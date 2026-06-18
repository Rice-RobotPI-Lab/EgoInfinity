"""
Run only SAM3D mesh reconstruction (Phase D-sam3d) for all favorites,
without re-running the rest of the pipeline. Reuses the SAM3 masks +
hand poses that are already in each clip's ``pipeline_result.pkl.gz``.

Per-oid mask scoring (Plan B):

    score(t, oid) = mask_area_norm[t][oid]
                  × (1 - hand_iou[t][oid])
                  × static_ratio[t]

  - **mask_area_norm**: mask pixels / (H*W); favours clean, large masks
  - **hand_iou**: IoU between mask and the projected hand silhouette
    (hand vertices_3d backprojected with dp_focal); penalises occlusion
  - **static_ratio**: derived from flow_rgb mean magnitude; penalises
    motion blur (SAM3D mesh quality degrades on blurred inputs)

The frame with the highest score per oid is the one fed to SAM3D.
SAM3D is invoked with quality='tier1' (gaussian_4 simplified output;
~70 KB - 10 MB per PLY).

Usage::

    python -m tools.rerun.refresh_sam3d_meshes               # all favorites
    python -m tools.rerun.refresh_sam3d_meshes --dry-run     # score + log only
    python -m tools.rerun.refresh_sam3d_meshes --only=ID1,ID2
    python -m tools.rerun.refresh_sam3d_meshes --skip-existing  # only oids without PLY
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Worker spawn/kill lives in the importable package (S2 refactor).
# Local aliases retain the underscore-prefixed names used downstream
# in this file, avoiding a large blast-radius rename.
from egoinfinity.pipeline.sam3d_runner import (  # noqa: E402
    spawn_sam3d_worker as _spawn_sam3d_worker,
    kill_sam3d_worker as _kill_sam3d_worker,
    SAM3D_WORKER_SOCKET,
)

_BROWSER_DIR = REPO_ROOT  # favorites cache root
FAVORITES_DIR = Path(os.environ.get(
    "ACTION100M_CACHE", str(_BROWSER_DIR / "cache"))) / "favorites"
LOG_DIR = REPO_ROOT / "tools" / "_batch_logs"

# Threshold in ten-thousandths (pct × 100): 50 = 0.5% (default); set
# SAM3D_MIN_MASK_AREA_PCT=10 (i.e. 0.1%) when intentionally regenerating
# small objects like tin cans / utensils that the default heuristic would
# otherwise filter out.
MIN_MASK_AREA_FRACTION = int(
    os.environ.get("SAM3D_MIN_MASK_AREA_PCT", "50")
) / 10000.0
STATIC_RATIO_GAIN = 6.0             # flow_mag * gain → motion penalty


def _decode_jpg_bgr(buf: bytes):
    """Decode JPG bytes to BGR ndarray (None if unparseable)."""
    if not isinstance(buf, (bytes, bytearray)) or not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _unpack_mask(mask_packed: bytes, mask_shape) -> np.ndarray | None:
    """Unpack a packbits mask back to a 2D bool array."""
    if mask_packed is None or mask_shape is None:
        return None
    shape = tuple(int(x) for x in mask_shape)
    n_total = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))
    if bits.size < n_total:
        return None
    return bits[:n_total].reshape(shape).astype(bool)


def _hand_mask_2d(verts_list, faces, dp_focal, cx, cy, H, W):
    """Project all hand vertices to 2D and rasterize a convex hull per hand.

    Cheap surrogate for hand silhouette: sometimes mask might overlap fingers
    only, but we want to know "is the hand covering this pixel?" Convex hull
    of projected vertices gives a slight over-estimate (fingers spread apart
    fold inward), but is robust and ~1ms.

    Returns (H, W) bool mask, or None if no valid hands.
    """
    if not verts_list or faces is None:
        return None
    out = np.zeros((H, W), dtype=np.uint8)
    found_any = False
    for v in verts_list:
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (778, 3):
            continue
        Z = v[:, 2]
        valid = (Z > 1e-6) & np.isfinite(Z)
        if valid.sum() < 50:
            continue
        u = v[valid, 0] / Z[valid] * dp_focal + cx
        vv = v[valid, 1] / Z[valid] * dp_focal + cy
        pts = np.stack([u, vv], axis=1).astype(np.int32)
        # Clip to image
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        if len(pts) >= 3:
            try:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(out, hull, 1)
                found_any = True
            except cv2.error:
                continue
    return out.astype(bool) if found_any else None


def _flow_motion_mag(flow_rgb_jpg: bytes) -> float:
    """Estimate per-frame motion magnitude from flow_rgb visualisation.

    flow_rgb is a HSV-style colour wheel of optical flow direction+magnitude;
    bright pixels = fast motion, dark = static. We use the mean intensity
    as a coarse proxy. Returns 0..1 (typical: 0.05 static, 0.4 fast motion).
    """
    img = _decode_jpg_bgr(flow_rgb_jpg)
    if img is None:
        return 0.0
    return float(img.mean()) / 255.0


def _score_frames_for_oid(fdata, oid, dp_focal, cx, cy, H, W, faces) -> np.ndarray:
    """Compute Plan B score per frame for a single oid. Returns (T,) float.

    Score components:
      area_norm           : mask area / image area  (large + clean wins)
      1 - hand_iou        : penalises hand-occluded frames (false-negative
                            when hand wraps *behind* the object — see below)
      static_ratio        : penalises motion-blur frames
      hand_proximity      : penalises frames where any 2D hand keypoint is
                            close to the object mask centroid — catches the
                            "hand wraps around object" case that hand_iou
                            misses (clip -8WOMg810tk_48.4_60.3 silver can,
                            where the fingers occlude the side of the can
                            but SAM2 dutifully excludes the finger pixels
                            from the can mask, so hand IoU = 0 even though
                            the can is heavily grasped).

    Plus an **area-outlier penalty** and a **hand-proximity penalty**,
    both gated by **hand_iou > DRIFT_HAND_IOU_THRESH**.

    The penalties target mid-action SAM2 mask drift, where the mask
    expands to swallow the hand/arm (packet + arm composite). Both
    failure modes share the same signature: oversized mask AND
    significant hand-pixel overlap. Frames where the object is held up
    cleanly (mask big but hand_iou ≈ 0) are NOT drift; they're the
    canonical-view frames we want to pick.

    Without the hand_iou gate, both penalties misfire in opposite
    directions: e.g. silver can in -8WOMg810tk_48.4_60.3 where the
    action-peak frames (t=75-81) have area 4.4× Q1 but hand_iou=0
    (can in palm, mask doesn't include hand pixels). The buggy old
    scoring rejected those frames in favour of a partial/occluded
    frame (t=98, area ≈ Q1 but a flat sliver silhouette), yielding a
    flat pancake mesh instead of a cylinder.

    Drift signatures the gated penalties still catch:
      - mask >> Q1 + hand_iou high  → arm-composite drift (was the
        original motivation, e.g. spice packet -8WOMg810tk_52.2_56.3)
      - prox to hand joints + hand_iou high → hand+object blob
    """
    T = len(fdata)
    scores = np.zeros(T, dtype=np.float32)
    img_area = float(H * W)
    AREA_OUTLIER_MULT = 2.0
    AREA_OUTLIER_PENALTY = 0.2
    # Drift gate: hand_iou must exceed this for area/prox penalties to fire.
    # Empirical: composite drift frames are ~0.4+; canonical-view frames
    # (object held up, hand below) sit at ~0.001.
    DRIFT_HAND_IOU_THRESH = 0.15
    # Hand-proximity: if min 2D hand-joint dist to mask centroid is below
    # HAND_PROX_PX, multiply score by HAND_PROX_PENALTY. Threshold scales
    # with image height so it stays geometrically meaningful across resolutions.
    HAND_PROX_PX = max(40.0, 0.08 * float(min(H, W)))
    HAND_PROX_PENALTY = 0.25

    # First pass: collect the per-frame raw mask areas for the outlier
    # baseline.  Use the 25th-percentile area (Q1) as the "stable" mask
    # size, not the median — when ~half the clip has SAM2 mask drift
    # (e.g. mid-action composite with the hand), the median is between
    # clean and drifted and 2× median doesn't penalise drift frames.
    # Q1 sits inside the clean cluster so 2×Q1 reliably separates them.
    areas_px = np.zeros(T, dtype=np.float64)
    for t, fd in enumerate(fdata):
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        mask = _unpack_mask(od.get("mask_packed"), od.get("mask_shape"))
        if mask is None or not mask.any():
            continue
        areas_px[t] = float(mask.sum())
    valid_areas = areas_px[areas_px > 0]
    stable_area = float(np.percentile(valid_areas, 25)) if valid_areas.size > 0 else 0.0
    outlier_cap = AREA_OUTLIER_MULT * stable_area if stable_area > 0 else float("inf")

    for t, fd in enumerate(fdata):
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        mask = _unpack_mask(od.get("mask_packed"), od.get("mask_shape"))
        if mask is None or not mask.any():
            continue
        area_px = float(mask.sum())
        area_norm = area_px / img_area
        if area_norm < MIN_MASK_AREA_FRACTION:
            continue
        # hand silhouette
        hand = _hand_mask_2d(
            fd.get("vertices_3d") or [], faces,
            dp_focal, cx, cy, H, W)
        if hand is not None:
            inter = int((mask & hand).sum())
            hand_iou = inter / max(int(mask.sum()), 1)
        else:
            hand_iou = 0.0
        # motion
        motion = _flow_motion_mag(fd.get("flow_rgb"))
        static = max(0.0, 1.0 - motion * STATIC_RATIO_GAIN)
        # Both penalties gate on hand_iou: only fire when mask actually
        # overlaps hand pixels (true drift composite), never on
        # canonical-view frames where the object is held up cleanly.
        is_drift = hand_iou > DRIFT_HAND_IOU_THRESH
        outlier_factor = (AREA_OUTLIER_PENALTY
                          if (area_px > outlier_cap and is_drift)
                          else 1.0)
        ys, xs = np.where(mask)
        mcx, mcy = float(xs.mean()), float(ys.mean())
        j2d = fd.get("joints_2d_pred")
        prox_factor = 1.0
        if j2d is not None and is_drift:
            jj = np.asarray(j2d, dtype=np.float32).reshape(-1, 2)
            valid_j = np.isfinite(jj).all(axis=1)
            if valid_j.any():
                dx = jj[valid_j, 0] - mcx
                dy = jj[valid_j, 1] - mcy
                min_d = float(np.sqrt(dx * dx + dy * dy).min())
                if min_d < HAND_PROX_PX:
                    prox_factor = HAND_PROX_PENALTY
        scores[t] = area_norm * (1.0 - hand_iou) * static * outlier_factor * prox_factor
    return scores


def _process_clip(fav_dir: Path, skip_existing: bool, dry_run: bool,
                  per_oid_worker=False, spawn_worker_fn=None, kill_worker_fn=None,
                  force_rerun_changed=False):
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fdata = data.get("frame_data") or []
    if not fdata:
        return "skip", "empty frame_data", {}

    # Get H, W without rehydrating: prefer mask_shape from any oid, else
    # decode img_rgb / depth_png. Avoid full pkl rehydration to keep this
    # tool fast and decoupled from pipeline_utils.
    H = W = None
    for fd in fdata:
        sd = fd.get("sam3_obj_data") or {}
        for od in sd.values():
            ms = od.get("mask_shape") if isinstance(od, dict) else None
            if ms is not None:
                H, W = int(ms[0]), int(ms[1])
                break
        if H is not None:
            break
    if H is None:
        # fallback: decode any frame's img_rgb
        for fd in fdata:
            ig = _decode_jpg_bgr(fd.get("img_rgb"))
            if ig is not None:
                H, W = ig.shape[:2]
                break
    if H is None:
        return "fail", "could not determine image dimensions", {}

    dp_focal = float(data.get("dp_focal", 0))
    cx = float(data.get("cx", W / 2))
    cy = float(data.get("cy", H / 2))
    faces = data.get("mano_faces")
    if dp_focal <= 0:
        return "fail", "no dp_focal", {}

    # Collect all oids that ever appeared
    oid_set = set()
    for fd in fdata:
        oid_set.update(int(k) for k in (fd.get("sam3_obj_data") or {}).keys())
    if not oid_set:
        return "skip", "no oids in any frame", {}

    mesh_dir = fav_dir / "sam3_meshes"
    mesh_dir.mkdir(exist_ok=True)
    sam3_mesh_info = data.get("sam3_mesh_info") or {}

    per_oid_results = {}
    for oid in sorted(oid_set):
        ply_path = mesh_dir / f"obj_{oid}.ply"
        prev_info = sam3_mesh_info.get(oid) or sam3_mesh_info.get(str(oid)) or {}
        forced = prev_info.get("init_frame_force") if isinstance(prev_info, dict) else None
        cur_init = prev_info.get("init_frame") if isinstance(prev_info, dict) else None
        ply_exists = ply_path.is_file() and ply_path.stat().st_size > 1000

        # Fast path: PLY exists, --skip-existing on, no force-rerun-changed
        # requested → skip without scoring. Avoids "all-0 score" fails on
        # oids that the new gating treats as unusable but where a prior
        # SAM3D run already succeeded (e.g. user-deleted background that
        # got restored).
        if ply_exists and skip_existing and not force_rerun_changed:
            per_oid_results[oid] = ("skip", f"PLY exists ({ply_path.stat().st_size//1024}KB)")
            continue

        # Manual override: if sam3_mesh_info[oid]['init_frame_force'] is set
        # in the pkl, skip scoring and use that frame. Used when SAM3D fails
        # on the auto-picked init_frame and a human wants to retry on a
        # different angle / less-occluded view.
        if forced is not None and 0 <= int(forced) < len(fdata):
            best_t = int(forced)
            best_score = float("nan")  # not from scoring
        else:
            scores = _score_frames_for_oid(fdata, oid, dp_focal, cx, cy, H, W, faces)
            if scores.max() <= 0:
                # If a PLY already exists, keep it (current scoring rejects
                # all frames, but a past run produced something usable).
                if ply_exists:
                    per_oid_results[oid] = ("skip", "PLY exists; current scoring rejects all frames")
                else:
                    per_oid_results[oid] = ("fail", "no usable frame (all 0 score)")
                continue
            best_t = int(np.argmax(scores))
            best_score = float(scores[best_t])

        # PLY-exists short-circuit for --force-rerun-changed path.
        if ply_exists:
            stale = (force_rerun_changed and cur_init is not None
                     and int(cur_init) != int(best_t))
            if skip_existing and not stale:
                per_oid_results[oid] = ("skip", f"PLY exists ({ply_path.stat().st_size//1024}KB)")
                continue
            if stale:
                action = "would delete" if dry_run else "deleting"
                print(f"  [force-rerun] oid {oid}: init_frame {cur_init} → {best_t}, "
                      f"{action} stale PLY", flush=True)
                if not dry_run:
                    try:
                        ply_path.unlink()
                    except FileNotFoundError:
                        pass

        # Read RGB frame from cache (4070 Ti path) or pkl-embedded img_rgb
        # (A100 multi-host path: frames/ is not uploaded to the private HF
        # transit dataset; pkl v2 carries per-frame JPG in img_rgb instead).
        frame_jpg = fav_dir / "frames" / f"frame_{best_t + 1:06d}.jpg"
        rgb_bgr = None
        if frame_jpg.is_file():
            rgb_bgr = cv2.imread(str(frame_jpg))
        if rgb_bgr is None:
            rgb_bgr = _decode_jpg_bgr(fdata[best_t].get("img_rgb"))
        if rgb_bgr is None:
            per_oid_results[oid] = (
                "fail",
                f"no RGB: missing {frame_jpg.name} and no img_rgb in pkl[{best_t}]")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        sd = fdata[best_t].get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        mask = _unpack_mask(od.get("mask_packed"), od.get("mask_shape"))
        if mask is None or mask.shape != rgb.shape[:2]:
            per_oid_results[oid] = ("fail", f"bad mask at frame {best_t}")
            continue

        # Feed FULL-frame RGB+mask, matching the A100 exo_pipeline path
        # (scripts/exo_pipeline.py:1789-1803). SAM3D's canonical-frame
        # inference depends on the surrounding context — bbox-cropping the
        # object recenters/rescales it inside the 518×518 input and
        # produces a different canonical_rotation_quat that does not match
        # hf_reference's quat → orientation formula.
        rgb_in = rgb
        mask_in = mask

        if dry_run:
            per_oid_results[oid] = (
                "dry", f"would reconstruct frame={best_t} score={best_score:.4f} "
                       f"area={int(mask.sum())}")
            continue

        # ── SAM3D inference ──
        # In per_oid_worker mode: spawn a fresh worker for this single oid,
        # call SAM3D, kill the worker. ~25s spawn + ~3s inference = ~28s/oid.
        # 100% success rate at the cost of speed. Use when the persistent
        # worker keeps OOMing on a 16 GB card.
        if per_oid_worker and not dry_run:
            if spawn_worker_fn is None or kill_worker_fn is None:
                per_oid_results[oid] = ("fail", "per_oid_worker missing fns")
                continue
            try:
                spawn_worker_fn()
            except Exception as e:
                per_oid_results[oid] = ("fail", f"worker spawn: {e}")
                continue

        from egoinfinity.pipeline.sam3d_client import reconstruct_object
        try:
            t0 = time.time()
            res = reconstruct_object(
                rgb=rgb_in, mask=mask_in,
                out_ply_path=str(ply_path),
                seed=42, quality="tier1", timeout=300.0,
            )
            elapsed = time.time() - t0
        except Exception as e:
            per_oid_results[oid] = ("fail", f"SAM3D inference: {type(e).__name__}: {e}")
            if per_oid_worker and not dry_run:
                try: kill_worker_fn()
                except Exception: pass
            continue
        finally:
            pass
        if per_oid_worker and not dry_run:
            try: kill_worker_fn()
            except Exception: pass

        # Preserve human-supplied fields (init_frame_force, notes, prompt
        # overrides, etc.) by merging into the prior info dict rather than
        # replacing it wholesale. Wholesale replacement was silently
        # dropping init_frame_force on rerun (an oid's force override would
        # otherwise be lost and fall back to score-based frame selection).
        new_info = dict(prev_info) if isinstance(prev_info, dict) else {}
        new_info.update({
            "ply_path": str(ply_path.relative_to(fav_dir)),
            "init_frame": best_t,
            "score": best_score,
            "n_points": int(res.n_points),
            "canonical_translation": res.translation.tolist(),
            "canonical_rotation_quat": res.rotation_quat.tolist(),
            "canonical_scale": float(res.scale),
        })
        sam3_mesh_info[int(oid)] = new_info
        sz_kb = ply_path.stat().st_size // 1024
        per_oid_results[oid] = (
            "ok", f"frame={best_t} score={best_score:.4f} pts={res.n_points} "
                  f"size={sz_kb}KB time={elapsed:.1f}s")

    if not dry_run and any(r[0] == "ok" for r in per_oid_results.values()):
        # Write back updated sam3_mesh_info
        data["sam3_mesh_info"] = sam3_mesh_info
        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)
        # Multi-host sentinel: this host just produced SAM3D meshes for
        # this clip.  4070 Ti's `refresh_pose_tracking --scan` will then
        # pick it up for D-track.  Non-fatal — main success path doesn't
        # depend on the state file.
        try:
            from egoinfinity.pipeline import pipeline_state as _ps
            _ps.mark_done(fav_dir, "D-sam3d")
        except Exception as _e:
            print(f"  [state] mark_done(D-sam3d) failed (non-fatal): {_e}")

    n_ok = sum(1 for r in per_oid_results.values() if r[0] == "ok")
    n_dry = sum(1 for r in per_oid_results.values() if r[0] == "dry")
    n_fail = sum(1 for r in per_oid_results.values() if r[0] == "fail")
    n_skip = sum(1 for r in per_oid_results.values() if r[0] == "skip")
    n_good = n_ok + n_dry  # treat dry-run as success for top-level status
    summary = f"{n_good}/{len(oid_set)} ok"
    if n_fail:
        summary += f", {n_fail} fail"
    if n_skip:
        summary += f", {n_skip} skip"
    return ("ok" if n_good > 0 else "fail" if n_fail else "skip",
            summary, per_oid_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated favorite ids "
                         "(use --only=ID1,ID2 for leading-dash ids)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip oids whose PLY already exists & non-empty")
    ap.add_argument("--force-rerun-changed", action="store_true",
                    help="Even with --skip-existing, rerun SAM3D for oids "
                         "whose current scoring picks a different init_frame "
                         "than what's stored in the pkl. Use after the "
                         "scoring formula changes to refresh affected meshes "
                         "without rebuilding all of them.")
    ap.add_argument("--scan", action="store_true",
                    help="Multi-host mode: skip clips whose pipeline_state.json "
                         "says D-sam3d is already done, AND skip clips that "
                         "haven't yet completed D-sam3 upstream.  Pairs with "
                         "scripts/exo_pipeline.py's mark_done() writes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Score + log; no SAM3D inference, no writes")
    ap.add_argument("--per-oid-worker", action="store_true",
                    help="Spawn fresh SAM3D worker per oid (~28s/oid). "
                         "Slow but 100%% success rate on 16 GB cards; "
                         "use when persistent worker OOMs.")
    args = ap.parse_args()

    favs = sorted(d for d in FAVORITES_DIR.iterdir()
                  if d.is_dir() and (d / "pipeline_result.pkl.gz").is_file())
    if args.only:
        only = set(s.strip() for s in args.only.split(",") if s.strip())
        favs = [d for d in favs if d.name in only]
    if args.scan:
        from egoinfinity.pipeline import pipeline_state as _ps
        before = len(favs)
        favs = [d for d in favs
                if _ps.is_phase_done(d, "D-sam3")
                and not _ps.is_phase_done(d, "D-sam3d")]
        print(f"[--scan] {before} eligible → {len(favs)} pending D-sam3d "
              f"(skipped: already done, or D-sam3 upstream missing)")
    if not favs:
        print("Nothing to process")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Spawn SAM3D worker (skip in dry-run AND per-oid-worker modes; the
    # latter spawns inside _process_clip's per-oid loop instead).
    proc = None
    os.environ["SAM3D_WORKER_SOCKET"] = SAM3D_WORKER_SOCKET

    def _spawn():
        nonlocal proc
        proc, msg = _spawn_sam3d_worker(
            SAM3D_WORKER_SOCKET, LOG_DIR / "_sam3d_refresh_worker.log")
        return msg

    def _kill():
        nonlocal proc
        if proc is not None:
            _kill_sam3d_worker(proc, SAM3D_WORKER_SOCKET)
            proc = None

    if not args.dry_run and not args.per_oid_worker:
        try:
            msg = _spawn()
            print(f"[sam3d] worker ready: {msg}")
        except Exception as e:
            print(f"[sam3d] worker spawn FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    counts = {"ok": 0, "fail": 0, "skip": 0}
    t0_all = time.time()
    # SAM3D accumulates GPU memory across calls; on a 16 GB card we OOM
    # after ~10 calls. Strategy: restart worker (a) every N OIDS ATTEMPTED
    # (ok or fail), and (b) immediately whenever an OOM is detected — once
    # OOM hits the worker is in a corrupted state and all subsequent calls
    # also OOM.
    RESTART_EVERY_ATTEMPTS = int(os.environ.get('SAM3D_RESTART_EVERY', '8'))
    attempts_since_restart = 0

    def _restart_worker():
        nonlocal proc, attempts_since_restart
        if args.dry_run or proc is None:
            return
        print("[sam3d] restarting worker ...", flush=True)
        _kill_sam3d_worker(proc, SAM3D_WORKER_SOCKET)
        proc, msg = _spawn_sam3d_worker(
            SAM3D_WORKER_SOCKET, LOG_DIR / "_sam3d_refresh_worker.log")
        print(f"[sam3d] worker ready: {msg}", flush=True)
        attempts_since_restart = 0

    try:
        for i, fav in enumerate(favs, 1):
            t0 = time.time()
            try:
                st, summary, per_oid = _process_clip(
                    fav, args.skip_existing, args.dry_run,
                    per_oid_worker=args.per_oid_worker,
                    spawn_worker_fn=_spawn, kill_worker_fn=_kill,
                    force_rerun_changed=args.force_rerun_changed)
            except Exception as e:
                st, summary, per_oid = "fail", f"crash: {type(e).__name__}: {e}", {}
            dt = time.time() - t0
            counts[st] = counts.get(st, 0) + 1
            marker = {"ok": "+", "skip": "·", "fail": "✗"}.get(st, "?")
            print(f"[{i:3d}/{len(favs)}] {marker} {fav.name:48s} "
                  f"{summary} ({dt:.1f}s)", flush=True)
            oom_seen = False
            for oid, (oid_st, oid_msg) in sorted(per_oid.items()):
                if oid_st in ("ok", "fail", "dry"):
                    sym = "+" if oid_st == "ok" else "✗" if oid_st == "fail" else "?"
                    print(f"           {sym} obj {oid}: {oid_msg}", flush=True)
                if oid_st in ("ok", "fail"):
                    attempts_since_restart += 1
                if oid_st == "fail" and "OutOfMemory" in oid_msg:
                    oom_seen = True
            if i >= len(favs):
                continue  # don't restart after last clip
            if oom_seen:
                _restart_worker()
            elif attempts_since_restart >= RESTART_EVERY_ATTEMPTS:
                _restart_worker()
    finally:
        if proc is not None:
            _kill_sam3d_worker(proc, SAM3D_WORKER_SOCKET)

    dt_total = time.time() - t0_all
    print(f"\nDone in {dt_total/60:.1f} min — "
          f"ok: {counts.get('ok', 0)}  "
          f"skip: {counts.get('skip', 0)}  "
          f"fail: {counts.get('fail', 0)}")


if __name__ == "__main__":
    main()
