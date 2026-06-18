"""Hand-mesh Z alignment in WRIST frames.

Background
==========
During WRIST mode (``state ∈ {grasped_l, grasped_r, grasped_both}``), the
SAM3D mesh's z-position is set by phase_d (Part B: bbox-center of obs cloud,
Kabsch palm binding).  Both signals share the underlying MoGe-2 monocular
depth noise — typical residual 30-60 mm of "mesh hovering above the palm"
or "mesh sunk into the fingers" in the viser viewer.

The earlier obs-cloud-vs-hand alignment approach (v1) under-corrected
because the SAM mask leaks onto hand pixels, contaminating the obs cloud
with hand-z samples and diluting the Δz estimate.  This version (v2)
aligns to the **mesh directly**: at this stage the mesh is the most
trustworthy object-shape signal (SAM3D + scale_sanity + R_anchor + obb
priority lock).  Per WRIST frame, find the K hand vertices closest to the
mesh surface, compute the mean z difference, shift the mesh by Δz.  Pure
geometric — no dependence on per-frame MoGe noise or SAM mask edges.

Where it slots in
=================
Between phase_d Pass 2 and ``bake_fp_pose``:

    phase_d Pass 1 → veto → phase_d Pass 2 → depth_align → bake → ...

Must run before bake because bake's R-chain reads T_seq for
``hand_rigid_grasp_lock`` and ``state_aware_lock`` decisions.

What it touches
===============
  ``pose_track_info[oid]['T_seq'][:, 2, 3]``
  ``frame_data[t]['sam3_obj_data'][oid]['pts'][:, 2]``        (if cached)
  ``frame_data[t]['sam3_obj_data'][oid]['pose_t'][2]``        (if cached)
  ``frame_data[t]['sam3_obj_data'][oid]['obb_corners'][:, 2]`` (if cached)
  ``data['depth_align_refresh'] = {'history': [...]}``

What it does NOT touch
======================
  - STATIC LOCK frames: lock pose comes from anchor, not the Kabsch chain.
  - DEPTH_TRACKED frames: obs cloud directly drives T_depth, no Kabsch bias.
  - Rotation: ``T_seq[:, :3, :3]`` is owned by bake.
  - ``pose_track_info_meta``: bake's canonical check
    (``fp_pose_bake.ts ≥ pose_track_info_meta.updated_at``) must still hold.

Idempotency
===========
Re-running on already-aligned pkl compounds the shift unless the source
data was rolled back. Use ``--skip-if-done`` to short-circuit clips with
prior history, or ``--force`` to ignore and re-align.

Usage
=====
    python -m egoinfinity.pipeline.post_tracking.depth_align --only=<CLIP_ID> --dry-run
    python -m egoinfinity.pipeline.post_tracking.depth_align --only=<CLIP_ID> --force
    python -m egoinfinity.pipeline.post_tracking.depth_align
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"

DEFAULT_K_PAIRS = 20             # # of closest hand-mesh vertex pairs
DEFAULT_FAR_REJECT_M = 0.20      # if even the closest pair is > 20cm apart,
                                  # the hand is not really near the object —
                                  # skip frame (likely residual veto FP)
DEFAULT_SAVGOL_WIN = 7
DEFAULT_SAVGOL_POLY = 2
DEFAULT_EDGE_RAMP = 2            # frames; only applied at INTERNAL segment
                                  # boundaries (clip edges keep full Δz).
                                  # Was 3; reduced to 2 (steeper) per visual
                                  # review — 2-frame transition over ~0.13s
                                  # @ 15fps is fast enough to avoid the
                                  # "mesh retreats at boundary" effect.
DEFAULT_MAX_DELTA_M = 0.10
DEFAULT_MIN_SEG_LEN = 3


# ── helpers ───────────────────────────────────────────────────────────
def _find_segments(b: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Inclusive (s, e) spans where b is True, length ≥ min_len."""
    T = len(b)
    segs = []
    i = 0
    while i < T:
        if b[i]:
            j = i
            while j < T and b[j]:
                j += 1
            if (j - i) >= min_len:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def _hand_verts_for_grasp(fd: dict, want_left: bool, want_right: bool
                           ) -> List[np.ndarray]:
    """Return the (778, 3) world-frame vertex arrays of the active hand(s)
    for this frame.  Filters out None entries and shape mismatches.

    Active hands selected from the per-frame `hand_is_right` list using
    the wrist_l_per_frame / wrist_r_per_frame booleans of *the object*
    (passed in from the caller as want_left / want_right)."""
    out = []
    verts_list = fd.get("vertices_3d") or []
    is_right_list = fd.get("hand_is_right") or []
    for v, ir in zip(verts_list, is_right_list):
        if v is None:
            continue
        v = np.asarray(v)
        if v.shape != (778, 3):
            continue
        if ir and want_right:
            out.append(v.astype(np.float32))
        elif (not ir) and want_left:
            out.append(v.astype(np.float32))
    return out


def _hand_mesh_dz(hand_verts_world: np.ndarray, mesh_world: np.ndarray,
                  k_pairs: int, far_reject_m: float
                  ) -> Optional[Tuple[float, float, int]]:
    """Return (hand_z_mean, mesh_z_mean, k_used) or None.

    For each hand vertex find its nearest mesh vertex.  Sort by distance,
    take the K closest pairs as the "contact patch" — these are the hand
    vertices that are physically nearest to the mesh surface (palm /
    fingertips, depending on grasp type).  Mean z over those K pairs.

    No fixed distance threshold: the mesh always exists somewhere, so
    nearest-K is always non-empty.  We reject only when even the single
    closest pair is > far_reject_m (means the hand is genuinely far from
    the object — likely a residual veto false positive that survived
    refresh_grasp_veto; skipping the frame avoids pulling the mesh toward
    a non-grasping hand).
    """
    if mesh_world is None or len(mesh_world) < 10:
        return None
    if hand_verts_world is None or len(hand_verts_world) < 10:
        return None
    tree = cKDTree(mesh_world)
    d, idx = tree.query(hand_verts_world, k=1)
    # Early reject if even the single best pair is far
    if float(d.min()) > far_reject_m:
        return None
    # Take K closest pairs
    k = min(int(k_pairs), len(d))
    order = np.argsort(d)[:k]
    hand_z = float(hand_verts_world[order, 2].mean())
    mesh_z = float(mesh_world[idx[order], 2].mean())
    return hand_z, mesh_z, k


def _apply_edge_ramp(delta: np.ndarray, segments: List[Tuple[int, int]],
                     edge_ramp: int, T: int) -> np.ndarray:
    """Linearly ramp delta 0 ↔ delta at INTERNAL segment boundaries only.

    The ramp exists to absorb Δz step changes between a WRIST segment
    (with Δz applied) and an adjacent non-WRIST segment (Δz=0).  At clip
    boundaries (frame 0 or T-1) there is no adjacent frame to blend with,
    so we keep the full Δz there — otherwise the mesh visibly "retreats"
    toward its un-aligned position at the start/end of the clip."""
    out = delta.astype(np.float64).copy()
    for (s, e) in segments:
        seg_len = e - s + 1
        ramp = min(edge_ramp, seg_len // 2)
        if ramp <= 0:
            continue
        head_internal = (s > 0)        # frame s-1 exists and is non-WRIST
        tail_internal = (e < T - 1)    # frame e+1 exists and is non-WRIST
        for k in range(ramp):
            w = (k + 1) / (ramp + 1)
            if head_internal:
                out[s + k] *= w
            if tail_internal:
                out[e - k] *= w
    return out


# ── per-clip / per-oid pass ───────────────────────────────────────────
def _process_object(
    info: dict,
    oid: int,
    fdata: list,
    mesh_pts_world: np.ndarray,   # raw PLY xyz × scale_correction (canonical frame)
    *,
    k_pairs: int,
    far_reject_m: float,
    savgol_win: int,
    savgol_poly: int,
    edge_ramp: int,
    max_delta_m: float,
    min_seg_len: int,
    dry_run: bool,
) -> dict:
    """Returns per-oid stats dict.  Mutates info / fdata when not dry_run."""
    T_seq = info.get("T_seq")
    if T_seq is None:
        return {"status": "skip", "reason": "no T_seq"}
    T_seq_arr = np.asarray(T_seq, dtype=np.float32)
    T = T_seq_arr.shape[0]
    if T_seq_arr.ndim != 3 or T_seq_arr.shape[1:] != (4, 4):
        return {"status": "skip", "reason": f"bad T_seq shape {T_seq_arr.shape}"}

    wl = info.get("wrist_l_per_frame")
    wr = info.get("wrist_r_per_frame")
    if wl is None or wr is None:
        return {"status": "skip", "reason": "no wrist_l/r_per_frame"}
    wl_arr = np.asarray(wl, dtype=bool)[:T]
    wr_arr = np.asarray(wr, dtype=bool)[:T]
    wrist_used = wl_arr | wr_arr
    if not wrist_used.any():
        return {"status": "skip", "reason": "no wrist frames"}

    segments = _find_segments(wrist_used, min_len=min_seg_len)
    if not segments:
        return {"status": "skip", "reason": "all wrist segments too short"}

    # Per-frame raw delta_z; NaN where no contact (hand too far from mesh)
    raw_delta = np.full(T, np.nan, dtype=np.float64)
    n_far_reject = 0
    k_used_per_frame = np.zeros(T, dtype=np.int32)

    for t in range(T):
        if not wrist_used[t]:
            continue
        # Compute mesh in world frame for this frame
        R_t = T_seq_arr[t, :3, :3].astype(np.float64)
        trans_t = T_seq_arr[t, :3, 3].astype(np.float64)
        mesh_world = mesh_pts_world @ R_t.T + trans_t
        hverts_list = _hand_verts_for_grasp(
            fdata[t], want_left=bool(wl_arr[t]), want_right=bool(wr_arr[t]))
        if not hverts_list:
            continue
        # Aggregate over (potentially) both hands.  In grasped_both, weight
        # each hand by its k_pairs contribution.
        weighted_hand_z = 0.0
        weighted_mesh_z = 0.0
        total_k = 0
        for hv in hverts_list:
            res = _hand_mesh_dz(hv, mesh_world, k_pairs, far_reject_m)
            if res is None:
                continue
            hz, mz, k = res
            weighted_hand_z += hz * k
            weighted_mesh_z += mz * k
            total_k += k
        if total_k == 0:
            n_far_reject += 1
            continue
        raw_delta[t] = (weighted_hand_z / total_k) - (weighted_mesh_z / total_k)
        k_used_per_frame[t] = total_k

    # Per-segment SavGol smoothing of the raw delta
    smoothed = np.zeros(T, dtype=np.float64)
    seg_stats = []
    n_segs_aligned = 0
    for (s, e) in segments:
        seg_len = e - s + 1
        seg = raw_delta[s:e + 1].copy()
        valid = np.isfinite(seg)
        if valid.sum() < 2:
            continue
        if not valid.all():
            x_full = np.arange(seg_len)
            seg = np.interp(x_full, x_full[valid], seg[valid])
        win = min(savgol_win, seg_len if seg_len % 2 == 1 else seg_len - 1)
        if win < savgol_poly + 2:
            seg_smooth = np.full(seg_len, float(np.median(seg)))
        else:
            if win % 2 == 0:
                win -= 1
            seg_smooth = savgol_filter(seg, win, savgol_poly)
        smoothed[s:e + 1] = seg_smooth
        seg_stats.append({
            "s": int(s), "e": int(e),
            "raw_median_mm": float(np.median(seg) * 1000),
            "smooth_median_mm": float(np.median(seg_smooth) * 1000),
            "n_contact_frames": int(valid.sum()),
            "seg_len": seg_len,
        })
        n_segs_aligned += 1

    if n_segs_aligned == 0:
        return {"status": "skip", "reason": "no contact across any wrist segment",
                "n_far_reject": n_far_reject}

    smoothed_ramped = _apply_edge_ramp(smoothed, segments, edge_ramp, T)
    capped = np.clip(smoothed_ramped, -max_delta_m, max_delta_m)
    n_capped = int(((smoothed_ramped != capped)).sum())
    final_delta = np.zeros(T, dtype=np.float64)
    for (s, e) in segments:
        final_delta[s:e + 1] = capped[s:e + 1]

    abs_dz = np.abs(final_delta[final_delta != 0.0])
    stats = {
        "status": "ok",
        "n_wrist_frames": int(wrist_used.sum()),
        "n_segments": len(segments),
        "n_segs_aligned": n_segs_aligned,
        "n_frames_aligned": int((final_delta != 0).sum()),
        "n_frames_far_reject": int(n_far_reject),
        "n_capped": n_capped,
        "abs_dz_median_mm": float(np.median(abs_dz) * 1000) if abs_dz.size else 0.0,
        "abs_dz_max_mm": float(abs_dz.max() * 1000) if abs_dz.size else 0.0,
        "abs_dz_mean_mm": float(abs_dz.mean() * 1000) if abs_dz.size else 0.0,
        "signed_dz_median_mm": float(np.median(final_delta[final_delta != 0]) * 1000)
                                if abs_dz.size else 0.0,
        "segments": seg_stats,
    }

    if dry_run:
        return stats

    T_seq_new = T_seq_arr.copy()
    T_seq_new[:, 2, 3] = T_seq_new[:, 2, 3] + final_delta.astype(np.float32)
    info["T_seq"] = T_seq_new

    for t in range(T):
        dz = float(final_delta[t])
        if abs(dz) < 1e-6:
            continue
        fd = fdata[t]
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        pts = od.get("pts")
        if pts is not None:
            pts_new = np.asarray(pts, dtype=np.float32).copy()
            pts_new[:, 2] += dz
            od["pts"] = pts_new
        pose_t = od.get("pose_t")
        if pose_t is not None:
            pt_new = np.asarray(pose_t, dtype=np.float32).copy()
            pt_new[2] += dz
            od["pose_t"] = pt_new
        obb = od.get("obb_corners")
        if obb is not None:
            obb_new = np.asarray(obb, dtype=np.float32).copy()
            obb_new[:, 2] += dz
            od["obb_corners"] = obb_new

    return stats


def _process_clip(
    fav_dir: Path, *,
    k_pairs: int,
    far_reject_m: float,
    savgol_win: int,
    savgol_poly: int,
    edge_ramp: int,
    max_delta_m: float,
    min_seg_len: int,
    dry_run: bool,
    skip_if_done: bool,
    force: bool,
) -> Tuple[str, str, dict]:
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if skip_if_done and not force:
        prov = (data.get("depth_align_refresh") or {}).get("history") or []
        if prov:
            return "skip", "already aligned", {}

    pti = data.get("pose_track_info") or {}
    if not pti:
        return "skip", "no pose_track_info", {}
    sam3_mesh_info = data.get("sam3_mesh_info") or {}
    if not sam3_mesh_info:
        return "skip", "no sam3_mesh_info", {}
    fdata = data.get("frame_data") or []
    if not fdata:
        return "skip", "no frame_data", {}

    from egoinfinity.pipeline.pose_tracker.utils import load_mesh_points_from_ply

    summary = {}
    any_change = False
    for oid_key, info in list(pti.items()):
        if not isinstance(info, dict):
            continue
        oid = int(oid_key) if not isinstance(oid_key, int) else oid_key

        # Load mesh PLY for this oid
        meta = sam3_mesh_info.get(oid_key) or sam3_mesh_info.get(int(oid_key))
        if not isinstance(meta, dict):
            summary[oid] = {"status": "skip", "reason": "no mesh_info"}
            continue
        ply_rel = meta.get("ply_path")
        if not ply_rel:
            summary[oid] = {"status": "skip", "reason": "no ply_path"}
            continue
        ply_path = Path(ply_rel)
        if not ply_path.is_absolute():
            ply_path = fav_dir / ply_path
        if not ply_path.exists():
            summary[oid] = {"status": "skip", "reason": f"missing PLY {ply_path}"}
            continue
        try:
            mesh_pts_raw, _ = load_mesh_points_from_ply(str(ply_path))
        except Exception as e:
            summary[oid] = {"status": "skip", "reason": f"PLY load fail: {e}"}
            continue
        mesh_pts_raw = np.asarray(mesh_pts_raw, dtype=np.float64)
        # The tracker's scale_correction (post refresh_scale_sanity) is the
        # absolute factor: raw PLY units → world metres.  Same convention as
        # _track_phase_d and the viser exporter.
        scale = float(info.get("scale_correction") or
                      meta.get("canonical_scale") or 1.0)
        mesh_pts_world_canonical = mesh_pts_raw * scale

        res = _process_object(
            info, oid, fdata, mesh_pts_world_canonical,
            k_pairs=k_pairs,
            far_reject_m=far_reject_m,
            savgol_win=savgol_win,
            savgol_poly=savgol_poly,
            edge_ramp=edge_ramp,
            max_delta_m=max_delta_m,
            min_seg_len=min_seg_len,
            dry_run=dry_run,
        )
        summary[oid] = res
        if res.get("status") == "ok" and res.get("n_frames_aligned", 0) > 0:
            any_change = True

    if not any_change:
        return "skip", "no oids aligned (no grasp in any wrist segment)", summary

    one_liner_parts = []
    for oid, s in sorted(summary.items()):
        if s.get("status") == "ok":
            one_liner_parts.append(
                f"obj{oid}:{s['n_frames_aligned']}f|Δz_signed_med={s['signed_dz_median_mm']:+.0f}mm"
                f"|abs_max={s['abs_dz_max_mm']:.0f}mm|cap={s['n_capped']}"
                f"|far_rej={s['n_frames_far_reject']}"
            )
    action = " ".join(one_liner_parts)

    if not dry_run:
        prov = data.get("depth_align_refresh") or {}
        history = prov.get("history") or []
        history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "algo_version": "v2-mesh",
            "k_pairs": k_pairs,
            "far_reject_m": far_reject_m,
            "savgol_win": savgol_win,
            "savgol_poly": savgol_poly,
            "edge_ramp": edge_ramp,
            "max_delta_m": max_delta_m,
            "min_seg_len": min_seg_len,
            "objs": {
                oid: {k: v for k, v in s.items()
                      if k not in ("segments",)}
                for oid, s in summary.items() if s.get("status") == "ok"
            },
        })
        data["depth_align_refresh"] = {"history": history}
        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)

    return "ok", action, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="comma-separated clip ids "
                         "(use --only=-ID for leading-dash ids)")
    ap.add_argument("--k-pairs", type=int, default=DEFAULT_K_PAIRS,
                    help=f"# of closest hand-mesh vertex pairs to use "
                         f"(default {DEFAULT_K_PAIRS})")
    ap.add_argument("--far-reject-m", type=float, default=DEFAULT_FAR_REJECT_M,
                    help=f"reject frame if even the closest hand-mesh pair "
                         f"exceeds this distance (default "
                         f"{DEFAULT_FAR_REJECT_M} m)")
    ap.add_argument("--savgol-win", type=int, default=DEFAULT_SAVGOL_WIN,
                    help=f"SavGol window for per-segment Δz smoothing "
                         f"(default {DEFAULT_SAVGOL_WIN})")
    ap.add_argument("--savgol-poly", type=int, default=DEFAULT_SAVGOL_POLY,
                    help=f"SavGol polyorder (default {DEFAULT_SAVGOL_POLY})")
    ap.add_argument("--edge-ramp", type=int, default=DEFAULT_EDGE_RAMP,
                    help=f"linear ramp frames at segment edges "
                         f"(default {DEFAULT_EDGE_RAMP})")
    ap.add_argument("--max-delta-m", type=float, default=DEFAULT_MAX_DELTA_M,
                    help=f"|Δz| cap (default {DEFAULT_MAX_DELTA_M} m)")
    ap.add_argument("--min-seg-len", type=int, default=DEFAULT_MIN_SEG_LEN,
                    help=f"min WRIST segment length to attempt alignment "
                         f"(default {DEFAULT_MIN_SEG_LEN})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute Δz stats without writing")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips already in depth_align_refresh.history")
    ap.add_argument("--force", action="store_true",
                    help="Ignore prior provenance and re-apply")
    args = ap.parse_args()

    allow = (set(s.strip() for s in args.only.split(",") if s.strip())
             if args.only else None)
    favs = sorted(p for p in FAV.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_depth_align (v2-mesh): {len(favs)} clips  dry_run={args.dry_run}"
          f"  K={args.k_pairs}  far_reject={args.far_reject_m}m"
          f"  savgol=({args.savgol_win},{args.savgol_poly})  ramp={args.edge_ramp}"
          f"  cap={args.max_delta_m}m")
    n_ok = n_skip = n_fail = 0
    abs_dz_all = []
    signed_dz_all = []
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        try:
            status, action, summary = _process_clip(
                fav,
                k_pairs=args.k_pairs,
                far_reject_m=args.far_reject_m,
                savgol_win=args.savgol_win,
                savgol_poly=args.savgol_poly,
                edge_ramp=args.edge_ramp,
                max_delta_m=args.max_delta_m,
                min_seg_len=args.min_seg_len,
                dry_run=args.dry_run,
                skip_if_done=args.skip_if_done,
                force=args.force,
            )
        except Exception as e:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  CRASH ({type(e).__name__}: {e})")
            continue
        dt = time.time() - t_clip
        if status == "ok":
            n_ok += 1
            for s in summary.values():
                if isinstance(s, dict) and s.get("status") == "ok":
                    abs_dz_all.append(s.get("abs_dz_median_mm", 0))
                    signed_dz_all.append(s.get("signed_dz_median_mm", 0))
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}"
                  f"+ {fav.name:<50}  {action}  ({dt:.1f}s)")
        elif status == "skip":
            n_skip += 1
            if action not in ("no oids aligned (no grasp in any wrist segment)",
                              "already aligned"):
                print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({action})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({action})")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — "
          f"ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")
    if abs_dz_all:
        a = np.array(abs_dz_all)
        s = np.array(signed_dz_all)
        print(f"Per-oid median |Δz|: mean={a.mean():.1f}mm  max={a.max():.1f}mm  "
              f"n={len(a)} oids aligned")
        print(f"Per-oid median signed Δz: mean={s.mean():+.1f}mm  "
              f"% negative (mesh pushed TOWARD camera): {100*(s<0).mean():.0f}%")


if __name__ == "__main__":
    main()
