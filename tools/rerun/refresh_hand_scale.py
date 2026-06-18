"""Refresh MANO hand metric scale to match WiLoR's native 2D detection.

Background
==========
The pipeline stores two parallel sources of hand 2D info per frame:

  joints_2d_pred  — WiLoR's pixel-space 2D joints (from un-cropped detector
                    output, in image pixels, MoGe-independent).
  vertices_3d /   — MANO 3D mesh, re-projected to MoGe's metric world via
  joints_3d_pred    align_hand_to_depth_multiscale (uses MoGe dp_focal).

The depths come from MoGe-2. The 2D from WiLoR. There is an *implicit* focal
length on each side. When MoGe's dp_focal differs from the effective focal
WiLoR used (it usually does, especially at close range), the existing
median-translation alignment (which assumes MANO's metric scale matches
the visible hand's real metric scale) breaks down — translation alone
cannot reconcile two hand-shapes-at-different-scales.

Concrete example (Pat bacon clip frame 0 hand 0):
  joints_2d_pred extent  : 158 x 173 px   ← matches visible hand in image
  joints_3d → 2D extent  :  85 x  97 px   ← what the stored 3D produces
  Ratio                  : ~1.85×

In the 3D viewer this shows up as: cutting board (D-track scaled to fit
SAM2 mask in MoGe depth) is correctly sized at ~1.5 m, but the hand stays
at MANO's intrinsic ~14 cm extent — so the hand looks ~5× too small
relative to the board.

Fix
===
For each (frame, hand), solve a Procrustes alignment (isotropic-scale +
translation) that maps MANO's current joints_3d_pred onto target 3D
points obtained by back-projecting joints_2d_pred through MoGe depth:

  target_3d[i] = ((u_i - cx) * d_i / f, (v_i - cy) * d_i / f, d_i)
                  where d_i = depth_map at joints_2d_pred[i].

  fit s, t :   s * (J[i] - centroid_J) + centroid_target  ≈  target_3d[i]

This jointly recovers BOTH the missing scale factor (which align_hand_to_
depth alone couldn't) AND a consistent translation (centroid replaces the
median-of-translations heuristic). The result satisfies the projection
equation through dp_focal AND keeps depth consistent with MoGe.

After this pass:
  - joints_3d_pred / vertices_3d are scaled isotropically to MoGe metric.
  - 3D→2D projection of the new 3D matches joints_2d_pred (overlay + mesh
    agree).
  - object meshes (D-track-fit to MoGe) are unchanged — hand and scene
    now share a single MoGe-consistent scale.

Usage
=====
  python -m tools.rerun.refresh_hand_scale --only=-8WOMg810tk_92.8_97.6
  python -m tools.rerun.refresh_hand_scale                       # all favorites
  python -m tools.rerun.refresh_hand_scale --dry-run            # report only
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"


def _sample_depth_patch(depth_map: np.ndarray, u: float, v: float, half: int = 3) -> float:
    """Median depth in a (2h+1) px patch around (u,v); NaN if all invalid."""
    H, W = depth_map.shape
    xi, yi = int(round(u)), int(round(v))
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return float("nan")
    x0, x1 = max(0, xi - half), min(W, xi + half + 1)
    y0, y1 = max(0, yi - half), min(H, yi + half + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.01]
    return float(np.median(valid)) if valid.size else float("nan")


def _backproject_joints(joints_2d: np.ndarray,
                        depth_map: np.ndarray,
                        focal: float, cx: float, cy: float) -> np.ndarray:
    """Back-project 21 2D joints to 3D using MoGe depth. NaN row if invalid."""
    out = np.full((joints_2d.shape[0], 3), np.nan, dtype=np.float32)
    for i, (u, v) in enumerate(joints_2d):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        d = _sample_depth_patch(depth_map, u, v)
        if not np.isfinite(d) or d <= 0:
            continue
        out[i, 0] = (u - cx) * d / focal
        out[i, 1] = (v - cy) * d / focal
        out[i, 2] = d
    return out


def _solve_iso_scale_translation(source: np.ndarray, target: np.ndarray
                                 ) -> tuple[float, np.ndarray]:
    """Procrustes-style: find isotropic scale s and translation t such that
    s * (source - centroid_source) + centroid_target ≈ target.

    Returns (s, t_apply) where t_apply is the translation to add AFTER scaling
    source's centroid-relative coords. Caller applies as:
        new = s * (source - centroid_source) + centroid_target

    Skips NaN rows. Returns (1.0, zeros) if too few valid pairs.
    """
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    if valid.sum() < 4:
        return 1.0, np.zeros(3, dtype=np.float32)
    S = source[valid]
    T = target[valid]
    cS = S.mean(axis=0)
    cT = T.mean(axis=0)
    S_c = S - cS
    T_c = T - cT
    # Isotropic scale: minimise sum |T_c - s * S_c|^2 → s = <T_c, S_c> / <S_c, S_c>
    num = float((T_c * S_c).sum())
    den = float((S_c * S_c).sum())
    if den < 1e-8:
        return 1.0, np.zeros(3, dtype=np.float32)
    s = num / den
    return s, cT - s * cS  # transform: new = s * source + (cT - s * cS)


def _per_frame_procrustes_scale(joints_3d: np.ndarray,
                                joints_2d: np.ndarray,
                                depth_map: np.ndarray,
                                focal: float, cx: float, cy: float
                                ) -> float | None:
    """Pass-1 only: compute the per-frame Procrustes scale ratio (no apply).
    Returns None if the fit can't be done (too few valid joint depths)."""
    if joints_2d is None or not np.isfinite(joints_2d).any():
        return None
    target_3d = _backproject_joints(joints_2d, depth_map, focal, cx, cy)
    s, _ = _solve_iso_scale_translation(joints_3d, target_3d)
    if not (0.5 < s < 3.5):
        return None
    return float(s)


def _apply_uniform_scale_around_wrist(verts_3d: np.ndarray,
                                      joints_3d: np.ndarray,
                                      s: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale mesh + skeleton uniformly by `s` around the wrist (joint 0).

    Wrist position is preserved → keeps the MoGe-anchored depth that
    align_hand_to_depth set, and the per-frame translation does not need
    to be re-estimated. This is what makes a single clip-wide scale
    feasible: every frame independently scales around its own wrist.
    """
    wrist = joints_3d[0:1]                    # (1,3)
    verts_new = (verts_3d - wrist) * s + wrist
    joints_new = (joints_3d - wrist) * s + wrist
    return verts_new.astype(np.float32), joints_new.astype(np.float32)


def _process_clip(fav_dir: Path, dry_run: bool,
                  scale_threshold: float = 1.25,
                  hand_extent_max_m: float = 0.30,
                  extent_cv_threshold: float = 0.10) -> tuple[str, str, dict]:
    """Two-pass clip-wide-median rescale.

    Pass 1: walk every (frame, hand), compute Procrustes scale s_t and
            current MANO mesh extent e_t (max-axis, m). Keep cache for pass 2.
    Pass 2: derive a SINGLE clip-wide target extent E_target from
            (median(s_t), median(e_t)). For each (frame, hand), apply the
            uniform scale  s_per_frame = E_target / e_t  around the wrist.

    Why uniform-around-wrist (not full Procrustes per frame): the prior
    per-frame Procrustes path baked depth-map noise + WiLoR 2D jitter +
    pose-dependent extent into a frame-specific s_t, producing visible
    "breathing" of the rendered hand (extent range up to 4×). A single
    target extent + per-frame scale-around-wrist preserves the
    MoGe-anchored wrist position while making the hand a CONSTANT
    physical size across the clip.

    Two action paths inside one algorithm:
      - "grow"   when median(s_t) is far from 1 (fresh data needs scaling
                 from MANO template ~14 cm to a real ~25 cm).
      - "flatten" when current extents have std/median > extent_cv_threshold
                  (prior per-frame fix left jitter; flatten to median current
                  extent, no further growth).
    A clip can need either, both, or neither.
    """
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    from scripts.pipeline_utils import decode_depth_png

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fdata = data.get("frame_data") or []
    if not fdata:
        return "skip", "empty frame_data", {}

    dp_focal = float(data.get("dp_focal", 0.0))
    cx = float(data.get("cx", 0.0))
    cy = float(data.get("cy", 0.0))
    if dp_focal <= 0:
        return "fail", "no dp_focal", {}

    # ── Pass 1: collect per-frame scales + size invariants ─────────
    # We use TWO size signals per frame:
    #   ext_m   = mesh max-axis extent — driven by both pose and WiLoR β.
    #             Used ONLY to compute the per-frame scale_to_apply
    #             (target / current) so the rescale is geometrically
    #             consistent.
    #   bone_m  = wrist→middle-MCP joint distance (joints[0]→joints[9]).
    #             Pose-invariant — only changes when WiLoR's β shape
    #             prediction jitters.  This is the signal we test
    #             extent_cv against, so natural fist↔palm motion does
    #             NOT trigger flatten.
    cache = []                   # [(fi, hi, verts, j3, ext_m, bone_m), ...]
    procrustes_scales = []
    extents_m = []
    bones_m = []
    n_hands = 0
    for fi, fd in enumerate(fdata):
        v_list = fd.get("vertices_3d") or []
        j3_list = fd.get("joints_3d_pred") or []
        j2_list = fd.get("joints_2d_pred") or []
        if not v_list:
            continue
        depth_png = fd.get("depth_png")
        if depth_png is None:
            continue
        depth_map = decode_depth_png(depth_png)
        for hi, (verts, j3) in enumerate(zip(v_list, j3_list)):
            if verts is None or j3 is None:
                continue
            verts = np.asarray(verts, dtype=np.float32)
            j3 = np.asarray(j3, dtype=np.float32)
            if verts.shape != (778, 3) or j3.shape != (21, 3):
                continue
            n_hands += 1
            ext_m = float((verts.max(axis=0) - verts.min(axis=0)).max())
            bone_m = float(np.linalg.norm(j3[9] - j3[0]))   # wrist→middle-MCP
            cache.append((fi, hi, verts, j3, ext_m, bone_m))
            extents_m.append(ext_m)
            bones_m.append(bone_m)
            j2 = (np.asarray(j2_list[hi], dtype=np.float32)
                  if hi < len(j2_list) and j2_list[hi] is not None else None)
            if j2 is not None:
                s_t = _per_frame_procrustes_scale(j3, j2, depth_map, dp_focal, cx, cy)
                if s_t is not None:
                    procrustes_scales.append(s_t)

    if not extents_m:
        return "skip", "no valid hand frames", {}

    s_arr = np.array(procrustes_scales, dtype=np.float32) if procrustes_scales else np.array([1.0])
    e_arr = np.array(extents_m, dtype=np.float32)
    b_arr = np.array(bones_m, dtype=np.float32)
    s_clip = float(np.median(s_arr))
    median_ext = float(np.median(e_arr))
    median_bone = float(np.median(b_arr))
    extent_cv = float(e_arr.std() / max(e_arr.mean(), 1e-6))
    bone_cv = float(b_arr.std() / max(b_arr.mean(), 1e-6))

    # Only act on UPWARD scale (close-up clips where MANO is undersized).
    # Downward Procrustes (s < 1) would shrink MANO below realistic adult
    # hand size — that's MoGe being noisy on small/distant hands, not a
    # real correction signal.  Threshold protects ~70 clips with s ∈ [0.6, 1].
    needs_grow = s_clip >= scale_threshold
    # Flatten is gated on bone_cv (pose-invariant) not extent_cv.  Natural
    # fist↔palm motion has extent_cv up to 0.20 but bone_cv stays < 0.05.
    needs_flatten = bone_cv > extent_cv_threshold

    if not needs_grow and not needs_flatten:
        return ("skip",
                f"no action (s_clip={s_clip:.3f}, bone_cv={bone_cv:.3f}, "
                f"median_ext={median_ext*100:.1f}cm)",
                {"s_clip": s_clip, "bone_cv": bone_cv, "extent_cv": extent_cv,
                 "median_ext_cm": median_ext * 100})

    # Determine clip-wide target BONE length (wrist→middle-MCP).
    # Bone is pose-invariant, so per-frame correction = target_bone /
    # current_bone preserves natural fist↔palm motion in mesh internals
    # while equalising hand size across the clip.
    target_bone = s_clip * median_bone if needs_grow else median_bone
    # Clamp via the equivalent extent cap (bone:extent ratio ≈ 0.5 for MANO)
    bone_max = hand_extent_max_m * 0.5
    if target_bone > bone_max:
        target_bone = bone_max

    # ── Pass 2: apply per-frame uniform-around-wrist scale ──────────
    n_rescaled = 0
    per_frame_s = []
    for fi, hi, verts, j3, ext_m, bone_m in cache:
        if bone_m < 0.02:                    # 2cm — guard degenerate fits
            continue
        s_t = target_bone / bone_m
        v_new, j3_new = _apply_uniform_scale_around_wrist(verts, j3, s_t)
        fdata[fi]["vertices_3d"][hi] = v_new
        fdata[fi]["joints_3d_pred"][hi] = j3_new
        n_rescaled += 1
        per_frame_s.append(s_t)

    p_s = np.array(per_frame_s, dtype=np.float32) if per_frame_s else np.array([1.0])
    stats = {
        "action": ("grow+flatten" if (needs_grow and needs_flatten)
                   else "grow" if needs_grow else "flatten"),
        "n_hands_total": n_hands,
        "n_hands_rescaled": n_rescaled,
        "s_clip_procrustes": s_clip,
        "extent_cv_before": extent_cv,
        "bone_cv_before": bone_cv,
        "median_ext_cm_before": median_ext * 100,
        "median_bone_cm_before": median_bone * 100,
        "target_bone_cm": target_bone * 100,
        "per_frame_s_p10": float(np.percentile(p_s, 10)),
        "per_frame_s_p90": float(np.percentile(p_s, 90)),
    }

    if not dry_run:
        prov = data.get("hand_scale_refresh") or {}
        prov_history = prov.get("history") or []
        prov_history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "algo": "clip-wide-median-v2",
            **stats,
        })
        data["hand_scale_refresh"] = {"history": prov_history}
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return "ok", stats["action"], stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated clip ids (equals form for leading-dash)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report scale stats, don't modify pkl")
    args = ap.parse_args()

    allow = None
    if args.only:
        allow = set(s.strip() for s in args.only.split(",") if s.strip())

    favs = sorted([p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_")])
    if allow:
        favs = [p for p in favs if p.name in allow]

    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_hand_scale: {len(favs)} clips, dry_run={args.dry_run}")
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        elapsed = time.time() - t0
        eta = (len(favs) - i) * elapsed / max(i - 1, 1) if i > 1 else 0
        t_clip = time.time()
        status, reason, stats = _process_clip(fav, args.dry_run)
        dt = time.time() - t_clip
        if status == "ok":
            n_ok += 1
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}+ {fav.name:<50}"
                  f"  {stats['action']:>12}  s_proc={stats['s_clip_procrustes']:.2f}"
                  f"  bone: {stats['median_bone_cm_before']:.1f}→{stats['target_bone_cm']:.1f}cm"
                  f"  bone_cv {stats['bone_cv_before']:.2f} (ext_cv {stats['extent_cv_before']:.2f})"
                  f"  ({stats['n_hands_rescaled']}/{stats['n_hands_total']})  {dt:.1f}s")
        elif status == "skip":
            n_skip += 1
            print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({reason})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({reason})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")


if __name__ == "__main__":
    main()
