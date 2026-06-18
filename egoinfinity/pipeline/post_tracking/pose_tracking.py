"""
Refresh ``pose_track_info`` in pkls without re-running any upstream phase.
This tool is **strictly independent** of the rest of the pipeline:

  Reads from pkl :  sam3_mesh_info, frame_data[*].sam3_obj_data, depth_png,
                    dp_focal, cx, cy, vertices_3d (for hand mask), mano_faces
  Reads from disk:  favorites/<id>/sam3_meshes/obj_<oid>.ply
  Writes to pkl  :  data['pose_track_info'][oid]   ← only this field
                    data['pose_track_info_meta']    ← provenance

Nothing else in the pkl is touched. Run as many times as you like with
different modes / params; each run overwrites the previous tracking
output and leaves the rest of the pipeline data alone.

Modes
=====

**position_first** (default, ~1s/oid)
    Mesh translation = SAM2 mask centroid in 3D, frame-by-frame.
    Mesh rotation    = R_anchor (from anchor frame FGR+ICP), constant.
    Translation post-processing: linear interpolation across gaps,
    Gaussian smoothing in time.
    Use when you want the mesh to *follow* the SAM2 cloud's position
    without orientation tracking complications.

**full_opt** (~30s/oid)
    Stage 1 anchor → Stage 2 flow PnP → Stage 4 7-loss LBFGS optimisation.
    All loss weights exposed via CLI flags (`--lambda-fit`, `--lambda-prox`,
    etc.). Reduce ``--lambda-prox`` and ``--lambda-noslip`` to weaken
    grasp-driven contributions.

Usage
=====

::

    # First run, position-first on all clips
    python -m egoinfinity.pipeline.post_tracking.pose_tracking

    # Iterate: try full_opt with reduced grasp weights
    python -m egoinfinity.pipeline.post_tracking.pose_tracking --mode=full_opt \
        --lambda-prox=10 --lambda-noslip=5 --lambda-fit=80

    # Just one clip
    python -m egoinfinity.pipeline.post_tracking.pose_tracking "--only=-9A2VyaIkX4_105.4_109.4"

    # Skip oids that already have pose_track_info
    python -m egoinfinity.pipeline.post_tracking.pose_tracking --skip-existing
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAVORITES_DIR = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO_ROOT / "cache"))) / "favorites"

ALGO_VERSION_POSITION = "v1-position-first"
ALGO_VERSION_FULLOPT = "v1-full-opt"
ALGO_VERSION_PHASE_D = "v1-phase-d"


# ── pkl helpers (rehydrate depth) ───────────────────────────────────────
def _decode_depth_png(buf, dtype=np.float32):
    """Decode uint16 mm-scale depth PNG → float32 metres ndarray."""
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return (img.astype(dtype) / 1000.0)  # mm → m


def _unpack_mask(mask_packed, mask_shape):
    if mask_packed is None or mask_shape is None:
        return None
    shape = tuple(int(x) for x in mask_shape)
    n_total = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))
    if bits.size < n_total:
        return None
    return bits[:n_total].reshape(shape).astype(bool)


# ── position-first algorithm ────────────────────────────────────────────
def _interp_gaps_3d(t_seq):
    """Linear-interpolate Nones in a list of (3,) translations."""
    T = len(t_seq)
    arr = np.full((T, 3), np.nan, dtype=np.float64)
    for i, v in enumerate(t_seq):
        if v is not None:
            arr[i] = v
    valid = ~np.isnan(arr[:, 0])
    if valid.sum() == 0:
        return arr
    if valid.sum() == 1:
        arr[~valid] = arr[valid][0]
        return arr
    idx = np.arange(T)
    for d in range(3):
        arr[~valid, d] = np.interp(idx[~valid], idx[valid], arr[valid, d])
    return arr


def _gaussian_smooth_1d(arr, sigma):
    """Gaussian smooth (T, 3) array independently per axis."""
    if sigma <= 0:
        return arr
    half = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.empty_like(arr)
    for d in range(arr.shape[1]):
        # Reflect-padded convolution
        padded = np.pad(arr[:, d], half, mode="edge")
        out[:, d] = np.convolve(padded, k, mode="valid")
    return out


def _decode_jpg_to_rgb(buf):
    if buf is None: return None
    arr = np.frombuffer(buf, dtype=np.uint8) if isinstance(buf, (bytes, bytearray)) else None
    if arr is None or arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _track_position_first(
    mesh_pts, mask_seq, depth_seq, hand_mask_seq, K,
    smooth_sigma=2.0, anchor_t=None, run_anchor_icp=True,
    sam3d_canonical_quat=None,
    sam3d_canonical_translation=None,
    sam3d_canonical_scale=None,
    fdata_for_flow=None,
    enclose_target_pct=95.0,
):
    """Position-first mode.

    Returns dict { 'T_seq': (T,4,4), 'anchor_t': int, 'mode': ..., 'R_anchor': ..., 'note': ... }
    """
    T = len(mask_seq)
    H, W = depth_seq[0].shape if depth_seq[0] is not None else (None, None)
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    mesh_centroid = np.mean(mesh_pts, axis=0).astype(np.float64)

    # 1) Anchor selection (cheapest version: max-area frame minus hand-occluded)
    if anchor_t is None:
        scores = np.zeros(T, dtype=np.float32)
        for t in range(T):
            m = mask_seq[t]
            if m is None or not m.any():
                continue
            area = float(m.sum())
            if hand_mask_seq is not None and hand_mask_seq[t] is not None:
                inter = int((m & hand_mask_seq[t]).sum())
                hand_iou = inter / max(area, 1)
            else:
                hand_iou = 0.0
            scores[t] = area * (1.0 - hand_iou)
        if scores.max() <= 0:
            return None
        anchor_t = int(np.argmax(scores))

    # 2) Anchor pose: PURE SAM3D canonical (no refinement).
    # Direct replica of hf_reference behavior when fitness < 0.5: no
    # ICP / Umeyama / sanity checks — just trust SAM3D's monocular
    # estimate. Failures of this approach (mesh too big/small/twisted)
    # are SAM3D's responsibility; no compensation is attempted.
    if sam3d_canonical_quat is not None and sam3d_canonical_scale is not None:
        from scipy.spatial.transform import Rotation as _Rot
        q = np.asarray(sam3d_canonical_quat, dtype=np.float64)
        # SAM3D outputs the rotation in Pytorch3D convention applied in
        # row form (p_p3d @ R_quat). To use as a column-form rotation in
        # OpenCV camera frame (R @ p_opencv) we need:
        #   1. transpose:  R_quat → R_quat.T  (row→column form)
        #   2. P3D → OpenCV camera flip:  F = diag(-1, -1, 1)
        # Final:  R_anchor = F @ R_quat.T
        # This matches the original pipeline's T_init_sim construction
        # at exo_pipeline.py:2055.
        R_quat = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        F = np.diag([-1.0, -1.0, 1.0])
        R_anchor = F @ R_quat.T
        scale_correction = float(sam3d_canonical_scale)
        icp_note = f"canonical_only (F@R^T) scale={scale_correction:.3f}"
    else:
        R_anchor = np.eye(3, dtype=np.float64)
        scale_correction = 1.0
        icp_note = "no_canonical_fallback identity"

    # 3) Per-frame translation = SAM2 mask centroid - R_anchor @ (scale × mesh_centroid)
    # Rotation stays at R_anchor for all frames (no per-frame R tracking).
    # Also keep per-frame obs cloud so the state-estimator can reuse it
    # for fingertip↔obs grasp detection and pc-motion-based moving detection.
    T_seq = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    T_seq[:, :3, :3] = R_anchor
    n_observed = 0

    t_seq_raw = []
    obs_clouds: list = []   # list[Optional[(N, 3) float32]]
    for t in range(T):
        m = mask_seq[t]
        d = depth_seq[t]
        if m is None or not m.any() or d is None:
            t_seq_raw.append(None); obs_clouds.append(None); continue
        m_clean = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                             iterations=2).astype(bool)
        if m_clean.sum() < 30:
            m_clean = m
        ys, xs = np.where(m_clean)
        if len(xs) < 30:
            t_seq_raw.append(None); obs_clouds.append(None); continue
        if len(xs) > 4000:
            sel = np.random.RandomState(t).choice(len(xs), 4000, replace=False)
            xs = xs[sel]; ys = ys[sel]
        z = d[ys, xs]
        valid = (z > 1e-3) & np.isfinite(z)
        if valid.sum() < 30:
            t_seq_raw.append(None); obs_clouds.append(None); continue
        xs, ys, z = xs[valid], ys[valid], z[valid]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        pts = np.stack([x, y, z], axis=-1)
        med = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - med, axis=1)
        d_med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - d_med)))
        pts = pts[dists <= d_med + 3.0 * 1.4826 * mad]
        if len(pts) < 20:
            t_seq_raw.append(None); obs_clouds.append(None); continue
        # bbox-center (not median) for mesh-position anchoring. Median is
        # sensitive to point-cloud density distribution, which differs between
        # this tool's mask processing (2-px erode + 4000-sample) and viser
        # export's (no pre-erode, all valid pts) — observed 3 cm X offset on
        # a cutting board frame, visible as a misaligned mesh on HF. bbox-
        # center is density-invariant: differs by ~0.5 mm between the two
        # pipelines for the same mask. Matches what main pipeline does
        # (pose_tracker/object_motion.py:compute_pc_motion_per_frame).
        c_obs = ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float64)
        mesh_centroid_world = R_anchor @ (scale_correction * mesh_centroid)
        t_xyz = c_obs - mesh_centroid_world
        t_seq_raw.append(t_xyz)
        obs_clouds.append(pts.astype(np.float32))
        n_observed += 1

    if n_observed < 1:
        return None
    t_seq = _interp_gaps_3d(t_seq_raw)
    t_seq = _gaussian_smooth_1d(t_seq, sigma=smooth_sigma)
    T_seq[:, :3, 3] = t_seq

    return {
        "mode": "position_first",
        "algo_version": ALGO_VERSION_POSITION,
        "anchor_t": anchor_t,
        "smooth_sigma": float(smooth_sigma),
        "n_observed": int(n_observed),
        "scale_correction": float(scale_correction),
        "T_seq": T_seq.astype(np.float32),
        "tracking_status": "ok",
        "note": icp_note,
        "_obs_clouds": obs_clouds,   # leading underscore = not persisted to pkl
    }


# ── State estimation (STATIC / GRASPED / MOVING) ───────────────────────
#
# Hierarchical: first ask "is this object globally static in 2D?". If yes,
# state="static" for every frame regardless of hand proximity. If no, then
# per-frame split between GRASPED (hand fingertip ≤ threshold of the obs
# cloud) and MOVING (everything else in the non-static segment).
#
# Why 2D for the static gate:
#   - SAM2 mask centroid is fast (mask is already in pkl as packed bits) and
#     free of MoGe-2 depth noise.
#   - Optical flow would catch in-plane slide on a flat surface too, but
#     pair_mag_list isn't persisted to pkl; recomputing it per refresh
#     would break the "seconds per clip" design.  Mask centroid is the
#     coarse-but-cheap proxy: it catches translation, misses pure
#     rotation-in-place — fine for the static/non-static distinction.
#   - Robust against MoGe z-noise (typical 1.5 cm/frame on stationary
#     objects) that the previous 3D-bbox-center approach had to fight
#     with via 5x looser z thresholds.
#
# Static gate combines two checks:
#   1. GLOBAL: p10-p90 span of mask 2D centroids ≤ STATIC_GLOBAL_PX_FRAC ×
#      min(H, W).  Trimmed range tolerates short SAM2 mask drift / fingertip
#      occlusion that briefly jumps the centroid.
#   2. PER-FRAME (only when 1 fails): Schmitt-trigger hysteresis on
#      ||centroid[t] - centroid[t-1]||.  Lets a non-static object still be
#      labelled STATIC during its quiet periods.
#
# GRASPED detection (unchanged): fingertip ≤ FINGERTIP_THRESHOLD_M of the
# 3D obs cloud, then morphological close+drop to bridge brief drops.  Uses
# 3D fingertip↔cloud distance not 2D mask overlap because the 2D hand mask
# isn't tracked here, and fingertip Z is fine even when the object surface
# Z is noisy.
#
# Final state (per frame):
#   static_global  →  "static"
#   else if grasp[t] → "grasped"
#   else if moving[t] → "moving"
#   else              → "static"  (object's quiet period inside a busy clip)

FINGERTIP_JOINT_IDX = [4, 8, 12, 16, 20]   # MANO/OpenPose tips: thumb,index,middle,ring,pinky


def _hysteresis(x, low, high):
    """Schmitt trigger: turn on when x > high, turn off when x < low.

    Single-pass O(T); state-preserving across frames so isolated noise
    above ``high`` doesn't flip when surrounded by low values.
    """
    T = len(x)
    out = np.zeros(T, dtype=bool)
    on = False
    for t in range(T):
        if on:
            if x[t] < low:
                on = False
        else:
            if x[t] > high:
                on = True
        out[t] = on
    return out


def _morpho_close_drop(b, bridge: int, min_run: int):
    """1D morphological close (bridge ≤ ``bridge`` internal 0-runs) then
    drop True runs shorter than ``min_run``.

    Edge-bounded gaps (start or end of sequence) are NOT bridged.
    """
    T = len(b)
    out = b.copy().astype(bool)
    # Close: fill internal 0-runs ≤ bridge
    i = 0
    while i < T:
        if not out[i]:
            j = i
            while j < T and not out[j]:
                j += 1
            run_len = j - i
            if i > 0 and j < T and out[i - 1] and out[j] and run_len <= bridge:
                out[i:j] = True
            i = j
        else:
            i += 1
    # Drop short True runs
    i = 0
    while i < T:
        if out[i]:
            j = i
            while j < T and out[j]:
                j += 1
            run_len = j - i
            if run_len < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _unpack_mask_2d(packed: np.ndarray, shape) -> np.ndarray:
    """packbits decoder for SAM2 masks stored in sam3_obj_data.

    Returns (H, W) bool. None if the input is invalid.
    """
    if packed is None or shape is None:
        return None
    H, W = int(shape[0]), int(shape[1])
    flat = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: H * W]
    return flat.astype(bool).reshape(H, W)


def _mask_centroid_2d(mask: np.ndarray) -> np.ndarray | None:
    """2D mask "center" used as the per-frame motion proxy.

    Uses **bbox midpoint** instead of pixel centroid: midpoint of
    (xmin, xmax) and (ymin, ymax).  Compared with the pixel mean, bbox
    midpoint is robust to the most common occlusion pattern in cooking
    videos — the hand sweeping over the *interior* of an object (e.g. hand
    on cutting board) — where the mask becomes annular but the outer
    bounds don't move.  It does still drift when the occluder eats into
    one side of the object (side-occlusion), but that's a less common
    case and the global p10-p90 trimming further reduces its impact.

    Returns (cx, cy) or None if the mask is empty / invalid.
    """
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    return np.array([
        0.5 * (float(xs.min()) + float(xs.max())),
        0.5 * (float(ys.min()) + float(ys.max())),
    ], dtype=np.float32)


def _estimate_state_per_frame(
    obs_clouds,                # list[Optional[np.ndarray]]: per-frame obs cloud (for grasp)
    joints_pf,                 # list[list[(21, 3) ndarray]]: per-frame hand joints
    hand_is_right_pf,          # list[list[bool]]: True=R-hand, False=L-hand, parallel to joints_pf
    masks_2d,                  # list[Optional[(H,W) bool]]: per-frame SAM2 mask
    img_hw,                    # (H, W) image size
    *,
    fingertip_threshold_m: float = 0.06,
    wrist_threshold_m: float = 0.05,           # 5cm — wrist near obs → grasp fallback
    mask_overlap_px: int = 30,                 # hand-mask ∩ obj-mask → 2D grasp
    grasp_bridge_frames: int = 30,             # was 10; 30 ≈ 2s @ 15fps — bridges
                                                # long pour / step-back gaps
    grasp_min_frames: int = 8,
    static_global_px_frac: float = 0.02,
    static_frame_low_px: float = 2.0,
    static_frame_high_px: float = 4.0,
    # Per-hand 2D mask rendering inputs (optional — when given, 2D mask ∩
    # mask_overlap_px is OR'd into the close[t] signal alongside the 3D
    # fingertip and wrist proximity checks).
    hand_verts_pf=None,                        # list[list[(778, 3)]]
    mano_faces=None,                           # (1538, 3) int
    K=None,                                    # (3, 3) intrinsics
):
    """Hierarchical per-frame state estimation:
       static_global ?
         → all "static"
         ↳ else: per-frame {grasped_l | grasped_r | grasped_both | moving | static-quiet}

    L/R disambiguation: per-frame the fingertip-close check is split by the
    hand_is_right flag.  Each hand's close[t] is morpho-filtered
    independently (so a brief mid-grasp drop in one hand doesn't kill its
    grasp run, while the other hand stays unaffected).

    Returns dict with:
      state_per_frame:        list[str] in {"static","grasped_l","grasped_r","grasped_both","moving"}
      grasp_hand_per_frame:   list[str|None] in {"L","R","both",None}
      is_static_global:       bool
      centroid_2d_per_frame:  list[[x,y]|None]
      state_counts:           {static, grasped_l, grasped_r, grasped_both, moving}
      # legacy fields (back-compat with consumers expecting single grasp bool):
      wrist_used_per_frame:   list[bool]
      wrist_l_per_frame:      list[bool]
      wrist_r_per_frame:      list[bool]
      is_moving_per_frame:    list[bool]
      close_per_frame:        list[bool]
    """
    T = len(obs_clouds)
    H, W = img_hw
    min_hw = float(min(H, W))

    # 1) Per-frame 2D mask centroid, NaN-tolerant
    centroids: list[np.ndarray | None] = [None] * T
    for t in range(T):
        m = masks_2d[t] if t < len(masks_2d) else None
        centroids[t] = _mask_centroid_2d(m)
    # Carry-forward through gaps (so disp diff doesn't spike when mask
    # is briefly missing).  We keep `centroids_raw` (with None) for the
    # report and use `centroids_filled` for differencing.
    centroids_filled: list[np.ndarray | None] = list(centroids)
    last_valid = None
    for t in range(T):
        if centroids_filled[t] is not None:
            last_valid = centroids_filled[t]
        elif last_valid is not None:
            centroids_filled[t] = last_valid

    # 2) GLOBAL static gate: p10-p90 span over the clip.
    valid_xy = np.array([c for c in centroids if c is not None], dtype=np.float32)
    static_global = False
    global_span_px = float("nan")
    if len(valid_xy) >= max(8, int(0.1 * T)):
        # Use 10th/90th percentile to ignore brief outliers
        p10 = np.percentile(valid_xy, 10, axis=0)
        p90 = np.percentile(valid_xy, 90, axis=0)
        global_span_px = float(np.linalg.norm(p90 - p10))
        static_global = global_span_px <= static_global_px_frac * min_hw

    # 3) Per-frame disp + Schmitt-trigger
    disp = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        a = centroids_filled[t - 1]
        b = centroids_filled[t]
        if a is None or b is None:
            continue
        disp[t] = float(np.linalg.norm(b - a))
    is_moving_2d = _hysteresis(disp, low=static_frame_low_px, high=static_frame_high_px)

    # 4) GRASP detection — combine three signals (any → close):
    #    (a) 2D primary: hand-mask ∩ obj-mask ≥ mask_overlap_px (depth-free)
    #    (b) 3D fingertip ≤ fingertip_threshold_m of obs cloud
    #    (c) 3D wrist ≤ wrist_threshold_m of obs cloud (handles full-occlusion)
    #
    #    2D is the primary because it's robust to MoGe depth noise and to
    #    the hand-mask-subtraction step that eats the obs cloud when the
    #    hand wraps around the object (causing the pour-drink clip's
    #    grasp gaps).  3D fingertip / wrist are kept as fallbacks for
    #    frames where the hand mask isn't fully rendered (e.g., partial
    #    WiLoR vertex detections).
    close_l = np.zeros(T, dtype=bool)
    close_r = np.zeros(T, dtype=bool)
    use_2d = (hand_verts_pf is not None and mano_faces is not None
              and K is not None)
    if use_2d:
        from egoinfinity.pipeline.pose_tracker.utils import render_hand_mask
    for t in range(T):
        obs = obs_clouds[t]
        hands = joints_pf[t] if t < len(joints_pf) else []
        is_right_arr = hand_is_right_pf[t] if t < len(hand_is_right_pf) else []
        if not hands:
            continue

        # 2D mask overlap (signal a) — runs first since it's primary
        if use_2d:
            obj_m = masks_2d[t] if t < len(masks_2d) else None
            verts_list = hand_verts_pf[t] if t < len(hand_verts_pf) else []
            if obj_m is not None and obj_m.any() and verts_list:
                for hi, hv in enumerate(verts_list):
                    if hv is None:
                        continue
                    hv_arr = np.asarray(hv)
                    if hv_arr.shape != (778, 3):
                        continue
                    is_right = bool(is_right_arr[hi]) if hi < len(is_right_arr) else False
                    try:
                        hand_m = render_hand_mask([hv_arr], np.asarray(mano_faces),
                                                   K, H, W, dilate_px=3)
                    except Exception:
                        continue
                    overlap = int((obj_m & hand_m).sum())
                    if overlap >= mask_overlap_px:
                        if is_right:
                            close_r[t] = True
                        else:
                            close_l[t] = True

        # 3D signals (b) + (c) — fingertip and wrist proximity
        if obs is not None and len(obs) >= 5:
            obs64 = np.asarray(obs, dtype=np.float64)
            for hi, j3d in enumerate(hands):
                j = np.asarray(j3d, dtype=np.float64)
                if j.shape != (21, 3) or not np.all(np.isfinite(j)):
                    continue
                is_right = bool(is_right_arr[hi]) if hi < len(is_right_arr) else False
                # Fingertip min-d
                tip_min_d = np.inf
                for ftip in FINGERTIP_JOINT_IDX:
                    d = float(np.min(np.linalg.norm(obs64 - j[ftip], axis=1)))
                    if d < tip_min_d:
                        tip_min_d = d
                        if tip_min_d <= fingertip_threshold_m:
                            break
                if tip_min_d <= fingertip_threshold_m:
                    if is_right: close_r[t] = True
                    else: close_l[t] = True
                # Wrist proximity (handles full-occlusion frames where
                # fingertips end up far due to hand-mask-subtracted obs
                # cloud being empty / mis-localised)
                wrist_d = float(np.min(np.linalg.norm(obs64 - j[0], axis=1)))
                if wrist_d <= wrist_threshold_m:
                    if is_right: close_r[t] = True
                    else: close_l[t] = True
    # Morpho-filter each hand independently
    wrist_l = _morpho_close_drop(
        close_l, bridge=grasp_bridge_frames, min_run=grasp_min_frames)
    wrist_r = _morpho_close_drop(
        close_r, bridge=grasp_bridge_frames, min_run=grasp_min_frames)
    wrist_used = wrist_l | wrist_r          # legacy single-hand bool
    close = close_l | close_r               # legacy single-hand raw

    # 5) Compose final state hierarchically (5 states: static / grasped_l /
    #    grasped_r / grasped_both / moving)
    states: list[str] = []
    grasp_hand: list = []                   # "L", "R", "both", or None
    if static_global:
        states = ["static"] * T
        grasp_hand = [None] * T
    else:
        for t in range(T):
            if wrist_l[t] and wrist_r[t]:
                states.append("grasped_both")
                grasp_hand.append("both")
            elif wrist_l[t]:
                states.append("grasped_l")
                grasp_hand.append("L")
            elif wrist_r[t]:
                states.append("grasped_r")
                grasp_hand.append("R")
            elif is_moving_2d[t]:
                states.append("moving")
                grasp_hand.append(None)
            else:
                states.append("static")  # object's quiet period
                grasp_hand.append(None)

    is_moving_legacy = is_moving_2d & ~wrist_used

    from collections import Counter
    state_counts_full = Counter(states)
    state_counts = {
        "static": state_counts_full.get("static", 0),
        "grasped_l": state_counts_full.get("grasped_l", 0),
        "grasped_r": state_counts_full.get("grasped_r", 0),
        "grasped_both": state_counts_full.get("grasped_both", 0),
        "moving": state_counts_full.get("moving", 0),
    }

    centroid_2d_list: list = []
    for c in centroids:
        centroid_2d_list.append([float(c[0]), float(c[1])] if c is not None else None)

    return {
        "state_per_frame": states,
        "grasp_hand_per_frame": grasp_hand,
        "is_static_global": bool(static_global),
        "centroid_2d_per_frame": centroid_2d_list,
        "centroid_2d_disp_per_frame": disp.tolist(),
        "global_span_px": global_span_px,
        "state_counts": state_counts,
        # legacy
        "wrist_used_per_frame": [bool(v) for v in wrist_used],
        "wrist_l_per_frame": [bool(v) for v in wrist_l],
        "wrist_r_per_frame": [bool(v) for v in wrist_r],
        "is_moving_per_frame": [bool(v) for v in is_moving_legacy],
        "close_per_frame": [bool(v) for v in close],
    }


# ── full-opt (legacy 7-loss) wrapper ────────────────────────────────────
def _track_full_opt(
    mesh_pts, mask_seq, depth_seq, hand_mask_seq, K, frames_rgb,
    lambdas, anchor_t=None,
):
    """Use existing flow_pnp.track_6dof + fill_optimize with custom λs."""
    from egoinfinity.pipeline.pose_tracker import (
        select_anchor_frame, estimate_anchor_pose, track_6dof,
    )
    from egoinfinity.pipeline.pose_tracker.fill_optimize import (
        optimize_pose_seq, DEFAULT_LAMBDAS,
    )
    if anchor_t is None:
        anchor_t = select_anchor_frame(mask_seq, hand_mask_seq)
    if anchor_t < 0:
        return None
    anchor_res = estimate_anchor_pose(
        mesh_pts.astype(np.float64),
        depth_seq[anchor_t], mask_seq[anchor_t],
        hand_mask_seq[anchor_t] if hand_mask_seq else None, K,
    )
    if anchor_res is None or anchor_res.fitness < 0.05:
        return None
    track_res = track_6dof(
        mesh_pts.astype(np.float64),
        anchor_res.T, anchor_t, frames_rgb, mask_seq, depth_seq, K,
    )
    if track_res is None:
        return None
    final_lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    opt_res = optimize_pose_seq(
        track_res.T_seq, mesh_pts.astype(np.float64),
        mask_seq, depth_seq, hand_mask_seq, K,
        lambdas=final_lambdas,
    )
    return {
        "mode": "full_opt",
        "algo_version": ALGO_VERSION_FULLOPT,
        "anchor_t": int(anchor_t),
        "lambdas": final_lambdas,
        "T_seq": opt_res.T_seq.astype(np.float32),
        "tracking_status": "ok",
        "note": f"icp_fitness={anchor_res.fitness:.2f}",
    }


# ── phase_d algorithm (state-machine: WRIST / STATIC LOCK / DEPTH_TRACKED) ──
def _track_phase_d(
    mesh_pts,                        # raw PLY xyz (N, 3) — scale-correction applied via sam3d_canonical_scale
    mask_seq,                        # per-frame SAM2 masks (T,)
    depth_seq,                       # per-frame depth maps (T,)
    hand_mask_seq,                   # per-frame hand masks (T,) from MANO render
    K,                               # camera intrinsics (3, 3)
    joints_pf,                       # per-frame joints_3d_pred lists
    hand_is_right_pf,                # per-frame handedness lists
    sam3d_canonical_quat,            # (4,) [w,x,y,z] from sam3_mesh_info[oid]
    sam3d_canonical_scale,           # scalar
    *,
    smooth_win: int = 11,
    smooth_poly: int = 3,
    enable_z_align: bool = False,            # Phase 2 flag (not used yet)
    grasp_point_per_frame=None,              # Phase 2 input (not used yet)
    existing_wrist_l_per_frame=None,         # ★ vetoed grasp from prior run, if available
    existing_wrist_r_per_frame=None,         # ★ vetoed grasp from prior run, if available
    existing_is_moving_per_frame=None,       # ★ 2D-based is_moving from prior _estimate_state_per_frame
                                              #   When given, state machine uses this instead of
                                              #   3D pc_motion (avoids MoGe-noise-driven false motion
                                              #   that makes "static" objects jitter via DEPTH_TRACKED).
):
    """Faithful port of ``scripts/exo_pipeline.py`` Phase D state machine to
    the refresh path.

    State machine per-frame:
        wrist_used[t] → T_wrist_seq[t]   (rigid_wrist_binding_propagation)
        else if static → static lock (lock_pose_for_segment / borrow neighbour)
        else           → T_depth_seq[t] (compute_depth_pose_per_frame)

    Then SavGol smooth_se3 (win=11, poly=3).

    Two deliberate differences from the main pipeline's Phase D:
      • is_moving uses pc_motion only (no optical flow — refresh path doesn't
        recompute RAFT/MEMFOF).  Boundary cases between "slowly moving" and
        "static" can differ by a few frames.
      • bg_template / bg_static_pf overrides skipped.  Affects only the few
        boundary frames where pc_motion is below threshold AND the object's
        depth profile matches its rest depth.

    Both differences are intentional — they shave dev time without changing
    the visual quality on the vast majority of clips.
    """
    from egoinfinity.pipeline.pose_tracker import (
        compute_depth_pose_per_frame,
        compute_obs_obb_per_frame,
        rigid_wrist_binding_propagation,
        lock_pose_for_segment,
        find_continuous_segments,
        hysteresis_filter,
        smooth_se3_savgol,
        complete_static_masks,
        detect_grasp_fingertip_persistent,
    )
    from egoinfinity.pipeline.pose_tracker.object_motion import compute_pc_motion_per_frame
    from egoinfinity.pipeline.object_tracker import mask_to_pointcloud

    T = len(mask_seq)
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    # ── Scale-correct mesh to world units, build R_anchor ──
    scale_correction = float(sam3d_canonical_scale) if sam3d_canonical_scale is not None else 1.0
    mesh_pts_world = mesh_pts.astype(np.float64) * scale_correction
    if sam3d_canonical_quat is not None:
        from scipy.spatial.transform import Rotation as _Rot
        q = np.asarray(sam3d_canonical_quat, dtype=np.float64)
        R_quat = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        F = np.diag([-1.0, -1.0, 1.0])
        R_anchor = F @ R_quat.T
    else:
        R_anchor = np.eye(3, dtype=np.float64)

    # ── Two cloud variants ──
    # (a) for depth-pose tracking / OBB: SAM2 mask MINUS hand mask, eroded.
    # (b) for grasp detection: RAW SAM2 mask (no hand subtract, no completion),
    #     step=2 sampling (matches scripts/exo_pipeline.py:2563 grasp_obs_clouds).
    obs_clouds_for_depth = []
    grasp_obs_clouds = []
    for t in range(T):
        m = mask_seq[t]
        d = depth_seq[t]
        if m is None or not m.any() or d is None:
            obs_clouds_for_depth.append(None)
            grasp_obs_clouds.append(None)
            continue
        # (b) raw cloud for grasp detection
        try:
            raw_pts = mask_to_pointcloud(m, d, fx, cx, cy, step=2)
            grasp_obs_clouds.append(raw_pts if len(raw_pts) >= 30 else None)
        except Exception:
            grasp_obs_clouds.append(None)
        # (a) hand-subtracted cloud for depth pose
        h = hand_mask_seq[t] if t < len(hand_mask_seq) else None
        m_eff = (m & ~h) if (h is not None) else m
        m_clean = cv2.erode(
            m_eff.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
        ).astype(bool)
        if m_clean.sum() < 30:
            m_clean = m_eff
        ys, xs = np.where(m_clean)
        if len(xs) < 30:
            obs_clouds_for_depth.append(None)
            continue
        if len(xs) > 4000:
            sel = np.random.RandomState(t).choice(len(xs), 4000, replace=False)
            xs = xs[sel]; ys = ys[sel]
        zz = d[ys, xs]
        valid = (zz > 1e-3) & np.isfinite(zz)
        if valid.sum() < 30:
            obs_clouds_for_depth.append(None)
            continue
        xs, ys, zz = xs[valid], ys[valid], zz[valid]
        xc = (xs - cx) * zz / fx
        yc = (ys - cy) * zz / fy
        pts = np.stack([xc, yc, zz], axis=-1)
        med = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - med, axis=1)
        d_med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - d_med)))
        pts = pts[dists <= d_med + 3.0 * 1.4826 * mad]
        if len(pts) < 20:
            obs_clouds_for_depth.append(None)
            continue
        obs_clouds_for_depth.append(pts.astype(np.float32))

    # ── Anchor frame: max area minus hand occlusion ──
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = mask_seq[t]
        if m is None or not m.any():
            continue
        area = float(m.sum())
        h = hand_mask_seq[t] if hand_mask_seq is not None and t < len(hand_mask_seq) else None
        hand_iou = (float((m & h).sum()) / max(area, 1.0)) if h is not None else 0.0
        scores[t] = area * (1.0 - hand_iou)
    if scores.max() <= 0:
        return None
    anchor_t = int(np.argmax(scores))

    # ── Per-frame obs OBB (sign-corrected against anchor) ──
    obs_obb_pf = compute_obs_obb_per_frame(obs_clouds_for_depth, anchor_idx=anchor_t)
    R_obb_anchor = None
    obb_a = obs_obb_pf[anchor_t] if 0 <= anchor_t < T else None
    if obb_a is not None and obb_a.get("trustworthy"):
        R_obb_anchor = np.asarray(obb_a["R"], dtype=np.float64)

    # ── Mask completion (only used to verify mc_is_static; obs cloud already
    #    computed from raw mask above) ──
    try:
        _completed, mc_is_static, _diag = complete_static_masks(mask_seq, hand_mask_seq)
    except Exception:
        mc_is_static = False

    # ── Grasp signal ──
    # If a prior run already wrote vetoed wrist_l/r_per_frame (via
    # egoinfinity.pipeline.post_tracking.grasp_veto), use those — the state machine below should be
    # driven by the *vetoed* grasp so FP-grasp frames don't trigger WRIST mode.
    # Otherwise (first / bootstrap run, no veto yet) fall back to the original
    # fingertip-persistent detector — equivalent to the main pipeline's Phase D.
    use_external_grasp = (
        existing_wrist_l_per_frame is not None
        and existing_wrist_r_per_frame is not None
        and len(existing_wrist_l_per_frame) == T
        and len(existing_wrist_r_per_frame) == T
    )
    if use_external_grasp:
        contact_soft = np.zeros((T, 2), dtype=np.float32)
        contact_soft[:, 0] = np.asarray(existing_wrist_l_per_frame, dtype=np.float32)
        contact_soft[:, 1] = np.asarray(existing_wrist_r_per_frame, dtype=np.float32)
        grasp_src = "vetoed (pkl)"
    else:
        contact_soft = detect_grasp_fingertip_persistent(
            obs_clouds=grasp_obs_clouds,
            joints_per_frame=joints_pf,
            hand_is_right_per_frame=hand_is_right_pf,
            fingertip_threshold_m=0.06,
            bridge_gap_frames=10,
            min_grasp_frames=8,
        )
        grasp_src = "fingertip_persistent (bootstrap)"

    # ── Motion / is_moving ──
    # Primary: 2D mask-centroid based signal from prior _estimate_state_per_frame
    # (matches what state_per_frame displays in viser).  This is immune to MoGe
    # depth noise on stationary objects, so static-state objects stay locked
    # instead of jittering via DEPTH_TRACKED mode.
    # Fallback: 3D pc_motion (Pass 1 / first run when no prior state exists).
    pc_motion_pf = compute_pc_motion_per_frame(obs_clouds_for_depth)
    pc_moving_pf = hysteresis_filter(pc_motion_pf, low=0.010, high=0.020)
    if (existing_is_moving_per_frame is not None
            and len(existing_is_moving_per_frame) == T):
        is_moving_pf = np.asarray(existing_is_moving_per_frame, dtype=bool)
        is_moving_src = "2D-centroid (pkl)"
    else:
        is_moving_pf = pc_moving_pf
        is_moving_src = "3D pc_motion (bootstrap)"

    # ── Fallback pose (used inside T_wrist computation as T_seq_in) ──
    fallback_pose = np.eye(4, dtype=np.float64)
    fallback_pose[:3, :3] = R_anchor
    # Set a sensible t: anchor frame's cloud bbox-center
    obs_a = obs_clouds_for_depth[anchor_t]
    if obs_a is not None:
        mesh_bbox_canon = (mesh_pts_world.min(0) + mesh_pts_world.max(0)) * 0.5
        target = (obs_a.min(0) + obs_a.max(0)) * 0.5
        fallback_pose[:3, 3] = target - R_anchor @ mesh_bbox_canon
    fallback_pose_seq = [fallback_pose.copy() for _ in range(T)]

    # ── T_depth_seq: per-frame depth-based pose ──
    T_depth_seq = compute_depth_pose_per_frame(
        R_anchor=R_anchor,
        mesh_pts=mesh_pts_world,
        obs_clouds=obs_clouds_for_depth,
        fallback_pose_seq=fallback_pose_seq,
        obs_obb_per_frame=obs_obb_pf,
        R_obb_anchor=R_obb_anchor,
        slerp_alpha=0.3,
        require_trust_consecutive=3,
    )

    # ── T_wrist_seq: rigid binding during grasp ──
    # pc_stable_per_frame = all False → t is ALWAYS palm-anchored during
    # grasp.  Matches scripts/exo_pipeline.py:2750 "5-1 reset" comment.
    T_wrist_seq, _hd_diag = rigid_wrist_binding_propagation(
        T_seq_in=T_depth_seq,
        mesh_pts=mesh_pts_world,
        R_obj_anchor=R_anchor,
        obj_masks=mask_seq,
        hand_masks=hand_mask_seq,
        obs_clouds=obs_clouds_for_depth,
        pc_stable_per_frame=np.zeros(T, dtype=bool),
        joints_per_frame=joints_pf,
        hand_is_right_per_frame=hand_is_right_pf,
        contact_soft=contact_soft,
        K=K,
        min_seg_len=3,
        soft_threshold=0.3,
    )

    # ── Per-frame mode selection (= Phase D logic verbatim) ──
    grasp_strong_pf = hysteresis_filter(contact_soft.max(axis=1), low=0.2, high=0.4)
    wrist_used_pf = grasp_strong_pf

    T_final_seq = [None] * T
    n_wrist = n_static = n_tracked = 0
    for i in range(T):
        if wrist_used_pf[i]:
            T_final_seq[i] = np.asarray(T_wrist_seq[i], dtype=np.float64).copy()
            n_wrist += 1
        elif not is_moving_pf[i]:
            T_final_seq[i] = None
            n_static += 1
        else:
            T_final_seq[i] = np.asarray(T_depth_seq[i], dtype=np.float64).copy()
            n_tracked += 1

    # ── Static lock per static segment ──
    static_segs = find_continuous_segments(~is_moving_pf, min_len=3)
    for (s, e) in static_segs:
        idxs_static = [t for t in range(s, e + 1) if not wrist_used_pf[t]]
        if not idxs_static:
            continue
        _s_idx = idxs_static[0]; _e_idx = idxs_static[-1]
        lock_T = None
        if _s_idx > 0 and wrist_used_pf[_s_idx - 1]:
            lock_T = np.asarray(T_wrist_seq[_s_idx - 1], dtype=np.float64).copy()
        elif _e_idx < T - 1 and wrist_used_pf[_e_idx + 1]:
            lock_T = np.asarray(T_wrist_seq[_e_idx + 1], dtype=np.float64).copy()
        if lock_T is None:
            seg_T_depth = [T_depth_seq[t] for t in idxs_static]
            lock_T = lock_pose_for_segment(seg_T_depth, 0, len(seg_T_depth) - 1, R_anchor)
        for t in idxs_static:
            T_final_seq[t] = lock_T.copy()

    # ── Any remaining None → T_depth fallback ──
    for i in range(T):
        if T_final_seq[i] is None:
            T_final_seq[i] = np.asarray(T_depth_seq[i], dtype=np.float64).copy()

    # ── SavGol SE3 smooth ──
    T_arr = np.stack(T_final_seq, axis=0)
    T_smooth = smooth_se3_savgol(T_arr, win=smooth_win, poly=smooth_poly)

    n_observed = int(sum(1 for o in obs_clouds_for_depth if o is not None))
    note = (f"phase_d state-machine: wrist={n_wrist} static={n_static} "
            f"tracked={n_tracked}  mc_static={bool(mc_is_static)}  "
            f"scale={scale_correction:.3f}  grasp_src={grasp_src}")

    return {
        "mode": "phase_d",
        "algo_version": ALGO_VERSION_PHASE_D,
        "anchor_t": anchor_t,
        "smooth_win": int(smooth_win),
        "smooth_poly": int(smooth_poly),
        "n_observed": n_observed,
        "scale_correction": scale_correction,
        "T_seq": T_smooth.astype(np.float32),
        "tracking_status": "ok",
        "note": note,
        "_obs_clouds": obs_clouds_for_depth,    # not persisted (underscore)
        # diagnostics for the GUI / debug:
        "n_frames_wrist": int(n_wrist),
        "n_frames_static": int(n_static),
        "n_frames_tracked": int(n_tracked),
    }


# ── orchestration ──────────────────────────────────────────────────────
def _process_clip(fav_dir: Path, mode: str, args, skip_existing: bool):
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fdata = data.get("frame_data") or []
    sam3_mesh_info = data.get("sam3_mesh_info") or {}
    if not fdata or not sam3_mesh_info:
        return "skip", "no frame_data or no meshes", {}

    dp_focal = float(data.get("dp_focal", 0))
    if dp_focal <= 0:
        return "fail", "no dp_focal", {}

    # Determine H, W from first available source
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
        # fallback: decode depth_png
        for fd in fdata:
            d = _decode_depth_png(fd.get("depth_png"))
            if d is not None:
                H, W = d.shape
                break
    if H is None:
        return "fail", "could not determine image dims", {}

    cx = float(data.get("cx", W / 2.0))
    cy = float(data.get("cy", H / 2.0))
    K = np.array([[dp_focal, 0, cx], [0, dp_focal, cy], [0, 0, 1]], dtype=np.float64)
    mano_faces = data.get("mano_faces")

    # Per-frame hand joints (for state estimation — fingertip↔obs distance).
    # joints_3d_pred is in WiLoR camera frame (joints_3d_rel + smoothed_cam_t),
    # same frame as the obs cloud we backproject from depth.
    joints_pf = [fd.get("joints_3d_pred") or [] for fd in fdata]
    hand_is_right_pf = [fd.get("hand_is_right") or [] for fd in fdata]

    # Decode all frames once (reuse for both anchor and per-frame loops)
    depth_seq = []
    hand_mask_seq = []
    frames_rgb = []
    for fd in fdata:
        depth_seq.append(_decode_depth_png(fd.get("depth_png")))
        # hand mask from vertices_3d
        if mano_faces is not None and fd.get("vertices_3d"):
            verts = [v for v in fd["vertices_3d"] if v is not None and len(v) > 0]
            if verts:
                from egoinfinity.pipeline.pose_tracker.utils import render_hand_mask
                try:
                    hand_mask_seq.append(render_hand_mask(
                        verts, np.asarray(mano_faces), K, H, W, dilate_px=3))
                except Exception:
                    hand_mask_seq.append(None)
            else:
                hand_mask_seq.append(None)
        else:
            hand_mask_seq.append(None)
        # rgb only needed for full_opt; lazy decode there
        frames_rgb.append(None)

    pose_track_info = data.get("pose_track_info") or {}
    per_oid_results = {}

    for oid_str, meta in sam3_mesh_info.items():
        oid = int(oid_str)
        ply_path_str = meta.get("ply_path")
        if not ply_path_str:
            per_oid_results[oid] = ("skip", "no ply_path in mesh_info")
            continue
        ply_path = Path(ply_path_str)
        if not ply_path.is_absolute():
            ply_path = fav_dir / ply_path
        if not ply_path.exists():
            per_oid_results[oid] = ("skip", f"missing PLY {ply_path}")
            continue

        if skip_existing and oid in pose_track_info:
            per_oid_results[oid] = ("skip", "pose_track_info exists")
            continue

        # Load mesh
        from egoinfinity.pipeline.pose_tracker.utils import load_mesh_points_from_ply
        mesh_pts, _ = load_mesh_points_from_ply(str(ply_path))
        # IMPORTANT: keep raw PLY xyz (do NOT pre-multiply by canonical_scale).
        # The tracker's scale_correction is now the ABSOLUTE factor: it
        # converts raw PLY units → world metres. Matches the A100 hf_reference
        # convention. Viser export just does `xyz_world = xyz_raw × scale_correction`.
        mesh_pts = mesh_pts.astype(np.float64)

        # Build per-frame mask sequence for this oid
        mask_seq = []
        for fd in fdata:
            sd = fd.get("sam3_obj_data") or {}
            od = sd.get(oid) or sd.get(int(oid))
            if isinstance(od, dict):
                mask_seq.append(_unpack_mask(od.get("mask_packed"), od.get("mask_shape")))
            else:
                mask_seq.append(None)

        n_with_mask = sum(1 for m in mask_seq if m is not None and m.any())
        if n_with_mask < 5:
            per_oid_results[oid] = ("fail", f"only {n_with_mask} frames have mask")
            continue

        try:
            if mode == "position_first":
                # Pass all SAM3D canonical fields so the anchor estimator
                # can build T_init_sim (the same prior the original
                # pipeline used at Phase D-track).
                q_can = meta.get("canonical_rotation_quat") if isinstance(meta, dict) else None
                t_can = meta.get("canonical_translation") if isinstance(meta, dict) else None
                s_can = meta.get("canonical_scale") if isinstance(meta, dict) else None
                res = _track_position_first(
                    mesh_pts, mask_seq, depth_seq, hand_mask_seq, K,
                    smooth_sigma=args.smooth_sigma,
                    run_anchor_icp=not args.no_icp,
                    sam3d_canonical_quat=q_can,
                    sam3d_canonical_translation=t_can,
                    sam3d_canonical_scale=s_can,
                    fdata_for_flow=fdata,
                    enclose_target_pct=args.enclose_pct,
                )
            elif mode == "phase_d":
                q_can = meta.get("canonical_rotation_quat") if isinstance(meta, dict) else None
                s_can = meta.get("canonical_scale") if isinstance(meta, dict) else None
                # If pkl already has vetoed wrist_l/r from a previous
                # refresh_grasp_veto pass, feed those into the phase_d state
                # machine so WRIST mode triggers on vetoed grasp only.  Else
                # the function falls back to its internal detector (bootstrap).
                prior = pose_track_info.get(oid) if isinstance(pose_track_info, dict) else None
                if args.rebuild_grasp:
                    ext_wl = ext_wr = ext_im = None    # force fingertip-persistent bootstrap
                else:
                    ext_wl = prior.get("wrist_l_per_frame") if isinstance(prior, dict) else None
                    ext_wr = prior.get("wrist_r_per_frame") if isinstance(prior, dict) else None
                    ext_im = prior.get("is_moving_per_frame") if isinstance(prior, dict) else None
                res = _track_phase_d(
                    mesh_pts, mask_seq, depth_seq, hand_mask_seq, K,
                    joints_pf=joints_pf,
                    hand_is_right_pf=hand_is_right_pf,
                    sam3d_canonical_quat=q_can,
                    sam3d_canonical_scale=s_can,
                    enable_z_align=False,
                    grasp_point_per_frame=None,
                    existing_wrist_l_per_frame=ext_wl,
                    existing_wrist_r_per_frame=ext_wr,
                    existing_is_moving_per_frame=ext_im,
                )
            else:
                lambdas = dict(
                    fit=args.lambda_fit, anchor=args.lambda_anchor,
                    temporal=args.lambda_temporal, static=args.lambda_static,
                    pen=args.lambda_pen, prox=args.lambda_prox,
                    noslip=args.lambda_noslip,
                )
                # full_opt needs RGB frames — decode lazily
                frames_rgb_full = []
                for fd in fdata:
                    arr = np.frombuffer(fd.get("img_rgb") or b"", dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
                    frames_rgb_full.append(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)
                res = _track_full_opt(
                    mesh_pts, mask_seq, depth_seq, hand_mask_seq, K,
                    frames_rgb_full, lambdas,
                )
        except Exception as e:
            per_oid_results[oid] = ("fail", f"{type(e).__name__}: {e}")
            continue

        if res is None:
            per_oid_results[oid] = ("fail", "tracker returned None")
            continue

        # State estimation: STATIC / GRASPED / MOVING per frame.
        # New 2D-mask-centroid based, hierarchical.  obs_clouds still used
        # for the grasp leg (fingertip ↔ 3D cloud); masks_2d (from sam3_obj_data
        # already loaded above as mask_seq) drives the static gate.
        #
        # ★ PRESERVE-VETO RULE: if the existing pose_track_info[oid] already
        # has a state from a prior refresh_grasp_veto pass, do NOT re-run
        # _estimate_state_per_frame — that would overwrite the veto.  Carry
        # the existing state fields forward instead.  The state machine inside
        # _track_phase_d also already read this veto via wrist_l/r_per_frame.
        #
        # The marker for "veto-bearing existing state" is the presence of
        # ``wrist_l_per_frame`` AND ``state_per_frame`` in the prior entry.
        # First-time runs (no prior entry) fall through to fresh estimation.
        prior_pti_entry = pose_track_info.get(oid) if isinstance(pose_track_info, dict) else None
        # --rebuild-grasp (Pass 1) skips the preserve-veto rule entirely so the
        # detector re-fires from scratch. Pass 2 (no flag) preserves the freshly-
        # vetoed wrist arrays even when they are all-False (that's the veto
        # saying "no real grasps in this clip", which IS authoritative).
        has_prior_state = (
            isinstance(prior_pti_entry, dict)
            and "wrist_l_per_frame" in prior_pti_entry
            and "state_per_frame" in prior_pti_entry
            and not args.rebuild_grasp
        )

        obs_clouds = res.pop("_obs_clouds", None)
        if has_prior_state:
            # Carry vetoed state forward verbatim — do not re-estimate.
            for k in ("state_per_frame", "grasp_hand_per_frame",
                      "is_static_global", "centroid_2d_per_frame",
                      "centroid_2d_disp_per_frame", "global_span_px",
                      "state_counts",
                      "wrist_used_per_frame", "wrist_l_per_frame",
                      "wrist_r_per_frame", "is_moving_per_frame",
                      "close_per_frame"):
                if k in prior_pti_entry:
                    res[k] = prior_pti_entry[k]
            sc = res.get("state_counts") or {}
            gtot = (sc.get('grasped_l', 0) + sc.get('grasped_r', 0)
                    + sc.get('grasped_both', 0))
            state_summary = (
                f"S={sc.get('static', 0)} G={gtot}(L{sc.get('grasped_l', 0)}"
                f"/R{sc.get('grasped_r', 0)}/B{sc.get('grasped_both', 0)})"
                f" M={sc.get('moving', 0)} [preserved-veto]")
        elif obs_clouds is not None:
            try:
                # Per-frame hand vertices for 2D mask-overlap path.
                hand_verts_pf_for_state = [
                    fd.get("vertices_3d") or [] for fd in fdata
                ]
                state = _estimate_state_per_frame(
                    obs_clouds, joints_pf, hand_is_right_pf, mask_seq, (H, W),
                    fingertip_threshold_m=args.fingertip_threshold_m,
                    grasp_bridge_frames=args.grasp_bridge_frames,
                    grasp_min_frames=args.grasp_min_frames,
                    static_global_px_frac=args.static_global_px_frac,
                    static_frame_low_px=args.static_frame_low_px,
                    static_frame_high_px=args.static_frame_high_px,
                    hand_verts_pf=hand_verts_pf_for_state,
                    mano_faces=mano_faces,
                    K=K,
                )
                res.update(state)
                sc = state["state_counts"]
                gs = state["global_span_px"]
                gs_str = "global" if state["is_static_global"] else f"span={gs:.1f}px"
                gtot = sc['grasped_l'] + sc['grasped_r'] + sc['grasped_both']
                state_summary = (
                    f"S={sc['static']} G={gtot}(L{sc['grasped_l']}/R{sc['grasped_r']}/B{sc['grasped_both']})"
                    f" M={sc['moving']} [{gs_str}]")
            except Exception as _e:
                state_summary = f"state-est failed: {type(_e).__name__}"
        else:
            state_summary = "no obs"

        # Preserve scale_sanity override (if it exists) — refresh_scale_sanity
        # writes the corrected scale into pti.scale_correction and leaves the
        # raw SAM3D value in pti.scale_correction_orig_sam3d as provenance.
        # Without this guard, D-track silently reverts every clip back to the
        # uncorrected SAM3D scale on every rerun (e.g. -MF7nEIDKLk_242.2_247.3
        # measuring cup: 18cm → 64cm regression on every D-track pass).
        prev = (pose_track_info.get(oid) or pose_track_info.get(str(oid)) or {})
        if isinstance(prev, dict) and prev.get("scale_correction_orig_sam3d") is not None:
            res["scale_correction"] = prev["scale_correction"]
            res["scale_correction_orig_sam3d"] = prev["scale_correction_orig_sam3d"]
        pose_track_info[oid] = res
        anchor = res.get("anchor_t")
        nobs = res.get("n_observed", "—")
        per_oid_results[oid] = (
            "ok",
            f"anchor={anchor} n_obs={nobs} {state_summary} {res.get('note', '')}")

    # Persist
    n_ok = sum(1 for r in per_oid_results.values() if r[0] == "ok")
    if n_ok > 0:
        data["pose_track_info"] = pose_track_info
        data["pose_track_info_meta"] = {
            "mode": mode,
            "algo_version": ALGO_VERSION_POSITION if mode == "position_first" else ALGO_VERSION_FULLOPT,
            "smooth_sigma": float(args.smooth_sigma) if mode == "position_first" else None,
            "lambdas": (None if mode == "position_first" else dict(
                fit=args.lambda_fit, anchor=args.lambda_anchor,
                temporal=args.lambda_temporal, static=args.lambda_static,
                pen=args.lambda_pen, prox=args.lambda_prox,
                noslip=args.lambda_noslip)),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)
        # Multi-host sentinel: D-track just produced fresh pose_track_info.
        # `--scan` mode at next invocation will skip this clip.
        try:
            from egoinfinity.pipeline import pipeline_state as _ps
            _ps.mark_done(fav_dir, "D-track")
        except Exception as _e:
            print(f"  [state] mark_done(D-track) failed (non-fatal): {_e}")

    n_fail = sum(1 for r in per_oid_results.values() if r[0] == "fail")
    n_skip = sum(1 for r in per_oid_results.values() if r[0] == "skip")
    summary = f"{n_ok}/{len(per_oid_results)} ok"
    if n_fail: summary += f", {n_fail} fail"
    if n_skip: summary += f", {n_skip} skip"
    return ("ok" if n_ok else "fail" if n_fail else "skip",
            summary, per_oid_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["position_first", "full_opt", "phase_d"],
                    default="position_first")
    ap.add_argument("--only", default=None,
                    help="comma-separated favorite ids "
                         "(use --only=ID1,ID2 for leading-dash ids)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--scan", action="store_true",
                    help="Multi-host mode: only process clips whose "
                         "pipeline_state.json says D-sam3d is done AND "
                         "D-track is NOT yet done.  Pairs with "
                         "scripts/exo_pipeline.py + refresh_sam3d_meshes "
                         "mark_done() writes.")
    # position_first knobs
    ap.add_argument("--smooth-sigma", type=float, default=2.0,
                    help="position-first: temporal Gaussian smooth sigma (frames)")
    ap.add_argument("--no-icp", action="store_true",
                    help="position-first: skip anchor pose estimate (R = identity)")
    ap.add_argument("--enclose-pct", type=float, default=95.0,
                    help="position-first: target obs-coverage percentile for "
                         "min-scale enclosing computation (default 95)")
    # state estimation knobs (WRIST / MOVING / STATIC)
    ap.add_argument("--fingertip-threshold-m", type=float, default=0.06,
                    help="WRIST trigger: min fingertip↔obs distance, default 6cm")
    ap.add_argument("--grasp-bridge-frames", type=int, default=30,
                    help="WRIST: bridge ≤N gap frames in close[t] (default 30 ≈ 2s @ "
                         "15fps — long pour / step-back gaps inside a real grasp run)")
    ap.add_argument("--grasp-min-frames", type=int, default=8,
                    help="WRIST: drop close[t] runs shorter than this (default 8 ≈ 0.53s @ 15fps)")
    ap.add_argument("--rebuild-grasp", action="store_true",
                    help="Pass 1 mode: ignore prior wrist_l/r_per_frame from pkl "
                         "and re-detect grasps via fingertip_persistent. Use BEFORE "
                         "running refresh_grasp_veto, so the veto sees fresh "
                         "detector output. Pass 2 (after veto) should NOT use this.")
    # STATIC gate: 2D mask centroid motion in image-plane pixels (depth-noise-free).
    # Global = p10-p90 span over the clip ≤ static_global_px_frac × min(H, W).
    # Per-frame = Schmitt-trigger on |centroid[t] - centroid[t-1]|.
    ap.add_argument("--static-global-px-frac", type=float, default=0.02,
                    help="STATIC global gate: p10-p90 mask centroid span ≤ "
                         "frac × min(H, W) (default 0.02 = 2%%)")
    ap.add_argument("--static-frame-low-px", type=float, default=2.0,
                    help="STATIC per-frame Schmitt OFF, px/frame (default 2)")
    ap.add_argument("--static-frame-high-px", type=float, default=4.0,
                    help="STATIC per-frame Schmitt ON, px/frame (default 4)")
    # full_opt knobs (override DEFAULT_LAMBDAS)
    ap.add_argument("--lambda-fit", type=float, default=80.0)
    ap.add_argument("--lambda-anchor", type=float, default=50.0)
    ap.add_argument("--lambda-temporal", type=float, default=30.0)
    ap.add_argument("--lambda-static", type=float, default=200.0)
    ap.add_argument("--lambda-pen", type=float, default=5.0)
    ap.add_argument("--lambda-prox", type=float, default=10.0,
                    help="grasp pull (default 10 = 10× lower than pipeline default 100)")
    ap.add_argument("--lambda-noslip", type=float, default=5.0,
                    help="grasp slip prevention (default 5 = 10× lower than 50)")
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
                if _ps.is_phase_done(d, "D-sam3d")
                and not _ps.is_phase_done(d, "D-track")]
        print(f"[--scan] {before} eligible → {len(favs)} pending D-track "
              f"(skipped: already done, or D-sam3d upstream missing)")
    if not favs:
        print("Nothing to process")
        return

    print(f"mode={args.mode}  smooth_sigma={args.smooth_sigma}  "
          f"clips={len(favs)}", flush=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0_all = time.time()
    for i, fav in enumerate(favs, 1):
        t0 = time.time()
        try:
            st, summary, per_oid = _process_clip(
                fav, args.mode, args, args.skip_existing)
        except Exception as e:
            st, summary, per_oid = "fail", f"crash: {type(e).__name__}: {e}", {}
        counts[st] = counts.get(st, 0) + 1
        marker = {"ok": "+", "skip": "·", "fail": "✗"}.get(st, "?")
        dt = time.time() - t0
        print(f"[{i:3d}/{len(favs)}] {marker} {fav.name:48s} "
              f"{summary} ({dt:.1f}s)", flush=True)
        for oid, (oid_st, oid_msg) in sorted(per_oid.items()):
            if oid_st == "ok":
                print(f"           + obj {oid}: {oid_msg}", flush=True)
            elif oid_st == "fail":
                print(f"           ✗ obj {oid}: {oid_msg}", flush=True)

    dt_total = time.time() - t0_all
    print(f"\nDone in {dt_total/60:.1f} min — "
          f"ok: {counts.get('ok', 0)}  "
          f"skip: {counts.get('skip', 0)}  "
          f"fail: {counts.get('fail', 0)}")


if __name__ == "__main__":
    main()
