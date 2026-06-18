"""Grasp detection — filter contact_soft to only frames where the
object is actually moving with the hand.

Layered on top of `detect_contact_2d_aware`:
    contact = "geometric proximity / mask overlap"
    grasp   = contact AND object motion correlated with hand motion

Resolves the false-positive case where a hand brushes / passes over a
static object (high contact_soft but object doesn't follow).

Inputs
------
- contact_soft : (T, 2) float — primary contact signal
- obj_mask_completed_per_frame : per-frame completed mask (so centroid stable)
- hand_mask_per_frame : combined hand mask, used to subtract for centroid
- joints_per_frame : list[list[(21,3)]] — for wrist 3D → 2D projection
- hand_is_right_per_frame : list[list[bool]] — handedness aligned

Output
------
- grasp_soft : (T, 2) float — refined contact, low when motion
  uncorrelated.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

log = logging.getLogger("pose_tracker.grasp")

DEFAULT_MOTION_WINDOW = 5             # ±N frames around current
DEFAULT_MIN_HAND_DISP_PX = 30         # below this hand is "static"
DEFAULT_MAX_OBJ_HAND_RATIO = 0.3      # obj/hand displacement; below = not following
DEFAULT_MIN_COS_SIMILARITY = 0.5      # direction agreement
DEFAULT_RAMP_WIN = 7                  # SavGol smoothing
DEFAULT_RAMP_POLY = 2


def _mask_centroid_uv(mask: np.ndarray) -> Optional[np.ndarray]:
    """Pixel centroid of a bool mask. Returns (u, v) float or None."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _project_3d_to_2d(point_3d: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
    """Pinhole projection. Returns (u, v) or None for behind-camera."""
    if point_3d is None or len(point_3d) < 3:
        return None
    z = float(point_3d[2])
    if z <= 1e-3:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([
        point_3d[0] * fx / z + cx,
        point_3d[1] * fy / z + cy,
    ], dtype=np.float64)


def detect_grasp_with_motion(
    contact_soft: np.ndarray,
    obj_mask_completed_per_frame: Sequence[Optional[np.ndarray]],
    hand_mask_per_frame: Sequence[Optional[np.ndarray]],
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    K: np.ndarray,
    *,
    window: int = DEFAULT_MOTION_WINDOW,
    min_hand_disp_px: float = DEFAULT_MIN_HAND_DISP_PX,
    max_obj_hand_ratio: float = DEFAULT_MAX_OBJ_HAND_RATIO,
    min_cos_similarity: float = DEFAULT_MIN_COS_SIMILARITY,
    ramp_window: int = DEFAULT_RAMP_WIN,
    ramp_poly: int = DEFAULT_RAMP_POLY,
    enable_segment_demote: bool = False,
) -> np.ndarray:
    """Compute per-frame, per-hand grasp soft weight.

    Algorithm per frame t, per hand h:
        if contact_soft[t, h] < 0.3 → grasp[t, h] = 0
        compute object 2D centroid (using mask ∖ hand_mask) at t±window
        compute hand wrist 2D position at t±window
        if hand_disp < min_hand_disp_px → trust contact (static grasp)
        elif obj/hand ratio < threshold → 0 (hand moves, obj doesn't)
        elif cos_sim > threshold → trust contact (correlated motion)
        else → demote to 0.3× contact (suspicious)
    SavGol smooth across time.
    """
    T = len(obj_mask_completed_per_frame)
    contact_soft = np.asarray(contact_soft, dtype=np.float32)
    if contact_soft.shape[0] != T:
        raise ValueError(
            f"contact_soft length {contact_soft.shape[0]} != mask length {T}")

    # Pre-compute per-frame centroids using (obj_mask & ~hand_mask)
    obj_uv = np.full((T, 2), np.nan, dtype=np.float64)
    for t in range(T):
        m = obj_mask_completed_per_frame[t]
        if m is None or not m.any():
            continue
        h = hand_mask_per_frame[t] if t < len(hand_mask_per_frame) else None
        eff = m & ~h if h is not None else m
        if not eff.any():
            eff = m
        c = _mask_centroid_uv(eff)
        if c is not None:
            obj_uv[t] = c

    # Pre-compute per-frame, per-hand wrist 2D
    wrist_uv = np.full((T, 2, 2), np.nan, dtype=np.float64)  # (T, hand_idx, uv)
    for t in range(T):
        joints_list = joints_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        for j, ir in zip(joints_list, is_right_list):
            if j is None or len(j) == 0:
                continue
            uv = _project_3d_to_2d(np.asarray(j[0]), K)
            if uv is not None:
                wrist_uv[t, 1 if ir else 0] = uv

    raw_grasp = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        if (contact_soft[t] < 0.3).all():
            continue
        t0 = max(0, t - window)
        t1 = min(T - 1, t + window)
        # Object displacement (over the window)
        obj_d = obj_uv[t1] - obj_uv[t0]
        if not np.isfinite(obj_d).all():
            obj_d = np.zeros(2, dtype=np.float64)
        obj_mag = float(np.linalg.norm(obj_d))

        for hand_idx in (0, 1):
            c_w = float(contact_soft[t, hand_idx])
            if c_w < 0.3:
                continue
            wrist_d = wrist_uv[t1, hand_idx] - wrist_uv[t0, hand_idx]
            if not np.isfinite(wrist_d).all():
                # No wrist data for this window → trust contact
                raw_grasp[t, hand_idx] = c_w
                continue
            wrist_mag = float(np.linalg.norm(wrist_d))

            if wrist_mag < min_hand_disp_px:
                # Hand is essentially still → static grasp (or static-near-obj);
                # trust contact signal
                raw_grasp[t, hand_idx] = c_w
                continue

            ratio = obj_mag / max(wrist_mag, 1e-6)
            if ratio < max_obj_hand_ratio:
                # Hand moves, object doesn't → not grasping (passing/brushing)
                raw_grasp[t, hand_idx] = 0.0
                continue

            cos_sim = float(np.dot(obj_d, wrist_d) /
                             (max(wrist_mag, 1e-6) * max(obj_mag, 1e-6)))
            if cos_sim > min_cos_similarity:
                # Correlated motion → confirm grasp
                raw_grasp[t, hand_idx] = c_w
            else:
                # Some motion but uncorrelated → weak grasp
                raw_grasp[t, hand_idx] = c_w * 0.3

    # ── Segment-level filter (DISABLED by default): contact segments with NO
    # motion at all could be "resting touch" (hand on a static plate), but
    # they could equally be "held still" (bottle held in fist while talking).
    # Per-frame logic above already handles this correctly via the
    # `wrist_mag < min_hand_disp_px → trust contact` branch.  Enabling this
    # segment-level demote causes false negatives on held-still objects;
    # leave OFF unless tuning a clip with rampant resting-touch artefacts.
    if enable_segment_demote:
        from .hand_driven import find_continuous_segments
        contact_any = (contact_soft.max(axis=1) > 0.3)
        contact_segs = find_continuous_segments(contact_any, min_len=3)
        SEG_MIN_OBJ_DISP_PX = 25.0
        SEG_MIN_WRIST_DISP_PX = 40.0
        SEG_DEMOTE_FACTOR = 0.15
        for (s, e) in contact_segs:
            obj_endpts = obj_uv[[s, e]]
            if not np.isfinite(obj_endpts).all():
                continue
            obj_d_seg = float(np.linalg.norm(obj_endpts[1] - obj_endpts[0]))
            max_wrist_d = 0.0
            for hand_idx in (0, 1):
                wrist_endpts = wrist_uv[[s, e], hand_idx]
                if np.isfinite(wrist_endpts).all():
                    d = float(np.linalg.norm(wrist_endpts[1] - wrist_endpts[0]))
                    if d > max_wrist_d:
                        max_wrist_d = d
            if (obj_d_seg < SEG_MIN_OBJ_DISP_PX
                    and max_wrist_d < SEG_MIN_WRIST_DISP_PX):
                raw_grasp[s:e+1] *= SEG_DEMOTE_FACTOR
                log.info(
                    f"grasp: demoted segment [{s}-{e}] "
                    f"(obj_disp={obj_d_seg:.1f}px, wrist_disp={max_wrist_d:.1f}px)")

    # SavGol smooth the grasp signal across time
    if T >= ramp_window:
        grasp_soft = savgol_filter(raw_grasp, ramp_window, ramp_poly, axis=0)
        grasp_soft = np.clip(grasp_soft, 0.0, 1.0).astype(np.float32)
    else:
        grasp_soft = raw_grasp

    n_grasp = int((grasp_soft.max(axis=1) > 0.5).sum())
    n_contact = int((contact_soft.max(axis=1) > 0.5).sum())
    log.info(
        f"grasp: {n_grasp}/{T} frames qualify as grasp "
        f"(filtered from {n_contact} contact frames)")
    return grasp_soft


# ---------------------------------------------------------------------------
# 2026-05-01 — Proximity + Object-Motion grasp detector
# ---------------------------------------------------------------------------
def detect_grasp_proximity_motion(
    obs_clouds: Sequence[Optional[np.ndarray]],
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    obj_motion_pf: np.ndarray,
    pc_motion_pf: np.ndarray,
    *,
    proximity_threshold_m: float = 0.10,        # 10 cm wrist↔obs centroid
    flow_motion_threshold_px: float = 1.0,      # px/frame
    pc_motion_threshold_m: float = 0.010,       # m/frame
    proximity_smooth_window: int = 7,
    motion_smooth_window: int = 7,
    ramp_window: int = DEFAULT_RAMP_WIN,
    ramp_poly: int = DEFAULT_RAMP_POLY,
) -> np.ndarray:
    """Per-frame, per-hand soft grasp weight from **proximity + motion**.

    Decoupled from 2D mask overlap and motion correlation.  Two conditions
    must both hold for a grasp:

        1. Hand–object **persistent proximity**: wrist 3D position close to
           the obs cloud centroid (uses obs as a robust object-position
           estimate that doesn't depend on mesh pose).
        2. Object **actually moving**: flow magnitude in object mask OR
           obs cloud bbox-center motion above threshold.

    Both signals are SavGol-smoothed first (so single-frame noise can't
    flip the verdict), then multiplied: ``grasp = proximity * motion``.
    Final SavGol pass smooths the boundaries.

    Returns (T, 2) float ∈ [0, 1].  Column 0 = left, 1 = right.
    """
    T = len(obs_clouds)
    obj_motion_pf = np.asarray(obj_motion_pf, dtype=np.float32)
    pc_motion_pf = np.asarray(pc_motion_pf, dtype=np.float32)

    # 1. Obs centroid per frame (NaN where unavailable)
    obs_centroid = np.full((T, 3), np.nan, dtype=np.float64)
    for t in range(T):
        obs = obs_clouds[t] if t < len(obs_clouds) else None
        if obs is not None and len(obs) >= 30:
            obs_centroid[t] = np.asarray(obs).mean(axis=0)

    # 2. Wrist 3D per frame per hand
    wrist_pos = np.full((T, 2, 3), np.nan, dtype=np.float64)
    for t in range(T):
        joints_list = joints_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        for j, ir in zip(joints_list, is_right_list):
            if j is None or len(j) == 0:
                continue
            col = 1 if ir else 0
            wrist_pos[t, col] = np.asarray(j[0])

    # 3. Distance per frame per hand
    dist = np.full((T, 2), np.inf, dtype=np.float64)
    for t in range(T):
        if not np.isfinite(obs_centroid[t]).all():
            continue
        for h in (0, 1):
            if np.isfinite(wrist_pos[t, h]).all():
                dist[t, h] = float(np.linalg.norm(wrist_pos[t, h] - obs_centroid[t]))

    # 4. Proximity (binary then SavGol-smoothed for persistence)
    close_raw = (dist < proximity_threshold_m).astype(np.float32)
    if T >= proximity_smooth_window and proximity_smooth_window >= 3:
        w = proximity_smooth_window if (proximity_smooth_window % 2 == 1) else (proximity_smooth_window + 1)
        close_smooth = savgol_filter(close_raw, w, 2, axis=0)
        close_smooth = np.clip(close_smooth, 0.0, 1.0)
    else:
        close_smooth = close_raw

    # 5. Object actually moving (flow OR pc bbox motion)
    flow_raw = (obj_motion_pf > flow_motion_threshold_px).astype(np.float32)
    pc_raw = (pc_motion_pf > pc_motion_threshold_m).astype(np.float32)
    obj_moving_raw = np.maximum(flow_raw, pc_raw)
    if T >= motion_smooth_window and motion_smooth_window >= 3:
        w = motion_smooth_window if (motion_smooth_window % 2 == 1) else (motion_smooth_window + 1)
        obj_moving_smooth = savgol_filter(obj_moving_raw, w, 2)
        obj_moving_smooth = np.clip(obj_moving_smooth, 0.0, 1.0)
    else:
        obj_moving_smooth = obj_moving_raw

    # 6. Grasp = proximity AND obj-moving (per hand × per frame)
    grasp_raw = close_smooth * obj_moving_smooth[:, None]   # (T, 2)

    # 7. Final smooth
    if T >= ramp_window and ramp_window >= 3:
        w = ramp_window if (ramp_window % 2 == 1) else (ramp_window + 1)
        grasp_soft = savgol_filter(grasp_raw, w, ramp_poly, axis=0)
        grasp_soft = np.clip(grasp_soft, 0.0, 1.0).astype(np.float32)
    else:
        grasp_soft = grasp_raw.astype(np.float32)

    n_grasp = int((grasp_soft.max(axis=1) > 0.5).sum())
    n_close = int((close_smooth.max(axis=1) > 0.5).sum())
    n_moving = int((obj_moving_smooth > 0.5).sum())
    log.info(
        f"grasp (proximity+motion): {n_grasp}/{T} frames "
        f"(persistent close={n_close}/{T}, obj moving={n_moving}/{T}, "
        f"prox_thr={proximity_threshold_m*100:.0f}cm)")
    return grasp_soft


# ── Fingertip-only persistent grasp (replaces proximity+motion) ──────

# MANO joint indices for the 5 fingertips (thumb, index, middle, ring, pinky).
_MANO_FINGERTIP_IDX = (4, 8, 12, 16, 20)


def _bridge_short_gaps(b: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill internal 0-runs of length ≤ ``max_gap`` with 1s.

    Boundary gaps (touching index 0 or T-1) are left alone — they're not
    "between two grasps" and shouldn't be filled.
    """
    out = b.copy()
    T = len(out)
    i = 0
    while i < T:
        if out[i]:
            i += 1
            continue
        j = i
        while j < T and not out[j]:
            j += 1
        if 0 < i and j < T and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def _drop_short_runs(b: np.ndarray, min_len: int) -> np.ndarray:
    """Remove 1-runs shorter than ``min_len`` frames."""
    out = b.copy()
    T = len(out)
    i = 0
    while i < T:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < T and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def detect_grasp_fingertip_persistent(
    obs_clouds: Sequence[Optional[np.ndarray]],
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    *,
    fingertip_threshold_m: float = 0.04,    # 4 cm
    bridge_gap_frames: int = 5,             # ≤ 5 = 0.33s @ 15fps
    min_grasp_frames: int = 8,              # ≥ 8 = 0.53s @ 15fps
) -> np.ndarray:
    """Per-frame, per-hand grasp from sustained fingertip-to-object proximity.

    Pure offline geometry, no motion / curl gating:

        1. dist[t, h] = min over MANO fingertips (joints 4, 8, 12, 16, 20) of
           min over obs-cloud points of Euclidean distance.
        2. close[t, h] = (dist ≤ fingertip_threshold_m).
        3. Bridge internal 0-runs of length ≤ bridge_gap_frames  (close gaps).
        4. Drop 1-runs of length < min_grasp_frames  (open noise).

    Step 3 handles brief occlusions / finger-extension blinks within a real
    grasp; step 4 rejects "hand passes by" momentary contacts that don't
    persist.  Boundary gaps are not bridged (a clip-edge run of 0s isn't
    "between two grasps"), but boundary runs are subject to ``min_grasp_frames``
    just like internal runs.

    Parameters
    ----------
    obs_clouds : (T,) sequence of (N_t, 3) arrays or None
        Per-frame observation point cloud for the object.
    joints_per_frame : (T,) sequence of per-hand (21, 3) arrays
        Per-frame MANO joint sets (wrist=0, thumb tip=4, …, pinky tip=20).
    hand_is_right_per_frame : (T,) sequence aligned with ``joints_per_frame``
        Handedness flag for each entry (True = right hand).

    Returns
    -------
    (T, 2) float32 array, values ∈ {0.0, 1.0}.  Column 0 = left, 1 = right.
    """
    T = len(obs_clouds)
    if T == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 1. Per-frame, per-hand minimum fingertip → obs cloud distance.
    dist = np.full((T, 2), np.inf, dtype=np.float64)
    for t in range(T):
        obs = obs_clouds[t] if t < len(obs_clouds) else None
        if obs is None:
            continue
        obs_arr = np.asarray(obs, dtype=np.float64)
        if obs_arr.ndim != 2 or obs_arr.shape[0] < 30 or obs_arr.shape[1] != 3:
            continue
        joints_list = joints_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        for j, ir in zip(joints_list, is_right_list):
            if j is None:
                continue
            j_arr = np.asarray(j, dtype=np.float64)
            if j_arr.shape != (21, 3):
                continue
            tips = j_arr[list(_MANO_FINGERTIP_IDX)]              # (5, 3)
            d = np.sqrt(((tips[:, None, :] - obs_arr[None, :, :]) ** 2).sum(-1))
            col = 1 if ir else 0
            dist[t, col] = float(d.min())

    # 2. Binary close mask, then 3-4 morph cleanup per hand.
    close = dist <= fingertip_threshold_m                          # (T, 2) bool
    grasp_bool = np.zeros_like(close)
    for h in (0, 1):
        bridged = _bridge_short_gaps(close[:, h], bridge_gap_frames)
        grasp_bool[:, h] = _drop_short_runs(bridged, min_grasp_frames)

    grasp = grasp_bool.astype(np.float32)

    n_close_R = int(close[:, 1].sum())
    n_close_L = int(close[:, 0].sum())
    n_grasp_R = int(grasp_bool[:, 1].sum())
    n_grasp_L = int(grasp_bool[:, 0].sum())
    log.info(
        f"grasp (fingertip persistent / morph): R={n_grasp_R}/{T} L={n_grasp_L}/{T} "
        f"(raw close R={n_close_R}/{T} L={n_close_L}/{T}; "
        f"thr={fingertip_threshold_m*100:.0f}cm, "
        f"bridge≤{bridge_gap_frames}f, min_run≥{min_grasp_frames}f)"
    )
    return grasp
