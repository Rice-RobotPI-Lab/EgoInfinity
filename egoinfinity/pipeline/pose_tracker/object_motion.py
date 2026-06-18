"""Per-object motion detection from optical flow magnitudes.

Used to distinguish DEPTH_STATIC (lock pose) vs DEPTH_TRACKED (per-frame
follow) within the depth-tracking path.  Static objects shouldn't get
per-frame T_depth updates because depth has mm-scale noise that would
manifest as visible jitter; instead we lock to a single median pose.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .hand_driven import find_continuous_segments

log = logging.getLogger("pose_tracker.object_motion")


DEFAULT_LOCK_THRESHOLD_PX = 1.0   # below: lock pose to segment median (DEPTH_STATIC)
DEFAULT_FAST_THRESHOLD_PX = 3.0   # above: clearly moving (GUI / diagnostic only)
DEFAULT_MOTION_THRESHOLD_PX = DEFAULT_LOCK_THRESHOLD_PX  # back-compat alias
DEFAULT_PC_LOCK_THRESHOLD_M = 0.015   # m/frame — pc bbox-center motion to unlock
                                       # (3x MoGe depth noise floor; below 1.5cm
                                       # → noise rather than real motion)
DEFAULT_RAMP_WIN = 7
DEFAULT_MIN_SEG_LEN = 5


def compute_pc_motion_per_frame(
    obs_clouds: Sequence[Optional[np.ndarray]],
) -> np.ndarray:
    """Per-frame point-cloud bbox-center displacement (m/frame).

    A more reliable motion signal than optical flow when:
      - the object is grasped (hand contaminates flow inside the mask)
      - large motion blur degrades flow accuracy
      - the object surface is uniform (low gradient → low flow magnitude)

    bbox center is density-invariant (vs obs.mean which biases forward) so
    it tracks the geometric centre, not the centre of visible-surface mass.
    """
    T = len(obs_clouds)
    centers: List[Optional[np.ndarray]] = []
    for obs in obs_clouds:
        if obs is None or len(obs) < 30:
            centers.append(None)
        else:
            centers.append((obs.min(axis=0) + obs.max(axis=0)) * 0.5)
    motion = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        if centers[t] is None or centers[t-1] is None:
            continue
        motion[t] = float(np.linalg.norm(centers[t] - centers[t-1]))
    return motion


def compute_object_motion_per_frame(
    pair_mag_list: Sequence[np.ndarray],
    obj_masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
) -> np.ndarray:
    """Mean optical flow magnitude inside ``obj_mask & ~hand_mask`` per frame.

    pair_mag_list[t] is the (H, W) flow magnitude going from frame t to t+1.
    The last frame reuses the previous magnitude.

    Returns
    -------
    motion : (T,) float32 — px/frame
    """
    T = len(obj_masks)
    motion = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = obj_masks[t]
        if m is None or not m.any():
            continue
        h = hand_masks[t] if t < len(hand_masks) else None
        eff = (m & ~h) if h is not None else m
        if not eff.any():
            eff = m
        idx = min(t, len(pair_mag_list) - 1)
        if idx < 0:
            continue
        mag = pair_mag_list[idx]
        if mag is None:
            continue
        motion[t] = float(mag[eff].mean())
    return motion


def detect_static_moving_segments(
    motion_per_frame: np.ndarray,
    *,
    pc_motion_per_frame: Optional[np.ndarray] = None,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD_PX,
    fast_threshold: float = DEFAULT_FAST_THRESHOLD_PX,
    pc_lock_threshold: float = DEFAULT_PC_LOCK_THRESHOLD_M,
    threshold: Optional[float] = None,        # deprecated alias for lock_threshold
    ramp_window: int = DEFAULT_RAMP_WIN,
    min_seg_len: int = DEFAULT_MIN_SEG_LEN,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Three-tier classification of each frame based on motion magnitude.

    - motion ≤ ``lock_threshold`` (after SavGol smoothing)  → STATIC (lock)
    - ``lock_threshold`` < motion ≤ ``fast_threshold``      → MICRO (per-frame T_depth)
    - motion > ``fast_threshold``                            → FAST  (per-frame T_depth)

    The lock-vs-not decision uses a binary threshold smoothed across frames
    (so single-frame noise doesn't flip status).  The fast-vs-micro split is
    GUI / diagnostic only — both tiers use per-frame T_depth for pose.

    Returns (moving_segments, static_segments, is_moving, is_fast):
        moving_segments : List[(s, e)]   continuous spans where is_moving==True
        static_segments : List[(s, e)]   continuous spans where is_moving==False
        is_moving       : (T,) bool      True → use per-frame T_depth (no lock)
        is_fast         : (T,) bool      True → motion above fast_threshold (diagnostic)
    """
    if threshold is not None:
        lock_threshold = threshold
    T = len(motion_per_frame)
    raw_lock = (motion_per_frame > lock_threshold).astype(np.float32)
    raw_fast = (motion_per_frame > fast_threshold).astype(np.float32)
    if T >= ramp_window:
        smooth_lock = savgol_filter(raw_lock, ramp_window, 2)
        smooth_fast = savgol_filter(raw_fast, ramp_window, 2)
    else:
        smooth_lock = raw_lock
        smooth_fast = raw_fast
    flow_moving = smooth_lock > 0.5
    is_fast = smooth_fast > 0.5
    # Point-cloud motion: any frame where bbox-center moved > threshold m/frame
    # (smoothed) overrides flow's "static" classification — point cloud is the
    # primary truth signal, flow is corroborative.
    if pc_motion_per_frame is not None:
        raw_pc = (np.asarray(pc_motion_per_frame, dtype=np.float32) >
                   pc_lock_threshold).astype(np.float32)
        if T >= ramp_window:
            smooth_pc = savgol_filter(raw_pc, ramp_window, 2)
        else:
            smooth_pc = raw_pc
        pc_moving = smooth_pc > 0.5
        is_moving = flow_moving | pc_moving
    else:
        pc_moving = np.zeros(T, dtype=bool)
        is_moving = flow_moving
    moving_segs = find_continuous_segments(is_moving, min_len=min_seg_len)
    static_segs = find_continuous_segments(~is_moving, min_len=min_seg_len)
    n_micro = int((is_moving & ~is_fast).sum())
    n_fast = int(is_fast.sum())
    n_lock = int((~is_moving).sum())
    n_pc_only = int((pc_moving & ~flow_moving).sum())
    log.info(
        f"object_motion: {len(moving_segs)} moving segs / "
        f"{len(static_segs)} static segs "
        f"(flow median={np.median(motion_per_frame):.2f} px/frame, "
        f"lock_thr={lock_threshold:.2f} fast_thr={fast_threshold:.2f}, "
        f"per-frame split lock={n_lock} micro={n_micro} fast={n_fast}, "
        f"pc-only-unlocked={n_pc_only})")
    return moving_segs, static_segs, is_moving, is_fast


# ---------------------------------------------------------------------------
# Per-frame stability flags (mask size + pointcloud density / variance)
# ---------------------------------------------------------------------------
DEFAULT_MASK_SIZE_THRESH = 800           # pixels
DEFAULT_DENSITY_THRESH = 0.30            # n_obs / n_mask_pixels
DEFAULT_DEPTH_VAR_THRESH_M2 = 1.0e-3     # 1 cm² ≈ 3 cm depth std


def per_frame_stability_flags(
    obj_masks: Sequence[Optional[np.ndarray]],
    obs_clouds: Sequence[Optional[np.ndarray]],
    *,
    mask_size_thresh: int = DEFAULT_MASK_SIZE_THRESH,
    density_thresh: float = DEFAULT_DENSITY_THRESH,
    depth_var_thresh_m2: float = DEFAULT_DEPTH_VAR_THRESH_M2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame booleans: (mask_too_small, pointcloud_unstable).

    Used to gate the WRIST override:
        wrist_used[t] = grasp[t] AND (mask_too_small[t] OR pc_unstable[t])
    """
    T = len(obj_masks)
    mask_small = np.zeros(T, dtype=bool)
    pc_unstable = np.zeros(T, dtype=bool)
    for t in range(T):
        m = obj_masks[t]
        if m is None or not m.any():
            mask_small[t] = True
            pc_unstable[t] = True
            continue
        n_mask = int(m.sum())
        if n_mask < mask_size_thresh:
            mask_small[t] = True
        obs = obs_clouds[t] if t < len(obs_clouds) else None
        if obs is None or len(obs) < 30:
            pc_unstable[t] = True
            continue
        density = len(obs) / max(n_mask, 1)
        depth_var = float(np.var(obs[:, 2]))
        if (density < density_thresh) or (depth_var > depth_var_thresh_m2):
            pc_unstable[t] = True
    return mask_small, pc_unstable
