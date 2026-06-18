"""
Depth-guided hand alignment: fuse WiLoR weak-perspective hands with monocular metric depth.

Two alignment methods:
  1. rescale_cam_t:   Simply rescale WiLoR's tz with a new focal length
  2. align_hand_depth: Robust multi-joint alignment using depth map

The key insight: WiLoR outputs root-relative 3D joints (good local structure from MANO)
plus a weak-perspective cam_t. We keep MANO's relative structure and replace the
absolute translation using monocular metric depth at multiple joint locations.

Borrows ideas from HaWoR's est_scale_hybrid (iterative median + robust loss)
but adapted for single-frame, static-camera, no-SLAM setting.
"""
import numpy as np
from typing import Optional


# Joints used for depth sampling — choose "thick" body parts where
# monocular depth is more reliable (wrist, MCP bases).
# Avoid fingertips (thin, noisy depth).
RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]  # wrist + 4 MCP joints
# Fallback to all joints if too few reliable ones have valid depth
ALL_JOINT_IDS = list(range(21))


def rescale_cam_t(cam_t: np.ndarray,
                  old_focal: float,
                  new_focal: float) -> np.ndarray:
    """Rescale cam_t for a different focal length.

    Only tz changes proportionally: tz = 2*f/(bbox*s).
    tx, ty are focal-independent in cam_crop_to_full.

    Args:
        cam_t: (3,) original [tx, ty, tz]
        old_focal: focal length used to compute cam_t
        new_focal: target focal length

    Returns:
        (3,) rescaled cam_t
    """
    cam_t_new = cam_t.copy()
    cam_t_new[2] = cam_t[2] * new_focal / old_focal
    return cam_t_new


def sample_depth_at_joints(depth_map: np.ndarray,
                           joints_2d: np.ndarray,
                           joint_ids: list,
                           patch_size: int = 5) -> np.ndarray:
    """Sample depth map at given 2D joint locations.

    Args:
        depth_map: (H, W) metric depth
        joints_2d: (21, 2) pixel coordinates
        joint_ids: which joints to sample
        patch_size: median filter patch size

    Returns:
        (len(joint_ids),) depth values, NaN for invalid
    """
    H, W = depth_map.shape
    half = patch_size // 2
    depths = np.full(len(joint_ids), np.nan)

    for i, jid in enumerate(joint_ids):
        x, y = joints_2d[jid]
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue
        x0, x1 = max(0, xi - half), min(W, xi + half + 1)
        y0, y1 = max(0, yi - half), min(H, yi + half + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0.01]
        if len(valid) > 0:
            depths[i] = float(np.median(valid))

    return depths


def backproject_2d_to_3d(joints_2d: np.ndarray,
                         depths: np.ndarray,
                         focal: float,
                         cx: float, cy: float,
                         joint_ids: list) -> np.ndarray:
    """Back-project 2D joints + depth to 3D using pinhole model.

    Args:
        joints_2d: (21, 2) pixel coords
        depths: (len(joint_ids),) metric depth at each joint
        focal: focal length in pixels
        cx, cy: principal point
        joint_ids: which joints

    Returns:
        (len(joint_ids), 3) 3D points in camera frame
    """
    points_3d = np.full((len(joint_ids), 3), np.nan)
    for i, jid in enumerate(joint_ids):
        d = depths[i]
        if np.isnan(d) or d <= 0:
            continue
        u, v = joints_2d[jid]
        points_3d[i, 0] = (u - cx) * d / focal
        points_3d[i, 1] = (v - cy) * d / focal
        points_3d[i, 2] = d
    return points_3d


def align_hand_to_depth(joints_3d_rel: np.ndarray,
                        joints_2d: np.ndarray,
                        depth_map: np.ndarray,
                        focal: float,
                        cx: float, cy: float,
                        patch_size: int = 7) -> Optional[np.ndarray]:
    """Compute metric cam_t by aligning WiLoR joints to metric depth.

    Method:
      1. Sample DP depth at reliable joints (wrist + MCPs)
      2. Back-project 2D+depth to 3D via pinhole: P_i = ((u-cx)*d/f, (v-cy)*d/f, d)
      3. WiLoR relative joint: j_i = joints_3d_rel[jid]
      4. Translation estimate per joint: t_i = P_i - j_i
      5. Robust median of t_i across valid joints → cam_t_aligned

    This is analogous to HaWoR's est_scale_hybrid but:
      - No SLAM (static camera, no scale ambiguity)
      - Uses pinhole back-projection instead of scale ratio
      - Multi-joint median instead of full-image pixel-wise alignment

    Args:
        joints_3d_rel: (21, 3) MANO root-relative joints
        joints_2d: (21, 2) 2D projections in pixels
        depth_map: (H, W) metric depth
        focal: estimated focal length in pixels
        cx, cy: principal point

    Returns:
        (3,) aligned cam_t, or None if alignment failed
    """
    # Try reliable joints first
    joint_ids = RELIABLE_JOINT_IDS
    depths = sample_depth_at_joints(depth_map, joints_2d, joint_ids, patch_size)
    valid_mask = ~np.isnan(depths) & (depths > 0.01)

    # Fallback to all joints if too few reliable ones
    if valid_mask.sum() < 2:
        joint_ids = ALL_JOINT_IDS
        depths = sample_depth_at_joints(depth_map, joints_2d, joint_ids, patch_size)
        valid_mask = ~np.isnan(depths) & (depths > 0.01)

    if valid_mask.sum() < 1:
        return None

    # Back-project valid joints to 3D
    pts_3d = backproject_2d_to_3d(joints_2d, depths, focal, cx, cy, joint_ids)

    # Compute per-joint translation: t_i = P_i - j_rel_i
    translations = np.full((len(joint_ids), 3), np.nan)
    for i, jid in enumerate(joint_ids):
        if valid_mask[i]:
            translations[i] = pts_3d[i] - joints_3d_rel[jid]

    # Robust estimate: median across valid joints
    valid_trans = translations[valid_mask]
    cam_t_aligned = np.median(valid_trans, axis=0)

    return cam_t_aligned


def align_hand_to_depth_multiscale(joints_3d_rel: np.ndarray,
                                   joints_2d: np.ndarray,
                                   depth_map: np.ndarray,
                                   focal: float,
                                   cx: float, cy: float,
                                   cam_t_wilor: np.ndarray,
                                   wilor_focal: float) -> np.ndarray:
    """Align hand with depth, falling back to WiLoR cam_t if alignment fails.

    Tries depth-based alignment first. If it fails (no valid depth at joints),
    falls back to rescaling WiLoR cam_t with the DP focal.

    Args:
        joints_3d_rel: (21, 3) MANO root-relative
        joints_2d: (21, 2) 2D pixel coords
        depth_map: (H, W) metric depth
        focal: DP estimated focal
        cx, cy: principal point
        cam_t_wilor: (3,) WiLoR's cam_t at wilor_focal
        wilor_focal: the scaled_focal WiLoR used

    Returns:
        (3,) best cam_t estimate
    """
    # Try robust multi-joint alignment
    cam_t_dp = align_hand_to_depth(joints_3d_rel, joints_2d, depth_map,
                                   focal, cx, cy)
    if cam_t_dp is not None:
        return cam_t_dp

    # Fallback: just rescale WiLoR cam_t with DP focal
    return rescale_cam_t(cam_t_wilor, wilor_focal, focal)
