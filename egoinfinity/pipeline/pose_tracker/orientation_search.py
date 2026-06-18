"""Layer A: try-multi-orientation anchor refinement.

SAM3D's canonical orientation is essentially arbitrary for symmetric / elongated
objects (knife flipped 90°, turkey flipped 180°, etc).  ICP on partial / occluded
observations can't escape big rotation errors (fitness ~0).  So we explicitly
search a discrete set of candidate orientations at the anchor frame and pick the
one whose rendered mesh mask best matches the SAM2 mask.

Candidates: the 24 proper rotations of the cube symmetry group + identity.
This is the standard finite subgroup of SO(3) that catches all common 90° /
180° / 270° errors around the principal axes.

For each candidate:
    1. Rotate mesh canonical by R_cand (pre-mul) → equivalent T new
    2. Re-run translation refinement (centroid alignment)
    3. Render mesh mask + score mask IoU vs SAM2

Pick highest IoU.  Rough cost: 24 × (50 ms = render + IoU + KDTree) ≈ 1.2 s/object.
"""
from __future__ import annotations

import itertools
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .trust_filter import render_mesh_mask, mask_iou
from .anchor import _make_anchor_pointcloud

log = logging.getLogger("pose_tracker.orientation_search")


# ---------------------------------------------------------------------------
# Cube symmetry rotations (24 proper)
# ---------------------------------------------------------------------------
def cube_symmetry_rotations() -> List[np.ndarray]:
    """Return the 24 proper rotations of the cube as 3x3 matrices.

    These map the unit cube to itself with positive determinant.
    Implementation: enumerate signed-permutation matrices of det=+1.
    """
    rots = []
    seen_strs = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if abs(np.linalg.det(R) - 1.0) < 1e-9:
                key = R.tobytes()
                if key in seen_strs:
                    continue
                seen_strs.add(key)
                rots.append(R)
    assert len(rots) == 24, f"expected 24 cube rotations, got {len(rots)}"
    return rots


def axis_aligned_quarter_rotations() -> List[np.ndarray]:
    """Lighter set: 12 rotations covering 0/90/180/270 around each principal axis.

    Faster than full cube (24); catches the most common SAM3D error modes
    (axis-aligned flips).  Includes identity.
    """
    rots: List[np.ndarray] = []
    seen = set()
    angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for axis in range(3):
        for ang in angles:
            R = np.eye(3)
            c, s = np.cos(ang), np.sin(ang)
            if axis == 0:
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 1:
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R = R.round(8)
            key = R.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rots.append(R)
    return rots


# ---------------------------------------------------------------------------
# Translation-only centroid refinement (mirrors pipeline's anchor t-refine)
# ---------------------------------------------------------------------------
def _t_refine(T: np.ndarray, mesh_pts: np.ndarray,
              obs_cloud: np.ndarray) -> np.ndarray:
    """Shift T's translation so mesh-in-cam centroid matches obs centroid."""
    if obs_cloud is None or len(obs_cloud) < 50:
        return T
    R, t = T[:3, :3], T[:3, 3]
    mesh_in_cam = mesh_pts @ R.T + t
    delta = obs_cloud.mean(0) - mesh_in_cam.mean(0)
    out = T.copy()
    out[:3, 3] = t + delta
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def search_anchor_orientation(
    T_anchor_init: np.ndarray,
    mesh_pts: np.ndarray,
    mask_anchor: np.ndarray,                # (H, W) bool — SAM2
    hand_mask_anchor: Optional[np.ndarray],
    depth_anchor: np.ndarray,
    K: np.ndarray,
    *,
    rotation_set: str = "cube",             # "cube" (24) or "axis" (12)
    min_iou_to_replace: float = 0.05,       # safety: only replace if best > base + this
) -> Tuple[np.ndarray, float, int, dict]:
    """Search rotation candidates and return the best anchor pose.

    Returns
    -------
    T_best : (4, 4) refined anchor transform
    iou_best : float IoU of best candidate
    idx_best : int index of winning candidate in the rotation set
    diag : dict with 'iou_base', 'iou_best', 'n_tried', 'iou_top5'
    """
    H, W = mask_anchor.shape
    if rotation_set == "cube":
        rots = cube_symmetry_rotations()
    elif rotation_set == "axis":
        rots = axis_aligned_quarter_rotations()
    else:
        raise ValueError(f"unknown rotation_set {rotation_set!r}")

    # Build observed cloud once for t-refine reuse
    obs_cloud = _make_anchor_pointcloud(
        depth_anchor, mask_anchor, hand_mask_anchor, K)

    # Score base pose first (so we know the starting IoU)
    T_base_refined = _t_refine(T_anchor_init, mesh_pts, obs_cloud)
    rendered_base = render_mesh_mask(mesh_pts, T_base_refined, K, H, W)
    iou_base = mask_iou(rendered_base, mask_anchor, hand_mask_anchor)

    # The base pose is "identity" rotation in the candidate set (rots[0]
    # should be I).  But to be safe, evaluate every candidate including ones
    # that match base, and pick the global max.
    R_init = T_anchor_init[:3, :3]
    iou_list: List[Tuple[float, int]] = []
    T_best = T_base_refined
    iou_best = iou_base
    idx_best = -1

    for i, R_cand in enumerate(rots):
        # Apply R_cand pre-multiplied on canonical mesh (i.e., rotate the mesh
        # canonical by R_cand before applying init transform).  Final mesh in
        # camera = (mesh @ R_cand.T) @ R_init.T + t
        # So new R = R_init @ R_cand.
        T_cand = T_anchor_init.copy()
        T_cand[:3, :3] = R_init @ R_cand
        T_cand = _t_refine(T_cand, mesh_pts, obs_cloud)

        rendered = render_mesh_mask(mesh_pts, T_cand, K, H, W)
        iou = mask_iou(rendered, mask_anchor, hand_mask_anchor)
        iou_list.append((iou, i))
        if iou > iou_best + min_iou_to_replace:
            iou_best = iou
            T_best = T_cand
            idx_best = i

    iou_list.sort(reverse=True)
    diag = {
        "iou_base": iou_base,
        "iou_best": iou_best,
        "n_tried": len(rots),
        "iou_top5": iou_list[:5],
        "replaced": idx_best >= 0,
        "winner_idx": idx_best,
    }
    log.info(
        f"orientation_search: base IoU={iou_base:.3f}, "
        f"best IoU={iou_best:.3f} at cand idx={idx_best} "
        f"(top5={[(round(v, 3), i) for v, i in iou_list[:5]]})")
    return T_best, iou_best, idx_best, diag
