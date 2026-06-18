"""Per-frame depth-based pose: align mesh world bbox-center to obs bbox-center.

This is the DEPTH path's core: trust the SAM2-mask backprojected pointcloud
as ground truth for object position.  Rotation is preserved from the anchor
or, when the obs PCA is trustworthy across multiple consecutive frames,
slowly tracked via SLERP from the anchor.

We use **bbox center** (midpoint of (min, max) along each axis) instead of
density-weighted mean.  Reason: obs cloud points concentrate on the
camera-facing surface (more pixels per area at closer distance), so the
density-weighted mean is biased forward toward the camera.  Mesh canonical
points may also be non-uniform.  Bbox center is invariant to density and
lines up the geometric extents instead — which is what we actually want
when both clouds are observations of the same physical bbox.

For the DEPTH_STATIC sub-mode (non-moving objects), pipeline aggregates
per-frame T_depth via SO(3) Riemannian median over the static segment to
lock the pose (robust to single-frame R outliers).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from .so3_utils import slerp_so3, so3_riemannian_median

log = logging.getLogger("pose_tracker.depth_pose")


def compute_depth_pose_per_frame(
    R_anchor: np.ndarray,                          # (3, 3) — rotation baseline at anchor
    mesh_pts: np.ndarray,                          # (M, 3) canonical, scale-corrected
    obs_clouds: Sequence[Optional[np.ndarray]],    # per-frame
    fallback_pose_seq: Sequence[np.ndarray],       # backup when obs missing
    obs_obb_per_frame: Optional[Sequence[Optional[dict]]] = None,
    R_obb_anchor: Optional[np.ndarray] = None,
    *,
    slerp_alpha: float = 0.3,
    require_trust_consecutive: int = 3,
) -> List[np.ndarray]:
    """Per-frame T_depth.

    For each frame:
        if obs_cloud[t] has enough points:
            obs_bbox_center = (obs.min(0) + obs.max(0)) / 2
            mesh_bbox_canon = (mesh.min(0) + mesh.max(0)) / 2

            # Rotation source:
            #   - When obs OBB trustworthy for the last `require_trust_consecutive`
            #     frames in a row, propose R_target = R_obb[t] @ R_obb_anchor.T @ R_anchor
            #     and SLERP from previous R toward R_target with `slerp_alpha`.
            #   - Otherwise fall back to R_anchor.
            T.R = R_t  (per above)
            T.t = obs_bbox_center - R_t @ mesh_bbox_canon
        else:
            T = fallback_pose_seq[t]

    Args:
        slerp_alpha: blend ratio for the per-frame rotation update (0=no
            change, 1=jump to obs PCA proposal).  0.3 means each frame
            covers ~30% of the residual rotation toward the proposal.
        require_trust_consecutive: number of consecutive trustworthy frames
            required before the obs PCA is allowed to drive R.  Avoids
            single-frame PCA flips.

    Returns list of (T,) ndarrays of shape (4, 4).
    """
    T = len(obs_clouds)
    mesh_bbox_canon = (mesh_pts.min(axis=0) + mesh_pts.max(axis=0)) * 0.5
    have_obb = (obs_obb_per_frame is not None and R_obb_anchor is not None)
    R_obb_anchor_T = R_obb_anchor.T if have_obb else None

    out: List[np.ndarray] = []
    n_used = 0
    n_R_from_obb = 0

    R_prev = np.asarray(R_anchor, dtype=np.float64).copy()
    consecutive_trust = 0

    for t in range(T):
        obs = obs_clouds[t]
        if obs is None or len(obs) < 30:
            out.append(np.asarray(fallback_pose_seq[t], dtype=np.float64).copy())
            consecutive_trust = 0
            continue

        # Per-frame R: SLERP toward obs PCA delta when trustworthy chain is long enough
        R_t = R_prev    # default: keep previous (smooth)
        if have_obb:
            obb_t = obs_obb_per_frame[t]
            is_trust = bool(obb_t is not None and obb_t.get('trustworthy'))
            if is_trust:
                consecutive_trust += 1
            else:
                consecutive_trust = 0

            if is_trust and consecutive_trust >= require_trust_consecutive:
                R_obb_t = np.asarray(obb_t['R'], dtype=np.float64)
                R_target = (R_obb_t @ R_obb_anchor_T) @ R_anchor
                R_t = slerp_so3(R_prev, R_target, slerp_alpha)
                n_R_from_obb += 1
            else:
                # Hold R steady on R_anchor while we don't have a confident
                # obs-PCA signal (e.g. first 1-2 frames of a moving span).
                # Drift slowly back to the anchor if previous R diverged.
                R_t = slerp_so3(R_prev, R_anchor, 0.10)
        # If no OBB at all, keep R = R_anchor for every frame.
        if not have_obb:
            R_t = R_anchor

        target = (obs.min(axis=0) + obs.max(axis=0)) * 0.5
        Td = np.eye(4, dtype=np.float64)
        Td[:3, :3] = R_t
        Td[:3, 3] = target - R_t @ mesh_bbox_canon
        out.append(Td)
        n_used += 1
        R_prev = R_t

    log.info(
        f"depth_pose: t computed for {n_used}/{T} frames; "
        f"R from obs PCA SLERP on {n_R_from_obb}/{n_used} frames "
        f"(slerp_alpha={slerp_alpha}, require_trust={require_trust_consecutive})")
    return out


def lock_pose_for_segment(
    T_depth_seq: Sequence[np.ndarray],
    seg_start: int,
    seg_end: int,
    R_anchor: np.ndarray,
) -> np.ndarray:
    """Median-lock pose over a static segment.

    Returns a single (4, 4) pose to use for all frames [seg_start, seg_end].
        - t : component-wise median of per-frame T_depth translations.
        - R : SO(3) Riemannian median of per-frame Rs (Weiszfeld's
              algorithm), robust to single-frame outliers.
    """
    seg_translations = []
    seg_rotations = []
    for t in range(seg_start, seg_end + 1):
        Td = T_depth_seq[t]
        if Td is None:
            continue
        Td_np = np.asarray(Td)
        seg_translations.append(Td_np[:3, 3])
        seg_rotations.append(Td_np[:3, :3])
    if not seg_translations:
        # All-None segment: identity-translated SE(3) from R_anchor
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R_anchor
        return out
    med_t = np.median(np.stack(seg_translations, axis=0), axis=0)
    R_med = so3_riemannian_median(seg_rotations)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_med
    out[:3, 3] = med_t
    return out
