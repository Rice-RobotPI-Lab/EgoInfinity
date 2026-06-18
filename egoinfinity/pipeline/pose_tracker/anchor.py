"""Stage 0 + Stage 1 of pose tracker:

Stage 0  Pick the best anchor frame (high visibility, low hand occlusion,
         object roughly stationary).
Stage 1  Solve anchor 6DoF — FGR (global) → ICP (refine) — and run a
         Umeyama scale-consistency check between the SAM3D mesh and the
         MoGe-2 metric depth.  Returns (T_anchor, scale_correction).

The same observed point cloud + cleaned mesh sample is reused across both
substeps; building it is the most expensive part (a few ms per frame).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .utils import (
    backproject_mask,
    filter_outliers_sor_np,
    umeyama_with_scale,
)

log = logging.getLogger("pose_tracker.anchor")


# ---------------------------------------------------------------------------
# Stage 0: anchor frame selection
# ---------------------------------------------------------------------------
def _mask_centroid(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return np.array([xs.mean(), ys.mean()])


def select_anchor_frame(
    masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
    motion_window: int = 2,
    motion_sigma: float = 30.0,
) -> int:
    """Score every frame and return the index of the best anchor.

    Score:  visibility * (1 − hand_overlap) * exp(−motion / sigma)
        - visibility = mask area / (70th percentile of all areas)
        - hand_overlap = |mask ∩ hand| / |mask|
        - motion = mean centroid displacement to ±motion_window neighbors

    Returns ``-1`` if no frame has any mask.
    """
    total = len(masks)
    areas = np.array([m.sum() if m is not None else 0 for m in masks], dtype=np.float64)
    if (areas > 0).sum() == 0:
        return -1
    p70 = np.percentile(areas[areas > 0], 70)
    centroids = [_mask_centroid(m) for m in masks]

    best_t, best_s = -1, -np.inf
    for t in range(total):
        if masks[t] is None or not masks[t].any():
            continue
        # (1) visibility (clip to [0, 1.5])
        vis = min(areas[t] / max(p70, 1.0), 1.5)
        # (2) occlusion (low is good)
        if hand_masks[t] is not None and hand_masks[t].any():
            inter = (masks[t] & hand_masks[t]).sum()
            occlusion = inter / max(areas[t], 1)
        else:
            occlusion = 0.0
        if occlusion >= 1.0:
            continue
        # (3) motion (small is good — exp decay)
        motion = 0.0; n_motion = 0
        for dt in range(-motion_window, motion_window + 1):
            if dt == 0:
                continue
            tt = t + dt
            if 0 <= tt < total and centroids[t] is not None and centroids[tt] is not None:
                motion += float(np.linalg.norm(centroids[t] - centroids[tt]))
                n_motion += 1
        motion = motion / max(n_motion, 1)
        score = vis * (1.0 - min(occlusion, 0.7)) * np.exp(-motion / motion_sigma)
        if score > best_s:
            best_s, best_t = score, t
    log.info(f"anchor_frame={best_t}  score={best_s:.3f}")
    return int(best_t)


# ---------------------------------------------------------------------------
# Stage 1: anchor pose (FGR + ICP) + scale check
# ---------------------------------------------------------------------------
@dataclass
class AnchorResult:
    frame_idx: int
    T: np.ndarray                 # (4, 4) mesh canonical → camera @ anchor
    scale_correction: float       # apply once to mesh.xyz
    inlier_rmse: float            # ICP convergence metric (m)
    fitness: float                # ICP overlap fraction
    n_obs_pts: int                # observed cloud size
    n_mesh_pts: int               # mesh sample size
    nn_mean_err_m: float          # mean nearest-neighbor distance after fit
    note: str = ""                # human-readable status


def _make_anchor_pointcloud(
    depth: np.ndarray, mask: np.ndarray, hand_mask: Optional[np.ndarray],
    K: np.ndarray, *, erode_px: int = 3, sobel_thresh: float = 0.10,
    sor_k: int = 20, step: int = 1,
) -> np.ndarray:
    """Build a clean 3D point cloud of the object surface from the
    anchor frame's depth map.

    Steps: subtract hand mask, erode, drop pixels with high depth gradient
    (Sobel — kills mask boundary noise), back-project, SOR.
    """
    O = mask & ~hand_mask if hand_mask is not None else mask.copy()
    if not O.any():
        return np.zeros((0, 3), dtype=np.float32)

    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1,) * 2, dtype=np.uint8)
        O = cv2.erode(O.astype(np.uint8), kernel).astype(bool)
        if not O.any():
            return np.zeros((0, 3), dtype=np.float32)

    # Sobel on depth; high-gradient pixels (mask edges, depth jumps) are noisy.
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx * dx + dy * dy)
    O &= (grad < sobel_thresh)
    if not O.any():
        return np.zeros((0, 3), dtype=np.float32)

    pts = backproject_mask(O, depth, K, step=step)
    if len(pts) > sor_k + 1:
        pts = filter_outliers_sor_np(pts, k=sor_k, std_ratio=2.0)
    return pts


def _sample_mesh_points(mesh_pts: np.ndarray, n: int = 5000) -> np.ndarray:
    if len(mesh_pts) <= n:
        return mesh_pts.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mesh_pts), n, replace=False)
    return mesh_pts[idx].astype(np.float64)


def _rough_align_scale_center(src: np.ndarray, tgt: np.ndarray
                              ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Translate/scale src so it roughly matches tgt in size and position.

    Returns (src_aligned, scale, src_centroid, tgt_centroid) so callers can
    invert the transform.  Used as a pre-pass so FGR doesn't fail when mesh
    canonical coords (~1-unit extent) meet metric observed coords (~0.4 m).
    """
    src_c = src.mean(0)
    tgt_c = tgt.mean(0)
    src_std = float(np.linalg.norm(src - src_c, axis=1).mean())
    tgt_std = float(np.linalg.norm(tgt - tgt_c, axis=1).mean())
    if src_std < 1e-6:
        return src.copy(), 1.0, src_c, tgt_c
    s = tgt_std / src_std
    src_aligned = (src - src_c) * s + tgt_c
    return src_aligned, s, src_c, tgt_c


def _fgr_then_icp(src: np.ndarray, tgt: np.ndarray,
                  voxel: Optional[float] = None,
                  T_init_sim: Optional[np.ndarray] = None,
                  ) -> tuple[np.ndarray, float, float, float]:
    """Run FGR for global init then point-to-plane ICP refinement.

    If ``T_init_sim`` (4×4 similarity transform mesh-canonical→camera) is
    provided, FGR is skipped and ICP is initialized from it.  Useful when a
    strong prior (e.g. SAM3D's reported single-frame pose) is available; this
    sidesteps FPFH rotation ambiguity for elongated / near-symmetric objects.

    Otherwise (a) pre-aligns mesh → observed via centroid + uniform scale and
    (b) runs FGR on the pre-aligned cloud.

    Voxel size is auto-picked from the observed cloud extent (~2% of bbox).
    Returns (T_final 4×4 mesh-canonical→camera, fitness, inlier_rmse, scale_applied).
    """
    import open3d as o3d

    # Pick voxel size from observed extent (~ 2 % of the bbox diagonal)
    if voxel is None:
        ext = tgt.max(0) - tgt.min(0)
        diag = float(np.linalg.norm(ext))
        voxel = max(diag * 0.02, 0.003)

    if T_init_sim is not None:
        # Decompose T_init into (s_pre, R_pre, t_pre) so we can match the
        # FGR-path interface: ICP runs in the "pre-aligned" frame.  We let
        # the pre-align scale = the scale baked into T_init's rotation
        # (so the rest of the transform in pre-aligned frame is identity).
        R_sim = T_init_sim[:3, :3]
        s_pre = float(np.cbrt(max(abs(np.linalg.det(R_sim)), 1e-12)))
        # mesh in camera under the prior init:
        ones = np.ones((len(src), 1))
        src_h = np.concatenate([src, ones], axis=1)
        src_pre = (src_h @ T_init_sim.T)[:, :3]   # already mesh-in-camera
        # ICP init = identity (prior placed mesh near observation already)
        T_init_for_icp = np.eye(4)
        # Pre-transform recovery: T_pre = T_init_sim
        T_pre_recovery = T_init_sim
    else:
        # FGR path: pre-align via centroid + uniform scale, then FGR.
        src_pre, s_pre, src_c, tgt_c = _rough_align_scale_center(src, tgt)
        T_pre = np.eye(4)
        T_pre[:3, :3] = s_pre * np.eye(3)
        T_pre[:3, 3] = tgt_c - s_pre * src_c
        T_pre_recovery = T_pre

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pre)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt)

    src_d = src_pcd.voxel_down_sample(voxel)
    tgt_d = tgt_pcd.voxel_down_sample(voxel)
    for pc in (src_d, tgt_d):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=2 * voxel, max_nn=30))

    if T_init_sim is None:
        src_f = o3d.pipelines.registration.compute_fpfh_feature(
            src_d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
        tgt_f = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
        fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_d, tgt_d, src_f, tgt_f,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=2.5 * voxel))
        T_init_for_icp = np.asarray(fgr.transformation, dtype=np.float64)

    # ICP refinement (point-to-plane)
    tgt_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=30))
    icp = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_correspondence_distance=5 * voxel,
        init=T_init_for_icp,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T_on_pre = np.asarray(icp.transformation, dtype=np.float64)

    # Compose: T_on_pre acts on the pre-aligned mesh; T_pre_recovery maps
    # raw canonical mesh → pre-aligned frame.  Final T_final acts on raw.
    T_final = T_on_pre @ T_pre_recovery

    return T_final, float(icp.fitness), float(icp.inlier_rmse), s_pre


def _verify_scale(mesh_in_cam: np.ndarray, observed: np.ndarray,
                  *, no_correct_below_m: float = 0.005,
                  warn_only_below_m: float = 0.010,
                  ) -> tuple[float, float]:
    """Compute mean nearest-neighbor distance from mesh-in-cam to observed,
    and decide whether to correct mesh scale via Umeyama."""
    if len(observed) < 50 or len(mesh_in_cam) < 50:
        return 1.0, float('inf')
    tree = cKDTree(observed)
    dists, idx = tree.query(mesh_in_cam, k=1)
    mean_err = float(np.mean(dists))
    if mean_err < no_correct_below_m:
        return 1.0, mean_err
    if mean_err < warn_only_below_m:
        # within tolerance — don't risk over-correcting
        return 1.0, mean_err
    # Use mean of best-50% closest pairs for Umeyama (avoids outliers)
    order = np.argsort(dists)[: len(dists) // 2]
    src = mesh_in_cam[order]
    dst = observed[idx[order]]
    s, _R, _t = umeyama_with_scale(src, dst)
    # Shrink-only: never trust Umeyama to grow the mesh past SAM3D's scale.
    # Empirically all problem cases so far were SAM3D over-sizing; if SAM3D
    # under-sized, the bbox check is intentionally one-directional too.
    s = float(np.clip(s, 0.1, 1.0))
    return s, mean_err


def estimate_anchor_pose(
    *,
    anchor_idx: int,
    mesh_pts_canonical: np.ndarray,        # (N, 3) PLY-loaded gaussian xyz
    depth: np.ndarray,                     # (H, W) float32 metric depth
    mask: np.ndarray,                      # (H, W) bool — SAM2 object mask
    hand_mask: Optional[np.ndarray],       # (H, W) bool or None
    K: np.ndarray,                         # (3, 3)
    voxel: float = 0.01,
    n_mesh_sample: int = 5000,
    T_init_sim: Optional[np.ndarray] = None,   # 4x4 similarity prior; skips FGR
    world_up: Optional[np.ndarray] = None,     # (3,) gravity unit vec in cam frame
) -> AnchorResult:
    """Solve mesh→camera SE(3) at the anchor frame and check scale.

    Coordinate convention: camera frame is OpenCV (+X right, +Y down, +Z forward).
    Returns ``AnchorResult`` with the 4×4 transform and a one-shot scale
    correction to apply to the mesh's vertices (so subsequent frames see
    the rescaled mesh).
    """
    obs = _make_anchor_pointcloud(depth, mask, hand_mask, K)
    if len(obs) < 100:
        return AnchorResult(
            frame_idx=anchor_idx, T=np.eye(4),
            scale_correction=1.0, inlier_rmse=float('inf'),
            fitness=0.0, n_obs_pts=len(obs),
            n_mesh_pts=len(mesh_pts_canonical),
            nn_mean_err_m=float('inf'),
            note=f"observed cloud too small ({len(obs)} pts)")

    mesh_src = _sample_mesh_points(mesh_pts_canonical, n_mesh_sample)

    T_sim, fitness, rmse, s_pre = _fgr_then_icp(
        mesh_src, obs.astype(np.float64), voxel=None,
        T_init_sim=T_init_sim)

    # Apply the final similarity to raw canonical mesh for scale check.
    ones = np.ones((len(mesh_src), 1))
    mesh_h = np.concatenate([mesh_src, ones], axis=1)
    mesh_in_cam = (mesh_h @ T_sim.T)[:, :3]
    # Only attempt Umeyama scale refinement when ICP found a meaningful overlap.
    # Low fitness means very few correspondences, so Umeyama is fitting noise
    # and tends to over-shrink (observed: turkey 0.32 → 0.27, ~15% loss).
    # Threshold 0.5 chosen so fitness=0.7 (clean fit) refines, 0.3 (sparse) does not.
    if fitness >= 0.5:
        refine_corr, nn_err = _verify_scale(mesh_in_cam, obs.astype(np.float64))
    else:
        # Trust SAM3D's canonical_scale when ICP can't validate it.
        refine_corr = 1.0
        from scipy.spatial import cKDTree as _kt
        if len(obs) >= 50 and len(mesh_in_cam) >= 50:
            _t = _kt(obs.astype(np.float64))
            _d, _ = _t.query(mesh_in_cam, k=1)
            nn_err = float(np.mean(_d))
        else:
            nn_err = float('inf')

    # Decompose T_sim into pure SE(3) + total scale.
    # T_sim[:3,:3] = s_baked * R_pure where s_baked ≈ s_pre.  We strip the
    # scale so callers get a true rigid pose; the full scale (pre + verify
    # refinement) is returned via ``scale_correction`` so downstream code
    # multiplies the canonical mesh once and tracks with rigid PnP.
    R_sim = T_sim[:3, :3]
    s_baked = float(np.cbrt(max(abs(np.linalg.det(R_sim)), 1e-12)))
    R_pure = R_sim / s_baked
    # Re-orthonormalize R via SVD to suppress numerical drift
    Ur, _, Vrt = np.linalg.svd(R_pure)
    R_pure = Ur @ Vrt
    if np.linalg.det(R_pure) < 0:
        Ur[:, -1] *= -1
        R_pure = Ur @ Vrt
    T = np.eye(4)
    T[:3, :3] = R_pure
    T[:3, 3] = T_sim[:3, 3]
    scale_corr = s_baked * refine_corr

    # 3D horizontal-plane sanity check (F1 + F4).
    #
    # The previous 2D render-IoU check was insensitive to small objects
    # (~px-level area noise dominates the ratio) and missed Z-axis errors
    # entirely.  We now compare extents in metric 3D, projected onto the
    # gravity-orthogonal horizontal plane:
    #
    #   horiz_ratio  = mesh_horiz_extent / obs_horiz_extent  (90-pct extent)
    #   chamfer_m    = mean partial mesh→obs nearest-neighbour distance
    #
    #   trigger:  horiz_ratio > 1.5  AND  chamfer_m > 0.02
    #     → shrink scale_corr by sqrt(horiz_ratio)  (clamped ≤ 30%)
    #
    # Why horizontal plane: obs cloud always misses the back-side of the
    # object (depth only sees front face) — that bias is in the gravity
    # direction.  Projecting to the horizontal plane drops the missing
    # axis and leaves extents that mesh and obs should both agree on.
    # Why chamfer corroboration: obs being mask-occluded on one side will
    # still shrink obs extent, but won't shift mesh→obs nn distances
    # systematically high; demanding both signals catches real oversize.
    pca_ratio = 1.0
    chamfer_m = float('nan')
    bbox_corr = 1.0
    try:
        if len(obs) >= 50:
            mesh_world_full = mesh_pts_canonical * scale_corr @ R_pure.T + T[:3, 3]
            # PCA on mesh (full body) and obs cloud (visible side).
            # Compare second-longest axis extents:
            #   - Longest axis often includes thin protrusions (handle, spout)
            #     that obs cloud doesn't capture, biasing the ratio upward.
            #   - Second-longest axis = "main body" extent, which obs cloud is
            #     likely to capture even with partial occlusion.
            def _pca_extents(p):
                p = np.asarray(p, dtype=np.float64)
                c = p.mean(axis=0)
                cov = np.cov((p - c).T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, order]
                proj = (p - c) @ eigvecs              # PCA-aligned coords
                lo = np.percentile(proj, 5.0, axis=0)
                hi = np.percentile(proj, 95.0, axis=0)
                return hi - lo                         # (3,) descending order
            mesh_ext = _pca_extents(mesh_world_full)
            obs_ext = _pca_extents(obs)
            # Second-longest extent (index 1) — robust to thin protrusions
            mesh_ext2 = float(mesh_ext[1])
            obs_ext2 = float(obs_ext[1])
            pca_ratio = mesh_ext2 / max(obs_ext2, 1e-3)
            # Chamfer corroboration
            from scipy.spatial import cKDTree as _kt
            tree = _kt(obs)
            d, _ = tree.query(mesh_world_full, k=1)
            d_sorted = np.sort(d)
            n_keep = max(int(len(d_sorted) * 0.5), 1)
            chamfer_m = float(np.mean(d_sorted[:n_keep]))
            if pca_ratio > 1.5 and chamfer_m > 0.02:
                k_lin = float(np.sqrt(pca_ratio))
                bbox_corr = max(1.0 / k_lin, 0.7)   # clamp ≤ 30% shrink
                log.warning(
                    f"anchor pose: mesh PCA-axis2 {pca_ratio:.2f}x obs "
                    f"(mesh={mesh_ext2*100:.1f}cm vs obs={obs_ext2*100:.1f}cm, "
                    f"chamfer={chamfer_m*1000:.0f}mm) — shrinking scale_corr "
                    f"{scale_corr:.3f} → {scale_corr * bbox_corr:.3f}")
                scale_corr *= bbox_corr
    except Exception as _bbe:
        log.debug(f"PCA-axis2 sanity check skipped: {_bbe}")

    note = (f"pre_s={s_pre:.3f} fitness={fitness:.3f} rmse={rmse*1000:.1f}mm "
            f"nn={nn_err*1000:.1f}mm scale_correction={scale_corr:.3f} "
            f"pca_ratio={pca_ratio:.2f} chamfer={chamfer_m*1000:.0f}mm")
    log.info(f"anchor pose: {note}")
    return AnchorResult(
        frame_idx=anchor_idx, T=T,
        scale_correction=scale_corr,
        inlier_rmse=rmse,
        fitness=fitness,
        n_obs_pts=len(obs),
        n_mesh_pts=len(mesh_src),
        nn_mean_err_m=nn_err,
        note=note,
    )


# ---------------------------------------------------------------------------
# 2026-05-01 — Prior-only anchor (skip ICP entirely)
# ---------------------------------------------------------------------------
def build_anchor_from_prior(
    *,
    anchor_idx: int,
    T_init_sim: np.ndarray,                       # (4, 4) mesh-canonical → camera (incl. SAM3D scale)
    mesh_pts_canonical: np.ndarray,               # (M, 3) raw mesh points
    depth: np.ndarray,                            # (H, W) depth map at anchor frame
    mask: np.ndarray,                             # (H, W) SAM2 object mask
    hand_mask: Optional[np.ndarray],              # (H, W) hand mask
    K: np.ndarray,                                # (3, 3)
) -> AnchorResult:
    """Build an anchor pose **directly** from the SAM3D similarity prior +
    observation point cloud — no FGR, no ICP, no Umeyama.

    R       : extracted from T_init_sim (SAM3D's canonical_rotation_quat).
    scale   : extracted from T_init_sim (SAM3D's canonical_scale).
              Optionally shrunk by the PCA-axis2 + chamfer sanity check
              (the only piece of "fit verification" we keep).
    t       : aligned so the (scale-corrected) mesh's mean lands on the
              observation cloud's mean.  This catches the constant SAM3D ↔
              MoGe-2 frame offset.

    Returns an ``AnchorResult`` populated with diagnostic fields:
        ``fitness=1.0`` (we trust the prior by definition)
        ``inlier_rmse`` = mean nearest-neighbour mesh→obs distance
    """
    obs = _make_anchor_pointcloud(depth, mask, hand_mask, K)

    # ---- Decompose T_init_sim into (R_pure, scale, t_init) ----
    M = np.asarray(T_init_sim[:3, :3], dtype=np.float64)
    s_pre = float(np.cbrt(max(abs(np.linalg.det(M)), 1e-12)))
    R_pure = M / s_pre
    U, _, Vt = np.linalg.svd(R_pure)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:
        U[:, -1] *= -1
        R_pure = U @ Vt

    scale_corr = s_pre

    # ---- t: align (scaled mesh) mean to obs mean ----
    if len(obs) >= 30:
        mesh_in_cam_no_t = (mesh_pts_canonical * scale_corr) @ R_pure.T
        mesh_c_no_t = mesh_in_cam_no_t.mean(axis=0)
        obs_c = obs.mean(axis=0)
        t_anchor = obs_c - mesh_c_no_t
    else:
        # No obs → fall back to the prior's translation
        t_anchor = np.asarray(T_init_sim[:3, 3], dtype=np.float64)

    # ---- PCA-axis2 + chamfer sanity check (shrink-only) ----
    pca_ratio = 1.0
    chamfer_m = float('nan')
    nn_err = float('nan')
    bbox_corr = 1.0
    n_obs = int(len(obs))
    try:
        if n_obs >= 50:
            mesh_world_full = mesh_pts_canonical * scale_corr @ R_pure.T + t_anchor

            def _pca_extents(p):
                p = np.asarray(p, dtype=np.float64)
                c = p.mean(axis=0)
                cov = np.cov((p - c).T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, order]
                proj = (p - c) @ eigvecs
                lo = np.percentile(proj, 5.0, axis=0)
                hi = np.percentile(proj, 95.0, axis=0)
                return hi - lo

            mesh_ext = _pca_extents(mesh_world_full)
            obs_ext = _pca_extents(obs)
            mesh_ext2 = float(mesh_ext[1])
            obs_ext2 = float(obs_ext[1])
            pca_ratio = mesh_ext2 / max(obs_ext2, 1e-3)

            from scipy.spatial import cKDTree as _kt
            tree = _kt(obs)
            d, _ = tree.query(mesh_world_full, k=1)
            d_sorted = np.sort(d)
            n_keep = max(int(len(d_sorted) * 0.5), 1)
            chamfer_m = float(np.mean(d_sorted[:n_keep]))
            nn_err = chamfer_m

            if pca_ratio > 1.5 and chamfer_m > 0.02:
                k_lin = float(np.sqrt(pca_ratio))
                bbox_corr = max(1.0 / k_lin, 0.7)
                log.warning(
                    f"build_anchor_from_prior: PCA-axis2 {pca_ratio:.2f}x obs "
                    f"(mesh={mesh_ext2*100:.1f}cm vs obs={obs_ext2*100:.1f}cm, "
                    f"chamfer={chamfer_m*1000:.0f}mm) — shrinking scale "
                    f"{scale_corr:.3f} → {scale_corr * bbox_corr:.3f}")
                scale_corr *= bbox_corr
                # Recompute t with the shrunk scale so mesh mean still lands on obs mean
                mesh_in_cam_no_t = (mesh_pts_canonical * scale_corr) @ R_pure.T
                t_anchor = obs.mean(axis=0) - mesh_in_cam_no_t.mean(axis=0)
    except Exception as e:
        log.debug(f"PCA-axis2 sanity check skipped: {e}")

    # ---- Compose final SE(3) (no scale baked in — caller multiplies mesh_xyz by scale_correction) ----
    T_final = np.eye(4, dtype=np.float64)
    T_final[:3, :3] = R_pure
    T_final[:3, 3] = t_anchor

    note = (f"prior-only anchor: scale={scale_corr:.3f} "
            f"pca_ratio={pca_ratio:.2f} chamfer={chamfer_m*1000:.0f}mm "
            f"n_obs={n_obs}")
    log.info(note)

    return AnchorResult(
        frame_idx=anchor_idx,
        T=T_final,
        scale_correction=scale_corr,
        inlier_rmse=nn_err if nn_err == nn_err else 0.0,
        fitness=1.0,                   # by construction — we trust the prior
        n_obs_pts=n_obs,
        n_mesh_pts=int(len(mesh_pts_canonical)),
        nn_mean_err_m=nn_err if nn_err == nn_err else 0.0,
        note=note,
    )
