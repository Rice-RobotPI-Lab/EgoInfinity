"""Stage 2: optical flow + RANSAC-PnP pose propagation.

Given a known pose at frame *t* and (I_t, I_{t+1}), propagate to t+1 by:

  1. Sample mesh surface points; transform to camera and project to pixels at t.
  2. Filter visibility: inside image, inside SAM2 mask at t,
     not inside hand mask at t.
  3. Run dense optical flow I_t -> I_{t+1} (OpenCV DIS; CPU, good quality,
     no GPU dep).  Sample flow at each kept pixel to get its
     corresponding location at t+1.
  4. Second-round filter: u_{t+1} inside mask_{t+1}, not inside hand_mask_{t+1}.
  5. solvePnPRansac on (3D mesh pts, 2D pixels at t+1).
  6. LM refine with inliers only.

Running in both directions from the anchor fills an 82-frame clip in a few
seconds (dominated by flow compute).

Gaps (propagation failures or below-threshold inlier count) are filled by
SE(3) interpolation between nearest trusted neighbors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import project_points

log = logging.getLogger("pose_tracker.flow_pnp")


# ---------------------------------------------------------------------------
# Optical flow — MEMFOF (deep, GPU)
# ---------------------------------------------------------------------------
def make_flow_engine():
    """Return the singleton MEMFOF flow engine."""
    from .memfof_flow import get_global_engine
    engine = get_global_engine()
    engine._ensure_loaded()
    return engine


def compute_flow(engine, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Dense flow from A to B.  Returns (H, W, 2) float32 (dx, dy in pixels)."""
    from .memfof_flow import compute_flow as memfof_compute
    return memfof_compute(engine, img_a, img_b)


# ---------------------------------------------------------------------------
# Visibility filter
# ---------------------------------------------------------------------------
def _inside_mask(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-point 0/1 — True if uv (N, 2 int) lies on a True pixel of mask."""
    if mask is None:
        return np.ones(len(uv), dtype=bool)
    H, W = mask.shape
    u = uv[:, 0]; v = uv[:, 1]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.zeros(len(uv), dtype=bool)
    if in_img.any():
        out[in_img] = mask[v[in_img].astype(np.int32), u[in_img].astype(np.int32)]
    return out


def _filter_visible(mesh_pts: np.ndarray, T: np.ndarray, K: np.ndarray,
                    H: int, W: int, mask: Optional[np.ndarray],
                    hand_mask: Optional[np.ndarray],
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Keep mesh points that are (a) in front of camera, (b) in image,
    (c) inside SAM2 mask, (d) NOT inside hand mask."""
    R, t = T[:3, :3], T[:3, 3]
    x_cam = mesh_pts @ R.T + t
    # (a) in front
    keep = x_cam[:, 2] > 1e-3
    if not keep.any():
        return np.zeros((0, 3)), np.zeros((0, 2))
    mesh_pts = mesh_pts[keep]; x_cam = x_cam[keep]
    # (b) project
    uv = project_points(K, x_cam)
    uv_int = uv.astype(np.int32)
    in_img = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) \
           & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    mesh_pts = mesh_pts[in_img]; uv = uv[in_img]; uv_int = uv_int[in_img]
    # (c) inside SAM2 mask
    if mask is not None:
        m = _inside_mask(uv_int, mask)
        mesh_pts = mesh_pts[m]; uv = uv[m]; uv_int = uv_int[m]
    # (d) not inside hand mask
    if hand_mask is not None and len(uv_int) > 0:
        h = _inside_mask(uv_int, hand_mask)
        mesh_pts = mesh_pts[~h]; uv = uv[~h]
    return mesh_pts, uv


# ---------------------------------------------------------------------------
# Propagate one step
# ---------------------------------------------------------------------------
@dataclass
class PropagationResult:
    T_next: Optional[np.ndarray]       # 4x4 or None on failure
    n_corr: int                        # correspondences fed to PnP
    n_inliers: int                     # RANSAC inliers
    inlier_ratio: float                # n_inliers / max(n_corr, 1)
    fail_reason: str = ""


def propagate(
    T_curr: np.ndarray,
    mesh_pts: np.ndarray,
    img_t: np.ndarray,
    img_tnext: np.ndarray,
    mask_t: Optional[np.ndarray],
    mask_tnext: Optional[np.ndarray],
    hand_mask_t: Optional[np.ndarray],
    hand_mask_tnext: Optional[np.ndarray],
    K: np.ndarray,
    flow_engine,
    n_mesh_sample: int = 1000,
    reproj_thresh: float = 3.0,
    min_correspondences: int = 4,           # PnP minimum
    min_inliers: int = 4,                   # bare minimum for solve
    max_flow: float = 80.0,
    max_translation_jump: float = 0.20,    # m, frame-to-frame
    seed: int = 0,
) -> PropagationResult:
    """Single-step pose propagation t -> t' via optical flow + RANSAC-PnP."""
    H, W = img_t.shape[:2]
    # 1. sub-sample mesh (deterministic)
    if len(mesh_pts) > n_mesh_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(mesh_pts), n_mesh_sample, replace=False)
        X_all = mesh_pts[idx].astype(np.float64)
    else:
        X_all = mesh_pts.astype(np.float64)

    # 2. visibility at t
    X_vis, u_t = _filter_visible(X_all, T_curr, K, H, W, mask_t, hand_mask_t)
    if len(X_vis) < min_correspondences:
        return PropagationResult(None, len(X_vis), 0, 0.0,
                                  f"too few visible at t ({len(X_vis)})")

    # 3. optical flow t -> t+1, sample at u_t
    flow = compute_flow(flow_engine, img_t, img_tnext)     # (H, W, 2)
    u_int = u_t.astype(np.int32)
    du = flow[u_int[:, 1], u_int[:, 0]]
    mag = np.linalg.norm(du, axis=1)
    ok_flow = mag < max_flow
    X_vis, u_t = X_vis[ok_flow], u_t[ok_flow]; du = du[ok_flow]
    u_next = u_t + du

    # 4. second filter: in t+1 image / mask / not-hand
    u_ni = u_next.astype(np.int32)
    in_img = (u_ni[:, 0] >= 0) & (u_ni[:, 0] < W) \
           & (u_ni[:, 1] >= 0) & (u_ni[:, 1] < H)
    X_vis, u_next, u_ni = X_vis[in_img], u_next[in_img], u_ni[in_img]
    if mask_tnext is not None and len(u_ni) > 0:
        m = _inside_mask(u_ni, mask_tnext)
        X_vis, u_next = X_vis[m], u_next[m]; u_ni = u_ni[m]
    if hand_mask_tnext is not None and len(u_ni) > 0:
        h = _inside_mask(u_ni, hand_mask_tnext)
        X_vis, u_next = X_vis[~h], u_next[~h]

    n_corr = len(X_vis)
    if n_corr < min_correspondences:
        return PropagationResult(None, n_corr, 0, 0.0,
                                  f"too few correspondences ({n_corr})")

    # 5. RANSAC PnP
    X_pnp = X_vis.astype(np.float32).reshape(-1, 1, 3)
    u_pnp = u_next.astype(np.float32).reshape(-1, 1, 2)
    # Use current pose as initial guess to help EPnP
    rvec0, _ = cv2.Rodrigues(T_curr[:3, :3])
    tvec0 = T_curr[:3, 3:4]
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_pnp, u_pnp, K.astype(np.float64), None,
            rvec=rvec0.astype(np.float64), tvec=tvec0.astype(np.float64),
            useExtrinsicGuess=True,
            iterationsCount=200, reprojectionError=reproj_thresh,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error as e:
        return PropagationResult(None, n_corr, 0, 0.0, f"solvePnPRansac error: {e}")
    if not ok or inliers is None or len(inliers) < min_inliers:
        n_in = 0 if inliers is None else len(inliers)
        return PropagationResult(None, n_corr, n_in, n_in / max(n_corr, 1),
                                  f"insufficient RANSAC inliers ({n_in})")

    # 6. LM refine
    inlier_idx = inliers.flatten()
    X_in = X_vis[inlier_idx].astype(np.float64).reshape(-1, 1, 3)
    u_in = u_next[inlier_idx].astype(np.float64).reshape(-1, 1, 2)
    rvec_r, tvec_r = cv2.solvePnPRefineLM(
        X_in, u_in, K.astype(np.float64), None, rvec, tvec)
    R_new, _ = cv2.Rodrigues(rvec_r)
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = tvec_r.flatten()

    # 7. Sanity: reject implausibly large frame-to-frame translation jumps.
    # Realistic single-frame motion at 30 fps is < 5 cm even for fast manipulation;
    # > 20 cm in one frame is almost always PnP solving to a degenerate pose
    # (very few correspondences, all clustered).  Reject so the gap-fill can
    # interpolate from the next clean frame instead of locking to a bad pose.
    dt = float(np.linalg.norm(T_new[:3, 3] - T_curr[:3, 3]))
    if dt > max_translation_jump:
        return PropagationResult(None, n_corr, len(inliers),
                                  len(inliers) / n_corr,
                                  f"translation jump {dt*100:.1f}cm > {max_translation_jump*100:.0f}cm")

    return PropagationResult(T_new, n_corr, len(inliers),
                              len(inliers) / n_corr, "")


# ---------------------------------------------------------------------------
# Bidirectional sweep + SE(3) gap-fill
# ---------------------------------------------------------------------------
def _se3_slerp(Ta: np.ndarray, Tb: np.ndarray, u: float) -> np.ndarray:
    """Interpolate between two SE(3) matrices: linear on translation, SLERP on rotation."""
    from scipy.spatial.transform import Rotation, Slerp
    Ra = Ta[:3, :3]; ta = Ta[:3, 3]
    Rb = Tb[:3, :3]; tb = Tb[:3, 3]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))
    R_u = slerp([u]).as_matrix()[0]
    t_u = ta * (1 - u) + tb * u
    T = np.eye(4); T[:3, :3] = R_u; T[:3, 3] = t_u
    return T


def _fill_gaps(T_seq: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Fill None entries by interpolating between nearest known neighbors.
    Extrapolate by holding the nearest known pose at both ends."""
    n = len(T_seq)
    known = [i for i, T in enumerate(T_seq) if T is not None]
    if not known:
        return T_seq
    out = list(T_seq)
    # leading Nones
    for i in range(0, known[0]):
        out[i] = T_seq[known[0]].copy()
    # trailing Nones
    for i in range(known[-1] + 1, n):
        out[i] = T_seq[known[-1]].copy()
    # interior gaps
    for a, b in zip(known, known[1:]):
        if b - a <= 1:
            continue
        for k in range(1, b - a):
            u = k / (b - a)
            out[a + k] = _se3_slerp(T_seq[a], T_seq[b], u)
    return out


@dataclass
class TrackResult:
    T_seq: List[np.ndarray]                   # per-frame 4x4 (no Nones after fill)
    inlier_ratios: List[float]                # per-frame (0.0 for gap-filled)
    n_corr: List[int]                         # per-frame correspondence count
    propagated: List[bool]                    # True if computed, False if gap-filled
    fail_reasons: List[str]                   # per-frame fail reason (empty if OK)


def track_6dof(
    mesh_pts: np.ndarray,
    anchor_idx: int,
    T_anchor: np.ndarray,
    frames_rgb: Sequence[np.ndarray],
    masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
    K: np.ndarray,
    *,
    n_mesh_sample: int = 1000,
    reproj_thresh: float = 3.0,
    min_correspondences: int = 8,
    min_inliers: int = 15,
) -> TrackResult:
    """Bidirectional optical-flow + PnP tracking from ``T_anchor``.

    Returns per-frame poses, each either computed by propagation or filled
    in by SE(3) interpolation (see ``propagated`` flag).
    """
    total = len(frames_rgb)
    T_seq: List[Optional[np.ndarray]] = [None] * total
    inlier_ratios = [0.0] * total
    n_corr = [0] * total
    propagated = [False] * total
    fail = [""] * total

    T_seq[anchor_idx] = T_anchor
    inlier_ratios[anchor_idx] = 1.0
    propagated[anchor_idx] = True

    flow_engine = make_flow_engine()

    def _step(t_from: int, t_to: int):
        if T_seq[t_from] is None:
            return
        res = propagate(
            T_seq[t_from], mesh_pts,
            frames_rgb[t_from], frames_rgb[t_to],
            masks[t_from], masks[t_to],
            hand_masks[t_from], hand_masks[t_to],
            K, flow_engine,
            n_mesh_sample=n_mesh_sample,
            reproj_thresh=reproj_thresh,
            min_correspondences=min_correspondences,
            min_inliers=min_inliers,
        )
        if res.T_next is not None:
            T_seq[t_to] = res.T_next
            propagated[t_to] = True
        inlier_ratios[t_to] = res.inlier_ratio
        n_corr[t_to] = res.n_corr
        fail[t_to] = res.fail_reason

    # forward
    for t in range(anchor_idx, total - 1):
        _step(t, t + 1)
    # backward
    for t in range(anchor_idx, 0, -1):
        _step(t, t - 1)

    # fill gaps with SE(3) interpolation
    T_filled = _fill_gaps(T_seq)

    return TrackResult(
        T_seq=[T for T in T_filled],         # type: ignore
        inlier_ratios=inlier_ratios,
        n_corr=n_corr,
        propagated=propagated,
        fail_reasons=fail,
    )
