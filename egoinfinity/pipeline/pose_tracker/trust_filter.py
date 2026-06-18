"""Stage 3: Triple-signal trust filter for per-frame pose validation.

Three independent signals computed against per-frame T_seq:

    1. Mask IoU       — render mesh point-cloud, morpho-close to mask, IoU vs SAM2
    2. Partial Chamfer — mesh-to-observed-cloud nearest-neighbour distance (best 50%)
    3. PnP inlier ratio — already computed in Stage 2; gap-filled / centroid-fallback
                          frames are flagged 0 to fix the false-positive 1.000 issue

A frame is `trusted` iff iou >= IOU_THR AND chamfer <= CHAMFER_THR AND ir >= IR_THR.

Output is **diagnostic only** — does not modify pose.  Stage 4 consumes the trust
array as a per-frame weight and the obs cloud per-frame as the L_fit target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .utils import project_points
from .anchor import _make_anchor_pointcloud

log = logging.getLogger("pose_tracker.trust_filter")

# Default thresholds (override via evaluate_trust kwargs).
IOU_THR = 0.6
CHAMFER_THR_M = 0.008      # 8 mm
IR_THR = 0.4


# ---------------------------------------------------------------------------
# Signal 1: Mask IoU via point-rasterise + morphological closing
# ---------------------------------------------------------------------------
def render_mesh_mask(mesh_pts: np.ndarray, T: np.ndarray, K: np.ndarray,
                     H: int, W: int, dilate_px: int = 2) -> np.ndarray:
    """Project mesh points and rasterise a binary occupancy mask.

    Used in lieu of triangle rasterisation because SAM3D meshes are gaussian
    splats with no faces.  Morphological closing fills the inter-point gaps
    so the rendered mask is a contiguous blob.
    """
    R, t = T[:3, :3], T[:3, 3]
    x_cam = mesh_pts @ R.T + t
    front = x_cam[:, 2] > 1e-3
    if not front.any():
        return np.zeros((H, W), dtype=bool)
    uv = project_points(K, x_cam[front]).astype(np.int32)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) \
          & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[valid]
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(uv) == 0:
        return mask.astype(bool)
    mask[uv[:, 1], uv[:, 0]] = 1
    k = 2 * dilate_px + 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             np.ones((k, k), dtype=np.uint8))
    return mask.astype(bool)


def mask_iou(pred: np.ndarray, sam2: np.ndarray,
             hand: Optional[np.ndarray]) -> float:
    """IoU between rendered mesh mask and SAM2 mask, both with hand subtracted."""
    obs = sam2 & ~hand if hand is not None else sam2
    pred_eff = pred & ~hand if hand is not None else pred
    inter = int((pred_eff & obs).sum())
    union = int((pred_eff | obs).sum())
    return inter / max(union, 1)


# ---------------------------------------------------------------------------
# Signal 2: Partial Chamfer (mesh → observed cloud)
# ---------------------------------------------------------------------------
def chamfer_partial(mesh_pts: np.ndarray, T: np.ndarray,
                    obs_cloud: np.ndarray, partial: float = 0.5) -> float:
    """Mean of best-`partial` nearest-neighbour distances mesh→obs.

    Partial because the mesh has a back-side that the observation never sees
    (single-view depth).  Picking the closest 50% suppresses this bias.
    Returns m (metric).
    """
    if obs_cloud is None or len(obs_cloud) < 50 or len(mesh_pts) < 50:
        return float('inf')
    R, t = T[:3, :3], T[:3, 3]
    mc = mesh_pts @ R.T + t
    tree = cKDTree(obs_cloud)
    d, _ = tree.query(mc, k=1)
    d_sorted = np.sort(d)
    n_keep = max(int(len(d_sorted) * partial), 1)
    return float(np.mean(d_sorted[:n_keep]))


# ---------------------------------------------------------------------------
# Combined trust evaluation
# ---------------------------------------------------------------------------
@dataclass
class TrustResult:
    trust: List[bool]                              # per-frame
    iou: List[float]                               # per-frame, NaN if no obs
    chamfer_m: List[float]                         # per-frame, inf if no obs
    inlier_ratio: List[float]                      # per-frame, 0 if non-PnP
    obs_clouds: List[Optional[np.ndarray]]         # per-frame, may be None
    trust_rate: float                              # fraction of trusted
    n_trusted: int


def evaluate_trust(
    T_seq: Sequence[np.ndarray],
    mesh_pts: np.ndarray,
    masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
    depth_maps: Sequence[np.ndarray],
    K: np.ndarray,
    inlier_ratios: Sequence[float],
    propagated: Sequence[bool],
    *,
    iou_thr: float = IOU_THR,
    chamfer_thr_m: float = CHAMFER_THR_M,
    ir_thr: float = IR_THR,
) -> TrustResult:
    """Compute trust signals for every frame.

    Args
    ----
    T_seq          : per-frame 4x4 mesh->camera (post-tracking, post-fallback)
    mesh_pts       : (M, 3) canonical-frame mesh, already scale-corrected
    masks          : per-frame SAM2 mask (bool HxW) or None
    hand_masks     : per-frame WiLoR hand mask (bool HxW) or None
    depth_maps     : per-frame metric depth (HxW float32)
    K              : 3x3 intrinsics
    inlier_ratios  : raw per-frame PnP ratios (untrusted/centroid frames may be 1.0)
    propagated     : True if frame's pose was solved by PnP (not gap-fill / centroid)
    """
    n = len(T_seq)
    iou_arr: List[float] = [float('nan')] * n
    cham_arr: List[float] = [float('inf')] * n
    ir_arr: List[float] = [0.0] * n
    obs_arr: List[Optional[np.ndarray]] = [None] * n
    trust_arr: List[bool] = [False] * n

    for i in range(n):
        m_i = masks[i]
        if m_i is None or not m_i.any():
            continue
        H, W = m_i.shape
        h_i = hand_masks[i] if hand_masks is not None else None

        # Signal 3: inlier ratio — only counts if PnP actually solved this frame
        if propagated[i]:
            ir = float(inlier_ratios[i])
        else:
            ir = 0.0
        ir_arr[i] = ir

        # Signal 1: mask IoU
        T_i = np.asarray(T_seq[i], dtype=np.float64)
        rendered = render_mesh_mask(mesh_pts, T_i, K, H, W)
        iou = mask_iou(rendered, m_i, h_i)
        iou_arr[i] = iou

        # Signal 2: partial Chamfer
        obs = _make_anchor_pointcloud(depth_maps[i], m_i, h_i, K)
        obs_arr[i] = obs if len(obs) >= 50 else None
        cham = chamfer_partial(mesh_pts, T_i, obs_arr[i])
        cham_arr[i] = cham

        # Triple AND
        trust_arr[i] = (iou >= iou_thr) and (cham <= chamfer_thr_m) and (ir >= ir_thr)

    n_trusted = int(sum(trust_arr))
    return TrustResult(
        trust=trust_arr,
        iou=iou_arr,
        chamfer_m=cham_arr,
        inlier_ratio=ir_arr,
        obs_clouds=obs_arr,
        trust_rate=n_trusted / max(n, 1),
        n_trusted=n_trusted,
    )
