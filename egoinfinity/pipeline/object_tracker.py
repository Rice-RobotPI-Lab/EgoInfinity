"""
SAM2-based object segmentation and tracking with 3D bounding box estimation.

Adapted from Fast-FoundationStereoPhysics/ffsd_demos/combined_sam2_stereo.py.

Pipeline:
  1. User provides initial prompt (bbox or point) on first frame
  2. SAM2 tracks object across frames → per-frame binary masks
  3. Mask + depth map → object point cloud → PCA-based 3D OBB
  4. Temporal smoothing on OBB (center, rotation, extent)
"""
import os
import sys
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

# SAM2 path (third_party/sam2/ in repo root)
_SAM2_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'sam2')


@dataclass
class ObjectState:
    """Per-frame object tracking result."""
    mask: np.ndarray                # (H, W) bool mask
    center_3d: np.ndarray           # (3,) smoothed 3D centroid
    extent_3d: np.ndarray           # (3,) smoothed half-widths along principal axes
    rotation_3d: np.ndarray         # (3, 3) smoothed rotation matrix (columns = principal axes)
    corners_3d: np.ndarray          # (8, 3) world-space OBB corners
    n_points: int                   # number of valid 3D points in mask


@dataclass
class OBBSmoother:
    """Temporal smoothing for oriented bounding box, following FFS approach."""
    # EMA state
    smooth_center: Optional[np.ndarray] = None
    smooth_rotation: Optional[np.ndarray] = None
    smooth_extent: Optional[np.ndarray] = None
    prev_axes: Optional[np.ndarray] = None

    # Extent stabilization
    extent_history: deque = field(default_factory=lambda: deque(maxlen=20))
    frame_count: int = 0

    # Parameters
    ema_alpha: float = 0.75           # EMA weight for center & rotation
    extent_alpha_init: float = 0.4
    extent_alpha_min: float = 0.02
    extent_alpha_decay: float = 0.92
    extent_max_change: float = 0.05   # max 5% per-frame change

    def update(self, center: np.ndarray, axes: np.ndarray,
               raw_extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update smoothed OBB with new observation.

        Args:
            center: (3,) raw centroid
            axes: (3, 3) raw rotation matrix (columns = principal axes)
            raw_extent: (3,) raw full-widths

        Returns:
            (smoothed_center, smoothed_rotation, smoothed_extent)
        """
        # Ensure consistent axis direction across frames
        if self.prev_axes is not None:
            for i in range(3):
                if np.dot(axes[:, i], self.prev_axes[:, i]) < 0:
                    axes[:, i] = -axes[:, i]
        self.prev_axes = axes.copy()

        self.frame_count += 1

        if self.smooth_center is None:
            # First frame
            self.smooth_center = center.copy()
            self.smooth_rotation = axes.copy()
            self.smooth_extent = raw_extent.copy()
        else:
            # Center & rotation: standard EMA
            a = self.ema_alpha
            self.smooth_center = a * center + (1 - a) * self.smooth_center
            self.smooth_rotation = a * axes + (1 - a) * self.smooth_rotation

            # Re-orthogonalize via Gram-Schmidt
            R = self.smooth_rotation
            u0 = R[:, 0] / np.linalg.norm(R[:, 0])
            u1 = R[:, 1] - np.dot(R[:, 1], u0) * u0
            u1 = u1 / np.linalg.norm(u1)
            u2 = np.cross(u0, u1)
            self.smooth_rotation = np.column_stack([u0, u1, u2])

            # Extent: decaying EMA + median + clamp
            self.extent_history.append(raw_extent.copy())
            ext_alpha = max(self.extent_alpha_min,
                            self.extent_alpha_init * (self.extent_alpha_decay ** self.frame_count))

            if len(self.extent_history) >= 3:
                median_ext = np.median(np.array(self.extent_history), axis=0)
                candidate = 0.5 * raw_extent + 0.5 * median_ext
            else:
                candidate = raw_extent

            max_delta = self.smooth_extent * self.extent_max_change
            delta = candidate - self.smooth_extent
            clamped = self.smooth_extent + np.clip(delta, -max_delta, max_delta)
            self.smooth_extent = ext_alpha * clamped + (1 - ext_alpha) * self.smooth_extent

        return self.smooth_center.copy(), self.smooth_rotation.copy(), self.smooth_extent.copy()

    def reset(self):
        self.smooth_center = None
        self.smooth_rotation = None
        self.smooth_extent = None
        self.prev_axes = None
        self.extent_history.clear()
        self.frame_count = 0


def smooth_obb_bidirectional(obj_states: List[Optional['ObjectState']],
                             alpha: float = 0.75) -> None:
    """Bidirectional EMA smoothing on OBB params, applied in-place.

    Runs a forward and backward EMA pass on center/rotation/extent,
    then averages to eliminate directional lag.
    """
    # Collect indices that have valid OBB (non-zero corners)
    valid = [(i, s) for i, s in enumerate(obj_states)
             if s is not None and s.n_points > 0 and np.any(s.corners_3d)]
    if len(valid) < 3:
        return

    indices = [v[0] for v in valid]
    centers = np.array([obj_states[i].center_3d for i in indices])
    rotations = np.array([obj_states[i].rotation_3d for i in indices])
    extents = np.array([obj_states[i].extent_3d for i in indices])
    N = len(indices)

    def _ema_pass(arr, alpha, reverse=False):
        out = np.empty_like(arr)
        rng = range(N - 1, -1, -1) if reverse else range(N)
        first = True
        prev = None
        for j in rng:
            if first:
                prev = arr[j].copy()
                first = False
            else:
                prev = alpha * arr[j] + (1 - alpha) * prev
            out[j] = prev
        return out

    def _ema_rot_pass(rots, alpha, reverse=False):
        """EMA on rotation matrices with axis-flip correction + re-orthogonalisation."""
        out = np.empty_like(rots)
        rng = range(N - 1, -1, -1) if reverse else range(N)
        first = True
        prev = None
        for j in rng:
            r = rots[j].copy()
            if first:
                prev = r.copy()
                first = False
            else:
                # Flip axes to stay consistent with prev
                for ax in range(3):
                    if np.dot(r[:, ax], prev[:, ax]) < 0:
                        r[:, ax] = -r[:, ax]
                prev = alpha * r + (1 - alpha) * prev
                # Gram-Schmidt
                u0 = prev[:, 0] / np.linalg.norm(prev[:, 0])
                u1 = prev[:, 1] - np.dot(prev[:, 1], u0) * u0
                u1 = u1 / np.linalg.norm(u1)
                u2 = np.cross(u0, u1)
                prev = np.column_stack([u0, u1, u2])
            out[j] = prev
        return out

    c_fwd = _ema_pass(centers, alpha, reverse=False)
    c_bwd = _ema_pass(centers, alpha, reverse=True)
    e_fwd = _ema_pass(extents, alpha, reverse=False)
    e_bwd = _ema_pass(extents, alpha, reverse=True)
    r_fwd = _ema_rot_pass(rotations, alpha, reverse=False)
    r_bwd = _ema_rot_pass(rotations, alpha, reverse=True)

    for j, idx in enumerate(indices):
        s = obj_states[idx]
        s.center_3d = 0.5 * (c_fwd[j] + c_bwd[j])
        s.extent_3d = 0.5 * (e_fwd[j] + e_bwd[j])

        # Average rotations and re-orthogonalise
        r_avg = 0.5 * (r_fwd[j] + r_bwd[j])
        u0 = r_avg[:, 0] / np.linalg.norm(r_avg[:, 0])
        u1 = r_avg[:, 1] - np.dot(r_avg[:, 1], u0) * u0
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(u0, u1)
        s.rotation_3d = np.column_stack([u0, u1, u2])

        s.corners_3d = obb_corners(s.center_3d, s.rotation_3d, s.extent_3d)


def compute_obb(points: np.ndarray,
                outlier_percentile: float = 90.0
                ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute oriented bounding box via PCA.

    Args:
        points: (N, 3) 3D points
        outlier_percentile: keep points within this percentile distance from centroid

    Returns:
        (center, axes, extent) or None if insufficient points
        - center: (3,) centroid
        - axes: (3, 3) rotation matrix (columns = principal axes, descending eigenvalue)
        - extent: (3,) full widths along each axis
    """
    if len(points) < 10:
        return None

    # Outlier filtering
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    thresh = np.percentile(dists, outlier_percentile)
    filtered = points[dists <= thresh]

    if len(filtered) < 10:
        return None

    # PCA
    center = filtered.mean(axis=0)
    cov = np.cov((filtered - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx]

    # Enforce right-hand coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]

    # Compute extent in local coordinates
    local = (filtered - center) @ axes
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    extent = local_max - local_min

    # Adjust center for asymmetric distribution
    local_center_offset = (local_max + local_min) / 2
    center = center + axes @ local_center_offset

    return center, axes, extent


def obb_corners(center: np.ndarray, rotation: np.ndarray,
                extent: np.ndarray) -> np.ndarray:
    """Compute 8 corners of an oriented bounding box.

    Returns:
        (8, 3) corner positions in world coordinates
    """
    half = extent / 2
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)
    corners_local = signs * half
    return corners_local @ rotation.T + center


@dataclass
class ObjectOBB:
    """Lightweight OBB descriptor produced by ``compute_obb_from_mask``.

    Axes are principal-component directions (descending eigenvalue) in
    camera coordinates. ``R`` is a proper rotation matrix (``det(R)=+1``).
    This gives a body→camera pose; for animating a mesh attached to the
    OBB, transform mesh vertices by ``R @ v_canonical + t`` each frame.
    """
    center_3d: np.ndarray          # (3,) t — camera-space
    R: np.ndarray                  # (3, 3) rotation body→camera
    extent: np.ndarray             # (3,) full widths along body axes
    corners_3d: np.ndarray         # (8, 3) camera-space corners
    n_points: int                  # number of valid points used

    @property
    def pose_t(self) -> np.ndarray:
        return self.center_3d

    @property
    def pose_R(self) -> np.ndarray:
        return self.R


def compute_obb_from_mask(
    mask: np.ndarray,
    depth_map: np.ndarray,
    focal: float, cx: float, cy: float,
    step: int = 2,
    sor_k: int = 20,
    sor_std_ratio: float = 2.0,
    min_points: int = 30,
    outlier_percentile: float = 90.0,
) -> Optional[ObjectOBB]:
    """Compute an OBB (center + rotation + extent) from a 2D mask and a depth map.

    Pipeline:
      1. Unproject masked pixels to 3D via pinhole model (``mask_to_pointcloud``).
      2. Statistical outlier removal.
      3. PCA OBB (``compute_obb``).

    Returns ``None`` if too few valid points or PCA degenerates.
    """
    if mask is None or not mask.any():
        return None
    pts = mask_to_pointcloud(mask, depth_map, focal, cx, cy, step=step)
    if len(pts) < min_points:
        return None
    if len(pts) >= sor_k + 1:
        pts = filter_outliers_sor(pts, k=sor_k, std_ratio=sor_std_ratio)
    if len(pts) < min_points:
        return None
    res = compute_obb(pts, outlier_percentile=outlier_percentile)
    if res is None:
        return None
    center, axes, extent = res
    corners = obb_corners(center, axes, extent)
    return ObjectOBB(
        center_3d=np.asarray(center, dtype=np.float64),
        R=np.asarray(axes, dtype=np.float64),
        extent=np.asarray(extent, dtype=np.float64),
        corners_3d=np.asarray(corners, dtype=np.float64),
        n_points=int(len(pts)),
    )


# OBB edge indices for wireframe rendering
OBB_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
]


def mask_to_pointcloud(mask: np.ndarray, depth: np.ndarray,
                       focal: float, cx: float, cy: float,
                       step: int = 1,
                       min_depth: float = 0.01,
                       max_depth: float = 5.0,
                       erode_pct: float = 0.03) -> np.ndarray:
    """Extract 3D point cloud from masked depth region.

    Args:
        mask: (H, W) bool mask
        depth: (H, W) metric depth
        focal: focal length in pixels
        cx, cy: principal point
        step: downsampling stride
        min_depth, max_depth: valid depth range
        erode_pct: erode mask border by this fraction of mask size (0.0 = off)

    Returns:
        (N, 3) points in camera frame
    """
    import cv2
    work_mask = mask
    if erode_pct > 0 and mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Use MIN of bbox dimensions, not MAX. The kernel applies in both
        # x and y, so sizing by MAX would over-erode the shorter axis on
        # thin/concave shapes (plate rims, knife blade) and obliterate them.
        # MIN sizes the kernel by the LIMITING dimension; chunky shapes still
        # get a meaningful erode, thin shapes lose only a sliver.
        bbox_size = max(min(rmax - rmin, cmax - cmin), 1)
        k = max(int(round(bbox_size * erode_pct)), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        work_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    H, W = depth.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ds = depth[0:H:step, 0:W:step]
    ms = work_mask[0:H:step, 0:W:step]

    valid = ms & (ds > min_depth) & (ds < max_depth)
    xs_v, ys_v, ds_v = xs[valid], ys[valid], ds[valid]

    if len(ds_v) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    X = (xs_v - cx) * ds_v / focal
    Y = (ys_v - cy) * ds_v / focal
    return np.stack([X, Y, ds_v], axis=-1).astype(np.float32)


def filter_outliers_sor(pts: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical Outlier Removal on a point cloud.

    For each point, compute mean distance to its K nearest neighbours.
    Points whose mean distance exceeds (global_mean + std_ratio * global_std)
    are removed.

    Args:
        pts: (N, 3) point cloud
        k: number of neighbours to consider
        std_ratio: multiplier on standard deviation for the threshold

    Returns:
        (M, 3) filtered point cloud (M <= N)
    """
    from scipy.spatial import cKDTree

    if len(pts) <= k:
        return pts

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)  # +1 because closest is self
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self (dist=0)

    mu = mean_dists.mean()
    sigma = mean_dists.std()
    threshold = mu + std_ratio * sigma

    return pts[mean_dists <= threshold]


class ObjectTracker:
    """SAM2-based object tracker with 3D OBB estimation.

    Usage:
        tracker = ObjectTracker(device='cuda')
        # Initialize on first frame with bbox or point
        tracker.init_track(first_frame_rgb, bbox=[x1, y1, x2, y2])
        # Track subsequent frames
        for frame_rgb in frames[1:]:
            state = tracker.track_frame(frame_rgb, depth_map, focal, cx, cy)
    """

    def __init__(self, device: str = 'cuda',
                 sam2_cfg: str = 'sam2.1/sam2.1_hiera_s.yaml',
                 sam2_checkpoint: Optional[str] = None):
        import torch

        self.device = device

        # SAM2 requires bfloat16
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if _SAM2_DIR not in sys.path:
            sys.path.insert(0, _SAM2_DIR)
        from sam2.build_sam import build_sam2_camera_predictor

        if sam2_checkpoint is None:
            from .config import SAM2_CHECKPOINT
            sam2_checkpoint = SAM2_CHECKPOINT

        self.predictor = build_sam2_camera_predictor(sam2_cfg, sam2_checkpoint)
        self.predictor.fill_hole_area = 0

        self.initialized = False
        self.smoothers: dict = {}  # obj_id -> OBBSmoother
        self.obj_ids_tracked: list = []

    def init_track(self, frame_rgb: np.ndarray,
                   bbox: Optional[List[float]] = None,
                   point: Optional[List[float]] = None,
                   obj_id: int = 1) -> np.ndarray:
        """Initialize tracking for one object on first frame.

        Can be called multiple times with different obj_ids before
        calling track_frame() to track multiple objects simultaneously.

        Args:
            frame_rgb: (H, W, 3) RGB uint8 image
            bbox: [x1, y1, x2, y2] bounding box
            point: [x, y] foreground point
            obj_id: object ID for multi-object tracking

        Returns:
            (H, W) bool mask of initial segmentation
        """
        if not self.initialized:
            self.predictor.load_first_frame(frame_rgb)

        if bbox is not None:
            x1, y1, x2, y2 = bbox[:4]
            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            _, _, masks = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, bbox=bbox_arr)
        elif point is not None:
            points = np.array([point], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            _, _, masks = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, points=points, labels=labels)
        else:
            raise ValueError("Must provide either bbox or point")

        mask = (masks[0] > 0.0).squeeze().cpu().numpy().astype(bool)
        self.smoothers[obj_id] = OBBSmoother()
        if obj_id not in self.obj_ids_tracked:
            self.obj_ids_tracked.append(obj_id)
        self.initialized = True
        return mask

    def _make_state(self, mask, depth_map, focal, cx, cy, pc_step, obj_id):
        """Build ObjectState for a single object mask."""
        if not np.any(mask):
            return None

        if depth_map is None or focal is None:
            return ObjectState(
                mask=mask, center_3d=np.zeros(3), extent_3d=np.zeros(3),
                rotation_3d=np.eye(3), corners_3d=np.zeros((8, 3)), n_points=0,
            )

        H, W = depth_map.shape
        if cx is None: cx = W / 2.0
        if cy is None: cy = H / 2.0

        pts = mask_to_pointcloud(mask, depth_map, focal, cx, cy, step=pc_step)

        if len(pts) < 10:
            return ObjectState(
                mask=mask, center_3d=np.zeros(3), extent_3d=np.zeros(3),
                rotation_3d=np.eye(3), corners_3d=np.zeros((8, 3)), n_points=len(pts),
            )

        obb_result = compute_obb(pts)
        if obb_result is None:
            return ObjectState(
                mask=mask, center_3d=pts.mean(axis=0), extent_3d=np.zeros(3),
                rotation_3d=np.eye(3), corners_3d=np.zeros((8, 3)), n_points=len(pts),
            )

        center, axes, extent = obb_result
        smoother = self.smoothers.get(obj_id)
        if smoother:
            sm_center, sm_rotation, sm_extent = smoother.update(center, axes, extent)
        else:
            sm_center, sm_rotation, sm_extent = center, axes, extent
        corners = obb_corners(sm_center, sm_rotation, sm_extent)

        return ObjectState(
            mask=mask, center_3d=sm_center, extent_3d=sm_extent,
            rotation_3d=sm_rotation, corners_3d=corners, n_points=len(pts),
        )

    def track_frame(self, frame_rgb: np.ndarray,
                    depth_map: Optional[np.ndarray] = None,
                    focal: Optional[float] = None,
                    cx: Optional[float] = None,
                    cy: Optional[float] = None,
                    pc_step: int = 2) -> 'dict[int, Optional[ObjectState]]':
        """Track all objects in next frame.

        Returns:
            dict mapping obj_id -> ObjectState (or None if lost).
            For single-object backward compat, also accessible as
            result[obj_id].
        """
        if not self.initialized:
            raise RuntimeError("Call init_track() first")

        out_obj_ids, mask_logits = self.predictor.track(frame_rgb)
        # out_obj_ids is a list of ints, mask_logits is (N, 1, H, W)

        results = {}
        for idx, oid in enumerate(out_obj_ids):
            oid = int(oid)
            mask = (mask_logits[idx] > 0.0).squeeze().cpu().numpy().astype(bool)
            results[oid] = self._make_state(mask, depth_map, focal, cx, cy, pc_step, oid)
        return results

    def reset(self):
        """Reset tracking state."""
        self.predictor.reset_state()
        self.smoothers.clear()
        self.obj_ids_tracked.clear()
        self.initialized = False

    def process_sequence(self, frames_rgb: List[np.ndarray],
                         depth_maps: List[np.ndarray],
                         focal: float, cx: float, cy: float,
                         init_bbox: Optional[List[float]] = None,
                         init_point: Optional[List[float]] = None,
                         pc_step: int = 2) -> List[Optional[ObjectState]]:
        """Process an entire sequence. Convenience wrapper.

        Args:
            frames_rgb: list of (H, W, 3) RGB uint8
            depth_maps: list of (H, W) metric depth
            focal, cx, cy: camera intrinsics
            init_bbox: [x1, y1, x2, y2] for object on first frame
            init_point: [x, y] for object on first frame

        Returns:
            list of ObjectState (one per frame, None if tracking lost)
        """
        if len(frames_rgb) == 0:
            return []

        # First frame: init
        first_mask = self.init_track(frames_rgb[0], bbox=init_bbox, point=init_point)

        # Build first frame's ObjectState with depth
        H, W = depth_maps[0].shape
        pts = mask_to_pointcloud(first_mask, depth_maps[0], focal, cx, cy, step=pc_step)
        obb_result = compute_obb(pts) if len(pts) >= 10 else None

        if obb_result is not None:
            center, axes, extent = obb_result
            sm_c, sm_r, sm_e = self.smoother.update(center, axes, extent)
            corners = obb_corners(sm_c, sm_r, sm_e)
            results = [ObjectState(
                mask=first_mask, center_3d=sm_c, extent_3d=sm_e,
                rotation_3d=sm_r, corners_3d=corners, n_points=len(pts))]
        else:
            results = [ObjectState(
                mask=first_mask, center_3d=pts.mean(axis=0) if len(pts) > 0 else np.zeros(3),
                extent_3d=np.zeros(3), rotation_3d=np.eye(3),
                corners_3d=np.zeros((8, 3)), n_points=len(pts))]

        # Subsequent frames
        for i in range(1, len(frames_rgb)):
            state = self.track_frame(frames_rgb[i], depth_maps[i], focal, cx, cy, pc_step)
            results.append(state)

        return results
