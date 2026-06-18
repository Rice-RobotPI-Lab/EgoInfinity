"""
Static-camera depth stabilization.

Core insight: with a static camera, background depth is constant across frames.
Any per-frame variation in background depth = estimation noise.

Pipeline:
  1. Build dynamic-region masks from hand bboxes (expanded)
  2. Compute temporal median of background depth → reference template
  3. Per-frame: robust scale alignment of background to template
  4. Apply correction to full depth map (including hand regions)
  5. Optional: temporal smoothing on hand translation

Borrows the iterative-median + robust-loss pattern from HaWoR's est_scale_hybrid,
but adapted: MoGe-2 gives metric depth (no SLAM scale ambiguity), so corrections
are small refinements (scale ≈ 1.0) rather than order-of-magnitude rescaling.
"""
import numpy as np
import cv2
import pathlib
from typing import Dict, List, Optional, Tuple


def build_optical_flow_masks(frames_gray: List[np.ndarray],
                             magnitude_threshold: float = 2.0,
                             temporal_window: int = 3,
                             dilate_px: int = 7,
                             debug_cache_dir: Optional["pathlib.Path"] = None,
                             debug_rgb_frames: Optional[List[np.ndarray]] = None,
                             return_flow_rgbs: bool = False,
                             return_pair_mag: bool = False,
                             ) -> "List[np.ndarray] | tuple":
    """Build dynamic masks from dense optical flow (static camera assumption).

    For each frame, computes the max flow magnitude over a temporal window
    of neighboring frame pairs, then thresholds to get a binary mask.

    Args:
        frames_gray: list of (H, W) uint8 grayscale images
        magnitude_threshold: flow magnitude (px) above which a pixel is dynamic
        temporal_window: number of neighboring frames to aggregate over
        dilate_px: morphological dilation radius for cleanup

    Returns:
        list of (H, W) bool masks, True = dynamic
    """
    n = len(frames_gray)
    if n < 2:
        H, W = frames_gray[0].shape[:2] if n > 0 else (1, 1)
        return [np.zeros((H, W), dtype=bool) for _ in range(n)]

    H, W = frames_gray[0].shape[:2]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    # Precompute pairwise flow magnitudes between consecutive frames (MEMFOF).
    from egoinfinity.pipeline.pose_tracker.memfof_flow import get_global_engine
    flow_engine = get_global_engine()
    pair_mag = []
    flow_rgbs: List[np.ndarray] = []     # HSV-coded BGR uint8 per pair (for viser overlay)
    for i in range(n - 1):
        ga = frames_gray[i]
        gb = frames_gray[i + 1]
        # MEMFOF expects RGB; broadcast gray to 3 channels.
        a_rgb = np.stack([ga, ga, ga], axis=-1) if ga.ndim == 2 else ga
        b_rgb = np.stack([gb, gb, gb], axis=-1) if gb.ndim == 2 else gb
        flow = flow_engine.compute_flow(a_rgb, b_rgb)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Color-encode flow once per pair (cheap, ~1 ms).  Used by viser's
        # camera-frustum overlay layer.  We also dump to disk if debug enabled.
        if return_flow_rgbs or debug_cache_dir is not None:
            try:
                from egoinfinity.pipeline.pose_tracker.debug_viz import flow_to_color
                color_bgr = flow_to_color(flow)        # (H, W, 3) BGR uint8
                if return_flow_rgbs:
                    flow_rgbs.append(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
                if debug_cache_dir is not None:
                    from egoinfinity.pipeline.pose_tracker.debug_viz import dump_flow_field
                    rgb_a_viz = (debug_rgb_frames[i]
                                 if debug_rgb_frames is not None and i < len(debug_rgb_frames)
                                 else None)
                    dump_flow_field(debug_cache_dir, i, flow, rgb_a=rgb_a_viz)
            except Exception:
                if return_flow_rgbs:
                    flow_rgbs.append(np.zeros((H, W, 3), dtype=np.uint8))
        pair_mag.append(mag)

    # Per-frame: aggregate max magnitude over temporal window
    half = temporal_window // 2
    masks = []
    for i in range(n):
        # Gather magnitudes from nearby pairs
        start = max(0, i - half)
        end = min(n - 1, i + half)  # pair index range is [0, n-2]
        if start >= len(pair_mag):
            start = len(pair_mag) - 1
        if end > len(pair_mag):
            end = len(pair_mag)

        if start < end:
            max_mag = np.max(np.stack(pair_mag[start:end], axis=0), axis=0)
        else:
            max_mag = pair_mag[min(start, len(pair_mag) - 1)]

        # Threshold + morphological cleanup
        raw_mask = max_mag > magnitude_threshold
        raw_u8 = raw_mask.astype(np.uint8) * 255
        # Close small holes then dilate
        raw_u8 = cv2.morphologyEx(raw_u8, cv2.MORPH_CLOSE, kernel)
        raw_u8 = cv2.dilate(raw_u8, kernel)
        masks.append(raw_u8 > 0)

    if return_flow_rgbs or return_pair_mag:
        # Pad flow_rgbs to length n (last entry duplicates).
        if flow_rgbs:
            flow_rgbs.append(flow_rgbs[-1])
        elif return_flow_rgbs:
            flow_rgbs = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]
        # Pad pair_mag similarly: the (i, i+1) magnitude maps to frame i;
        # last frame reuses the previous magnitude.
        pair_mag_full: List[np.ndarray] = []
        if return_pair_mag:
            for i in range(n):
                idx = min(i, len(pair_mag) - 1)
                if idx < 0:
                    pair_mag_full.append(np.zeros((H, W), dtype=np.float32))
                else:
                    pair_mag_full.append(pair_mag[idx])
        if return_flow_rgbs and return_pair_mag:
            return masks, flow_rgbs, pair_mag_full
        if return_flow_rgbs:
            return masks, flow_rgbs
        return masks, pair_mag_full
    return masks


def build_dynamic_mask(H: int, W: int,
                       bboxes: List[np.ndarray],
                       margin: float = 0.3) -> np.ndarray:
    """Create binary mask marking dynamic (hand/object) regions.

    Args:
        H, W: image size
        bboxes: list of [x1,y1,x2,y2] arrays for detected hands
        margin: fractional expansion of bbox (0.3 = 30% on each side)

    Returns:
        (H, W) bool mask, True = dynamic
    """
    mask = np.zeros((H, W), dtype=bool)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * margin, bh * margin
        x1m = max(0, int(x1 - mx))
        y1m = max(0, int(y1 - my))
        x2m = min(W, int(x2 + mx))
        y2m = min(H, int(y2 + my))
        mask[y1m:y2m, x1m:x2m] = True
    return mask


def compute_background_template(depth_maps: List[np.ndarray],
                                dynamic_masks: List[np.ndarray],
                                min_valid_ratio: float = 0.3) -> np.ndarray:
    """Compute stable background depth by temporal median.

    For each pixel, take the median over frames where it is:
      - not in a dynamic region
      - has valid depth (> 0.01)

    Args:
        depth_maps: list of (H, W) depth arrays
        dynamic_masks: list of (H, W) bool masks (True = dynamic)
        min_valid_ratio: min fraction of frames a pixel needs to be valid

    Returns:
        (H, W) background depth template (0 where insufficient data)
    """
    n_frames = len(depth_maps)
    H, W = depth_maps[0].shape
    min_valid = max(1, int(n_frames * min_valid_ratio))

    # Stack masked depth: set dynamic/invalid to NaN
    stack = np.full((n_frames, H, W), np.nan, dtype=np.float32)
    for i in range(n_frames):
        d = depth_maps[i].copy()
        d[(dynamic_masks[i]) | (d < 0.01)] = np.nan
        stack[i] = d

    # Temporal median (ignoring NaN)
    with np.errstate(all='ignore'):
        template = np.nanmedian(stack, axis=0)

    # Zero out pixels with too few valid observations
    valid_count = np.sum(~np.isnan(stack), axis=0)
    template[valid_count < min_valid] = 0
    template = np.nan_to_num(template, nan=0.0)

    return template


def estimate_frame_correction(depth_frame: np.ndarray,
                              template: np.ndarray,
                              dynamic_mask: np.ndarray,
                              near: float = 0.1,
                              far: float = 10.0) -> Tuple[float, float]:
    """Estimate scale and offset to align a single frame's background to the template.

    Model: corrected = scale * depth_frame + offset
    On background: scale * D_frame + offset ≈ D_template

    Uses iterative robust median estimation.

    Args:
        depth_frame: (H, W) current frame depth
        template: (H, W) background depth template
        dynamic_mask: (H, W) bool, True = dynamic region
        near, far: depth validity range

    Returns:
        (scale, offset) correction parameters
    """
    # Valid background pixels
    valid = (~dynamic_mask &
             (depth_frame > near) & (depth_frame < far) &
             (template > near) & (template < far))

    if valid.sum() < 100:
        return 1.0, 0.0

    df = depth_frame[valid]
    dt = template[valid]

    # Stage 1: estimate scale via iterative median of ratio
    ratio = dt / df
    scale = float(np.median(ratio))

    for _ in range(5):
        corrected = df * scale
        residual = np.abs(corrected - dt)
        # Reject outliers (> 2 * median residual)
        thresh = 2.0 * np.median(residual)
        inlier = residual < max(thresh, 0.01)
        if inlier.sum() < 50:
            break
        ratio_inlier = dt[inlier] / df[inlier]
        scale = float(np.median(ratio_inlier))

    # Stage 2: estimate offset on inlier set
    corrected = df * scale
    residual = dt - corrected
    offset = float(np.median(residual))

    return scale, offset


def stabilize_depth_sequence(depth_maps: List[np.ndarray],
                             hand_bboxes_per_frame: List[List[np.ndarray]],
                             bbox_margin: float = 0.3,
                             return_template: bool = False,
                             flow_masks: Optional[List[np.ndarray]] = None):
    """Stabilize a sequence of depth maps using background consistency.

    Args:
        depth_maps: list of (H, W) MoGe-2 outputs
        hand_bboxes_per_frame: per-frame list of [x1,y1,x2,y2] bboxes
        bbox_margin: bbox expansion ratio for dynamic mask
        return_template: if True, also return (bg_template, dynamic_masks)
        flow_masks: optional list of (H, W) bool masks from optical flow.
                    If provided, merged (union) with bbox-based masks for
                    more complete foreground coverage.

    Returns:
        list of (H, W) stabilized depth maps
        (optionally) bg_template, dynamic_masks
    """
    if len(depth_maps) == 0:
        if return_template:
            return depth_maps, np.zeros_like(depth_maps[0]) if depth_maps else None, []
        return depth_maps

    H, W = depth_maps[0].shape

    # Step 1: Build dynamic masks (bbox + optional optical flow)
    dynamic_masks = []
    for i, bboxes in enumerate(hand_bboxes_per_frame):
        bbox_mask = build_dynamic_mask(H, W, bboxes, bbox_margin)
        if flow_masks is not None and i < len(flow_masks):
            bbox_mask = bbox_mask | flow_masks[i]
        dynamic_masks.append(bbox_mask)

    # Step 2: Background template
    template = compute_background_template(depth_maps, dynamic_masks)

    # Step 3: Per-frame correction
    corrected = []
    for i in range(len(depth_maps)):
        scale, offset = estimate_frame_correction(
            depth_maps[i], template, dynamic_masks[i])
        d_corr = depth_maps[i] * scale + offset
        d_corr = np.maximum(d_corr, 0)  # no negative depth
        corrected.append(d_corr)

    if return_template:
        return corrected, template, dynamic_masks
    return corrected


def smooth_translations(cam_ts: List[Optional[np.ndarray]],
                        window: int = 5) -> List[Optional[np.ndarray]]:
    """Temporal median smoothing of cam_t sequence.

    Args:
        cam_ts: list of (3,) arrays or None
        window: smoothing window size (odd preferred)

    Returns:
        list of smoothed (3,) arrays
    """
    n = len(cam_ts)
    if n == 0:
        return cam_ts

    smoothed = []
    half = window // 2

    for i in range(n):
        if cam_ts[i] is None:
            smoothed.append(None)
            continue

        # Gather valid neighbors in window
        neighbors = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if cam_ts[j] is not None:
                neighbors.append(cam_ts[j])

        if len(neighbors) == 0:
            smoothed.append(cam_ts[i])
        else:
            # Per-axis median
            stacked = np.stack(neighbors, axis=0)
            smoothed.append(np.median(stacked, axis=0))

    return smoothed


def smooth_joints_savgol(
    track_joints: Dict[int, Dict[int, np.ndarray]],
    window: int = 7,
    polyorder: int = 2,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Savitzky-Golay smoothing on per-track 3D joint sequences.

    Args:
        track_joints: {track_id: {frame_idx: (21,3) joints_3d}}
        window: SavGol window length (odd, >= polyorder+2)
        polyorder: polynomial order

    Returns:
        Same structure with smoothed joint positions.
    """
    from scipy.signal import savgol_filter

    smoothed = {}
    for tid, frame_dict in track_joints.items():
        if len(frame_dict) < window:
            # Too few frames for SavGol — fall back to no smoothing
            smoothed[tid] = dict(frame_dict)
            continue

        frames_sorted = sorted(frame_dict.keys())
        # Stack into (T, 21, 3)
        seq = np.stack([frame_dict[f] for f in frames_sorted], axis=0)
        T, J, D = seq.shape

        # Reshape to (T, 63) for per-column filtering
        flat = seq.reshape(T, -1)
        win = min(window, T)
        if win % 2 == 0:
            win -= 1
        win = max(win, polyorder + 2)
        if win % 2 == 0:
            win += 1

        filtered = savgol_filter(flat, window_length=win, polyorder=polyorder, axis=0)
        filtered = filtered.reshape(T, J, D)

        smoothed[tid] = {f: filtered[i] for i, f in enumerate(frames_sorted)}

    return smoothed


def reject_spikes(
    track_data: Dict[int, Dict[int, np.ndarray]],
    threshold_factor: float = 3.0,
) -> int:
    """Detect and replace single-frame outlier spikes in per-track sequences.

    A frame is a spike if its distance from BOTH the previous and next frame
    exceeds the threshold (median consecutive displacement * threshold_factor).
    This avoids falsely flagging neighbors of a spike.

    Replaced with linear interpolation of neighbors.  Works in-place.

    Args:
        track_data: {track_id: {frame_idx: (J, D) array}}
        threshold_factor: multiplier on median consecutive displacement

    Returns:
        Number of spikes removed.
    """
    n_removed = 0
    for tid, frame_dict in track_data.items():
        frames_sorted = sorted(frame_dict.keys())
        T = len(frames_sorted)
        if T < 3:
            continue

        vals = [frame_dict[f] for f in frames_sorted]

        # Compute consecutive displacements: dist(t, t+1)
        consec = np.array([np.linalg.norm(vals[t + 1] - vals[t])
                           for t in range(T - 1)])
        med = np.median(consec)
        if med < 1e-6:
            med = 1e-6
        thresh = threshold_factor * med

        # A frame is a spike if dist to BOTH neighbors exceeds threshold
        spike_indices = []
        for t in range(1, T - 1):
            d_prev = consec[t - 1]  # dist(t-1, t)
            d_next = consec[t]      # dist(t, t+1)
            if d_prev > thresh and d_next > thresh:
                spike_indices.append(t)

        # Replace spikes with neighbor midpoint
        for t in spike_indices:
            frame_dict[frames_sorted[t]] = 0.5 * (vals[t - 1] + vals[t + 1])
            n_removed += 1

    return n_removed


def refine_dynamic_masks(dynamic_masks: List[np.ndarray],
                         depth_maps: List[np.ndarray],
                         object_masks: Optional[List[Optional[np.ndarray]]] = None,
                         depth_variance_threshold: float = 3.0,
                         dilate_px: int = 5) -> List[np.ndarray]:
    """Refine dynamic masks by incorporating object masks and depth variance.

    Args:
        dynamic_masks: list of (H, W) bool masks from hand bboxes
        depth_maps: list of (H, W) stabilized depth maps
        object_masks: optional list of (H, W) bool masks from SAM2
        depth_variance_threshold: not used currently (reserved)
        dilate_px: dilation radius for expanding masks

    Returns:
        list of refined (H, W) bool masks
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    refined = []
    for i, dm in enumerate(dynamic_masks):
        mask = dm.copy()
        # Merge object mask if available
        if object_masks is not None and i < len(object_masks) and object_masks[i] is not None:
            mask = mask | object_masks[i]
        # Dilate to cover edges
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.dilate(mask_u8, kernel)
        refined.append(mask_u8 > 0)
    return refined


def compute_background_rgb(frames_rgb: List[np.ndarray],
                           dynamic_masks: List[np.ndarray]) -> np.ndarray:
    """Compute a stable background RGB image by temporal median over static pixels.

    Args:
        frames_rgb: list of (H, W, 3) uint8 RGB images
        dynamic_masks: list of (H, W) bool masks (True = dynamic)

    Returns:
        (H, W, 3) uint8 background RGB image
    """
    n = len(frames_rgb)
    H, W, C = frames_rgb[0].shape

    # Use a subset of frames for memory efficiency
    step = max(1, n // 30)
    indices = list(range(0, n, step))

    stack = np.zeros((len(indices), H, W, C), dtype=np.float32)
    for j, i in enumerate(indices):
        img = frames_rgb[i].astype(np.float32)
        img[dynamic_masks[i]] = np.nan
        stack[j] = img

    with np.errstate(all='ignore'):
        bg = np.nanmedian(stack, axis=0)

    bg = np.nan_to_num(bg, nan=0.0)
    return bg.astype(np.uint8)
