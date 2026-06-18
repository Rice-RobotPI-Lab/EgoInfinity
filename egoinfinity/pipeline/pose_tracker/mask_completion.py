"""Static-object SAM2 mask completion.

Problem: when a hand passes over a static object (e.g. plate on the table),
SAM2's per-frame visible mask shrinks → the mask centroid jumps to the
non-occluded portion → looks like the object "moves" → grasp detection
falsely triggers HELD.

Solution: for objects that are static (high IoU between non-hand-occluded
frames' masks), replace occluded-frame masks with the median mask computed
from clean frames.  This keeps mask centroid stable through occlusion.

CRITICAL: only use the completed mask for **2D operations** (contact
overlap, mask centroid trajectory).  For 3D back-projection (depth point
cloud), use the original SAM2 mask minus hand mask — the completed pixels
correspond to the *hand's* depth, not the object's.

Static-camera assumption: this works because our setup is a fixed exo
camera; static objects have approximately constant pixel-coordinate masks
across frames.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("pose_tracker.mask_completion")

DEFAULT_OCCLUSION_THRESHOLD = 0.05      # mask < 5% covered by hand → "clean"
DEFAULT_STATIC_IOU_THRESHOLD = 0.85     # avg clean-frame IoU > this → static
DEFAULT_MIN_CLEAN_FRAMES = 5
DEFAULT_MORPH_CLOSE_PX = 3              # smooth out median artifacts


def complete_static_masks(
    sam2_masks_per_frame: Sequence[Optional[np.ndarray]],
    hand_mask_per_frame: Sequence[Optional[np.ndarray]],
    *,
    occlusion_threshold: float = DEFAULT_OCCLUSION_THRESHOLD,
    static_iou_threshold: float = DEFAULT_STATIC_IOU_THRESHOLD,
    min_clean_frames: int = DEFAULT_MIN_CLEAN_FRAMES,
    morph_close_px: int = DEFAULT_MORPH_CLOSE_PX,
) -> Tuple[List[Optional[np.ndarray]], bool, dict]:
    """Complete (heal) per-frame masks for one object across the clip.

    Returns
    -------
    completed_masks : list of (H, W) bool — same length as input.  For
        non-static objects, returns the input unchanged.  For static
        objects, occluded frames have median mask substituted.
    is_static : bool
    diag : dict with keys 'avg_iou', 'n_clean', 'reason' (when applicable)
    """
    T = len(sam2_masks_per_frame)
    if T == 0:
        return list(sam2_masks_per_frame), False, {"reason": "empty"}

    # Find a representative shape from the first non-None mask
    H, W = None, None
    for m in sam2_masks_per_frame:
        if m is not None:
            H, W = m.shape
            break
    if H is None:
        return list(sam2_masks_per_frame), False, {"reason": "all None"}

    # Step 1: identify clean frames (hand barely overlaps object mask)
    clean_t: List[int] = []
    for t in range(T):
        m = sam2_masks_per_frame[t]
        if m is None or not m.any():
            continue
        h = hand_mask_per_frame[t] if t < len(hand_mask_per_frame) else None
        if h is None:
            clean_t.append(t)
            continue
        overlap = float((m & h).sum()) / float(max(m.sum(), 1))
        if overlap < occlusion_threshold:
            clean_t.append(t)

    if len(clean_t) < min_clean_frames:
        return list(sam2_masks_per_frame), False, {
            "reason": "too_few_clean", "n_clean": len(clean_t)}

    # Step 2: pixel-wise median mask over clean frames
    stack = np.stack([sam2_masks_per_frame[t].astype(np.float32)
                      for t in clean_t], axis=0)        # (n_clean, H, W)
    median_mask = (stack.mean(axis=0) > 0.5)            # majority vote

    # Step 3: average clean-frame IoU vs median → is_static?
    ious = []
    for t in clean_t:
        m = sam2_masks_per_frame[t]
        inter = float((m & median_mask).sum())
        union = float((m | median_mask).sum())
        ious.append(inter / max(union, 1.0))
    avg_iou = float(np.mean(ious))
    is_static = avg_iou > static_iou_threshold

    diag = {"avg_iou": avg_iou, "n_clean": len(clean_t), "n_total": T}

    if not is_static:
        return list(sam2_masks_per_frame), False, diag

    # Step 4: morphological close to clean median artefacts
    if morph_close_px > 0 and median_mask.any():
        k = 2 * morph_close_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        median_mask = cv2.morphologyEx(
            median_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # Step 5: build output — clean frames keep original, others get median
    clean_set = set(clean_t)
    completed: List[Optional[np.ndarray]] = []
    for t in range(T):
        if t in clean_set:
            completed.append(sam2_masks_per_frame[t])
        else:
            completed.append(median_mask.copy())

    log.info(
        f"mask_completion: STATIC, avg_iou={avg_iou:.2f}, "
        f"healed {T - len(clean_t)}/{T} occluded frames")
    return completed, True, diag
