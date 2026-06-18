"""Flow visualisation with per-object motion overlay.

Renders a grayscale magnitude image (static = black, moving = white) and
overlays each object's mask outline coloured by its `obj_motion_pf` scalar:
  • < 1 px/f  → grey   (static)
  • 1–3 px/f  → yellow (low motion)
  • >= 3 px/f → red    (significant motion)

Self-contained — does NOT depend on ``debug_viz.flow_to_color``; that
function keeps an HSV-coded representation for legacy debug PNG dumps.
This module is the "viser flow overlay" baked into pipeline_result.pkl.gz.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _magnitude_to_grayscale(
    pair_mag: np.ndarray, max_magnitude: float
) -> np.ndarray:
    """(H, W) float32 magnitude → (H, W, 3) BGR uint8 grayscale.

    Static pixels are black, moving pixels increase toward white.  No hue —
    direction is dropped intentionally to avoid the false-colour speckle the
    HSV variant produces over static-camera backgrounds.
    """
    v = np.clip(pair_mag / max(max_magnitude, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def motion_to_color(v: float) -> Tuple[int, int, int]:
    """obj_motion_pf scalar (px/frame) → BGR colour for the contour."""
    if not np.isfinite(v):
        return (128, 128, 128)
    if v < 1.0:
        return (200, 200, 200)
    if v < 3.0:
        return (0, 215, 255)
    return (0, 0, 255)


def render_flow_with_motion_overlay(
    pair_mag: np.ndarray,
    global_max: float,
    object_overlays: Sequence[Tuple[Optional[np.ndarray], float]],
    *,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Grayscale flow magnitude + per-object coloured contours.

    Args
    ----
    pair_mag : (H, W) float32 — flow magnitude for one frame pair.
    global_max : clip-level normalisation constant (use 99th percentile of
        all pair_mag in the clip — passed in by caller, not recomputed
        per-frame, so brightness is comparable across frames).
    object_overlays : sequence of (mask, motion_value).  ``mask`` may be None
        or empty; those entries are skipped.
    contour_thickness : px, default 2.
    """
    img = _magnitude_to_grayscale(pair_mag, global_max)
    for mask, v in object_overlays:
        if mask is None:
            continue
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            m = m.astype(bool)
        if not m.any():
            continue
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(img, contours, -1, motion_to_color(float(v)),
                         contour_thickness)
    return img
