"""Debug visualisation helpers — dump masks and flow as PNGs to disk.

Off by default; enable via `EGOINFINITY_DEBUG_VIZ=1` env var.  Outputs go
under ``cache/<clip>/debug/...`` for offline inspection (any image viewer).

Per-frame outputs:
    debug/masks/obj{oid}/{frame:04d}_orig.png       SAM2 mask, raw
    debug/masks/obj{oid}/{frame:04d}_completed.png  after mask completion
    debug/masks/obj{oid}/{frame:04d}_overlay.png    side-by-side comparison
                                                     overlay on RGB
    debug/flow/{frame:04d}_to_{frame+1:04d}.png     HSV-coded optical flow

Lightweight: ~ a few hundred KB per object per clip.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger("pose_tracker.debug_viz")


def is_debug_enabled() -> bool:
    return os.environ.get("EGOINFINITY_DEBUG_VIZ", "0").lower() in ("1", "true", "yes")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Mask visualisation
# ---------------------------------------------------------------------------
def dump_mask_pair(
    out_dir: Path,
    oid: int,
    frame_idx: int,
    raw_mask: Optional[np.ndarray],
    completed_mask: Optional[np.ndarray],
    rgb: Optional[np.ndarray] = None,
    suffix: str = "",
) -> None:
    """Write 3 PNGs: raw mask, completed mask, side-by-side overlay on RGB.

    Args
    ----
    out_dir: cache root, e.g. cache/<clip>/debug
    oid: object id
    frame_idx: frame number
    raw_mask: (H, W) bool — original SAM2 mask
    completed_mask: (H, W) bool — after completion (may equal raw)
    rgb: (H, W, 3) uint8 RGB frame for overlay (optional)
    suffix: extra tag for filename (e.g. 'static' / 'dynamic')
    """
    if raw_mask is None:
        return
    out_obj = _ensure_dir(out_dir / "masks" / f"obj{oid}")
    base = f"{frame_idx:04d}{('_' + suffix) if suffix else ''}"

    cv2.imwrite(str(out_obj / f"{base}_orig.png"),
                (raw_mask.astype(np.uint8) * 255))
    if completed_mask is not None:
        cv2.imwrite(str(out_obj / f"{base}_completed.png"),
                    (completed_mask.astype(np.uint8) * 255))

    if rgb is not None and completed_mask is not None:
        H, W = raw_mask.shape
        # left: raw mask in red overlay, right: completed in green overlay
        left = rgb.copy()
        right = rgb.copy()
        left[raw_mask] = (left[raw_mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        right[completed_mask] = (
            right[completed_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        sep = np.full((H, 4, 3), 128, dtype=np.uint8)
        sbs = np.concatenate([left, sep, right], axis=1)
        # Annotate
        cv2.putText(sbs, f"obj{oid} f{frame_idx} RAW",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sbs, "COMPLETED",
                    (W + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2 expects BGR
        sbs_bgr = cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_obj / f"{base}_overlay.png"), sbs_bgr)


def dump_completed_masks(
    cache_dir: Path,
    oid: int,
    raw_masks: Sequence[Optional[np.ndarray]],
    completed_masks: Sequence[Optional[np.ndarray]],
    rgb_frames: Optional[Sequence[Optional[np.ndarray]]] = None,
    suffix: str = "",
) -> None:
    """Bulk dump for an entire clip's masks for one object."""
    if not is_debug_enabled():
        return
    debug_dir = _ensure_dir(Path(cache_dir) / "debug")
    T = len(raw_masks)
    for t in range(T):
        if raw_masks[t] is None:
            continue
        rgb = rgb_frames[t] if rgb_frames is not None and t < len(rgb_frames) else None
        dump_mask_pair(
            out_dir=debug_dir,
            oid=oid, frame_idx=t,
            raw_mask=raw_masks[t],
            completed_mask=completed_masks[t] if t < len(completed_masks) else None,
            rgb=rgb,
            suffix=suffix,
        )
    log.info(f"dump_masks: obj{oid} → {debug_dir/'masks'/f'obj{oid}'}")


# ---------------------------------------------------------------------------
# Flow visualisation — magnitude-only grayscale (no direction / hue)
# ---------------------------------------------------------------------------
def flow_to_color(flow: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    """(H, W, 2) flow OR (H, W) magnitude → (H, W, 3) BGR uint8 grayscale.

    Static pixels are black, moving pixels increase toward white.  Direction
    information is dropped intentionally — the rainbow-coded HSV version
    produced strong false-color in the background of static-camera clips
    (any 0.1–0.3 px ghost flow got rendered as fully saturated colour).
    """
    if flow.ndim == 3 and flow.shape[2] == 2:
        mag = np.linalg.norm(flow, axis=-1)
    elif flow.ndim == 2:
        mag = flow
    else:
        raise ValueError(f"expected (H,W,2) flow or (H,W) magnitude, got {flow.shape}")
    if max_magnitude is None:
        max_magnitude = float(np.percentile(mag, 99) + 1e-6)
    v = np.clip(mag / max(max_magnitude, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def dump_flow_field(
    cache_dir: Path,
    frame_idx: int,
    flow: np.ndarray,
    rgb_a: Optional[np.ndarray] = None,
) -> None:
    """One PNG per pair of frames.  Off when EGOINFINITY_DEBUG_VIZ unset."""
    if not is_debug_enabled():
        return
    out_dir = _ensure_dir(Path(cache_dir) / "debug" / "flow")
    color = flow_to_color(flow)
    if rgb_a is not None:
        # Overlay flow over RGB at 50% blend
        rgb_bgr = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2BGR)
        blend = (rgb_bgr.astype(np.float32) * 0.5
                  + color.astype(np.float32) * 0.5).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{frame_idx:04d}_to_{frame_idx+1:04d}.png"),
                    np.concatenate([color, blend], axis=1))
    else:
        cv2.imwrite(str(out_dir / f"{frame_idx:04d}_to_{frame_idx+1:04d}.png"), color)
