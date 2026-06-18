"""MEMFOF optical flow engine — drop-in replacement for cv2.DISOpticalFlow.

MEMFOF (ICCV 2025): memory-efficient multi-frame deep optical flow.
Same `compute_flow(engine, img_a, img_b) -> (H, W, 2)` API as the DIS engine
in `flow_pnp.py`, so callers don't need to change.

Loaded lazily on first use (avoids GPU init cost when only Farneback is wanted).
HuggingFace weights are cached under ``$HF_HOME``; ensure
``HF_HUB_OFFLINE=1`` is *unset* the very first time, then re-set after weights
are cached locally.

Key design points:

- Pair flow via 3-frame trick: feed [a, a, b] so backward (a→a) is trivially
  zero and forward (a→b) is the real flow.  Works because MEMFOF is causal
  on a 3-frame window.
- Auto downsamples very large inputs to keep VRAM bounded; rescales flow back
  to native resolution.
- Single global instance (lazy-loaded singleton) so repeated calls share weights.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("pose_tracker.memfof_flow")

DEFAULT_CKPT = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"


class MEMFOFFlowEngine:
    """Stateful flow engine.  Construct once, reuse `compute_flow` across calls."""

    def __init__(self, device: Optional[str] = None,
                 checkpoint: str = DEFAULT_CKPT,
                 max_side: int = 960):
        """
        Parameters
        ----------
        device : "cuda" | "cpu" | None (auto)
        checkpoint : HuggingFace repo (default: TartanAir-T-TSKH variant — best
                     for real videos per MEMFOF authors)
        max_side : if max(H, W) > this, downsample input to keep VRAM
                   bounded; flow gets bilinearly upscaled afterwards.
                   480x854 frames pass through without rescaling.
        """
        import torch
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.max_side = max_side
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from memfof import MEMFOF
            except ImportError as e:
                raise ImportError(
                    "memfof package not installed. "
                    "Run: pip install git+https://github.com/msu-video-group/memfof"
                ) from e
            log.info(f"loading MEMFOF from {self.checkpoint} ...")
            # Allow HF download once even with HF_HUB_OFFLINE; relies on weights
            # already being in $HF_HOME after first run.
            offline_was = os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                self._model = MEMFOF.from_pretrained(self.checkpoint).eval().to(self.device)
            finally:
                if offline_was is not None:
                    os.environ["HF_HUB_OFFLINE"] = offline_was
            log.info(f"MEMFOF ready on {self.device} "
                     f"({sum(p.numel() for p in self._model.parameters())/1e6:.1f} M params)")

    def compute_flow(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Forward dense flow from img_a to img_b.

        Inputs (H, W, 3) uint8.  Output (H, W, 2) float32 (du, dv) in pixels
        at *input* resolution.
        """
        torch = self._torch
        self._ensure_loaded()

        if img_a.ndim != 3 or img_a.shape[2] != 3:
            raise ValueError(f"img_a must be (H, W, 3), got {img_a.shape}")
        H_in, W_in = img_a.shape[:2]

        # Optional downsample to bound VRAM
        scale = 1.0
        side = max(H_in, W_in)
        if side > self.max_side:
            scale = self.max_side / side
            new_H = int(round(H_in * scale))
            new_W = int(round(W_in * scale))
            import cv2
            a_s = cv2.resize(img_a, (new_W, new_H), interpolation=cv2.INTER_AREA)
            b_s = cv2.resize(img_b, (new_W, new_H), interpolation=cv2.INTER_AREA)
        else:
            a_s, b_s = img_a, img_b

        # Build 3-frame triplet [a, a, b].  Backward (a→a) = 0; forward (a→b) is what we want.
        # Shape: (1, T=3, C=3, H, W) uint8.
        triplet = np.stack([a_s, a_s, b_s], axis=0)              # (3, H, W, 3)
        x = torch.from_numpy(triplet).permute(0, 3, 1, 2)        # (3, 3, H, W)
        x = x.unsqueeze(0).to(self.device)                       # (1, 3, 3, H, W)

        with torch.inference_mode():
            out = self._model(x)
            # out["flow"][-1] shape: (B=1, T_pair=2, C=2, H, W) — last refinement step
            backward, forward = out["flow"][-1].unbind(dim=1)
            flow = forward[0].permute(1, 2, 0).cpu().numpy()     # (H, W, 2)

        if scale != 1.0:
            import cv2
            inv_scale_h = H_in / flow.shape[0]
            inv_scale_w = W_in / flow.shape[1]
            flow = cv2.resize(flow, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= inv_scale_w
            flow[..., 1] *= inv_scale_h

        return flow.astype(np.float32)


# ── Singleton helpers ───────────────────────────────────────────────────────
_global_engine: Optional[MEMFOFFlowEngine] = None
_global_engine_lock = threading.Lock()


def get_global_engine(**kwargs) -> MEMFOFFlowEngine:
    global _global_engine
    if _global_engine is None:
        with _global_engine_lock:
            if _global_engine is None:
                _global_engine = MEMFOFFlowEngine(**kwargs)
    return _global_engine


def make_flow_engine() -> MEMFOFFlowEngine:
    """Drop-in replacement for cv2.DISOpticalFlow_create().

    Returns a `MEMFOFFlowEngine` instance.  Callers should pass it to
    `compute_flow(engine, img_a, img_b)`.
    """
    return get_global_engine()


def compute_flow(engine: MEMFOFFlowEngine, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Compatible signature with the cv2-DIS path: (engine, img_a, img_b) -> (H, W, 2)."""
    if img_a.ndim == 2:
        # gray → RGB
        img_a = np.stack([img_a] * 3, axis=-1)
    if img_b.ndim == 2:
        img_b = np.stack([img_b] * 3, axis=-1)
    return engine.compute_flow(img_a, img_b)
