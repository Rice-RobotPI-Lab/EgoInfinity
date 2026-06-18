"""Hand detection + reconstruction contracts (Phase B).

Default backends:
  hand_detect -> YOLO  (egoinfinity.pipeline.hand_detector.HandDetector)
  hand_recon  -> WiLoR (egoinfinity.pipeline.hand_reconstructor.HandReconstructor)
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IHandDetector(Protocol):
    """Per-frame hand bbox + handedness detection.

    Constructed with ``(model_path=..., ...)`` (defaults baked in).
    ``detect(img_bgr, track=True)`` returns a list of detections, each
    with at least a bbox and a handedness flag. (Phase B-1.)
    """

    def detect(self, img_bgr: np.ndarray, track: bool = True) -> List:
        ...


@runtime_checkable
class IHandReconstructor(Protocol):
    """Per-detection MANO reconstruction.

    Constructed with ``(checkpoint=..., ...)`` (defaults baked in).
    ``reconstruct(img_bgr, ...)`` returns a hand-result object carrying
    at least ``joints_3d_rel`` / ``joints_2d`` / ``vertices`` and the
    camera translation used to lift to metric. The pipeline passes
    ``focal_length=dp_focal`` from Phase A. (Phase B-2.)
    """

    def reconstruct(self, img_bgr: np.ndarray, *args, **kwargs):
        ...
