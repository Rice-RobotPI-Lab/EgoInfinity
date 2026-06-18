"""IDepthEstimator — Phase A metric depth backbone contract.

Default backend: MoGe-2 (``egoinfinity.pipeline.moge2_estimator.MoGe2Estimator``).
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IDepthEstimator(Protocol):
    """Per-frame metric depth + focal estimation.

    A backend is constructed with ``(device='cuda', ...)`` and exposes
    ``estimate(image, known_focal=None)`` returning an object with at
    least ``.depth`` (HxW float32 metric depth, meters) and
    ``.focal_length_px`` (float). The pipeline calls ``estimate`` once
    per frame and collects ``result.depth`` into the in-memory
    ``depth_maps`` list (scripts/exo_pipeline.py Phase A).

    To register a new backend, see egoinfinity/pipeline/backends/__init__.py.
    """

    def estimate(self, image: np.ndarray,
                 known_focal: Optional[float] = None):
        """Return a depth result for one RGB frame.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 RGB.
        known_focal : float, optional
            If provided, the estimator may use it instead of predicting.

        Returns
        -------
        result : object with ``.depth`` (HxW float32 meters) and
            ``.focal_length_px`` (float).
        """
        ...
