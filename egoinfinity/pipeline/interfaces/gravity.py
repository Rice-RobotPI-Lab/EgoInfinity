"""IGravityEstimator — Phase A-grav world-up direction contract.

Default backend: GeoCalib (``egoinfinity.pipeline.gravity_estimator.GravityEstimator``).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IGravityEstimator(Protocol):
    """Camera gravity / world-up estimation from a single image.

    Constructed with ``(device='cuda')``. ``estimate(image)`` returns an
    object with at least ``.up_direction`` (3-vector, unit), ``.roll_deg``,
    ``.pitch_deg``, and ``.focal_length_px``. The pipeline samples a few
    frames and aggregates (scripts/exo_pipeline.py Phase A-grav).
    """

    def estimate(self, image: np.ndarray):
        """Return a gravity result for one RGB frame.

        Returns
        -------
        result : object with ``.up_direction`` (3,), ``.roll_deg`` (float),
            ``.pitch_deg`` (float), ``.focal_length_px`` (float).
        """
        ...
