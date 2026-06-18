"""
MoGe-2 metric depth estimation wrapper.

MoGe-2 metric depth estimation wrapper.
Input:  BGR image (numpy)
Output: DepthResult (metric depth map in meters + estimated focal length in pixels)
"""
import math
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from typing import Optional

from .config import MOGE2_MODEL


@dataclass
class DepthResult:
    """Per-frame depth estimation result."""
    depth: np.ndarray          # (H, W) metric depth in meters
    focal_length_px: float     # estimated focal length in pixels (at input resolution)


class MoGe2Estimator:
    """Microsoft MoGe-2 metric depth + focal length estimator."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 device: str = 'cuda',
                 resolution_level: int = 9,
                 precision=None):       # accepted for API compat, not used
        from moge.model.v2 import MoGeModel
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(
            model_name or MOGE2_MODEL
        ).to(self.device)
        self.model.eval()
        self.resolution_level = resolution_level

    @torch.no_grad()
    def estimate(self,
                 image: np.ndarray,
                 known_focal: Optional[float] = None,
                 ) -> DepthResult:
        """Estimate metric depth and focal length for a single BGR image.

        Args:
            image: BGR numpy (H, W, 3)
            known_focal: if provided, use this focal length (px) instead of
                         the model's estimated value

        Returns:
            DepthResult with metric depth map and focal length
        """
        H, W = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(
            rgb / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        kwargs = dict(
            resolution_level=self.resolution_level,
            apply_mask=True,
            force_projection=True,
            use_fp16=True,
        )
        if known_focal is not None:
            fov_x = 2 * math.atan(W / (2 * known_focal))
            kwargs['fov_x'] = math.degrees(fov_x)

        output = self.model.infer(input_tensor, **kwargs)

        depth = output['depth'].cpu().numpy()
        mask = output['mask'].cpu().numpy()
        intrinsics = output['intrinsics'].cpu().numpy()

        # MoGe sets invalid pixels to inf; pipeline expects 0 for invalid
        depth[~mask] = 0.0

        # Normalized intrinsics -> pixel focal length
        fx_px = float(intrinsics[0, 0]) * W
        fy_px = float(intrinsics[1, 1]) * H
        focal = (fx_px + fy_px) / 2

        return DepthResult(depth=depth, focal_length_px=focal)

    def unload(self):
        del self.model
        import gc; gc.collect()
        torch.cuda.empty_cache()
