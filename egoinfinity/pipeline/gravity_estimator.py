"""
GeoCalib gravity direction estimation wrapper.

Input:  BGR image (numpy)
Output: GravityResult (up direction in camera frame + estimated focal length)

GeoCalib convention: gravity.vec3d is the UP direction in camera frame.
  - Level camera, roll=0, pitch=0 → [0, -1, 0] (OpenCV: -Y = up)
  - This matches the pipeline's WORLD_UP convention directly.

Reference: Veicht et al., "GeoCalib: Single-image Calibration with
Geometric Optimization", ECCV 2024.
"""
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GravityResult:
    """Gravity estimation result."""
    up_direction: np.ndarray   # (3,) unit vector, up direction in camera frame
    roll_deg: float            # roll angle in degrees
    pitch_deg: float           # pitch angle in degrees
    focal_length_px: float     # GeoCalib estimated focal length in pixels


class GravityEstimator:
    """GeoCalib-based gravity direction estimator."""

    def __init__(self, device: str = 'cuda'):
        from geocalib import GeoCalib
        self.device = torch.device(device)
        self.model = GeoCalib(weights='pinhole').to(self.device)
        self.model.eval()

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> GravityResult:
        """Estimate gravity (up direction) from a single BGR image.

        Args:
            image: BGR numpy (H, W, 3), uint8 or float

        Returns:
            GravityResult with up direction, roll/pitch, and focal length
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        result = self.model.calibrate(img_tensor)

        gravity = result['gravity']
        up_vec = gravity.vec3d[0].cpu().numpy()
        roll_deg = float(gravity.roll[0].item()) * 57.2957795
        pitch_deg = float(gravity.pitch[0].item()) * 57.2957795

        camera = result['camera']
        focal_px = float(camera.f[0, 0].item())

        return GravityResult(
            up_direction=up_vec,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            focal_length_px=focal_px,
        )

    def estimate_robust(self,
                        images: List[np.ndarray],
                        sample_indices: Optional[List[int]] = None,
                        ) -> GravityResult:
        """Estimate gravity from multiple frames, taking the median.

        For static cameras, gravity is constant across frames.
        Running on 1-3 representative frames and taking the median
        provides robustness against single-frame anomalies.

        Args:
            images: list of BGR numpy arrays (full frame list)
            sample_indices: which frame indices to use.
                If None, uses first, middle, and last frame.

        Returns:
            GravityResult with median up direction and focal length
        """
        n = len(images)
        if sample_indices is None:
            if n == 1:
                sample_indices = [0]
            elif n == 2:
                sample_indices = [0, n - 1]
            else:
                sample_indices = [0, n // 2, n - 1]

        results = []
        for idx in sample_indices:
            r = self.estimate(images[idx])
            results.append(r)

        # Median of up directions, then re-normalize
        up_vecs = np.array([r.up_direction for r in results])
        median_up = np.median(up_vecs, axis=0)
        median_up = median_up / (np.linalg.norm(median_up) + 1e-8)

        median_roll = float(np.median([r.roll_deg for r in results]))
        median_pitch = float(np.median([r.pitch_deg for r in results]))
        median_focal = float(np.median([r.focal_length_px for r in results]))

        return GravityResult(
            up_direction=median_up,
            roll_deg=median_roll,
            pitch_deg=median_pitch,
            focal_length_px=median_focal,
        )

    def unload(self):
        """Release GPU memory."""
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
