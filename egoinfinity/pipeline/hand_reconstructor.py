"""
WiLoR-based 3D hand reconstruction.

Input:  BGR image + list of HandDetection
Output: list of HandResult (MANO params, 3D joints, 2D joints, camera-space translation)
"""
import sys
import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

# WiLoR is under third_party/
_THIRD_PARTY_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full

from .config import WILOR_CHECKPOINT, WILOR_CFG, RESCALE_FACTOR
from .hand_detector import HandDetection


@dataclass
class HandResult:
    """Per-hand 3D reconstruction result in camera frame."""
    # ── MANO parameters ───────────────────────────────────────────
    global_orient: np.ndarray   # (1, 3, 3) root rotation matrix
    hand_pose: np.ndarray       # (15, 3, 3) finger joint rotations
    betas: np.ndarray           # (10,) shape parameters
    # ── 3D outputs ────────────────────────────────────────────────
    joints_3d: np.ndarray       # (21, 3) joints in camera frame (with cam_t applied)
    joints_3d_rel: np.ndarray   # (21, 3) root-relative joints (before cam_t)
    vertices: np.ndarray        # (778, 3) mesh vertices in camera frame
    cam_t: np.ndarray           # (3,) camera-space translation [tx, ty, tz]
    # ── 2D outputs ────────────────────────────────────────────────
    joints_2d: np.ndarray       # (21, 2) projected joints in pixel coords
    # ── focal info (for later rescaling) ─────────────────────────
    scaled_focal: float         # focal used to compute cam_t (5000/256*max(H,W) or real)
    # ── metadata ──────────────────────────────────────────────────
    is_right: bool
    bbox: np.ndarray            # original detection bbox [x1,y1,x2,y2]
    confidence: float
    track_id: int


class HandReconstructor:
    """WiLoR single-frame hand reconstruction."""

    def __init__(self, checkpoint: str = WILOR_CHECKPOINT,
                 cfg_path: str = WILOR_CFG,
                 device: str = 'cuda'):
        self.device = torch.device(device)

        self.model, self.model_cfg = load_wilor(
            checkpoint_path=checkpoint, cfg_path=cfg_path)

        self.model = self.model.to(self.device).eval()
        self.model.half()

        self.cfg_focal = self.model_cfg.EXTRA.FOCAL_LENGTH   # default 5000
        self.img_size_cfg = self.model_cfg.MODEL.IMAGE_SIZE  # 256

    @torch.no_grad()
    def reconstruct(self, img_bgr: np.ndarray,
                    detections: List[HandDetection],
                    focal_length: Optional[float] = None,
                    ) -> List[HandResult]:
        """Run WiLoR on detected hands.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image
            detections: hand detections from HandDetector
            focal_length: real camera focal length in pixels.
                          If None, uses WiLoR's default (weak-perspective only).

        Returns:
            list of HandResult, one per detection
        """
        if len(detections) == 0:
            return []

        H, W = img_bgr.shape[:2]
        boxes = np.array([d.bbox for d in detections])
        right = np.array([d.is_right for d in detections])

        # Build WiLoR dataset for this frame
        dataset = ViTDetDataset(
            self.model_cfg, img_bgr, boxes, right,
            rescale_factor=RESCALE_FACTOR)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(detections), shuffle=False, num_workers=0)

        results = []
        for batch in loader:
            batch = recursive_to(batch, self.device)
            batch['img'] = batch['img'].half()

            with torch.amp.autocast('cuda'):
                out = self.model(batch)

            # Fix left/right hand x-flip
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam'].float()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()

            # Camera translation with WiLoR's default focal (for 2D projection)
            scaled_focal_render = (self.cfg_focal / self.img_size_cfg
                                   * img_size.max(dim=1).values)
            cam_t_render = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size,
                scaled_focal_render).detach().cpu().numpy()

            # Camera translation with real focal (for metric 3D)
            if focal_length is not None:
                focal_tensor = torch.tensor(
                    focal_length, device=self.device, dtype=torch.float32)
                cam_t_metric = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size,
                    focal_tensor).detach().cpu().numpy()
            else:
                cam_t_metric = cam_t_render

            for n in range(len(detections)):
                det = detections[n]
                is_right_n = batch['right'][n].item()

                # 3D joints (root-relative from MANO)
                joints_3d = out['pred_keypoints_3d'][n].detach().float().cpu().numpy()
                vertices = out['pred_vertices'][n].detach().float().cpu().numpy()

                # Flip x for left hands
                joints_3d[:, 0] = (2 * is_right_n - 1) * joints_3d[:, 0]
                vertices[:, 0] = (2 * is_right_n - 1) * vertices[:, 0]

                # Keep root-relative copy before adding cam_t
                joints_3d_rel = joints_3d.copy()

                # Apply camera translation for metric 3D position
                cam_t_m = cam_t_metric[n]
                joints_3d_cam = joints_3d + cam_t_m
                vertices_cam = vertices + cam_t_m

                # 2D projection using render focal
                cam_t_r = cam_t_render[n]
                scaled_f = float(scaled_focal_render[n].cpu())
                cx, cy = W / 2.0, H / 2.0
                pts = joints_3d_rel + cam_t_r
                joints_2d = np.zeros((21, 2))
                joints_2d[:, 0] = pts[:, 0] / pts[:, 2] * scaled_f + cx
                joints_2d[:, 1] = pts[:, 1] / pts[:, 2] * scaled_f + cy

                # The focal used for cam_t computation
                effective_focal = float(focal_length) if focal_length is not None else scaled_f

                # MANO parameters
                mano_params = out['pred_mano_params']
                global_orient = mano_params['global_orient'][n].detach().float().cpu().numpy()
                hand_pose = mano_params['hand_pose'][n].detach().float().cpu().numpy()
                betas = mano_params['betas'][n].detach().float().cpu().numpy()

                results.append(HandResult(
                    global_orient=global_orient,
                    hand_pose=hand_pose,
                    betas=betas,
                    joints_3d=joints_3d_cam,
                    joints_3d_rel=joints_3d_rel,
                    vertices=vertices_cam,
                    cam_t=cam_t_m,
                    joints_2d=joints_2d,
                    scaled_focal=effective_focal,
                    is_right=bool(is_right_n > 0.5),
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    track_id=det.track_id,
                ))

        return results
