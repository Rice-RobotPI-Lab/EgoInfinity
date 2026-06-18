"""
Hand detection using YOLO + simple IoU-based frame-to-frame tracking.

Input:  BGR image (numpy)
Output: list of HandDetection (bbox, handedness, track_id)
"""
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from ultralytics import YOLO

from .config import DETECTOR_PATH, DETECTOR_CONF

# Opt-in multi-scale TTA for hard clips (fast motion, top-down view, sleeves
# bleeding into wrist patch, etc). Set EGOINFINITY_HAND_TTA=1 to enable.
# Cost: ~3× detector latency. Default OFF.
_TTA_ENABLED = os.environ.get("EGOINFINITY_HAND_TTA", "0") == "1"
_TTA_SCALES = [640, 960, 1280]
_TTA_NMS_IOU = 0.5


@dataclass
class HandDetection:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    is_right: float           # 1.0 = right, 0.0 = left
    confidence: float
    track_id: int = -1


class HandDetector:
    """YOLO-based hand detector with simple IoU tracking."""

    def __init__(self, model_path: str = DETECTOR_PATH,
                 conf_thresh: float = DETECTOR_CONF,
                 device: str = 'cuda'):
        self.model = YOLO(model_path).to(device)
        self.conf_thresh = conf_thresh
        self._prev_dets: List[HandDetection] = []
        self._next_id = 0

    def detect(self, img_bgr: np.ndarray, track: bool = True) -> List[HandDetection]:
        """Detect hands in a single BGR image.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image
            track: if True, assign track IDs via IoU matching to previous frame

        Returns:
            list of HandDetection
        """
        if _TTA_ENABLED:
            dets = self._detect_tta(img_bgr)
        else:
            results = self.model(img_bgr, conf=self.conf_thresh, verbose=False)[0]

            dets = []
            for det in results:
                bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right = det.boxes.cls.cpu().detach().squeeze().item()
                conf = float(bbox[4]) if len(bbox) > 4 else 1.0
                dets.append(HandDetection(
                    bbox=bbox[:4].astype(np.float32),
                    is_right=float(is_right),
                    confidence=conf,
                ))

        if track and len(dets) > 0:
            self._assign_track_ids(dets)
        else:
            for d in dets:
                d.track_id = self._next_id
                self._next_id += 1

        self._prev_dets = dets
        return dets

    def reset_tracking(self):
        """Reset tracking state (call when starting a new video)."""
        self._prev_dets = []
        self._next_id = 0

    def _detect_tta(self, img_bgr: np.ndarray) -> List[HandDetection]:
        """Multi-scale TTA: run YOLO at several imgsz and merge per-class via NMS."""
        from torchvision.ops import nms as _tv_nms

        pooled = []
        for s in _TTA_SCALES:
            res = self.model(img_bgr, conf=self.conf_thresh, imgsz=s, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            pooled.append(np.concatenate([xyxy, confs[:, None], clss[:, None]],
                                          axis=1).astype(np.float32))
        if not pooled:
            return []
        pooled = np.concatenate(pooled, axis=0)

        kept_rows = []
        for c in np.unique(pooled[:, 5]):
            mask = pooled[:, 5] == c
            sub = pooled[mask]
            boxes = torch.from_numpy(sub[:, :4])
            scores = torch.from_numpy(sub[:, 4])
            idx = _tv_nms(boxes, scores, _TTA_NMS_IOU).numpy()
            kept_rows.append(sub[idx])
        kept = np.concatenate(kept_rows, axis=0)

        return [
            HandDetection(
                bbox=row[:4].astype(np.float32),
                is_right=float(row[5]),
                confidence=float(row[4]),
            )
            for row in kept
        ]

    # ── internals ──────────────────────────────────────────────────

    def _assign_track_ids(self, dets: List[HandDetection]):
        """Greedy IoU matching to previous frame detections."""
        if not self._prev_dets:
            for d in dets:
                d.track_id = self._next_id
                self._next_id += 1
            return

        prev_boxes = np.array([d.bbox for d in self._prev_dets])
        curr_boxes = np.array([d.bbox for d in dets])
        iou_matrix = self._compute_iou_matrix(prev_boxes, curr_boxes)

        used_prev = set()
        used_curr = set()
        # Greedy matching by descending IoU
        pairs = []
        for pi in range(len(self._prev_dets)):
            for ci in range(len(dets)):
                pairs.append((iou_matrix[pi, ci], pi, ci))
        pairs.sort(reverse=True)

        for iou_val, pi, ci in pairs:
            if iou_val < 0.2:
                break
            if pi in used_prev or ci in used_curr:
                continue
            dets[ci].track_id = self._prev_dets[pi].track_id
            used_prev.add(pi)
            used_curr.add(ci)

        # Assign new IDs to unmatched
        for ci in range(len(dets)):
            if ci not in used_curr:
                dets[ci].track_id = self._next_id
                self._next_id += 1

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of [x1,y1,x2,y2] boxes."""
        N, M = len(boxes_a), len(boxes_b)
        iou = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                x1 = max(boxes_a[i, 0], boxes_b[j, 0])
                y1 = max(boxes_a[i, 1], boxes_b[j, 1])
                x2 = min(boxes_a[i, 2], boxes_b[j, 2])
                y2 = min(boxes_a[i, 3], boxes_b[j, 3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (boxes_b[j, 3] - boxes_b[j, 1])
                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0
        return iou
