"""Visual analysis: YOLO hand detection + dense optical flow camera motion."""
import cv2
import numpy as np
import torch

# detector.pt saved with older PyTorch; allow unpickling ultralytics classes
_torch_load_orig = torch.load
torch.load = lambda *a, **kw: _torch_load_orig(*a, **{**kw, "weights_only": False})

from ultralytics import YOLO  # noqa: E402

_model = None


def _get_model(detector_path: str):
    """Lazy-load YOLO hand detector."""
    global _model
    if _model is None:
        _model = YOLO(detector_path)
    return _model


def _is_truncated(xyxy: np.ndarray, h: int, w: int, edge_px: int) -> bool:
    """Check if any hand bbox touches the frame border."""
    if len(xyxy) == 0:
        return False
    return bool(
        (xyxy[:, 0] < edge_px).any()
        or (xyxy[:, 1] < edge_px).any()
        or (xyxy[:, 2] > w - edge_px).any()
        or (xyxy[:, 3] > h - edge_px).any()
    )


def _camera_motion_score_flow(
    gray1: np.ndarray, gray2: np.ndarray, grid: tuple[int, int] = (4, 4)
) -> float:
    """Dense optical flow + grid statistics for background motion detection.

    Splits the frame into a grid, computes median flow magnitude per cell,
    then returns the 25th percentile across cells.  This naturally excludes
    foreground (hand) motion which only affects a few cells.

    Returns background motion in pixels/frame.  Always produces a value
    (no failure mode like ORB).
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    H, W = mag.shape
    gh, gw = grid
    cell_h, cell_w = H // gh, W // gw

    region_medians = []
    for r in range(gh):
        for c in range(gw):
            cell = mag[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            region_medians.append(float(np.median(cell)))

    return float(np.percentile(region_medians, 20))


def analyze_frames(
    frames: list[np.ndarray],
    detector_path: str,
    hand_conf: float = 0.3,
    trunc_edge_px: int = 5,
    motion_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Run hand detection + camera motion analysis on BGR frames.

    Args:
        frames: BGR frames (uniformly sampled) for hand detection + viz.
        motion_pairs: list of (gray1, gray2) closely-spaced frame pairs for
            background motion detection.  If None, falls back to computing
            flow on consecutive ``frames`` (less accurate due to large gap).

    Returns dict with:
        hand_ratio, avg_hand_size, trunc_ratio, bg_flow  (aggregate metrics)
        per_frame: list of {xyxy, confs, truncated} per frame  (for visualization)
    """
    if not frames:
        return {"hand_ratio": 0.0, "avg_hand_size": 0.0, "trunc_ratio": 0.0,
                "bg_flow": 999.0, "per_frame": []}

    # --- Batch hand detection ---
    model = _get_model(detector_path)
    results = model(frames, conf=hand_conf, verbose=False)

    hand_count = 0
    trunc_count = 0
    all_hand_sizes: list[float] = []
    per_frame: list[dict] = []

    for r in results:
        h, w = r.orig_shape
        frame_area = h * w
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        truncated = _is_truncated(xyxy, h, w, trunc_edge_px) if len(xyxy) > 0 else False

        per_frame.append({"xyxy": xyxy, "confs": confs, "truncated": truncated})

        if len(xyxy) > 0:
            hand_count += 1
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            all_hand_sizes.extend((areas / frame_area).tolist())
            if truncated:
                trunc_count += 1

    # --- Camera motion via dense optical flow ---
    if motion_pairs:
        # Dedicated closely-spaced pairs (preferred)
        scores = [_camera_motion_score_flow(g1, g2) for g1, g2 in motion_pairs]
    else:
        # Fallback: use consecutive uniformly-sampled frames
        scores = []
        prev_gray = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                scores.append(_camera_motion_score_flow(prev_gray, gray))
            prev_gray = gray

    bg_flow = float(np.median(scores)) if scores else 0.0

    n = len(frames)
    return {
        "hand_ratio": hand_count / n,
        "avg_hand_size": float(np.mean(all_hand_sizes)) if all_hand_sizes else 0.0,
        "trunc_ratio": trunc_count / n,
        "bg_flow": bg_flow,
        "per_frame": per_frame,
    }
