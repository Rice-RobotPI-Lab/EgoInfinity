"""
Shared visualization and GPU utilities for pipeline scripts.
"""
import gc
import numpy as np
import cv2
import yaml

import os, sys
REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from egoinfinity.pipeline.config import HAND_EDGES

FINGER_COLORS_BGR = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
FRUSTUM_SCALE = 0.15
OBB_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def free_gpu():
    gc.collect()
    import torch  # lazy: keeps `--load-cache` viser launches off the 11s torch import path
    torch.cuda.empty_cache()


def bone_colors_bgr():
    return [FINGER_COLORS_BGR[fi] for fi in range(5) for _ in range(4)]


def make_line_segments_3d(kpts_3d):
    return np.array([[kpts_3d[s], kpts_3d[e]] for s, e in HAND_EDGES])


def draw_skeleton_2d(img, kpts_2d, bone_color_list, joint_color, thickness=2, radius=3):
    for ei, (s, e) in enumerate(HAND_EDGES):
        cv2.line(img, tuple(kpts_2d[s].astype(int)), tuple(kpts_2d[e].astype(int)), bone_color_list[ei], thickness)
    for j in range(kpts_2d.shape[0]):
        cv2.circle(img, tuple(kpts_2d[j].astype(int)), radius, joint_color, -1)


def depth_to_colormap(depth):
    d = depth.copy()
    valid = d[d > 0]
    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(valid, [2, 98])
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def depth_to_pointcloud(depth, focal, cx, cy, step=4, max_depth=5.0, img_rgb=None):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    ds = depth[0:H:step, 0:W:step]
    mask = (ds > 0.01) & (ds < max_depth)
    xs, ys, ds = xs[mask], ys[mask], ds[mask]
    if len(ds) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    X = (xs - cx) * ds / focal
    Y = (ys - cy) * ds / focal
    points = np.stack([X, Y, ds], axis=-1).astype(np.float32)
    if img_rgb is not None:
        colors_rgb = img_rgb[ys, xs].copy()
    else:
        valid_d = depth[depth > 0]
        vmin, vmax = (np.percentile(valid_d, [2, 98]) if len(valid_d) > 0 else (0, 1))
        norm_u8 = np.clip((ds - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        norm_u8 = (norm_u8 * 255).astype(np.uint8)
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_INFERNO)[0]
        colors_rgb = lut[norm_u8][:, ::-1].copy()
    return points, colors_rgb


def get_focal_from_calib(calibration_dir, camera_serial, default=378.0):
    """Read focal length from HO-Cap calibration file."""
    calib_file = os.path.join(calibration_dir, 'intrinsics', f'{camera_serial}.yaml')
    if os.path.exists(calib_file):
        with open(calib_file) as f:
            return yaml.safe_load(f)['color']['fx']
    return default


# ── Compact-pkl encode / decode helpers ──────────────────────────────────
# As of pkl format v2 we store depth maps as uint16-mm PNG bytes (~9x smaller
# than raw float32) and drop redundant per-frame `pts` from sam3_obj_data
# (recomputed on load from mask_packed + depth_map).  Format detection is
# automatic so legacy v1 caches still load unchanged.

PKL_FORMAT_VERSION = 2


def encode_depth_png(depth_f32, max_m=65.535):
    """float32 metric depth (H, W) -> PNG-encoded uint16 millimeter bytes.
    Returns bytes; quantization rounds to 1 mm. Pixels >= max_m clip."""
    if depth_f32 is None:
        return None
    d_mm = np.clip(depth_f32 * 1000.0, 0, int(max_m * 1000)).astype(np.uint16)
    ok, buf = cv2.imencode('.png', d_mm)
    return bytes(buf) if ok else None


def decode_depth_png(png_bytes):
    """PNG bytes (uint16 mm) -> float32 (H, W) metric depth in meters."""
    if png_bytes is None:
        return None
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    d_mm = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if d_mm is None:
        return None
    return (d_mm.astype(np.float32) / 1000.0)


def _unpack_mask(mask_packed, mask_shape):
    """packbits-compressed bool mask -> (H, W) bool ndarray."""
    if mask_packed is None or mask_shape is None:
        return None
    H, W = int(mask_shape[0]), int(mask_shape[1])
    flat = np.unpackbits(mask_packed)[:H * W]
    return flat.reshape(H, W).astype(bool)


def _recompute_sam3_pts(mask, depth, dp_focal, cx, cy):
    """Mirror exo_pipeline.py's per-frame Phase E pts construction:
        mask_to_pointcloud(mask, depth, dp_focal, cx, cy, step=2)
        + filter_outliers_sor(pts, k=20, std_ratio=2.0) if len > 20
    Used to re-create the dropped `pts` field on cache load."""
    from egoinfinity.pipeline.object_tracker import (
        mask_to_pointcloud, filter_outliers_sor)
    pts = mask_to_pointcloud(mask, depth, dp_focal, cx, cy, step=2)
    if len(pts) > 20:
        pts = filter_outliers_sor(pts, k=20, std_ratio=2.0)
    return pts


def rehydrate_pkl(data, recompute_pts=True):
    """Bring a freshly pickle.load'd pipeline_result dict back to its in-memory
    canonical shape:

      - 'depth_png' bytes -> 'depth_map' float32 metric (H, W) per frame
      - top-level 'bg_template_png' -> 'bg_template' float32
      - sam3_obj_data[oid]['pts'] recomputed from mask_packed + depth_map +
        dp_focal (when missing and recompute_pts=True)

    Legacy v1 caches (already canonical) pass through with a no-op.
    Mutates `data` in place and returns it.
    """
    fd_list = data.get('frame_data') or []
    dp_focal = float(data.get('dp_focal') or 0.0)
    cx = float(data.get('cx') or 0.0)
    cy = float(data.get('cy') or 0.0)

    for fd in fd_list:
        # depth_png -> depth_map
        if 'depth_png' in fd and fd.get('depth_map') is None:
            fd['depth_map'] = decode_depth_png(fd.pop('depth_png'))
        # If still no depth_map (very old corrupt cache), leave alone
        if recompute_pts and dp_focal > 0 and fd.get('depth_map') is not None:
            depth = fd['depth_map']
            for oid, entry in (fd.get('sam3_obj_data') or {}).items():
                if not isinstance(entry, dict):
                    continue
                if entry.get('pts') is not None:
                    continue
                mask = _unpack_mask(entry.get('mask_packed'), entry.get('mask_shape'))
                if mask is None or not mask.any():
                    entry['pts'] = np.zeros((0, 3), dtype=np.float32)
                    continue
                try:
                    entry['pts'] = _recompute_sam3_pts(mask, depth, dp_focal, cx, cy)
                except Exception:
                    entry['pts'] = np.zeros((0, 3), dtype=np.float32)

    # Top-level bg_template
    if 'bg_template_png' in data and data.get('bg_template') is None:
        data['bg_template'] = decode_depth_png(data.pop('bg_template_png'))

    return data
