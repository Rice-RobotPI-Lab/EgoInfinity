"""Helpers for pose tracker: K matrix, projection, mask rasterization,
mesh loading, Umeyama scale alignment."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------
def build_K(focal: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole intrinsic matrix.  Returns (3, 3) float64."""
    return np.array([[focal, 0, cx],
                     [0, focal, cy],
                     [0, 0,    1]], dtype=np.float64)


def project_points(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame → (N, 2) pixel coords (no distortion)."""
    z = np.maximum(pts_cam[:, 2:3], 1e-6)
    uvw = pts_cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def backproject_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                     step: int = 1) -> np.ndarray:
    """Unproject masked pixels into 3D camera-frame points.

    Returns (N, 3) float32 ndarray.  Drops pixels whose depth is invalid
    (<= 0 or NaN).  Optionally subsamples by ``step``.
    """
    H, W = depth.shape
    if step > 1:
        m = np.zeros_like(mask)
        m[::step, ::step] = mask[::step, ::step]
    else:
        m = mask
    ys, xs = np.where(m)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    valid = (z > 1e-3) & np.isfinite(z)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Hand mask rasterization (avoid pytorch3d dependency)
# ---------------------------------------------------------------------------
def render_hand_mask(verts_3d_list, faces: np.ndarray, K: np.ndarray,
                     H: int, W: int, dilate_px: int = 3) -> np.ndarray:
    """Project per-hand 3D vertex sets into image plane and rasterize a
    binary occupancy mask (one mask covering all hands).

    ``verts_3d_list``: list of (V, 3) float arrays (one per hand in this frame),
    or None / empty list if no hand.
    ``faces``: (F, 3) int32 — MANO topology, shared across hands.

    Uses ``cv2.fillPoly`` over each face triangle.  Optional dilation
    accounts for finger-edge pixels.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not verts_3d_list:
        return mask.astype(bool)
    for verts in verts_3d_list:
        if verts is None or len(verts) == 0:
            continue
        verts = np.asarray(verts, dtype=np.float64)
        # Drop points behind camera
        if (verts[:, 2] <= 1e-3).all():
            continue
        # Project all vertices once
        uv = project_points(K, verts)        # (V, 2)
        # Filter faces whose vertices are all in front of camera
        z = verts[:, 2]
        face_z = z[faces]                    # (F, 3)
        keep = (face_z > 1e-3).all(axis=1)
        if not keep.any():
            continue
        face_uv = uv[faces[keep]]            # (F', 3, 2) int32
        face_uv = face_uv.astype(np.int32)
        cv2.fillPoly(mask, face_uv, color=1)
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1,) * 2, dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Mesh PLY loader (gaussian splat — same parser as exo_pipeline.py)
# ---------------------------------------------------------------------------
def load_mesh_points_from_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a SAM3D gaussian-splat PLY.  Returns (xyz, rgb_uint8)."""
    with open(path, 'rb') as f:
        props = []
        n_vertex = 0
        while True:
            line = f.readline().decode('utf-8', errors='replace').strip()
            if line.startswith('element vertex'):
                n_vertex = int(line.split()[2])
            elif line.startswith('property float'):
                props.append(line.split()[2])
            elif line == 'end_header':
                break
        dtype = np.dtype([(p, '<f4') for p in props])
        data = np.fromfile(f, dtype=dtype, count=n_vertex)
    xyz = np.stack([data['x'], data['y'], data['z']], -1).astype(np.float32)
    C = 0.28209479177387814
    if 'f_dc_0' in data.dtype.names:
        dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], -1)
        rgb = np.clip(dc * C + 0.5, 0, 1)
    else:
        rgb = np.full((n_vertex, 3), 0.7)
    rgb_u8 = (rgb * 255).astype(np.uint8)
    return xyz, rgb_u8


# ---------------------------------------------------------------------------
# Umeyama similarity transform (with scale)
# ---------------------------------------------------------------------------
def umeyama_with_scale(src: np.ndarray, dst: np.ndarray
                       ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find (s, R, t) so that ``s * R @ src + t ≈ dst`` (least squares).

    src, dst: (N, 3) corresponding point pairs.
    Returns (scale, R 3×3, t 3,) all float64.
    Reference: Umeyama 1991.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    sc = src - src_mean
    dc = dst - dst_mean
    # cross-covariance
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = (sc ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = dst_mean - scale * R @ src_mean
    return scale, R, t


def filter_outliers_sor_np(pts: np.ndarray, k: int = 20,
                           std_ratio: float = 2.0) -> np.ndarray:
    """Statistical Outlier Removal — direct numpy version (avoids open3d)."""
    if len(pts) < k + 1:
        return pts
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    d_mean = d[:, 1:].mean(axis=1)
    thresh = d_mean.mean() + std_ratio * d_mean.std()
    return pts[d_mean <= thresh]
