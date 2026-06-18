"""Per-frame oriented bounding box (OBB) for the observed point cloud.

PCA on the SAM2-mask backprojected point cloud gives the object's
principal axes in the world frame.  When the cloud is dense and
non-degenerate the principal axes track the true object orientation
across frames — useful as a rotation cue for boxes / books / phones
during grasp.

Returned per-frame structure:
    R       : (3, 3)  principal axes as columns (right-handed)
    center  : (3,)    cloud centroid
    extents : (3,)    full width along each principal axis (max - min in that axis)
    eigvals : (3,)    PCA eigenvalues sorted descending — diagnostic of degeneracy

Sign-correction across frames keeps eigenvectors aligned with the
previous trustworthy frame so the rotation sequence is continuous.
PCA is otherwise sign-arbitrary and would flip ±180° between frames.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("pose_tracker.obs_obb")


# ---------------------------------------------------------------------------
# Trustworthiness gate
# ---------------------------------------------------------------------------
DEFAULT_MIN_POINTS = 200
DEFAULT_MAX_AR_12 = 0.92      # eigenvalue ratio λ2/λ1 — too high = degenerate
DEFAULT_MIN_AR_31 = 0.015     # eigenvalue ratio λ3/λ1 — too low = flat sliver


def is_trustworthy(eigvals: np.ndarray, n_pts: int) -> bool:
    """Reject obs clouds with too few points or degenerate principal axes."""
    if n_pts < DEFAULT_MIN_POINTS:
        return False
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvals[0] < 1e-9:
        return False
    if eigvals[1] / eigvals[0] > DEFAULT_MAX_AR_12:
        return False
    if eigvals[2] / eigvals[0] < DEFAULT_MIN_AR_31:
        return False
    return True


# ---------------------------------------------------------------------------
# Single-frame OBB
# ---------------------------------------------------------------------------
def compute_obb(obs: np.ndarray) -> Optional[dict]:
    """Compute PCA-based OBB for one cloud.

    Returns dict with keys (R, center, extents, eigvals, trustworthy, n_pts)
    or None when the cloud is too small to do PCA at all.
    """
    if obs is None or len(obs) < 4:
        return None
    obs = np.asarray(obs, dtype=np.float64)
    center = obs.mean(axis=0)
    centered = obs - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip to descending so col 0 = largest
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Force right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    proj = centered @ eigvecs            # coords in PCA frame
    extents = proj.max(axis=0) - proj.min(axis=0)
    return {
        "R": eigvecs.astype(np.float64),
        "center": center.astype(np.float64),
        "extents": extents.astype(np.float64),
        "eigvals": eigvals.astype(np.float64),
        "trustworthy": bool(is_trustworthy(eigvals, len(obs))),
        "n_pts": int(len(obs)),
    }


# ---------------------------------------------------------------------------
# Sign-correct PCA columns against a reference (anchor or previous frame).
# ---------------------------------------------------------------------------
def sign_correct(R_new: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Flip column signs of R_new so each axis points the same way as R_ref."""
    R = R_new.copy()
    for i in range(3):
        if float(R[:, i] @ R_ref[:, i]) < 0:
            R[:, i] *= -1
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


# ---------------------------------------------------------------------------
# Per-frame batch with cross-frame sign continuity.
# ---------------------------------------------------------------------------
def compute_obs_obb_per_frame(
    obs_clouds: Sequence[Optional[np.ndarray]],
    anchor_idx: int = 0,
) -> List[Optional[dict]]:
    """Compute OBB for every frame; sign-correct against anchor first, then
    run forward + backward to also align frames where anchor is far away."""
    T = len(obs_clouds)
    out: List[Optional[dict]] = [None] * T
    for t in range(T):
        out[t] = compute_obb(obs_clouds[t])

    # Reference axes from anchor (or first trustworthy frame near anchor)
    ref = None
    if 0 <= anchor_idx < T and out[anchor_idx] is not None:
        ref = out[anchor_idx]["R"]
    if ref is None:
        for t in range(T):
            if out[t] is not None and out[t]["trustworthy"]:
                ref = out[t]["R"]
                break
    if ref is None:
        return out  # nothing to align to

    # Forward sweep
    last_R = ref.copy()
    for t in range(T):
        if out[t] is None:
            continue
        out[t]["R"] = sign_correct(out[t]["R"], last_R)
        if out[t]["trustworthy"]:
            last_R = out[t]["R"]

    # Backward sweep — align frames before the first trustworthy frame
    last_R = ref.copy()
    for t in range(T - 1, -1, -1):
        if out[t] is None:
            continue
        out[t]["R"] = sign_correct(out[t]["R"], last_R)
        if out[t]["trustworthy"]:
            last_R = out[t]["R"]

    n_trust = sum(1 for o in out if o is not None and o["trustworthy"])
    log.info(f"obs_obb: {n_trust}/{T} trustworthy frames")
    return out


# ---------------------------------------------------------------------------
# Helpers for visualization — 8 corners + 3 axis lines.
# ---------------------------------------------------------------------------
def obb_corners(obb: dict) -> np.ndarray:
    """Return (8, 3) world coords of the 8 OBB corners."""
    R = obb["R"]
    c = obb["center"]
    e = obb["extents"] * 0.5
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ], dtype=np.float64)
    local = signs * e
    return (local @ R.T) + c


def obb_axes(obb: dict, length_scale: float = 0.5) -> np.ndarray:
    """Return (3, 2, 3) — three line segments for principal axes from center.

    length_scale: fraction of half-extent to draw axes (0.5 = midway out).
    """
    R = obb["R"]
    c = obb["center"]
    half = obb["extents"] * 0.5 * length_scale
    out = np.zeros((3, 2, 3), dtype=np.float64)
    for i in range(3):
        out[i, 0] = c
        out[i, 1] = c + R[:, i] * half[i]
    return out
