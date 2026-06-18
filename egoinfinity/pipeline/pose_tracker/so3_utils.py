"""SO(3) utilities: log/exp maps, geodesic SLERP, Riemannian median."""
from __future__ import annotations
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Log / exp maps on SO(3)
# ---------------------------------------------------------------------------

def logmap_so3(R: np.ndarray) -> np.ndarray:
    """Map a rotation matrix to its tangent (axis * angle) vector in R^3.

    Robust to ``angle ≈ 0`` (returns near-zero) and ``angle ≈ pi`` (uses
    eigen-decomposition path).
    """
    R = np.asarray(R, dtype=np.float64)
    cos_t = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_t))
    if angle < 1e-8:
        # near identity — first-order skew part
        return np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) * 0.5
    if abs(angle - np.pi) < 1e-6:
        # angle ≈ pi: skew part vanishes; recover axis from R + I (rank-1 sym)
        M = (R + np.eye(3)) * 0.5
        # find column with max diagonal value
        diag = np.array([M[0, 0], M[1, 1], M[2, 2]])
        i = int(np.argmax(diag))
        col = M[:, i]
        col_n = float(np.linalg.norm(col))
        if col_n < 1e-9:
            return np.zeros(3)
        axis = col / col_n
        return axis * angle
    # standard branch
    sin_t = float(np.sin(angle))
    skew = (R - R.T) * (angle / (2.0 * sin_t))
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def expmap_so3(omega: np.ndarray) -> np.ndarray:
    """Map a tangent vector (axis * angle) back to a rotation matrix."""
    omega = np.asarray(omega, dtype=np.float64).ravel()
    angle = float(np.linalg.norm(omega))
    if angle < 1e-8:
        # first-order
        wx, wy, wz = omega
        K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        return np.eye(3) + K
    axis = omega / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return (np.eye(3)
            + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K))


def slerp_so3(R_a: np.ndarray, R_b: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic SLERP between two rotations: ``alpha=0 → R_a``, ``alpha=1 → R_b``."""
    if alpha <= 0.0:
        return np.asarray(R_a, dtype=np.float64).copy()
    if alpha >= 1.0:
        return np.asarray(R_b, dtype=np.float64).copy()
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    delta = logmap_so3(R_a.T @ R_b)
    return R_a @ expmap_so3(alpha * delta)


# ---------------------------------------------------------------------------
# Riemannian median (Weiszfeld's algorithm on SO(3))
# ---------------------------------------------------------------------------

def so3_riemannian_median(
    R_list: Sequence[np.ndarray],
    n_iter: int = 12,
    eps: float = 1e-6,
) -> np.ndarray:
    """Weiszfeld's iteratively reweighted L1 median on SO(3).

    Robust to outliers; converges to the geometric median (geodesic-distance
    sense) of the input rotations.
    """
    R_arr = [np.asarray(R, dtype=np.float64) for R in R_list if R is not None]
    if not R_arr:
        return np.eye(3, dtype=np.float64)
    if len(R_arr) == 1:
        return R_arr[0].copy()

    # Initialise from the chronological middle (cheap heuristic)
    R_med = R_arr[len(R_arr) // 2].copy()

    for _ in range(n_iter):
        # log-map differences from current median to each sample
        log_diffs = [logmap_so3(R_med.T @ R) for R in R_arr]
        norms = np.array([np.linalg.norm(d) for d in log_diffs], dtype=np.float64)
        # IRWLS weights: 1 / |d|, clamped
        w = 1.0 / np.maximum(norms, eps)
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            break
        delta = sum(wi * di for wi, di in zip(w, log_diffs)) / w_sum
        # step
        R_new = R_med @ expmap_so3(delta)
        if np.linalg.norm(delta) < eps:
            R_med = R_new
            break
        R_med = R_new

    return R_med


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def so3_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance (radians) between two rotations."""
    cos_t = float(np.clip((np.trace(R_a.T @ R_b) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_t))
