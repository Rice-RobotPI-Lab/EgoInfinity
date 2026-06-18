"""Stage 5: non-causal SavGol smoothing of T_seq output.

Applied AFTER Stage 4 LBFGS optimisation.  Kills residual per-frame jitter
that L_fit tracking induces from noisy depth observations.  SavGol is a
local-polynomial fitter — it preserves smooth motion (e.g. hand transporting
a knife) while suppressing high-frequency depth noise.

Translation and rotation handled separately:
    - translation: SavGol on (T, 3) directly
    - rotation: SavGol on (T, 4) quaternion, with antipodal sign-fix first
                to ensure continuity, then re-normalise after smoothing
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter

from .fill_optimize import _mat_to_quat


DEFAULT_WIN = 11        # frames — must be odd, > poly
DEFAULT_POLY = 3


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(4,) [w, x, y, z] -> (3, 3); not assumed normalised."""
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _unwrap_quat_signs(qs: np.ndarray) -> np.ndarray:
    """Flip sign of q[i] if q[i] · q[i-1] < 0, so the time series is
    continuous on the 4-sphere (q and -q represent the same rotation)."""
    out = qs.copy()
    for i in range(1, len(out)):
        if (out[i] * out[i - 1]).sum() < 0:
            out[i] = -out[i]
    return out


def smooth_se3_savgol(
    T_seq: Sequence[np.ndarray],
    win: int = DEFAULT_WIN,
    poly: int = DEFAULT_POLY,
) -> np.ndarray:
    """Non-causal SavGol on T_seq.  Returns (N, 4, 4) float64.

    If ``len(T_seq) <= win``, returns input unchanged.
    """
    T_arr = np.asarray(T_seq, dtype=np.float64)
    n = len(T_arr)
    if n <= win:
        return T_arr

    # Translations
    ts = T_arr[:, :3, 3]                                    # (N, 3)
    ts_smooth = savgol_filter(ts, win, poly, axis=0)

    # Rotations as quaternions
    qs = np.stack([_mat_to_quat(T_arr[i, :3, :3]) for i in range(n)])
    qs = _unwrap_quat_signs(qs)
    qs_smooth = savgol_filter(qs, win, poly, axis=0)
    qs_smooth /= np.linalg.norm(qs_smooth, axis=1, keepdims=True).clip(min=1e-8)

    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1
    out[:, :3, 3] = ts_smooth
    for i in range(n):
        out[i, :3, :3] = _quat_to_mat(qs_smooth[i])
    return out
