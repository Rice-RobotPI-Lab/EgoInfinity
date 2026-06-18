"""Schmitt-trigger style hysteresis filter for binary state signals.

Used by Phase D-track to convert noisy real-valued signals (contact_soft,
flow / pc motion magnitudes, mask area, density) into clean booleans whose
boundaries don't flicker around the threshold.
"""
from __future__ import annotations
from typing import Sequence, Union

import numpy as np
from scipy.signal import savgol_filter


def hysteresis_filter(
    signal: Union[Sequence[float], np.ndarray],
    low: float,
    high: float,
    *,
    ramp_window: int = 7,
    smooth: bool = True,
) -> np.ndarray:
    """Schmitt-trigger style filter.

    Output goes ``True`` when the signal rises above ``high`` and goes
    ``False`` when it falls below ``low``. The dead-zone ``[low, high]``
    suppresses single-frame noise around a boundary.

    By default the input signal is first SavGol-smoothed with
    ``ramp_window`` (poly=2) so brief spikes don't even reach ``high``.

    Args:
        signal: 1-D array of floats.
        low: lower threshold (state goes False below this).
        high: upper threshold (state goes True above this).
        ramp_window: SavGol window for pre-smoothing.
        smooth: set False to skip pre-smoothing.

    Returns:
        np.ndarray of bool, same length as ``signal``.
    """
    s = np.asarray(signal, dtype=np.float32).ravel()
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if smooth and n >= ramp_window and ramp_window >= 3:
        # SavGol requires odd window
        w = ramp_window if (ramp_window % 2 == 1) else (ramp_window + 1)
        try:
            s = savgol_filter(s, w, 2)
        except Exception:
            pass

    out = np.zeros(n, dtype=bool)
    state = False
    # Bootstrap the initial state from the first few samples to avoid an
    # arbitrary "always start at False" bias when the clip begins inside a
    # high region.
    for i in range(min(3, n)):
        if s[i] > high:
            state = True
            break
        if s[i] < low:
            state = False
            break
    for i in range(n):
        v = s[i]
        if not state and v > high:
            state = True
        elif state and v < low:
            state = False
        out[i] = state
    return out
