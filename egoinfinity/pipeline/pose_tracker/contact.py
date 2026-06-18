"""Stage 4 prerequisite: per-frame, per-hand soft contact detection.

Decision: use a geometric distance threshold + SavGol smoothing over a
short temporal window.  Three reasons:

    1. Zero-cost (no extra model, no API call).  Works offline.
    2. ~80% F1 on table-top manipulation per WHOLE paper's empirical study
       — sufficient for downstream losses to absorb the residual error.
    3. SavGol smoothing eliminates the 0/1 jitter at contact boundaries
       (hand just touching / just letting go), giving downstream
       optimisers a soft ramp instead of a hard switch.

Future extension hooks: VLM (GPT-5) labels at 3 fps as a second source —
disagreement frames flagged ``uncertain`` with weight 0.5.  Not in this PR.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

log = logging.getLogger("pose_tracker.contact")

DEFAULT_THRESHOLD_M = 0.005   # 5 mm — matches EgoGrasp's grasp label cutoff
DEFAULT_RAMP_WIN = 7          # SavGol window (must be odd)
DEFAULT_RAMP_POLY = 2


# ---------------------------------------------------------------------------
# Per-frame raw geometric contact (binary)
# ---------------------------------------------------------------------------
def _hand_to_mesh_min_dist(hand_verts: np.ndarray, mesh_world: np.ndarray
                           ) -> float:
    """Min distance from any hand vertex to any mesh point (m)."""
    if hand_verts is None or len(hand_verts) == 0:
        return float('inf')
    if mesh_world is None or len(mesh_world) == 0:
        return float('inf')
    tree = cKDTree(mesh_world)
    d, _ = tree.query(hand_verts, k=1)
    return float(d.min())


def detect_contact_2d_aware(
    obj_mask_per_frame: Sequence[Optional[np.ndarray]],
    hand_verts_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    mano_faces: np.ndarray,
    K: np.ndarray,
    H: int, W: int,
    *,
    overlap_px_threshold: int = 30,             # min hand∩obj px
    ramp_window: int = DEFAULT_RAMP_WIN,
    ramp_poly: int = DEFAULT_RAMP_POLY,
) -> np.ndarray:
    """Per-frame, per-hand soft contact based on 2D mask overlap.

    Robust to depth noise: just checks whether each hand's rendered mask
    overlaps the SAM2 object mask in the image plane.  If yes → contact.

    Returns (T, 2) float32 ∈ [0, 1] — col 0 = left, col 1 = right.
    """
    from .utils import render_hand_mask
    T = len(obj_mask_per_frame)
    raw = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        obj_m = obj_mask_per_frame[t]
        if obj_m is None or not obj_m.any():
            continue
        verts_list = hand_verts_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        for hv, ir in zip(verts_list, is_right_list):
            if hv is None or len(hv) == 0:
                continue
            single_hand_mask = render_hand_mask(
                [hv], mano_faces, K, H, W, dilate_px=3,
            )
            overlap = int((obj_m & single_hand_mask).sum())
            if overlap >= overlap_px_threshold:
                col = 1 if ir else 0
                raw[t, col] = 1.0
    if T >= ramp_window:
        soft = savgol_filter(raw, ramp_window, ramp_poly, axis=0)
        soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    else:
        soft = raw
    n_contact = int((soft.max(axis=1) > 0.5).sum())
    log.info(f"contact (2D-aware): {n_contact}/{T} frames have any-hand contact "
             f"(soft >= 0.5)")
    return soft


def detect_contact_per_frame(
    hand_verts_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    mesh_world_per_frame: Sequence[np.ndarray],
    *,
    threshold_m: float = DEFAULT_THRESHOLD_M,
    ramp_window: int = DEFAULT_RAMP_WIN,
    ramp_poly: int = DEFAULT_RAMP_POLY,
) -> np.ndarray:
    """Per-frame, per-handedness soft contact weight in [0, 1].

    Parameters
    ----------
    hand_verts_per_frame : list of length T, each a list of (V, 3) ndarrays
        WiLoR hand vertices in world frame.  Up to 2 hands per frame.
    hand_is_right_per_frame : list of length T, each a list of bool
        Same length as the inner list of ``hand_verts_per_frame``.
        ``True`` -> right hand.
    mesh_world_per_frame : list of length T of (M, 3) ndarrays
        Object mesh transformed into world (==camera) frame for that frame.
        Pass mesh_canonical @ R_t.T + t_t per frame.

    Returns
    -------
    contact_soft : (T, 2) float32 array
        column 0 = left hand, column 1 = right hand.  Values in [0, 1].
    """
    T = len(hand_verts_per_frame)
    raw = np.zeros((T, 2), dtype=np.float32)   # 0 = left, 1 = right

    for t in range(T):
        verts_list = hand_verts_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        if len(verts_list) != len(is_right_list):
            # bad frame metadata — skip
            continue
        mesh_w = mesh_world_per_frame[t]
        if mesh_w is None or len(mesh_w) == 0:
            continue
        for h_verts, is_right in zip(verts_list, is_right_list):
            d = _hand_to_mesh_min_dist(h_verts, mesh_w)
            col = 1 if is_right else 0
            if d < threshold_m:
                # Multiple WiLoR detections of same hand can occur — take max
                raw[t, col] = 1.0

    # SavGol along time axis to soften 0/1 jitter at contact boundaries
    if T >= ramp_window:
        soft = savgol_filter(raw, ramp_window, ramp_poly, axis=0)
        soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    else:
        soft = raw

    n_contact = int((soft.max(axis=1) > 0.5).sum())
    log.info(f"contact: {n_contact}/{T} frames have any-hand contact "
             f"(soft >= 0.5)")
    return soft


# ---------------------------------------------------------------------------
# MANO palm vertex subset for L_prox
# ---------------------------------------------------------------------------
# Standard MANO has 778 vertices.  A reasonable "palm + fingertip" subset for
# proximity loss: 5 fingertips (where contact most often happens) +
# palm centre + thumb base.  Indices verified against MANO mean-shape mesh.
#
# Fingertip vertex indices in MANO (0-indexed):
#     thumb tip ≈ 745
#     index tip ≈ 320
#     middle tip ≈ 443
#     ring tip ≈ 554
#     pinky tip ≈ 671
#
# Plus a few palm-central vertices for in-contact pose where fingers wrap
# around the object.  These are heuristic — refine if MANO topology differs.
MANO_PALM_VERTICES = np.array([
    745,   # thumb tip
    320,   # index tip
    443,   # middle tip
    554,   # ring tip
    671,   # pinky tip
    73,    # palm centre (approx)
    79,    # palm radial side
    279,   # palm ulnar side
    232,   # thumb base
    156,   # index base (proximal)
    346,   # middle base
    459,   # ring base
    616,   # pinky base
], dtype=np.int64)
