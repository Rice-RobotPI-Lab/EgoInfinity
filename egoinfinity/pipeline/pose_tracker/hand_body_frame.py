"""Explicit hand body coordinate frame (option B in flow3r-attach plan).

Builds a full SE(3) pose at the wrist for every frame, defined by:

    origin = wrist (joint 0)
    x      = (joint_5 (index MCP) - wrist).normalize()
    y      = palm normal (out of palm, dorsal-to-palmar direction)
    z      = x × y                                  (closes the right-handed basis)

Why this frame: gives a stable, articulation-invariant 6D pose at the wrist
that can be used as the parent frame for grasped-object rigid binding.
Using only palm landmarks (wrist + index MCP + pinky MCP) keeps the frame
unaffected by finger flexion.

Construction details:
    y is computed as ((joint_17 - wrist) × x) then re-orthogonalized so that
    the resulting basis is orthonormal even when index MCP and pinky MCP
    are nearly colinear with the wrist (degenerate handful of frames are
    flagged as invalid and returned with NaN).

For LEFT hands the basis chirality is preserved (still right-handed in the
returned matrix), which means the object-relative pose computed against
this frame will differ slightly between left and right hands. Per-segment
canonical computation handles this naturally (it averages within one
hand's frames). single_canonical_grasp_lock's auto-detect via the y_obs
sign branch picks the correct mesh-end-to-palm alignment for either hand.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


WRIST = 0
IDX_MCP = 5
PNK_MCP = 17


def build_hand_body_frame_single(joints_3d: np.ndarray
                                  ) -> Optional[np.ndarray]:
    """Single-frame, single-hand body frame as 4x4 SE(3) in world coords.

    Args:
        joints_3d: (21, 3) MANO joints in camera/world coords (meters).

    Returns:
        (4, 4) np.float32 or None if degenerate.
    """
    if joints_3d is None:
        return None
    j = np.asarray(joints_3d, dtype=np.float64)
    if j.shape != (21, 3) or not np.all(np.isfinite(j)):
        return None

    wrist = j[WRIST]
    v_x = j[IDX_MCP] - wrist
    v_y_raw = j[PNK_MCP] - wrist

    nx = np.linalg.norm(v_x)
    if nx < 1e-6:
        return None
    x_axis = v_x / nx

    # Project out the x-component of v_y_raw to get an in-palm-plane vector
    in_plane = v_y_raw - np.dot(v_y_raw, x_axis) * x_axis
    in_plane_norm = np.linalg.norm(in_plane)
    if in_plane_norm < 1e-6:
        return None

    # palm normal = x × (in-plane direction);  this is the "y" of our frame
    # (perpendicular to both wrist→index and the wrist→pinky in-plane component)
    y_axis = np.cross(x_axis, in_plane / in_plane_norm)
    ny = np.linalg.norm(y_axis)
    if ny < 1e-6:
        return None
    y_axis = y_axis / ny

    z_axis = np.cross(x_axis, y_axis)
    # z should already be unit-length by construction, but normalize for safety
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-9)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = wrist
    return T


def build_hand_body_frame_per_frame(
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    dominant_hand_per_frame: Sequence[Optional[str]],
) -> np.ndarray:
    """Returns (T, 4, 4) array of hand body frames (NaN where invalid).

    Selects the dominant hand at each frame; if dominant hand has no valid
    palm joints, the entry is NaN. The dominant_hand spec mirrors the
    convention used by `hand_rigid_grasp_lock` (string "L"/"R" per frame
    or None).
    """
    T_frames = len(joints_per_frame)
    out = np.full((T_frames, 4, 4), np.nan, dtype=np.float32)

    for t in range(T_frames):
        dom = dominant_hand_per_frame[t] if t < len(dominant_hand_per_frame) else None
        if dom not in ("L", "R"):
            continue
        use_right = (dom == "R")

        j_list = joints_per_frame[t] if t < len(joints_per_frame) else []
        r_list = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
        if not j_list or not r_list:
            continue

        for j, ir in zip(j_list, r_list):
            if bool(ir) != use_right:
                continue
            T_hand = build_hand_body_frame_single(j)
            if T_hand is not None:
                out[t] = T_hand
                break
    return out


def hand_frame_valid_mask(T_hand_seq: np.ndarray) -> np.ndarray:
    """(T,) bool: True where T_hand_seq[t] is a valid finite SE(3)."""
    return np.isfinite(T_hand_seq).reshape(T_hand_seq.shape[0], -1).all(axis=1)
