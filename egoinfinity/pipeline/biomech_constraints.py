"""
Biomechanical constraints for MANO hand pose parameters.

Clamps each joint's rotation to physiologically plausible ranges using
swing-twist decomposition. Inspired by Dyn-HaMR (ECCV 2024).

MANO 15-joint ordering (hand_pose, excluding global_orient):
  0: Index MCP    3: Middle MCP    6: Ring MCP     9: Pinky MCP
  1: Index PIP    4: Middle PIP    7: Ring PIP    10: Pinky PIP
  2: Index DIP    5: Middle DIP    8: Ring DIP    11: Pinky DIP
 12: Thumb CMC   13: Thumb MCP    14: Thumb IP

Each joint's axis-angle is decomposed into:
  - twist: rotation around the bone axis (abduction/adduction)
  - swing: rotation perpendicular to the bone axis (flexion/extension)

PIP/DIP joints are hinge joints (twist ≈ 0, swing = flexion only).
MCP joints allow flexion + small abduction.
Thumb joints have wider ranges.
"""
import numpy as np

# ── Per-joint limits (radians) ────────────────────────────────────────
# Format: (max_swing, max_twist)
# swing = flexion angle [0, max_swing]
# twist = abduction angle [-max_twist, +max_twist]

_DEG = np.pi / 180.0

# Finger PIP/DIP: hinge joints, flexion only
_PIP_DIP = (110 * _DEG, 5 * _DEG)   # near-zero twist allowed for numerical slack

# Finger MCP: flexion + abduction
_MCP = (90 * _DEG, 25 * _DEG)

# Thumb: more permissive
_THUMB_CMC = (60 * _DEG, 40 * _DEG)
_THUMB_MCP = (80 * _DEG, 15 * _DEG)
_THUMB_IP = (80 * _DEG, 5 * _DEG)

JOINT_LIMITS = {
    # Index
    0: _MCP, 1: _PIP_DIP, 2: _PIP_DIP,
    # Middle
    3: _MCP, 4: _PIP_DIP, 5: _PIP_DIP,
    # Ring
    6: _MCP, 7: _PIP_DIP, 8: _PIP_DIP,
    # Pinky
    9: _MCP, 10: _PIP_DIP, 11: _PIP_DIP,
    # Thumb
    12: _THUMB_CMC, 13: _THUMB_MCP, 14: _THUMB_IP,
}


# ── Swing-twist decomposition ────────────────────────────────────────

def _aa_to_rotmat(aa):
    """Axis-angle (3,) -> rotation matrix (3,3). Rodrigues formula."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    k = aa / angle
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotmat_to_aa(R):
    """Rotation matrix (3,3) -> axis-angle (3,). Inverse Rodrigues."""
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    # Extract axis from skew-symmetric part
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3)
    return (angle / axis_norm) * axis


def _decompose_swing_twist(aa, twist_axis):
    """Decompose axis-angle rotation into swing and twist components.

    twist: rotation around twist_axis (bone direction)
    swing: rotation perpendicular to twist_axis (flexion/extension)
    R = R_swing @ R_twist

    Args:
        aa: (3,) axis-angle rotation
        twist_axis: (3,) unit vector along the bone

    Returns:
        swing_angle: float, angle of swing rotation (always >= 0)
        swing_axis: (3,) unit axis of swing rotation
        twist_angle: float, signed angle of twist rotation
    """
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return 0.0, np.array([1.0, 0.0, 0.0]), 0.0

    rot_axis = aa / angle

    # Project rotation axis onto twist_axis to get twist component
    twist_proj = np.dot(rot_axis, twist_axis)
    twist_angle = angle * twist_proj

    # Swing = remainder
    R_full = _aa_to_rotmat(aa)
    R_twist = _aa_to_rotmat(twist_axis * twist_angle)
    R_swing = R_full @ R_twist.T

    swing_aa = _rotmat_to_aa(R_swing)
    swing_angle = np.linalg.norm(swing_aa)
    if swing_angle < 1e-8:
        swing_axis = np.array([1.0, 0.0, 0.0])
    else:
        swing_axis = swing_aa / swing_angle

    return swing_angle, swing_axis, twist_angle


def _compose_swing_twist(swing_angle, swing_axis, twist_angle, twist_axis):
    """Recompose axis-angle from swing and twist components.
    R = R_swing @ R_twist -> axis-angle
    """
    R_swing = _aa_to_rotmat(swing_axis * swing_angle)
    R_twist = _aa_to_rotmat(twist_axis * twist_angle)
    R = R_swing @ R_twist
    return _rotmat_to_aa(R)


# ── Main constraint function ────────────────────────────────────────

# Default twist axis: local x-axis (bone direction in MANO's rest pose)
# MANO joints flex around x-axis in their local coordinate frame.
_TWIST_AXIS = np.array([1.0, 0.0, 0.0])


def clamp_hand_pose(hand_pose_aa):
    """Apply biomechanical constraints to a single hand's pose.

    Args:
        hand_pose_aa: (15, 3) axis-angle per joint

    Returns:
        clamped: (15, 3) axis-angle with clamped rotations
        changed: bool, True if any joint was modified
    """
    clamped = hand_pose_aa.copy()
    changed = False

    for j in range(15):
        max_swing, max_twist = JOINT_LIMITS[j]
        aa = clamped[j]

        angle = np.linalg.norm(aa)
        if angle < 1e-8:
            continue

        swing_angle, swing_axis, twist_angle = _decompose_swing_twist(
            aa, _TWIST_AXIS)

        need_clamp = False

        # Clamp swing (flexion): must be in [0, max_swing]
        # Negative swing = hyperextension = not allowed for most joints
        clamped_swing = swing_angle
        if swing_angle > max_swing:
            clamped_swing = max_swing
            need_clamp = True
        # Allow small negative swing for numerical tolerance only
        if swing_angle < -5 * _DEG:
            clamped_swing = 0.0
            need_clamp = True

        # Clamp twist (abduction): must be in [-max_twist, max_twist]
        clamped_twist = twist_angle
        if abs(twist_angle) > max_twist:
            clamped_twist = np.sign(twist_angle) * max_twist
            need_clamp = True

        if need_clamp:
            clamped[j] = _compose_swing_twist(
                clamped_swing, swing_axis, clamped_twist, _TWIST_AXIS)
            changed = True

    return clamped, changed


def apply_biomech_constraints(hand_results_per_frame, mano_model=None):
    """Apply biomechanical constraints to all hands in all frames.

    Modifies HandResult objects in-place. If a MANO model is provided,
    recomputes joints_3d_rel and vertices for modified frames.

    Args:
        hand_results_per_frame: list of list of HandResult
        mano_model: optional MANO model for recomputing geometry

    Returns:
        n_clamped: number of hands that were modified
    """
    import torch
    from .infiller_utils.rotation import angle_axis_to_rotation_matrix

    n_clamped = 0

    for frame_hands in hand_results_per_frame:
        for h in frame_hands:
            # Convert rotmat (15, 3, 3) -> axis-angle (15, 3)
            pose_aa = np.zeros((15, 3), dtype=np.float32)
            for j in range(15):
                pose_aa[j] = _rotmat_to_aa(h.hand_pose[j])

            clamped_aa, changed = clamp_hand_pose(pose_aa)

            if not changed:
                continue

            n_clamped += 1

            # Convert back to rotation matrices
            clamped_rotmat = np.zeros((15, 3, 3), dtype=np.float32)
            for j in range(15):
                clamped_rotmat[j] = _aa_to_rotmat(clamped_aa[j])
            h.hand_pose = clamped_rotmat

            # Recompute joints and vertices if MANO available
            if mano_model is not None:
                go = torch.from_numpy(h.global_orient).float().unsqueeze(0)
                hp = torch.from_numpy(clamped_rotmat).float().unsqueeze(0)
                b = torch.from_numpy(h.betas).float().unsqueeze(0)

                with torch.no_grad():
                    out = mano_model(
                        global_orient=go, hand_pose=hp, betas=b, pose2rot=False)

                joints = out.joints[0].cpu().numpy()
                vertices = out.vertices[0].cpu().numpy()

                flip = (2 * int(h.is_right) - 1)
                joints[:, 0] *= flip
                vertices[:, 0] *= flip

                h.joints_3d_rel = joints
                h.joints_3d = joints + h.cam_t
                h.vertices = vertices + h.cam_t

    return n_clamped
