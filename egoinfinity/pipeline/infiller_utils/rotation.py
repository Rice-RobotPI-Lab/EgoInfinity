"""
Rotation conversion utilities for the motion infiller.
Adapted from HaWoR (infiller/hand_utils/rotation.py, lib/utils/geometry.py).
Self-contained — no external MANO dependency.
"""
import torch
import numpy as np
from torch.nn import functional as F


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert angle-axis to quaternion (WXYZ order).
    :param angle_axis (*, 3)
    :returns quaternion (*, 4) WXYZ
    """
    theta_sq = torch.sum(angle_axis ** 2, dim=-1, keepdim=True)
    valid = theta_sq > 0
    theta = torch.sqrt(theta_sq)
    half_theta = 0.5 * theta
    ones = torch.ones_like(half_theta)
    k = torch.where(valid, torch.sin(half_theta) / theta, 0.5 * ones)
    w = torch.where(valid, torch.cos(half_theta), ones)
    return torch.cat([w, k * angle_axis], dim=-1)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (WXYZ) to rotation matrix.
    :param quaternion (*, 4)
    :returns rotation_matrix (*, 3, 3)
    """
    q = F.normalize(quaternion, p=2, dim=-1, eps=1e-12)
    dims = q.shape[:-1]
    w, x, y, z = torch.chunk(q, chunks=4, dim=-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    one = torch.tensor(1.0, device=q.device, dtype=q.dtype)
    matrix = torch.stack((
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy),
    ), dim=-1).view(*dims, 3, 3)
    return matrix


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert angle-axis (*, 3) to rotation matrix (*, 3, 3)."""
    return quaternion_to_rotation_matrix(angle_axis_to_quaternion(angle_axis))


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert rotation matrix (*, 3, 3) to quaternion (*, 4) WXYZ.
    Adapted from HaWoR infiller/hand_utils/rotation.py — works with (*, 3, 3) directly.
    """
    original_shape = rotation_matrix.shape[:-2]
    rmat_t = torch.transpose(rotation_matrix.reshape(-1, 3, 3), -1, -2)

    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0,
                       rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                       rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                       rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1,
                       rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                       rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                       rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                       rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                       rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = (mask_d2 * mask_d0_d1).view(-1, 1).type_as(q0)
    mask_c1 = (mask_d2 * ~mask_d0_d1).view(-1, 1).type_as(q1)
    mask_c2 = (~mask_d2 * mask_d0_nd1).view(-1, 1).type_as(q2)
    mask_c3 = (~mask_d2 * ~mask_d0_nd1).view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +
                     t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q.reshape(*original_shape, 4)


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (*, 3, 3) to angle-axis (*, 3)."""
    original_shape = rotation_matrix.shape[:-2]
    flat = rotation_matrix.reshape(-1, 3, 3)
    quat = rotation_matrix_to_quaternion(flat)
    # quaternion_to_angle_axis
    q1, q2, q3 = quat[..., 1], quat[..., 2], quat[..., 3]
    sin_sq = q1 * q1 + q2 * q2 + q3 * q3
    sin_t = torch.sqrt(sin_sq)
    cos_t = quat[..., 0]
    two_theta = 2.0 * torch.where(
        cos_t < 0.0, torch.atan2(-sin_t, -cos_t), torch.atan2(sin_t, cos_t))
    k = torch.where(sin_sq > 0.0, two_theta / sin_t, 2.0 * torch.ones_like(sin_t))
    aa = torch.zeros_like(quat)[..., :3]
    aa[..., 0] += q1 * k
    aa[..., 1] += q2 * k
    aa[..., 2] += q3 * k
    aa[torch.isnan(aa)] = 0.0
    return aa.reshape(*original_shape, 3)


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (..., 3, 3) to 6D representation (..., 6)."""
    return rotmat[..., :2].reshape(*rotmat.shape[:-2], 6)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation (..., 6) to rotation matrix (..., 3, 3).
    Based on Zhou et al., CVPR 2019.
    """
    original_shape = x.shape[:-1]
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    mat = torch.stack((b1, b2, b3), dim=-1)
    return mat.reshape(*original_shape, 3, 3)
