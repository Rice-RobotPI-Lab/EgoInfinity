"""
Motion infiller pre/post-processing utilities.
Adapted from HaWoR (lib/eval_utils/filling_utils.py).
Uses MANO forward pass for correct canonical-frame conversion
(root_loc offset), matching HaWoR's training distribution.
"""
import copy
import numpy as np
import torch
from scipy.spatial.transform import Slerp, Rotation

from .rotation import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
)


# ── Interpolation helpers ────────────────────────────────────────────

def slerp_interpolation_aa(pos, valid):
    """SLERP interpolation for angle-axis rotation sequences.
    Args:
        pos: (B, T, N, 3) angle-axis rotations
        valid: (B, T) boolean validity mask
    Returns:
        (B, T, N, 3) interpolated rotations
    """
    B, T, N, _ = pos.shape
    pos_interp = pos.copy()
    for b in range(B):
        for n in range(N):
            aa_bn = pos[b, :, n, :]
            valid_bn = valid[b, :]
            invalid_idxs = np.where(~valid_bn)[0]
            valid_idxs = np.where(valid_bn)[0]
            if len(invalid_idxs) == 0 or len(valid_idxs) < 2:
                if len(valid_idxs) == 1:
                    for idx in invalid_idxs:
                        pos_interp[b, idx, n, :] = aa_bn[valid_idxs[0]]
                continue
            valid_rots = Rotation.from_rotvec(aa_bn[valid_idxs])
            slerp = Slerp(valid_idxs, valid_rots)
            for idx in invalid_idxs:
                if idx < valid_idxs[0]:
                    pos_interp[b, idx, n, :] = aa_bn[valid_idxs[0]]
                elif idx > valid_idxs[-1]:
                    pos_interp[b, idx, n, :] = aa_bn[valid_idxs[-1]]
                else:
                    pos_interp[b, idx, n, :] = slerp([idx]).as_rotvec()[0]
    return pos_interp


def linear_interpolation_nd(pos, valid):
    """Linear interpolation for N-D value sequences.
    Args:
        pos: (B, T, D) values
        valid: (B, T) boolean validity mask
    Returns:
        (B, T, D) interpolated values
    """
    B, T = pos.shape[:2]
    feature_dim = pos.shape[2]
    pos_interp = pos.copy()
    for b in range(B):
        valid_b = valid[b, :]
        invalid_idxs = np.where(~valid_b)[0]
        valid_idxs = np.where(valid_b)[0]
        if len(invalid_idxs) == 0 or len(valid_idxs) < 2:
            continue
        for idx in range(feature_dim):
            pos_b_idx = pos[b, :, idx].copy()
            pos_b_idx[invalid_idxs] = np.interp(invalid_idxs, valid_idxs, pos_b_idx[valid_idxs])
            pos_interp[b, :, idx] = pos_b_idx
    return pos_interp


# ── MANO-aware canonical-frame conversion ────────────────────────────

@torch.no_grad()
def _mano_root_loc(mano, trans, rot_aa, hand_pose_aa, betas, is_right):
    """Run MANO forward to get wrist joint location at reference frame.

    Args:
        mano: MANO model (right-hand, WiLoR-style)
        trans: (3,) translation
        rot_aa: (3,) root orientation axis-angle
        hand_pose_aa: (15, 3) hand pose axis-angle
        betas: (10,) shape
        is_right: bool

    Returns:
        root_loc: (3,) wrist joint position
    """
    go_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(rot_aa).float().unsqueeze(0)
    ).unsqueeze(0)  # (1, 1, 3, 3)
    hp_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(hand_pose_aa).float()
    ).unsqueeze(0)  # (1, 15, 3, 3)
    b = torch.from_numpy(betas).float().unsqueeze(0)  # (1, 10)

    out = mano(global_orient=go_mat, hand_pose=hp_mat, betas=b,
               transl=torch.from_numpy(trans).float().unsqueeze(0),
               pose2rot=False)
    joints = out.joints[0].cpu()  # (21, 3)

    # WiLoR flips x for left hands
    flip = (2 * int(is_right) - 1)
    joints[:, 0] *= flip

    return joints[0]  # wrist = joint 0


def _world2canonical_convert(R_c2w, t_c2w, root_rot_mat, trans, root_loc_ref,
                              offset, T):
    """Transform from world/camera space to canonical space.

    Matches HaWoR's world2canonical_convert logic:
      - Rotate root orientation: R_canon = R_w2c @ R_world
      - Compute per-frame translation using root_loc:
        canon_trans = R_w2c @ root_loc + t_w2c + offset

    For the simplified case (no per-frame MANO forward), we approximate
    root_loc ≈ trans - offset (since offset = trans_ref - root_loc_ref is constant).

    Args:
        R_c2w: (T, 3, 3) canonical-to-world rotation (or its inverse for w2c)
        t_c2w: (T, 3) canonical-to-world translation
        root_rot_mat: (1, T, 3, 3) root orientation matrices
        trans: (1, T, 3) translations
        root_loc_ref: (3,) wrist location at reference frame
        offset: (3,) constant offset = trans_ref - root_loc_ref
        T: number of frames

    Returns:
        rot_aa: (1, T, 3) canonical root orientation (axis-angle)
        canon_trans: (1, T, 3) canonical translation
    """
    # Rotate root orientation
    rotated = torch.einsum("tij,btjk->btik", R_c2w, root_rot_mat)
    rot_aa = rotation_matrix_to_angle_axis(rotated.reshape(-1, 3, 3)).reshape(1, T, 3)

    # Approximate root_loc for all frames: root_loc ≈ trans - offset
    approx_root_loc = trans - offset.unsqueeze(0).unsqueeze(0)  # (1, T, 3)

    # Transform translation: R_w2c @ root_loc + t_w2c + offset
    canon_trans = (
        torch.einsum("tij,btj->bti", R_c2w, approx_root_loc)
        + t_c2w.unsqueeze(0)
        + offset.unsqueeze(0).unsqueeze(0)
    )

    return rot_aa, canon_trans


def filling_preprocess(item, mano=None):
    """Preprocess hand motion data for the infiller network.

    Converts to canonical frame using MANO root_loc offset (if MANO available),
    interpolates missing frames, encodes as rot6d.

    Args:
        item: dict with keys:
            trans:     (2, T, 3) global translation
            rot:       (2, T, 3) root orientation (angle-axis)
            hand_pose: (2, T, 45) finger joint rotations (angle-axis, 15*3)
            betas:     (2, T, 10) shape parameters
            valid:     (2, T) boolean validity mask
        mano: optional MANO model for root_loc computation

    Returns:
        global_pose_vec: (T, 218) concatenated rot6d vector for both hands
        transform_info: dict with canonical<->world transform matrices
    """
    num_joints = 15

    global_trans = item['trans']      # (2, T, 3)
    global_rot = item['rot']          # (2, T, 3)
    hand_pose = item['hand_pose']     # (2, T, 45)
    betas = item['betas']             # (2, T, 10)
    valid = item['valid']             # (2, T) bool

    N, T, _ = global_trans.shape
    hand_pose_reshaped = hand_pose.reshape(N, T, num_joints, 3)

    transform_info = {}

    canonical_rots = []
    canonical_trans = []
    for hand_idx in range(N):
        is_right = (hand_idx == 1)
        valid_h = valid[hand_idx]
        valid_frames = np.where(valid_h)[0]

        if len(valid_frames) == 0:
            canonical_rots.append(torch.from_numpy(global_rot[hand_idx:hand_idx+1]).float())
            canonical_trans.append(torch.from_numpy(global_trans[hand_idx:hand_idx+1]).float())
            transform_info[f'R_w2c_{hand_idx}'] = torch.eye(3).unsqueeze(0).expand(T, -1, -1)
            transform_info[f't_w2c_{hand_idx}'] = torch.zeros(T, 3)
            transform_info[f'R_c2w_{hand_idx}'] = torch.eye(3).unsqueeze(0).expand(T, -1, -1)
            transform_info[f't_c2w_{hand_idx}'] = torch.zeros(T, 3)
            transform_info[f'offset_{hand_idx}'] = torch.zeros(3)
            continue

        ref_frame = valid_frames[0]
        ref_rot_aa = global_rot[hand_idx, ref_frame]
        ref_trans = global_trans[hand_idx, ref_frame]
        ref_hand_pose = hand_pose_reshaped[hand_idx, ref_frame]
        ref_betas = betas[hand_idx, ref_frame]

        # Compute root_loc offset using MANO
        if mano is not None:
            root_loc_ref = _mano_root_loc(
                mano, ref_trans, ref_rot_aa, ref_hand_pose, ref_betas, is_right)
            offset = torch.from_numpy(ref_trans).float() - root_loc_ref  # constant
        else:
            offset = torch.zeros(3)
            root_loc_ref = torch.from_numpy(ref_trans).float()

        # R_w2c = inverse of reference frame rotation
        R_w2c = angle_axis_to_rotation_matrix(
            torch.from_numpy(ref_rot_aa).float()
        ).t()

        # t_w2c: matches HaWoR — t = -R_w2c @ root_loc_ref - offset
        t_w2c = -torch.mv(R_w2c, root_loc_ref) - offset

        R_w2c_T = R_w2c.unsqueeze(0).expand(T, -1, -1)
        t_w2c_T = t_w2c.unsqueeze(0).expand(T, -1)

        root_rot_mat = angle_axis_to_rotation_matrix(
            torch.from_numpy(global_rot[hand_idx:hand_idx+1]).float().reshape(-1, 3)
        ).reshape(1, T, 3, 3)
        trans_tensor = torch.from_numpy(global_trans[hand_idx:hand_idx+1]).float()

        canon_rot_aa, canon_trans = _world2canonical_convert(
            R_w2c_T, t_w2c_T, root_rot_mat, trans_tensor,
            root_loc_ref, offset, T)

        canonical_rots.append(canon_rot_aa)
        canonical_trans.append(canon_trans)

        R_c2w = R_w2c.t()
        t_c2w = -torch.mv(R_c2w, t_w2c)
        transform_info[f'R_w2c_{hand_idx}'] = R_w2c_T
        transform_info[f't_w2c_{hand_idx}'] = t_w2c_T
        transform_info[f'R_c2w_{hand_idx}'] = R_c2w.unsqueeze(0).expand(T, -1, -1)
        transform_info[f't_c2w_{hand_idx}'] = t_c2w.unsqueeze(0).expand(T, -1)
        transform_info[f'offset_{hand_idx}'] = offset

    # Merge canonical data
    global_rot_canon = torch.cat(canonical_rots, dim=0).numpy()
    global_trans_canon = torch.cat(canonical_trans, dim=0).numpy()

    global_rot_canon = global_rot_canon.reshape(N, T, 1, 3)

    # SLERP and linear interpolation
    global_trans_lerped = linear_interpolation_nd(global_trans_canon, valid)
    betas_lerped = linear_interpolation_nd(betas, valid)
    global_rot_slerped = slerp_interpolation_aa(global_rot_canon, valid)
    hand_pose_slerped = slerp_interpolation_aa(hand_pose_reshaped, valid)

    # Convert to rot6d
    global_rot_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(global_rot_slerped.reshape(N * T, -1)).float()
    )
    global_rot_rot6d = rotmat_to_rot6d(global_rot_mat).reshape(N, T, -1).numpy()

    hand_pose_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(hand_pose_slerped.reshape(N * T * num_joints, -1)).float()
    )
    hand_pose_rot6d = rotmat_to_rot6d(hand_pose_mat).reshape(N, T, -1).numpy()

    global_pose_vec = np.concatenate(
        (global_trans_lerped, betas_lerped, global_rot_rot6d, hand_pose_rot6d),
        axis=-1
    ).transpose(1, 0, 2).reshape(T, -1)

    return global_pose_vec, transform_info


def filling_postprocess(output, transform_info):
    """Postprocess infiller output back to camera-space MANO parameters.

    Args:
        output: (T, 2, 109) tensor from infiller
        transform_info: dict from filling_preprocess

    Returns:
        dict with keys:
            trans:     (2, T, 3) global translation
            rot:       (2, T, 3) root orientation (angle-axis)
            hand_pose: (2, T, 45) finger rotations (angle-axis)
            betas:     (2, T, 10) shape parameters
    """
    output = output.permute(1, 0, 2)  # (2, T, 109)
    N, T, _ = output.shape

    canon_trans = output[:, :, :3]
    betas = output[:, :, 3:13]
    canon_rot_rot6d = output[:, :, 13:19]
    hand_pose_rot6d = output[:, :, 19:109].reshape(N, T, 15, 6)

    canon_rot_mat = rot6d_to_rotmat(canon_rot_rot6d.reshape(-1, 6)).reshape(N, T, 3, 3)
    hand_pose_mat = rot6d_to_rotmat(hand_pose_rot6d.reshape(-1, 6)).reshape(N, T, 15, 3, 3)

    hand_pose_aa = rotation_matrix_to_angle_axis(
        hand_pose_mat.reshape(-1, 3, 3)
    ).reshape(N, T, 15, 3)

    # Transform each hand back from canonical to camera space
    world_rots = []
    world_trans = []
    for hand_idx in range(N):
        R_c2w = transform_info[f'R_c2w_{hand_idx}']  # (T, 3, 3)
        t_c2w = transform_info[f't_c2w_{hand_idx}']  # (T, 3)
        offset = transform_info[f'offset_{hand_idx}']  # (3,)

        # Rotate root orientation back
        rotated = torch.einsum(
            "tij,btjk->btik", R_c2w,
            canon_rot_mat[hand_idx:hand_idx+1]
        )
        rot_aa = rotation_matrix_to_angle_axis(
            rotated.reshape(-1, 3, 3)
        ).reshape(1, T, 3)

        # Inverse of canonical transform:
        # canon_trans = R_w2c @ (trans - offset) + t_w2c + offset
        # => trans - offset = R_c2w @ (canon_trans - offset - t_w2c)
        # => trans = R_c2w @ (canon_trans - offset) + t_c2w + offset
        # Simplified: since t_c2w = -R_c2w @ t_w2c, this is equivalent to
        # the reverse of the forward transform.
        canon_minus_offset = canon_trans[hand_idx:hand_idx+1] - offset.unsqueeze(0).unsqueeze(0)
        trans_back = (
            torch.einsum("tij,btj->bti", R_c2w, canon_minus_offset)
            + t_c2w.unsqueeze(0)
            + offset.unsqueeze(0).unsqueeze(0)
        )

        world_rots.append(rot_aa)
        world_trans.append(trans_back)

    global_rot = torch.cat(world_rots, dim=0).numpy()
    global_trans = torch.cat(world_trans, dim=0).numpy()

    return {
        "trans": global_trans,
        "rot": global_rot,
        "hand_pose": hand_pose_aa.flatten(-2).numpy(),
        "betas": betas.numpy(),
    }
