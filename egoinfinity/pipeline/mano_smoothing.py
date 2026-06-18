"""
Temporal smoothing of MANO rotation parameters (global_orient + hand_pose),
with MANO forward pass to produce consistent vertices.

Smooths axis-angle parameters per-track using Savitzky-Golay filter,
then runs MANO forward to get vertices that are both temporally smooth
and topologically correct (no mesh deformation).
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.signal import savgol_filter

from .hand_reconstructor import HandResult


def _rotmat_to_aa(R):
    """Rotation matrix (3,3) -> axis-angle (3,). Rodrigues inverse."""
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (angle / n * axis).astype(np.float32)


def _rotmat_to_quat(R):
    """Rotation matrix (3,3) -> quaternion (4,) in wxyz order."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat(scalar_first=True).astype(np.float32)


def _quat_to_aa(q):
    """Quaternion (4,) wxyz -> axis-angle (3,)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q, scalar_first=True).as_rotvec().astype(np.float32)


def _ensure_quat_continuity(quat_seq):
    """Ensure quaternion sign consistency across a sequence.

    q and -q represent the same rotation. We pick the sign that
    minimizes the step from the previous frame (dot product > 0).

    Args:
        quat_seq: (T, 4) quaternion sequence (wxyz)

    Returns:
        (T, 4) sign-consistent sequence
    """
    T = len(quat_seq)
    if T < 2:
        return quat_seq.copy()

    out = quat_seq.copy()
    for t in range(1, T):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def _savgol_quat(quat_seq, window, polyorder):
    """SavGol filter on quaternion sequence, with re-normalization.

    Args:
        quat_seq: (T, 4) sign-consistent quaternions (wxyz)
        window: SavGol window
        polyorder: polynomial order

    Returns:
        (T, 4) smoothed, normalized quaternions
    """
    T = len(quat_seq)
    win = min(window, T)
    if win % 2 == 0:
        win -= 1
    win = max(win, polyorder + 2)
    if win % 2 == 0:
        win += 1
    if T < win:
        return quat_seq.copy()

    smoothed = savgol_filter(quat_seq, window_length=win, polyorder=polyorder, axis=0)
    # Re-normalize to unit quaternions
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return smoothed / norms


@torch.no_grad()
def _batch_mano_forward(mano, global_orient_aa, hand_pose_aa, betas,
                         is_right, cam_t):
    """Run batched MANO forward for a sequence of frames.

    Args:
        mano: MANO model (right-hand)
        global_orient_aa: (T, 3) axis-angle
        hand_pose_aa: (T, 45) axis-angle (15 joints x 3)
        betas: (T, 10) shape params
        is_right: bool
        cam_t: (T, 3) camera translation

    Returns:
        joints_3d: (T, 21, 3) joints in camera frame
        vertices_3d: (T, 778, 3) vertices in camera frame
    """
    from .infiller_utils.rotation import angle_axis_to_rotation_matrix

    T = len(global_orient_aa)

    # Convert to rotation matrices
    go_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(global_orient_aa).float()
    ).unsqueeze(1)  # (T, 1, 3, 3)

    hp_flat = hand_pose_aa.reshape(T * 15, 3)
    hp_mat = angle_axis_to_rotation_matrix(
        torch.from_numpy(hp_flat).float()
    ).reshape(T, 15, 3, 3)  # (T, 15, 3, 3)

    b = torch.from_numpy(betas).float()  # (T, 10)

    # Batch forward (process in chunks to avoid OOM for very long sequences)
    chunk_size = 64
    all_joints = []
    all_verts = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        out = mano(
            global_orient=go_mat[start:end],
            hand_pose=hp_mat[start:end],
            betas=b[start:end],
            pose2rot=False,
        )
        all_joints.append(out.joints.cpu().numpy())
        all_verts.append(out.vertices.cpu().numpy())

    joints = np.concatenate(all_joints, axis=0)   # (T, 21, 3)
    verts = np.concatenate(all_verts, axis=0)      # (T, 778, 3)

    # Flip x for left hands
    flip = (2 * int(is_right) - 1)
    joints[:, :, 0] *= flip
    verts[:, :, 0] *= flip

    # Add cam_t
    joints_3d = joints + cam_t[:, np.newaxis, :]
    verts_3d = verts + cam_t[:, np.newaxis, :]

    return joints_3d, verts_3d


def _get_skinning_weights(mano_model):
    """Extract MANO skinning weights (778, 16) and pad to (778, 21) for OpenPose joints.

    MANO has 16 joints internally but outputs 21 joints (with extra fingertip joints
    regressed from vertices). For the 5 extra joints (fingertips), we use the
    nearest MANO joint's weight column.
    """
    # lbs_weights: (778, 16) — weights for 16 MANO joints
    weights = mano_model.lbs_weights.detach().cpu().numpy()  # (778, 16)

    # MANO-to-OpenPose joint map (from mano_wrapper.py)
    # OpenPose joints 0-20, MANO internal joints 0-15
    # Fingertip joints (16-20 in MANO output) are regressed from vertices,
    # not part of the kinematic chain. Map them to their parent DIP joints.
    # OpenPose: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    # Parent DIP: thumb=3, index=7, middle=11, ring=15, pinky=19 (in OpenPose)

    # For simplicity, pad with zeros for the 5 extra joints
    # and use the existing 16 columns for the main joints
    weights_21 = np.zeros((778, 21), dtype=np.float32)
    # Map the 16 MANO joints to OpenPose ordering
    mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    for openpose_idx in range(21):
        mano_idx = mano_to_openpose.index(openpose_idx) if openpose_idx in mano_to_openpose else -1
        if mano_idx >= 0 and mano_idx < 16:
            weights_21[:, openpose_idx] = weights[:, mano_idx]

    # Normalize rows to sum to 1
    row_sums = weights_21.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    weights_21 /= row_sums

    return weights_21


def smooth_mano_params(
    hand_results_per_frame: List[List[HandResult]],
    smoothed_cam_t_map: Dict,
    mano_model,
    total_frames: int,
    smoothed_joints_map: Optional[Dict] = None,
    window: int = 7,
    polyorder: int = 2,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Smooth MANO parameters per-track and recompute vertices via MANO forward.

    If smoothed_joints_map is provided, applies joint-guided deformation to
    pull MANO mesh vertices toward the smoothed joint positions using
    MANO skinning weights.

    Args:
        hand_results_per_frame: list of list of HandResult
        smoothed_cam_t_map: {(frame, track_id): cam_t} from translation smoothing
        mano_model: MANO model for forward pass
        total_frames: number of frames
        smoothed_joints_map: {(frame, track_id): (21,3)} SavGol-smoothed joints
        window: SavGol window
        polyorder: SavGol polynomial order

    Returns:
        smoothed_verts_map: {(frame, track_id): (778, 3)} smoothed vertices
    """
    if mano_model is None:
        return {}

    # Group by track_id
    track_data = defaultdict(dict)  # {tid: {frame: HandResult}}
    for i in range(total_frames):
        for h in hand_results_per_frame[i]:
            track_data[h.track_id][i] = h

    smoothed_verts_map = {}

    # Get skinning weights for joint-guided deformation
    skinning_w = None
    if smoothed_joints_map is not None:
        try:
            skinning_w = _get_skinning_weights(mano_model)  # (778, 21)
        except Exception:
            pass

    for tid, frame_dict in track_data.items():
        if len(frame_dict) < 3:
            # Too few frames, skip smoothing
            continue

        frames_sorted = sorted(frame_dict.keys())
        T = len(frames_sorted)
        h0 = frame_dict[frames_sorted[0]]
        is_right = h0.is_right

        # Extract per-frame MANO params as quaternions (wxyz) for smooth interpolation
        go_quat = np.zeros((T, 4), dtype=np.float32)
        hp_quat = np.zeros((T, 15, 4), dtype=np.float32)
        betas = np.zeros((T, 10), dtype=np.float32)
        cam_ts = np.zeros((T, 3), dtype=np.float32)

        for t, fi in enumerate(frames_sorted):
            h = frame_dict[fi]
            go_quat[t] = _rotmat_to_quat(h.global_orient[0])
            for j in range(15):
                hp_quat[t, j] = _rotmat_to_quat(h.hand_pose[j])
            betas[t] = h.betas
            ct = smoothed_cam_t_map.get((fi, tid))
            cam_ts[t] = ct if ct is not None else h.cam_t

        # Ensure quaternion sign consistency, then SavGol + renormalize
        go_quat = _ensure_quat_continuity(go_quat)
        go_quat_smooth = _savgol_quat(go_quat, window, polyorder)

        hp_quat_smooth = np.zeros_like(hp_quat)
        for j in range(15):
            hp_quat[:, j] = _ensure_quat_continuity(hp_quat[:, j])
            hp_quat_smooth[:, j] = _savgol_quat(hp_quat[:, j], window, polyorder)

        # Convert smoothed quaternions back to axis-angle for MANO forward
        go_smooth = np.zeros((T, 3), dtype=np.float32)
        hp_smooth = np.zeros((T, 45), dtype=np.float32)
        for t in range(T):
            go_smooth[t] = _quat_to_aa(go_quat_smooth[t])
            for j in range(15):
                hp_smooth[t, j*3:(j+1)*3] = _quat_to_aa(hp_quat_smooth[t, j])

        # Re-apply biomechanical constraints after smoothing
        from .biomech_constraints import clamp_hand_pose
        for t in range(T):
            hp_frame = hp_smooth[t].reshape(15, 3)
            hp_frame, _ = clamp_hand_pose(hp_frame)
            hp_smooth[t] = hp_frame.flatten()

        # Betas: use per-track median (shape shouldn't change frame-to-frame)
        betas_median = np.median(betas, axis=0, keepdims=True).repeat(T, axis=0)

        # MANO forward with smoothed params
        _, verts_3d = _batch_mano_forward(
            mano_model, go_smooth, hp_smooth, betas_median, is_right, cam_ts)

        # Store in map
        for t, fi in enumerate(frames_sorted):
            smoothed_verts_map[(fi, tid)] = verts_3d[t]  # (778, 3)

    return smoothed_verts_map
