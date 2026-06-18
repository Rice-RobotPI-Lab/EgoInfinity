"""
Root frame estimation pipeline — everything between model inference and IK.

Functions cover:
  - estimating per-window root poses from the neural model
  - clustering and scoring candidate anchors
  - blending, interpolating, and smoothing per-frame root frames
  - converting camera-frame wrist trajectories to root-frame IK targets
  - bilateral separation rescaling
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


# ── window estimation ─────────────────────────────────────────────────────────

def estimate_root_poses(
    model,
    left_cam:     np.ndarray,
    right_cam:    np.ndarray,
    g_cam_t:      torch.Tensor,
    seq_len:      int,
    seq_fps:      float,
    window_secs:  float,
    window_stride: int,
    device:       torch.device,
    batch_size:   int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model at strided window positions; return (kf_positions, Rs, ts).

    window_stride = window_len  → sparse keyframes
    window_stride = 1           → fully sliding window
    window_stride = N           → intermediate density
    """
    window_len   = max(4, int(round(window_secs * seq_fps)))
    half         = window_len // 2
    kf_positions = sorted(set(list(range(0, seq_len, window_stride)) + [seq_len - 1]))

    lt_all = np.stack([
        left_cam[max(0, min(k - half, seq_len - window_len)):
                 max(0, min(k - half, seq_len - window_len)) + window_len]
        for k in kf_positions
    ])
    rt_all = np.stack([
        right_cam[max(0, min(k - half, seq_len - window_len)):
                  max(0, min(k - half, seq_len - window_len)) + window_len]
        for k in kf_positions
    ])

    Rs, ts = [], []
    for start in range(0, len(kf_positions), batch_size):
        end = min(start + batch_size, len(kf_positions))
        b   = end - start
        lt  = torch.tensor(lt_all[start:end], dtype=torch.float32).to(device)
        rt  = torch.tensor(rt_all[start:end], dtype=torch.float32).to(device)
        g_b = g_cam_t.expand(b, -1) if g_cam_t.shape[0] == 1 else g_cam_t[:b]
        with torch.no_grad():
            R, t = model.sample(lt, rt, g_cam=g_b)
        Rs.append(R.cpu().numpy())
        ts.append(t.cpu().numpy())

    Rs = np.concatenate(Rs, axis=0)
    ts = np.concatenate(ts, axis=0)
    print(f"  {len(kf_positions)} windows  "
          f"(window={window_secs}s/{window_len}fr  stride={window_stride}fr)")
    return np.array(kf_positions, dtype=float), Rs, ts


# ── anchor selection ──────────────────────────────────────────────────────────

def select_best_anchor(
    Rs:               np.ndarray,
    ts:               np.ndarray,
    left_cam:         np.ndarray,
    right_cam:        np.ndarray,
    opt,
    workspace_center: dict | None,
    n_clusters:       int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """K-medoids cluster → score all K candidates via IK → return best (R, t, ik_rate)."""
    from models.root_opt import cluster_se3
    K = min(n_clusters, len(Rs))
    R_cands, t_cands, _ = cluster_se3(Rs, ts, K=K)
    T = len(left_cam)

    all_lp, all_lq, all_rp, all_rq = [], [], [], []
    for k in range(K):
        R_pf = np.broadcast_to(R_cands[k], (T, 3, 3))
        t_pf = np.broadcast_to(t_cands[k], (T, 3))
        lp, lq, rp, rq = cam_to_root_targets(left_cam, right_cam, R_pf, t_pf, workspace_center)
        all_lp.append(lp); all_lq.append(lq)
        all_rp.append(rp); all_rq.append(rq)

    lp_KT = torch.tensor(np.concatenate(all_lp), dtype=torch.float32)
    lq_KT = torch.tensor(np.concatenate(all_lq), dtype=torch.float32)
    rp_KT = torch.tensor(np.concatenate(all_rp), dtype=torch.float32)
    rq_KT = torch.tensor(np.concatenate(all_rq), dtype=torch.float32)

    with torch.no_grad():
        _, info_l = opt.ik_left.solve_batch( lp_KT, lq_KT)
        _, info_r = opt.ik_right.solve_batch(rp_KT, rq_KT)

    conv_l   = info_l["converged"].cpu().numpy().reshape(K, T)
    conv_r   = info_r["converged"].cpu().numpy().reshape(K, T)
    ik_rates = (conv_l.mean(axis=1) + conv_r.mean(axis=1)) / 2

    for k in range(K):
        print(f"  anchor cand {k}: ik_rate={ik_rates[k]*100:.1f}%  t={t_cands[k].round(3)}")

    best_k = int(np.argmax(ik_rates))
    print(f"  best anchor: ik_rate={ik_rates[best_k]*100:.1f}%  t={t_cands[best_k].round(3)}")
    return R_cands[best_k], t_cands[best_k], float(ik_rates[best_k])


# ── blending ──────────────────────────────────────────────────────────────────

def blend_keyframes(
    Rs:        np.ndarray,
    ts:        np.ndarray,
    alpha:     float,
    alpha_rot: float = 1.0,
    R_anchor:  np.ndarray | None = None,
    t_anchor:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend per-window estimates toward the anchor. alpha=0 → anchor only, 1 → per-window."""
    R_global = R_anchor if R_anchor is not None else Rotation.from_matrix(Rs).mean().as_matrix()
    t_global = t_anchor if t_anchor is not None else ts.mean(axis=0)

    ts_blended = (1.0 - alpha) * t_global[None] + alpha * ts

    if alpha_rot >= 1.0:
        Rs_blended = Rs.copy()
    elif alpha_rot <= 0.0:
        Rs_blended = np.broadcast_to(R_global, Rs.shape).copy()
    else:
        Rs_blended = np.zeros_like(Rs)
        for k in range(len(Rs)):
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_global, Rs[k]])))
            Rs_blended[k] = slerp([alpha_rot]).as_matrix()[0]

    return Rs_blended, ts_blended


# ── interpolation and smoothing ───────────────────────────────────────────────

def interpolate_root_frames(
    kf_positions: np.ndarray,
    Rs:           np.ndarray,
    ts:           np.ndarray,
    T:            int,
) -> tuple[np.ndarray, np.ndarray]:
    """SLERP rotations + linear interpolation of translations → (T, 3, 3), (T, 3)."""
    frames         = np.arange(T, dtype=float)
    frames_clamped = np.clip(frames, kf_positions[0], kf_positions[-1])

    t_interp = np.stack([
        np.interp(frames, kf_positions, ts[:, i]) for i in range(3)
    ], axis=1)

    if len(kf_positions) == 1:
        R_interp = np.broadcast_to(Rs[0], (T, 3, 3)).copy()
    else:
        slerp    = Slerp(kf_positions, Rotation.from_matrix(Rs))
        R_interp = slerp(frames_clamped).as_matrix()

    return R_interp, t_interp


def smooth_root_frames(
    Rs:    np.ndarray,
    ts:    np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-smooth per-frame root poses along the time axis."""
    if sigma <= 0:
        return Rs, ts
    ts_smooth = gaussian_filter1d(ts, sigma=sigma, axis=0)
    quats = Rotation.from_matrix(Rs).as_quat()
    for t in range(1, len(quats)):
        if np.dot(quats[t], quats[t-1]) < 0:
            quats[t] = -quats[t]
    quats_smooth = gaussian_filter1d(quats, sigma=sigma, axis=0)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    return Rotation.from_quat(quats_smooth).as_matrix(), ts_smooth


# ── coordinate transforms ─────────────────────────────────────────────────────

def cam_to_root_targets(
    left_cam:         np.ndarray,
    right_cam:        np.ndarray,
    R_interp:         np.ndarray,
    t_interp:         np.ndarray,
    workspace_center: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert camera-frame wrist trajectory to per-frame root-frame IK targets."""
    T = len(left_cam)
    left_pos  = np.einsum("tij,tj->ti", R_interp.transpose(0, 2, 1),
                          left_cam[:, :3] - t_interp)
    right_pos = np.einsum("tij,tj->ti", R_interp.transpose(0, 2, 1),
                          right_cam[:, :3] - t_interp)

    left_quat  = np.zeros((T, 4))
    right_quat = np.zeros((T, 4))
    for t in range(T):
        R_inv = Rotation.from_matrix(R_interp[t].T)
        ql = Rotation.from_quat(left_cam[t,  [4, 5, 6, 3]])
        qr = Rotation.from_quat(right_cam[t, [4, 5, 6, 3]])
        left_quat[t]  = (R_inv * ql).as_quat()[[3, 0, 1, 2]]
        right_quat[t] = (R_inv * qr).as_quat()[[3, 0, 1, 2]]

    if workspace_center is not None:
        left_pos  = left_pos  - left_pos.mean(0)  + workspace_center["left"]
        right_pos = right_pos - right_pos.mean(0) + workspace_center["right"]

    return (left_pos.astype(np.float32),  left_quat.astype(np.float32),
            right_pos.astype(np.float32), right_quat.astype(np.float32))


def rescale_bilateral_separation(
    left_cam: np.ndarray,
    right_cam: np.ndarray,
    target_sep: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both wrist trajectories so their mean separation matches target_sep."""
    l_pos, r_pos = left_cam[:, :3], right_cam[:, :3]
    l_mean, r_mean = l_pos.mean(0), r_pos.mean(0)
    cur_sep = float(np.linalg.norm(l_mean - r_mean))
    if cur_sep < 1e-6:
        return left_cam.copy(), right_cam.copy()
    scale = target_sep / cur_sep
    mid   = 0.5 * (l_mean + r_mean)
    return (
        np.concatenate([mid + (l_pos - mid) * scale, left_cam[:,  3:]], 1).astype(left_cam.dtype),
        np.concatenate([mid + (r_pos - mid) * scale, right_cam[:, 3:]], 1).astype(right_cam.dtype),
    )
