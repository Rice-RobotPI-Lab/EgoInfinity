"""Stage 4: physics-aware test-time optimisation of object pose sequence.

7-loss LBFGS over the entire clip's T_seq, refining the per-frame mesh-to-
world transform.  All losses are soft / continuous (no binary trust gate)
because Stage 3 showed binary trust is too sparse on real data.

Loss summary (lambda weights given):

    L_fit (100)       — mesh ↔ observation Chamfer, IoU-weighted per frame
    L_anchor (50)     — drift penalty against initial T_seq, IoU-weighted
    L_temporal (1)    — frame-to-frame SE(3) smoothness
    L_static (50)     — non-contact-frame velocity ≈ 0  (table-top key)
    L_pen (5)         — hand vertices outside object mesh (hinge)
    L_prox (5)        — palm vertices near object surface during contact (hinge)
    L_noslip (10)     — hand vertices stationary in object's local frame

Pose parameterisation: per-frame (quaternion ∈ R^4, translation ∈ R^3).
Quaternions are renormalised every step.  LBFGS with strong-Wolfe line search.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("pose_tracker.fill_optimize")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LAMBDAS = dict(
    fit=30.0,         # was 100 — too greedy: pulled mesh to per-frame noisy depth
    anchor=50.0,
    temporal=30.0,    # explicit smoothness; uses L1 on velocity
    static=200.0,     # non-contact static is the strongest table-top prior
    pen=5.0,
    prox=100.0,       # was 5 — L1 hinge now; need to close 5cm centroid offset (knife)
    noslip=50.0,      # was 10 — once close, drives rotation following hand
)
DEFAULT_DELTA_PEN_M = 0.002    # 2 mm — penetration safety margin
DEFAULT_D0_PROX_M = 0.005      # 5 mm — desired contact gap


# ---------------------------------------------------------------------------
# Quaternion utilities (PyTorch)
# ---------------------------------------------------------------------------
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (4,) [w, x, y, z]."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                          (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                          (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if R[1,1] > R[2,2]:
        s = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                          0.25*s, (R[1,2]+R[2,1])/s])
    s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                      (R[1,2]+R[2,1])/s, 0.25*s])


def _quat_to_mat_torch(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) [w, x, y, z] -> (N, 3, 3) — assumes already normalised."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ], dim=-1),
        torch.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)    ], dim=-1),
        torch.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=-2)
    return R


def _quat_dist_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """1 - |q1 . q2|, antipodal-aware quaternion distance."""
    return 1 - (q1 * q2).sum(dim=-1).abs()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class OptimizeResult:
    T_seq: np.ndarray              # (T, 4, 4) optimised
    L_history: dict                # per-loss-component history
    n_iter: int


def optimize_pose_seq(
    T_seq_init: Sequence[np.ndarray],
    mesh_pts: np.ndarray,
    obs_clouds: Sequence[Optional[np.ndarray]],
    iou_per_frame: Sequence[float],
    hand_verts_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    mano_palm_indices: np.ndarray,
    contact_soft: np.ndarray,                     # (T, 2)
    *,
    lambdas: dict = None,
    n_iter: int = 50,
    n_mesh_sample: int = 800,
    n_hand_sample_pen: int = 256,
    delta_pen_m: float = DEFAULT_DELTA_PEN_M,
    d0_prox_m: float = DEFAULT_D0_PROX_M,
    device: str = 'cuda',
    verbose: bool = True,
) -> OptimizeResult:
    """Run LBFGS optimisation on T_seq.

    Notes on inputs
    ---------------
    - ``mesh_pts`` already has ``ares.scale_correction`` applied; canonical
      frame, no further scaling done here.
    - ``obs_clouds[t]`` may be None (mask empty / depth invalid).  Skip those
      frames in L_fit.
    - ``hand_verts_per_frame[t]`` is a list of (V, 3) per detected hand,
      world frame.  Length 0/1/2.  Lists empty when no hand.
    - ``contact_soft[t, 0]`` = left, ``[t, 1]`` = right.  In [0, 1].
    """
    lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    T = len(T_seq_init)
    M_mesh = len(mesh_pts)

    # ---- Bake initial poses into quat + trans tensors ----
    q_init_np = np.stack([_mat_to_quat(np.asarray(Tt[:3, :3], dtype=np.float64))
                           for Tt in T_seq_init]).astype(np.float32)
    t_init_np = np.stack([np.asarray(Tt[:3, 3], dtype=np.float32)
                           for Tt in T_seq_init])
    q_init = torch.tensor(q_init_np, device=device)
    t_init = torch.tensor(t_init_np, device=device)
    quats = q_init.clone().requires_grad_(True)
    trans = t_init.clone().requires_grad_(True)

    # ---- Subsample mesh once to keep Chamfer cost bounded ----
    if M_mesh > n_mesh_sample:
        rng = np.random.default_rng(0)
        idx_m = rng.choice(M_mesh, n_mesh_sample, replace=False)
        mesh_sub_np = mesh_pts[idx_m].astype(np.float32)
    else:
        mesh_sub_np = mesh_pts.astype(np.float32)
    mesh_sub = torch.tensor(mesh_sub_np, device=device)        # (Ms, 3)
    Ms = mesh_sub.shape[0]

    # ---- Pre-stage per-frame observation clouds (variable length) ----
    obs_clouds_t = []
    for c in obs_clouds:
        if c is None or len(c) < 20:
            obs_clouds_t.append(None)
        else:
            # Subsample observation if huge
            cc = np.asarray(c, dtype=np.float32)
            if len(cc) > 2000:
                rng = np.random.default_rng(0)
                cc = cc[rng.choice(len(cc), 2000, replace=False)]
            obs_clouds_t.append(torch.tensor(cc, device=device))

    iou_w = torch.tensor(
        [max(float(iou_per_frame[t]) if np.isfinite(iou_per_frame[t]) else 0, 0.05)
         for t in range(T)],
        device=device, dtype=torch.float32)

    # ---- Pre-stage per-frame hand vertex tensors (sub-sampled) ----
    # We use a single concatenated hand-verts tensor per frame if multiple
    # hands present; track an index that maps back per-hand for L_noslip.
    hand_verts_t_pen = [None] * T          # full-hand sample for L_pen
    hand_palm_t_left = [None] * T          # palm subset, left hand
    hand_palm_t_right = [None] * T         # palm subset, right hand
    hand_full_t_left = [None] * T          # full-hand for L_noslip, left
    hand_full_t_right = [None] * T         # ditto, right

    for t in range(T):
        verts_list = hand_verts_per_frame[t] or []
        is_right_list = hand_is_right_per_frame[t] or []
        if not verts_list:
            continue
        full_concat_list = []
        for h_verts, is_right in zip(verts_list, is_right_list):
            if h_verts is None or len(h_verts) == 0:
                continue
            hv = np.asarray(h_verts, dtype=np.float32)
            full_concat_list.append(hv)
            # palm subset
            if hv.shape[0] >= mano_palm_indices.max() + 1:
                palm = hv[mano_palm_indices]
            else:
                palm = hv  # fallback, full hand
            palm_t = torch.tensor(palm, device=device)
            full_t = torch.tensor(hv, device=device)
            if is_right:
                hand_palm_t_right[t] = palm_t
                hand_full_t_right[t] = full_t
            else:
                hand_palm_t_left[t] = palm_t
                hand_full_t_left[t] = full_t
        if full_concat_list:
            concat = np.concatenate(full_concat_list, axis=0)
            if len(concat) > n_hand_sample_pen:
                rng = np.random.default_rng(t)
                concat = concat[rng.choice(len(concat), n_hand_sample_pen, replace=False)]
            hand_verts_t_pen[t] = torch.tensor(concat, device=device)

    contact_t = torch.tensor(contact_soft, device=device, dtype=torch.float32)
    # max over 2 hands gives "any-hand contact" weight per frame
    contact_any = contact_t.max(dim=1).values                  # (T,)
    non_contact = 1.0 - contact_any                            # (T,)

    # ---- Loss closure ----
    L_history = {k: [] for k in lambdas.keys()}
    L_history['total'] = []

    def closure():
        optimizer.zero_grad()

        q_norm = quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Rs = _quat_to_mat_torch(q_norm)                        # (T, 3, 3)

        # mesh in world per frame: (T, Ms, 3)
        mesh_world = torch.einsum('tij,mj->tmi', Rs, mesh_sub) + trans[:, None, :]

        # =========== L_fit: weighted partial Chamfer mesh -> obs ===========
        L_fit = trans.new_zeros(())
        total_w = trans.new_zeros(())
        for t in range(T):
            obs = obs_clouds_t[t]
            if obs is None:
                continue
            d = torch.cdist(mesh_world[t], obs)                # (Ms, Ko_t)
            min_d, _ = d.min(dim=1)
            n_keep = max(min_d.shape[0] // 2, 8)
            sorted_d, _ = min_d.sort()
            partial = sorted_d[:n_keep].mean()
            L_fit = L_fit + iou_w[t] * partial
            total_w = total_w + iou_w[t]
        L_fit = L_fit / total_w.clamp(min=1e-6)

        # =========== L_anchor: stay close to init, IoU-weighted ===========
        L_anc_t = ((trans - t_init) ** 2).sum(dim=-1) * iou_w
        L_anc_q = _quat_dist_torch(q_norm, q_init) * iou_w
        L_anchor = L_anc_t.mean() + L_anc_q.mean()

        # =========== L_temporal: frame-to-frame smoothness (L1) ============
        # L1 (Huber-like) is closer in magnitude to L_fit (~cm) so the
        # weight ratio is meaningful.  L2 made temporal vanish under depth jitter.
        dt = trans[1:] - trans[:-1]
        L_temp_t = dt.norm(dim=-1).mean()                      # L1: mean ‖Δt‖
        L_temp_q = _quat_dist_torch(q_norm[1:], q_norm[:-1]).mean()
        L_temporal = L_temp_t + L_temp_q

        # =========== L_static: non-contact frame velocity ≈ 0 (L1) ============
        nc_pair = non_contact[:-1] * non_contact[1:]           # (T-1,)
        L_static = (dt.norm(dim=-1) * nc_pair).mean()
        L_static = L_static + (
            _quat_dist_torch(q_norm[1:], q_norm[:-1]) * nc_pair
        ).mean()

        # =========== L_pen: hand verts outside object (hinge) ============
        L_pen = trans.new_zeros(())
        n_pen = 0
        for t in range(T):
            hv = hand_verts_t_pen[t]
            if hv is None:
                continue
            d = torch.cdist(hv, mesh_world[t]).min(dim=1).values   # (V_sub,)
            # penalise when distance < delta_pen
            pen = F.relu(delta_pen_m - d) ** 2
            L_pen = L_pen + pen.mean()
            n_pen += 1
        L_pen = L_pen / max(n_pen, 1)

        # =========== L_prox: palm verts inside [0, d_0] of mesh, when contact ============
        # L1 hinge — squared was too weak when palm was 5cm away (loss tiny).
        # Linear gradient pulls just as hard at 5cm as at 1cm.
        L_prox = trans.new_zeros(())
        n_prox = 0
        for t in range(T):
            for col, palm_t in [(0, hand_palm_t_left[t]),
                                 (1, hand_palm_t_right[t])]:
                if palm_t is None:
                    continue
                c_w = contact_t[t, col]
                if c_w < 0.05:
                    continue
                d = torch.cdist(palm_t, mesh_world[t]).min(dim=1).values
                # penalise distance > d_0 (palm too far from object) — L1
                prox = F.relu(d - d0_prox_m)
                L_prox = L_prox + c_w * prox.mean()
                n_prox += 1
        L_prox = L_prox / max(n_prox, 1)

        # =========== L_noslip: hand verts in object frame should not move ============
        L_noslip = trans.new_zeros(())
        n_ns = 0
        for k in (1, 2, 3):
            for t in range(T - k):
                # use whichever hand has stronger contact
                left_w = min(contact_t[t, 0].item(), contact_t[t + k, 0].item())
                right_w = min(contact_t[t, 1].item(), contact_t[t + k, 1].item())
                if left_w < 0.05 and right_w < 0.05:
                    continue
                if left_w >= right_w:
                    h0, h1, c_w = hand_full_t_left[t], hand_full_t_left[t + k], left_w
                else:
                    h0, h1, c_w = hand_full_t_right[t], hand_full_t_right[t + k], right_w
                if h0 is None or h1 is None:
                    continue
                # transform h0 to obj-local using (R_t, t_t); h1 via (R_{t+k}, t_{t+k})
                R_t, R_tk = Rs[t], Rs[t + k]
                tt, tk = trans[t], trans[t + k]
                # obj-local: R^T @ (h - t)
                h0_obj = (h0 - tt) @ R_t                         # (V, 3)
                h1_obj = (h1 - tk) @ R_tk
                # Soft per-vertex contact weight.  Bandwidth widened 1cm → 5cm
                # so vertices that *should* be touching but aren't yet (knife
                # at 5cm offset before L_prox closes the gap) still get
                # meaningful gradient on rotation.
                with torch.no_grad():
                    d_to_mesh = torch.cdist(h0_obj, mesh_sub).min(dim=1).values
                    w_v = torch.exp(-d_to_mesh / 0.05)
                slip = ((h1_obj - h0_obj) ** 2).sum(dim=-1) * w_v
                L_noslip = L_noslip + c_w * slip.mean() / k
                n_ns += 1
        L_noslip = L_noslip / max(n_ns, 1)

        # ---- Combine ----
        L_total = (
            lambdas['fit']     * L_fit
          + lambdas['anchor']  * L_anchor
          + lambdas['temporal']* L_temporal
          + lambdas['static']  * L_static
          + lambdas['pen']     * L_pen
          + lambdas['prox']    * L_prox
          + lambdas['noslip']  * L_noslip
        )
        L_total.backward()

        L_history['fit'].append(float(L_fit))
        L_history['anchor'].append(float(L_anchor))
        L_history['temporal'].append(float(L_temporal))
        L_history['static'].append(float(L_static))
        L_history['pen'].append(float(L_pen))
        L_history['prox'].append(float(L_prox))
        L_history['noslip'].append(float(L_noslip))
        L_history['total'].append(float(L_total))
        return L_total

    optimizer = torch.optim.LBFGS(
        [quats, trans],
        lr=0.2,
        max_iter=n_iter,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        line_search_fn='strong_wolfe',
    )
    optimizer.step(closure)

    # Repack to numpy (T, 4, 4)
    with torch.no_grad():
        q_n = (quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-8))
        Rs_n = _quat_to_mat_torch(q_n).cpu().numpy()
        trans_n = trans.detach().cpu().numpy()
    T_out = np.zeros((T, 4, 4), dtype=np.float64)
    T_out[:, 3, 3] = 1
    T_out[:, :3, :3] = Rs_n
    T_out[:, :3, 3] = trans_n

    n_done = len(L_history['total'])
    if verbose and n_done > 0:
        first, last = L_history['total'][0], L_history['total'][-1]
        log.info(
            f"opt done: total {first:.4g} → {last:.4g} "
            f"({n_done} closures); "
            f"final fit={L_history['fit'][-1]:.4g}, "
            f"static={L_history['static'][-1]:.4g}, "
            f"pen={L_history['pen'][-1]:.4g}, "
            f"prox={L_history['prox'][-1]:.4g}, "
            f"noslip={L_history['noslip'][-1]:.4g}"
        )

    return OptimizeResult(T_seq=T_out, L_history=L_history, n_iter=n_done)
