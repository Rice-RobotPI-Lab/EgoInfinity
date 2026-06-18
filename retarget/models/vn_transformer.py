"""
NeuralRootFrameEstimator: SE(3)-equivariant estimation of the robot root frame
pose from bilateral wrist trajectories observed in the camera frame.

The "root frame" is the robot's root body as defined in its MJCF config
(e.g. "torso_link" for G1, "torso_frame" for Robonaut2/XLeRobot, "base" for
Franka).  The model outputs the SE(3) pose of that body expressed in camera
coordinates: R ∈ SO(3) whose columns are the root-body axes in the camera
frame, and t ∈ R³ which is the root-body origin in the camera frame.

Mode: flow matching
-------------------
Flow-matching denoiser: (trajectory, noisy SE(3) sample, tau) → velocity.
Models the full posterior p(T | trajectory), which is multi-modal because
many camera placements make the same observed motion valid.  At inference,
run an ODE integrator (e.g. 10-step Euler) from a prior sample to get a
draw from the posterior.

Equivariance guarantee
-----------------------
For any g = (R_g, t_g) in SE(3):

    f(g * X, g * T_noisy, τ) = g * f(X, T_noisy, τ)

The noisy SE(3) condition is encoded as VN channels and therefore transforms
covariantly, preserving equivariance end-to-end.

Architecture
------------
Backbone
  1. Centroid centering:   c = mean of all wrist positions across time and sides
  2. Per-frame VN encoding: (pos − c, rotmat columns) → 5 VN channels per side
  3. Input projection:      VNLinear → d_model channels per frame
  4. VN-Transformer encoder (num_layers Pre-LN blocks)
  5. Mean-pool over T                                   → (B, d_model, 3)

Flow-matching heads (noisy SE(3) + τ injected before transformer)
  5'. Encode noisy (R_noisy, t_noisy) → VN features → d_model channels
      τ → sinusoidal embedding → MLP → invariant scale
      fused = (backbone_out + noisy_enc) * tau_scale
  6a. Rot velocity:   invariant MLP on body-frame features → (B, 3) ∈ so(3)
  6b. Trans velocity: VNLinear(d_model, 1) → (B, 3)

References
----------
Deng et al., "Vector Neurons," ICCV 2021.
Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
roma library for differentiable SO(3) utilities.
"""

from __future__ import annotations

import math

import roma
import torch
import torch.nn as nn
try:
    from vn_layers import VNLayerNorm, VNLinear, VNTransformerLayer
except ImportError:
    from models.vn_layers import VNLayerNorm, VNLinear, VNTransformerLayer


# ── helpers ───────────────────────────────────────────────────────────────────

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (w,x,y,z) → 3×3 rotation matrices."""
    return roma.unitquat_to_rotmat(q[..., [1, 2, 3, 0]])  # roma uses (x,y,z,w)


def _pose7_to_vn(
    traj: torch.Tensor,           # (B, T, 7)
    centroid: torch.Tensor,       # (B, 3)
    g_cam: torch.Tensor | None,   # (B, 3) gravity in camera frame, or None
) -> torch.Tensor:
    """
    Encode a (B, T, 7) pose trajectory as VN features (B, T, 5, 3).

    Channels: [pos − centroid, R[:,0], R[:,1], R[:,2], g_cam]
    g_cam is broadcast over T. When unavailable pass None (or zero) —
    the zero vector contributes nothing through VNLinear (equivariance preserved).
    """
    pos  = traj[..., :3]                               # (B, T, 3)
    quat = traj[..., 3:]                               # (B, T, 4)
    pos_centered = pos - centroid[:, None, :]          # (B, T, 3)
    R = _quat_to_rotmat(quat)                          # (B, T, 3, 3)
    if g_cam is None:
        g = torch.zeros(traj.shape[0], 3, device=traj.device)  # (B, 3)
    else:
        g = g_cam                                      # (B, 3)
    g_expanded = g[:, None, :].expand(-1, traj.shape[1], -1)   # (B, T, 3)
    return torch.stack(
        [pos_centered, R[..., 0], R[..., 1], R[..., 2], g_expanded], dim=-2
    )                                                  # (B, T, 5, 3)


def _sinusoidal_attn_bias(T: int, num_heads: int, device: torch.device) -> torch.Tensor:
    """
    Invariant temporal attention bias: PE[t]·PE[s] / sqrt(d_pe).
    Returns (1, H, T, T), broadcastable over batch.
    """
    d_pe = 64
    pos  = torch.arange(T, device=device).float()
    dim  = torch.arange(0, d_pe, 2, device=device).float()
    freq = torch.exp(-dim * math.log(10000.0) / d_pe)
    pe   = torch.zeros(T, d_pe, device=device)
    pe[:, 0::2] = torch.sin(pos[:, None] * freq[None, :])
    pe[:, 1::2] = torch.cos(pos[:, None] * freq[None, :])
    bias = pe @ pe.T / math.sqrt(d_pe)                # (T, T)
    return bias[None, None].expand(1, num_heads, T, T)


def _sinusoidal_tau_emb(tau: torch.Tensor, d: int = 64) -> torch.Tensor:
    """
    Sinusoidal embedding for the flow time τ ∈ [0, 1].

    Parameters
    ----------
    tau : (B,)
    d   : embedding dimension (must be even)

    Returns
    -------
    (B, d)
    """
    half = d // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device) / half
    )
    emb = tau[:, None] * freq[None, :]          # (B, half)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, d)


def _se3_to_vn(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Encode an SE(3) element as 4 VN channels (B, 4, 3).

    Channels: [t, R[:,0], R[:,1], R[:,2]]
    Equivariant under the same SE(3) action as the trajectory encoding.
    """
    return torch.stack([t, R[:, :, 0], R[:, :, 1], R[:, :, 2]], dim=-2)  # (B, 4, 3)


# ── main model ────────────────────────────────────────────────────────────────

class NeuralRootFrameEstimator(nn.Module):
    """
    SE(3)-equivariant flow-matching model that estimates the robot root frame
    pose from bilateral wrist trajectories observed in the camera frame.

    Parameters
    ----------
    d_model         : VN feature width (channels per frame)
    num_heads       : attention heads (must divide d_model)
    num_layers      : number of VNTransformerLayer blocks
    dim_feedforward : FFN hidden width
    dropout         : dropout probability

    Forward
    -------
    v_R, v_t = model(left_traj, right_traj, R_noisy, t_noisy, tau)
      R_noisy : (B, 3, 3)  noisy rotation sample at flow time τ
      t_noisy : (B, 3)     noisy translation sample at flow time τ
      tau     : (B,)       flow time ∈ [0, 1]
      v_R : (B, 3)   rotation    velocity in so(3) (axis-angle tangent vector)
      v_t : (B, 3)   translation velocity

    Inference: call model.sample(left_traj, right_traj) to integrate the ODE.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads

        # ── backbone ──────────────────────────────────────────────────────
        # 5 VN channels: [pos−c, R[:,0], R[:,1], R[:,2], g_cam]
        # g_cam is zeroed when unavailable — zero contributes nothing through
        # VNLinear, preserving equivariance.
        self.left_proj  = VNLinear(5, d_model)
        self.right_proj = VNLinear(5, d_model)
        self.fuse_proj  = VNLinear(2 * d_model, d_model)

        self.layers = nn.ModuleList([
            VNTransformerLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = VNLayerNorm(d_model)

        # ── flow-matching heads ───────────────────────────────────────────
        # Encode noisy SE(3): 4 VN channels → d_model
        self.noisy_proj = VNLinear(4, d_model)

        # τ → invariant scale applied to the fused conditioning
        _d_tau = 64
        self.tau_mlp = nn.Sequential(
            nn.Linear(_d_tau, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        # Rotation velocity: invariant MLP.
        # x_body = R_tau^T @ x is invariant (Q cancels).
        # Flatten d_model*3 scalars → MLP → v_body (3 invariant scalars).
        # v_R = R_tau @ v_body is equivariant ✓.
        self.rot_vel_inv_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )
        # Translation velocity: equivariant VN head
        self.trans_vel_head = VNLinear(d_model, 1)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        left_traj:   torch.Tensor,
        right_traj:  torch.Tensor,
        R_noisy:     torch.Tensor,
        t_noisy:     torch.Tensor,
        tau:         torch.Tensor,
        g_cam:       torch.Tensor | None = None,
        left_mask:   torch.Tensor | None = None,  # (B, T) bool, True=valid
        right_mask:  torch.Tensor | None = None,  # (B, T) bool, True=valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flow-matching forward: (left_traj, right_traj, R_noisy, t_noisy, tau) → (v_R, v_t)

        g_cam : (B, 3) gravity direction in camera frame, or None.
                Pass None (or omit) when gravity is unavailable — the gravity
                channel is zeroed and the model falls back gracefully.
        """

        B, T, _ = left_traj.shape
        device  = left_traj.device

        # ── centroid ──────────────────────────────────────────────────────
        centroid = torch.cat(
            [left_traj[..., :3], right_traj[..., :3]], dim=1
        ).mean(dim=1)                                        # (B, 3)

        # ── per-frame VN encoding ─────────────────────────────────────────
        left_vn  = _pose7_to_vn(left_traj,  centroid, g_cam)  # (B, T, 5, 3)
        right_vn = _pose7_to_vn(right_traj, centroid, g_cam)  # (B, T, 5, 3)

        # ── per-hand feature masking (occluded frames → zero VN features) ─
        if left_mask is not None:
            left_vn  = left_vn  * left_mask.float().unsqueeze(-1).unsqueeze(-1)
        if right_mask is not None:
            right_vn = right_vn * right_mask.float().unsqueeze(-1).unsqueeze(-1)

        # ── key_padding_mask: True = at least one hand valid ──────────────
        key_padding_mask: torch.Tensor | None = None
        if left_mask is not None or right_mask is not None:
            lm = left_mask  if left_mask  is not None else torch.ones(B, T, dtype=torch.bool, device=device)
            rm = right_mask if right_mask is not None else torch.ones(B, T, dtype=torch.bool, device=device)
            key_padding_mask = lm | rm                           # (B, T)

        # ── project & fuse ────────────────────────────────────────────────
        x = self.fuse_proj(
            torch.cat([self.left_proj(left_vn),
                       self.right_proj(right_vn)], dim=-2)  # (B,T,2*d,3)
        )                                                    # (B, T, d_model, 3)

        # ── flow conditioning injected BEFORE transformer ─────────────────
        # Critical: noisy_enc must enter before attention layers so the
        # transformer can compute inner products between R_tau columns and
        # trajectory features — this nonlinear cross-interaction is what
        # enables learning Log(R_tau^T @ R_gt).
        noisy_vn  = _se3_to_vn(R_noisy, t_noisy - centroid)  # (B, 4, 3)
        noisy_enc = self.noisy_proj(noisy_vn)                 # (B, d_model, 3)
        tau_scale = self.tau_mlp(
            _sinusoidal_tau_emb(tau, d=64)
        ).unsqueeze(-1)                                        # (B, d_model, 1)
        x = x + (noisy_enc * tau_scale).unsqueeze(1)          # (B, T, d_model, 3)

        # ── VN-Transformer encoder ────────────────────────────────────────
        attn_bias = _sinusoidal_attn_bias(T, self.num_heads, device)
        for layer in self.layers:
            x = layer(x, attn_bias, key_padding_mask)
        x = self.final_norm(x).mean(dim=1)                  # (B, d_model, 3)

        return self._decode_flow(x, R_noisy)

    # ── decoder ───────────────────────────────────────────────────────────────

    def _decode_flow(
        self,
        x: torch.Tensor,       # (B, d_model, 3)  transformer output (already conditioned)
        R_noisy: torch.Tensor, # (B, 3, 3) R_tau — lifts body-frame velocity to camera frame
        # centroid, t_noisy, tau are not needed here: flow conditioning
        # (noisy SE(3) + tau) was already injected into x before the transformer.
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        # Rotation velocity via invariant MLP.
        # x_body = R_tau^T @ x: invariant under camera rotation Q ✓
        # MLP on flat scalars can express Log(R_tau^T @ R_gt) — fully expressive.
        # v_R = R_tau @ v_body: equivariant output ✓
        R_tau_T = R_noisy.transpose(-1, -2)                              # (B, 3, 3)
        x_body  = (R_tau_T[:, None] @ x.unsqueeze(-1)).squeeze(-1)      # (B, d_model, 3)
        x_inv   = x_body.reshape(B, -1)                                  # (B, d_model*3)
        v_body  = self.rot_vel_inv_mlp(x_inv)                            # (B, 3)
        v_R     = (R_noisy @ v_body.unsqueeze(-1)).squeeze(-1)           # (B, 3)

        # Translation velocity: equivariant VN head
        v_t = self.trans_vel_head(x).squeeze(-2)                         # (B, 3)
        return v_R, v_t

    # ── inference utilities ───────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        left_traj:  torch.Tensor,
        right_traj: torch.Tensor,
        n_steps:    int = 20,
        g_cam:      torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flow-matching inference: integrate the learned ODE from a prior sample
        to draw from p(T | trajectory).

        Prior: R_0 ~ Uniform(SO(3)), t_0 ~ N(centroid, 0.5²I).
        ODE:   dx/dτ = v_θ(x_τ, trajectory, τ)
        Integrator: Euler with n_steps steps.

        Parameters
        ----------
        left_traj, right_traj : (B, T, 7)
        n_steps               : number of Euler steps

        Returns
        -------
        R : (B, 3, 3)
        t : (B, 3)
        """
        B      = left_traj.shape[0]
        device = left_traj.device

        centroid = torch.cat(
            [left_traj[..., :3], right_traj[..., :3]], dim=1
        ).mean(dim=1)                                        # (B, 3)

        # Stochastic priors — same distributions as training
        R = roma.random_rotmat(B).to(device)                    # R_0 ~ Uniform(SO(3))
        t = centroid + 0.5 * torch.randn(B, 3, device=device)  # t_0 ~ N(centroid, 0.5²I)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((B,), i * dt, device=device)
            # Velocity prediction: network outputs (v_R, v_t) in camera frame
            v_R, v_t = self(left_traj, right_traj, R, t, tau, g_cam)

            # Euler step on SE(3):
            # v_body = R^T @ v_R  (bring equivariant velocity to body frame)
            # R ← R @ exp(dt * v_body)
            # t ← t + dt * v_t
            v_body = (R.transpose(-1, -2) @ v_R.unsqueeze(-1)).squeeze(-1)  # (B, 3)
            R = R @ roma.rotvec_to_rotmat(dt * v_body)
            t = t + dt * v_t

        return R, t
