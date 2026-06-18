"""
Training pipeline for NeuralRootFrameEstimator.

Data generation per epoch
--------------------------
1. N robot environments initialised at zero config + small joint perturbation.
2. Random-walk joint trajectories (2 s, 30 Hz → T=60 frames) generated in
   joint space, clipped to joint limits, then forward-kinematics run via
   JaxVecEnv to produce bilateral wrist trajectories in the torso/world frame.
3. Per sample, a random camera frame is drawn such that the camera's z-axis
   points toward the hand centroid, mimicking a hand-focused camera.
4. Trajectories are projected to the camera frame.

Training step
-------------
5. Camera-frame trajectories → NeuralRootFrameEstimator → flow velocity prediction.
6. Flow-matching velocity MSE loss vs. ground-truth SE(3) path.

Usage
-----
    python3 scripts/train.py
    python3 scripts/train.py --robot franka --log_dir runs/franka

Logs (TensorBoard)
------------------
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import roma
from torch.utils.tensorboard import SummaryWriter

# Disable JAX GPU memory preallocation so PyTorch can share the same GPU.
# Must be set before jax is imported.
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Suppress warp import warnings from mujoco_mjx
with contextlib.redirect_stdout(io.StringIO()):
    import jax  # noqa: F401 — imported here to ensure XLA_PYTHON_CLIENT_PREALLOCATE takes effect

from models.vn_transformer import NeuralRootFrameEstimator
from sim.vec_env_jax import JaxVecEnv, ENV_CONFIGS as _ENV_CONFIGS
from sim.traj_sampler import taskspace_traj, random_joint_traj, collect_trajectories, T


# ── camera frame sampling ─────────────────────────────────────────────────────

def sample_camera_se3(
    centroids:   torch.Tensor,   # (B, 3)  hand centroid in world frame
    behind_prob: float = 0.15,   # probability of sampling a rear-arc camera
    dist_lo:     float = 0.4,    # minimum distance from centroid [m]
    dist_hi:     float = 2.5,    # maximum distance from centroid [m]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a camera SE(3) per batch element.

    Most samples place the camera in front of the robot (azimuth ±120°).
    With probability `behind_prob` the camera is placed in the rear arc
    (azimuth 120°–180°) to expose the model to partial back views.
    Distance range is widened relative to HOCap to include close-up and
    far-away viewpoints.

    Returns
    -------
    R_cam : (B, 3, 3)  — columns = camera axes expressed in world frame
    t_cam : (B, 3)     — camera origin in world frame
    """
    B      = centroids.shape[0]
    device = centroids.device

    # ── sample spherical coordinates relative to world frame ─────────────────
    # Front arc: ±120°.  Rear arc (behind_prob): ±(120°–180°), randomly left/right.
    front_mask = torch.rand(B, device=device) >= behind_prob
    az_front   = (torch.rand(B, device=device) - 0.5) * (4.0 * math.pi / 3.0)
    az_rear    = (torch.rand(B, device=device).sign() *
                  (math.pi * 2.0 / 3.0 + torch.rand(B, device=device) * math.pi / 3.0))
    azimuth    = torch.where(front_mask, az_front, az_rear)

    # Elevation: −10° to 45°
    elev_lo, elev_hi = -math.pi / 18.0, math.pi / 4.0
    elevation = torch.rand(B, device=device) * (elev_hi - elev_lo) + elev_lo
    # Distance: widened range covers close-up and room-scale viewpoints
    distance  = torch.rand(B, device=device) * (dist_hi - dist_lo) + dist_lo

    # Direction vector from centroid to camera (unit sphere)
    cos_e = torch.cos(elevation)
    direction = torch.stack([
        cos_e * torch.cos(azimuth),   # +x  =  forward (in front of robot)
        cos_e * torch.sin(azimuth),   # ±y  =  lateral
        torch.sin(elevation),         # +z  =  up
    ], dim=-1)                                                    # (B, 3)

    t_cam = centroids + direction * distance.unsqueeze(-1)        # (B, 3)

    # ── build camera rotation: z-axis toward centroid ─────────────────────────
    z = F.normalize(centroids - t_cam, dim=-1)

    # Random x-axis perpendicular to z via Gram-Schmidt
    rand = torch.randn(B, 3, device=device)
    rand = rand - (rand * z).sum(-1, keepdim=True) * z
    x = F.normalize(rand, dim=-1)

    y = torch.linalg.cross(z, x)

    # Columns of R_cam are the camera axes expressed in world frame
    R_cam = torch.stack([x, y, z], dim=-1)                       # (B, 3, 3)
    return R_cam, t_cam


def project_to_camera(
    traj_world: torch.Tensor,  # (B, T, 7)
    R_cam:      torch.Tensor,  # (B, 3, 3)  camera-to-world rotation
    t_cam:      torch.Tensor,  # (B, 3)     camera origin in world frame
) -> torch.Tensor:
    """
    Transform a (B, T, 7) pose trajectory from world to camera frame.

    p_cam = R_cam^T @ (p_world − t_cam)
    q_cam = quat(R_cam^T) ⊗ q_world
    """
    pos_world  = traj_world[..., :3]                                 # (B, T, 3)
    quat_world = traj_world[..., 3:]                                 # (B, T, 4)

    R_inv  = R_cam.transpose(-1, -2)                                 # (B, 3, 3)
    delta  = pos_world - t_cam.unsqueeze(1)                          # (B, T, 3)
    pos_cam = (R_inv.unsqueeze(1) @ delta.unsqueeze(-1)).squeeze(-1) # (B, T, 3)

    # Rotate quaternion: q_cam = quat(R_inv) ⊗ q_world
    R_inv_q_xyzw = roma.rotmat_to_unitquat(R_inv)                    # (B, 4) x,y,z,w
    R_inv_q      = R_inv_q_xyzw[..., [3, 0, 1, 2]]                  # (B, 4) w,x,y,z
    quat_cam = _quat_mul(R_inv_q.unsqueeze(1).expand_as(quat_world), quat_world)

    return torch.cat([pos_cam, quat_cam], dim=-1)                    # (B, T, 7)


# ── loss ─────────────────────────────────────────────────────────────────────

def flow_matching_loss(
    model:      NeuralRootFrameEstimator,
    left_cam:   torch.Tensor,               # (B, T, 7)
    right_cam:  torch.Tensor,               # (B, T, 7)
    R_gt:       torch.Tensor,               # (B, 3, 3)
    t_gt:       torch.Tensor,               # (B, 3)
    rot_weight:   float = 1.0,
    trans_weight: float = 1.0,
    g_cam:      torch.Tensor | None = None, # (B, 3) or None
    left_mask:  torch.Tensor | None = None, # (B, T) bool or None
    right_mask: torch.Tensor | None = None, # (B, T) bool or None
) -> tuple[torch.Tensor, dict]:
    """
    Conditional flow matching loss on SE(3).

    Straight-line probability paths from source (I, centroid_cam) to target
    (R_gt, t_gt):

        R_τ = Exp(τ · Log(R_gt))             rotation geodesic
        t_τ = (1−τ)·centroid_cam + τ·t_gt   linear interpolation

    The constant target velocity field is:
        v_R = Log(R_gt)   [rad]   axis-angle tangent vector
        v_t = t_gt − centroid_cam  [m]

    tau is resampled every call so that gradient steps reusing the same
    (left_cam, right_cam, R_gt, t_gt) batch still see diverse flow times.
    """
    B      = R_gt.shape[0]
    device = R_gt.device
    # tau resampled every grad step — critical when batch is reused across steps
    tau = torch.rand(B, device=device)

    centroid_cam = torch.cat(
        [left_cam[..., :3], right_cam[..., :3]], dim=1
    ).mean(dim=1)                                                    # (B, 3)

    # Stochastic priors — detached from graph (not learned, no gradients needed)
    #   R_0 ~ Uniform(SO(3))
    #   t_0 ~ N(centroid_cam, 0.5² I)
    with torch.no_grad():
        R_0 = roma.random_rotmat(B).to(device)                               # (B, 3, 3)
        t_0 = centroid_cam + 0.5 * torch.randn(B, 3, device=device)         # (B, 3)
        log_R_rel = roma.rotmat_to_rotvec(R_0.transpose(-1, -2) @ R_gt)     # (B, 3)
        R_tau = R_0 @ roma.rotvec_to_rotmat(tau.unsqueeze(-1) * log_R_rel)  # (B, 3, 3)
        t_tau = (1 - tau).unsqueeze(-1) * t_0 + tau.unsqueeze(-1) * t_gt   # (B, 3)
        # Velocity targets — constant along the straight-line path
        # v_R_gt = R_tau @ Log(R_0^T @ R_gt)  (equivariant lift of body-frame velocity)
        # v_t_gt = t_gt - t_0                  (constant for linear interpolation)
        v_R_gt = (R_tau @ log_R_rel.unsqueeze(-1)).squeeze(-1)              # (B, 3)
        v_t_gt = t_gt - t_0                                                 # (B, 3)

    # Velocity prediction: network outputs (v_R, v_t), targets are constant
    # along the straight-line path from (R_0, t_0) to (R_gt, t_gt).
    v_R, v_t = model(left_cam, right_cam, R_tau, t_tau, tau, g_cam=g_cam,
                     left_mask=left_mask, right_mask=right_mask)

    loss_R = rot_weight   * (v_R - v_R_gt).pow(2).mean()
    loss_t = trans_weight * (v_t - v_t_gt).pow(2).mean()
    total  = loss_R + loss_t
    return total, {"rot_vel": loss_R.item(), "trans_vel": loss_t.item()}


# ── training ──────────────────────────────────────────────────────────────────

def sample_batch(
    env:    JaxVecEnv,
    device: torch.device,
    rng:    np.random.Generator,
    args:   argparse.Namespace,
) -> tuple:
    """
    Sample one batch of trajectories and project to camera frame.
    Returns (left_cam, right_cam, R_gt, t_gt) ready for the model.
    Separated from the grad step so the same batch can be reused for
    steps_per_epoch gradient updates.
    """
    limits_l = np.array(env._joint_limits["left"])
    limits_r = np.array(env._joint_limits["right"])

    _start_l = args._start_config_l
    _start_r = args._start_config_r

    _sc = args._sample_cfg
    if args.sample_mode == "taskspace":
        left_q_np  = taskspace_traj(args.num_envs, "left",  rng, args._ik_left,
                                    _sc, q_ref=_start_l)
        right_q_np = taskspace_traj(args.num_envs, "right", rng, args._ik_right,
                                    _sc, q_ref=_start_r)
    else:
        base_l = np.tile(_start_l, (args.num_envs, 1))
        base_r = np.tile(_start_r, (args.num_envs, 1))
        left_q_np  = random_joint_traj(args.num_envs, limits_l, rng, _sc,
                                       side="left",  shared_base=base_l)
        right_q_np = random_joint_traj(args.num_envs, limits_r, rng, _sc,
                                       side="right", shared_base=base_r)

    left_world, right_world = collect_trajectories(env, left_q_np, right_q_np)
    left_world  = left_world.to(device)
    right_world = right_world.to(device)

    # reject arm-crossing samples
    no_cross = (left_world[..., 1] > right_world[..., 1]).all(dim=1)
    if not no_cross.all():
        valid = no_cross.nonzero(as_tuple=True)[0]
        if len(valid) > 0:
            fill = valid[torch.arange(args.num_envs, device=device) % len(valid)]
            left_world  = left_world[fill]
            right_world = right_world[fill]

    centroid_world = torch.cat(
        [left_world[..., :3], right_world[..., :3]], dim=1
    ).mean(dim=1)
    R_cam, t_cam = sample_camera_se3(
        centroid_world,
        behind_prob=args.cam_behind_prob,
        dist_lo=args.cam_dist_lo,
        dist_hi=args.cam_dist_hi,
    )

    left_cam  = project_to_camera(left_world,  R_cam, t_cam)
    right_cam = project_to_camera(right_world, R_cam, t_cam)

    R_cam_T = R_cam.transpose(-1, -2)
    R_gt = R_cam_T @ args._R_torso_world.to(device).unsqueeze(0)
    t_gt = (R_cam_T @ (args._torso_pos_world.to(device) - t_cam).unsqueeze(-1)).squeeze(-1)

    # Gravity in camera frame: g_world = [0,0,-1], g_cam = R_cam^T @ g_world
    g_world = torch.tensor([0., 0., -1.], device=device).unsqueeze(0)  # (1, 3)
    g_cam   = (R_cam_T @ g_world.unsqueeze(-1)).squeeze(-1)            # (B, 3)

    # ── augmentations ─────────────────────────────────────────────────────────

    # 1. Tracking noise: perturb wrist positions and orientations
    if args.track_pos_noise > 0:
        left_cam[..., :3]  += torch.randn_like(left_cam[..., :3])  * args.track_pos_noise
        right_cam[..., :3] += torch.randn_like(right_cam[..., :3]) * args.track_pos_noise
    if args.track_ori_noise > 0:
        for traj in (left_cam, right_cam):
            B_, T_, _ = traj.shape
            axes  = F.normalize(torch.randn(B_, T_, 3, device=device), dim=-1)
            angle = torch.rand(B_, T_, 1, device=device) * args.track_ori_noise
            # Rodrigues: q_perturb = [cos(a/2), sin(a/2)*axis]
            half  = angle * 0.5
            dq    = torch.cat([torch.cos(half), torch.sin(half) * axes], dim=-1)  # (B,T,4) wxyz
            traj[..., 3:] = _quat_mul(dq, traj[..., 3:])

    # 2. Tracking jumps: for a short random block, replace one hand's pose with
    #    a large random offset — simulates tracker snapping to a wrong detection.
    if args.hand_jump_prob > 0:
        B_ = left_cam.shape[0]
        T_ = left_cam.shape[1]
        jump_mask = torch.rand(B_, device=device) < args.hand_jump_prob
        if jump_mask.any():
            jump_idx  = jump_mask.nonzero(as_tuple=True)[0]
            side      = torch.randint(0, 2, (len(jump_idx),), device=device)
            max_len   = max(1, T_ // 4)
            jmp_lens  = torch.randint(1, max_len + 1, (len(jump_idx),))
            jmp_strt  = torch.randint(0, T_, (len(jump_idx),))
            for i, b in enumerate(jump_idx):
                s = int(jmp_strt[i].item())
                e = min(s + int(jmp_lens[i].item()), T_)
                traj = left_cam if side[i] == 0 else right_cam
                # Large position offset (uniform in a sphere of radius jump_mag)
                offset = F.normalize(torch.randn(3, device=device), dim=-1) \
                         * (torch.rand(1, device=device) * args.hand_jump_mag)
                traj[b, s:e, :3] = traj[b, s:e, :3] + offset
                # Large orientation perturbation (up to π radians)
                axis  = F.normalize(torch.randn(3, device=device), dim=-1)
                angle = torch.rand(1, device=device) * math.pi
                half  = angle * 0.5
                dq    = torch.cat([torch.cos(half), torch.sin(half) * axis]).unsqueeze(0)
                traj[b, s:e, 3:] = _quat_mul(dq.expand(e - s, -1), traj[b, s:e, 3:])

    # 3. Hand occlusion: mask out a contiguous block for one arm and zero features
    B_ = left_cam.shape[0]
    T_ = left_cam.shape[1]
    left_mask  = torch.ones(B_, T_, dtype=torch.bool, device=device)
    right_mask = torch.ones(B_, T_, dtype=torch.bool, device=device)
    if args.hand_occlusion_prob > 0:
        occ_mask = torch.rand(B_, device=device) < args.hand_occlusion_prob
        if occ_mask.any():
            occ_idx  = occ_mask.nonzero(as_tuple=True)[0]
            side     = torch.randint(0, 2, (len(occ_idx),), device=device)
            max_len  = max(1, T_ // 2)
            occ_lens = torch.randint(1, max_len + 1, (len(occ_idx),))
            occ_strt = torch.randint(0, T_, (len(occ_idx),))
            for i, b in enumerate(occ_idx):
                s = int(occ_strt[i].item())
                e = min(s + int(occ_lens[i].item()), T_)
                if side[i] == 0:
                    left_mask[b, s:e]  = False
                    left_cam[b, s:e]   = 0.0
                else:
                    right_mask[b, s:e] = False
                    right_cam[b, s:e]  = 0.0

    # 3. Gravity noise: rotate g_cam by a small random angle around a random
    #    axis perpendicular to gravity — models IMU/estimation noise.
    if args.gravity_noise > 0:
        angle   = torch.rand(B_, device=device) * args.gravity_noise  # (B,)
        rand_ax = torch.randn(B_, 3, device=device)
        rand_ax = rand_ax - (rand_ax * g_cam).sum(-1, keepdim=True) * g_cam
        rand_ax = F.normalize(rand_ax, dim=-1)
        c = torch.cos(angle).unsqueeze(-1)
        s = torch.sin(angle).unsqueeze(-1)
        g_cam = (g_cam * c
                 + torch.linalg.cross(rand_ax, g_cam) * s
                 + rand_ax * (rand_ax * g_cam).sum(-1, keepdim=True) * (1 - c))
        g_cam = F.normalize(g_cam, dim=-1)

    # Gravity dropout: zero out gravity for ~30% of samples
    if args.gravity_dropout > 0:
        mask  = (torch.rand(g_cam.shape[0], device=device) > args.gravity_dropout)
        g_cam = g_cam * mask.unsqueeze(-1).float()

    return left_cam, right_cam, R_gt, t_gt, g_cam, left_mask, right_mask


def train_step(
    model:      NeuralRootFrameEstimator,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    args:       argparse.Namespace,
    left_cam:   torch.Tensor,
    right_cam:  torch.Tensor,
    R_gt:       torch.Tensor,
    t_gt:       torch.Tensor,
    g_cam:      torch.Tensor | None = None,
    left_mask:  torch.Tensor | None = None,
    right_mask: torch.Tensor | None = None,
) -> tuple[dict, float]:
    """One gradient step on a pre-sampled batch."""
    model.train()
    optimizer.zero_grad()

    loss, metrics = flow_matching_loss(model, left_cam, right_cam, R_gt, t_gt,
                                       args.rot_weight, args.trans_weight,
                                       g_cam=g_cam,
                                       left_mask=left_mask, right_mask=right_mask)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
    ).item()
    optimizer.step()

    metrics["total"] = loss.item()
    return metrics, grad_norm


def train_epoch(
    model:     NeuralRootFrameEstimator,
    env:       JaxVecEnv,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    rng:       np.random.Generator,
    args:      argparse.Namespace,
    epoch:     int = 0,
) -> tuple[dict, float]:
    """
    Sample one batch, then run steps_per_epoch gradient steps on it.
    steps_per_epoch controls both gradient update count and batch reuse — no
    extra parameter needed.
    """
    from tqdm import tqdm
    # sample once per epoch
    left_cam, right_cam, R_gt, t_gt, g_cam, left_mask, right_mask = \
        sample_batch(env, device, rng, args)

    sum_metrics: dict = {}
    sum_grad_norm = 0.0
    bar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch:4d}",
               leave=False, dynamic_ncols=True)
    for _ in bar:
        m, gn = train_step(model, optimizer, device, args,
                           left_cam, right_cam, R_gt, t_gt, g_cam,
                           left_mask, right_mask)
        for k, v in m.items():
            sum_metrics[k] = sum_metrics.get(k, 0.0) + v
        sum_grad_norm += gn
        n = bar.n or 1
        bar.set_postfix({k: f"{v/n:.4f}" for k, v in sum_metrics.items()})
    return {k: v / args.steps_per_epoch for k, v in sum_metrics.items()}, sum_grad_norm / args.steps_per_epoch


def _saveable_args(args: argparse.Namespace) -> dict:
    """Extract only JSON-serialisable CLI args — excludes runtime objects (_*)."""
    return {k: v for k, v in vars(args).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.log_dir is None:
        args.log_dir = f"runs/{args.robot}/taskspace" \
                       if args.sample_mode == "taskspace" \
                       else f"runs/{args.robot}"

    # ── per-robot config ──────────────────────────────────────────────────
    from sim.robots import ROBOT_CONFIGS as _RC, SAMPLE_CONFIGS as _SC
    robot_cfg  = _RC[args.robot]
    env_config = _ENV_CONFIGS[args.robot]

    args._start_config_l = np.asarray(robot_cfg["start_config"]["left"],  dtype=np.float32)
    args._start_config_r = np.asarray(robot_cfg["start_config"]["right"], dtype=np.float32)
    args._sample_cfg     = _SC[args.robot]

    # ── sim environment ───────────────────────────────────────────────────
    print(f"Robot: {args.robot}  |  Initialising JaxVecEnv with {args.num_envs} envs …")
    env = JaxVecEnv(env_config, num_envs=args.num_envs)
    env.warmup()
    print("JIT compilation done.")

    # torso body pose — fixed after reset (robot base never moves during training)
    import mujoco as _mj
    _model = _mj.MjModel.from_xml_path(str(env_config.mjcf_path))
    _data  = _mj.MjData(_model)
    _mj.mj_resetData(_model, _data)
    _mj.mj_forward(_model, _data)
    _torso_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, robot_cfg["torso_body"])
    args._torso_pos_world = torch.tensor(
        _data.xpos[_torso_id].copy(), dtype=torch.float32
    )
    _R_torso = np.zeros((3, 3))
    _mj.mju_quat2Mat(_R_torso.ravel(), _data.xquat[_torso_id].copy())
    args._R_torso_world = torch.tensor(_R_torso, dtype=torch.float32)
    print(f"{robot_cfg['torso_body']} world pos:  {args._torso_pos_world.numpy().round(4)}")
    print(f"{robot_cfg['torso_body']} world R:\n{args._R_torso_world.numpy().round(4)}")

    # IK solvers — only needed for task-space sampling
    if args.sample_mode == "taskspace":
        from kinematics.wrist_ik import WristIK, RobotIKConfig
        print("Building WristIK solvers for task-space sampling…")
        ik_robot = robot_cfg["ik_robot"]
        if ik_robot is None:
            args._ik_left  = WristIK(side="left",  ori_weight=0.0, max_iter=50, tol_pos=0.01)
            args._ik_right = WristIK(side="right", ori_weight=0.0, max_iter=50, tol_pos=0.01)
        else:
            ik_cfg_l = getattr(RobotIKConfig, ik_robot)("left")
            ik_cfg_r = getattr(RobotIKConfig, ik_robot)("right")
            args._ik_left  = WristIK(side="left",  robot=ik_cfg_l, ori_weight=0.0, max_iter=50, tol_pos=0.01)
            args._ik_right = WristIK(side="right", robot=ik_cfg_r, ori_weight=0.0, max_iter=50, tol_pos=0.01)
        print("IK solvers ready.")

    # ── model + optimiser ─────────────────────────────────────────────────
    if args.dim_feedforward is None:
        args.dim_feedforward = 4 * args.d_model

    # ── resume from checkpoint ────────────────────────────────────────────
    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_ckpt = Path(args.resume)
    else:
        # Auto-resume: find the latest checkpoint in the log dir
        ckpt_dir_probe = Path(args.log_dir) / "checkpoints"
        if ckpt_dir_probe.exists():
            existing = sorted(ckpt_dir_probe.glob("epoch_*.pt"))
            if existing:
                resume_ckpt = existing[-1]
                print(f"Auto-resuming from {resume_ckpt}")

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        saved = ckpt.get("args", {})
        # Use saved model architecture args
        args.d_model        = saved.get("d_model",        args.d_model)
        args.num_heads      = saved.get("num_heads",      args.num_heads)
        args.num_layers     = saved.get("num_layers",     args.num_layers)
        args.dim_feedforward= saved.get("dim_feedforward",args.dim_feedforward)
        args.dropout        = saved.get("dropout",        args.dropout)
        start_epoch         = ckpt["epoch"] + 1
        print(f"Resuming from epoch {ckpt['epoch']}  →  training epochs {start_epoch}–{args.epochs}")

    model = NeuralRootFrameEstimator(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    if resume_ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Loaded weights + optimizer + scheduler from {resume_ckpt}")

    # ── tensorboard + checkpoint dir ──────────────────────────────────────
    log_dir  = Path(args.log_dir)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Log hparams so they appear in the HParams tab
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(hparams, metric_dict={})

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        metrics, grad_norm = train_epoch(model, env, optimizer, device, rng, args, epoch=epoch)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # TensorBoard: one scalar per metric + learning rate + grad norm
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/grad_norm", grad_norm, epoch)

        if epoch % args.log_every == 0:
            parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"epoch {epoch:4d}/{args.epochs}  {parts}  grad={grad_norm:.2f}  lr={lr:.2e}", flush=True)

        if epoch % args.save_every == 0:
            ckpt = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "args":       _saveable_args(args),
            }, ckpt)
            print(f"  → saved {ckpt}")

    # Final checkpoint
    torch.save({
        "epoch":     args.epochs,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args":      _saveable_args(args),
    }, ckpt_dir / "final.pt")
    writer.close()
    print(f"Training complete. Logs: {log_dir}  |  Final ckpt: {ckpt_dir / 'final.pt'}")


# ── quaternion helper ─────────────────────────────────────────────────────────

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Train NeuralRootFrameEstimator")

    # Robot
    p.add_argument("--robot", default="g1", choices=["g1", "franka", "robonaut2", "xlerobot"],
                   help="Which robot to train on (default: g1)")

    # Data
    p.add_argument("--num_envs",    type=int, default=1024)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--sample_mode", default="taskspace",
                   choices=["jointspace", "taskspace"],
                   help="jointspace: random joint angles → FK  |  "
                        "taskspace: random Cartesian waypoints → IK → FK (default)")

    # Model
    p.add_argument("--d_model",         type=int, default=128)
    p.add_argument("--num_heads",       type=int, default=4)
    p.add_argument("--num_layers",      type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=None,
                   help="FFN hidden width (default: 4 × d_model)")
    p.add_argument("--dropout",         type=float, default=0.1)

    # Loss
    p.add_argument("--rot_weight",       type=float, default=1.0)
    p.add_argument("--trans_weight",     type=float, default=1.0)
    p.add_argument("--track_pos_noise",   type=float, default=0.01,
                   help="Std of Gaussian position noise on wrist poses [m] (0=off)")
    p.add_argument("--track_ori_noise",   type=float, default=0.05,
                   help="Max orientation perturbation on wrist poses [rad] (0=off)")
    p.add_argument("--hand_jump_prob",      type=float, default=0.2,
                   help="Probability of a tracking jump event for one arm (0=off)")
    p.add_argument("--hand_jump_mag",       type=float, default=0.15,
                   help="Max position offset for tracking jump [m] (default 0.15)")
    p.add_argument("--hand_occlusion_prob", type=float, default=0.2,
                   help="Probability of randomly occluding one arm for a block of frames (0=off)")
    p.add_argument("--gravity_noise",     type=float, default=0.1,
                   help="Max gravity direction perturbation [rad] (0=off)")
    p.add_argument("--cam_behind_prob",   type=float, default=0.15,
                   help="Probability of placing camera in rear arc (0=front-only)")
    p.add_argument("--cam_dist_lo",       type=float, default=0.4,
                   help="Minimum camera distance from centroid [m]")
    p.add_argument("--cam_dist_hi",       type=float, default=2.5,
                   help="Maximum camera distance from centroid [m]")
    p.add_argument("--gravity_dropout",  type=float, default=0.3,
                   help="Fraction of samples where gravity channel is zeroed "
                        "(0 = always provide gravity, 1 = never provide)")

    # Optimiser
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs",    type=int,   default=500)

    # Steps per epoch
    p.add_argument("--steps_per_epoch", type=int, default=20,
                   help="Gradient steps per epoch; each step uses a fresh sim batch")

    # Logging / checkpoints
    p.add_argument("--log_every",  type=int, default=1)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_dir",    default=None,
                   help="Checkpoint/log directory. Defaults to runs/{robot}/{mode}")
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from. "
                        "If omitted, auto-resumes from the latest checkpoint in log_dir.")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
