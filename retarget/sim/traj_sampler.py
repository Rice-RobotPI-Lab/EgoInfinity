"""
Trajectory samplers for bilateral arm training data generation.

Per-robot sampling parameters live in sim/robots/<name>/sample_config.py.
Pass the robot's SAMPLE_CONFIG dict to taskspace_traj() or random_joint_traj().
"""
from __future__ import annotations

import numpy as np

FPS       = 30
TRAJ_SECS = 2
T         = FPS * TRAJ_SECS  # 60 time steps per clip

N_CTRL_TS  = 7     # number of Cartesian knots in the OU walk (taskspace_traj)
INIT_NOISE = 0.05  # std of per-env start-pose noise around base config (rad)

# Per-joint waypoint spread (rad) used in random_joint_traj.
# Joint order matches the G1 / most robots:
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
# Larger spread on pitch and elbow (depth joints) produces enough forward-backward
# wrist variation for the model to infer torso yaw from bilateral trajectories.
# Wrist joints get a smaller spread so palm orientation stays natural.
WAYPOINT_SPREAD = np.array([0.40, 0.15, 0.15, 0.30, 0.15, 0.10, 0.10], dtype=np.float32)


def taskspace_traj(
    num_envs: int,
    side: str,       # "left" or "right"
    rng: np.random.Generator,
    ik,              # WristIK instance
    sample_cfg: dict,
    q_ref: np.ndarray | None = None,  # (n_dof,) anchor config; typically start_config
) -> np.ndarray:
    """
    Sample smooth wrist trajectories by walking in Cartesian space and solving IK.

    Returns (num_envs, T, n_dof) float32.
    """
    from scipy.interpolate import CubicSpline
    import torch as _torch

    n_dof  = ik.n_dof
    jnt_lo = ik.limits[:, 0].cpu().numpy()
    jnt_hi = ik.limits[:, 1].cpu().numpy()

    # ── 1. Build per-env base config from q_ref + diversity noise ────────────
    # Tile q_ref across envs, then add:
    #   lateral_joint jitter  — shifts the arm left/right while keeping it on its
    #                           own side of the torso midline
    #   proximal_jitter       — randomises shoulder and elbow angles so FK gives
    #                           diverse wrist anchor positions across envs
    q_base = np.tile(q_ref.astype(np.float32), (num_envs, 1)) \
             if q_ref is not None \
             else np.zeros((num_envs, n_dof), dtype=np.float32)

    lat = sample_cfg["lateral_joint"]
    q_base[:, lat["index"]] += rng.uniform(*lat[side], num_envs).astype(np.float32)
    for jidx, (lo_j, hi_j) in sample_cfg["proximal_jitter"].items():
        q_base[:, jidx] += rng.uniform(lo_j, hi_j, num_envs).astype(np.float32)
    q_base = np.clip(q_base, jnt_lo, jnt_hi)

    # ── 2. FK → Cartesian anchor for each env ────────────────────────────────
    # The anchor is the wrist position that corresponds to q_base. The OU walk
    # in step 3 orbits around this point, so each env gets trajectories
    # centred on a different natural workspace location.
    with _torch.no_grad():
        pos_ref, _ = ik._fk(
            _torch.tensor(q_base, dtype=_torch.float32, device=ik.device)
        )
    anchor = pos_ref.cpu().numpy()   # (num_envs, 3)

    # ── 3. Ornstein-Uhlenbeck random walk → N_CTRL_TS Cartesian knots ───────
    # Each step adds uniform noise then springs back toward the anchor.
    # ou_spring controls how tightly the walk stays near the anchor:
    #   low  (≈0.04) → broader, slower drift — larger workspace excursions
    #   high (≈0.08) → tighter, faster return — more localised motion
    # ou_step sets how far the wrist can move per knot (m).
    ts_step, ts_spring = sample_cfg["ou_step"], sample_cfg["ou_spring"]
    ctrl_pos = np.zeros((num_envs, N_CTRL_TS, 3), dtype=np.float32)
    pos = anchor + rng.uniform(-ts_step, ts_step, (num_envs, 3)).astype(np.float32)
    for k in range(N_CTRL_TS):
        ctrl_pos[:, k, :] = pos
        pos = pos + ts_spring * (anchor - pos) + \
              rng.uniform(-ts_step, ts_step, (num_envs, 3)).astype(np.float32)

    # ── 4. Sequential position-only IK at each knot ──────────────────────────
    # Warm-starting from the previous knot's solution keeps the joint trajectory
    # smooth and avoids large discontinuities between knots.
    # id_quat = identity quaternion → position-only IK (orientation unconstrained).
    # q_base acts as the null-space target, keeping redundant DOFs near start_config.
    id_quat = _torch.tensor(
        np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (num_envs, 1))
    )
    q_ctrl  = np.zeros((num_envs, N_CTRL_TS, n_dof), dtype=np.float32)
    q_prev  = q_base
    q_base_t = _torch.tensor(q_base, dtype=_torch.float32)
    for k in range(N_CTRL_TS):
        q_t, _ = ik.solve_batch(
            _torch.tensor(ctrl_pos[:, k, :], dtype=_torch.float32),
            id_quat,
            q_init=_torch.tensor(q_prev, dtype=_torch.float32),
            q_ref=q_base_t,
        )
        q_ctrl[:, k, :] = q_t.cpu().numpy()
        q_prev = q_ctrl[:, k, :]

    # ── 5. Wrist joints: independent from IK, sampled per clip ──────────────
    # The position IK only constrains proximal DOFs; wrist joints are free.
    # We sample a single wrist pose per clip and add small per-knot variation
    # (±10 % of range) to keep palm orientation roughly constant within a clip
    # while varying it across clips.
    # wrist_relative=True: limits are offsets from the mean start_config wrist
    #                      value (good for robots where the zero pose is unusual)
    # wrist_relative=False: limits are absolute joint-space bounds
    ws, we = sample_cfg["wrist_joints"]
    if n_dof > ws:
        lims = sample_cfg["wrist_limits"]
        if isinstance(lims, dict):   # per-side limits (e.g. Robonaut2 forearm_roll)
            lims = lims[side]
        if sample_cfg.get("wrist_relative", False):
            centre   = q_base[:, ws:we].mean(axis=0)
            wrist_lo = np.clip(centre + lims[:, 0], jnt_lo[ws:we], jnt_hi[ws:we])
            wrist_hi = np.clip(centre + lims[:, 1], jnt_lo[ws:we], jnt_hi[ws:we])
        else:
            wrist_lo = np.clip(lims[:, 0], jnt_lo[ws:we], jnt_hi[ws:we])
            wrist_hi = np.clip(lims[:, 1], jnt_lo[ws:we], jnt_hi[ws:we])
        n_wrist      = we - ws
        wrist_base   = rng.uniform(wrist_lo, wrist_hi,
                                   size=(num_envs, n_wrist)).astype(np.float32)
        wrist_spread = (wrist_hi - wrist_lo) * 0.10
        wrist_delta  = rng.uniform(-wrist_spread, wrist_spread,
                                   size=(num_envs, N_CTRL_TS, n_wrist)).astype(np.float32)
        q_ctrl[:, :, ws:we] = np.clip(
            wrist_base[:, None, :] + wrist_delta, wrist_lo, wrist_hi
        )

    # ── 6. Cubic spline through joint-space knots → T frames ─────────────────
    # Fitting a spline through N_CTRL_TS knots gives smooth, curved trajectories
    # without sharp velocity discontinuities between knots.
    t_knots = np.linspace(0.0, 1.0, N_CTRL_TS)
    t_eval  = np.linspace(0.0, 1.0, T)
    y    = q_ctrl.transpose(1, 0, 2).reshape(N_CTRL_TS, num_envs * n_dof)
    cs   = CubicSpline(t_knots, y, bc_type="not-a-knot")
    traj = cs(t_eval).reshape(T, num_envs, n_dof).transpose(1, 0, 2).astype(np.float32)
    return traj


def random_joint_traj(
    num_envs: int,
    limits: np.ndarray,   # (n_dof, 2) raw joint limits from the env
    rng: np.random.Generator,
    sample_cfg: dict,
    side: str = "left",
    shared_base: np.ndarray | None = None,  # (num_envs, n_dof); typically tiled start_config
) -> np.ndarray:
    """
    Sample joint trajectories by interpolating between random waypoints in joint space.

    Simpler and faster than taskspace_traj (no IK), but produces less natural
    Cartesian motion. Used as a fallback or for ablations.

    Returns (num_envs, T, n_dof) float32.
    """
    n_dof = limits.shape[0]
    lo, hi = limits[:, 0].copy(), limits[:, 1].copy()

    # Narrow the raw joint limits to the manipulation workspace.
    # workspace_bias provides absolute per-joint clamps (e.g. elbow always bent,
    # shoulder_pitch always forward). Empty for robots whose limits are already tight.
    for jidx, (lo_j, hi_j) in sample_cfg.get("workspace_bias", {}).items():
        lo[jidx] = max(lo[jidx], lo_j)
        hi[jidx] = min(hi[jidx], hi_j)

    # Clamp the lateral joint (shoulder roll / rotation) per side to prevent the
    # arm from crossing the torso midline.
    lat = sample_cfg["lateral_joint"]
    lat_lo, lat_hi = lat[side]
    lo[lat["index"]] = max(lo[lat["index"]], lat_lo)
    hi[lat["index"]] = min(hi[lat["index"]], lat_hi)

    spread      = WAYPOINT_SPREAD[:n_dof]
    N_WAYPOINTS = 3   # number of target waypoints after the start pose

    # Base config: the centre of all waypoints. Using shared_base (= start_config)
    # keeps both arms at a similar height, mimicking natural bilateral manipulation.
    if shared_base is not None:
        base = np.clip(shared_base, lo, hi)
    else:
        base = rng.uniform(lo, hi, size=(num_envs, n_dof)).astype(np.float32)

    # Start slightly off the base so the arm isn't frozen at frame 0.
    start = base[:, None, :] + \
            rng.standard_normal((num_envs, 1, n_dof)).astype(np.float32) * INIT_NOISE
    start = np.clip(start, lo, hi)

    # Target waypoints: small perturbations around the base. WAYPOINT_SPREAD limits
    # how far each joint can stray per waypoint, keeping motion localised.
    delta    = rng.uniform(-spread, spread,
                           size=(num_envs, N_WAYPOINTS, n_dof)).astype(np.float32)
    targets  = np.clip(base[:, None, :] + delta, lo, hi)
    waypoints = np.concatenate([start, targets], axis=1)  # (N, N_WAYPOINTS+1, n_dof)

    # Cosine-ease between consecutive waypoints so velocity is zero at each knot.
    seg_len = T // N_WAYPOINTS
    traj    = np.empty((num_envs, T, n_dof), dtype=np.float32)
    for seg in range(N_WAYPOINTS):
        q0 = waypoints[:, seg,     :]
        q1 = waypoints[:, seg + 1, :]
        t_start = seg * seg_len
        t_end   = t_start + seg_len if seg < N_WAYPOINTS - 1 else T
        frames  = t_end - t_start
        alpha = (1 - np.cos(np.linspace(0, np.pi, frames))) / 2  # 0 → 1 with ease-in/out
        traj[:, t_start:t_end, :] = (
            q0[:, None, :] * (1 - alpha[None, :, None]) +
            q1[:, None, :] *      alpha[None, :, None]
        )
    return traj


def collect_trajectories(
    env,
    left_q_np:  np.ndarray,   # (N, T, n_dof)
    right_q_np: np.ndarray,   # (N, T, n_dof)
) -> tuple:
    """
    Roll out joint trajectories through the sim and collect wrist poses.

    Returns (left_traj, right_traj), each (N, T, 7) float32 CPU tensor
    with layout [pos(3) | quat_wxyz(4)] in the world frame.
    """
    import jax.numpy as jnp
    import torch

    N = left_q_np.shape[0]
    left_pos   = np.empty((N, T, 3), dtype=np.float32)
    left_quat  = np.empty((N, T, 4), dtype=np.float32)
    right_pos  = np.empty((N, T, 3), dtype=np.float32)
    right_quat = np.empty((N, T, 4), dtype=np.float32)

    state = env.reset()
    for t in range(T):
        state = env.step_joints(state, {
            "left":  jnp.array(left_q_np[:,  t, :]),
            "right": jnp.array(right_q_np[:, t, :]),
        })
        obs = env.get_obs(state)
        left_pos[:,  t, :] = np.array(obs["left"]["pos"])
        left_quat[:, t, :] = np.array(obs["left"]["quat"])
        right_pos[:,  t, :] = np.array(obs["right"]["pos"])
        right_quat[:, t, :] = np.array(obs["right"]["quat"])

    left_traj  = torch.from_numpy(np.concatenate([left_pos,  left_quat],  axis=-1))
    right_traj = torch.from_numpy(np.concatenate([right_pos, right_quat], axis=-1))
    return left_traj, right_traj
