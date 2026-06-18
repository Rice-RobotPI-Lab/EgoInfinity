"""
Generic JAX/MJX vectorized robot environment for GPU-parallel training.

Runs N environments in parallel on GPU using jax.vmap + jit.
State is a batched mjx.Data pytree (leading dim = num_envs).

Design
------
Robots are described by a RobotConfig dataclass:
  - joint_groups  : named groups of controllable joints, e.g. {"left": [...], "right": [...]}
  - end_effectors : body to track per group, e.g. {"left": "left_hand_frame", ...}
  - mjcf_path     : path to the MJCF file

All env operations are keyed by group name, making the API robot-agnostic.
A pre-built G1_CONFIG is provided for the Unitree G1.

Usage
-----
    # G1 (bilateral arms)
    from sim.vec_env_jax import JaxVecEnv, G1_CONFIG

    env = JaxVecEnv(G1_CONFIG, num_envs=256)
    state = env.reset()

    # step: joint_dict keys are group names defined in the config
    state = env.step_joints(state, {"left": q_left, "right": q_right})  # (N,7) each
    obs   = env.get_obs(state)   # {"left": {"pos":(N,3), "quat":(N,4)}, "right": ...}

    # rollout N trajectories in parallel, returns mean per-env loss over T steps
    loss = env.rollout(
        joint_trajs  = {"left": (N,T,7), "right": (N,T,7)},
        target_trajs = {"left": {"pos":(N,T,3), "quat":(N,T,4)},
                        "right": {"pos":(N,T,3), "quat":(N,T,4)}},
    )
    # loss["left"]["total"]  : (N,) — per-env mean loss for the left arm
    # loss["right"]["total"] : (N,)
    # loss["total"]          : (N,) — sum across all groups

    # New robot
    my_config = RobotConfig(
        mjcf_path   = Path("robots/my_robot/robot.xml"),
        joint_groups = {"arm": ["joint1", "joint2", "joint3"]},
        end_effectors = {"arm": "end_effector_body"},
    )
    env = JaxVecEnv(my_config, num_envs=128)
"""

from __future__ import annotations

import contextlib
import io

import mujoco
with contextlib.redirect_stdout(io.StringIO()):
    from mujoco import mjx  # suppress optional-backend warnings (warp, mujoco_warp)
import jax
import jax.numpy as jnp


from sim.robot_config import RobotConfig  # noqa: E402
from sim.robots import ENV_CONFIGS        # noqa: E402


# ── generic JAX vectorized environment ───────────────────────────────────────

class JaxVecEnv:
    """
    Generic JAX/MJX vectorized robot environment.

    Works with any MJCF robot described by a RobotConfig.
    All operations are group-keyed dicts so the API is robot-agnostic.

    Parameters
    ----------
    config : RobotConfig
        Robot description (joints, end-effectors, MJCF path).
    num_envs : int
        Number of parallel environments.
    pos_weight, ori_weight, joint_limit_weight : float
        Weights for the tracking loss terms.
    """

    def __init__(
        self,
        config: RobotConfig,
        num_envs: int,
    ):
        self.config = config
        self.num_envs = num_envs

        model = mujoco.MjModel.from_xml_path(str(config.mjcf_path))
        self.mx = mjx.put_model(model)

        # ── cache indices (Python ints — static at JAX trace time) ────────
        self._qpos_adr: dict[str, list[int]] = {}
        self._n_dof: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}
        self._joint_limits: dict[str, jnp.ndarray] = {}

        for group, joints in config.joint_groups.items():
            jids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in joints
            ]
            self._qpos_adr[group] = [model.jnt_qposadr[jid] for jid in jids]
            self._n_dof[group] = len(joints)
            self._joint_limits[group] = jnp.array(
                [model.jnt_range[jid] for jid in jids]
            )  # (n_dof, 2)

        for group, body_name in config.end_effectors.items():
            self._body_ids[group] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )

        # ── batched initial state ─────────────────────────────────────────
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        dx0 = mjx.put_data(model, data)
        self._init_state: mjx.Data = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None], num_envs, axis=0), dx0
        )

        # ── jit-compile hot paths (lazy: compiles on first call) ──────────
        self._step_fn = jax.jit(jax.vmap(self._apply_joints_single))

    # ── single-env primitive (vmapped internally) ─────────────────────────

    def _apply_joints_single(self, dx: mjx.Data, joint_dict: dict) -> mjx.Data:
        """
        Apply joint angles for one or more groups to a SINGLE env and run FK.

        joint_dict : {group: (n_dof,)}  — only provided groups are updated;
                     joints not in joint_dict keep their current qpos.
        """
        qpos = dx.qpos
        for group, q in joint_dict.items():
            for i, adr in enumerate(self._qpos_adr[group]):
                qpos = qpos.at[adr].set(q[i])
        return mjx.kinematics(self.mx, dx.replace(qpos=qpos))

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> mjx.Data:
        """Return batched home-pose state (N copies of zero-joint config)."""
        return self._init_state

    def step_joints(self, state: mjx.Data, joint_dict: dict) -> mjx.Data:
        """
        Apply joint angles for all N envs and run forward kinematics.

        Parameters
        ----------
        state      : batched mjx.Data from reset() or a prior step
        joint_dict : {group: (N, n_dof)}  — any subset of config.joint_groups;
                     groups not listed keep their current qpos.

        Returns
        -------
        New batched mjx.Data with updated xpos / xquat.

        Notes
        -----
        JIT re-traces when joint_dict keys change (different robots / subsets).
        Within a training loop the dict structure is fixed, so this happens once.
        """
        return self._step_fn(state, joint_dict)

    def get_obs(self, state: mjx.Data) -> dict:
        """
        Return end-effector poses for all N envs.

        Returns
        -------
        dict keyed by group name (matching config.end_effectors), each entry::

            {"pos":  (N, 3),   # world-frame position
             "quat": (N, 4)}   # world-frame orientation (w, x, y, z)
        """
        obs = {}
        for group, bid in self._body_ids.items():
            obs[group] = {
                "pos":  state.xpos[:, bid, :],   # (N, 3)
                "quat": state.xquat[:, bid, :],  # (N, 4)
            }
        return obs

    def warmup(self):
        """JIT-compile step_joints and get_obs with dummy inputs."""
        dummy_q = {g: jnp.zeros((self.num_envs, n)) for g, n in self._n_dof.items()}
        state = self.step_joints(self.reset(), dummy_q)
        _ = self.get_obs(state)
