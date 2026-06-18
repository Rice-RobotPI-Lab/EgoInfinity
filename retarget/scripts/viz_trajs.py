"""
Visualize training-data trajectories.

Each iteration samples one clip using the same pipeline as train.py,
animates the robot in a MuJoCo viewer with colored wrist trails, and prints a
per-clip stats table.

Overlay colours (world frame)
  Blue   = left  wrist trajectory
  Orange = right wrist trajectory

Usage
-----
    python3 scripts/viz_trajs.py                   # 10 clips, G1
    python3 scripts/viz_trajs.py --n_clips 20
    python3 scripts/viz_trajs.py --robot franka
    python3 scripts/viz_trajs.py --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
with contextlib.redirect_stdout(io.StringIO()):
    import jax  # noqa: F401 — must be imported after XLA_PYTHON_CLIENT_PREALLOCATE

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.vec_env_jax import JaxVecEnv
from sim.robots import ROBOT_CONFIGS as _ROBOT_CONFIGS, SAMPLE_CONFIGS as _SAMPLE_CONFIGS, ENV_CONFIGS as _ENV_CONFIGS
from sim.traj_sampler import taskspace_traj, random_joint_traj, collect_trajectories, T

DT = 1.0 / 30


# ── geometry helpers ──────────────────────────────────────────────────────────

def _draw_traj(scene, positions: np.ndarray, rgba: np.ndarray, radius: float = 0.012):
    for pos in positions:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=np.float64),
            pos.astype(np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        scene.ngeom += 1


# ── statistics ────────────────────────────────────────────────────────────────

def _clip_stats(left_pos: np.ndarray, right_pos: np.ndarray) -> dict:
    dl = np.linalg.norm(np.diff(left_pos,  axis=0), axis=-1)
    dr = np.linalg.norm(np.diff(right_pos, axis=0), axis=-1)
    return {
        "speed":      (dl.mean() + dr.mean()) / 2 / DT,
        "path":       (dl.sum()  + dr.sum())  / 2,
        "motion_std": (left_pos.std(axis=0).mean() + right_pos.std(axis=0).mean()) / 2,
        "separation": np.linalg.norm(left_pos - right_pos, axis=-1).mean(),
    }


def _print_stats(stats: dict, clip_idx: int):
    print(f"\n  ── clip {clip_idx:3d} ─────────────────────────────")
    print(f"  {'speed':<18s}  {stats['speed']:>7.3f} m/s")
    print(f"  {'path/hand':<18s}  {stats['path']:>7.3f} m")
    print(f"  {'motion std':<18s}  {stats['motion_std']:>7.3f} m")
    print(f"  {'L-R sep.':<18s}  {stats['separation']:>7.3f} m")


def _print_aggregate(all_stats: list[dict]):
    keys   = ["speed", "path", "motion_std", "separation"]
    labels = ["speed (m/s)", "path/hand (m)", "motion std (m)", "L-R sep. (m)"]
    print(f"\n{'='*48}")
    print(f"  AGGREGATE over {len(all_stats)} clips")
    print(f"{'='*48}")
    print(f"  {'Metric':<20s}  {'Mean':>7s}  {'Std':>7s}")
    print(f"  {'-'*44}")
    for key, label in zip(keys, labels):
        vals = np.array([s[key] for s in all_stats])
        print(f"  {label:<20s}  {vals.mean():>7.3f}  {vals.std():>7.3f}")
    print(f"{'='*48}\n")


# ── main ──────────────────────────────────────────────────────────────────────

_LEFT_RGBA  = np.array([0.15, 0.35, 0.95, 0.85], dtype=np.float32)
_RIGHT_RGBA = np.array([0.95, 0.50, 0.10, 0.85], dtype=np.float32)


def visualize(args: argparse.Namespace):
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
    rng  = np.random.default_rng(seed)
    print(f"Robot: {args.robot}  |  Seed: {seed}  |  sample_mode: {args.sample_mode}")

    _cfg = _ROBOT_CONFIGS[args.robot]
    _sc  = _SAMPLE_CONFIGS[args.robot]
    start_l = np.asarray(_cfg["start_config"]["left"],  dtype=np.float32)
    start_r = np.asarray(_cfg["start_config"]["right"], dtype=np.float32)

    # ── environment setup ─────────────────────────────────────────────────────
    print("Initialising JaxVecEnv (num_envs=1)…")
    env_jax = JaxVecEnv(_ENV_CONFIGS[args.robot], num_envs=1)
    env_jax.warmup()
    limits_l = np.array(env_jax._joint_limits["left"])
    limits_r = np.array(env_jax._joint_limits["right"])
    print("JIT compilation done.\n")

    env_vis = _cfg["env_cls"](mjcf_path=_cfg["scene_path"],
                              start_config=_cfg["start_config"])
    env_vis.reset()

    def collect_wrist_pos(lq, rq):
        lw, rw = collect_trajectories(env_jax, lq, rq)
        return lw[0, :, :3].numpy(), rw[0, :, :3].numpy()

    # ── IK solvers for task-space mode ────────────────────────────────────────
    ik_left = ik_right = None
    if args.sample_mode == "taskspace":
        from kinematics.wrist_ik import WristIK, RobotIKConfig
        import mujoco as _mj
        print("Building WristIK solvers…")
        ik_robot = _cfg["ik_robot"]
        if ik_robot is None:
            ik_left  = WristIK(side="left",  ori_weight=0.0, max_iter=50, tol_pos=0.01, q_default=start_l)
            ik_right = WristIK(side="right", ori_weight=0.0, max_iter=50, tol_pos=0.01, q_default=start_r)
        else:
            ik_left  = WristIK(side="left",
                               robot=getattr(RobotIKConfig, ik_robot)("left"),
                               ori_weight=0.0, max_iter=50, tol_pos=0.01, q_default=start_l)
            ik_right = WristIK(side="right",
                               robot=getattr(RobotIKConfig, ik_robot)("right"),
                               ori_weight=0.0, max_iter=50, tol_pos=0.01, q_default=start_r)
        torso_id  = _mj.mj_name2id(env_vis.model, _mj.mjtObj.mjOBJ_BODY, _cfg["torso_body"])
        print(f"{_cfg['torso_body']} pos: {env_vis.data.xpos[torso_id].round(4)}\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    all_stats: list[dict] = []
    print("Legend:  Blue = left wrist   Orange = right wrist")
    print(f"Running {args.n_clips} clips — press Ctrl+C to stop early.\n")

    with mujoco.viewer.launch_passive(env_vis.model, env_vis.data) as viewer:
        viewer.cam.azimuth   = _cfg["cam_azimuth"]
        viewer.cam.elevation = _cfg["cam_elevation"]
        viewer.cam.distance  = _cfg["cam_distance"]
        viewer.cam.lookat[:] = _cfg["cam_lookat"]

        env_vis.set_arm_joints("left",  start_l.astype(np.float64))
        env_vis.set_arm_joints("right", start_r.astype(np.float64))
        viewer.sync()

        for clip_idx in range(1, args.n_clips + 1):
            if not viewer.is_running():
                break

            if args.sample_mode == "taskspace":
                left_q  = taskspace_traj(1, "left",  rng, ik_left,  _sc, q_ref=start_l)
                right_q = taskspace_traj(1, "right", rng, ik_right, _sc, q_ref=start_r)
            else:
                left_q  = random_joint_traj(1, limits_l, rng, _sc, side="left",
                                            shared_base=np.tile(start_l, (1, 1)))
                right_q = random_joint_traj(1, limits_r, rng, _sc, side="right",
                                            shared_base=np.tile(start_r, (1, 1)))

            left_pos, right_pos = collect_wrist_pos(left_q, right_q)

            stats = _clip_stats(left_pos, right_pos)
            all_stats.append(stats)
            _print_stats(stats, clip_idx)

            env_vis.set_arm_joints("left",  left_q[0, 0])
            env_vis.set_arm_joints("right", right_q[0, 0])
            viewer.sync()

            for frame in range(T):
                if not viewer.is_running():
                    break
                env_vis.set_arm_joints("left",  left_q[0, frame])
                env_vis.set_arm_joints("right", right_q[0, frame])
                viewer.user_scn.ngeom = 0
                _draw_traj(viewer.user_scn, left_pos,  _LEFT_RGBA)
                _draw_traj(viewer.user_scn, right_pos, _RIGHT_RGBA)
                viewer.sync()
                time.sleep(DT)

    if all_stats:
        _print_aggregate(all_stats)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Visualize training trajectories")
    p.add_argument("--robot",        default="g1",
                   choices=list(_ROBOT_CONFIGS.keys()))
    p.add_argument("--n_clips",      type=int, default=10)
    p.add_argument("--seed",         type=int, default=None)
    p.add_argument("--sample_mode",  default="taskspace",
                   choices=["jointspace", "taskspace"])
    return p.parse_args()


if __name__ == "__main__":
    visualize(_parse())
