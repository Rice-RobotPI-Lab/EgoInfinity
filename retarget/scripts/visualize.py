"""
Robot arm visualizer.

Usage
-----
    python scripts/visualize.py                     # G1 at start config
    python scripts/visualize.py --mode home         # home configuration
    python scripts/visualize.py --mode zero         # zero configuration
    python scripts/visualize.py --robot franka      # different robot
    python scripts/visualize.py --show-body-frames  # overlay torso and wrist axes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer
import numpy as np

from sim.robots import ROBOT_CONFIGS as _ROBOT_CONFIGS

_FRAME_AXIS_WIDTH = 0.008
_AXIS_COLORS = (
    np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
)


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)  ],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z), 2*(y*z - x*w)  ],
        [2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def _add_axis(scene, origin: np.ndarray, tip: np.ndarray, rgba: np.ndarray):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([_FRAME_AXIS_WIDTH] * 3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, _FRAME_AXIS_WIDTH,
                         origin.astype(np.float64), tip.astype(np.float64))
    scene.ngeom += 1


def _draw_frame(scene, pos: np.ndarray, quat: np.ndarray, axis_length: float):
    R = _quat_to_rotmat(quat)
    for i, color in enumerate(_AXIS_COLORS):
        _add_axis(scene, pos, pos + axis_length * R[:, i], color)


def _draw_body_frames(viewer, env, robot_cfg: dict):
    viewer.user_scn.ngeom = 0
    bodies = [
        (robot_cfg["torso_body"],              0.28),
        (robot_cfg["wrist_body"]["left"],      0.16),
        (robot_cfg["wrist_body"]["right"],     0.16),
    ]
    for name, length in bodies:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            _draw_frame(viewer.user_scn,
                        env.data.xpos[bid].copy(),
                        env.data.xquat[bid].copy(),
                        length)


def run_visualizer(robot: str = "g1", mode: str = "start", show_body_frames: bool = False):
    robot_cfg = _ROBOT_CONFIGS[robot]
    env = robot_cfg["env_cls"](mjcf_path=robot_cfg["scene_path"])
    env.reset()

    for side, q in robot_cfg[f"{mode}_config"].items():
        env.set_arm_joints(side, np.asarray(q, dtype=np.float64))

    print(f"Robot: {robot}  |  Config: {mode}")
    for side in ("left", "right"):
        pos, quat = env.get_wrist_pose(side)
        print(f"  [{side}] pos={np.round(pos, 3)}  quat={np.round(quat, 3)}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth   = robot_cfg["cam_azimuth"]
        viewer.cam.elevation = robot_cfg["cam_elevation"]
        viewer.cam.distance  = robot_cfg["cam_distance"]
        viewer.cam.lookat[:] = robot_cfg["cam_lookat"]

        while viewer.is_running():
            if show_body_frames:
                _draw_body_frames(viewer, env, robot_cfg)
            viewer.sync()
            time.sleep(0.033)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot arm visualizer")
    parser.add_argument("--robot", choices=list(_ROBOT_CONFIGS.keys()), default="g1")
    parser.add_argument("--mode", choices=["home", "start", "zero"], default="start")
    parser.add_argument("--show-body-frames", action="store_true",
                        help="Overlay RGB axes on torso and wrist frames")
    args = parser.parse_args()
    run_visualizer(robot=args.robot, mode=args.mode, show_body_frames=args.show_body_frames)
