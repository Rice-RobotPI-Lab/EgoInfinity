"""
Dual Franka FR3 MuJoCo environment.

Two FR3 arms on a fixed base:
  left  arm at [0, +0.5, 0]
  right arm at [0, -0.5, 0]

The virtual "torso frame" is the world origin (identity pose).

Action:  dict with optional keys "left" and "right", each (7,) joint angles [rad]
         for [joint1 .. joint7]

Observation: dict with optional keys "left" and "right", each containing:
    - "pos":  (3,) end-effector position in world frame (at the flange, +107 mm from link7)
    - "quat": (4,) end-effector orientation (w, x, y, z) in world frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from sim.base_env import BaseEnv

# ── robot description ─────────────────────────────────────────────────────────

_ROBOT_DIR     = Path(__file__).parents[3] / "robots" / "franka_fr3"
_SCENE_PATH    = _ROBOT_DIR / "scene_vis.xml"
_MJCF_MJX_PATH = _ROBOT_DIR / "fr3_dual_mjx.xml"

_ARM_JOINTS = {
    "left":  ["left_joint1",  "left_joint2",  "left_joint3",  "left_joint4",
               "left_joint5",  "left_joint6",  "left_joint7"],
    "right": ["right_joint1", "right_joint2", "right_joint3", "right_joint4",
               "right_joint5", "right_joint6", "right_joint7"],
}

_EE_BODY = {"left": "left_ee", "right": "right_ee"}

_GRIPPER_ACTUATOR = {"left": "left_gripper", "right": "right_gripper"}

_GRIPPER_JOINTS = {
    "left":  ["left_finger_joint1",  "left_finger_joint2"],
    "right": ["right_finger_joint1", "right_finger_joint2"],
}

_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, 0.7854], dtype=np.float64)


class FrankaEnv(BaseEnv):
    """
    Single-instance dual Franka FR3 simulation environment.

    Parameters
    ----------
    mjcf_path : str | Path, optional
        Path to the MJCF file. Defaults to robots/franka_fr3/scene_vis.xml.
    start_config : dict, optional
        {"left": (7,), "right": (7,)} applied after reset.
    """

    def __init__(
        self,
        mjcf_path: Optional[Path] = None,
        start_config: Optional[dict] = None,
    ):
        mjcf_path = Path(mjcf_path) if mjcf_path is not None else _SCENE_PATH
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data  = mujoco.MjData(self.model)

        self._start_config      = start_config

        self._joint_ids:    dict[str, list[int]] = {}
        self._actuator_ids: dict[str, list[int]] = {}
        self._body_ids:     dict[str, int]       = {}

        for side, joints in _ARM_JOINTS.items():
            self._joint_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in joints
            ]
            self._actuator_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
                for j in joints
            ]
            self._body_ids[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY[side]
            )

        self._gripper_actuator_ids = {
            side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for side, name in _GRIPPER_ACTUATOR.items()
        }
        self._gripper_joint_ids = {
            side: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
            for side, joints in _GRIPPER_JOINTS.items()
        }

        self._joint_limits: dict[str, np.ndarray] = {}
        for side, jids in self._joint_ids.items():
            self._joint_limits[side] = np.array(
                [self.model.jnt_range[jid] for jid in jids]
            )

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            for side in ("left", "right"):
                for i, jid in enumerate(self._joint_ids[side]):
                    adr = self.model.jnt_qposadr[jid]
                    self.data.qpos[adr] = _HOME_QPOS[i]
                    self.data.ctrl[self._actuator_ids[side][i]] = _HOME_QPOS[i]
        mujoco.mj_forward(self.model, self.data)
        if self._start_config is not None:
            for side, q in self._start_config.items():
                self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── action ────────────────────────────────────────────────────────────────

    def set_arm_joints(self, side: str, q: np.ndarray):
        assert q.shape == (7,), f"Expected (7,) joint angles, got {q.shape}"
        for i, aid in enumerate(self._actuator_ids[side]):
            self.data.ctrl[aid] = q[i]
        for i, jid in enumerate(self._joint_ids[side]):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, side: str, width: float):
        width = float(np.clip(width, 0.0, 0.04))
        self.data.ctrl[self._gripper_actuator_ids[side]] = width
        for jid in self._gripper_joint_ids[side]:
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = width
        mujoco.mj_forward(self.model, self.data)

    def set_finger_joints(self, q: np.ndarray, joint_names: list[str]):
        for i, name in enumerate(joint_names):
            if "gripper" in name:
                side = "left" if "left" in name else "right"
                self.set_gripper(side, float(q[i]))

    def step_joints(self, action: dict) -> dict:
        for side, q in action.items():
            self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {
            side: {"pos": self.data.xpos[bid].copy(), "quat": self.data.xquat[bid].copy()}
            for side, bid in self._body_ids.items()
        }

    def get_wrist_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_ids[side]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
