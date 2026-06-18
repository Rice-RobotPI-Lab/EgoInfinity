"""
XLeRobot MuJoCo environment.

Two SO-ARM100 5-DoF arms mounted on a locked Raskog-cart base.

torso_frame is at base_link origin (world origin, identity orientation).
All IK targets and EE poses are expressed in this frame, which coincides
with the world frame since the base is stationary.

Joint order per arm (5 DoF):
  [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]

Action:  dict with optional keys "left"/"right", each (5,) joint angles [rad]
Observation: dict with "left"/"right", each containing:
    - "pos":  (3,) EE position in world frame
    - "quat": (4,) EE orientation (w, x, y, z) in world frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from sim.base_env import BaseEnv

# ── robot description ─────────────────────────────────────────────────────────

_ROBOT_DIR     = Path(__file__).parents[3] / "robots" / "xlerobot"
_SCENE_PATH    = _ROBOT_DIR / "scene_vis.xml"
_MJCF_MJX_PATH = _ROBOT_DIR / "xlerobot_mjx.xml"

_ARM_JOINTS = {
    "left":  ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"],
    "right": ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"],
}

_EE_BODY    = {"left": "left_hand_frame", "right": "right_hand_frame"}
_TORSO_BODY = "torso_frame"
_N_DOF      = 5


class XLeRobotEnv(BaseEnv):
    """
    Single-instance XLeRobot simulation environment.

    Parameters
    ----------
    mjcf_path : str | Path, optional
        Path to the MJCF file. Defaults to robots/xlerobot/scene_vis.xml.
    start_config : dict, optional
        {"left": (5,), "right": (5,)} applied after reset.
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

        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _TORSO_BODY
        )

        self._joint_limits: dict[str, np.ndarray] = {}
        for side, jids in self._joint_ids.items():
            self._joint_limits[side] = np.array(
                [self.model.jnt_range[jid] for jid in jids]
            )

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self._start_config is not None:
            for side, q in self._start_config.items():
                self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── action ────────────────────────────────────────────────────────────────

    def set_arm_joints(self, side: str, q: np.ndarray):
        assert q.shape == (_N_DOF,), f"Expected ({_N_DOF},) joint angles, got {q.shape}"
        for i, aid in enumerate(self._actuator_ids[side]):
            self.data.ctrl[aid] = q[i]
        for i, jid in enumerate(self._joint_ids[side]):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def set_finger_joints(self, q: np.ndarray, joint_names: list[str]):
        """Drive jaw gripper from WilorHandRetargeter output (q in radians)."""
        _jaw = {"left": "Jaw_L", "right": "Jaw_R"}
        for i, name in enumerate(joint_names):
            if "gripper" not in name:
                continue
            side = "left" if "left" in name else "right"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, _jaw[side])
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _jaw[side])
            if aid >= 0:
                self.data.ctrl[aid] = float(np.clip(q[i], lo, hi))
        mujoco.mj_forward(self.model, self.data)

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
