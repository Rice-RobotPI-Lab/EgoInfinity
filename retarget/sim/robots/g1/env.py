"""
Single G1 MuJoCo environment.

Action:  dict with optional keys "left" and "right", each (7,) joint angles [rad]
         for [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
              wrist_roll, wrist_pitch, wrist_yaw]

Observation: dict with optional keys "left" and "right", each containing:
    - "pos":  (3,) wrist position in world frame
    - "quat": (4,) wrist orientation quaternion (w, x, y, z) in world frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from sim.base_env import BaseEnv

# ── robot description ─────────────────────────────────────────────────────────

_ROBOT_DIR     = Path(__file__).parents[3] / "robots" / "unitree_g1"
_MJCF_PATH     = _ROBOT_DIR / "g1.xml"
_MJCF_MJX_PATH = _ROBOT_DIR / "g1_mjx.xml"
_SCENE_PATH    = _ROBOT_DIR / "scene_vis.xml"

_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

_EE_BODY    = {"left": "left_hand_frame", "right": "right_hand_frame"}
_TORSO_BODY = "torso_link"


class G1Env(BaseEnv):
    """
    Single-instance G1 arm simulation environment.

    Parameters
    ----------
    mjcf_path : str | Path, optional
        Path to the G1 MJCF file. Defaults to robots/unitree_g1/g1.xml.
    start_config : dict, optional
        {"left": (7,), "right": (7,)} applied after reset.
    """

    def __init__(
        self,
        mjcf_path: Optional[Path] = None,
        start_config: Optional[dict] = None,
    ):
        mjcf_path = Path(mjcf_path) if mjcf_path is not None else _MJCF_PATH
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
        assert q.shape == (7,), f"Expected (7,) joint angles, got {q.shape}"
        for i, aid in enumerate(self._actuator_ids[side]):
            self.data.ctrl[aid] = q[i]
        for i, jid in enumerate(self._joint_ids[side]):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def set_finger_joints(self, q: np.ndarray, joint_names: list[str]):
        """Set finger joints by name; silently skips joints absent from the model."""
        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
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
            side: {
                "pos":  self.data.xpos[bid].copy(),
                "quat": self.data.xquat[bid].copy(),
            }
            for side, bid in self._body_ids.items()
        }

    def get_wrist_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_ids[side]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
