"""
Dual Robonaut 2 MuJoCo environment.

Two 7-DoF arms attached to a fixed torso body.
Joint names:  /r2/left_arm/joint0 … /r2/left_arm/joint6
              /r2/right_arm/joint0 … /r2/right_arm/joint6
EE bodies:    left_hand_frame, right_hand_frame  (canonical leaf bodies)
Torso body:   torso_frame                        (canonical root body)

IK targets and reported wrist poses are expressed in the torso frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from sim.base_env import BaseEnv

# ── robot description ─────────────────────────────────────────────────────────

_ROBOT_DIR     = Path(__file__).parents[3] / "robots" / "robonaut2"
_SCENE_PATH    = _ROBOT_DIR / "scene_vis.xml"
_MJCF_MJX_PATH = _ROBOT_DIR / "r2_mjx.xml"

_ARM_JOINTS = {
    "left": [
        "/r2/left_arm/joint0", "/r2/left_arm/joint1", "/r2/left_arm/joint2",
        "/r2/left_arm/joint3", "/r2/left_arm/joint4",
        "/r2/left_arm/joint5", "/r2/left_arm/joint6",
    ],
    "right": [
        "/r2/right_arm/joint0", "/r2/right_arm/joint1", "/r2/right_arm/joint2",
        "/r2/right_arm/joint3", "/r2/right_arm/joint4",
        "/r2/right_arm/joint5", "/r2/right_arm/joint6",
    ],
}

_EE_BODY    = {"left": "left_hand_frame",  "right": "right_hand_frame"}
_TORSO_BODY = "torso_frame"

# Maps retargeter short names → MJCF joint name suffixes per side.
_FINGER_JOINT_NAMES = {
    "index_abd":        "hand/index/joint0",
    "index_mcp":        "hand/index/joint1",
    "index_pip":        "hand/index/joint2",
    "index_dip":        "hand/index/joint3",
    "middle_abd":       "hand/middle/joint0",
    "middle_mcp":       "hand/middle/joint1",
    "middle_pip":       "hand/middle/joint2",
    "middle_dip":       "hand/middle/joint3",
    "little_prox":      "hand/little/joint0",
    "little_med":       "hand/little/joint1",
    "little_dist":      "hand/little/joint2",
    "ring_prox":        "hand/ring/joint0",
    "ring_med":         "hand/ring/joint1",
    "ring_dist":        "hand/ring/joint2",
    "thumb_cmc_spread": "hand/thumb/joint0",
    "thumb_mcp":        "hand/thumb/joint1",
    "thumb_ip":         "hand/thumb/joint2",
    "thumb_dip":        "hand/thumb/joint3",
}


class Robonaut2Env(BaseEnv):
    """
    Single-instance dual Robonaut 2 simulation environment.

    Parameters
    ----------
    mjcf_path : str | Path, optional
        Defaults to robots/robonaut2/scene_vis.xml.
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
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j + "_servo")
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
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
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
        """Set R2 finger joints from retargeter output (left_/right_ prefixed names)."""
        for i, name in enumerate(joint_names):
            if name.startswith("left_"):
                side, short = "left", name[len("left_"):]
            elif name.startswith("right_"):
                side, short = "right", name[len("right_"):]
            else:
                continue
            if short.endswith("_joint"):
                short = short[:-len("_joint")]
            suffix = _FINGER_JOINT_NAMES.get(short)
            if suffix is None:
                continue
            mjcf_name = f"/r2/{side}_arm/{suffix}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjcf_name)
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
