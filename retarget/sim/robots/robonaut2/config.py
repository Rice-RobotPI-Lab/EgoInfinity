import numpy as np

from sim.robot_config import RobotConfig
from sim.robots.robonaut2.env import Robonaut2Env, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY

_START_L = np.array([-0.0825, -1.0044, -1.8417, -1.6244,  1.0565,  0.0503, -0.0029], dtype=np.float32)
_START_R = np.array([ 0.0835, -0.9661,  1.8180, -1.6204, -1.0206,  0.0744, -0.0030], dtype=np.float32)

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              Robonaut2Env,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             "robonaut2",
    "retargeter":           "robonaut2",
    # ── trajectory sampling ───────────────────────────────────────────────────
    "apply_workspace_bias": False,
    # ── MuJoCo body names ─────────────────────────────────────────────────────
    "torso_body":           "torso_frame",
    "wrist_body":           {"left": "left_hand_frame", "right": "right_hand_frame"},
    "extra_bodies":         [],
    # ── joint configurations ──────────────────────────────────────────────────
    "zero_config": {
        "left":  np.zeros(7, dtype=np.float32),
        "right": np.zeros(7, dtype=np.float32),
    },
    "home_config": {
        "left":  _START_L.copy(),
        "right": _START_R.copy(),
    },
    "start_config": {
        "left":  _START_L.copy(),
        "right": _START_R.copy(),
    },
    # ── viewer camera ─────────────────────────────────────────────────────────
    "cam_azimuth":   60,
    "cam_elevation": -20,
    "cam_distance":  4.34,
    "cam_lookat":    [0.0, 0.0, 0.63],
}

ENV_CONFIG = RobotConfig(
    mjcf_path=_MJCF_MJX_PATH,
    joint_groups=_ARM_JOINTS,
    end_effectors=_EE_BODY,
)
