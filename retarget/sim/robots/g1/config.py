import numpy as np

from sim.robot_config import RobotConfig
from sim.robots.g1.env import G1Env, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              G1Env,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             None,       # WristIK default = G1
    "retargeter":           "g1",
    # ── trajectory sampling ───────────────────────────────────────────────────
    "apply_workspace_bias": True,
    # ── MuJoCo body names ─────────────────────────────────────────────────────
    "torso_body":           "torso_link",
    "wrist_body":           {"left": "left_hand_frame", "right": "right_hand_frame"},
    "extra_bodies":         [],
    # ── joint configurations ──────────────────────────────────────────────────
    "zero_config": {
        "left":  np.zeros(7, dtype=np.float32),
        "right": np.zeros(7, dtype=np.float32),
    },
    "home_config": {
        "left":  np.array([ 0.2,  0.2, 0.0, 1.28, 0.0, 0.0, 0.0], dtype=np.float32),
        "right": np.array([ 0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0], dtype=np.float32),
    },
    "start_config": {
        "left":  np.array([-0.363,  0.371, -0.195, 0.336, -0.419, 0.0, 0.0], dtype=np.float32),
        "right": np.array([-0.363, -0.371,  0.195, 0.336,  0.419, 0.0, 0.0], dtype=np.float32),
    },
    # ── viewer camera ─────────────────────────────────────────────────────────
    "cam_azimuth":   140,
    "cam_elevation": -20,
    "cam_distance":  2.59,
    "cam_lookat":    [0.10, 0.0, 0.63],
}

ENV_CONFIG = RobotConfig(
    mjcf_path=_MJCF_MJX_PATH,
    joint_groups=_ARM_JOINTS,
    end_effectors=_EE_BODY,
)
