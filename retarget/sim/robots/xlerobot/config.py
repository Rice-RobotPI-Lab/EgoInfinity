import numpy as np

from sim.robot_config import RobotConfig
from sim.robots.xlerobot.env import XLeRobotEnv, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              XLeRobotEnv,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             "xlerobot",
    "retargeter":           "xlerobot",
    # ── trajectory sampling ───────────────────────────────────────────────────
    "apply_workspace_bias": False,
    # ── MuJoCo body names ─────────────────────────────────────────────────────
    # torso_frame at midpoint between arm bases: (-0.135, 0, 0.760) in base_link
    # Zero config EE in torso frame:   right=(0.480, -0.133, 0.086)  left=(0.480, 0.133, 0.086)
    # Start config EE in torso frame:  right=(0.396, -0.123, 0.237)  left=(0.396, 0.143, 0.237)
    "torso_body":           "torso_frame",
    "wrist_body":           {"left": "left_hand_frame", "right": "right_hand_frame"},
    "extra_bodies":         [],
    # ── joint configurations ──────────────────────────────────────────────────
    "zero_config": {
        "left":  np.zeros(5, dtype=np.float32),
        "right": np.zeros(5, dtype=np.float32),
    },
    "home_config": {
        "left":  np.zeros(5, dtype=np.float32),
        "right": np.zeros(5, dtype=np.float32),
    },
    "start_config": {
        "left":  np.array([0, np.pi/2, np.pi/2, 0, np.pi/2], dtype=np.float32),
        "right": np.array([0, np.pi/2, np.pi/2, 0, np.pi/2], dtype=np.float32),
    },
    # ── viewer camera ─────────────────────────────────────────────────────────
    "cam_azimuth":   140,
    "cam_elevation": -20,
    "cam_distance":  1.67,
    "cam_lookat":    [0.05, 0.03, 0.60],
    # ── bilateral scaling ─────────────────────────────────────────────────────
    # EE at start_config in torso frame: (0.396, ±0.133, 0.237)
    # bilateral_target_sep = |y_right| + |y_left| = 0.123 + 0.143 ≈ 0.266
    "bilateral_target_sep": 0.266,
    "workspace_center": {
        "left":  np.array([0.396,  0.143, 0.237], dtype=np.float32),
        "right": np.array([0.396, -0.123, 0.237], dtype=np.float32),
    },
}

ENV_CONFIG = RobotConfig(
    mjcf_path=_MJCF_MJX_PATH,
    joint_groups=_ARM_JOINTS,
    end_effectors=_EE_BODY,
)
