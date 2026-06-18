import numpy as np

from sim.robot_config import RobotConfig
from sim.robots.franka.env import FrankaEnv, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY

_FR3_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7854], dtype=np.float32)

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              FrankaEnv,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             "franka",
    "retargeter":           "franka",
    # ── trajectory sampling ───────────────────────────────────────────────────
    "apply_workspace_bias": False,
    # ── MuJoCo body names ─────────────────────────────────────────────────────
    "torso_body":           "base",
    "wrist_body":           {"left": "left_ee", "right": "right_ee"},
    "extra_bodies":         [],
    # ── joint configurations ──────────────────────────────────────────────────
    # zero_config: joint4 and joint6 clamped to nearest valid limits
    "zero_config": {
        "left":  np.array([0.0, 0.0, 0.0, -0.1518, 0.0, 0.5445, 0.0], dtype=np.float32),
        "right": np.array([0.0, 0.0, 0.0, -0.1518, 0.0, 0.5445, 0.0], dtype=np.float32),
    },
    "home_config": {
        "left":  _FR3_HOME.copy(),
        "right": _FR3_HOME.copy(),
    },
    # Optimized for: x≈0.55, y≈±0.214 (G1-like 0.43 m bilateral), z≈0.57,
    # EE y-axis forward (+x_base), z-axis inward (left→-y_base, right→+y_base),
    # perfect bilateral symmetry, min joint-limit margin 0.84 rad, manip=0.092.
    "start_config": {
        "left":  np.array([ 0.8409,  0.9450, -1.7640, -1.8648, -0.6507, 2.6473, -0.7854], dtype=np.float32),
        "right": np.array([-0.8409,  0.9450,  1.7640, -1.8648,  0.6507, 2.6473, -0.7854], dtype=np.float32),
    },
    # ── viewer camera ─────────────────────────────────────────────────────────
    "cam_azimuth":   140,
    "cam_elevation": -10,
    "cam_distance":  3.0,
    "cam_lookat":    [0.23, 0.006, 0.36],
}

ENV_CONFIG = RobotConfig(
    mjcf_path=_MJCF_MJX_PATH,
    joint_groups=_ARM_JOINTS,
    end_effectors=_EE_BODY,
)
