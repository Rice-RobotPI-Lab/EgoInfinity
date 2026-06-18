"""
RobotConfig dataclass used by JaxVecEnv.

Lives here (not in vec_env_jax.py) so per-robot config.py files can import it
without creating a circular dependency on the full simulation module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RobotConfig:
    """
    Robot description for JaxVecEnv.

    Parameters
    ----------
    mjcf_path : Path
        Path to the MJCF model file.
    joint_groups : dict[str, list[str]]
        Named groups of controllable joints.
        e.g. {"left": ["left_shoulder_pitch_joint", ...], "right": [...]}
        Each group is treated as one unit for step_joints / rollout.
    end_effectors : dict[str, str]
        Body to observe per group.
        e.g. {"left": "left_hand_frame", "right": "right_hand_frame"}
        Keys must be a subset of joint_groups keys.
    """
    mjcf_path: Path
    joint_groups: dict[str, list[str]]
    end_effectors: dict[str, str]
