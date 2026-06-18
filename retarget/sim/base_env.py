"""
Abstract base class for robot arm simulation environments.

Any robot env must subclass BaseEnv and implement the abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """
    Interface for a single-robot arm simulation environment.

    Conventions
    -----------
    - "side" is always "left" or "right"
    - Observations are dicts keyed by side, each with "pos" (3,) and "quat" (4,) (w,x,y,z)
    - Joint trajectories are (T, n_arm_dof) arrays
    - Loss dicts always contain: "pos", "ori", "joint_limit", "total"
    """

    @abstractmethod
    def reset(self) -> dict:
        """Reset to home pose. Returns initial observation."""

    @abstractmethod
    def set_arm_joints(self, side: str, q: np.ndarray):
        """Set joint angles for one arm and forward-simulate."""

    @abstractmethod
    def step_joints(self, action: dict) -> dict:
        """
        Set joint angles for one or both arms, forward-simulate, return obs.

        Parameters
        ----------
        action : dict with keys "left" and/or "right", each (n_dof,) ndarray
        """

    @abstractmethod
    def get_wrist_pose(self, side: str) -> tuple:
        """Return (pos, quat) of the wrist end-effector in world frame."""


