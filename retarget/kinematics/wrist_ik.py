"""
Robot-agnostic wrist IK using pytorch_kinematics.

Implements Damped Least Squares (DLS) IK in PyTorch, supporting:
  - GPU-accelerated batch IK (multiple targets solved in parallel)
  - Differentiable FK (gradients flow through joint angles)
  - Joint limit clamping at every iteration
  - Warm-starting for trajectory solving
  - Null-space secondary objectives (limit margin, manipulability, smoothness,
    self-collision avoidance)

Coordinate frame
----------------
Target pose (pos, quat) must be expressed in the root body frame (the frame
the kinematic chain is built from).

Usage
-----
    # Unitree G1 (default)
    ik = WristIK(side="left", device="cuda")

    # Any other robot — pass config explicitly
    cfg = RobotIKConfig.simplified_athenazero("right")
    ik  = WristIK(side="right", robot=cfg)

    # Single target
    q, info = ik.solve(target_pos, target_quat)

    # Batch (B targets in parallel on GPU)
    q_batch, info = ik.solve_batch(target_pos_batch, target_quat_batch)

    # Trajectory with warm-starting
    q_traj, infos = ik.solve_trajectory(pos_traj, quat_traj)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_kinematics as pk

_ROBOTS_DIR = Path(__file__).parent.parent / "robots"


# ── per-robot IK configuration ────────────────────────────────────────────────

@dataclass
class RobotIKConfig:
    """
    All robot-specific constants needed by WristIK.

    Parameters
    ----------
    mjcf_path      : path to the robot MJCF file
    end_link_name  : name of the end-effector body in the MJCF
    root_link_name : name of the root body for the kinematic chain
                     (the "torso" of the robot — IK targets are expressed
                     in this body's frame)
    joint_limits   : (n_dof, 2) tensor of [lower, upper] limits in radians
    """
    mjcf_path:      Path
    end_link_name:  str
    root_link_name: str
    joint_limits:   torch.Tensor   # (n_dof, 2)

    # ── built-in configs ──────────────────────────────────────────────────

    @staticmethod
    def unitree_g1(side: str) -> "RobotIKConfig":
        """Unitree G1 7-DoF arm (default robot)."""
        _limits = {
            "left": torch.tensor([
                [-3.0892,  2.6704],  # shoulder_pitch
                [-1.5882,  2.2515],  # shoulder_roll
                [-2.6180,  2.6180],  # shoulder_yaw
                [-1.0472,  2.0944],  # elbow
                [-1.9722,  1.9722],  # wrist_roll
                [-1.6144,  1.6144],  # wrist_pitch
                [-1.6144,  1.6144],  # wrist_yaw
            ]),
            "right": torch.tensor([
                [-3.0892,  2.6704],
                [-2.2515,  1.5882],
                [-2.6180,  2.6180],
                [-1.0472,  2.0944],
                [-1.9722,  1.9722],
                [-1.6144,  1.6144],
                [-1.6144,  1.6144],
            ]),
        }
        _ee = {"left": "left_hand_frame", "right": "right_hand_frame"}
        return RobotIKConfig(
            mjcf_path      = _ROBOTS_DIR / "unitree_g1" / "g1.xml",
            end_link_name  = _ee[side],
            root_link_name = "torso_link",
            joint_limits   = _limits[side],
        )

    @staticmethod
    def simplified_athenazero(side: str) -> "RobotIKConfig":
        """Simplified AthenaZero 7-DoF arm."""
        _limits = {
            "right": torch.tensor([
                [-0.7854,  3.0369],  # RJ1
                [-3.2812,  0.1396],  # RJ2
                [-2.5133,  0.9250],  # RJ3
                [ 0.0000,  2.5831],  # RJ4
                [-1.5184,  1.5184],  # RJ5
                [-1.2217,  1.2217],  # RJ6
                [-0.3491,  0.3491],  # RJ7
            ]),
            "left": torch.tensor([
                [-3.0369,  0.7854],  # LJ1
                [-0.1396,  3.2812],  # LJ2
                [-0.9250,  2.5133],  # LJ3
                [-2.5831,  0.0000],  # LJ4
                [-1.5184,  1.5184],  # LJ5
                [-1.2217,  1.2217],  # LJ6
                [-0.3491,  0.3491],  # LJ7
            ]),
        }
        _ee = {"right": "RHAND", "left": "LHAND"}
        return RobotIKConfig(
            mjcf_path      = _ROBOTS_DIR / "simplified_athenazero" / "simplified_athenazero.xml",
            end_link_name  = _ee[side],
            root_link_name = "TL1",
            joint_limits   = _limits[side],
        )

    @staticmethod
    def franka(side: str) -> "RobotIKConfig":
        """Dual Franka FR3 — one 7-DoF arm (left at y=+0.5, right at y=-0.5)."""
        _limits = torch.tensor([
            [-2.7437,  2.7437],  # joint1
            [-1.7837,  1.7837],  # joint2
            [-2.9007,  2.9007],  # joint3
            [-3.0421, -0.1518],  # joint4
            [-2.8065,  2.8065],  # joint5
            [ 0.5445,  4.5169],  # joint6
            [-3.0159,  3.0159],  # joint7
        ])
        _ee = {"left": "left_ee", "right": "right_ee"}
        return RobotIKConfig(
            mjcf_path      = _ROBOTS_DIR / "franka_fr3" / "fr3_dual.xml",
            end_link_name  = _ee[side],
            root_link_name = "base",
            joint_limits   = _limits,
        )

    @staticmethod
    def robonaut2(side: str) -> "RobotIKConfig":
        """Dual Robonaut 2 — 7-DoF arm (joint0..6)."""
        _limits = {
            "left": torch.tensor([
                [-1.5700,  1.5700],  # joint0
                [-1.6600,  0.1750],  # joint1
                [-4.8900, -0.7900],  # joint2
                [-2.7900, -0.0870],  # joint3
                [-1.3000,  4.4500],  # joint4
                [-1.2200,  1.2200],  # joint5
                [-0.7850,  0.7850],  # joint6
            ]),
            "right": torch.tensor([
                [-1.5700,  1.5700],  # joint0
                [-1.6600,  0.1750],  # joint1
                [ 0.7850,  4.8900],  # joint2
                [-2.7900, -0.0870],  # joint3
                [-4.4500,  1.3100],  # joint4
                [-1.2200,  1.2200],  # joint5
                [-0.7850,  0.7850],  # joint6
            ]),
        }
        _ee = {"left": "left_hand_frame", "right": "right_hand_frame"}
        return RobotIKConfig(
            mjcf_path      = _ROBOTS_DIR / "robonaut2" / "r2.xml",
            end_link_name  = _ee[side],
            root_link_name = "torso_frame",
            joint_limits   = _limits[side],
        )

    @staticmethod
    def xlerobot(side: str) -> "RobotIKConfig":
        """XLeRobot 5-DoF SO-ARM100 arm (Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll)."""
        _limits = torch.tensor([
            [-2.1000,  2.1000],  # Rotation   (shoulder swing)
            [-0.1000,  3.4500],  # Pitch       (upper arm)
            [-0.2000,  3.1416],  # Elbow
            [-1.8000,  1.8000],  # Wrist_Pitch
            [-3.1416,  3.1416],  # Wrist_Roll
        ])
        _ee = {"left": "left_hand_frame", "right": "right_hand_frame"}
        return RobotIKConfig(
            mjcf_path      = _ROBOTS_DIR / "xlerobot" / "xlerobot_mjx.xml",
            end_link_name  = _ee[side],
            root_link_name = "torso_frame",
            joint_limits   = _limits,
        )


# ── default G1 config (kept for backward compatibility) ───────────────────────

_MJCF_PATH  = _ROBOTS_DIR / "unitree_g1" / "g1.xml"
_WRIST_BODY = {"left": "left_hand_frame", "right": "right_hand_frame"}
_JOINT_LIMITS = {
    s: RobotIKConfig.unitree_g1(s).joint_limits for s in ("left", "right")
}

# ── quaternion utilities ──────────────────────────────────────────────────────

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. Both (..., 4) in (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of unit quaternion (..., 4) in (w, x, y, z)."""
    return q * torch.tensor([1., -1., -1., -1.], device=q.device)


def _quat_error(q_cur: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    """
    Orientation error as axis-angle vector (..., 3).
    Both inputs (..., 4) in (w, x, y, z).
    """
    q_err = _quat_mul(q_tgt, _quat_conj(q_cur))
    # Short path: ensure w >= 0
    q_err = torch.where(q_err[..., :1] < 0, -q_err, q_err)
    return 2.0 * q_err[..., 1:]  # (w, x, y, z) -> axis * angle


# ── IK solver ─────────────────────────────────────────────────────────────────

class WristIK:
    """
    GPU-capable Damped Least Squares IK for one G1 arm.

    Parameters
    ----------
    side : "left" | "right"
    device : str
        PyTorch device ("cuda", "cpu"). Defaults to cuda if available.
    damping : float
        DLS damping factor λ.
    max_iter : int
        Maximum iterations per solve.
    tol_pos : float
        Position convergence threshold [m].
    tol_ori : float
        Orientation convergence threshold [rad].
    pos_weight, ori_weight : float
        Relative weighting in the error vector.
    """

    def __init__(
        self,
        side: str,
        device: Optional[str] = None,
        damping: float = 1e-3,
        max_iter: int = 30,
        tol_pos: float = 5e-3,
        tol_ori: float = 5e-2,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        # ── null-space secondary objectives ───────────────────────────────
        null_weight_lim: float = 0.5,
        # Weight for pulling joints toward the centre of their range.
        # Improves limit margin and is a proxy for self-collision avoidance.
        null_weight_man: float = 0.1,
        # Weight for maximising Yoshikawa manipulability.
        # Gradient computed numerically every `null_man_freq` iterations.
        null_man_freq: int = 5,
        # How often (in IK iterations) to recompute the manipulability gradient.
        null_weight_smo: float = 0.2,
        # Weight for trajectory smoothness (pull toward q_ref when provided).
        null_weight_col: float = 1.0,
        # Weight for self-collision avoidance (sphere-based link repulsion).
        col_margin: float = 0.08,
        # Minimum allowed distance [m] between non-adjacent link centres.
        null_col_freq: int = 5,
        # How often (in IK iterations) to recompute the self-collision gradient.
        # ── robot config (pass RobotIKConfig for non-G1 robots) ──────────
        robot: Optional[RobotIKConfig] = None,
        # ── default initial joint angles (used by solve_batch when q_init is None) ─
        q_default: Optional[np.ndarray] = None,
    ):
        self.side = side
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.damping = damping
        self.max_iter = max_iter
        self.tol_pos = tol_pos
        self.tol_ori = tol_ori
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.null_weight_lim = null_weight_lim
        self.null_weight_man = null_weight_man
        self.null_man_freq   = null_man_freq
        self.null_weight_smo = null_weight_smo
        self.null_weight_col = null_weight_col
        self.col_margin      = col_margin
        self.null_col_freq   = null_col_freq

        # Resolve robot config — default to G1 if not provided
        cfg = robot if robot is not None else RobotIKConfig.unitree_g1(side)

        # Build kinematic chain from MJCF.
        # - chdir to robot dir so relative asset paths resolve
        # - strip <freejoint> — pytorch_kinematics only handles hinge/slide
        import os, re
        _prev_dir = os.getcwd()
        os.chdir(cfg.mjcf_path.parent)
        try:
            mjcf_str = cfg.mjcf_path.read_text()
            mjcf_str = re.sub(r'<freejoint[^/]*/>', '', mjcf_str)
            mjcf_str = re.sub(r'<keyframe>.*?</keyframe>', '', mjcf_str, flags=re.DOTALL)
            self.chain = pk.build_serial_chain_from_mjcf(
                mjcf_str,
                end_link_name=cfg.end_link_name,
                root_link_name=cfg.root_link_name,
            )
            self.chain = self.chain.to(device=self.device)
        finally:
            os.chdir(_prev_dir)

        self.n_dof = len(self.chain.get_joint_parameter_names())
        self.limits = cfg.joint_limits.to(self.device)  # (n_dof, 2)
        self._has_analytic_jacobian = hasattr(self.chain, "jacobian")
        self.q_default = (
            torch.tensor(q_default, dtype=torch.float32).to(self.device)
            if q_default is not None else None
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, q: torch.Tensor) -> torch.Tensor:
        """Clamp (..., n_dof) to joint limits."""
        limits = self.limits.to(q.device)
        return torch.clamp(q, limits[:, 0], limits[:, 1])

    def _fk(self, q: torch.Tensor):
        """
        Forward kinematics. q: (B, n_dof) → pos (B, 3), quat (B, 4) wxyz.
        """
        tf = self.chain.forward_kinematics(q)
        mat = tf.get_matrix()          # (B, 4, 4)
        pos = mat[:, :3, 3]            # (B, 3)
        # Extract quaternion from rotation matrix
        rot = mat[:, :3, :3]           # (B, 3, 3)
        quat = pk.matrix_to_quaternion(rot)  # (B, 4) already (w,x,y,z)
        return pos, quat

    def _jacobian_fd(self, q: torch.Tensor) -> torch.Tensor:
        """
        Numerical Jacobian (6, n_dof) via finite differences.
        q: (B, n_dof). Returns (B, 6, n_dof).
        """
        B = q.shape[0]
        eps = 1e-4
        J = torch.zeros(B, 6, self.n_dof, device=self.device)
        pos0, quat0 = self._fk(q)
        for i in range(self.n_dof):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            pos1, quat1 = self._fk(self._clamp(q + dq))
            J[:, :3, i] = (pos1 - pos0) / eps
            J[:, 3:, i] = (_quat_error(quat0, quat1)) / eps
        return J

    def _jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """
        Jacobian (B, 6, n_dof), preferring the analytic implementation from
        pytorch_kinematics and falling back to finite differences if needed.
        """
        if not self._has_analytic_jacobian:
            return self._jacobian_fd(q)
        try:
            J = self.chain.jacobian(q)
            if J.shape[1] != 6:
                return self._jacobian_fd(q)
            return J
        except Exception:
            return self._jacobian_fd(q)

    def _manipulability_grad(self, q: torch.Tensor) -> torch.Tensor:
        """
        Numerical gradient of Yoshikawa manipulability w.r.t. q.
        q: (B, n_dof) → (B, n_dof).
        """
        eps = 1e-4
        m0 = self.manipulability(q)          # (B,)
        grad = torch.zeros_like(q)
        for i in range(self.n_dof):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            m1 = self.manipulability(self._clamp(q + dq))
            grad[:, i] = (m1 - m0) / eps
        return grad                          # (B, n_dof)

    def _fk_link_positions(self, q: torch.Tensor) -> torch.Tensor:
        """
        Positions of all chain links. q: (B, n_dof) → (B, n_links, 3).
        Uses pytorch_kinematics end_only=False to get intermediate transforms.
        """
        tfs = self.chain.forward_kinematics(q, end_only=False)
        # tfs is an OrderedDict {link_name: Transform3d}
        positions = torch.stack(
            [tf.get_matrix()[:, :3, 3] for tf in tfs.values()], dim=1
        )  # (B, n_links, 3)
        return positions

    def _self_collision_cost(self, q: torch.Tensor) -> torch.Tensor:
        """
        Sphere-based self-collision cost: sum of squared penetrations between
        non-adjacent link centres closer than col_margin.
        q: (B, n_dof) → (B,) cost (lower = no collision).
        """
        pos = self._fk_link_positions(q)   # (B, n_links, 3)
        n   = pos.shape[1]
        cost = torch.zeros(pos.shape[0], device=self.device)
        for i in range(n):
            for j in range(i + 2, n):      # skip adjacent links (i, i+1)
                dist = (pos[:, i] - pos[:, j]).norm(dim=-1)   # (B,)
                penetration = (self.col_margin - dist).clamp(min=0.0)
                cost = cost + penetration ** 2
        return cost                        # (B,)

    def _self_collision_grad(self, q: torch.Tensor) -> torch.Tensor:
        """
        Numerical gradient of self-collision cost w.r.t. q.
        Returns (B, n_dof) repulsive gradient (points away from collision).
        """
        eps  = 1e-4
        c0   = self._self_collision_cost(q)
        grad = torch.zeros_like(q)
        for i in range(self.n_dof):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            c1 = self._self_collision_cost(self._clamp(q + dq))
            grad[:, i] = (c1 - c0) / eps
        return -grad   # negate: we want to push cost down → move opposite to gradient

    # ── public API ────────────────────────────────────────────────────────────

    def solve_batch(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        q_init: Optional[torch.Tensor] = None,
        q_ref: Optional[torch.Tensor] = None,
        avoid_self_collision: bool = False,
    ) -> tuple:
        """
        Solve IK for a batch of targets in parallel.

        Parameters
        ----------
        target_pos           : (B, 3) target positions in world frame
        target_quat          : (B, 4) target orientations (w,x,y,z) in world frame
        q_init               : (B, n_dof) initial joint angles, or None (zeros)
        q_ref                : (B, n_dof) reference angles for smoothness (prev solution)
        avoid_self_collision : bool — add sphere-based self-collision repulsion in null-space

        Returns
        -------
        q    : (B, n_dof) solved joint angles
        info : dict with "converged" (B,), "pos_err" (B,), "ori_err" (B,), "iters"
        """
        B = target_pos.shape[0]
        target_pos  = target_pos.to(self.device)
        target_quat = target_quat.to(self.device)

        q_ref_t = q_ref.clone().to(self.device) if q_ref is not None else None

        if q_init is not None:
            q = q_init.clone().to(self.device)
        elif self.q_default is not None:
            q = self.q_default.unsqueeze(0).expand(B, -1).clone()
        else:
            q = torch.zeros(B, self.n_dof, device=self.device)
        q = self._clamp(q)

        I_n = torch.eye(self.n_dof, device=self.device).unsqueeze(0)  # (1, n, n)
        converged = torch.zeros(B, dtype=torch.bool, device=self.device)
        man_grad = torch.zeros(B, self.n_dof, device=self.device)
        col_grad = torch.zeros(B, self.n_dof, device=self.device)

        for i in range(self.max_iter):
            pos_cur, quat_cur = self._fk(q)

            err_pos = self.pos_weight * (target_pos - pos_cur)   # (B, 3)
            err_ori = self.ori_weight * _quat_error(quat_cur, target_quat)  # (B, 3)
            err = torch.cat([err_pos, err_ori], dim=-1)  # (B, 6)

            pos_err = err_pos.norm(dim=-1) / self.pos_weight
            if self.ori_weight > 0:
                ori_err = err_ori.norm(dim=-1) / self.ori_weight
                converged = (pos_err < self.tol_pos) & (ori_err < self.tol_ori)
            else:
                converged = pos_err < self.tol_pos
            if converged.all():
                break

            J = self._jacobian(q)  # (B, 6, n_dof)
            JJT = J @ J.transpose(-1, -2) + self.damping * torch.eye(
                6, device=self.device).unsqueeze(0)

            # DLS pseudoinverse: J^+ = J^T (J J^T + λI)^{-1}
            J_pinv = J.transpose(-1, -2) @ torch.linalg.solve(JJT,
                         torch.eye(6, device=self.device).unsqueeze(0).expand(B, -1, -1))
            dq_task = (J_pinv @ err.unsqueeze(-1)).squeeze(-1)  # (B, n_dof)

            # Null-space projector N = I - J^+ J
            N = I_n - J_pinv @ J  # (B, n_dof, n_dof)

            # Secondary objective: gradient of null-space cost
            dq0 = torch.zeros(B, self.n_dof, device=self.device)

            # 1. Pull joints toward range midpoint (limit margin)
            if self.null_weight_lim > 0:
                mid = (self.limits[:, 0] + self.limits[:, 1]) / 2.0  # (n_dof,)
                dq0 = dq0 + self.null_weight_lim * (mid - q)

            # 2. Manipulability gradient (recomputed every null_man_freq iters)
            if self.null_weight_man > 0:
                if i % self.null_man_freq == 0:
                    man_grad = self._manipulability_grad(q)
                dq0 = dq0 + self.null_weight_man * man_grad

            # 3. Smoothness: pull toward previous solution
            if self.null_weight_smo > 0 and q_ref_t is not None:
                dq0 = dq0 + self.null_weight_smo * (q_ref_t - q)

            # 4. Self-collision repulsion (sphere-based, optional)
            if avoid_self_collision and self.null_weight_col > 0:
                if i % self.null_col_freq == 0:
                    col_grad = self._self_collision_grad(q)
                dq0 = dq0 + self.null_weight_col * col_grad

            dq = dq_task + (N @ dq0.unsqueeze(-1)).squeeze(-1)
            q_new = self._clamp(q + dq)
            # Freeze frames that already converged so they can't overshoot
            q = torch.where(converged.unsqueeze(-1), q, q_new)

        pos_cur, quat_cur = self._fk(q)
        pos_err = (target_pos - pos_cur).norm(dim=-1)
        ori_err = _quat_error(quat_cur, target_quat).norm(dim=-1)
        if self.ori_weight > 0:
            converged = (pos_err < self.tol_pos) & (ori_err < self.tol_ori)
        else:
            converged = pos_err < self.tol_pos

        lower = self.limits[:, 0]
        upper = self.limits[:, 1]
        jl_margin = torch.minimum(q - lower, upper - q).min(dim=-1).values

        return q, {
            "converged":          converged,
            "pos_err":            pos_err,
            "ori_err":            ori_err,
            "iters":              i + 1,
            "joint_limit_margin": jl_margin,
            "manipulability":     self.manipulability(q),
        }

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        q_ref: Optional[np.ndarray] = None,
        avoid_self_collision: bool = False,
    ) -> tuple:
        """
        Solve IK for a single target. Numpy in, numpy out.

        Parameters
        ----------
        target_pos  : (3,) in world frame
        target_quat : (4,) (w,x,y,z) in world frame
        q_init      : (n_dof,) or None
        q_ref       : (n_dof,) reference for null-space smoothness, or None

        Returns
        -------
        q    : (n_dof,) numpy array
        info : dict with scalar "converged", "pos_err", "ori_err", "iters"
        """
        tp = torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)
        tq = torch.tensor(target_quat, dtype=torch.float32).unsqueeze(0)
        qi = torch.tensor(q_init, dtype=torch.float32).unsqueeze(0) \
            if q_init is not None else None
        qr = torch.tensor(q_ref, dtype=torch.float32).unsqueeze(0) \
            if q_ref is not None else None

        q_t, info = self.solve_batch(tp, tq, qi, qr,
                                     avoid_self_collision=avoid_self_collision)

        return q_t[0].cpu().numpy(), {
            "converged":          bool(info["converged"][0].item()),
            "pos_err":            float(info["pos_err"][0].item()),
            "ori_err":            float(info["ori_err"][0].item()),
            "iters":              info["iters"],
            "joint_limit_margin": float(info["joint_limit_margin"][0].item()),
            "manipulability":     float(info["manipulability"][0].item()),
        }

    def manipulability(self, q: torch.Tensor) -> torch.Tensor:
        """
        Yoshikawa manipulability index: sqrt(det(J_pos @ J_pos^T)).

        Parameters
        ----------
        q : (B, n_dof)

        Returns
        -------
        (B,) manipulability values (higher = further from singularity)
        """
        q = q.to(self.device)
        J    = self._jacobian(q)              # (B, 6, n_dof)
        Jp   = J[:, :3, :]                    # positional rows only (B, 3, n_dof)
        JJT  = Jp @ Jp.transpose(-1, -2)      # (B, 3, 3)
        det  = torch.linalg.det(JJT).clamp(min=0.0)
        return det.sqrt()                     # (B,)

    def solve_trajectory(
        self,
        target_pos_traj: Optional[np.ndarray],
        target_quat_traj: Optional[np.ndarray],
        q_init: Optional[np.ndarray] = None,
        verbose: bool = False,
        desc: str = "",
    ) -> tuple:
        """
        Solve IK sequentially, warm-starting each frame from the previous solution.
        Null-space smoothness (q_ref = q_prev) ensures temporal consistency.

        Parameters
        ----------
        target_pos_traj  : (T, 3) or None
        target_quat_traj : (T, 4) or None
        q_init           : (n_dof,) initial joint angles for frame 0
        verbose          : print per-frame progress
        desc             : label shown in progress output

        Returns
        -------
        q_traj : (T, n_dof)
        infos  : list of T info dicts
        """
        assert target_pos_traj is not None or target_quat_traj is not None
        T = (target_pos_traj if target_pos_traj is not None
             else target_quat_traj).shape[0]

        q_traj = np.zeros((T, self.n_dof))
        infos = []
        if q_init is not None:
            q = q_init
        elif self.q_default is not None:
            q = self.q_default.cpu().numpy()
        else:
            q = None

        for t in range(T):
            tp = target_pos_traj[t] if target_pos_traj is not None else None
            tq = target_quat_traj[t] if target_quat_traj is not None else None

            # If one is None, hold current FK value
            if tp is None:
                qi_t = torch.tensor(q if q is not None else np.zeros(self.n_dof),
                                    dtype=torch.float32).unsqueeze(0).to(self.device)
                pos_cur, _ = self._fk(qi_t)
                tp = pos_cur[0].cpu().numpy()
            if tq is None:
                qi_t = torch.tensor(q if q is not None else np.zeros(self.n_dof),
                                    dtype=torch.float32).unsqueeze(0).to(self.device)
                _, quat_cur = self._fk(qi_t)
                tq = quat_cur[0].cpu().numpy()

            q_sol, info = self.solve(tp, tq, q_init=q, q_ref=None)
            q_traj[t] = q_sol
            infos.append(info)
            if verbose:
                tag = f"[{desc}] " if desc else ""
                n_conv = sum(info["converged"] for info in infos)
                n_fail = (t + 1) - n_conv
                filled = int(30 * (t + 1) / T)
                bar = "█" * filled + "░" * (30 - filled)
                print(f"\r  {tag}|{bar}| {t+1}/{T}  fail={n_fail}",
                      end="", flush=True)
            q = q_sol

        if verbose:
            tag = f"[{desc}] " if desc else ""
            n_conv = sum(info["converged"] for info in infos)
            print(f"\r  {tag}|{'█'*30}| {T}/{T}  converged={n_conv}/{T}")

        return q_traj, infos

    def refine_manipulability(
        self,
        q_traj: np.ndarray,
        n_iter: int = 20,
        step_size: float = 0.05,
        smo_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Improve manipulability per frame via sequential null-space gradient ascent.

        Frames are processed in order so each frame is initialised from the
        previous solution. The null-space update blends manipulability gradient
        ascent with a pull toward the previous frame (temporal smoothness):
            dq0 = ∇m(q) / ||∇m(q)|| + smo_weight * (q_prev - q)
            dq  = N @ dq0   (projected into null-space → EE preserved)
            q   = clamp(q + step_size * dq)

        Parameters
        ----------
        q_traj     : (T, n_dof) joint trajectory
        n_iter     : null-space gradient steps per frame
        step_size  : step size [rad]
        smo_weight : weight on temporal smoothness pull (higher = smoother,
                     lower = more manipulability gain)

        Returns
        -------
        q_refined : (T, n_dof)
        """
        T = q_traj.shape[0]
        q_out = q_traj.copy()
        I_n = torch.eye(self.n_dof, device=self.device).unsqueeze(0)  # (1, n, n)

        for t in range(T):
            q = torch.tensor(q_out[t], dtype=torch.float32,
                             device=self.device).unsqueeze(0)  # (1, n_dof)
            q_prev = torch.tensor(q_out[t - 1] if t > 0 else q_out[t],
                                  dtype=torch.float32, device=self.device).unsqueeze(0)

            for _ in range(n_iter):
                J = self._jacobian(q)  # (1, 6, n_dof)
                JJT = J @ J.transpose(-1, -2) + self.damping * torch.eye(
                    6, device=self.device).unsqueeze(0)
                J_pinv = J.transpose(-1, -2) @ torch.linalg.solve(
                    JJT, torch.eye(6, device=self.device).unsqueeze(0))
                N = I_n - J_pinv @ J  # (1, n_dof, n_dof)

                man_grad = self._manipulability_grad(q)  # (1, n_dof)
                man_grad = man_grad / (man_grad.norm(dim=-1, keepdim=True) + 1e-8)
                smo_grad = smo_weight * (q_prev - q)     # (1, n_dof)
                dq0 = man_grad + smo_grad
                dq = (N @ dq0.unsqueeze(-1)).squeeze(-1)
                q = self._clamp(q + step_size * dq)

            q_out[t] = q[0].cpu().numpy()

        return q_out

    def smooth_trajectory(
        self,
        q_init_traj: np.ndarray,
        target_pos_traj: np.ndarray,
        target_quat_traj: Optional[np.ndarray] = None,
        pos_weight: float = 50.0,
        ori_weight: float = 1.0,
        vel_weight: float = 1.0,
        acc_weight: float = 0.1,
        n_iter: int = 100,
    ) -> np.ndarray:
        """
        Post-process a joint trajectory globally for smoothness while tracking EE targets.

        Minimizes over the full trajectory:
            pos_weight * mean ||FK(q_t) - p_t||^2
          + ori_weight * mean ||quat_err(q_t, o_t)||^2
          + vel_weight * mean ||q_{t+1} - q_t||^2
          + acc_weight * mean ||q_{t+2} - 2*q_{t+1} + q_t||^2

        Uses L-BFGS with joint-limit projection after each step.

        Parameters
        ----------
        q_init_traj      : (T, n_dof) warm-start (from solve_trajectory)
        target_pos_traj  : (T, 3) EE position targets
        target_quat_traj : (T, 4) EE orientation targets, or None
        pos_weight       : weight for EE position tracking
        ori_weight       : weight for EE orientation tracking
        vel_weight       : weight for joint velocity (smoothness)
        acc_weight       : weight for joint acceleration
        n_iter           : number of L-BFGS outer steps

        Returns
        -------
        q_smooth : (T, n_dof)
        """
        T = q_init_traj.shape[0]
        tp = torch.tensor(target_pos_traj, dtype=torch.float32).to(self.device)
        tq = (torch.tensor(target_quat_traj, dtype=torch.float32).to(self.device)
              if target_quat_traj is not None and ori_weight > 0 else None)

        lo = self.limits[:, 0]
        hi = self.limits[:, 1]

        q = torch.tensor(q_init_traj, dtype=torch.float32,
                         device=self.device).clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([q], lr=1.0, max_iter=20,
                                  tolerance_grad=1e-7, tolerance_change=1e-9,
                                  line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            q_c = torch.clamp(q, lo, hi)
            pos_cur, quat_cur = self._fk(q_c)
            loss = pos_weight * ((pos_cur - tp) ** 2).mean()
            if tq is not None:
                loss = loss + ori_weight * (_quat_error(quat_cur, tq) ** 2).mean()
            if T > 1:
                dq = q_c[1:] - q_c[:-1]
                loss = loss + vel_weight * (dq ** 2).mean()
                if acc_weight > 0 and T > 2:
                    loss = loss + acc_weight * ((dq[1:] - dq[:-1]) ** 2).mean()
            loss.backward()
            return loss

        for _ in range(n_iter):
            opt.step(closure)
            with torch.no_grad():
                q.data.clamp_(lo, hi)

        return q.detach().clamp(lo, hi).cpu().numpy()
