"""
Post-process a bilateral arm trajectory to remove cross-arm self-collision.

Collision detection uses MuJoCo contacts; optimization uses numerical
gradient descent to push joints away from penetration while staying
close to the original trajectory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco


class CollisionFilter:
    """
    Per-frame gradient-based post-processor that removes cross-arm
    self-collision from a bilateral joint trajectory.

    Parameters
    ----------
    robot_cfg       : entry from ROBOT_CONFIGS (provides env_cls, scene_path,
                      wrist_body, start_config)
    margin          : minimum clearance [m] — contacts closer than this are penalised
    hand_sep_margin : minimum wrist-to-wrist distance [m] — extra bilateral
                      repulsion when EEs are too close (handles cases where
                      MuJoCo contacts are missed or margin alone is insufficient)
    lam_sep         : weight on the EE-separation soft constraint
    max_iter        : max gradient steps per colliding frame
    lr              : gradient step size [rad]
    lam_prox        : weight on ||q - q_orig||² to prevent large deviations
    fd_eps          : finite-difference epsilon [rad]
    verbose         : print progress summary
    """

    def __init__(
        self,
        robot_cfg:       dict,
        margin:          float = 0.02,
        hand_sep_margin: float = 0.05,
        lam_sep:         float = 5.0,
        max_iter:        int   = 80,
        lr:              float = 0.03,
        lam_prox:        float = 1.5,
        fd_eps:          float = 1e-3,
        verbose:         bool  = True,
    ):
        self.margin          = margin
        self.hand_sep_margin = hand_sep_margin
        self.lam_sep         = lam_sep
        self.max_iter        = max_iter
        self.lr              = lr
        self.lam_prox        = lam_prox
        self.fd_eps          = fd_eps
        self.verbose         = verbose

        self._env   = robot_cfg["env_cls"](
            mjcf_path=robot_cfg["scene_path"],
            start_config=robot_cfg["start_config"],
        )
        self._env.reset()
        self._model = self._env.model
        self._data  = self._env.data

        self._left_geoms, self._right_geoms = self._build_geom_sets(
            robot_cfg["wrist_body"]
        )

        # Body IDs for wrist-to-wrist separation cost
        _wb = robot_cfg["wrist_body"]
        self._wrist_bid_l = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, _wb["left"])
        self._wrist_bid_r = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, _wb["right"])

        if verbose:
            print(f"[CollisionFilter] "
                  f"left={len(self._left_geoms)} geoms  "
                  f"right={len(self._right_geoms)} geoms  "
                  f"margin={margin:.3f}m  hand_sep={hand_sep_margin:.3f}m")

    # ── geometry setup ────────────────────────────────────────────────────────

    def _chain_to_root(self, body_name: str) -> set[int]:
        """Body IDs on the path from body_name up to (not including) worldbody."""
        bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        ids: set[int] = set()
        while bid > 0:
            ids.add(bid)
            bid = int(self._model.body_parentid[bid])
        return ids

    def _build_geom_sets(self, wrist_body: dict) -> tuple[set[int], set[int]]:
        left_chain  = self._chain_to_root(wrist_body["left"])
        right_chain = self._chain_to_root(wrist_body["right"])
        shared      = left_chain & right_chain   # torso/base — excluded
        left_only   = left_chain  - shared
        right_only  = right_chain - shared

        left_geoms:  set[int] = set()
        right_geoms: set[int] = set()
        for gid in range(self._model.ngeom):
            bid = int(self._model.geom_bodyid[gid])
            if bid in left_only:
                left_geoms.add(gid)
            elif bid in right_only:
                right_geoms.add(gid)
        return left_geoms, right_geoms

    # ── collision helpers ─────────────────────────────────────────────────────

    def _forward(self, q_l: np.ndarray, q_r: np.ndarray) -> None:
        self._env.set_arm_joints("left",  q_l.astype(np.float64))
        self._env.set_arm_joints("right", q_r.astype(np.float64))
        mujoco.mj_forward(self._model, self._data)

    def _penetration(self) -> float:
        """Total margin-inclusive penetration depth over cross-arm contacts."""
        total = 0.0
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self._left_geoms  and g2 in self._right_geoms) or
                    (g1 in self._right_geoms and g2 in self._left_geoms)):
                total += max(0.0, self.margin - float(c.dist))
        return total

    def _hand_separation_cost(self) -> float:
        """Soft cost when wrist bodies are closer than hand_sep_margin."""
        pos_l = self._data.xpos[self._wrist_bid_l]
        pos_r = self._data.xpos[self._wrist_bid_r]
        dist = float(np.linalg.norm(pos_l - pos_r))
        return self.lam_sep * max(0.0, self.hand_sep_margin - dist) ** 2

    def _needs_correction(self) -> bool:
        """True if there is any geom penetration or hands are too close."""
        if self._penetration() > 0.0:
            return True
        pos_l = self._data.xpos[self._wrist_bid_l]
        pos_r = self._data.xpos[self._wrist_bid_r]
        dist = float(np.linalg.norm(pos_l - pos_r))
        return dist < self.hand_sep_margin

    # ── per-frame optimisation ────────────────────────────────────────────────

    def _loss(
        self,
        q_l: np.ndarray, q_r: np.ndarray,
        q_l_ref: np.ndarray, q_r_ref: np.ndarray,
    ) -> float:
        self._forward(q_l, q_r)
        pen  = self._penetration()
        sep  = self._hand_separation_cost()
        prox = self.lam_prox * (
            float(np.dot(q_l - q_l_ref, q_l - q_l_ref)) +
            float(np.dot(q_r - q_r_ref, q_r - q_r_ref))
        )
        return pen + sep + prox

    def _gradient(
        self,
        q_l: np.ndarray, q_r: np.ndarray,
        q_l_ref: np.ndarray, q_r_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = self.fd_eps
        f0  = self._loss(q_l, q_r, q_l_ref, q_r_ref)
        gl  = np.empty_like(q_l)
        gr  = np.empty_like(q_r)
        for i in range(len(q_l)):
            q_l[i] += eps
            gl[i]   = (self._loss(q_l, q_r, q_l_ref, q_r_ref) - f0) / eps
            q_l[i] -= eps
        for i in range(len(q_r)):
            q_r[i] += eps
            gr[i]   = (self._loss(q_l, q_r, q_l_ref, q_r_ref) - f0) / eps
            q_r[i] -= eps
        return gl, gr

    def _optimize_frame(
        self,
        q_l0: np.ndarray,
        q_r0: np.ndarray,
        q_l_ref: np.ndarray,
        q_r_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (q_l, q_r, resolved).

        q_l0/q_r0   : starting point (original IK solution for this frame)
        q_l_ref/q_r_ref : proximity anchor (previous corrected frame's output)
        """
        q_l = q_l0.copy()
        q_r = q_r0.copy()
        best_loss = float("inf")
        best_l, best_r = q_l.copy(), q_r.copy()

        for _ in range(self.max_iter):
            self._forward(q_l, q_r)
            if not self._needs_correction():
                best_l[:] = q_l
                best_r[:] = q_r
                return best_l, best_r, True
            loss = self._loss(q_l, q_r, q_l_ref, q_r_ref)
            if loss < best_loss:
                best_loss  = loss
                best_l[:] = q_l
                best_r[:] = q_r
            gl, gr = self._gradient(q_l, q_r, q_l_ref, q_r_ref)
            q_l -= self.lr * gl
            q_r -= self.lr * gr

        self._forward(best_l, best_r)
        return best_l, best_r, not self._needs_correction()

    # ── public API ────────────────────────────────────────────────────────────

    def process(
        self,
        q_left:  np.ndarray,   # (T, n_dof)
        q_right: np.ndarray,   # (T, n_dof)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove cross-arm self-collision from a bilateral joint trajectory.

        Frames are processed sequentially so each correction is anchored to
        the previous corrected frame, preventing jitter at collision boundaries.
        Always returns copies — does not modify inputs.
        """
        T = q_left.shape[0]
        q_l_out = q_left.copy()
        q_r_out = q_right.copy()
        n_bad = 0
        n_fixed = 0

        for t in range(T):
            self._forward(q_l_out[t], q_r_out[t])
            if not self._needs_correction():
                continue
            n_bad += 1
            # Proximity anchor = previous frame's output → corrections chain smoothly
            q_l_ref = q_l_out[t - 1] if t > 0 else q_l_out[t]
            q_r_ref = q_r_out[t - 1] if t > 0 else q_r_out[t]
            q_l_new, q_r_new, resolved = self._optimize_frame(
                q_l_out[t], q_r_out[t], q_l_ref, q_r_ref,
            )
            q_l_out[t] = q_l_new
            q_r_out[t] = q_r_new
            if resolved:
                n_fixed += 1

        if self.verbose:
            if n_bad == 0:
                print("[CollisionFilter] "
                      "trajectory already collision-free — skipping")
            else:
                n_remain = n_bad - n_fixed
                print(f"[CollisionFilter] "
                      f"fixed {n_fixed}/{n_bad}  remaining={n_remain}")

        return q_l_out, q_r_out
