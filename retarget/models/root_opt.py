"""
RootPoseOptimizer — post-processing class to find the single best robot root
frame SE(3) pose for a full motion sequence given per-clip model predictions.

Uses WristIK (ik/wrist_ik.py) for IK solving and manipulability computation.

Pipeline
--------
1. ``generate_hypotheses(clips)``
       Run NeuralRootFrameEstimator on each 2-second clip → N SE(3) hypotheses.

2. ``cluster(Rs, ts)``
       K-medoids on SE(3) geodesic distance → K representative candidates.

3. ``score(R_cand, t_cand, left_cam, right_cam)``
       For each candidate, transform wrist poses to root frame, run batch IK,
       compute quality metrics:
           • IK convergence rate   — fraction of frames IK solved
           • Mean manipulability   — √det(J_pos J_pos^T), arm away from singularity
           • Joint smoothness      — negative mean squared joint velocity

4. ``optimize(clips, left_full, right_full)``
       Full pipeline: generate → cluster → score → return best SE(3) + IK solution.

Example
-------
    from models.root_opt import RootPoseOptimizer

    opt = RootPoseOptimizer(model, device="cuda")

    result = opt.optimize(clips, left_full, right_full, g_cam=g_cam_tensor)
    print(result.R_torso, result.t_torso)
    result.save("best_root.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from kinematics.wrist_ik import WristIK, RobotIKConfig
from sim.robots import SAMPLE_CONFIGS


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Output of RootPoseOptimizer.optimize()."""
    R_torso:      np.ndarray          # (3, 3) best torso rotation in camera frame
    t_torso:      np.ndarray          # (3,)   best torso origin in camera frame
    q_left:       np.ndarray          # (T, n_dof) joint angles for left arm
    q_right:      np.ndarray          # (T, n_dof) joint angles for right arm
    scores:       dict                # per-metric scores for the best candidate
    info_l:       dict = field(default_factory=dict)   # IK info for left arm
    info_r:       dict = field(default_factory=dict)   # IK info for right arm
    all_scores:   list[dict] = field(default_factory=list)
    R_candidates: Optional[np.ndarray] = None   # (K, 3, 3)
    t_candidates: Optional[np.ndarray] = None   # (K, 3)

    def save(self, path: str | Path) -> None:
        """Save result to .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            R_torso = self.R_torso,
            t_torso = self.t_torso,
            q_left  = self.q_left,
            q_right = self.q_right,
            **{f"score_{k}": v for k, v in self.scores.items()
               if isinstance(v, (int, float, np.ndarray))},
        )


# ── SE(3) clustering ──────────────────────────────────────────────────────────

def _se3_distance(R1, t1, R2, t2, rot_scale: float = 1.0) -> float:
    """Geodesic SE(3) distance: ||Log(R1^T R2)|| + rot_scale * ||t1 - t2||."""
    return float(_Rot.from_matrix(R1.T @ R2).magnitude()) + \
           rot_scale * float(np.linalg.norm(t1 - t2))


def cluster_se3(
    Rs:        np.ndarray,   # (N, 3, 3)
    ts:        np.ndarray,   # (N, 3)
    K:         int,
    n_iter:    int   = 50,
    rot_scale: float = 1.0,
    seed:      int   = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-medoids clustering on SE(3) using geodesic distance.

    Returns
    -------
    R_centers : (K, 3, 3)
    t_centers : (K, 3)
    labels    : (N,) cluster assignment for each hypothesis
    """
    N   = len(Rs)
    K   = min(K, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, K, replace=False)

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iter):
        for i in range(N):
            dists    = [_se3_distance(Rs[i], ts[i], Rs[idx[k]], ts[idx[k]], rot_scale)
                        for k in range(K)]
            labels[i] = int(np.argmin(dists))

        new_idx = idx.copy()
        for k in range(K):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue
            best, best_d = members[0], float("inf")
            for m in members:
                d = sum(_se3_distance(Rs[m], ts[m], Rs[j], ts[j], rot_scale)
                        for j in members)
                if d < best_d:
                    best_d, best = d, m
            new_idx[k] = best

        if np.array_equal(new_idx, idx):
            break
        idx = new_idx

    return Rs[idx], ts[idx], labels


# ── q_init sampling (shared by optimizer and eval scripts) ───────────────────

def sample_q_inits(
    n:        int,
    rng:      np.random.Generator,
    ik_left,
    ik_right,
    ik_robot: Optional[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n diverse q_inits using the same per-robot jitter as training."""
    robot = ik_robot or "g1"

    def _get_default(ik):
        return ik.q_default.cpu().numpy() if ik.q_default is not None \
               else np.zeros(ik.n_dof, dtype=np.float32)

    sc = SAMPLE_CONFIGS.get(robot)

    def _jitter(q_def, ik, side):
        lo    = ik.limits[:, 0].cpu().numpy()
        hi    = ik.limits[:, 1].cpu().numpy()
        n_dof = ik.n_dof
        q = np.tile(q_def, (n, 1))
        if sc is not None:
            lat = sc["lateral_joint"]
            q[:, lat["index"]] += rng.uniform(*lat[side], n).astype(np.float32)
            for jidx, (lo_j, hi_j) in sc["proximal_jitter"].items():
                q[:, jidx] += rng.uniform(lo_j, hi_j, n).astype(np.float32)
        else:
            jitter = (hi - lo) * 0.10
            q += rng.uniform(-jitter, jitter, (n, n_dof)).astype(np.float32)
        q[0] = q_def   # first sample always unperturbed
        return np.clip(q, lo, hi)

    q_inits_l = _jitter(_get_default(ik_left),  ik_left,  "left")
    q_inits_r = _jitter(_get_default(ik_right), ik_right, "right")
    return q_inits_l, q_inits_r


def compute_scores(
    q_l_flat:     "torch.Tensor",   # (B*T, n_dof)
    q_r_flat:     "torch.Tensor",
    info_l:       dict,
    info_r:       dict,
    B:            int,
    T:            int,
    ik_left_traj,
    ik_right_traj,
    w_ik:         float = 1.0,
    w_man:        float = 0.5,
    w_smo:        float = 0.1,
    w_lim:        float = 0.5,
    w_col:        float = 0.5,
) -> dict:
    """
    Compute all scoring metrics for B pairs in parallel.

    Returns dict with (B,) arrays:
        scores, ik_rates, mans, roughnesses, lim_margins, col_costs,
        q_left (B, T, n_dof_l), q_right (B, T, n_dof_r)
    """
    import torch as _torch
    n_dof_l = q_l_flat.shape[-1]
    n_dof_r = q_r_flat.shape[-1]
    q_l_np  = q_l_flat.reshape(B, T, n_dof_l).cpu().numpy()
    q_r_np  = q_r_flat.reshape(B, T, n_dof_r).cpu().numpy()

    ik_rates = 0.5 * (
        info_l["converged"].reshape(B, T).float().mean(dim=1) +
        info_r["converged"].reshape(B, T).float().mean(dim=1)
    ).cpu().numpy()

    mans = 0.5 * (
        ik_left_traj.manipulability(q_l_flat).reshape(B, T).mean(dim=1) +
        ik_right_traj.manipulability(q_r_flat).reshape(B, T).mean(dim=1)
    ).cpu().numpy()

    roughnesses = 0.5 * (
        np.mean(np.diff(q_l_np, axis=1) ** 2, axis=(1, 2)) +
        np.mean(np.diff(q_r_np, axis=1) ** 2, axis=(1, 2))
    )

    def _lim_batch(q_np, limits):
        lo  = limits[:, 0].cpu().numpy()
        hi  = limits[:, 1].cpu().numpy()
        rng = hi - lo
        return np.minimum((q_np - lo) / rng, (hi - q_np) / rng).mean(axis=(1, 2))

    lim_margins = 0.5 * (
        _lim_batch(q_l_np, ik_left_traj.limits) +
        _lim_batch(q_r_np, ik_right_traj.limits)
    )

    col_costs = 0.5 * (
        ik_left_traj._self_collision_cost(q_l_flat).reshape(B, T).mean(dim=1) +
        ik_right_traj._self_collision_cost(q_r_flat).reshape(B, T).mean(dim=1)
    ).cpu().numpy()

    scores = (w_ik  * ik_rates
            + w_man * mans
            + w_lim * lim_margins
            - w_smo * roughnesses
            - w_col * col_costs)

    return {
        "scores":      scores,
        "ik_rates":    ik_rates,
        "mans":        mans,
        "roughnesses": roughnesses,
        "lim_margins": lim_margins,
        "col_costs":   col_costs,
        "q_left":      q_l_np,
        "q_right":     q_r_np,
    }


def _interpolate_failed(
    q_traj:    np.ndarray,
    converged,
    q_default: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Linearly interpolate joint angles at frames where IK failed.
    Falls back to nearest good frame if no good neighbour exists on one side.
    If all frames failed, fills with q_default (start config) if provided,
    otherwise returns q_traj unchanged.
    """
    if hasattr(converged, "cpu"):
        converged = converged.cpu().numpy()
    conv = np.asarray(converged, dtype=bool)
    good = np.where(conv)[0]
    if len(good) == len(conv):
        return q_traj
    if len(good) == 0:
        if q_default is not None:
            q = q_traj.copy()
            q[:] = q_default
            return q
        return q_traj
    q = q_traj.copy()
    for t in np.where(~conv)[0]:
        before = good[good < t]
        after  = good[good > t]
        if len(before) == 0:
            q[t] = q[after[0]]
        elif len(after) == 0:
            q[t] = q[before[-1]]
        else:
            t0, t1 = int(before[-1]), int(after[0])
            alpha  = (t - t0) / (t1 - t0)
            q[t]   = (1.0 - alpha) * q[t0] + alpha * q[t1]
    return q


# ── main class ────────────────────────────────────────────────────────────────

class RootPoseOptimizer:
    """
    Post-processing optimizer that finds the single best robot root frame SE(3)
    pose for a full motion sequence.

    Parameters
    ----------
    model      : NeuralRootFrameEstimator (deterministic or flow)
    device     : torch device
    w_ik       : weight for IK convergence rate in composite score
    w_man      : weight for mean manipulability
    w_smo      : weight (penalty) for joint-space roughness
    w_lim      : weight for joint-limit margin (avoids arms at stretch/limit)
    n_clusters : number of SE(3) cluster centres to evaluate
    ik_max_iter: max IK iterations used during scoring
    """

    def __init__(
        self,
        model,
        device:      str | torch.device = "cpu",
        w_ik:        float = 1.0,
        w_man:       float = 0.5,
        w_smo:       float = 0.1,
        w_lim:       float = 0.5,
        w_col:       float = 0.5,
        n_clusters:           int   = 5,
        ik_max_iter:          int   = 50,
        avoid_self_collision: bool  = False,
        ik_robot:             Optional[str] = None,
        workspace_center:     Optional[dict] = None,
        start_config:         Optional[dict] = None,
        ori_weight:           Optional[float] = None,
        tol_pos:              Optional[float] = None,
        null_weight_man:      Optional[float] = None,
        null_weight_smo:      Optional[float] = None,
        n_q_init:             int   = 5,
    ):
        self.model      = model
        self.device     = torch.device(device)
        self.w_ik       = w_ik
        self.w_man      = w_man
        self.w_smo      = w_smo
        self.w_lim      = w_lim
        self.w_col      = w_col
        self.n_clusters = n_clusters
        self.n_q_init   = n_q_init

        self.avoid_self_collision = avoid_self_collision
        self.workspace_center     = workspace_center
        self.ik_robot             = ik_robot
        import numpy as _np
        _q_def_l  = _np.asarray(start_config["left"],  dtype=_np.float32) if start_config else None
        _q_def_r  = _np.asarray(start_config["right"], dtype=_np.float32) if start_config else None
        _ori_w    = ori_weight      if ori_weight      is not None else 1.0
        _tol_pos  = tol_pos        if tol_pos         is not None else (0.1 if ik_robot == "xlerobot" else 0.01)
        _null_man = null_weight_man if null_weight_man is not None else (0.5 if ik_robot == "franka" else 0.1)
        _null_smo = null_weight_smo if null_weight_smo is not None else (0.5 if ik_robot == "franka" else 0.2)
        if ik_robot is not None and ik_robot != "g1":
            ik_cfg_factory = getattr(RobotIKConfig, ik_robot)
            # Frame-0 IK: null-space ON to find good branch
            self.ik_left  = WristIK("left",  robot=ik_cfg_factory("left"),
                                    max_iter=ik_max_iter, tol_pos=_tol_pos,
                                    ori_weight=_ori_w,
                                    null_weight_man=_null_man, null_weight_smo=_null_smo,
                                    device=str(self.device), q_default=_q_def_l)
            self.ik_right = WristIK("right", robot=ik_cfg_factory("right"),
                                    max_iter=ik_max_iter, tol_pos=_tol_pos,
                                    ori_weight=_ori_w,
                                    null_weight_man=_null_man, null_weight_smo=_null_smo,
                                    device=str(self.device), q_default=_q_def_r)
            # Trajectory IK: null-space OFF for temporal consistency
            self.ik_left_traj  = WristIK("left",  robot=ik_cfg_factory("left"),
                                          max_iter=ik_max_iter, tol_pos=_tol_pos,
                                          ori_weight=_ori_w,
                                          null_weight_lim=0.0, null_weight_man=0.0,
                                          null_weight_smo=0.0,
                                          device=str(self.device), q_default=_q_def_l)
            self.ik_right_traj = WristIK("right", robot=ik_cfg_factory("right"),
                                          max_iter=ik_max_iter, tol_pos=_tol_pos,
                                          ori_weight=_ori_w,
                                          null_weight_lim=0.0, null_weight_man=0.0,
                                          null_weight_smo=0.0,
                                          device=str(self.device), q_default=_q_def_r)
        else:
            self.ik_left  = WristIK(side="left",  max_iter=ik_max_iter, tol_pos=_tol_pos,
                                    ori_weight=_ori_w,
                                    null_weight_man=_null_man, null_weight_smo=_null_smo,
                                    device=str(self.device), q_default=_q_def_l)
            self.ik_right = WristIK(side="right", max_iter=ik_max_iter, tol_pos=_tol_pos,
                                    ori_weight=_ori_w,
                                    null_weight_man=_null_man, null_weight_smo=_null_smo,
                                    device=str(self.device), q_default=_q_def_r)
            self.ik_left_traj  = WristIK(side="left",  max_iter=ik_max_iter, tol_pos=_tol_pos,
                                          ori_weight=_ori_w,
                                          null_weight_lim=0.0, null_weight_man=0.0,
                                          null_weight_smo=0.0,
                                          device=str(self.device), q_default=_q_def_l)
            self.ik_right_traj = WristIK(side="right", max_iter=ik_max_iter, tol_pos=_tol_pos,
                                          ori_weight=_ori_w,
                                          null_weight_lim=0.0, null_weight_man=0.0,
                                          null_weight_smo=0.0,
                                          device=str(self.device), q_default=_q_def_r)

    # ── hypothesis generation ─────────────────────────────────────────────

    def generate_hypotheses(
        self,
        clips: list[dict],
        g_cam: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the alignment model on a list of clips.

        Parameters
        ----------
        clips : list of dicts, each with:
            "left_traj"  : (T, 7) np.ndarray  wrist poses in camera frame
            "right_traj" : (T, 7) np.ndarray
        g_cam : (1, 3) gravity in camera frame, or None

        Returns
        -------
        Rs : (N, 3, 3)
        ts : (N, 3)
        """
        Rs, ts = [], []
        self.model.eval()

        for clip in clips:
            left_np  = clip.get("left_traj")
            right_np = clip.get("right_traj")
            if left_np is None or right_np is None:
                continue
            if np.isnan(left_np).any() or np.isnan(right_np).any():
                continue

            left_t  = torch.tensor(left_np,  dtype=torch.float32).unsqueeze(0).to(self.device)
            right_t = torch.tensor(right_np, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.model.mode == "flow":
                    R_p, t_p = self.model.sample(left_t, right_t, g_cam=g_cam)
                else:
                    R_p, t_p = self.model(left_t, right_t, g_cam=g_cam)

            Rs.append(R_p[0].cpu().numpy())
            ts.append(t_p[0].cpu().numpy())

        if not Rs:
            raise ValueError("No valid clips — all had NaN or missing hands.")

        return np.stack(Rs), np.stack(ts)

    # ── candidate clustering ──────────────────────────────────────────────

    def cluster(
        self,
        Rs: np.ndarray,
        ts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-medoids → (K, 3, 3), (K, 3) representative candidates."""
        K = min(self.n_clusters, len(Rs))
        R_cands, t_cands, _ = cluster_se3(Rs, ts, K=K)
        return R_cands, t_cands

    # ── single-candidate scoring ──────────────────────────────────────────

    def _ee_targets_in_torso_frame(
        self,
        R_cand:    np.ndarray,   # (3, 3)
        t_cand:    np.ndarray,   # (3,)
        left_cam:  np.ndarray,   # (T, 7)
        right_cam: np.ndarray,   # (T, 7)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform camera-frame wrist poses into torso frame for IK.

        Returns (left_pos, left_quat, right_pos, right_quat) all (T, 3/4).
        """
        T   = left_cam.shape[0]
        R_T = R_cand.T

        left_pos  = (R_T @ (left_cam[:,  :3] - t_cand).T).T
        right_pos = (R_T @ (right_cam[:, :3] - t_cand).T).T

        if self.workspace_center is not None:
            left_pos  = left_pos  - left_pos.mean(0)  + self.workspace_center["left"]
            right_pos = right_pos - right_pos.mean(0) + self.workspace_center["right"]

        left_rot  = np.stack([R_T @ _Rot.from_quat(left_cam[ f, [4,5,6,3]]).as_matrix()
                               for f in range(T)])
        right_rot = np.stack([R_T @ _Rot.from_quat(right_cam[f, [4,5,6,3]]).as_matrix()
                               for f in range(T)])

        left_quat  = _Rot.from_matrix(left_rot ).as_quat()[:, [3,0,1,2]]  # wxyz
        right_quat = _Rot.from_matrix(right_rot).as_quat()[:, [3,0,1,2]]

        return left_pos, left_quat, right_pos, right_quat

    def _sample_q_inits(self, n: int, rng: np.random.Generator):
        return sample_q_inits(n, rng, self.ik_left, self.ik_right, self.ik_robot)

    def _compute_scores(self, q_l_flat, q_r_flat, info_l, info_r, B, T) -> dict:
        return compute_scores(q_l_flat, q_r_flat, info_l, info_r, B, T,
                              self.ik_left_traj, self.ik_right_traj,
                              self.w_ik, self.w_man, self.w_smo, self.w_lim, self.w_col)

    # ── full pipeline ─────────────────────────────────────────────────────

    def optimize(
        self,
        clips:     list[dict],
        left_cam:  np.ndarray,
        right_cam: np.ndarray,
        g_cam:     Optional[torch.Tensor] = None,
        verbose:   bool = True,
    ) -> OptimizationResult:
        """
        Full pipeline: generate hypotheses → cluster → score → return best.

        Parameters
        ----------
        clips     : list of dicts with "left_traj" / "right_traj" (T, 7)
        left_cam  : (T_all, 7) full-sequence left wrist trajectory (camera frame)
        right_cam : (T_all, 7) full-sequence right wrist trajectory (camera frame)
        g_cam     : (1, 3) gravity in camera frame, or None
        verbose   : print progress

        Returns
        -------
        OptimizationResult
        """
        if verbose:
            print(f"[RootPoseOptimizer] {len(clips)} clips, "
                  f"{len(left_cam)} scoring frames")

        Rs, ts = self.generate_hypotheses(clips, g_cam=g_cam)
        if verbose:
            print(f"  → {len(Rs)} valid hypotheses")

        K = min(self.n_clusters, len(Rs))
        R_cands, t_cands = self.cluster(Rs, ts)
        if verbose:
            print(f"  Clustering → {K} candidates  |  q_inits per candidate: {self.n_q_init}")
            print(f"  {'cand':>4}  {'init':>4}  {'score':>8}  {'IK%':>6}  {'man':>8}  {'lim':>8}  {'rough':>10}")
            print(f"  {'-'*58}")

        rng = np.random.default_rng(0)
        q_inits_l, q_inits_r = self._sample_q_inits(self.n_q_init, rng)

        M  = self.n_q_init
        KM = K * M

        # ── EE targets for all K candidates ───────────────────────────────
        all_lp, all_lq, all_rp, all_rq = [], [], [], []
        for k in range(K):
            lp, lq, rp, rq = self._ee_targets_in_torso_frame(
                R_cands[k], t_cands[k], left_cam, right_cam)
            all_lp.append(lp); all_lq.append(lq)
            all_rp.append(rp); all_rq.append(rq)
        left_pos_K   = np.stack(all_lp)   # (K, T, 3)
        left_quat_K  = np.stack(all_lq)   # (K, T, 4)
        right_pos_K  = np.stack(all_rp)
        right_quat_K = np.stack(all_rq)
        T_all = left_pos_K.shape[1]

        # ── Frame 0: B = K*M ───────────────────────────────────────────────
        # repeat each candidate's frame 0 M times → (K*M, 3/4)
        f0_lp = torch.tensor(np.repeat(left_pos_K[:,  0, :], M, axis=0), dtype=torch.float32)
        f0_lq = torch.tensor(np.repeat(left_quat_K[:, 0, :], M, axis=0), dtype=torch.float32)
        f0_rp = torch.tensor(np.repeat(right_pos_K[:, 0, :], M, axis=0), dtype=torch.float32)
        f0_rq = torch.tensor(np.repeat(right_quat_K[:,0, :], M, axis=0), dtype=torch.float32)
        # tile q_inits K times → (K*M, n_dof)
        qi_l0 = torch.tensor(np.tile(q_inits_l, (K, 1)), dtype=torch.float32)
        qi_r0 = torch.tensor(np.tile(q_inits_r, (K, 1)), dtype=torch.float32)
        q0_l, _ = self.ik_left.solve_batch(f0_lp, f0_lq, q_init=qi_l0)   # (K*M, n_dof)
        q0_r, _ = self.ik_right.solve_batch(f0_rp, f0_rq, q_init=qi_r0)

        # ── All T frames: B = K*M*T ────────────────────────────────────────
        # repeat each candidate's T targets M times → (K*M, T, 3/4)
        left_pos_KM   = np.repeat(left_pos_K,   M, axis=0)
        left_quat_KM  = np.repeat(left_quat_K,  M, axis=0)
        right_pos_KM  = np.repeat(right_pos_K,  M, axis=0)
        right_quat_KM = np.repeat(right_quat_K, M, axis=0)
        tp_l = torch.tensor(left_pos_KM.reshape(KM * T_all, 3),   dtype=torch.float32)
        tq_l = torch.tensor(left_quat_KM.reshape(KM * T_all, 4),  dtype=torch.float32)
        tp_r = torch.tensor(right_pos_KM.reshape(KM * T_all, 3),  dtype=torch.float32)
        tq_r = torch.tensor(right_quat_KM.reshape(KM * T_all, 4), dtype=torch.float32)
        # each q0 repeated T times → (K*M*T, n_dof)
        qi_l_traj = q0_l.unsqueeze(1).expand(KM, T_all, -1).reshape(KM * T_all, -1)
        qi_r_traj = q0_r.unsqueeze(1).expand(KM, T_all, -1).reshape(KM * T_all, -1)
        q_l_flat, info_l = self.ik_left_traj.solve_batch(
            tp_l, tq_l, q_init=qi_l_traj, avoid_self_collision=self.avoid_self_collision)
        q_r_flat, info_r = self.ik_right_traj.solve_batch(
            tp_r, tq_r, q_init=qi_r_traj, avoid_self_collision=self.avoid_self_collision)

        # ── Vectorized scoring over K*M ────────────────────────────────────
        sc = self._compute_scores(q_l_flat, q_r_flat, info_l, info_r, B=KM, T=T_all)

        all_results = []
        for km in range(KM):
            k = km // M
            m = km %  M
            s = slice(km * T_all, (km + 1) * T_all)
            info_l_m = {key: val[s] if isinstance(val, torch.Tensor) else val
                        for key, val in info_l.items()}
            info_r_m = {key: val[s] if isinstance(val, torch.Tensor) else val
                        for key, val in info_r.items()}
            res = {
                "score":      float(sc["scores"][km]),
                "ik_rate":    float(sc["ik_rates"][km]),
                "man":        float(sc["mans"][km]),
                "roughness":  float(sc["roughnesses"][km]),
                "lim_margin": float(sc["lim_margins"][km]),
                "col_cost":   float(sc["col_costs"][km]),
                "q_left":     sc["q_left"][km],
                "q_right":    sc["q_right"][km],
                "info_l":     info_l_m,
                "info_r":     info_r_m,
            }
            all_results.append((k, m, res))
            if verbose:
                print(f"  {k:>4}  {m:>4}  {sc['scores'][km]:>8.4f}  "
                      f"{sc['ik_rates'][km]*100:>5.1f}%  "
                      f"{sc['mans'][km]:>8.4f}  "
                      f"{sc['lim_margins'][km]:>8.4f}  "
                      f"{sc['roughnesses'][km]:>10.6f}")

        best_idx = int(np.argmax([r[2]["score"] for r in all_results]))
        best_k, best_m, best_res = all_results[best_idx]

        # Interpolate joint angles at frames where IK failed
        def _q_def(ik):
            return ik.q_default.cpu().numpy() if ik.q_default is not None else None
        best_res["q_left"]  = _interpolate_failed(
            best_res["q_left"],  best_res["info_l"]["converged"], _q_def(self.ik_left))
        best_res["q_right"] = _interpolate_failed(
            best_res["q_right"], best_res["info_r"]["converged"], _q_def(self.ik_right))

        if verbose:
            print(f"\n  Best: cand={best_k}  init={best_m}  "
                  f"score={best_res['score']:.4f}  "
                  f"roughness={best_res['roughness']:.6f}  "
                  f"t={t_cands[best_k].round(3)}")

        return OptimizationResult(
            R_torso      = R_cands[best_k],
            t_torso      = t_cands[best_k],
            q_left       = best_res["q_left"],
            q_right      = best_res["q_right"],
            scores       = {k: v for k, v in best_res.items()
                            if k not in ("q_left", "q_right", "info_l", "info_r")},
            info_l       = best_res.get("info_l", {}),
            info_r       = best_res.get("info_r", {}),
            all_scores   = [r[2] for r in all_results],
            R_candidates = R_cands,
            t_candidates = t_cands,
        )
