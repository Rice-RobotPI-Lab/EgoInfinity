"""Per-segment full SE(3) hand-rigid binding for grasped objects.

Replaces the rotation-only behavior of `fp_compose.hand_rigid_grasp_lock`
when the user opts into the explicit hand-body-frame attach mode.

Strategy (per grasp segment):
  1. Compute T_obj_rel(t) = inv(T_hand(t)) @ T_obj_obs(t) for every frame
     where both T_hand and T_obj_obs are valid.
  2. Reject R outliers (frames whose rel-R is > thresh deg from segment R median)
     and t outliers (MAD-gated).
  3. R_canonical = chordal mean of remaining rel R values, projected to SO(3).
     t_canonical = median of remaining rel t values.
  4. Optionally snap t_canonical so the closest object mesh point sits at the
     hand grasp point (closest hand vertex) within target_gap (object thickness).
  5. Propagate: T_obj_new(t) = T_hand(t) @ T_canonical for every frame in segment.
  6. Boundary SLERP ramp to blend with neighbouring (non-grasp) frames.

Result: within each segment, the object is strictly rigid w.r.t. the hand
body frame (zero relative motion by construction). Different segments may
have different T_canonical (e.g. user re-grips the object); SLERP boundary
ramp hides cross-segment jumps.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import numpy as np

from .hand_driven import (
    find_continuous_segments,
    snap_to_hand,
)


def _ang_deg_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    R_rel = R1.T @ R2
    cos_a = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _chordal_mean_so3(Rs: np.ndarray) -> np.ndarray:
    """SVD-based chordal mean of a stack (N, 3, 3) of rotations. Returns (3, 3)."""
    if Rs.ndim != 3 or Rs.shape[1:] != (3, 3) or Rs.shape[0] < 1:
        raise ValueError(f"Rs must be (N,3,3), got {Rs.shape}")
    M = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R.astype(np.float32)


def _median_seed_R(Rs: np.ndarray) -> np.ndarray:
    """Pick the R in the set that has the smallest sum of geodesic distances
    to all others. This is the geometric median, used as outlier reference."""
    n = Rs.shape[0]
    if n == 1:
        return Rs[0]
    if n == 2:
        return _chordal_mean_so3(Rs)
    dists = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i] += _ang_deg_between(Rs[i], Rs[j])
    return Rs[int(np.argmin(dists))]


def _filter_R_outliers(Rs: np.ndarray, threshold_deg: float = 30.0
                       ) -> np.ndarray:
    """Returns boolean mask (N,) of inliers (within threshold of median R)."""
    if Rs.shape[0] <= 2:
        return np.ones(Rs.shape[0], dtype=bool)
    R_ref = _median_seed_R(Rs)
    inlier = np.array([_ang_deg_between(R_ref, R) <= threshold_deg
                       for R in Rs], dtype=bool)
    # Always keep at least 1 (the reference itself qualifies trivially)
    if inlier.sum() == 0:
        inlier[:] = True
    return inlier


def _filter_t_outliers(ts: np.ndarray, mad_factor: float = 3.0,
                       fallback_threshold_m: float = 0.10
                       ) -> np.ndarray:
    """Returns boolean mask (N,) of inlier t vectors. MAD gate on per-axis,
    with a hard fallback threshold to catch pathological inputs."""
    if ts.shape[0] <= 2:
        return np.ones(ts.shape[0], dtype=bool)
    med = np.median(ts, axis=0)
    abs_dev = np.abs(ts - med)
    mad = np.median(abs_dev, axis=0) + 1e-9
    inlier = np.all(abs_dev <= mad_factor * mad, axis=1) & \
             np.all(abs_dev <= fallback_threshold_m, axis=1)
    if inlier.sum() == 0:
        inlier[:] = True
    return inlier


def _slerp_R(R_from: np.ndarray, R_to: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP between two rotations, alpha in [0,1]."""
    from scipy.spatial.transform import Rotation as Rsc, Slerp
    key_R = Rsc.from_matrix(np.stack([R_from, R_to]))
    s = Slerp([0, 1], key_R)
    return s([alpha])[0].as_matrix().astype(np.float32)


def single_canonical_grasp_lock(
    fp_pose_seq: np.ndarray,                                # (T, 4, 4) input pose
    T_hand_seq: np.ndarray,                                  # (T, 4, 4) hand body frame, NaN where invalid
    is_grasp: Sequence[bool],                                # (T,) per-object grasp mask
    *,
    mesh_pts: Optional[np.ndarray] = None,                   # for snap_to_hand
    hand_verts_per_frame: Optional[Sequence] = None,         # for snap_to_hand
    hand_is_right_per_frame: Optional[Sequence] = None,
    dominant_hand_per_frame: Optional[Sequence] = None,
    joints_per_frame: Optional[Sequence] = None,             # for palm-centroid grasp point
    min_seg_len: int = 5,
    max_gap_for_merge: int = 5,
    boundary_ramp: int = 5,
    R_outlier_thresh_deg: float = 30.0,
    t_outlier_mad_factor: float = 3.0,
    snap_target_gap_m: float = 0.005,
    snap_max_m: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """Returns (T_obj_new (T,4,4), diag dict).

    Per grasp segment:
      - Compute per-frame rel pose (in hand body frame).
      - Reject R and t outliers.
      - Aggregate canonical (R: chordal mean, t: median).
      - Optionally snap canonical t so the mesh contacts the hand at a
        representative segment frame.
      - Propagate strictly within segment.
      - SLERP/lerp ramp at segment ends to blend back to fp_pose_seq.
    """
    T = len(fp_pose_seq)
    out = fp_pose_seq.copy().astype(np.float32)
    diag = {"segments": [], "n_locked_frames": 0, "skipped": []}

    # Merge short gaps (≤ max_gap_for_merge) to avoid fragmentation
    is_g_arr = np.asarray(is_grasp, dtype=bool)
    raw_segs = find_continuous_segments(is_g_arr, min_len=min_seg_len)
    # local copy of merge logic
    merged = []
    for s, e in raw_segs:
        if merged and (s - merged[-1][1] - 1) <= max_gap_for_merge:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    segs = merged

    # ★ Fix B: split each grasp segment by dominant-hand (L vs R) runs.
    # Within a single grasp segment, dominant_hand_per_frame can switch
    # (e.g. user passes object L→R). A single T_canon mixes L-hand frames
    # and R-hand frames; because palm normals point opposite, the median
    # ends up wrong on the minority side. Split into per-hand sub-segments
    # each with its own T_canon.
    if dominant_hand_per_frame is not None:
        sub_segs = []
        for s_start, s_end in segs:
            cur_dom = None
            cur_start = s_start
            for t in range(s_start, s_end + 1):
                d_t = dominant_hand_per_frame[t] \
                      if t < len(dominant_hand_per_frame) else None
                if d_t != cur_dom:
                    if cur_dom is not None and (t - cur_start) >= min_seg_len:
                        sub_segs.append((cur_start, t - 1))
                    cur_dom = d_t
                    cur_start = t
            # tail
            if cur_dom is not None and (s_end + 1 - cur_start) >= min_seg_len:
                sub_segs.append((cur_start, s_end))
            elif cur_dom is None or (s_end + 1 - cur_start) < min_seg_len:
                # If tail is too short, merge it into previous sub_seg of same
                # parent segment (if any)
                if sub_segs and sub_segs[-1][0] >= s_start:
                    sub_segs[-1] = (sub_segs[-1][0], s_end)
                elif cur_dom is not None:
                    sub_segs.append((cur_start, s_end))
        segs = sub_segs

    for (s_start, s_end) in segs:
        seg_idx = list(range(s_start, s_end + 1))

        # Validity: both T_hand and fp_pose finite
        hand_valid = np.all(np.isfinite(T_hand_seq[seg_idx].reshape(len(seg_idx), -1)), axis=1)
        obj_valid = np.all(np.isfinite(fp_pose_seq[seg_idx].reshape(len(seg_idx), -1)), axis=1)
        good_mask = hand_valid & obj_valid
        good_t = [seg_idx[i] for i in range(len(seg_idx)) if good_mask[i]]
        if len(good_t) < 2:
            diag["skipped"].append({"start": s_start, "end": s_end,
                                     "reason": f"only {len(good_t)} valid frames"})
            continue

        # Per-frame relative pose
        rel_R = np.zeros((len(good_t), 3, 3), dtype=np.float64)
        rel_t = np.zeros((len(good_t), 3), dtype=np.float64)
        for k, t in enumerate(good_t):
            R_h = T_hand_seq[t, :3, :3].astype(np.float64)
            t_h = T_hand_seq[t, :3, 3].astype(np.float64)
            R_o = fp_pose_seq[t, :3, :3].astype(np.float64)
            t_o = fp_pose_seq[t, :3, 3].astype(np.float64)
            rel_R[k] = R_h.T @ R_o
            rel_t[k] = R_h.T @ (t_o - t_h)

        # Outlier filtering
        R_in = _filter_R_outliers(rel_R, threshold_deg=R_outlier_thresh_deg)
        t_in = _filter_t_outliers(rel_t, mad_factor=t_outlier_mad_factor)
        both_in = R_in & t_in
        if both_in.sum() < 2:
            both_in = R_in  # fall back: trust R inlier set even if t outlier

        # Canonical aggregation
        R_canon = _chordal_mean_so3(rel_R[both_in])
        t_canon_obs = np.median(rel_t[both_in], axis=0)   # observed median in hand frame

        # ★ Closed-form grasp placement: align mesh CENTROID to PALM CENTROID
        # in the palm plane (XZ in hand body frame), and align mesh BOTTOM to
        # PALM SURFACE in palm normal direction (Y).
        #
        # Previous version used `t_canon_obs.xz` (observed mask×depth centroid)
        # for XZ — but that's wherever the visible point cloud center was, NOT
        # the grasp point. For a hand-occluded object, observed centroid is
        # systematically biased AWAY from the grasp point (only the unoccluded
        # part contributes), so mesh ends up floating far from hand.
        snap_dist = 0.0
        if mesh_pts is not None:
            mesh_pts_64 = np.asarray(mesh_pts, dtype=np.float64)
            mesh_in_hand = mesh_pts_64 @ R_canon.T  # (N, 3) in hand body frame
            mesh_centroid_in_hand = mesh_in_hand.mean(axis=0)

            # ── Palm centroid in hand body frame ─────────────────────────
            # Use joints at the same anchor frame used for snap. Palm
            # landmarks = wrist (0) + 4 MCPs (5, 9, 13, 17). All should be
            # near y=0 (palm plane) since hand frame is constructed from them.
            anchor_t = good_t[len(good_t) // 2]
            palm_centroid_in_hand = None
            if joints_per_frame is not None and \
                    hand_is_right_per_frame is not None and \
                    dominant_hand_per_frame is not None:
                dom_a = dominant_hand_per_frame[anchor_t] \
                        if anchor_t < len(dominant_hand_per_frame) else None
                if dom_a in ("L", "R"):
                    use_right = (dom_a == "R")
                    j_list = joints_per_frame[anchor_t] \
                             if anchor_t < len(joints_per_frame) else []
                    r_list = hand_is_right_per_frame[anchor_t] \
                             if anchor_t < len(hand_is_right_per_frame) else []
                    joints_chosen = None
                    for j, ir in zip(j_list, r_list):
                        if bool(ir) == use_right and j is not None:
                            j_arr = np.asarray(j, dtype=np.float64)
                            if j_arr.shape == (21, 3) and np.all(np.isfinite(j_arr)):
                                joints_chosen = j_arr
                                break
                    if joints_chosen is not None:
                        T_h = T_hand_seq[anchor_t].astype(np.float64)
                        wrist_world = T_h[:3, 3]
                        R_h = T_h[:3, :3]
                        # joint_in_hand_frame = R_h.T @ (joint_world - wrist_world)
                        joints_in_hand = (joints_chosen - wrist_world) @ R_h
                        palm_centroid_in_hand = \
                            joints_in_hand[[0, 5, 9, 13, 17]].mean(axis=0)

            # Sanity check (item 5): which side of palm is the object on?
            #
            # Hand chirality determines this deterministically:
            #   - R hand: cross-product y in build_hand_body_frame points OUT
            #     of palm. A palm-grasped object has positive y in hand
            #     coords → its mesh's -y extreme (mesh.y.min) is the
            #     palm-touching end.
            #   - L hand: cross-product y points INTO the palm (chirality
            #     flip; see hand_body_frame.py docstring). A palm-grasped
            #     object has negative y in hand coords → its mesh's +y
            #     extreme (mesh.y.max) is the palm-touching end.
            #
            # We used to gate this on y_obs sign (median of observed rel_t.y),
            # but FP++ T noise frequently flipped the sign (oil-bottle clips
            # had y_obs > 1000mm or 66mm despite being L-hand palm grasps),
            # pulling mesh onto the back of the L hand. Per-hand deterministic
            # branch eliminates that failure mode without breaking the
            # working cases (y_obs sign matched the chirality choice for
            # every R-hand case we've observed).
            #
            # If dominant hand is unknown (rare), fall back to the original
            # y_obs sign auto-detect.
            y_obs = float(t_canon_obs[1])
            seg_dom = dominant_hand_per_frame[anchor_t] \
                      if (dominant_hand_per_frame is not None
                          and anchor_t < len(dominant_hand_per_frame)) \
                      else None
            if seg_dom == "R":
                y_offset = -float(mesh_in_hand[:, 1].min())
                y_branch_note = "R_chirality"
            elif seg_dom == "L":
                y_offset = -float(mesh_in_hand[:, 1].max())
                y_branch_note = "L_chirality"
            elif y_obs >= 0:
                y_offset = -float(mesh_in_hand[:, 1].min())
                y_branch_note = "fallback_y_obs>=0"
            else:
                y_offset = -float(mesh_in_hand[:, 1].max())
                y_branch_note = "fallback_y_obs<0"

            # Sanity check (item 1): degenerate R_canon → fall back to obs
            y_extent = float(mesh_in_hand[:, 1].max() - mesh_in_hand[:, 1].min())
            if y_extent < 0.005:
                t_canon = t_canon_obs.copy()
            elif palm_centroid_in_hand is not None:
                # Align mesh centroid to palm centroid in XZ; mesh bottom to
                # palm surface in Y.
                t_canon = palm_centroid_in_hand - mesh_centroid_in_hand
                t_canon[1] = y_offset
            else:
                # Fallback if palm centroid couldn't be computed: use observed
                # XZ (previous behavior).
                t_canon = np.array([
                    t_canon_obs[0],
                    y_offset,
                    t_canon_obs[2],
                ], dtype=np.float64)
            snap_dist = float(np.linalg.norm(t_canon - t_canon_obs))

        # Build T_canonical in hand body frame
        T_canon_hb = np.eye(4, dtype=np.float32)
        T_canon_hb[:3, :3] = R_canon
        T_canon_hb[:3, 3] = t_canon.astype(np.float32)

        # Propagate within segment, including frames whose obj or hand was missing
        # (use linear interp of hand frame for missing hand entries via SLERP-from-neighbour;
        # if hand still missing, keep input fp_pose for that frame)
        n_replaced = 0
        for t in seg_idx:
            T_h = T_hand_seq[t]
            if not np.all(np.isfinite(T_h)):
                continue
            T_new = T_h.astype(np.float64) @ T_canon_hb.astype(np.float64)
            out[t] = T_new.astype(np.float32)
            n_replaced += 1

        # Boundary SLERP ramp: blend segment-end with the next/prev frame's
        # original pose to avoid visual jumps when the segment boundary is
        # mid-motion.
        if boundary_ramp > 0:
            # Leading ramp (before s_start): if input frames exist, lerp pose
            for k in range(1, boundary_ramp + 1):
                t = s_start - k
                if t < 0:
                    break
                if not np.all(np.isfinite(T_hand_seq[t])):
                    continue
                alpha = k / (boundary_ramp + 1)  # smaller k = closer to segment
                R_target = (T_hand_seq[t, :3, :3] @ R_canon).astype(np.float32)
                t_target = (T_hand_seq[t, :3, :3] @ t_canon + T_hand_seq[t, :3, 3]).astype(np.float32)
                out[t, :3, :3] = _slerp_R(R_target, fp_pose_seq[t, :3, :3].astype(np.float32),
                                          alpha)
                out[t, :3, 3] = (1 - alpha) * t_target + alpha * fp_pose_seq[t, :3, 3]
            # Trailing ramp (after s_end)
            for k in range(1, boundary_ramp + 1):
                t = s_end + k
                if t >= T:
                    break
                if not np.all(np.isfinite(T_hand_seq[t])):
                    continue
                alpha = k / (boundary_ramp + 1)
                R_target = (T_hand_seq[t, :3, :3] @ R_canon).astype(np.float32)
                t_target = (T_hand_seq[t, :3, :3] @ t_canon + T_hand_seq[t, :3, 3]).astype(np.float32)
                out[t, :3, :3] = _slerp_R(R_target, fp_pose_seq[t, :3, :3].astype(np.float32),
                                          alpha)
                out[t, :3, 3] = (1 - alpha) * t_target + alpha * fp_pose_seq[t, :3, 3]

        diag["segments"].append({
            "start": s_start, "end": s_end,
            "n_frames_in_seg": len(seg_idx),
            "n_valid": len(good_t),
            "n_R_inliers": int(R_in.sum()),
            "n_t_inliers": int(t_in.sum()),
            "n_used": int(both_in.sum()),
            "n_replaced": n_replaced,
            "snap_dist_m": float(snap_dist),
            "t_canonical_m": [float(x) for x in t_canon],
            "y_branch": locals().get("y_branch_note", "n/a"),
        })
        diag["n_locked_frames"] += n_replaced

    return out, diag
