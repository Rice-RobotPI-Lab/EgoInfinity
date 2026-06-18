"""Layer B: wrist-binding hand-driven pose propagation for held objects.

Replaces the earlier all-hand-vert Kabsch.  Key improvements:

    1. Use only the rigid-palm 6-keypoint subset (wrist + 5 MCPs) instead
       of full MANO 778 verts.  Finger flexion no longer corrupts the
       rigid-motion estimate.
    2. Detect *stable* contact segments (≥ N consecutive frames) rather
       than per-frame thresholding.
    3. For each segment, anchor on the **last trusted frame BEFORE contact
       starts** if Stage 3 trust is available.  Pre-contact frames have
       independent geometric evidence; in-contact frames don't.
    4. If no trusted frame exists (severely under-aligned objects like
       knife with bad SAM3D canonical), fall back to the
       layer-A-refined anchor pose at the segment's first frame.
    5. **hand-snap**: before propagating, translate the anchor pose so the
       mesh is actually touching the hand at the anchor frame.  Without
       this, all subsequent rigid-bound frames inherit the residual depth-
       based offset (typically 4-5 cm for held objects).

This is closer in spirit to EgoGrasp's no-slip loss but as a closed-form
pre-LBFGS step rather than gradient descent.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("pose_tracker.hand_driven")


# ---------------------------------------------------------------------------
# Palm-frame keypoints (6 joints that move rigidly with the wrist)
# ---------------------------------------------------------------------------
# WiLoR / MANO 21-joint convention:
#   0  = wrist
#   1-4 = thumb (CMC, MCP, IP, TIP)  → use joint 1 (thumb CMC, near wrist)
#   5-8 = index (MCP, PIP, DIP, TIP) → use joint 5 (index MCP)
#   9-12 = middle                    → use joint 9
#   13-16 = ring                     → use joint 13
#   17-20 = pinky                    → use joint 17
PALM_FRAME_JOINT_INDICES = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)


def palm_keypoints(joints_3d: np.ndarray) -> Optional[np.ndarray]:
    """Extract (6, 3) palm-rigid keypoints from a 21x3 joint array."""
    if joints_3d is None:
        return None
    j = np.asarray(joints_3d, dtype=np.float64)
    if j.shape[0] < int(PALM_FRAME_JOINT_INDICES.max()) + 1:
        return None
    return j[PALM_FRAME_JOINT_INDICES]


# ---------------------------------------------------------------------------
# Kabsch rigid alignment
# ---------------------------------------------------------------------------
def kabsch_rigid(P: np.ndarray, Q: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Find rigid (R, t) such that ``Q ≈ R @ P + t`` (least squares).

    Both arrays (N, 3).  No scale.  Returns (R 3x3, t 3,).
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    Pc = P.mean(0)
    Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


# ---------------------------------------------------------------------------
# Stable contact segment detection
# ---------------------------------------------------------------------------
def find_continuous_segments(mask: np.ndarray, min_len: int = 5
                              ) -> List[Tuple[int, int]]:
    """Find contiguous True-runs of length >= min_len.  Returns inclusive (start, end)."""
    n = len(mask)
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i in range(n):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif not mask[i] and in_seg:
            if i - start >= min_len:
                segments.append((start, i - 1))
            in_seg = False
    if in_seg and (n - start) >= min_len:
        segments.append((start, n - 1))
    return segments


def find_last_trusted_before(t: int, trust: Sequence[bool]) -> Optional[int]:
    """Find the largest i < t with trust[i] True, else None."""
    for i in range(t - 1, -1, -1):
        if trust[i]:
            return i
    return None


# ---------------------------------------------------------------------------
# Helper: extract palm kp for the dominant contact hand at frame t
# ---------------------------------------------------------------------------
def _get_palm_kps_for_hand(
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    t: int,
    use_right: bool,
) -> Optional[np.ndarray]:
    j_list = joints_per_frame[t] or []
    r_list = hand_is_right_per_frame[t] or []
    for j, ir in zip(j_list, r_list):
        if (ir and use_right) or (not ir and not use_right):
            kps = palm_keypoints(j)
            if kps is not None:
                return kps
    return None


def _get_hand_verts_for_hand(
    hand_verts_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    t: int,
    use_right: bool,
) -> Optional[np.ndarray]:
    v_list = hand_verts_per_frame[t] or []
    r_list = hand_is_right_per_frame[t] or []
    for v, ir in zip(v_list, r_list):
        if (ir and use_right) or (not ir and not use_right):
            if v is not None and len(v) > 0:
                return np.asarray(v, dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# mask + wrist-Z anchor: place mesh centroid at (mask_uv backproject, wrist_z)
# ---------------------------------------------------------------------------
def compute_mask_wrist_anchor(
    R_canonical: np.ndarray,                # (3, 3) — fixed from SAM3D
    mesh_pts: np.ndarray,                   # (M, 3) canonical
    sam2_mask: np.ndarray,                  # (H, W) bool
    hand_mask: Optional[np.ndarray],        # (H, W) bool (subtract)
    wrist_world: np.ndarray,                # (3,) world coords of MANO joint 0
    K: np.ndarray,                          # (3, 3)
    *,
    z_offset_m: float = 0.0,                # optional bias if wrist Z too shallow
) -> Tuple[Optional[np.ndarray], dict]:
    """Build a 4x4 pose where:
        rotation = R_canonical (SAM3D as-is, not refined)
        translation = (X_target, Y_target, Z_target) - R @ mesh_centroid
        with X, Y from (mask ∖ hand_mask) 2D centroid backprojected at Z = wrist Z.

    Returns (T_anchor, diag) or (None, diag) if mask is empty.
    """
    eff_mask = sam2_mask & ~hand_mask if hand_mask is not None else sam2_mask
    if not eff_mask.any():
        eff_mask = sam2_mask
        if not eff_mask.any():
            return None, {"reason": "empty mask"}
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.where(eff_mask)
    u_mask = float(xs.mean())
    v_mask = float(ys.mean())

    Z_target = float(wrist_world[2]) + z_offset_m
    if Z_target <= 1e-3 or not np.isfinite(Z_target):
        return None, {"reason": "bad wrist Z", "Z": Z_target}

    X_target = (u_mask - cx) * Z_target / fx
    Y_target = (v_mask - cy) * Z_target / fy
    target_world = np.array([X_target, Y_target, Z_target], dtype=np.float64)

    mesh_centroid_canon = mesh_pts.mean(axis=0)
    t_new = target_world - R_canonical @ mesh_centroid_canon

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_canonical
    T[:3, 3] = t_new
    return T, {
        "u_mask": u_mask, "v_mask": v_mask,
        "Z_wrist": Z_target,
        "target_world": target_world.tolist(),
    }


# ---------------------------------------------------------------------------
# hand-snap: translate anchor pose so mesh touches hand
# ---------------------------------------------------------------------------
def snap_to_hand(
    T_obj_anchor: np.ndarray,
    mesh_pts: np.ndarray,
    hand_verts: np.ndarray,
    *,
    target_gap_m: float = 0.005,
    max_snap_m: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """Translate ``T_obj_anchor`` so the closest mesh point is ~``target_gap_m``
    from the closest hand vertex.

    The snap is *translation only* — no rotation change.  Cap the magnitude
    at ``max_snap_m`` to prevent runaway snap when the wrong hand was chosen
    or the mesh orientation is grossly wrong.

    Returns
    -------
    T_snapped : (4, 4)
    snap_distance : float  (m, applied)
    """
    R, t = T_obj_anchor[:3, :3], T_obj_anchor[:3, 3]
    mesh_world = mesh_pts @ R.T + t
    if len(hand_verts) < 3 or len(mesh_world) < 3:
        return T_obj_anchor, 0.0
    tree = cKDTree(hand_verts)
    d_mesh_to_hand, idx_hand = tree.query(mesh_world, k=1)
    closest_mesh_i = int(d_mesh_to_hand.argmin())
    closest_hand_i = int(idx_hand[closest_mesh_i])
    mesh_pt = mesh_world[closest_mesh_i]
    hand_pt = hand_verts[closest_hand_i]
    direction = hand_pt - mesh_pt
    current_dist = float(np.linalg.norm(direction))
    if current_dist <= target_gap_m:
        return T_obj_anchor, 0.0
    snap_amount = min(current_dist - target_gap_m, max_snap_m)
    delta_t = (direction / current_dist) * snap_amount
    out = T_obj_anchor.copy()
    out[:3, 3] = t + delta_t
    return out, snap_amount


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def wrist_binding_propagation(
    T_seq_in: Sequence[np.ndarray],
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    contact_soft: np.ndarray,                       # (T, 2)
    trust: Sequence[bool],                          # (T,) from Stage 3
    *,
    mesh_pts: Optional[np.ndarray] = None,                                 # for hand-snap
    hand_verts_per_frame: Optional[Sequence[Sequence[Optional[np.ndarray]]]] = None,
    enable_hand_snap: bool = True,
    snap_target_gap_m: float = 0.005,
    snap_max_m: float = 0.10,
    min_seg_len: int = 5,
    soft_threshold: float = 0.5,
    boundary_ramp: int = 5,
) -> Tuple[np.ndarray, dict]:
    """Replace per-frame object pose with wrist-binding propagation.

    For every stable contact segment:
        1. Find anchor = last trusted frame BEFORE segment start.  If none,
           use first frame of segment.
        2. Get palm keypoints (wrist + 5 MCPs) of the dominant contact hand
           at the anchor and at every in-segment frame.
        3. Kabsch on the 6-keypoint set yields (R_h, t_h) — rigid hand motion.
        4. T_obj_t = (R_h, t_h) @ T_obj_anchor.
        5. Apply a 5-frame SLERP ramp at segment boundaries to avoid jumps.

    Returns (T_seq_out, diag).  diag has 'segments', 'used_right', 'n_replaced'.
    """
    T_in = np.asarray(T_seq_in, dtype=np.float64)
    n = len(T_in)
    out = T_in.copy()

    soft_max = contact_soft.max(axis=1) > soft_threshold
    segments = find_continuous_segments(soft_max, min_len=min_seg_len)
    if not segments:
        return out, {"segments": [], "n_replaced": 0, "used_right": None}

    # Decide handedness via integral over contact frames
    contact_idx_all = np.where(soft_max)[0]
    hot_left = float(contact_soft[contact_idx_all, 0].sum())
    hot_right = float(contact_soft[contact_idx_all, 1].sum())
    use_right = hot_right >= hot_left

    n_replaced = 0
    seg_diag = []

    for (t_start, t_end) in segments:
        # Anchor: last trusted frame before segment start.
        t_anchor = find_last_trusted_before(t_start, trust)
        anchor_source = "trusted_before"
        if t_anchor is None:
            # Fall back to segment start (whose pose was set by Layer A)
            t_anchor = t_start
            anchor_source = "segment_start_fallback"

        kps_anchor = _get_palm_kps_for_hand(
            joints_per_frame, hand_is_right_per_frame, t_anchor, use_right)
        if kps_anchor is None:
            # try the other hand if the anchor frame has no keypoints for chosen hand
            kps_anchor = _get_palm_kps_for_hand(
                joints_per_frame, hand_is_right_per_frame, t_anchor, not use_right)
            if kps_anchor is None:
                continue

        T_obj_anchor = T_in[t_anchor].copy()
        snap_dist = 0.0
        # ── hand-snap: bring mesh to actual contact at the anchor frame ──
        if (enable_hand_snap and mesh_pts is not None
                and hand_verts_per_frame is not None):
            # Use anchor-frame palm-side hand verts; if anchor is "trusted_before"
            # contact, the hand might not be in grasp pose yet — pick the
            # contact hand at segment START instead for snap reference.
            t_snap = t_start if anchor_source == "trusted_before" else t_anchor
            hand_v_snap = _get_hand_verts_for_hand(
                hand_verts_per_frame, hand_is_right_per_frame, t_snap, use_right)
            if hand_v_snap is None:
                hand_v_snap = _get_hand_verts_for_hand(
                    hand_verts_per_frame, hand_is_right_per_frame, t_snap, not use_right)
            if hand_v_snap is not None:
                # Snap the anchor pose, evaluated at t_snap.  But we need
                # the mesh→world at t_snap, not t_anchor.  Use Kabsch first
                # to project T_obj_anchor to t_snap, snap there, then
                # back-propagate.
                if t_snap != t_anchor:
                    kps_snap = _get_palm_kps_for_hand(
                        joints_per_frame, hand_is_right_per_frame,
                        t_snap, use_right)
                    if kps_snap is not None:
                        R_h, t_h = kabsch_rigid(kps_anchor, kps_snap)
                        T_delta = np.eye(4, dtype=np.float64)
                        T_delta[:3, :3] = R_h
                        T_delta[:3, 3] = t_h
                        T_at_snap = T_delta @ T_obj_anchor
                        T_at_snap_snapped, snap_dist = snap_to_hand(
                            T_at_snap, mesh_pts, hand_v_snap,
                            target_gap_m=snap_target_gap_m,
                            max_snap_m=snap_max_m,
                        )
                        # Back-propagate snap to anchor frame: invert T_delta
                        T_obj_anchor = np.linalg.inv(T_delta) @ T_at_snap_snapped
                else:
                    T_obj_anchor, snap_dist = snap_to_hand(
                        T_obj_anchor, mesh_pts, hand_v_snap,
                        target_gap_m=snap_target_gap_m,
                        max_snap_m=snap_max_m,
                    )

        seg_replaced = 0
        for t in range(t_start, t_end + 1):
            kps_t = _get_palm_kps_for_hand(
                joints_per_frame, hand_is_right_per_frame, t, use_right)
            if kps_t is None:
                kps_t = _get_palm_kps_for_hand(
                    joints_per_frame, hand_is_right_per_frame, t, not use_right)
                if kps_t is None:
                    continue
            R_h, t_h = kabsch_rigid(kps_anchor, kps_t)
            T_delta = np.eye(4, dtype=np.float64)
            T_delta[:3, :3] = R_h
            T_delta[:3, 3] = t_h
            out[t] = T_delta @ T_obj_anchor
            seg_replaced += 1
        n_replaced += seg_replaced

        # ── boundary ramp: blend out[t_end+1 ..] toward original T_in over `boundary_ramp` frames
        # so non-contact frames smoothly resume the upstream pose.
        if boundary_ramp > 0 and t_end + 1 < n:
            from scipy.spatial.transform import Rotation as Rot, Slerp
            for k in range(boundary_ramp):
                tk = t_end + 1 + k
                if tk >= n:
                    break
                u = (k + 1) / (boundary_ramp + 1)        # 1/6, 2/6, ...
                # SLERP rotation
                R_a = out[tk][:3, :3]
                R_b = T_in[tk][:3, :3]
                slerp = Slerp([0.0, 1.0],
                              Rot.from_matrix(np.stack([R_a, R_b])))
                R_blend = slerp([u]).as_matrix()[0]
                t_blend = (1 - u) * out[tk][:3, 3] + u * T_in[tk][:3, 3]
                out[tk][:3, :3] = R_blend
                out[tk][:3, 3] = t_blend

        seg_diag.append({
            "start": int(t_start), "end": int(t_end),
            "anchor": int(t_anchor), "anchor_source": anchor_source,
            "n_replaced": seg_replaced,
            "snap_distance_m": float(snap_dist),
        })

    snap_summary = ", ".join(
        f"{s['n_replaced']}fr@{s['snap_distance_m']*100:.1f}cm"
        for s in seg_diag)
    log.info(
        f"wrist_binding: hand={'R' if use_right else 'L'}, "
        f"{len(segments)} segs, replaced {n_replaced} frames "
        f"[{snap_summary}]")
    return out, {"segments": seg_diag, "n_replaced": n_replaced,
                  "used_right": use_right}


# Back-compat alias
hand_driven_pose_propagation = wrist_binding_propagation


# ---------------------------------------------------------------------------
# Mask + Wrist-Z Anchor + Wrist-Kabsch Propagation
# ---------------------------------------------------------------------------
def _merge_short_gap_segments(
    segments: Sequence[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    """Merge contact segments separated by gaps <= max_gap frames.

    A "gap" is the number of NON-contact frames between segment_i.end and
    segment_{i+1}.start.  Short gaps usually mean SavGol-borderline
    fluctuation, not real release; merge so the rotation reference stays
    continuous through them.
    """
    if not segments:
        return []
    merged: List[List[int]] = [list(segments[0])]
    for s, e in segments[1:]:
        prev_end = merged[-1][1]
        gap = s - prev_end - 1
        if gap <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def mask_wrist_anchor_propagation(
    T_seq_in: Sequence[np.ndarray],
    mesh_pts: np.ndarray,
    R_canonical: np.ndarray,
    obj_masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    contact_soft: np.ndarray,                       # (T, 2)
    K: np.ndarray,
    *,
    min_seg_len: int = 3,
    soft_threshold: float = 0.3,
    boundary_ramp: int = 5,
    max_short_gap: int = 10,                        # ≤ this = same grasp episode
) -> Tuple[np.ndarray, dict]:
    """Per-grasp-episode mask + wrist-Z anchor + wrist Kabsch propagation.

    Algorithm:
        1. Find raw contact segments (≥ min_seg_len frames, soft > threshold).
        2. **Merge segments separated by ≤ max_short_gap frames** — these
           are usually SavGol-borderline fluctuations, not real release.
           Each merged "grasp episode" represents one continuous hold.
        3. For each grasp episode:
             a. Pick global anchor = frame with max contact_soft inside the
                episode (best grip moment).
             b. Build T_anchor: R = R_canonical (SAM3D), t from mask UV
                backprojected at wrist Z.
             c. Apply Kabsch propagation to **every frame in the episode**,
                including the small dropout gaps (palm joints are still
                detected by WiLoR even when contact_soft fluctuates).
        4. Boundary ramp blends back to upstream pose at episode edges.

    Long dropouts (> max_short_gap) → independent grasp episode → fresh anchor.
    """
    T_in = np.asarray(T_seq_in, dtype=np.float64)
    n = len(T_in)
    out = T_in.copy()

    is_contact = contact_soft.max(axis=1) > soft_threshold
    raw_segments = find_continuous_segments(is_contact, min_len=min_seg_len)
    segments = _merge_short_gap_segments(raw_segments, max_gap=max_short_gap)
    if raw_segments and len(segments) != len(raw_segments):
        log.info(
            f"merged {len(raw_segments)} raw segments → {len(segments)} grasp episodes "
            f"(max_short_gap={max_short_gap})")
    if not segments:
        return out, {"segments": [], "n_replaced": 0}

    seg_diag: List[dict] = []
    n_replaced = 0

    for (t_start, t_end) in segments:
        # ---- Determine handedness FOR THIS SEGMENT (knife may switch hands across grasps) ----
        seg_left = float(contact_soft[t_start:t_end+1, 0].sum())
        seg_right = float(contact_soft[t_start:t_end+1, 1].sum())
        use_right = seg_right >= seg_left

        # ---- Anchor frame = strongest grip in segment ----
        seg_max = contact_soft[t_start:t_end+1, 1 if use_right else 0]
        t_anchor = int(t_start + np.argmax(seg_max))

        # ---- Anchor pose: mask + wrist-Z ----
        if obj_masks[t_anchor] is None or not obj_masks[t_anchor].any():
            continue
        joints_list = joints_per_frame[t_anchor] or []
        is_right_list = hand_is_right_per_frame[t_anchor] or []
        wrist = None
        for j, ir in zip(joints_list, is_right_list):
            if (ir and use_right) or (not ir and not use_right):
                if j is not None and len(j) > 0:
                    wrist = np.asarray(j[0], dtype=np.float64)
                    break
        if wrist is None:
            continue
        T_anchor_obj, anchor_diag = compute_mask_wrist_anchor(
            R_canonical=R_canonical,
            mesh_pts=mesh_pts,
            sam2_mask=obj_masks[t_anchor],
            hand_mask=hand_masks[t_anchor] if hand_masks else None,
            wrist_world=wrist,
            K=K,
        )
        if T_anchor_obj is None:
            continue

        # ---- Anchor palm keypoints for Kabsch propagation ----
        kps_anchor = _get_palm_kps_for_hand(
            joints_per_frame, hand_is_right_per_frame, t_anchor, use_right)
        if kps_anchor is None:
            continue

        # ---- Propagate every frame in segment ----
        seg_replaced = 0
        for t in range(t_start, t_end + 1):
            kps_t = _get_palm_kps_for_hand(
                joints_per_frame, hand_is_right_per_frame, t, use_right)
            if kps_t is None:
                continue
            R_h, t_h = kabsch_rigid(kps_anchor, kps_t)
            T_delta = np.eye(4, dtype=np.float64)
            T_delta[:3, :3] = R_h
            T_delta[:3, 3] = t_h
            out[t] = T_delta @ T_anchor_obj
            seg_replaced += 1
        n_replaced += seg_replaced

        # ---- Boundary ramp ----
        if boundary_ramp > 0 and t_end + 1 < n:
            from scipy.spatial.transform import Rotation as Rot, Slerp
            for k in range(boundary_ramp):
                tk = t_end + 1 + k
                if tk >= n:
                    break
                u = (k + 1) / (boundary_ramp + 1)
                R_a = out[tk][:3, :3]
                R_b = T_in[tk][:3, :3]
                slerp = Slerp([0.0, 1.0],
                              Rot.from_matrix(np.stack([R_a, R_b])))
                R_blend = slerp([u]).as_matrix()[0]
                t_blend = (1 - u) * out[tk][:3, 3] + u * T_in[tk][:3, 3]
                out[tk][:3, :3] = R_blend
                out[tk][:3, 3] = t_blend

        seg_diag.append({
            "start": int(t_start), "end": int(t_end),
            "anchor": int(t_anchor),
            "use_right": bool(use_right),
            "n_replaced": seg_replaced,
            "anchor_target_world": anchor_diag.get("target_world"),
            "u_mask": anchor_diag.get("u_mask"),
            "v_mask": anchor_diag.get("v_mask"),
            "Z_wrist": anchor_diag.get("Z_wrist"),
        })

    log.info(
        f"mask+wrist anchor propagation: {len(segments)} segs, "
        f"replaced {n_replaced} frames")
    return out, {"segments": seg_diag, "n_replaced": n_replaced}


# ---------------------------------------------------------------------------
# 2026-05-01 — Rigid wrist binding (replaces Kabsch propagation drift)
# ---------------------------------------------------------------------------
def _split_episode_by_handedness(
    t_start: int, t_end: int,
    contact_soft: np.ndarray,
    soft_threshold: float,
    min_sub_len: int = 5,
) -> List[Tuple[int, int, bool]]:
    """Split a WRIST episode into sub-segments by per-frame dominant hand.

    Without this, the original code picks one ``use_right`` for the whole
    episode (by total contact-frame count).  When an episode contains a
    real L↔R hand transition (e.g. demonstrator switches hands mid-clip),
    the loser-hand frames stay bound to the winner's palm via Kabsch, and
    the mesh tracks the wrong hand by tens of cm in XY.

    Per-frame label assignment:
        L-only  (wl=1, wr=0) → 'L'
        R-only  (wl=0, wr=1) → 'R'
        BOTH    (wl=1, wr=1) → inherit previous frame's label
                              (avoid spurious sub-segment splits while
                               the demonstrator briefly grips with both)

    Sub-segments shorter than ``min_sub_len`` are merged into the longer
    neighbour — short flicker is noise, not a real transition.

    Returns list of (sub_start, sub_end, use_right) tuples covering
    [t_start, t_end] without overlap.
    """
    n = t_end - t_start + 1
    if n == 0:
        return []
    hand = np.zeros(n, dtype=np.int8)   # 0=L, 1=R
    last: Optional[int] = None
    for i in range(n):
        t = t_start + i
        wl_t = float(contact_soft[t, 0]) > soft_threshold
        wr_t = float(contact_soft[t, 1]) > soft_threshold
        if wl_t and not wr_t:
            hand[i] = 0; last = 0
        elif wr_t and not wl_t:
            hand[i] = 1; last = 1
        else:                           # BOTH or NEITHER — inherit
            hand[i] = last if last is not None else 0

    # Build raw sub-segments
    subs: List[List[int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and hand[j] == hand[i]:
            j += 1
        subs.append([i, j - 1, int(hand[i])])
        i = j

    # Merge sub-segments shorter than min_sub_len into the longer neighbour
    if min_sub_len > 1:
        changed = True
        while changed and len(subs) > 1:
            changed = False
            for k in range(len(subs)):
                si, ei, _ = subs[k]
                if ei - si + 1 >= min_sub_len:
                    continue
                # Pick neighbour to merge into
                left = k - 1 if k > 0 else None
                right = k + 1 if k < len(subs) - 1 else None
                if left is not None and right is not None:
                    ll = subs[left][1] - subs[left][0] + 1
                    rl = subs[right][1] - subs[right][0] + 1
                    target = left if ll >= rl else right
                elif left is not None:
                    target = left
                elif right is not None:
                    target = right
                else:
                    break
                # Extend target span; drop the short sub
                if target == left:
                    subs[left][1] = ei
                else:
                    subs[right][0] = si
                subs.pop(k)
                changed = True
                break

    # Convert to absolute frame indices + bool use_right
    return [(t_start + si, t_start + ei, bool(hi)) for si, ei, hi in subs]


def rigid_wrist_binding_propagation(
    T_seq_in: Sequence[np.ndarray],
    mesh_pts: np.ndarray,
    R_obj_anchor: np.ndarray,                       # (3, 3) baseline R from Phase D anchor
    obj_masks: Sequence[Optional[np.ndarray]],
    hand_masks: Sequence[Optional[np.ndarray]],
    obs_clouds: Sequence[Optional[np.ndarray]],
    pc_stable_per_frame: np.ndarray,                # (T,) bool — drives t source
    joints_per_frame: Sequence[Sequence[Optional[np.ndarray]]],
    hand_is_right_per_frame: Sequence[Sequence[bool]],
    contact_soft: np.ndarray,                       # (T, 2)
    K: np.ndarray,
    *,
    min_seg_len: int = 3,
    soft_threshold: float = 0.3,
    max_short_gap: int = 10,
    handedness_split_min_sub_len: int = 5,          # split sub-segs by dominant hand
    sub_seg_ramp: int = 3,                          # frames to blend at sub-seg boundary
) -> Tuple[np.ndarray, dict]:
    """Per-grasp-segment **rigid binding**: R_obj follows R_palm with a
    constant relative rotation; t_obj uses observation bbox center when the
    point cloud is stable, otherwise stays rigidly attached to the palm.

    Differs from ``mask_wrist_anchor_propagation`` in two ways:
        1. R: uses ``R_obj[t] = R_palm[t] @ R_rel`` with R_rel computed once
           per segment (eliminates Kabsch propagation drift).
        2. t: dual-source — obs bbox center if pc stable; else palm + a
           constant in-palm offset captured at the segment's strongest grip.

    Frames outside grasp segments are left untouched (filled with T_seq_in).
    """
    T_in = np.asarray(T_seq_in, dtype=np.float64)
    n = len(T_in)
    out = T_in.copy()
    pc_stable_per_frame = np.asarray(pc_stable_per_frame, dtype=bool)

    is_contact = contact_soft.max(axis=1) > soft_threshold
    raw_segments = find_continuous_segments(is_contact, min_len=min_seg_len)
    segments = _merge_short_gap_segments(raw_segments, max_gap=max_short_gap)
    if not segments:
        return out, {"segments": [], "n_replaced": 0}

    mesh_centroid_canon = (mesh_pts.min(axis=0) + mesh_pts.max(axis=0)) * 0.5

    seg_diag: List[dict] = []
    n_replaced = 0
    n_t_from_pc = 0
    n_t_from_palm = 0

    for (t_start, t_end) in segments:
        # ---- Split episode by dominant hand (handles L↔R transitions) ----
        sub_segs = _split_episode_by_handedness(
            t_start, t_end, contact_soft, soft_threshold,
            min_sub_len=handedness_split_min_sub_len)

        episode_replaced = 0
        sub_diags: List[dict] = []

        for (s_sub, e_sub, use_right) in sub_segs:
            # ---- Verify the chosen hand actually grasps in this sub-seg ----
            seg_max = contact_soft[s_sub:e_sub + 1, 1 if use_right else 0]
            if seg_max.size == 0 or float(seg_max.max()) < soft_threshold:
                # Sub-segment is entirely BOTH-inherited with no real contact
                # on the chosen hand — fall back to the other hand.
                other_col = 0 if use_right else 1
                alt_max = contact_soft[s_sub:e_sub + 1, other_col]
                if alt_max.size == 0 or float(alt_max.max()) < soft_threshold:
                    continue
                use_right = not use_right

            # ---- Anchor frame: minimise |obs_bbox − chosen_wrist| ----
            # Picking the first contact frame (e.g. transition into the new
            # hand) gives a stale obs_cloud that's still at the OLD hand's
            # position — t_offset_in_palm then becomes huge and the Kabsch
            # propagation puts the mesh in the wrong place for the rest of
            # the sub-seg.  Anchoring on the frame where obs is closest to
            # the chosen wrist gives a steady-state grip moment.
            best_d = float("inf")
            t_a = -1
            for t in range(s_sub, e_sub + 1):
                if obs_clouds[t] is None or len(obs_clouds[t]) < 30:
                    continue
                obs_t = obs_clouds[t]
                obs_center = (obs_t.min(axis=0) + obs_t.max(axis=0)) * 0.5
                j_list = joints_per_frame[t] or []
                ir_list = hand_is_right_per_frame[t] or []
                wrist_t = None
                for j, ir in zip(j_list, ir_list):
                    if (ir and use_right) or (not ir and not use_right):
                        if j is not None and len(j) > 0:
                            wrist_t = np.asarray(j[0], dtype=np.float64)
                            break
                if wrist_t is None:
                    continue
                d = float(np.linalg.norm(obs_center - wrist_t))
                if d < best_d:
                    best_d = d
                    t_a = t
            if t_a < 0:
                # Fallback to first frame with contact + chosen hand detected
                seg_max = contact_soft[s_sub:e_sub + 1, 1 if use_right else 0]
                t_a = int(s_sub + np.argmax(seg_max))

            # ---- Anchor palm keypoints (reference frame for R_palm[t]) ----
            kps_a = _get_palm_kps_for_hand(
                joints_per_frame, hand_is_right_per_frame, t_a, use_right)
            if kps_a is None:
                continue

            # ---- Anchor t_obj: prefer obs bbox at t_a; fall back to mask+wrist Z ----
            t_obj_anchor = None
            if obs_clouds[t_a] is not None and len(obs_clouds[t_a]) >= 30:
                obs_a = obs_clouds[t_a]
                bbox_center = (obs_a.min(axis=0) + obs_a.max(axis=0)) * 0.5
                t_obj_anchor = bbox_center - R_obj_anchor @ mesh_centroid_canon
            if t_obj_anchor is None:
                joints_list = joints_per_frame[t_a] or []
                is_right_list = hand_is_right_per_frame[t_a] or []
                wrist = None
                for j, ir in zip(joints_list, is_right_list):
                    if (ir and use_right) or (not ir and not use_right):
                        if j is not None and len(j) > 0:
                            wrist = np.asarray(j[0], dtype=np.float64)
                            break
                if wrist is None or obj_masks[t_a] is None or not obj_masks[t_a].any():
                    continue
                T_anchor_obj_legacy, _ = compute_mask_wrist_anchor(
                    R_canonical=R_obj_anchor,
                    mesh_pts=mesh_pts,
                    sam2_mask=obj_masks[t_a],
                    hand_mask=hand_masks[t_a] if hand_masks else None,
                    wrist_world=wrist,
                    K=K,
                )
                if T_anchor_obj_legacy is None:
                    continue
                t_obj_anchor = T_anchor_obj_legacy[:3, 3]

            kps_a_origin = kps_a[0].copy()
            t_offset_in_palm = t_obj_anchor - kps_a_origin

            sub_replaced = 0
            for t in range(s_sub, e_sub + 1):
                kps_t = _get_palm_kps_for_hand(
                    joints_per_frame, hand_is_right_per_frame, t, use_right)
                if kps_t is None:
                    continue
                R_palm_t, _t_palm_t = kabsch_rigid(kps_a, kps_t)
                palm_origin_t = kps_t[0]
                R_obj_t = R_palm_t @ R_obj_anchor

                # ★ grasped_both: prefer the FARTHER hand's z so the mesh
                # doesn't occlude the more-distant hand.  Naive pick (one
                # sub-segment ⇒ one hand) drives mesh z to the nearer hand,
                # leaving the farther hand peeking out behind the mesh.
                # Only apply when |L-R| z difference is < 25 cm (otherwise
                # one wrist is likely an outlier not on the object).
                palm_z_for_t = palm_origin_t[2]
                both_grasp = (
                    t < contact_soft.shape[0]
                    and float(contact_soft[t, 0]) > soft_threshold
                    and float(contact_soft[t, 1]) > soft_threshold
                )
                if both_grasp:
                    kps_other = _get_palm_kps_for_hand(
                        joints_per_frame, hand_is_right_per_frame, t, not use_right)
                    if kps_other is not None:
                        other_z = float(kps_other[0, 2])
                        if abs(palm_z_for_t - other_z) < 0.25:
                            palm_z_for_t = max(palm_z_for_t, other_z)

                # Hybrid translation source:
                #   XY ← obs cloud bbox (SAM2 mask × depth → pixel-accurate
                #        2D projection, immune to MoGe depth noise)
                #   Z  ← raw palm wrist z, with R-rotated mesh-centroid
                #        offset subtracted so the mesh centroid lands AT
                #        palm.z.  We deliberately do NOT use t_offset_in_palm.z
                #        — that offset was captured against obs_cloud.z at
                #        the anchor frame, which carries MoGe's per-pixel
                #        depth error (up to 70 cm in pour-drink clips).  Pure
                #        palm.z is the only signal that's MoGe-free.
                # Fallback: when obs cloud is missing / too sparse for XY,
                # fall back to t_offset_in_palm for XY only (Z stays palm).
                mesh_offset = R_obj_t @ mesh_centroid_canon
                if obs_clouds[t] is not None and len(obs_clouds[t]) >= 30:
                    obs_t = obs_clouds[t]
                    bbox_center_t = (obs_t.min(axis=0) + obs_t.max(axis=0)) * 0.5
                    t_obj_t = np.array([
                        bbox_center_t[0] - mesh_offset[0],   # X ← obs
                        bbox_center_t[1] - mesh_offset[1],   # Y ← obs
                        palm_z_for_t - mesh_offset[2],       # Z ← palm.z (or farther wrist.z in grasped_both)
                    ], dtype=np.float64)
                    n_t_from_pc += 1
                else:
                    # Fallback when obs cloud unavailable: use palm-anchored
                    # XY (better than nothing) but keep MoGe-free Z.
                    palm_anchored_t = palm_origin_t + R_palm_t @ t_offset_in_palm
                    t_obj_t = np.array([
                        palm_anchored_t[0],
                        palm_anchored_t[1],
                        palm_z_for_t - mesh_offset[2],
                    ], dtype=np.float64)
                    n_t_from_palm += 1

                T_t = np.eye(4, dtype=np.float64)
                T_t[:3, :3] = R_obj_t
                T_t[:3, 3] = t_obj_t
                out[t] = T_t
                sub_replaced += 1

            episode_replaced += sub_replaced
            sub_diags.append({
                "sub_start": int(s_sub),
                "sub_end": int(e_sub),
                "anchor": int(t_a),
                "use_right": bool(use_right),
                "n_replaced": int(sub_replaced),
            })

        # ---- Blend across sub-segment boundaries inside the episode ----
        # When sub_idx-1 binds to L and sub_idx binds to R, out[s_sub] will
        # jump in XY by tens of cm (L wrist ↔ R wrist).  Slerp R and lerp t
        # over the first ``sub_seg_ramp`` frames of each non-first sub-seg.
        if sub_seg_ramp > 0 and len(sub_diags) > 1:
            from scipy.spatial.transform import Rotation as Rot, Slerp
            for k in range(1, len(sub_diags)):
                prev_end = sub_diags[k - 1]["sub_end"]
                curr_start = sub_diags[k]["sub_start"]
                curr_end = sub_diags[k]["sub_end"]
                ramp = min(sub_seg_ramp, curr_end - curr_start + 1)
                R_prev = out[prev_end][:3, :3].copy()
                t_prev = out[prev_end][:3, 3].copy()
                for r in range(ramp):
                    tk = curr_start + r
                    u = (r + 1) / (ramp + 1)
                    R_new = out[tk][:3, :3].copy()
                    t_new = out[tk][:3, 3].copy()
                    slerp = Slerp([0.0, 1.0],
                                  Rot.from_matrix(np.stack([R_prev, R_new])))
                    R_blend = slerp([u]).as_matrix()[0]
                    t_blend = (1 - u) * t_prev + u * t_new
                    out[tk][:3, :3] = R_blend
                    out[tk][:3, 3] = t_blend

        n_replaced += episode_replaced
        seg_diag.append({
            "start": int(t_start),
            "end": int(t_end),
            "n_replaced": int(episode_replaced),
            "n_sub_segs": int(len(sub_diags)),
            "sub_segs": sub_diags,
        })

    log.info(
        f"rigid_wrist_binding: {len(segments)} segs, "
        f"{n_replaced} frames replaced "
        f"(t from pc={n_t_from_pc}, t from palm={n_t_from_palm})")
    return out, {
        "segments": seg_diag,
        "n_replaced": int(n_replaced),
        "n_t_from_pc": int(n_t_from_pc),
        "n_t_from_palm": int(n_t_from_palm),
    }
