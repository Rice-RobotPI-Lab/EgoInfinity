"""Composed-pose helpers extracted from FoundationPose-plus-plus/scripts/
viser_multiclip.py. Library only — no viser/GUI.

Public API:
  - ClipData (dataclass holding everything compose_display_pose needs)
  - load_clip(testcase: Path, pkl: Path, oid_override=None) -> ClipData
  - compose_display_pose(clip, do_unflip, do_hand_rigid, do_pca,
                          do_obs_anchor, do_ei_t, do_state_lock, do_smooth)
      -> np.ndarray (T, 4, 4)

Used by egoinfinity.pipeline.post_tracking.bake_fp_pose to write per-clip composed FP++ poses into
EI pkl's pose_track_info[oid]['T_seq'] for HF export.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# ────────────────── decode helpers ─────────────────────────────────────────
def load_pkl(p: Path) -> dict:
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def decode_jpeg(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_depth_m(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def unpack_mask(packed: bytes, shape) -> np.ndarray:
    H, W = int(shape[0]), int(shape[1])
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[: H * W]
    return bits.reshape(H, W).astype(bool)


SH_C0 = 0.28209479177387814

def load_gs_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("property float"):
                props.append(line.split()[2])
            elif line == "end_header":
                break
        dt = np.dtype([(p, "<f4") for p in props])
        data = np.fromfile(f, dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    if "f_dc_0" in data.dtype.names:
        dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], -1)
        rgb = np.clip(dc * SH_C0 + 0.5, 0, 1)
    else:
        rgb = np.full((n, 3), 0.7)
    return xyz, (rgb * 255).astype(np.uint8)


def tint(rgb: np.ndarray, color, mix: float = 0.5) -> np.ndarray:
    t = np.asarray(color, dtype=np.float32)
    out = rgb.astype(np.float32) * (1 - mix) + t * mix
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_to_pointcloud(mask, depth_m, focal, cx, cy, step=2) -> np.ndarray:
    ys, xs = np.where(mask)
    if step > 1:
        ys, xs = ys[::step], xs[::step]
    z = depth_m[ys, xs]
    keep = (z > 0.05) & (z < 10.0)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    x = (xs - cx) * z / focal
    y = (ys - cy) * z / focal
    return np.stack([x, y, z], -1).astype(np.float32)


# ────────────────── SE(3) smoothing ────────────────────────────────────────
_FLIP_180 = np.stack([
    Rotation.from_rotvec(np.array([np.pi, 0.0, 0.0])).as_matrix(),
    Rotation.from_rotvec(np.array([0.0, np.pi, 0.0])).as_matrix(),
    Rotation.from_rotvec(np.array([0.0, 0.0, np.pi])).as_matrix(),
]).astype(np.float32)


def _ang(R1, R2):
    Rd = R1 @ R2.T
    cos_a = np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def align_to_ei_canonical(fp_pose: np.ndarray, ei_pose: Optional[np.ndarray],
                           ref_frame: int = 0) -> np.ndarray:
    """If FP++'s register chose a 180°-equivalent symmetry pose, apply the
    inverse flip in mesh-local frame to ALL frames so the cyan and orange
    meshes share a canonical orientation.

    `ref_frame` is where we compare FP↔EI to detect the symmetry flip.
    Defaults to 0 but should be set to SAM3D's `init_frame` — that's the
    most trustworthy EI pose (= the frame whose RGB+D actually grew the
    SAM3D mesh, so the canonical orientation is best defined there).
    """
    if ei_pose is None or len(ei_pose) < 1 or len(fp_pose) < 1:
        return fp_pose.copy()
    k = int(np.clip(ref_frame, 0, min(len(fp_pose), len(ei_pose)) - 1))
    R_fp_k = fp_pose[k, :3, :3]
    R_ei_k = ei_pose[k, :3, :3]
    base = _ang(R_fp_k, R_ei_k)
    best_F = None
    best_diff = base
    for F in _FLIP_180:
        d = _ang(R_fp_k @ F, R_ei_k)
        if d < best_diff:
            best_diff = d
            best_F = F
    if best_F is None or best_diff > base - 60.0:
        return fp_pose.copy()
    out = fp_pose.copy().astype(np.float32)
    out[:, :3, :3] = fp_pose[:, :3, :3] @ best_F
    print(f"[align_to_ei_canonical] frame-{k} R diff {base:.0f}° → {best_diff:.0f}° via 180° flip")
    return out


def remove_180_flips(poses: np.ndarray, max_jump_deg: float = 60.0,
                     min_improvement_deg: float = 30.0) -> np.ndarray:
    """Cancel frame-to-frame 180° flips caused by mesh-symmetry ambiguity.

    For each consecutive pair where the rotation jumped > max_jump_deg, try
    applying 180° rotation around each of the 3 mesh-local principal axes;
    keep whichever brings the jump closest to the previous frame, provided
    the improvement is at least `min_improvement_deg` (so we don't fight
    against real fast motion). Translation is left alone — symmetry flips
    don't change the visual centroid.

    At 15 fps a real-motion jump < 60°/frame implies < 900°/s, well above
    typical hand-manipulation rates; anything beyond this is almost always
    refiner-induced symmetry flicker, not physical rotation.
    """
    T = len(poses)
    if T < 2:
        return poses.copy()
    out = poses.copy().astype(np.float32)

    n_fixed = 0
    for i in range(1, T):
        R_prev = out[i - 1, :3, :3]
        R_curr = out[i, :3, :3]
        cur_jump = _ang(R_curr, R_prev)
        if cur_jump <= max_jump_deg:
            continue
        # Try 180° flip around each mesh-local principal axis (composed on
        # the right — flip the MESH before the camera transform).
        best_R = R_curr
        best_jump = cur_jump
        for F in _FLIP_180:
            R_cand = R_curr @ F
            j = _ang(R_cand, R_prev)
            if j < best_jump:
                best_jump = j
                best_R = R_cand
        if best_jump <= cur_jump - min_improvement_deg:
            out[i, :3, :3] = best_R
            n_fixed += 1
    if n_fixed:
        print(f"[remove_180_flips] fixed {n_fixed}/{T-1} intra-clip flips")
    return out


# ───────────── EI-derived helpers (kabsch + palm kp + segments) ───────────
# Copied from EgoInfinity/egoinfinity/pipeline/pose_tracker/hand_driven.py so this
# script stays self-contained.
PALM_FRAME_JOINT_INDICES = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)


def palm_keypoints(joints_3d) -> Optional[np.ndarray]:
    """Wrist + 5 MCPs = 6 keypoints that move rigidly with the wrist."""
    if joints_3d is None:
        return None
    j = np.asarray(joints_3d, dtype=np.float64)
    if j.shape[0] < int(PALM_FRAME_JOINT_INDICES.max()) + 1:
        return None
    return j[PALM_FRAME_JOINT_INDICES]


def kabsch_rigid(P: np.ndarray, Q: np.ndarray):
    """Rigid (R, t) such that Q ≈ R @ P + t. No scale."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, Qc - R @ Pc


def find_continuous_segments(mask, min_len: int = 5):
    """Contiguous True-runs of length ≥ min_len, inclusive (start, end)."""
    n = len(mask)
    segs = []
    in_seg = False
    start = 0
    for i in range(n):
        if mask[i] and not in_seg:
            in_seg = True
            start = i
        elif (not mask[i]) and in_seg:
            if i - start >= min_len:
                segs.append((start, i - 1))
            in_seg = False
    if in_seg and (n - start) >= min_len:
        segs.append((start, n - 1))
    return segs


def compute_palm_angular(joints_per_frame, hand_is_right_per_frame) -> np.ndarray:
    """Per-frame palm angular delta (deg) — frame-to-frame Kabsch on the
    palm 6-keypoint subset. 0 when palm kp unavailable for both frames.

    Catches in-hand rotation that the translation-only `motion_score` misses
    (e.g., user holds an object steady but rotates wrist → palm rotates ~5-20°/
    frame but obj cloud bbox center barely moves).
    """
    T = len(joints_per_frame)
    out = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        js0 = joints_per_frame[t - 1] if t - 1 < len(joints_per_frame) else []
        ir0 = hand_is_right_per_frame[t - 1] if t - 1 < len(hand_is_right_per_frame) else []
        js1 = joints_per_frame[t] if t < len(joints_per_frame) else []
        ir1 = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
        for use_right in (True, False):
            kps0 = _get_palm_kps_for_hand(js0, ir0, use_right)
            kps1 = _get_palm_kps_for_hand(js1, ir1, use_right)
            if kps0 is not None and kps1 is not None:
                try:
                    R_h, _ = kabsch_rigid(kps0, kps1)
                except np.linalg.LinAlgError:
                    # SVD non-convergence on degenerate palm kp (NaN, all-zero,
                    # or colinear). Treat this frame as "no angular signal"
                    # rather than crashing the whole bake.
                    out[t] = 0.0
                    break
                cos_a = float(np.clip((np.trace(R_h) - 1.0) / 2.0, -1.0, 1.0))
                out[t] = float(np.degrees(np.arccos(cos_a)))
                break
    return out


def hysteresis_gate(values: np.ndarray, high_thresh: float, low_thresh: float) -> np.ndarray:
    """Schmitt-trigger style gate. Turn ON when values >= high_thresh; turn OFF
    when values < low_thresh; otherwise hold previous state. Eliminates
    boundary flicker from a single threshold."""
    out = np.zeros(len(values), dtype=bool)
    on = False
    for i in range(len(values)):
        if values[i] >= high_thresh:
            on = True
        elif values[i] < low_thresh:
            on = False
        out[i] = on
    return out


def merge_short_gap_segments(segments, max_gap: int = 5):
    """Merge segments whose gap (non-True frames between them) ≤ max_gap.
    Real grasps survive brief hand-detection or contact-signal drop-outs;
    treating them as one continuous episode lets Kabsch propagate cleanly
    through occlusion frames instead of resetting at every "false release".
    """
    if not segments:
        return []
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        prev_end = merged[-1][1]
        if s - prev_end - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _get_palm_kps_for_hand(joints_list, is_right_list, use_right: bool) -> Optional[np.ndarray]:
    for j, ir in zip(joints_list or [], is_right_list or []):
        if (bool(ir) and use_right) or ((not bool(ir)) and (not use_right)):
            kps = palm_keypoints(j)
            if kps is not None:
                return kps
    return None


# ───────────── derive hand signals (which hand grasps, distances) ─────────
def derive_hand_signals(joints_3d_per_frame, hand_is_right_per_frame,
                        fp_pose: np.ndarray,
                        wrist_used: np.ndarray, close: np.ndarray,
                        contact_thresh_m: float = 0.30) -> dict:
    """For each frame, compute the dominant hand (closest wrist to FP++ object
    position) and a binary is_grasp signal (wrist_used OR (close AND dominant
    exists within threshold)).

    Returns dict of (T,) arrays: dominant_hand (list of 'L'/'R'/None),
    is_grasp (bool), d_L, d_R (m).
    """
    T = len(fp_pose)
    obj_t = fp_pose[:, :3, 3]
    dom = []
    d_L = np.full(T, np.nan, dtype=np.float32)
    d_R = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        js = joints_3d_per_frame[t] if t < len(joints_3d_per_frame) else []
        ir = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
        best_L = np.inf
        best_R = np.inf
        for j, isR in zip(js or [], ir or []):
            if j is None or len(j) == 0:
                continue
            wrist = np.asarray(j[0], dtype=np.float64)
            d = float(np.linalg.norm(wrist - obj_t[t]))
            if bool(isR):
                best_R = min(best_R, d)
            else:
                best_L = min(best_L, d)
        d_L[t] = best_L if best_L < np.inf else np.nan
        d_R[t] = best_R if best_R < np.inf else np.nan
        if best_R < best_L and best_R < contact_thresh_m:
            dom.append("R")
        elif best_L <= best_R and best_L < contact_thresh_m:
            dom.append("L")
        else:
            dom.append(None)

    wrist_used = np.asarray(wrist_used, dtype=bool)[:T] if wrist_used is not None else np.zeros(T, dtype=bool)
    close = np.asarray(close, dtype=bool)[:T] if close is not None else np.zeros(T, dtype=bool)
    is_grasp = np.zeros(T, dtype=bool)
    for t in range(T):
        if wrist_used[t]:
            is_grasp[t] = True
        elif close[t] and dom[t] is not None:
            is_grasp[t] = True
    return {"dominant_hand": dom, "is_grasp": is_grasp, "d_L": d_L, "d_R": d_R}


# ───────────── state-aware lock (STATIC/MOVING/GRASPED gating) ─────────────
STATE_STATIC = 0
STATE_MOVING_NOT_GRASPED = 1
STATE_GRASPED = 2
STATE_NAME = {0: "STATIC", 1: "MOVING_NOT_GRASPED", 2: "GRASPED"}


def compute_state_per_frame(is_moving, is_grasp):
    """State priority (grasp > motion > rest), because EI's `is_moving` and
    `wrist_used` are essentially mutually exclusive — when the object is
    held, EI freezes its pose (so is_moving=False even though the hand is
    moving it). The user's semantics require GRASPED to take priority so
    orientation is allowed to change while held.
    """
    T = len(is_moving)
    is_moving = np.asarray(is_moving, dtype=bool)[:T]
    is_grasp = np.asarray(is_grasp, dtype=bool)[:T]
    state = np.zeros(T, dtype=np.int8)
    for t in range(T):
        if is_grasp[t]:
            state[t] = STATE_GRASPED
        elif is_moving[t]:
            state[t] = STATE_MOVING_NOT_GRASPED
        else:
            state[t] = STATE_STATIC
    return state


def state_aware_lock(fp_pose: np.ndarray,
                     motion_score: np.ndarray,
                     init_frame: int,
                     T_anchor: np.ndarray,
                     motion_thresh: float = 0.050,
                     palm_angular: Optional[np.ndarray] = None,
                     palm_ang_thresh: float = 3.0,
                     is_grasp: Optional[np.ndarray] = None,
                     state_per_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """**init_frame-anchored** static lock. Stage 1 of the SAM3D-anchored
    redesign.

    For each contiguous stretch of low-motion frames (`motion_score <
    motion_thresh`):
      • if `init_frame` is inside the stretch → lock all frames to `T_anchor`
        (= T_seq[init_frame], SAM3D's golden reference pose)
      • otherwise → lock to fp_pose at the stretch boundary closer to
        init_frame (its closest "clean" sample). Avoids forcing two different
        rest positions to the same SAM3D pose if the object moved between
        them.

    Non-static frames pass through unchanged. ``is_grasp`` (per-object, per
    frame) takes priority — grasp frames are NEVER treated as static here so
    we don't second-guess hand_rigid's output; the object's R for grasp
    frames is exactly what hand_rigid produced. ``palm_angular`` is global
    (hand activity, not per-object) so it is ONLY consulted for grasp
    frames anyway, but since we now exclude those entirely from is_static,
    the palm_angular gate is effectively irrelevant — keep the param for
    backward-compat but it has no effect when is_grasp is given.
    """
    T = len(fp_pose)
    motion_score = np.asarray(motion_score, dtype=np.float64)[:T]
    out = fp_pose.copy()

    # ★ STRICT static lock: trust EI's per-object state_per_frame classification.
    # For each contiguous STATIC stretch:
    #   - if init_frame ∈ stretch → lock all to T_anchor (SAM3D's golden pose)
    #   - otherwise → lock to median(fp_pose) WITHIN the stretch
    # (Previously: ALL STATIC frames forced to T_anchor regardless of stretch,
    # which caused multi-stretch objects like "sauce bottle held then put down"
    # to teleport from grasp position to T_anchor's mid-grasp position.)
    if state_per_frame is not None:
        sp = np.asarray(state_per_frame, dtype=np.int8)[:T]
        strict_static = (sp == STATE_STATIC)
        T_anchor_f32 = T_anchor.astype(np.float32)
        strict_stretches = find_continuous_segments(strict_static, min_len=1)
        n_anchor = n_local = 0
        for s_start, s_end in strict_stretches:
            if s_start <= init_frame <= s_end:
                # init_frame in stretch → T_anchor
                for t in range(s_start, s_end + 1):
                    out[t] = T_anchor_f32
                n_anchor += (s_end - s_start + 1)
            else:
                # Use median fp_pose in stretch (chordal mean R + median t)
                stretch_R = fp_pose[s_start:s_end + 1, :3, :3]
                stretch_t = fp_pose[s_start:s_end + 1, :3, 3]
                # Filter finite frames
                fin = np.all(np.isfinite(stretch_R), axis=(1, 2)) & \
                      np.all(np.isfinite(stretch_t), axis=1)
                if fin.sum() < 1:
                    # Degenerate: fall back to T_anchor
                    for t in range(s_start, s_end + 1):
                        out[t] = T_anchor_f32
                    n_anchor += (s_end - s_start + 1)
                    continue
                R_med = _chordal_mean_R(stretch_R[fin])
                t_med = np.median(stretch_t[fin], axis=0).astype(np.float32)
                lock_T = T_anchor_f32.copy()
                lock_T[:3, :3] = R_med
                lock_T[:3, 3] = t_med
                for t in range(s_start, s_end + 1):
                    out[t] = lock_T
                n_local += (s_end - s_start + 1)
        if n_anchor + n_local:
            print(f"[init_static_lock] strict (state==STATIC): {n_anchor + n_local}/{T} "
                  f"({n_anchor} → T_anchor, {n_local} → per-stretch median)")
        skip_strict_mask = strict_static
    else:
        skip_strict_mask = np.zeros(T, dtype=bool)

    is_static = (motion_score < motion_thresh) & ~skip_strict_mask
    if is_grasp is not None:
        # Grasp frames are managed by hand_rigid; trust its R there.
        grasped = np.asarray(is_grasp, dtype=bool)[:T]
        is_static = is_static & ~grasped
    elif palm_angular is not None:
        # Legacy path (no is_grasp passed): keep the palm gate to avoid
        # locking when hand is rotating in place.
        pa = np.asarray(palm_angular, dtype=np.float64)[:T]
        is_static = is_static & (pa < palm_ang_thresh)

    segments = find_continuous_segments(is_static, min_len=1)
    n_locked_to_anchor = 0
    n_locked_to_stretch = 0
    for s_start, s_end in segments:
        if s_start <= init_frame <= s_end:
            # init_frame in this stretch → use SAM3D-anchored T_anchor verbatim
            ref_pose = T_anchor.astype(np.float32)
            n_locked_to_anchor += (s_end - s_start + 1)
            for t in range(s_start, s_end + 1):
                out[t] = ref_pose
        else:
            # Non-init stretch → use the IN-STRETCH aggregate of fp_pose,
            # UNLESS the aggregate is close to T_anchor — then prefer
            # T_anchor (kills 10-30° PCA bias on disk-degenerate meshes
            # for objects that are globally static across the whole clip).
            seg_R = fp_pose[s_start:s_end + 1, :3, :3]
            seg_t = fp_pose[s_start:s_end + 1, :3, 3]
            R_med = _chordal_mean_R(seg_R)
            t_med = np.median(seg_t, axis=0).astype(np.float32)
            # Distance from aggregate to anchor
            R_rel = R_med @ T_anchor[:3, :3].T
            cos_a = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
            ang_to_anchor = float(np.degrees(np.arccos(cos_a)))
            if ang_to_anchor < 45.0:
                # Same rest pose as the init_frame anchor → snap to T_anchor
                R_use = T_anchor[:3, :3].astype(np.float32)
                n_locked_to_anchor += (s_end - s_start + 1)
            else:
                # Genuinely different rest pose (object was moved between
                # static stretches) → use the in-stretch aggregate
                R_use = R_med
                n_locked_to_stretch += (s_end - s_start + 1)
            for t in range(s_start, s_end + 1):
                out[t, :3, :3] = R_use
                out[t, :3, 3] = t_med

    if n_locked_to_anchor + n_locked_to_stretch:
        print(f"[init_static_lock] init_frame={init_frame}  "
              f"locked: {n_locked_to_anchor} → T_anchor, "
              f"{n_locked_to_stretch} → per-stretch aggregate")
    return out


# ───────────── hand-rigid grasp lock (Kabsch propagation per segment) ─────
# ─── occlusion signal ─────────────────────────────────────────────────────
def compute_occlusion(obj_data_per_frame, occlusion_thresh: float = 0.30):
    """Per-frame occlusion proxy: True if SAM2 mask area drops below
    `occlusion_thresh` × peak-area-in-clip. Threshold defaults to 30% — only
    severe occlusion triggers, leaving FP++ in charge of moderately-occluded
    frames where it usually still has enough signal. Empirically, frames with
    30-50% visibility often have better FP++ R than hand_rigid (because hand
    kp jitter ~ 1-2° while FP++ R is still well-anchored on the visible part).

    Returns (mask_area: np.ndarray int (T,), is_occluded: np.ndarray bool (T,))
    """
    T = len(obj_data_per_frame)
    area = np.zeros(T, dtype=np.int32)
    for t in range(T):
        od = obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        area[t] = int(bits[: H * W].sum())
    peak = int(area.max()) if area.size else 0
    is_occ = (peak > 0) & (area < occlusion_thresh * peak)
    return area, is_occ


# ─── dual-source translation (obs bbox when visible, palm-rel when occluded) ─
def observation_anchored_translation(fp_pose: np.ndarray,
                                      mesh_xyz: np.ndarray,
                                      obj_data_per_frame,
                                      depth_seq,
                                      focal: float, cx: float, cy: float,
                                      is_grasp: Optional[np.ndarray] = None,
                                      is_occluded: Optional[np.ndarray] = None,
                                      joints_per_frame=None,
                                      hand_is_right_per_frame=None,
                                      dominant_hand_per_frame=None,
                                      mask_step: int = 2) -> np.ndarray:
    """Dual-source translation:
       - non-occluded frames: t from SAM2-mask × depth bbox center
       - occluded grasp frames: t from palm origin + offset captured at a
         "high-trust" anchor in the same segment (offset_in_palm =
         R_palm[anchor]^T @ (t_obj_anchor - palm_origin[anchor]))

    This matches EgoInfinity's `rigid_wrist_binding_propagation` dual-source
    convention. Same MAD + mesh-half-diag outlier filters for the obs branch.
    """
    T = len(fp_pose)
    out = fp_pose.copy()
    mesh_centroid_canon = ((mesh_xyz.min(axis=0) + mesh_xyz.max(axis=0)) * 0.5
                            ).astype(np.float32)
    mesh_half_diag = 0.5 * float(np.linalg.norm(
        mesh_xyz.max(axis=0) - mesh_xyz.min(axis=0)))

    is_occluded = (np.asarray(is_occluded, dtype=bool) if is_occluded is not None
                   else np.zeros(T, dtype=bool))[:T]
    is_grasp_arr = (np.asarray(is_grasp, dtype=bool) if is_grasp is not None
                    else np.zeros(T, dtype=bool))[:T]

    # First pass: compute obs-bbox t for every frame that has a usable mask.
    obs_center = np.full((T, 3), np.nan, dtype=np.float32)
    for t in range(T):
        if t >= len(obj_data_per_frame):
            continue
        od = obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        mask = bits.reshape(H, W).astype(bool)
        if not mask.any():
            continue
        pts = mask_to_pointcloud(mask, depth_seq[t], focal, cx, cy, step=mask_step)
        if len(pts) < 10:
            continue
        # ★ Same iterative direction-aware cleanup as PCA stage — keeps
        # both stages' obs cloud interpretations consistent.
        mesh_extent_vec = mesh_xyz.max(0) - mesh_xyz.min(0)
        clean = clean_obs_cloud_directional(pts, mesh_extent_vec, n_iter=3)
        if len(clean) < 10:
            clean = pts
        # ★ Robust "soft bbox" center: p10/p90 instead of min/max. The cleaned
        # cloud's extremes vary frame-to-frame (different subsets of points
        # survive cleanup), so min/max bbox center jitters. Percentile-based
        # center ignores the most volatile 20% and is much more stable.
        lo = np.percentile(clean, 10.0, axis=0)
        hi = np.percentile(clean, 90.0, axis=0)
        obs_center[t] = ((lo + hi) * 0.5).astype(np.float32)

    # ★ Temporal median filter (window 3) on each contiguous valid segment.
    # NaN-safe by construction — we only smooth within runs where obs_center
    # is defined. Window 3 kills 1-frame spikes without lagging real motion.
    valid = ~np.any(np.isnan(obs_center), axis=1)
    try:
        from scipy.signal import medfilt
    except Exception:
        medfilt = None
    if medfilt is not None:
        smoothed = obs_center.copy()
        for s, e in find_continuous_segments(valid, min_len=3):
            seg = obs_center[s:e + 1]
            L = len(seg)
            win = 3 if L >= 3 else 1
            if win >= 3:
                for ax in range(3):
                    smoothed[s:e + 1, ax] = medfilt(seg[:, ax], kernel_size=win)
        obs_center = smoothed

    # Second pass: for grasp segments, compute palm-relative anchor offset.
    # Anchor = first frame in segment that is NOT occluded and has a valid obs_center.
    palm_relative_anchor = {}   # seg_idx -> (use_right, kps_anchor_wrist (3,), offset_in_palm (3,))
    if joints_per_frame is not None and hand_is_right_per_frame is not None:
        grasp_segments = find_continuous_segments(is_grasp_arr, min_len=3)
        grasp_segments = merge_short_gap_segments(grasp_segments, max_gap=5)
        for si, (s, e) in enumerate(grasp_segments):
            # Decide handedness
            seg_doms = [dominant_hand_per_frame[t] for t in range(s, e + 1)
                        if dominant_hand_per_frame is not None
                        and t < len(dominant_hand_per_frame)
                        and dominant_hand_per_frame[t] is not None]
            if not seg_doms:
                continue
            use_right = sum(1 for d in seg_doms if d == "R") >= sum(1 for d in seg_doms if d == "L")
            # Pick anchor: first visible frame with palm kp + valid obs_center
            anchor = None
            for t in range(s, e + 1):
                if is_occluded[t]:
                    continue
                if np.any(np.isnan(obs_center[t])):
                    continue
                js = joints_per_frame[t] if t < len(joints_per_frame) else []
                ir = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
                kps = _get_palm_kps_for_hand(js, ir, use_right)
                if kps is None:
                    continue
                anchor = (t, kps, use_right)
                break
            if anchor is None:
                continue
            t_a, kps_a, ur = anchor
            wrist_a = kps_a[0]                                # (3,)
            R_a = out[t_a, :3, :3].astype(np.float64)
            # The pose-frame "object position" we want to maintain
            t_obj_a = (obs_center[t_a] - (R_a @ mesh_centroid_canon)).astype(np.float64)
            # Offset of object from wrist, captured at anchor (in world frame).
            # When propagating, we'll re-apply via the palm's Kabsch rotation.
            offset_world_at_anchor = t_obj_a - wrist_a
            palm_relative_anchor[(s, e)] = {
                "use_right": ur,
                "wrist_a": wrist_a,
                "kps_a": kps_a,
                "offset": offset_world_at_anchor,
            }

    # Third pass: assign per-frame t.
    n_obs = n_palm = n_skip = 0
    for t in range(T):
        if not np.any(np.isnan(obs_center[t])) and not is_occluded[t]:
            R = out[t, :3, :3]
            out[t, :3, 3] = obs_center[t] - (R @ mesh_centroid_canon).astype(np.float32)
            n_obs += 1
            continue
        # Occluded (or no mask) → try palm-relative fallback if in a grasp segment
        seg_key = None
        for k in palm_relative_anchor.keys():
            if k[0] <= t <= k[1]:
                seg_key = k
                break
        if seg_key is not None:
            info = palm_relative_anchor[seg_key]
            js = joints_per_frame[t] if t < len(joints_per_frame) else []
            ir = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
            kps_t = _get_palm_kps_for_hand(js, ir, info["use_right"])
            if kps_t is not None:
                R_h, _ = kabsch_rigid(info["kps_a"], kps_t)
                wrist_t = kps_t[0]
                t_new = wrist_t + R_h @ info["offset"]
                R = out[t, :3, :3]
                out[t, :3, 3] = (t_new - (R @ mesh_centroid_canon)).astype(np.float32)
                n_palm += 1
                continue
        # Last resort: leave as-is
        n_skip += 1

    if n_obs + n_palm > 0:
        print(f"[obs_anchored_t] t from obs: {n_obs}, t from palm: {n_palm}, "
              f"unchanged: {n_skip}  (occluded: {int(is_occluded.sum())})")
    return out


# ─── PCA-based rotation estimation (stage 2 of SAM3D-anchored redesign) ──
def compute_pca(pts: np.ndarray):
    """Right-handed PCA frame (3, 3) and descending eigenvalues (3,)."""
    c = pts.mean(axis=0)
    centered = pts - c
    C = (centered.T @ centered) / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(-eigvals)
    R = eigvecs[:, order]
    eig = eigvals[order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R, eig


def clean_obs_cloud_directional(pts: np.ndarray,
                                  mesh_extent: np.ndarray,
                                  n_iter: int = 3) -> np.ndarray:
    """Iterative direction-aware outlier filter for SAM2-mask-derived clouds.

    The problem: SAM2 masks of held objects often include background pixels
    near the mask boundary (table edges, fingers, etc.). Backprojected via
    metric depth, these become long "tails" in the cloud — typically along
    the camera depth axis. A single MAD pass can't kill them because they're
    NUMEROUS (not sparse outliers) and their direction biases the median.

    Iterative direction-aware strategy:
      1) MAD-cap by L2 distance (drops sparse outliers)
      2) PCA the (partially-cleaned) cloud → 3 principal axes
      3) Project pts onto each axis; cap projection along axis i by
         ``mesh_extent[i] / 2 * pad`` (≈ how far along axis i the cloud
         can plausibly extend given mesh size)
      4) Iterate (median + PCA naturally tighten on dense cluster)

    `mesh_extent` is the mesh's full extent along each canonical axis (max-min).
    We use the SORTED sizes (large→small) to match the obs PCA axis order.
    """
    if len(pts) < 30:
        return pts
    mesh_sorted = np.sort(mesh_extent)[::-1]   # large → small, matches PCA order
    half_extents = mesh_sorted * 0.5
    diag = float(np.linalg.norm(mesh_extent))
    pad = 1.5

    for _ in range(n_iter):
        if len(pts) < 30:
            break
        # (1) MAD by L2 + global diag cap
        med = np.median(pts, axis=0)
        d_med = np.linalg.norm(pts - med, axis=1)
        d_mid = float(np.median(d_med))
        mad_v = float(np.median(np.abs(d_med - d_mid)))
        cap = min(d_mid + 3.0 * 1.4826 * mad_v, pad * diag * 0.5)
        new_pts = pts[d_med <= cap]
        if len(new_pts) < 30:
            break

        # (2) PCA on partially cleaned cloud
        R_pca, eigs = compute_pca(new_pts)
        center = new_pts.mean(axis=0)
        # (3) Project pts onto each PCA axis; cap by mesh half-extent (sorted)
        proj = (new_pts - center) @ R_pca   # (N, 3)
        keep = np.ones(len(new_pts), dtype=bool)
        for i in range(3):
            limit = pad * half_extents[i]
            keep &= np.abs(proj[:, i]) <= limit
        new_pts2 = new_pts[keep]
        if len(new_pts2) < 30:
            # too aggressive — relax
            new_pts2 = new_pts
        # converged?
        if len(new_pts2) == len(pts):
            pts = new_pts2
            break
        pts = new_pts2
    return pts


def _min_rot_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimum-angle rotation matrix R such that R @ v_from ≈ v_to."""
    n_from = v_from / (np.linalg.norm(v_from) + 1e-9)
    n_to = v_to / (np.linalg.norm(v_to) + 1e-9)
    cos_a = float(np.clip(np.dot(n_from, n_to), -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle < 1e-5:
        return np.eye(3)
    axis = np.cross(n_from, n_to)
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-9:                       # antiparallel — pick arbitrary perpendicular
        # find any vector not parallel to n_from
        ref = np.array([1.0, 0.0, 0.0]) if abs(n_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(n_from, ref)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
    else:
        axis = axis / axis_n
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _axis_aligned_R(obs_pca: np.ndarray, mesh_pca_R: np.ndarray,
                     T_anchor_R: np.ndarray, mesh_axis_idx: int,
                     max_deviation_deg: float = 60.0):
    """For degenerate shapes (disks → align smallest axis; rods → align largest),
    use a SINGLE-AXIS alignment instead of the D₂ sign-flip enumeration.

    Aligns mesh's axis `mesh_axis_idx` (e.g. 2 = normal for a disk; 0 = long
    axis for a rod) to the corresponding observed PCA axis, and preserves
    T_anchor's rotation around the orthogonal subspace (which is geometrically
    ill-defined for degenerate shapes).

    ★ Continuity guard: if the obs-derived axis deviates from the anchor axis
    by more than ``max_deviation_deg``, return None. A plate's normal can't
    physically flip from horizontal to vertical between frames; if obs PCA
    claims that, the cloud shape is misleading (partial rim view, hand
    occlusion arc, etc.) and we should fall back to T_anchor.
    """
    mesh_axis_canon = mesh_pca_R[:, mesh_axis_idx]
    anchor_axis_world = T_anchor_R @ mesh_axis_canon
    obs_axis = obs_pca[:, mesh_axis_idx]
    # Sign disambiguation: flip if closer to negative anchor direction
    if float(np.dot(obs_axis, anchor_axis_world)) < 0:
        obs_axis = -obs_axis
    # Sanity / continuity check
    cos_a = float(np.clip(np.dot(obs_axis, anchor_axis_world)
                          / (np.linalg.norm(obs_axis) * np.linalg.norm(anchor_axis_world) + 1e-9),
                          -1.0, 1.0))
    dev_deg = float(np.degrees(np.arccos(cos_a)))
    if dev_deg > max_deviation_deg:
        return None
    R_correction = _min_rot_align(anchor_axis_world, obs_axis)
    return R_correction @ T_anchor_R


def symmetry_aware_pca_R(obs_pts: np.ndarray,
                          mesh_pca_R: np.ndarray,
                          T_anchor_R: np.ndarray,
                          mesh_eigvals: Optional[np.ndarray] = None,
                          degeneracy_ratio: float = 1.5,
                          min_pts: int = 30,
                          min_anisotropy: float = 2.0):
    """Estimate object world-rotation from observed point cloud's PCA, with
    sign anchored to ``T_anchor_R`` to handle symmetry naturally.

    Math
    ----
    If world cloud ≈ R_world @ mesh_canonical_pts + t, then
        obs_pca_cols ≈ R_world @ mesh_pca_cols
        →  R_world = obs_pca @ mesh_pca^T   (modulo column signs)

    The 8 sign-flip variants of `obs_pca` (4 of which are right-handed) span
    the D₂ symmetry group of the mesh's PCA frame. For symmetric objects
    (cereal box, can, etc.), several variants give physically equivalent
    poses — picking the one closest to T_anchor.R avoids spurious rotation
    around symmetry axes. For asymmetric objects, only the geometrically
    correct variant is close to anchor; PCA picks the right one.

    Returns
    -------
    (R_world, trust)
      R_world: (3,3) np.float32 or None
      trust:   float — eigenvalue ratio λ_max / λ_min (high = anisotropic,
                       low = nearly isotropic / unreliable)
    """
    if len(obs_pts) < min_pts:
        return None, 0.0
    obs_pca, eigs = compute_pca(obs_pts)
    trust = float(eigs[0] / max(eigs[2], 1e-9))
    if trust < min_anisotropy:
        return None, trust

    # ── Degeneracy detection (uses MESH eigenvalues, not obs — mesh shape is
    # the ground truth for symmetry. obs eigs are noisier and frame-dependent.)
    is_disk = False
    is_rod = False
    if mesh_eigvals is not None:
        if mesh_eigvals[0] / max(mesh_eigvals[1], 1e-9) < degeneracy_ratio:
            is_disk = True   # λ₀ ≈ λ₁  → SO(2) symmetry around 3rd axis (plate)
        elif mesh_eigvals[1] / max(mesh_eigvals[2], 1e-9) < degeneracy_ratio:
            is_rod = True    # λ₁ ≈ λ₂  → SO(2) symmetry around 1st axis (cylinder)

    if is_disk:
        # ★ Extra trust gate for disks: obs cloud's "smallest axis" is only
        # reliable when the cloud is itself clearly disk-like (λ₀ ≫ λ₂ AND
        # λ₁ ≈ λ₀). For partial views of a disk (e.g., only the rim, only
        # the top), the obs PCA's smallest axis points along the camera
        # depth direction instead of the disk's true normal — leading to a
        # 30-40° wrong rotation. Skip when obs isn't disk-shaped enough.
        obs_disk = (eigs[0] / max(eigs[1], 1e-9) < 1.8) and (eigs[1] / max(eigs[2], 1e-9) > 3.0)
        if not obs_disk:
            return None, trust
        # Align mesh's smallest axis (= disk normal) to obs's smallest axis.
        # Rotation around the normal is undefined; preserved from T_anchor.
        R = _axis_aligned_R(obs_pca, mesh_pca_R, T_anchor_R, mesh_axis_idx=2)
        if R is None:                          # continuity guard tripped
            return None, trust
        return R.astype(np.float32), trust

    if is_rod:
        # Obs cloud must be rod-shaped (λ₀ ≫ λ₁) for the long-axis estimate
        # to be reliable. Partial views may give different shapes.
        obs_rod = eigs[0] / max(eigs[1], 1e-9) > 3.0
        if not obs_rod:
            return None, trust
        # Align mesh's largest axis (= rod long axis) to obs's largest axis.
        # Rotation around the long axis is undefined; preserved from T_anchor.
        R = _axis_aligned_R(obs_pca, mesh_pca_R, T_anchor_R, mesh_axis_idx=0)
        if R is None:                          # continuity guard tripped
            return None, trust
        return R.astype(np.float32), trust

    # Non-degenerate: D₂ symmetry handling via 4-variant sign-flip enumeration
    best_R = None
    best_ang = np.inf
    for s0 in (1, -1):
        for s1 in (1, -1):
            for s2 in (1, -1):
                V = obs_pca.copy()
                V[:, 0] *= s0
                V[:, 1] *= s1
                V[:, 2] *= s2
                if np.linalg.det(V) < 0:
                    continue
                R_world = V @ mesh_pca_R.T
                ang = _ang(R_world, T_anchor_R)
                if ang < best_ang:
                    best_ang = ang
                    best_R = R_world
    return (best_R.astype(np.float32) if best_R is not None else None), trust


def pca_anchored_rotation(fp_pose: np.ndarray,
                           obj_data_per_frame, depth_seq,
                           focal: float, cx: float, cy: float,
                           mesh_xyz: np.ndarray,
                           T_anchor: np.ndarray,
                           is_grasp,
                           min_anisotropy: float = 2.0,
                           degeneracy_ratio: float = 1.5,
                           mask_step: int = 2) -> np.ndarray:
    """For all non-grasp frames (motion or static), replace R with the
    PCA-derived rotation. Grasp frames are skipped (hand_rigid handles them).

    Per-frame logic:
      - Compute observed cloud, do MAD outlier filter.
      - symmetry_aware_pca_R picks the best R: D₂ enumeration for non-
        degenerate meshes, or single-axis alignment for disks / rods.
      - Frames with low observation anisotropy (e.g. tiny partial mask)
        keep their input R.

    Stage 2 fix: dropped motion threshold. PCA now applies to static frames
    too, so non-init-frame static stretches get observation-driven R instead
    of inheriting a noisy boundary fp_pose.
    """
    T = len(fp_pose)
    out = fp_pose.copy()
    mesh_pca_R, mesh_eigs = compute_pca(mesh_xyz)
    T_anchor_R = T_anchor[:3, :3].astype(np.float64)

    # is_grasp accepted for backward compatibility (no longer used as a skip).
    _ = is_grasp

    # ★ Apply PCA to ALL frames (including grasp). hand_rigid runs AFTER and
    # overrides PCA only on actively-moving sub-stretches of grasp segments
    # (see hysteresis gate in hand_rigid_grasp_lock). For "hand near static
    # object" frames (grasp=True but motion≈0), PCA's geometric estimate is
    # KEPT — which is what we want (mesh stays at observed orientation).
    n_pca = n_skip_trust = n_skip_nomask = 0
    for t in range(T):
        if t >= len(obj_data_per_frame):
            continue
        od = obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            n_skip_nomask += 1
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        mask = bits.reshape(H, W).astype(bool)
        if not mask.any():
            n_skip_nomask += 1
            continue
        pts = mask_to_pointcloud(mask, depth_seq[t], focal, cx, cy, step=mask_step)
        if len(pts) < 30:
            n_skip_nomask += 1
            continue
        # ★ Iterative direction-aware outlier removal (kills depth-tail
        # contamination from SAM2 mask boundary pixels)
        mesh_extent_vec = mesh_xyz.max(0) - mesh_xyz.min(0)
        clean = clean_obs_cloud_directional(pts, mesh_extent_vec, n_iter=3)
        if len(clean) < 30:
            clean = pts

        R_new, trust = symmetry_aware_pca_R(
            clean, mesh_pca_R, T_anchor_R,
            mesh_eigvals=mesh_eigs,
            degeneracy_ratio=degeneracy_ratio,
            min_anisotropy=min_anisotropy)
        if R_new is None:
            n_skip_trust += 1
            continue
        out[t, :3, :3] = R_new
        n_pca += 1

    if n_pca:
        tag = ""
        if mesh_eigs[0] / max(mesh_eigs[1], 1e-9) < degeneracy_ratio:
            tag = " [disk-degenerate]"
        elif mesh_eigs[1] / max(mesh_eigs[2], 1e-9) < degeneracy_ratio:
            tag = " [rod-degenerate]"
        print(f"[pca_anchored_R] applied to {n_pca}/{T} frames{tag}  "
              f"(skipped: no-mask={n_skip_nomask}, low-trust={n_skip_trust})")
    return out


def _chordal_mean_R(Rs: np.ndarray) -> np.ndarray:
    """Chordal mean of (N, 3, 3) rotation matrices, SVD-projected back to SO(3)."""
    M = Rs.astype(np.float64).mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    d = float(np.sign(np.linalg.det(U @ Vt)))
    return (U @ np.diag([1.0, 1.0, d]) @ Vt).astype(np.float32)


def hand_rigid_grasp_lock(fp_pose: np.ndarray,
                           joints_per_frame, hand_is_right_per_frame,
                           is_grasp, dominant_hand,
                           is_occluded=None,
                           motion_score=None,
                           palm_angular=None,
                           init_frame: Optional[int] = None,
                           T_anchor: Optional[np.ndarray] = None,
                           motion_thresh: float = 0.030,
                           min_seg_len: int = 5,
                           boundary_ramp: int = 5,
                           anchor_window: int = 1) -> np.ndarray:
    """**ROTATION-ONLY** rigid binding to the dominant hand during each
    contiguous grasp segment. For frame t in a segment:

        R_out[t] = R_h(anchor → t) @ R_anchor
        t_out[t] = fp_pose[t, :3, 3]                      # ← keep mask-aligned t

    Keeping translation from the input preserves whatever xy-correction
    `run_egoinfinity.py` did via the SAM2 mask center (Kalman update on
    `est.pose_last.xy`). Rigidly binding TRANSLATION too (the previous
    behavior) coupled the mesh to the wrist joint and dragged it off the
    observed cloud by 5-15cm — the wrist isn't the object center.

    `anchor_window` (default 5) frames of valid keypoints at the segment
    start get their R averaged (chordal mean) to make the reference R less
    sensitive to a single noisy refiner output.

    Boundary SLERP ramp blends rotation back to upstream pose at segment end.
    """
    T = len(fp_pose)
    out = fp_pose.copy().astype(np.float32)
    segments = find_continuous_segments(is_grasp, min_len=min_seg_len)
    segments = merge_short_gap_segments(segments, max_gap=5)
    if not segments:
        return out

    from scipy.spatial.transform import Slerp

    n_replaced = 0
    for s_start, s_end in segments:
        # Majority handedness in this segment, with hard preference (not
        # frame-by-frame fallback — fall-back caused phantom flips when
        # the dominant hand wasn't detected briefly).
        seg_doms = [dominant_hand[t] for t in range(s_start, s_end + 1)
                    if t < len(dominant_hand) and dominant_hand[t] is not None]
        if not seg_doms:
            continue
        use_right = sum(1 for d in seg_doms if d == "R") >= sum(1 for d in seg_doms if d == "L")

        # Collect frames with palm-kp for the CHOSEN hand only (no fallback).
        valid = []
        for t in range(s_start, s_end + 1):
            js = joints_per_frame[t] if t < len(joints_per_frame) else []
            ir = hand_is_right_per_frame[t] if t < len(hand_is_right_per_frame) else []
            kps = _get_palm_kps_for_hand(js, ir, use_right)
            if kps is not None:
                valid.append((t, kps))
        if len(valid) < 2:
            continue

        # ★ Reference selection — gated by init_frame's relationship to the
        # grasp segment:
        #
        #  (A) init_frame ∈ [s_start, s_end] AND palm kp at init_frame
        #      available for chosen hand:
        #         R_anchor = T_anchor.R  (= SAM3D's pose at init_frame)
        #         kps_ref  = palm[init_frame]
        #      Per frame: R[t] = Kabsch(palm[init], palm[t]) @ T_anchor.R
        #      Mathematically exact: when init_frame is mid-grasp, T_anchor's
        #      R is the object's true R then, and palm rotation correctly
        #      tracks subsequent reorientation.
        #
        #  (B) init_frame NOT in this grasp segment, OR palm unavailable:
        #         Fall back to pre-grasp PCA chordal mean (existing logic).
        #      Don't reuse T_anchor as anchor here — object may have been
        #      at a different orientation when grasp started (e.g., plate
        #      clip: init_frame=181 with plate vertical, but earlier grasp
        #      frames had plate going from flat to vertical).
        kps_ref = None
        R_anchor = None
        if (init_frame is not None and T_anchor is not None
                and s_start <= init_frame <= s_end
                and 0 <= init_frame < len(joints_per_frame)):
            js_init = joints_per_frame[init_frame]
            ir_init = hand_is_right_per_frame[init_frame]
            init_palm = _get_palm_kps_for_hand(js_init, ir_init, use_right)
            if init_palm is not None:
                kps_ref = init_palm
                R_anchor = T_anchor[:3, :3].astype(np.float64)

        if kps_ref is None:
            ref_t, kps_ref = valid[0]
            pre_window = 5
            pre_start = max(0, s_start - pre_window)
            if pre_start < s_start:
                R_anchor = _chordal_mean_R(
                    fp_pose[pre_start:s_start, :3, :3]).astype(np.float64)
            else:
                pool = valid[0: max(1, anchor_window)]
                R_anchor = _chordal_mean_R(
                    np.stack([fp_pose[i, :3, :3] for i, _ in pool])).astype(np.float64)
            # ★ Snap to T_anchor when close. For globally-static objects
            # (e.g. plate on table), pre-grasp PCA mean carries a 10-30°
            # disk-degenerate bias from cloud asymmetry. T_anchor (SAM3D's
            # init_frame pose) is the gold reference; if the pre-grasp
            # aggregate is within 30° of it, the object's true rest pose
            # is T_anchor's pose and we should use it as R_anchor.
            if T_anchor is not None:
                T_anchor_R64 = T_anchor[:3, :3].astype(np.float64)
                R_rel = R_anchor @ T_anchor_R64.T
                cos_a = float(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
                if float(np.degrees(np.arccos(cos_a))) < 30.0:
                    R_anchor = T_anchor_R64

        # ★ Segment-level motion+rotation gate. If the OBJECT barely moves
        # AND the user's palm barely rotates, the user is touching but not
        # reorienting the object (e.g. hand resting on a plate on the
        # table) → freeze the whole segment at R_anchor.
        #
        # The palm-rotation check is critical: an object being dispensed
        # (e.g. seasoning bottle being tilted to pour) has small per-frame
        # translation (median < 50mm) but the hand IS rotating it. Without
        # the palm gate, those legitimate reorientations get frozen too.
        #
        # Median (not max) on motion is robust to depth-tail spikes on the
        # mask boundary; mean on palm_angular catches sustained low-rate
        # rotation (e.g. 2°/frame for 100 frames = 200° cumulative).
        static_seg_thresh_m = 0.050   # 50 mm median → "object basically still"
        palm_active_thresh_deg = 1.0  # 1°/frame mean → palm actively rotating
        seg_static = False
        if motion_score is not None:
            seg_m = np.asarray(motion_score[s_start:s_end + 1], dtype=np.float64)
            mot_low = len(seg_m) > 0 and float(np.median(seg_m)) < static_seg_thresh_m
            palm_quiet = True
            if palm_angular is not None:
                seg_p = np.asarray(palm_angular[s_start:s_end + 1], dtype=np.float64)
                palm_quiet = (len(seg_p) > 0
                              and float(np.mean(seg_p)) < palm_active_thresh_deg)
            if mot_low and palm_quiet:
                seg_static = True
        if seg_static:
            for t in range(s_start, s_end + 1):
                out[t, :3, :3] = R_anchor.astype(np.float32)
                n_replaced += 1
            mp = (float(np.mean(np.asarray(palm_angular[s_start:s_end + 1])))
                  if palm_angular is not None else 0.0)
            print(f"[hand_rigid_grasp_lock]   seg [{s_start},{s_end}] static "
                  f"(med mot={float(np.median(seg_m)*1000):.0f}mm, "
                  f"mean palm={mp:.1f}°) → frozen at R_anchor")
            continue

        # ★ Apply hand_rigid to ALL grasp frames with detected palm (no
        # per-frame active gate). Previously we hysteresis-gated on motion +
        # palm_angular and only acted on "active" sub-stretches, but the
        # transitions at active↔inactive boundaries created 100°+ R jumps
        # (hand_rigid R diverges by 90° from PCA R for symmetric meshes).
        valid_by_t = dict(valid)
        last_R = None
        for t in range(s_start, s_end + 1):
            if t in valid_by_t:
                kps_t = valid_by_t[t]
                R_h, _ = kabsch_rigid(kps_ref, kps_t)
                new_R = (R_h @ R_anchor).astype(np.float32)
                out[t, :3, :3] = new_R
                last_R = new_R
                n_replaced += 1
            elif last_R is not None:
                # palm missing this frame → hold previous hand_rigid R
                out[t, :3, :3] = last_R
                n_replaced += 1

        # Boundary SLERP ramp on rotation only
        if boundary_ramp > 0 and s_end + 1 < T:
            for k in range(boundary_ramp):
                tk = s_end + 1 + k
                if tk >= T:
                    break
                u = (k + 1) / (boundary_ramp + 1)
                R_a = out[tk, :3, :3]
                R_b = fp_pose[tk, :3, :3]
                slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_a, R_b])))
                out[tk, :3, :3] = slerp([u]).as_matrix()[0].astype(np.float32)

    if n_replaced:
        print(f"[hand_rigid_grasp_lock] {len(segments)} seg(s), {n_replaced} R-frames bound "
              f"(t preserved from input)")
    return out


def _mask_iou_per_segment(obj_data_per_frame, s: int, e: int) -> float:
    """Median frame-to-frame mask IoU across segment [s, e]. High IoU
    (> 0.85) → object's image footprint is essentially stationary."""
    ious = []
    prev = None
    for t in range(s, e + 1):
        if t >= len(obj_data_per_frame):
            break
        od = obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            prev = None
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        mask = bits.reshape(H, W).astype(bool)
        if prev is not None and prev.shape == mask.shape:
            inter = int((prev & mask).sum())
            union = int((prev | mask).sum())
            if union > 0:
                ious.append(inter / union)
        prev = mask
    return float(np.median(ious)) if ious else 0.0


def _per_frame_pca_R(clip) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R_per_frame: (T,3,3), ok_per_frame: (T,) bool). For frames
    where symmetry_aware_pca_R failed (low trust / no mask), ok=False."""
    T = clip.T
    mesh_pca_R, mesh_eigs = compute_pca(clip.mesh_xyz)
    T_anchor_R64 = clip.T_anchor[:3, :3].astype(np.float64)
    pca_R = np.full((T, 3, 3), np.nan, dtype=np.float64)
    pca_ok = np.zeros(T, dtype=bool)
    mesh_ext = clip.mesh_xyz.max(0) - clip.mesh_xyz.min(0)
    for t in range(T):
        if t >= len(clip.obj_data_per_frame):
            continue
        od = clip.obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        mask = bits.reshape(H, W).astype(bool)
        if not mask.any():
            continue
        pts = mask_to_pointcloud(mask, clip.depth_seq[t],
                                  clip.focal, clip.cx, clip.cy, step=2)
        if len(pts) < 30:
            continue
        clean = clean_obs_cloud_directional(pts, mesh_ext, n_iter=3)
        if len(clean) < 30:
            clean = pts
        R_new, _ = symmetry_aware_pca_R(clean, mesh_pca_R, T_anchor_R64,
                                         mesh_eigvals=mesh_eigs)
        if R_new is not None:
            pca_R[t] = R_new
            pca_ok[t] = True
    return pca_R, pca_ok


def _per_frame_obs_eigs(clip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (long_axis_unit_vec, λ1/λ2, λ2/λ3) for the obs cloud.

    Used by the OBB_LONG_AXIS branch in ``obb_priority_lock``.  Long axis
    is the first principal direction (descending eigenvalue order); the
    two ratios characterise how rod-like the cloud is.

    Returns:
        long_axis: (T, 3) — unit vector of the dominant axis, NaN for
                            frames where PCA failed
        r12, r23 : (T,)   — eigenvalue ratios (1 if frame failed)
    """
    T = clip.T
    long_axis = np.full((T, 3), np.nan, dtype=np.float64)
    r12 = np.ones(T, dtype=np.float64)
    r23 = np.ones(T, dtype=np.float64)
    mesh_ext = clip.mesh_xyz.max(0) - clip.mesh_xyz.min(0)
    for t in range(T):
        if t >= len(clip.obj_data_per_frame):
            continue
        od = clip.obj_data_per_frame[t]
        if od is None or "mask_packed" not in od:
            continue
        H, W = int(od["mask_shape"][0]), int(od["mask_shape"][1])
        bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H * W]
        mask = bits.reshape(H, W).astype(bool)
        if not mask.any():
            continue
        pts = mask_to_pointcloud(mask, clip.depth_seq[t],
                                  clip.focal, clip.cx, clip.cy, step=2)
        if len(pts) < 30:
            continue
        clean = clean_obs_cloud_directional(pts, mesh_ext, n_iter=3)
        if len(clean) < 30:
            clean = pts
        R_pca, eigs = compute_pca(clean)
        long_axis[t] = R_pca[:, 0]
        r12[t] = float(eigs[0] / max(eigs[1], 1e-9))
        r23[t] = float(eigs[1] / max(eigs[2], 1e-9))
    return long_axis, r12, r23


def _long_axis_dominant_R(obs_long_world: np.ndarray, hand_R: np.ndarray,
                           mesh_long_canonical: np.ndarray) -> np.ndarray:
    """Build R such that the mesh's CANONICAL long axis maps to the
    observed long axis in world, while staying as close to hand_R as
    possible.

    SAM3D meshes have their long axis along an arbitrary canonical
    direction (not necessarily +X), so a naive ``R[:, 0] = obs_long``
    would align the wrong mesh axis — visually putting the knife
    perpendicular to its real direction.  Instead we:

      1. compute m_world = hand_R @ mesh_long_canonical  — where the
         mesh's true long axis currently points in world frame
      2. find the minimum-angle rotation R_align that takes m_world to
         obs_long (sign-corrected)
      3. return R_align @ hand_R

    The result has ``R @ mesh_long_canonical = obs_long`` (so the mesh's
    real long axis aligns with the obs OBB) while the rotation AROUND
    that axis follows hand_R.
    """
    w = np.asarray(obs_long_world, dtype=np.float64)
    n_w = np.linalg.norm(w)
    if n_w < 1e-6:
        return hand_R.astype(np.float64)
    w = w / n_w
    m = np.asarray(mesh_long_canonical, dtype=np.float64)
    n_m = np.linalg.norm(m)
    if n_m < 1e-6:
        return hand_R.astype(np.float64)
    m = m / n_m
    hand_R = hand_R.astype(np.float64)
    m_world = hand_R @ m
    nmw = np.linalg.norm(m_world)
    if nmw < 1e-6:
        return hand_R
    m_world = m_world / nmw
    # Note: w must already be sign-corrected by the caller (temporal
    # stitching across the segment).  We do NOT flip its sign here —
    # that would override the caller's choice and reintroduce frame-to-
    # frame 180° flips.
    cos_a = float(np.clip(m_world @ w, -1.0, 1.0))
    if cos_a > 1.0 - 1e-9:
        return hand_R   # already aligned
    axis = np.cross(m_world, w)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # Anti-parallel (shouldn't happen after sign-correction, but defensive)
        # Pick any perpendicular axis to m_world.
        if abs(m_world[0]) < 0.9:
            axis = np.array([1.0, 0.0, 0.0]) - m_world[0] * m_world
        else:
            axis = np.array([0.0, 1.0, 0.0]) - m_world[1] * m_world
        axis = axis / (np.linalg.norm(axis) + 1e-9)
    else:
        axis = axis / axis_norm
    angle = float(np.arccos(cos_a))
    # Rodrigues' formula for R_align
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    R_align = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R_align @ hand_R


def obb_priority_lock(fp_pose: np.ndarray, clip,
                       obb_dR_static_max: float = 1.0,
                       obb_dR_track_max: float = 15.0,
                       pca_conf_min: float = 0.70,
                       mask_iou_min: float = 0.85,
                       outlier_R_deg: float = 30.0,
                       rod_r12_min: float = 2.5,
                       rod_r23_min: float = 1.3) -> np.ndarray:
    """For each grasp segment, decide R policy via three tiers:

      1. OBB stable & static  (PCA confident ≥70%, median per-frame ΔR < 1°)
         → chordal mean of PCA R across segment (one R)
      2. OBB stable & tracking (median per-frame ΔR < 5°)
         → per-frame PCA R with outlier-clamp at 30° from chordal mean
      3. mask IoU ≥ 0.85
         → freeze at T_anchor
      4. else
         → leave input alone (hand_rigid output)

    Run AFTER hand_rigid_grasp_lock and BEFORE state_aware_lock, so the
    OBB-driven R overrides hand_rigid's Kabsch when the obs cloud agrees
    on a stable orientation. Final smooth_se3 cleans the boundaries.
    """
    out = fp_pose.copy()
    if not clip.is_grasp_per_frame.any():
        return out
    grasp_segs = find_continuous_segments(clip.is_grasp_per_frame, min_len=3)
    grasp_segs = merge_short_gap_segments(grasp_segs, max_gap=5)
    if not grasp_segs:
        return out

    pca_R, pca_ok = _per_frame_pca_R(clip)
    # Per-frame obs cloud long-axis vector + eigenvalue ratios (for
    # OBB_LONG_AXIS rod-priority branch).
    long_axis, r12_per_frame, r23_per_frame = _per_frame_obs_eigs(clip)
    # Mesh's CANONICAL long axis direction (which canonical direction
    # corresponds to the mesh's longest extent).  Needed to map the obs
    # long axis to the right mesh axis (not necessarily canonical X).
    mesh_canonical_R, _mesh_eigs = compute_pca(clip.mesh_xyz)
    mesh_long_canonical = mesh_canonical_R[:, 0].astype(np.float64)
    n_obb_static = n_obb_track = n_static = n_hand = n_long_axis = 0
    T_anchor_R = clip.T_anchor[:3, :3].astype(np.float32)

    # Palm-rotation gate shared by both OBB_STATIC and IoU-static branches.
    # Even when obs PCA looks stable, if the palm is clearly rotating the
    # object is being reoriented — never freeze R.  Rod-degenerate obs
    # clouds (long thin bottles) produce a stable but ARBITRARY R for the
    # two weak axes; only palm motion can disambiguate whether the object
    # is really rotating.
    def _palm_active(s, e):
        if clip.palm_angular_per_frame is None:
            return False
        seg_p = np.asarray(clip.palm_angular_per_frame[s:e + 1],
                            dtype=np.float64)
        return len(seg_p) > 0 and float(np.mean(seg_p)) >= 1.0   # 1°/f

    for s, e in grasp_segs:
        seg_pca_ok = pca_ok[s:e + 1]
        rate = float(seg_pca_ok.sum() / max(len(seg_pca_ok), 1))
        diffs = [_ang(pca_R[t], pca_R[t - 1])
                 for t in range(s + 1, e + 1) if pca_ok[t] and pca_ok[t - 1]]
        median_dR = float(np.median(diffs)) if diffs else 180.0
        palm_active = _palm_active(s, e)

        # ★ OBB_LONG_AXIS branch (checked FIRST, before palm_active).
        # For rod-strong objects (λ1/λ2 > 2.5 AND λ2/λ3 > 1.3) the obs
        # cloud's long axis is well-defined.  Use it as the mesh's
        # principal axis, and resolve the rotation AROUND that axis
        # from hand_rigid R.  This gives thin tools (knife, fork, screw-
        # driver) the correct long-axis alignment without losing hand-
        # driven roll information.
        #
        # Differs from OBB_TRACK: that branch uses the *full* PCA R
        # (long + two weak axes), which is unreliable on rods because
        # the two weak axes are nearly degenerate (λ2 ≈ λ3) and can
        # flip 90° randomly.  OBB_LONG_AXIS only trusts the LONG axis
        # and uses hand_rigid for the remaining two.
        seg_r12 = r12_per_frame[s:e + 1]
        seg_r23 = r23_per_frame[s:e + 1]
        median_r12 = float(np.median(seg_r12))
        median_r23 = float(np.median(seg_r23))
        is_rod_strong = (median_r12 > rod_r12_min
                         and median_r23 > rod_r23_min)

        if is_rod_strong:
            # ★ Segment-level long-axis correction (hand-driven frame-to-frame).
            # Compute ONE rotation R_corr per segment that aligns the
            # mesh's long axis with the segment-averaged obs PCA long axis;
            # apply R_corr to every frame's hand_rigid R.  Hand R drives
            # all per-frame variation; obs OBB only provides the overall
            # axis reference (no per-frame PCA noise propagated into R).
            #
            # Step 1: collect per-frame obs_long, sign-stitch across segment.
            stitched = []
            prev_v1 = None
            for t in range(s, e + 1):
                v1 = long_axis[t]
                if not np.all(np.isfinite(v1)):
                    continue
                v1 = v1 / (np.linalg.norm(v1) + 1e-9)
                if prev_v1 is None:
                    # First valid frame: sign against hand_R at that frame
                    hand_R_t = fp_pose[t, :3, :3].astype(np.float64)
                    m_world_t = hand_R_t @ mesh_long_canonical
                    nmw = np.linalg.norm(m_world_t)
                    if nmw > 1e-6:
                        if float(v1 @ (m_world_t / nmw)) < 0:
                            v1 = -v1
                else:
                    if float(v1 @ prev_v1) < 0:
                        v1 = -v1
                stitched.append(v1)
                prev_v1 = v1

            if len(stitched) < 3:
                # Not enough valid PCA frames — fall through to hand_rigid
                n_hand += 1
                continue

            # Step 2: segment-averaged direction (renormalised mean).
            seg_obs_long = np.mean(np.stack(stitched, axis=0), axis=0)
            n_seg = np.linalg.norm(seg_obs_long)
            if n_seg < 0.3:
                # Too dispersed (per-frame directions inconsistent) — the
                # rod might actually be rotating frame-to-frame.  Fall back.
                n_hand += 1
                continue
            seg_obs_long = seg_obs_long / n_seg

            # Step 3: anchor frame = segment midpoint.
            t_anchor = s + (e - s) // 2
            hand_R_anchor = fp_pose[t_anchor, :3, :3].astype(np.float64)
            m_anchor = hand_R_anchor @ mesh_long_canonical
            nma = np.linalg.norm(m_anchor)
            if nma < 1e-6:
                n_hand += 1
                continue
            m_anchor = m_anchor / nma

            # Step 4: minimum-angle rotation taking m_anchor → seg_obs_long.
            cos_a = float(np.clip(m_anchor @ seg_obs_long, -1.0, 1.0))
            if cos_a > 1.0 - 1e-9:
                R_corr = np.eye(3)
            else:
                axis = np.cross(m_anchor, seg_obs_long)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-9:
                    # 180° apart (shouldn't happen after sign-stitching)
                    n_hand += 1
                    continue
                axis = axis / axis_norm
                angle = float(np.arccos(cos_a))
                K = np.array([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ], dtype=np.float64)
                R_corr = (np.eye(3)
                           + np.sin(angle) * K
                           + (1.0 - np.cos(angle)) * (K @ K))

            # Step 5: apply R_corr to every frame's hand_R.
            for t in range(s, e + 1):
                hand_R_t = fp_pose[t, :3, :3].astype(np.float64)
                out[t, :3, :3] = (R_corr @ hand_R_t).astype(np.float32)
            n_long_axis += 1
            continue

        # ★ Palm-active short-circuit: when the palm is actively rotating,
        # the user IS reorienting the object regardless of what the obs
        # cloud's PCA looks like.  Skip remaining OBB override branches
        # (OBB_STATIC, OBB_TRACK, mask-IoU freeze) and keep hand_rigid R.
        # Only applies when the cloud isn't rod-strong (those are handled
        # above by OBB_LONG_AXIS).
        if palm_active:
            n_hand += 1
            continue

        if rate >= pca_conf_min and median_dR < obb_dR_track_max:
            # OBB stable
            Rs = np.stack([pca_R[t] for t in range(s, e + 1) if pca_ok[t]])
            R_mean = _chordal_mean_R(Rs)
            if median_dR < obb_dR_static_max:
                # OBB_STATIC: one R for whole segment.  palm_active already
                # short-circuited above; this branch only fires when palm
                # is quiet AND obs PCA is stable → safe to freeze.
                for t in range(s, e + 1):
                    out[t, :3, :3] = R_mean.astype(np.float32)
                n_obb_static += 1
            else:
                # OBB_TRACK: per-frame PCA, clamp outliers to R_mean
                last = R_mean.astype(np.float32)
                for t in range(s, e + 1):
                    if pca_ok[t] and _ang(pca_R[t], R_mean) <= outlier_R_deg:
                        last = pca_R[t].astype(np.float32)
                    out[t, :3, :3] = last
                n_obb_track += 1
        else:
            # PCA not confident enough for OBB_TRACK; fall back to
            # IoU-based freeze (palm already known to be quiet because
            # palm_active short-circuited at top of loop).
            iou = _mask_iou_per_segment(clip.obj_data_per_frame, s, e)
            if iou >= mask_iou_min:
                # mask stationary → freeze to T_anchor.
                for t in range(s, e + 1):
                    out[t, :3, :3] = T_anchor_R
                n_static += 1
            else:
                # mask isn't stationary either → leave hand_rigid output.
                n_hand += 1

    if n_obb_static + n_obb_track + n_static + n_hand + n_long_axis:
        print(f"[obb_priority_lock] segments: OBB_LONG_AXIS={n_long_axis} "
              f"OBB_STATIC={n_obb_static} OBB_TRACK={n_obb_track} "
              f"STATIC={n_static} HAND={n_hand}")
    return out


def smooth_se3(poses: np.ndarray, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Translation: SavGol per-axis. Rotation: SavGol on each matrix element
    (9 channels), then SVD-project back to SO(3) per frame.

    The previous rotvec-based version blew up when the rotation sequence
    happened to cross the ±π discontinuity of axis-angle, causing 100°+ max
    jumps. Matrix-element smoothing followed by SVD projection avoids that
    entirely while still being numerically equivalent for smooth segments.
    """
    T = len(poses)
    if T < 5:
        return poses.copy()
    w = min(window, T)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return poses.copy()

    out = poses.copy()
    # translation
    for i in range(3):
        out[:, i, 3] = savgol_filter(poses[:, i, 3], w, polyorder)
    # rotation: smooth each entry of R, then nearest-SO(3) projection per frame
    R_flat = poses[:, :3, :3].reshape(T, 9)
    R_smooth = np.stack([savgol_filter(R_flat[:, i], w, polyorder) for i in range(9)],
                         axis=-1).reshape(T, 3, 3)
    for t in range(T):
        U, _, Vt = np.linalg.svd(R_smooth[t].astype(np.float64))
        d = float(np.sign(np.linalg.det(U @ Vt)))
        out[t, :3, :3] = (U @ np.diag([1.0, 1.0, d]) @ Vt).astype(np.float32)
    return out


# ────────────────── module-level constants used by load_clip ───────────────
COLOR_FP = (0, 200, 255)
COLOR_EI = (255, 180, 60)
COLOR_OBS = (200, 200, 200)

# ────────────────── clip data container ────────────────────────────────────
@dataclass
class ClipData:
    clip_id: str
    T: int
    focal: float
    cx: float
    cy: float
    H_img: int
    W_img: int
    rgb_seq: List[np.ndarray]
    depth_seq: List[np.ndarray]
    fp_pose: np.ndarray              # (T, 4, 4) raw
    fp_pose_unflipped: np.ndarray    # raw + 180° flip removal (cached)
    ei_pose: Optional[np.ndarray]    # (T, 4, 4) or None
    # state signals (per frame)
    is_moving_per_frame: np.ndarray
    wrist_used_per_frame: np.ndarray
    close_per_frame: np.ndarray
    is_grasp_per_frame: np.ndarray
    dominant_hand_per_frame: list    # 'L'|'R'|None per frame
    grasp_hand_raw_per_frame: list   # 'L'|'R'|'both'|None per frame (from pkl, pre-resolution)
    d_L_per_frame: np.ndarray        # (T,) m
    d_R_per_frame: np.ndarray
    state_per_frame: np.ndarray      # 0=STATIC, 1=MOVING_NOT_GRASPED, 2=GRASPED
    is_occluded_per_frame: np.ndarray   # bool, mask area < 50% peak
    mask_area_per_frame: np.ndarray     # int pixel count
    anchor_t: int                       # EI pose_tracker's strongest-grip frame
    # ★ SAM3D-anchored reference (the golden truth in this clip) ──────────
    init_frame: int                  # SAM3D reconstruction frame
    motion_score_per_frame: np.ndarray  # (T,) m — pc_motion_xy + pc_motion_z
    palm_angular_per_frame: np.ndarray  # (T,) deg — Kabsch on palm kp t-1→t
    T_anchor: np.ndarray             # (4, 4) — T_seq[init_frame] when EI present,
                                     #          else fp_pose_unflipped[init_frame]
    mesh_xyz: np.ndarray
    mesh_rgb: np.ndarray
    fp_mesh_rgb: np.ndarray
    ei_mesh_rgb: np.ndarray
    mano_faces: Optional[np.ndarray]
    gravity_up: np.ndarray
    joints_3d: List[List[np.ndarray]]
    vertices_3d: List[List[np.ndarray]]
    hand_is_right: List[List[bool]]
    obj_data_per_frame: List[Optional[dict]]
    grid_origin: np.ndarray
    oid: int


def load_clip(testcase: Path, pkl: Path, oid_override: Optional[int] = None,
              max_w: int = 480) -> ClipData:
    """Load a clip from disk (pkl + testcase/pose.npy) into memory."""
    d = load_pkl(pkl)
    fd = d["frame_data"]
    pti = d.get("pose_track_info") or {}
    mi = d.get("sam3_mesh_info") or {}

    # pick oid
    oid = oid_override
    if oid is None:
        # use whichever oid the converter picked (mirror auto-pick logic)
        fd0_sd = fd[0].get("sam3_obj_data") or {}
        cands = []
        for k, pi in pti.items():
            if pi.get("T_seq") is None: continue
            od = fd0_sd.get(k)
            if od and "mask_packed" in od and "mask_shape" in od:
                m = unpack_mask(od["mask_packed"], od["mask_shape"])
                if m.any():
                    cands.append(k)
        if not cands:
            raise RuntimeError(f"{testcase.name}: no valid oid")
        oid = sorted(cands)[0]

    fp_pose = np.load(testcase / "pose.npy").astype(np.float32)
    T = min(len(fd), len(fp_pose))
    fp_pose = fp_pose[:T]

    ei_pose = None
    pi = pti.get(oid, {})
    if pi.get("T_seq") is not None:
        ei_pose = np.asarray(pi["T_seq"], dtype=np.float32)[:T]

    # FP++ pose.npy may contain NaN frames at [0, init_frame_idx-1] when
    # SAM2 mask was absent at frame 0 (register frame had to be moved
    # forward). Fill those frames with EI's T_seq so downstream stages
    # see valid data. bake_fp_pose still falls back to old T_seq for
    # NaN frames via its own check.
    if ei_pose is not None:
        for t in range(T):
            if not np.all(np.isfinite(fp_pose[t])):
                fp_pose[t] = ei_pose[t]

    info = mi[oid]
    # Read SAM3D's reconstruction frame (the golden anchor frame for this clip).
    # SAM3D often picks a clear-view frame mid/late in the clip, not frame 0;
    # `init_frame` tells us which one.
    init_frame_idx = int(info.get("init_frame", 0))
    init_frame_idx = max(0, min(init_frame_idx, T - 1))

    # Anti-flip pipeline: compare FP vs EI rotation at init_frame (the most
    # trustworthy EI pose) and apply the matching 180° flip globally.
    fp_pose_unflipped = remove_180_flips(
        align_to_ei_canonical(fp_pose, ei_pose, ref_frame=init_frame_idx))
    ply_path = (pkl.parent / info["ply_path"]).resolve()
    xyz, rgb = load_gs_ply(ply_path)
    sc = pi.get("scale_correction") or info.get("canonical_scale") or 1.0
    xyz = xyz * float(sc)

    focal = float(d["dp_focal"])
    cx = float(d["cx"])
    cy = float(d["cy"])

    mano_faces = (np.asarray(d["mano_faces"], dtype=np.int32)
                  if d.get("mano_faces") is not None else None)

    rgb_seq, depth_seq = [], []
    joints_3d, vertices_3d, hand_is_right = [], [], []
    obj_data_per_frame = []
    for f in fd[:T]:
        rgb_im = decode_jpeg(f["img_rgb"])
        h, w = rgb_im.shape[:2]
        if w > max_w:
            s = max_w / w
            rgb_im = cv2.resize(rgb_im, (max_w, int(h * s)))
        rgb_seq.append(rgb_im)
        depth_seq.append(decode_depth_m(f["depth_png"]))
        joints_3d.append(f.get("joints_3d_pred") or [])
        vertices_3d.append(f.get("vertices_3d") or [])
        hand_is_right.append(f.get("hand_is_right") or [])
        obj_data_per_frame.append((f.get("sam3_obj_data") or {}).get(oid))
    H_img, W_img = depth_seq[0].shape

    gu = np.asarray(d.get("gravity_up", [0, -1, 0]), dtype=np.float32)
    gu = gu / (np.linalg.norm(gu) + 1e-9)

    # grid origin: median wrist position
    wrists = []
    for js in joints_3d:
        for j in js:
            if j is not None and len(j) > 0:
                wrists.append(np.asarray(j[0], dtype=np.float32))
    origin = np.median(np.array(wrists), axis=0) if wrists else np.array([0, 0, 1.0])
    origin = origin - gu * 0.5

    # ── state signals from EI pose_track_info + derived dominant hand ──
    is_moving = np.asarray(pi.get("is_moving_per_frame") or np.zeros(T, dtype=bool),
                           dtype=bool)[:T]
    wrist_used = np.asarray(pi.get("wrist_used_per_frame") or np.zeros(T, dtype=bool),
                            dtype=bool)[:T]
    close = np.asarray(pi.get("close_per_frame") or np.zeros(T, dtype=bool),
                       dtype=bool)[:T]
    anchor_t = int(pi.get("anchor_t") or 0)
    sigs = derive_hand_signals(joints_3d, hand_is_right,
                                fp_pose_unflipped, wrist_used, close)
    # refresh_grasp_veto curates wrist_l/r_per_frame using MEMFOF object flow;
    # trust those as authoritative is_grasp. derive_hand_signals's (close AND
    # dom) fallback would resurrect flow-vetoed FP grasps (e.g., plate sitting
    # on table near hand) since `close` is not touched by veto.
    pkl_wl = pi.get("wrist_l_per_frame")
    pkl_wr = pi.get("wrist_r_per_frame")
    if pkl_wl is not None and pkl_wr is not None:
        is_grasp = (np.asarray(pkl_wl, dtype=bool)[:T] |
                    np.asarray(pkl_wr, dtype=bool)[:T])
    else:
        is_grasp = sigs["is_grasp"]
    # ★ Also count frames where grasp_hand_per_frame says L/R/both as grasped,
    # even if wrist_l/r_per_frame is False there. Without this, some brief
    # grasps (object in hand but flow-vetoed wrist) wouldn't trigger the
    # full_se3 segment lock and would fall back to rotation_only.
    _pkl_ghp = pi.get("grasp_hand_per_frame")
    if _pkl_ghp is not None and len(_pkl_ghp) >= T:
        is_grasp = is_grasp | np.array(
            [g in ("L", "R", "both") for g in _pkl_ghp[:T]], dtype=bool)
    # ★ Prefer pkl's grasp_hand_per_frame (refreshed by grasp_v2 L/R split) as
    # authoritative dominant hand. derive_hand_signals's proximity-based
    # dominant hand misses frames where the fingertip-to-mask distance is
    # above its threshold but pkl's wrist_l/r is True (memfof-confirmed grasp).
    # 'both' → resolved per-frame: use wrist_l vs wrist_r to break tie; default 'L'.
    pkl_ghp = pi.get("grasp_hand_per_frame")
    # ★ All-'both' fingertip-vote tiebreak. When all grasp frames are 'both'
    # (no explicit L or R from state estimation), the existing per-frame
    # wrist_l/r resolution always lands on 'L' (because grasp_both requires
    # both wrists True, hitting the wrist_l branch first). This silently
    # mis-assigns clips where the right hand is actually the dominant
    # holder (e.g. knife in `-Oj2xzQbo4s_236.4_240.6`). We resolve by
    # computing per-frame min fingertip-to-observation distance for each
    # hand on every 'both' frame; clip-wide majority wins. Fingertip-min
    # captures "which fingers are in contact with the object surface",
    # which is more discriminative than wrist-to-centroid (wrists can be
    # mis-located on tools/elongated meshes where the centroid sits far
    # from the held end).
    #
    # Trigger: only when n_L_raw == 0 AND n_R_raw == 0 AND n_both >= 5.
    # Other 'both' cases are handled by Fix A's clip-global dominant
    # override later in compose_display_pose.
    both_dom_override = None
    if pkl_ghp is not None and len(pkl_ghp) >= T:
        n_L_raw_lc = sum(1 for g in pkl_ghp[:T] if g == "L")
        n_R_raw_lc = sum(1 for g in pkl_ghp[:T] if g == "R")
        both_idx = [t for t in range(T) if pkl_ghp[t] == "both"]
        if (n_L_raw_lc == 0 and n_R_raw_lc == 0 and len(both_idx) >= 5):
            FINGERTIPS = (4, 8, 12, 16, 20)
            n_Lf = n_Rf = 0
            for t in both_idx:
                od = obj_data_per_frame[t]
                if od is None: continue
                mp = od.get("mask_packed"); ms = od.get("mask_shape")
                if mp is None or ms is None: continue
                try:
                    H, W = int(ms[0]), int(ms[1])
                    n_total = H * W
                    bits = np.unpackbits(np.frombuffer(mp, dtype=np.uint8))
                    if bits.size < n_total: continue
                    mask = bits[:n_total].reshape(H, W).astype(bool)
                except Exception:
                    continue
                if not mask.any(): continue
                pts = mask_to_pointcloud(mask, depth_seq[t], focal, cx, cy, step=2)
                if pts is None or len(pts) < 10: continue
                pts = np.asarray(pts, dtype=np.float64)
                j_list = joints_3d[t]; r_list = hand_is_right[t]
                L_j = R_j = None
                for h, ir in zip(j_list, r_list):
                    if h is None: continue
                    j_arr = np.asarray(h, dtype=np.float64)
                    if j_arr.shape != (21, 3): continue
                    if not np.all(np.isfinite(j_arr)): continue
                    if bool(ir): R_j = j_arr
                    else:        L_j = j_arr
                if L_j is None or R_j is None: continue
                L_tips, R_tips = L_j[list(FINGERTIPS)], R_j[list(FINGERTIPS)]
                d_L = float(min(np.linalg.norm(pts - tip, axis=1).min() for tip in L_tips))
                d_R = float(min(np.linalg.norm(pts - tip, axis=1).min() for tip in R_tips))
                if d_L < d_R: n_Lf += 1
                else: n_Rf += 1
            if n_Lf + n_Rf >= 5:
                both_dom_override = "L" if n_Lf > n_Rf else "R"
                print(f"[fingertip_tiebreak] all-'both' clip oid={oid}  "
                      f"both={len(both_idx)}  fingertip vote L:{n_Lf}/R:{n_Rf} "
                      f"→ {both_dom_override}")

    if pkl_ghp is not None and len(pkl_ghp) >= T:
        dom_override = list(sigs["dominant_hand"])
        for t in range(T):
            g = pkl_ghp[t]
            if g in ("L", "R"):
                dom_override[t] = g
            elif g == "both":
                if both_dom_override is not None:
                    dom_override[t] = both_dom_override
                elif pkl_wl is not None and pkl_wl[t]:
                    dom_override[t] = "L"
                elif pkl_wr is not None and pkl_wr[t]:
                    dom_override[t] = "R"
                else:
                    dom_override[t] = "L"
        sigs["dominant_hand"] = dom_override
    state_pf = compute_state_per_frame(is_moving, is_grasp)
    mask_area, is_occluded = compute_occlusion(obj_data_per_frame)

    # Motion score: prefer pkl's pc_motion_xy/z. Fall back to computing from
    # obs cloud bbox center delta if pkl is missing those fields (some
    # earlier pose_tracker runs didn't populate them — e.g., the plate clip
    # `-0RheyDV3a0_474.8_487.3 oid=1` has all-zero motion in pkl, which made
    # every frame look "static" and locked the whole clip to T_anchor).
    pc_xy = np.asarray(pi.get("pc_motion_xy_per_frame") or np.zeros(T),
                        dtype=np.float64)[:T]
    pc_z = np.asarray(pi.get("pc_motion_z_per_frame") or np.zeros(T),
                       dtype=np.float64)[:T]
    motion_score = np.sqrt(pc_xy ** 2 + pc_z ** 2)
    if motion_score.max() < 1e-4:
        # All-zero motion in pkl → compute ourselves from obs bbox center
        # delta. Same convention as EI's pc_motion_*: distance moved by the
        # object cloud's robust centroid between consecutive frames.
        centers = []
        for f_idx in range(T):
            od = obj_data_per_frame[f_idx]
            if od is None or "mask_packed" not in od:
                centers.append(None)
                continue
            H_m, W_m = int(od["mask_shape"][0]), int(od["mask_shape"][1])
            bits = np.unpackbits(np.frombuffer(od["mask_packed"], dtype=np.uint8))[: H_m * W_m]
            mask = bits.reshape(H_m, W_m).astype(bool)
            if not mask.any():
                centers.append(None)
                continue
            pts = mask_to_pointcloud(mask, depth_seq[f_idx], focal, cx, cy, step=4)
            if len(pts) < 10:
                centers.append(None)
                continue
            mm = np.median(pts, axis=0)
            d_med = np.linalg.norm(pts - mm, axis=1)
            mad_v = np.median(np.abs(d_med - np.median(d_med)))
            clean = pts[d_med <= np.median(d_med) + 3.0 * 1.4826 * mad_v]
            if len(clean) < 10: clean = pts
            centers.append((clean.min(0) + clean.max(0)) * 0.5)
        motion_score = np.zeros(T, dtype=np.float64)
        for f_idx in range(1, T):
            if centers[f_idx] is not None and centers[f_idx - 1] is not None:
                motion_score[f_idx] = float(np.linalg.norm(centers[f_idx] - centers[f_idx - 1]))
        print(f"[load_clip] pkl pc_motion missing/zero → recomputed from obs: "
              f"min={motion_score.min()*1000:.0f}mm max={motion_score.max()*1000:.0f}mm "
              f"median={np.median(motion_score)*1000:.0f}mm")

    # Palm angular motion: catches in-hand rotation that translation misses.
    palm_angular = compute_palm_angular(joints_3d, hand_is_right)

    # T_anchor: EI's pose at init_frame is the SAM3D-anchored reference.
    # Fall back to fp_pose_unflipped[init_frame] if EI pose missing.
    if ei_pose is not None and init_frame_idx < len(ei_pose):
        T_anchor = ei_pose[init_frame_idx].astype(np.float32).copy()
    else:
        T_anchor = fp_pose_unflipped[init_frame_idx].astype(np.float32).copy()
    n_static = int((state_pf == STATE_STATIC).sum())
    n_grasp = int((state_pf == STATE_GRASPED).sum())
    n_movng = int((state_pf == STATE_MOVING_NOT_GRASPED).sum())
    print(f"[load_clip] {testcase.name}  oid={oid}  T={T}  "
          f"STATIC={n_static}  GRASPED={n_grasp}  MOVING_NG={n_movng}  "
          f"init_frame={init_frame_idx}  anchor_t={anchor_t}  "
          f"motion_score range {motion_score.min()*1000:.0f}-{motion_score.max()*1000:.0f}mm")

    return ClipData(
        clip_id=testcase.name, T=T,
        focal=focal, cx=cx, cy=cy, H_img=H_img, W_img=W_img,
        rgb_seq=rgb_seq, depth_seq=depth_seq,
        fp_pose=fp_pose,
        fp_pose_unflipped=fp_pose_unflipped,
        ei_pose=ei_pose,
        is_moving_per_frame=is_moving,
        wrist_used_per_frame=wrist_used,
        close_per_frame=close,
        is_grasp_per_frame=is_grasp,
        dominant_hand_per_frame=sigs["dominant_hand"],
        grasp_hand_raw_per_frame=(list(pkl_ghp) if (pkl_ghp is not None
                                                     and len(pkl_ghp) >= T)
                                  else [None] * T),
        d_L_per_frame=sigs["d_L"],
        d_R_per_frame=sigs["d_R"],
        state_per_frame=state_pf,
        is_occluded_per_frame=is_occluded,
        mask_area_per_frame=mask_area,
        anchor_t=anchor_t,
        init_frame=init_frame_idx,
        motion_score_per_frame=motion_score.astype(np.float32),
        palm_angular_per_frame=palm_angular,
        T_anchor=T_anchor,
        mesh_xyz=xyz, mesh_rgb=rgb,
        fp_mesh_rgb=tint(rgb, COLOR_FP, 0.5),
        ei_mesh_rgb=tint(rgb, COLOR_EI, 0.5),
        mano_faces=mano_faces,
        gravity_up=gu,
        joints_3d=joints_3d, vertices_3d=vertices_3d, hand_is_right=hand_is_right,
        obj_data_per_frame=obj_data_per_frame,
        grid_origin=origin,
        oid=oid,
    )


# ────────────────── per-frame update ───────────────────────────────────────
# ───────────── compose FP++ display pose from toggle stack ────────────────
def use_ei_translation(fp_pose: np.ndarray, ei_pose: Optional[np.ndarray]) -> np.ndarray:
    """Replace each frame's translation with EI's t (yellow mesh's t).
    Keeps FP's rotation; only patches valid (finite) EI frames."""
    if ei_pose is None or len(ei_pose) == 0:
        return fp_pose
    out = fp_pose.copy()
    n = min(len(out), len(ei_pose))
    for t in range(n):
        et = ei_pose[t, :3, 3]
        if np.all(np.isfinite(et)):
            out[t, :3, 3] = et.astype(out.dtype)
    return out


def compose_display_pose(clip: ClipData, do_unflip: bool, do_hand_rigid: bool,
                          do_pca: bool,
                          do_obs_anchor: bool,
                          do_ei_t: bool,
                          do_state_lock: bool, do_smooth: bool,
                          do_obb_priority: bool = True,
                          do_full_se3_bind: bool = False) -> np.ndarray:
    """Pipeline (each toggleable):
       raw → [unflip] → [pca: non-grasp → PCA R] → [hand_rigid: grasp → hand R,
                                                     anchored to pre-grasp PCA R]
           → [obb_priority: grasp segment OBB / mask static override]
           → [obs_anchor: t from mask×depth] → [state_lock: static → T_anchor]
           → [smooth]

       PCA runs BEFORE hand_rigid so the latter's grasp-segment anchor R comes
       from the OBSERVATION-corrected pre-grasp orientation. ``obb_priority``
       runs AFTER hand_rigid so it can OVERRIDE Kabsch with OBB-stable PCA R
       when the obs cloud agrees the object isn't rotating with the hand.
    """
    p = clip.fp_pose_unflipped if do_unflip else clip.fp_pose
    if do_pca:
        p = pca_anchored_rotation(
            p, clip.obj_data_per_frame, clip.depth_seq,
            clip.focal, clip.cx, clip.cy,
            clip.mesh_xyz, clip.T_anchor,
            is_grasp=clip.is_grasp_per_frame)
    if do_full_se3_bind:
        # Per-segment FULL SE(3) rigid bind to explicit hand body frame.
        # Replaces (not augments) hand_rigid_grasp_lock when enabled.
        from egoinfinity.pipeline.pose_tracker.hand_body_frame import (
            build_hand_body_frame_per_frame)
        from egoinfinity.pipeline.pose_tracker.single_canonical_bind import (
            single_canonical_grasp_lock)

        # ★ Fix A: clip-level global dominant hand override using RAW pkl grasp
        # info (where 'both' is kept distinct from L/R). Loadclip resolves
        # 'both' → L by default, hiding the original ambiguity; we read the
        # raw to detect "overwhelmingly one-hand + some 'both'" clips and
        # force 'both' frames to the dominant hand instead of L default.
        dom_per_frame = list(clip.dominant_hand_per_frame)
        raw_gh = clip.grasp_hand_raw_per_frame
        is_g_for_a = np.asarray(clip.is_grasp_per_frame, dtype=bool)
        g_frames_a = np.where(is_g_for_a)[0]
        if len(g_frames_a) >= 5 and raw_gh is not None:
            n_L_raw = sum(1 for t in g_frames_a if raw_gh[t] == "L")
            n_R_raw = sum(1 for t in g_frames_a if raw_gh[t] == "R")
            n_both = sum(1 for t in g_frames_a if raw_gh[t] == "both")
            major = max(n_L_raw, n_R_raw)
            minor = min(n_L_raw, n_R_raw)
            # Fire when:
            #   - 'both' frames exist (something to resolve), n_both >= 5
            #     (the 'both' signal itself is statistically meaningful)
            # AND either:
            #   - minor == 0 AND major >= 1 (no evidence of second hand; the
            #     explicit hand is unambiguous — when 'both' dominates, even
            #     a single explicit-hand frame is enough to break the tie
            #     because the 'both' frames have BOTH hands present anyway,
            #     so picking the explicit-hand for those is safe), OR
            #   - minor < 20% of major AND major >= 5 AND n_both <= major
            #     (mild second-hand evidence; require larger major and cap
            #     n_both to avoid overriding genuine two-hand manipulation)
            unambiguous_single = (minor == 0 and major >= 1)
            mild_second_hand = (minor < 0.2 * major and major >= 5
                                and n_both <= major)
            if (n_both >= 5 and (unambiguous_single or mild_second_hand)):
                global_dom = "L" if n_L_raw > n_R_raw else "R"
                n_forced = 0
                for t in g_frames_a:
                    if raw_gh[t] == "both" and dom_per_frame[t] != global_dom:
                        dom_per_frame[t] = global_dom
                        n_forced += 1
                print(f"[fix_A dom_override] global_dom={global_dom}  "
                      f"L_raw={n_L_raw} R_raw={n_R_raw} both={n_both}  "
                      f"forced {n_forced} 'both' frames → {global_dom}")

        T_hand_seq = build_hand_body_frame_per_frame(
            clip.joints_3d, clip.hand_is_right, dom_per_frame)
        p, bind_diag = single_canonical_grasp_lock(
            p, T_hand_seq, clip.is_grasp_per_frame,
            mesh_pts=clip.mesh_xyz,
            hand_verts_per_frame=clip.vertices_3d,
            hand_is_right_per_frame=clip.hand_is_right,
            dominant_hand_per_frame=dom_per_frame,
            joints_per_frame=clip.joints_3d,
        )
        print(f"[full_se3_bind] {len(bind_diag['segments'])} seg(s), "
              f"{bind_diag['n_locked_frames']} frames locked, "
              f"skipped={len(bind_diag['skipped'])}")
    elif do_hand_rigid:
        p = hand_rigid_grasp_lock(
            p, clip.joints_3d, clip.hand_is_right,
            clip.is_grasp_per_frame, clip.dominant_hand_per_frame,
            motion_score=clip.motion_score_per_frame,
            palm_angular=clip.palm_angular_per_frame,
            init_frame=clip.init_frame,
            T_anchor=clip.T_anchor)
    if do_obb_priority:
        p = obb_priority_lock(p, clip)
    if do_obs_anchor:
        p = observation_anchored_translation(
            p, clip.mesh_xyz, clip.obj_data_per_frame, clip.depth_seq,
            clip.focal, clip.cx, clip.cy,
            is_grasp=clip.is_grasp_per_frame,
            is_occluded=clip.is_occluded_per_frame,
            joints_per_frame=clip.joints_3d,
            hand_is_right_per_frame=clip.hand_is_right,
            dominant_hand_per_frame=clip.dominant_hand_per_frame)
    if do_ei_t:
        # ★ Direct override: copy EI (yellow mesh) translation per-frame.
        # Runs LAST among t-modifiers so it wins over obs_anchor when both on.
        p = use_ei_translation(p, clip.ei_pose)
    if do_state_lock:
        p = state_aware_lock(p, clip.motion_score_per_frame,
                              init_frame=clip.init_frame, T_anchor=clip.T_anchor,
                              palm_angular=clip.palm_angular_per_frame,
                              is_grasp=clip.is_grasp_per_frame,
                              state_per_frame=clip.state_per_frame)
    if do_smooth:
        p = smooth_se3(p)
    return p


