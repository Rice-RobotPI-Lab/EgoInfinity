"""Z-axis-specific temporal smoothing for hands AND objects.

Background
==========
After ``refresh_pose_tracking`` and ``refresh_hand_scale``, the per-frame
hand wrist Z and object T_seq Z still jitter heavily — typically 50 mm
mean Δz/frame, with 500+ mm spikes when MoGe depth lands on a fingertip-
on-background pixel.  Existing pipeline smoothing windows (5-frame
median for hand cam_t, 11-frame SavGol for object SE3) don't kill this
because the MoGe noise is correlated across multiple frames.

This tool runs **post-hoc** on the favorite pkl with one-pass cost
(decode each frame's depth_png once, smooth both hand and object Z
sequences in place) and writes back:

  Hand :
    For each (track_id, frame) sample MoGe depth at the 5 RELIABLE_JOINT
    pixel locations.  MAD-filter outliers (|d - median| > mad×MAD).  If
    inliers ≥ min_inliers, anchor_z = mean(inliers); else trust=0.  Apply
    NaN-tolerant Gaussian σ=sigma_z to the anchor sequence.  For each
    frame, ``delta_z = smoothed - current_wrist_z`` then translate
    ``vertices_3d`` and ``joints_3d_pred`` by ``(0, 0, delta_z)``
    (preserves XY, hand pose, internal mesh structure).

  Object :
    Per-frame anchor_z = current ``T_seq[t][2, 3]`` (D-track output).
    Trust = n_points-after-SOR ≥ ``min_pts``.  Same Gaussian smooth.
    ``delta_z = smoothed - current_T_z`` then translate three things by
    ``(0, 0, delta_z)``:
      1. ``T_seq[t][2, 3]``        — mesh world-coord position
      2. ``sam3_obj_data[t][oid].pts[:, 2]`` — per-frame point cloud
      3. ``sam3_obj_data[t][oid].pose_t[2]`` and ``.obb_corners[:, 2]``
         — per-frame OBB pose

Why translate point cloud too: the 3D viewer renders both the smoothed
mesh AND the per-frame point cloud.  If only the mesh moves, mesh and
cloud become misaligned.  Translating the cloud by the same Δz keeps
them visually consistent.

Idempotency
===========
Provenance ``depth_smooth_refresh`` recorded per clip.  Re-running the
smoother on already-smoothed data produces a small additional smoothing
(approx square-root composition) — safe but unnecessary.  ``--dry-run``
reports before/after |Δz| stats without writing.

Usage
=====
  python -m egoinfinity.pipeline.post_tracking.depth_smooth --only=-8WOMg810tk_92.8_97.6 --dry-run
  python -m egoinfinity.pipeline.post_tracking.depth_smooth --only=-8WOMg810tk_92.8_97.6
  python -m egoinfinity.pipeline.post_tracking.depth_smooth --hands-only ...
  python -m egoinfinity.pipeline.post_tracking.depth_smooth --objects-only ...
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"

# Same set used by egoinfinity.pipeline.depth_align — wrist + 4 MCP bases.
# Avoid fingertips: their MoGe depth is noisy because the joint often
# overlaps the object surface or background by a few pixels.
RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]


# ── helpers ───────────────────────────────────────────────────────────
def _sample_depth_patch(depth_map: np.ndarray, u: float, v: float, half: int = 3) -> float:
    """Median MoGe depth in (2h+1) px patch around (u,v); NaN if invalid."""
    H, W = depth_map.shape
    xi, yi = int(round(u)), int(round(v))
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return float("nan")
    x0, x1 = max(0, xi - half), min(W, xi + half + 1)
    y0, y1 = max(0, yi - half), min(H, yi + half + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.01]
    return float(np.median(valid)) if valid.size else float("nan")


def _gaussian_smooth_weighted(values: np.ndarray, weights: np.ndarray,
                              sigma: float) -> np.ndarray:
    """1D Gaussian smoothing with weights.  Returns smoothed array.

    weighted_sum / weighted_norm form so weights=0 frames don't pull the
    smoother. NaN-tolerant: NaN values get replaced with 0 (paired with
    weight=0 from the caller), since numpy's ``NaN * 0 == NaN`` poisons the
    convolution otherwise — one NaN frame propagates NaN out to a radius
    of ``3σ`` and silently kills the smoothing for the rest of the clip.
    """
    n = len(values)
    if n == 0:
        return values
    # Sanitize: replace NaN values with 0 (weight is already 0 there from
    # the caller). Without this, NaN × 0 = NaN poisons the running sum.
    values = np.where(np.isfinite(values), values, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    half = max(int(np.ceil(3 * sigma)), 1)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    for k, w in enumerate(kernel):
        offset = k - half
        i0 = max(0, -offset); i1 = min(n, n - offset)
        # Skip empty / wrap-around windows. When kernel half > n (short clips
        # with wide σ), i1 can drop ≤ i0, and numpy's negative-index slicing
        # would silently wrap weights[a:b] to (n-|b|,) — broadcast mismatch.
        if i1 <= i0:
            continue
        out[i0:i1] += w * weights[i0 + offset:i1 + offset] * values[i0 + offset:i1 + offset]
        norm[i0:i1] += w * weights[i0 + offset:i1 + offset]
    smoothed = np.where(norm > 1e-9, out / np.maximum(norm, 1e-9), values)
    return smoothed


def _stabilize_hand_indices(fdata: list) -> dict:
    """Re-order per-frame hand entries so a consistent INDEX always tracks
    the same physical hand.

    Why: WiLoR's hand detection ORDER can swap between frames (the entries
    at js[0] and js[1] physically exchange positions). The is_right label
    follows whichever entry is at each index, so per-frame labels remain
    locally correct, but ACROSS frames the SAME INDEX alternates between
    physical hands. Downstream per-INDEX smoothing then operates on a
    mixed-identity series, and viser export's side-routing (R if
    is_right[hi] else L) yields trails that oscillate by 10+ cm in z as
    the routing alternately picks each physical hand.

    Fix: pick a canonical ordering (index 0 = LEFT hand, index 1 = RIGHT
    hand) and at every bi-hand frame, swap entries if the current ordering
    doesn't match. Single-hand frames are left alone (no ambiguity).

    Swaps all four parallel arrays in place on fdata:
      joints_3d_pred, vertices_3d, joints_2d_pred, hand_is_right
    """
    T = len(fdata)
    n_swaps = 0
    n_checked = 0
    for t in range(T):
        fr = fdata[t]
        js = fr.get("joints_3d_pred") or []
        ir = fr.get("hand_is_right") or []
        if len(js) < 2 or len(ir) < 2 or js[0] is None or js[1] is None:
            continue
        n_checked += 1
        # Canonical order: index 0 must be LEFT (is_right=False), index 1
        # must be RIGHT (is_right=True). If the order is reversed, swap.
        if bool(ir[0]) and not bool(ir[1]):
            # entry 0 is R, entry 1 is L → swap to canonical
            for key in ("joints_3d_pred", "vertices_3d", "joints_2d_pred",
                        "hand_is_right"):
                seq = fr.get(key)
                if seq is None or len(seq) < 2:
                    continue
                seq[0], seq[1] = seq[1], seq[0]
            n_swaps += 1
        # If both are same side (both R or both L), leave alone — WiLoR
        # gave inconsistent labels but we have no signal to choose.
    return {"n_swaps": n_swaps, "n_checked": n_checked}


def _temporal_outlier_reject(series: np.ndarray, trust: np.ndarray,
                              window: int = 5, mad_factor: float = 5.0) -> np.ndarray:
    """Time-domain MAD outlier rejection. Returns updated trust array with
    spike frames set to 0 (causing the Gaussian smoother to interpolate over
    them instead of being pulled by the spike).

    A frame ``t`` is flagged when its deviation from the local-window median
    exceeds ``mad_factor × local_MAD``. Window covers ``[t-window, t+window]``.

    Use case: a hand wrist z that's stable at 1.78m but jumps to 1.99m for
    a SINGLE frame is a depth-tracker outlier — neighbors are good, the
    spike is bad. Setting its trust to 0 lets the smoother fit through the
    neighbors and produce a smoothed value near 1.78m at the spike frame.
    """
    n = len(series)
    out_trust = trust.astype(np.float64).copy()
    for t in range(n):
        if not np.isfinite(series[t]) or trust[t] <= 0:
            continue
        lo = max(0, t - window); hi = min(n, t + window + 1)
        nbr = series[lo:hi]
        nbr_trust = trust[lo:hi]
        valid = np.isfinite(nbr) & (nbr_trust > 0)
        # Need ≥3 neighbors for a reliable local stat
        if valid.sum() < 3:
            continue
        v = nbr[valid]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) + 1e-6
        if abs(float(series[t]) - med) > mad_factor * mad:
            out_trust[t] = 0.0
    return out_trust


# ── hand pass ─────────────────────────────────────────────────────────
def _hand_anchor_per_frame(fdata, depth_maps, hi: int,
                           dp_focal: float, cx: float, cy: float,
                           mad_factor: float, min_inliers: int):
    """Returns (anchor_z[T], trust[T], current_wrist_z[T]) for one hand
    index across all frames.  hi is the per-frame list index, NOT
    track_id — assumes hand index is stable per frame (typical when
    only 0/1 hands)."""
    T = len(fdata)
    anchor = np.full(T, np.nan, dtype=np.float64)
    trust = np.zeros(T, dtype=np.float64)
    current = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        fd = fdata[t]
        j2_list = fd.get("joints_2d_pred") or []
        j3_list = fd.get("joints_3d_pred") or []
        if hi >= len(j2_list) or hi >= len(j3_list):
            continue
        j2 = np.asarray(j2_list[hi], dtype=np.float32) if j2_list[hi] is not None else None
        j3 = np.asarray(j3_list[hi], dtype=np.float32) if j3_list[hi] is not None else None
        if j2 is None or j3 is None or j2.shape != (21, 2) or j3.shape != (21, 3):
            continue
        current[t] = float(j3[0, 2])
        depth_map = depth_maps[t]
        if depth_map is None:
            continue
        # Sample MoGe depth at reliable joints
        depths = []
        for jid in RELIABLE_JOINT_IDS:
            d = _sample_depth_patch(depth_map, j2[jid, 0], j2[jid, 1])
            if np.isfinite(d) and d > 0:
                depths.append(d)
        if len(depths) < min_inliers:
            continue
        depths = np.array(depths, dtype=np.float64)
        med = float(np.median(depths))
        mad = float(np.median(np.abs(depths - med))) + 1e-6
        inliers = depths[np.abs(depths - med) <= mad_factor * mad]
        if len(inliers) < min_inliers:
            continue
        anchor[t] = float(np.mean(inliers))
        trust[t] = 1.0
    return anchor, trust, current


def _smooth_hand_z(fdata, depth_maps, dp_focal: float, cx: float, cy: float,
                   sigma_z: float, mad_factor: float, min_inliers: int,
                   max_delta_z_m: float, dry_run: bool,
                   collect_traces: bool = False) -> dict:
    """Smooth hand wrist Z and translate vertices+joints by Δz.  Returns
    stats dict with before/after |Δz| samples.  If collect_traces, the
    dict also contains ``traces`` = list of (name, before_z, after_z)."""
    # Detect hand count (max over frames)
    n_hands_max = 0
    for fd in fdata:
        n_hands_max = max(n_hands_max, len(fd.get("joints_3d_pred") or []))
    if n_hands_max == 0:
        return {"hands_smoothed": 0, "n_hand_tracks": 0,
                "before_dz_mm_mean": 0.0, "after_dz_mm_mean": 0.0,
                "before_dz_mm_max":  0.0, "after_dz_mm_max":  0.0,
                "traces": []}

    before_all, after_all = [], []
    traces = []
    n_smoothed = 0
    for hi in range(n_hands_max):
        anchor, trust, current = _hand_anchor_per_frame(
            fdata, depth_maps, hi, dp_focal, cx, cy,
            mad_factor=mad_factor, min_inliers=min_inliers)
        valid = np.isfinite(anchor) & np.isfinite(current)
        if valid.sum() < 4:
            continue
        # ★ Time-domain outlier rejection on the anchor series. Kills single-
        # frame spikes (e.g. wrist depth jumps 200mm for 1 frame, neighbors
        # fine) by setting trust=0 → Gaussian smoothes through the gap.
        trust = _temporal_outlier_reject(anchor, trust, window=5, mad_factor=5.0)
        # Smooth anchor
        anchor_filled = np.where(np.isfinite(anchor), anchor, current)
        smoothed = _gaussian_smooth_weighted(anchor_filled, trust, sigma_z)
        # delta = smoothed - current (capped)
        delta = np.where(np.isfinite(current), smoothed - current, 0.0)
        delta = np.clip(delta, -max_delta_z_m, max_delta_z_m)
        # Stats: before = |Δz| of current sequence; after = |Δz| of (current + delta)
        before_dz = np.abs(np.diff(current[np.isfinite(current)]))
        new_z = current + delta
        after_dz = np.abs(np.diff(new_z[np.isfinite(new_z)]))
        before_all.extend(before_dz.tolist())
        after_all.extend(after_dz.tolist())
        if collect_traces:
            traces.append((f"hand{hi}_wrist_z", current.copy(), new_z.copy()))
        # Apply to vertices_3d + joints_3d_pred
        if not dry_run:
            for t in range(len(fdata)):
                if not np.isfinite(delta[t]) or abs(delta[t]) < 1e-6:
                    continue
                v_list = fdata[t].get("vertices_3d") or []
                j3_list = fdata[t].get("joints_3d_pred") or []
                if hi >= len(v_list) or hi >= len(j3_list):
                    continue
                v = v_list[hi]; j = j3_list[hi]
                if v is not None and np.asarray(v).shape == (778, 3):
                    new_v = np.asarray(v, dtype=np.float32).copy()
                    new_v[:, 2] += float(delta[t])
                    fdata[t]["vertices_3d"][hi] = new_v
                if j is not None and np.asarray(j).shape == (21, 3):
                    new_j = np.asarray(j, dtype=np.float32).copy()
                    new_j[:, 2] += float(delta[t])
                    fdata[t]["joints_3d_pred"][hi] = new_j
        n_smoothed += 1

    b = np.array(before_all) if before_all else np.array([0.0])
    a = np.array(after_all) if after_all else np.array([0.0])
    return {
        "hands_smoothed": n_smoothed,
        "n_hand_tracks": n_hands_max,
        "before_dz_mm_mean": float(b.mean() * 1000),
        "after_dz_mm_mean":  float(a.mean() * 1000),
        "before_dz_mm_max":  float(b.max() * 1000),
        "after_dz_mm_max":   float(a.max() * 1000),
        "traces": traces,
    }


def _smooth_hand_verts_mean_z(fdata: list, sigma_z: float, max_delta_z_m: float,
                               dry_run: bool, collect_traces: bool = False) -> dict:
    """Pass 2: smooth the per-hand verts-mean-z time series, applying the
    delta to *only the vertices* (not the joints).

    Why: After pass 1, joints[0].z is smooth (anchor-based smoothing). But
    verts.mean_z still wiggles ~20mm/frame because MANO pose noise makes
    the cloud of 778 vertices shift relative to the wrist independently of
    the global translation. Pass 1 shifts joints + verts uniformly by the
    SAME delta, so the relative drift (verts.mean - wrist) is preserved.

    This pass smooths verts.mean_z directly as its own scalar series, and
    re-shifts the mesh per-frame so the verts.mean trajectory becomes smooth.
    Joints are NOT touched here (skeleton trajectory comes from pass 1).

    Result: mesh and skeleton each have their own smooth z-trajectory; the
    relative offset (verts.mean - joints[0]) becomes a slow-varying delta
    instead of a 20mm/frame jitter, so visually the mesh sticks to the
    skeleton without internal-pose-driven z wobble.
    """
    if not fdata:
        return {"hands_v_smoothed": 0,
                "before_dz_mm_mean": 0.0, "after_dz_mm_mean": 0.0,
                "before_dz_mm_max":  0.0, "after_dz_mm_max":  0.0,
                "traces": []}
    T = len(fdata)
    n_hands_max = 0
    for fd in fdata:
        n_hands_max = max(n_hands_max, len(fd.get("vertices_3d") or []))
    if n_hands_max == 0:
        return {"hands_v_smoothed": 0,
                "before_dz_mm_mean": 0.0, "after_dz_mm_mean": 0.0,
                "before_dz_mm_max":  0.0, "after_dz_mm_max":  0.0,
                "traces": []}

    before_all, after_all, traces = [], [], []
    n_smoothed = 0
    for hi in range(n_hands_max):
        vmz = np.full(T, np.nan, dtype=np.float64)
        trust = np.zeros(T, dtype=np.float64)
        for t in range(T):
            v_list = fdata[t].get("vertices_3d") or []
            if hi >= len(v_list) or v_list[hi] is None:
                continue
            v = np.asarray(v_list[hi])
            if v.shape != (778, 3):
                continue
            vmz[t] = float(v[:, 2].mean())
            trust[t] = 1.0
        valid = np.isfinite(vmz)
        if valid.sum() < 4:
            continue
        # Temporal outlier rejection (same as pass 1)
        trust = _temporal_outlier_reject(vmz, trust, window=5, mad_factor=5.0)
        vmz_filled = np.where(np.isfinite(vmz), vmz, 0.0)
        smoothed = _gaussian_smooth_weighted(vmz_filled, trust, sigma_z)
        delta = np.where(np.isfinite(vmz), smoothed - vmz, 0.0)
        delta = np.clip(delta, -max_delta_z_m, max_delta_z_m)
        # Stats
        before_dz = np.abs(np.diff(vmz[valid]))
        new_vmz = vmz + delta
        after_dz = np.abs(np.diff(new_vmz[np.isfinite(new_vmz)]))
        before_all.extend(before_dz.tolist())
        after_all.extend(after_dz.tolist())
        if collect_traces:
            traces.append((f"hand{hi}_verts_mean_z", vmz.copy(), new_vmz.copy()))
        # Apply: shift each vertex's z by delta[t] (only verts, NOT joints)
        if not dry_run:
            for t in range(T):
                if not np.isfinite(delta[t]) or abs(delta[t]) < 1e-6:
                    continue
                v_list = fdata[t].get("vertices_3d") or []
                if hi >= len(v_list) or v_list[hi] is None:
                    continue
                v = np.asarray(v_list[hi])
                if v.shape != (778, 3):
                    continue
                new_v = v.astype(np.float32).copy()
                new_v[:, 2] += float(delta[t])
                fdata[t]["vertices_3d"][hi] = new_v
        n_smoothed += 1

    b = np.array(before_all) if before_all else np.array([0.0])
    a = np.array(after_all) if after_all else np.array([0.0])
    return {
        "hands_v_smoothed": n_smoothed,
        "before_dz_mm_mean": float(b.mean() * 1000),
        "after_dz_mm_mean":  float(a.mean() * 1000),
        "before_dz_mm_max":  float(b.max() * 1000),
        "after_dz_mm_max":   float(a.max() * 1000),
        "traces": traces,
    }


# ── object pass ───────────────────────────────────────────────────────
def _smooth_object_z(data: dict, fdata: list, sigma_z: float,
                     min_mask_px: int, max_delta_z_m: float,
                     dry_run: bool, collect_traces: bool = False) -> dict:
    """Smooth per-object T_seq Z + per-frame OBB Z.

    Source signals:
      anchor[t] = current T_seq[t][2, 3] (D-track output)
      trust[t]  = mask pixel count ≥ min_mask_px in sam3_obj_data[t][oid].
                  pkl pre-computes mask_packed (packed bits) but NOT pts —
                  point cloud is derived viewer-side from mask × depth.
                  Mask presence is the right "is this object observed in
                  this frame" signal.
    """
    pti = data.get("pose_track_info") or {}
    if not pti:
        return {"objects_smoothed": 0,
                "before_dz_mm_mean": 0.0, "after_dz_mm_mean": 0.0,
                "before_dz_mm_max":  0.0, "after_dz_mm_max":  0.0,
                "traces": []}

    T = len(fdata)
    before_all, after_all = [], []
    traces = []
    n_smoothed = 0

    for oid_str, info in list(pti.items()):
        if not isinstance(info, dict):
            continue
        T_seq = info.get("T_seq")
        if T_seq is None:
            continue
        T_seq = np.asarray(T_seq, dtype=np.float32).copy()
        if T_seq.ndim != 3 or T_seq.shape[0] != T or T_seq.shape[1:] != (4, 4):
            continue
        oid = int(oid_str) if not isinstance(oid_str, int) else oid_str

        anchor = T_seq[:, 2, 3].astype(np.float64)        # current Z
        trust = np.zeros(T, dtype=np.float64)
        for t in range(T):
            sd = (fdata[t].get("sam3_obj_data") or {})
            od = sd.get(oid) or sd.get(int(oid))
            if not isinstance(od, dict):
                continue
            mp = od.get("mask_packed")
            ms = od.get("mask_shape")
            if mp is None or ms is None:
                continue
            # Cheap mask area check: count set bits in packed bytes
            n_set = int(np.unpackbits(np.asarray(mp, dtype=np.uint8))[:int(ms[0]) * int(ms[1])].sum())
            if n_set >= min_mask_px:
                trust[t] = 1.0
        # Need at least 4 trusted frames to smooth meaningfully
        if trust.sum() < 4:
            continue

        smoothed = _gaussian_smooth_weighted(anchor, trust, sigma_z)
        delta = np.clip(smoothed - anchor, -max_delta_z_m, max_delta_z_m)

        before_dz = np.abs(np.diff(anchor))
        after_dz = np.abs(np.diff(anchor + delta))
        before_all.extend(before_dz.tolist())
        after_all.extend(after_dz.tolist())
        if collect_traces:
            traces.append((f"obj{oid}_z", anchor.copy(), (anchor + delta).copy()))

        if not dry_run:
            # 1) T_seq Z
            T_seq[:, 2, 3] = (anchor + delta).astype(np.float32)
            info["T_seq"] = T_seq
            # 2) Per-frame point cloud + OBB pose Z
            for t in range(T):
                dz = float(delta[t])
                if abs(dz) < 1e-6:
                    continue
                sd = fdata[t].get("sam3_obj_data") or {}
                od = sd.get(oid) or sd.get(int(oid))
                if not isinstance(od, dict):
                    continue
                pts = od.get("pts")
                if pts is not None:
                    pts = np.asarray(pts, dtype=np.float32).copy()
                    pts[:, 2] += dz
                    od["pts"] = pts
                pose_t = od.get("pose_t")
                if pose_t is not None:
                    pt2 = np.asarray(pose_t, dtype=np.float32).copy()
                    pt2[2] += dz
                    od["pose_t"] = pt2
                obb = od.get("obb_corners")
                if obb is not None:
                    obb2 = np.asarray(obb, dtype=np.float32).copy()
                    obb2[:, 2] += dz
                    od["obb_corners"] = obb2
        n_smoothed += 1

    b = np.array(before_all) if before_all else np.array([0.0])
    a = np.array(after_all) if after_all else np.array([0.0])
    return {
        "objects_smoothed": n_smoothed,
        "before_dz_mm_mean": float(b.mean() * 1000),
        "after_dz_mm_mean":  float(a.mean() * 1000),
        "before_dz_mm_max":  float(b.max() * 1000),
        "after_dz_mm_max":   float(a.max() * 1000),
        "traces": traces,
    }


# ── audit plot ─────────────────────────────────────────────────────────
def _save_audit_plot(out_dir: Path, clip_name: str, sigma_z: float, stats: dict):
    """Save a multi-row PNG with before/after Z traces for visual review.
    Each row = one trace (hand_wrist or obj). Top-of-row title shows the
    Δz mean/max for that signal. Matplotlib import is local so we do not
    pull it unless --audit-plot is used."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [audit] matplotlib unavailable ({e}) — skip plot")
        return

    traces = []
    traces.extend((stats.get("hand") or {}).get("traces") or [])
    traces.extend((stats.get("object") or {}).get("traces") or [])
    if not traces:
        return

    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, before, after) in zip(axes, traces):
        ax.plot(before, color="#aaaaaa", lw=1.0, label="before")
        ax.plot(after,  color="#cc4040", lw=1.2, label="after")
        valid_b = before[np.isfinite(before)]
        valid_a = after[np.isfinite(after)]
        db = float(np.abs(np.diff(valid_b)).mean() * 1000) if len(valid_b) > 1 else 0
        da = float(np.abs(np.diff(valid_a)).mean() * 1000) if len(valid_a) > 1 else 0
        ax.set_title(f"{name}   |Δz| mean: {db:.1f} → {da:.1f} mm",
                     fontsize=9, loc="left")
        ax.grid(alpha=0.2)
        ax.set_ylabel("z (m)", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"{clip_name}    σ_z = {sigma_z}", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── orchestration ─────────────────────────────────────────────────────
def _process_clip(fav_dir: Path, sigma_z: float, mad_factor: float,
                  min_inliers: int, min_mask_px: int, max_delta_z_m: float,
                  do_hands: bool, do_objects: bool,
                  dry_run: bool,
                  skip_if_done: bool = False,
                  force: bool = False,
                  audit_plot_dir: Path = None) -> tuple[str, str, dict]:
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    from scripts.pipeline_utils import decode_depth_png

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fdata = data.get("frame_data") or []
    if not fdata:
        return "skip", "empty frame_data", {}

    # ── Skip-if-done: any prior smoothing pass present in provenance ──
    # Re-running compounds Gaussian σ (σ_eff ≈ √(σ_prev² + σ_new²)) which
    # silently drifts results across clips, so the safe default when
    # asked to "skip already smoothed" is to skip unconditionally.
    if skip_if_done and not force:
        prov = data.get("depth_smooth_refresh") or {}
        hist = prov.get("history") or []
        if hist:
            sigmas = [h.get("sigma_z") for h in hist if "sigma_z" in h]
            return "skip", f"already smoothed (σ={sigmas})", {}

    dp_focal = float(data.get("dp_focal", 0.0))
    if dp_focal <= 0:
        return "fail", "no dp_focal", {}
    cx = float(data.get("cx", 0.0))
    cy = float(data.get("cy", 0.0))

    # Decode all depth_pngs once (only needed for hand pass)
    depth_maps = []
    if do_hands:
        for fd in fdata:
            png = fd.get("depth_png")
            depth_maps.append(decode_depth_png(png) if png is not None else None)
    else:
        depth_maps = [None] * len(fdata)

    collect = audit_plot_dir is not None
    stats = {"action": []}
    if do_hands:
        # ★ Pre-pass: stabilize per-frame hand entry order so a consistent
        # INDEX always tracks the same physical hand. WiLoR's detection
        # order can swap entries (js[0] and js[1] physically exchange)
        # between frames; the is_right label follows the new order, so the
        # labels are locally correct but per-INDEX series mix physical
        # hands. Without this fix, downstream per-INDEX smoothing operates
        # on mixed identities and viser's side-routing oscillates wildly.
        # Mutates fdata in place if not dry_run; in dry_run we still do
        # the swap because subsequent smoothing pass reads the (swapped)
        # entries — but we revert at the end (handled below).
        if not dry_run:
            si_stats = _stabilize_hand_indices(fdata)
        else:
            # snapshot, swap, run downstream stats, restore
            import copy
            _backup = [{k: copy.deepcopy(fr.get(k))
                        for k in ("joints_3d_pred", "vertices_3d",
                                   "joints_2d_pred", "hand_is_right")}
                       for fr in fdata]
            si_stats = _stabilize_hand_indices(fdata)
        stats["stabilize_idx"] = si_stats
        h_stats = _smooth_hand_z(fdata, depth_maps, dp_focal, cx, cy,
                                 sigma_z=sigma_z, mad_factor=mad_factor,
                                 min_inliers=min_inliers,
                                 max_delta_z_m=max_delta_z_m,
                                 dry_run=dry_run, collect_traces=collect)
        stats["hand"] = h_stats
        if h_stats["hands_smoothed"] > 0:
            stats["action"].append("hand")
        # Pass 2: smooth the verts.mean z trajectory independently of the
        # wrist anchor — kills the MANO-pose-noise-driven mesh z wiggle.
        hv_stats = _smooth_hand_verts_mean_z(
            fdata, sigma_z=sigma_z, max_delta_z_m=max_delta_z_m,
            dry_run=dry_run, collect_traces=collect)
        stats["hand_v"] = hv_stats
        if hv_stats["hands_v_smoothed"] > 0:
            stats["action"].append("hand_v")
    if do_objects:
        o_stats = _smooth_object_z(data, fdata, sigma_z=sigma_z,
                                   min_mask_px=min_mask_px,
                                   max_delta_z_m=max_delta_z_m,
                                   dry_run=dry_run, collect_traces=collect)
        stats["object"] = o_stats
        if o_stats["objects_smoothed"] > 0:
            stats["action"].append("object")

    if not stats["action"]:
        # restore dry-run backup if applicable
        if dry_run and do_hands and "_backup" in dir():
            for fr, b in zip(fdata, _backup):
                for k, v in b.items():
                    if v is not None: fr[k] = v
        return "skip", "nothing to smooth", stats

    if audit_plot_dir is not None:
        _save_audit_plot(audit_plot_dir, fav_dir.name, sigma_z, stats)

    if not dry_run:
        prov = data.get("depth_smooth_refresh") or {}
        prov_history = prov.get("history") or []
        prov_history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sigma_z": sigma_z, "mad_factor": mad_factor,
            "min_inliers": min_inliers, "min_mask_px": min_mask_px,
            "max_delta_z_m": max_delta_z_m,
            "did_hands": do_hands, "did_objects": do_objects,
            **{f"hand_{k}": v for k, v in stats.get("hand", {}).items()},
            **{f"hand_v_{k}": v for k, v in stats.get("hand_v", {}).items()},
            **{f"stabilize_{k}": v for k, v in stats.get("stabilize_idx", {}).items()},
            **{f"obj_{k}": v for k, v in stats.get("object", {}).items()},
        })
        data["depth_smooth_refresh"] = {"history": prov_history}
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return "ok", "+".join(stats["action"]), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    ap.add_argument("--sigma-z", type=float, default=5.0,
                    help="Gaussian σ in frames (default 5 ≈ 0.33s @ 15fps)")
    ap.add_argument("--mad-factor", type=float, default=2.0,
                    help="hand-joint MAD outlier reject (default 2.0)")
    ap.add_argument("--min-inliers", type=int, default=3,
                    help="min reliable hand joints with valid MoGe depth (default 3 of 5)")
    ap.add_argument("--min-mask-px", type=int, default=200,
                    help="min SAM2 mask pixel count for object trust (default 200)")
    ap.add_argument("--max-delta-z-m", type=float, default=0.30,
                    help="cap on per-frame Δz (default 0.30 m, kills catastrophic outliers)")
    ap.add_argument("--hands-only", action="store_true")
    ap.add_argument("--objects-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips that already have depth_smooth_refresh history "
                         "(avoid Gaussian σ compounding — σ_eff=√(σ_old²+σ_new²))")
    ap.add_argument("--force", action="store_true",
                    help="Ignore prior provenance — apply smoothing regardless. "
                         "Use this to redo clips affected by older buggy versions; "
                         "be aware that this will compound σ if the previous run "
                         "actually applied (it didn't, for the pre-fix NaN-poisoning "
                         "bug — those runs were no-ops on clips with any NaN frame).")
    ap.add_argument("--audit-plot", default=None, type=str,
                    help="Directory to save per-clip before/after z-trace PNGs "
                         "(visual confirmation that σ is right)")
    args = ap.parse_args()
    audit_dir = Path(args.audit_plot).resolve() if args.audit_plot else None

    if args.hands_only and args.objects_only:
        print("error: --hands-only and --objects-only are mutually exclusive")
        sys.exit(1)
    do_hands = not args.objects_only
    do_objects = not args.hands_only

    allow = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    favs = sorted(p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_depth_smooth: {len(favs)} clips, dry_run={args.dry_run}, "
          f"σ_z={args.sigma_z}, hands={do_hands}, objects={do_objects}")
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        status, action, stats = _process_clip(
            fav, sigma_z=args.sigma_z, mad_factor=args.mad_factor,
            min_inliers=args.min_inliers, min_mask_px=args.min_mask_px,
            max_delta_z_m=args.max_delta_z_m,
            do_hands=do_hands, do_objects=do_objects,
            dry_run=args.dry_run,
            skip_if_done=args.skip_if_done,
            force=args.force,
            audit_plot_dir=audit_dir,
        )
        dt = time.time() - t_clip
        if status == "ok":
            n_ok += 1
            h = stats.get("hand", {})
            hv = stats.get("hand_v", {})
            o = stats.get("object", {})
            si = stats.get("stabilize_idx", {})
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}+ {fav.name:<50}"
                  f"  swaps={si.get('n_swaps',0):>3}/{si.get('n_checked',0):>3}"
                  f"  hand |Δz|: {h.get('before_dz_mm_mean',0):.0f}→{h.get('after_dz_mm_mean',0):.0f}mm"
                  f" (max {h.get('before_dz_mm_max',0):.0f}→{h.get('after_dz_mm_max',0):.0f})"
                  f"  hand_v: {hv.get('before_dz_mm_mean',0):.0f}→{hv.get('after_dz_mm_mean',0):.0f}mm"
                  f"  obj: {o.get('before_dz_mm_mean',0):.0f}→{o.get('after_dz_mm_mean',0):.0f}mm"
                  f"  ({dt:.1f}s)")
        elif status == "skip":
            n_skip += 1
            print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({action})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({action})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")


if __name__ == "__main__":
    main()
