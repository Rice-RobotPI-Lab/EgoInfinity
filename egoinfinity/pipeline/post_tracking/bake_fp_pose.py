"""Bake FoundationPose++ composed-display-pose orientation into EI pkls.

Reads ``FoundationPose-plus-plus/testcase/<clip>{,__oid<N>}/pose.npy`` (raw
FP++ output), runs the full composed-pose pipeline from
``egoinfinity.pipeline.post_tracking.fp_compose.compose_display_pose`` (unflip
-> PCA-anchored R -> hand-rigid grasp lock -> state lock -> SE3 smooth),
and writes the resulting rotation **only** back into
``pose_track_info[oid]['T_seq'][:, :3, :3]``.

The translation column ``T_seq[:, :3, 3]`` is left UNCHANGED — it's already
the smoothed EI position from ``egoinfinity.pipeline.post_tracking.depth_smooth``.
So the HF demo sees: orientation from FP++ refined pipeline, position from
EI smoothed.

Provenance: ``data['fp_pose_bake'] = {'history': [...]}``  with timestamp,
toggle settings, and per-oid R-change stats.

Usage::

    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose --dry-run                     # all favorites
    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose --only=-DCmm-dTi0o_188.7_194.8 --dry-run
    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose                               # write back
    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose --skip-if-done                # idempotent
    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose --force                       # re-bake
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"
DEFAULT_FP_ROOT = Path(
    os.environ.get("EGOINFINITY_FP_ROOT")
    or (REPO.parent / "FoundationPose-plus-plus" / "testcase"))


def _ang_R(R1: np.ndarray, R2: np.ndarray) -> float:
    Rd = R1 @ R2.T
    cos_a = float(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _pts_pca_center(pts) -> Optional[np.ndarray]:
    """Robust 3D centroid replicating viser's OBB wireframe pipeline.

    Steps (match the viser OBB wireframe centroid pipeline):
      1. MAD-based outlier filter on per-point distance to median
         (threshold = median + 3 * 1.4826 * MAD)
      2. Mean of inliers (= PCA centroid)

    Returns None on degenerate input (too few points). Used by Fix A's
    static-frame T override so the mesh aligns with the wireframe viser
    actually renders, instead of with sam3_obj_data.pose_t (which is the
    raw (min+max)/2 bbox center, sensitive to depth-noise outliers and
    can diverge from PCA center by 5-30cm).
    """
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 10:
        return None
    med = np.median(arr, axis=0)
    d = np.linalg.norm(arr - med, axis=1)
    d_med = float(np.median(d))
    mad = float(np.median(np.abs(d - d_med)))
    thresh = d_med + 3.0 * 1.4826 * mad
    inliers = arr[d <= thresh]
    if len(inliers) < 5:
        return None
    return inliers.mean(axis=0)


_DEPTH_CACHE: dict = {}

def _decode_depth_cached(depth_png_bytes, key):
    """Decode depth_png once per frame_id, cache result (cleared per clip)."""
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    from scripts.pipeline_utils import decode_depth_png
    depth = decode_depth_png(depth_png_bytes)
    _DEPTH_CACHE[key] = depth
    return depth


def _compute_pts_inline(od, fd_frame, dp_focal, cx, cy, frame_key):
    """Compute observed point cloud from mask_packed + depth_png on-the-fly.

    Necessary because bake_fp_pose does NOT rehydrate the pkl (would bloat
    saved file by 100s of MB / clip from cached depth_map), so the standard
    sam3_obj_data['pts'] field is absent.
    """
    if not isinstance(od, dict): return None
    mp = od.get('mask_packed')
    ms = od.get('mask_shape')
    if mp is None or ms is None: return None
    depth_png = fd_frame.get('depth_png')
    depth_map = fd_frame.get('depth_map')
    if depth_map is not None:
        depth = np.asarray(depth_map)
    elif depth_png is not None:
        depth = _decode_depth_cached(depth_png, frame_key)
    else:
        return None
    try:
        H, W = int(ms[0]), int(ms[1])
        n_total = H * W
        bits = np.unpackbits(np.frombuffer(mp, dtype=np.uint8))
        if bits.size < n_total: return None
        mask = bits[:n_total].reshape(H, W).astype(bool)
    except Exception:
        return None
    if not mask.any(): return None
    if depth.shape[:2] != (H, W): return None
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    valid = (z > 1e-3) & np.isfinite(z)
    if valid.sum() < 30: return None
    xs, ys, z = xs[valid], ys[valid], z[valid]
    x = (xs - cx) * z / dp_focal
    y = (ys - cy) * z / dp_focal
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _read_oid_marker(testcase: Path) -> Optional[int]:
    """Read the explicit oid marker written by from_egoinfinity (oid.txt).
    Returns None if the marker is absent (legacy testcase folder)."""
    f = testcase / "oid.txt"
    if not f.is_file():
        return None
    try:
        return int(f.read_text().strip())
    except Exception:
        return None


def _fp_testcase_for_oid(fp_root: Path, clip_id: str, oid: int) -> Optional[Path]:
    """Locate the FP++ testcase dir for a given (clip, oid).

    Lookup order:
      1. ``<clip_id>__oid<N>/`` if it has pose.npy AND its oid.txt matches
         (oid.txt missing on this sub is treated as a match for backwards
         compat with explicit-suffix legacy folders).
      2. ``<clip_id>/`` (primary) if it has pose.npy AND its oid.txt matches.
         If oid.txt is missing (legacy primary from before 2026-05),
         fall back to the historical convention "primary == oid 0".

    from_egoinfinity.py writes oid.txt into every testcase folder it
    builds; bake_fp_pose only needs the legacy fallback for testcase
    folders generated before that marker was added.
    """
    # 1) suffixed sibling
    sub = fp_root / f"{clip_id}__oid{oid}"
    if (sub / "pose.npy").exists():
        marker = _read_oid_marker(sub)
        if marker is None or marker == oid:
            return sub

    # 2) primary
    primary = fp_root / clip_id
    if (primary / "pose.npy").exists():
        marker = _read_oid_marker(primary)
        if marker is not None:
            return primary if marker == oid else None
        # legacy fallback: assume primary == oid 0 only when no marker
        return primary if oid == 0 else None

    return None


def _process_clip(fav_dir: Path, fp_root: Path,
                   dry_run: bool, skip_if_done: bool, force: bool,
                   toggles: dict) -> tuple[str, str, dict]:
    """Process one favorite clip. Returns (status, action, stats)."""
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if skip_if_done and not force:
        prov = data.get("fp_pose_bake") or {}
        if prov.get("history"):
            return "skip", "already baked", {}

    pti = data.get("pose_track_info") or {}
    fd = data.get("frame_data") or []
    if not pti or not fd:
        return "skip", "no pti or fdata", {}
    T_pkl = len(fd)
    sam3_mesh_info = data.get("sam3_mesh_info") or {}
    # Camera intrinsics for inline pts computation (Fix A)
    dp_focal = float(data.get("dp_focal", 0) or 0)
    cx_img = float(data.get("cx", 0) or 0)
    cy_img = float(data.get("cy", 0) or 0)
    # Clear depth decode cache for new clip
    _DEPTH_CACHE.clear()

    # Lazy-import — fp_compose pulls cv2, scipy, etc.
    from egoinfinity.pipeline.post_tracking.fp_compose import load_clip, compose_display_pose


    def _r_anchor_from_sam3d(meta: dict) -> Optional[np.ndarray]:
        """SAM3D canonical_rotation_quat (wxyz, pytorch3d row-form) → world R.
        Matches the convention in
        ``egoinfinity/pipeline/pose_tracker._track_phase_d``."""
        if not isinstance(meta, dict):
            return None
        q = meta.get("canonical_rotation_quat")
        if q is None:
            return None
        try:
            from scipy.spatial.transform import Rotation as _Rot
            q_arr = np.asarray(q, dtype=np.float64)
            R_q = _Rot.from_quat([q_arr[1], q_arr[2], q_arr[3], q_arr[0]]).as_matrix()
            F = np.diag([-1.0, -1.0, 1.0])
            return F @ R_q.T
        except Exception:
            return None

    n_baked = 0
    n_missing = 0
    n_skipped = 0
    r_change_means = []
    r_change_maxes = []
    per_oid_log = []

    for oid_key in list(pti.keys()):
        # Normalize oid to int for FP++ dir matching
        try:
            oid = int(oid_key)
        except (ValueError, TypeError):
            n_skipped += 1
            continue
        info = pti.get(oid_key)
        if not isinstance(info, dict):
            n_skipped += 1
            continue
        T_seq = info.get("T_seq")
        if T_seq is None:
            n_skipped += 1
            continue
        T_seq = np.asarray(T_seq, dtype=np.float32).copy()
        if T_seq.ndim != 3 or T_seq.shape[1:] != (4, 4):
            n_skipped += 1
            continue

        # ── is_static_global short-circuit ──
        # For globally-static objects, skip the entire FP++ + compose chain
        # and lock R to SAM3D's initial canonical_rotation_quat for ALL
        # frames.  Reasoning: a globally-static object has no per-frame
        # rotation to learn, and SAM3D's one-shot estimate is the most
        # reliable signal we have.  Going through pca_anchored_rotation +
        # state_aware_lock + smooth_se3 introduces obs-cloud PCA noise
        # (90° symmetry flips) that the 45° snap threshold can't catch.
        if bool(info.get("is_static_global", False)):
            R_anchor_s3d = _r_anchor_from_sam3d(
                sam3_mesh_info.get(oid) or sam3_mesh_info.get(str(oid)) or {})
            if R_anchor_s3d is not None:
                old_R_static = T_seq[:, :3, :3].copy()
                new_R_static = np.tile(R_anchor_s3d.astype(np.float32),
                                        (T_seq.shape[0], 1, 1))
                angs_static = np.array([_ang_R(new_R_static[t], old_R_static[t])
                                         for t in range(T_seq.shape[0])])
                r_change_means.append(float(angs_static.mean()))
                r_change_maxes.append(float(angs_static.max()))
                per_oid_log.append(
                    f"oid{oid}=Δ{angs_static.mean():.0f}°[static_global]")
                # ★ Lock translation to median of per-frame PCA-MAD pts
                # centroid (= the exact signal the viser OBB wireframe uses).
                # Fallbacks: median(pose_t) if pts unavailable,
                # then median(input T_seq.t) if both missing.
                #
                # Why not pose_t directly: sam3_obj_data.pose_t is bbox
                # center (min+max)/2 which is sensitive to depth-noise
                # outliers (e.g. strainer has obb_center.z=1.87 but
                # PCA-MAD center.z=2.04, where the actual visible point
                # cloud lives). Across 106 clips: 92% of static stretches
                # have |pose_t - pca_center| > 10mm, 54% > 50mm, 10% > 200mm.
                pca_list = []
                pose_t_list = []
                T_seq_n = T_seq.shape[0]
                fd_data = data.get("frame_data") or []
                for t in range(min(T_seq_n, len(fd_data))):
                    sd = fd_data[t].get("sam3_obj_data") or {}
                    od_ = sd.get(oid)
                    if od_ is None:
                        od_ = sd.get(int(oid)) if not isinstance(oid, int) else None
                    if not isinstance(od_, dict):
                        continue
                    # Compute pts inline (bake does NOT rehydrate pkl, so the
                    # standard od_['pts'] field is absent).
                    pts = _compute_pts_inline(
                        od_, fd_data[t], dp_focal, cx_img, cy_img,
                        frame_key=(fav_dir.name, t))
                    c = _pts_pca_center(pts)
                    if c is not None and np.all(np.isfinite(c)):
                        pca_list.append(c.astype(np.float32))
                    pt = od_.get("pose_t")
                    if pt is not None:
                        pt = np.asarray(pt, dtype=np.float32)
                        if pt.shape == (3,) and np.all(np.isfinite(pt)):
                            pose_t_list.append(pt)
                if pca_list:
                    t_static = np.median(np.stack(pca_list), axis=0)
                elif pose_t_list:
                    t_static = np.median(np.stack(pose_t_list), axis=0)
                else:
                    # Last-resort fallback (median of input T_seq.t)
                    t_seq_old = T_seq[:, :3, 3]
                    t_finite = np.isfinite(t_seq_old).all(axis=1)
                    if t_finite.any():
                        t_static = np.median(t_seq_old[t_finite], axis=0)
                    else:
                        t_static = np.zeros(3, dtype=np.float32)
                if not dry_run:
                    T_seq_new = T_seq.copy()
                    T_seq_new[:, :3, :3] = new_R_static
                    T_seq_new[:, :3, 3] = t_static.astype(T_seq_new.dtype)
                    info["T_seq"] = T_seq_new
                n_baked += 1
                continue
            # If canonical_rotation_quat is missing, fall through to normal
            # bake (rare; sam3d should always emit it)

        # Find FP++ testcase dir for this oid
        fp_dir = _fp_testcase_for_oid(fp_root, fav_dir.name, oid)
        if fp_dir is None:
            n_missing += 1
            per_oid_log.append(f"oid{oid}=miss")
            continue

        try:
            clip = load_clip(fp_dir, pkl_path, oid_override=oid)
        except Exception as e:
            n_missing += 1
            per_oid_log.append(f"oid{oid}=load-fail")
            continue

        T_compose = min(clip.T, T_pkl, T_seq.shape[0])

        # Run composed pose pipeline
        try:
            composed = compose_display_pose(clip, **toggles)  # (T, 4, 4)
        except Exception as e:
            n_missing += 1
            per_oid_log.append(f"oid{oid}=compose-fail:{type(e).__name__}")
            continue

        # ★ R always replaced. T replaced ONLY in full_se3 mode.
        # In rotation_only mode (legacy), composed T is discarded and the
        # EI smoothed translation from depth_align stays. In full_se3 mode,
        # composed T = T_hand @ T_canonical for grasp frames + T_anchor for
        # state-locked STATIC frames, and that's the lock the user wants.
        old_R = T_seq[:T_compose, :3, :3].copy()
        new_R = composed[:T_compose, :3, :3].astype(np.float32)
        old_t = T_seq[:T_compose, :3, 3].copy()
        new_t = composed[:T_compose, :3, 3].astype(np.float32)
        full_se3_mode = bool(toggles.get("do_full_se3_bind", False))

        # Sanity: replace NaN frames with old values (compose can yield NaN
        # if an upstream stage failed for that frame).
        for t in range(T_compose):
            if not np.all(np.isfinite(new_R[t])):
                new_R[t] = old_R[t]
            if full_se3_mode and not np.all(np.isfinite(new_t[t])):
                new_t[t] = old_t[t]

        # R change stats: angle between old and new per-frame
        angs = np.array([_ang_R(new_R[t], old_R[t]) for t in range(T_compose)])
        r_change_means.append(float(angs.mean()))
        r_change_maxes.append(float(angs.max()))
        if full_se3_mode:
            t_shift = np.linalg.norm(new_t - old_t, axis=1) * 1000  # mm
            per_oid_log.append(
                f"oid{oid}=ΔR{angs.mean():.0f}°/Δt{t_shift.mean():.0f}mm")
        else:
            per_oid_log.append(f"oid{oid}=Δ{angs.mean():.0f}°")

        if not dry_run:
            T_seq_new = T_seq.copy()
            T_seq_new[:T_compose, :3, :3] = new_R
            # State-based T overrides operate on the FULL T_seq range, not
            # just T_compose. FP++ pose.npy may be shorter than the clip
            # (e.g. 51 frames vs 59) but state_per_frame + sam3_obj_data
            # cover all frames. Without this, late static frames stay at
            # their stale position_first values.
            T_full = T_seq.shape[0]
            if full_se3_mode:
                # ★ T override policy (post-2026-05-28):
                #   - GRASP frames: use composed T = T_hand @ T_canonical
                #     (object follows the hand during grasp).
                #   - STATIC frames: median of sam3_obj_data[oid].pose_t over
                #     the strict-static stretch (= OBB center per frame, the
                #     same signal viser renders as the wireframe bbox). FP++
                #     raw T frequently drifts to the occluding hand's depth
                #     (sauce bottle: 22 cm z-error to LEFT-hand depth) and
                #     position_first's eroded-mask bbox also misses the true
                #     bottle z by ~10cm. pose_t (sam3_obj_data per-frame OBB
                #     center) is the one signal that stably tracks the object
                #     during static stretches.
                #   - Other (moving / no_state) frames: keep input old_t
                #     (= position_first's output, post depth-align).
                is_grasp_arr = np.asarray(clip.is_grasp_per_frame,
                                           dtype=bool)[:T_compose]
                # final_t starts as full-length copy of input T_seq.t,
                # then we override the first T_compose frames per the
                # grasp_t_mask. Frames beyond T_compose keep their input
                # T_seq.t (= position_first output, possibly stale).
                final_t = T_seq[:, :3, 3].copy().astype(np.float32)
                grasp_t_mask = is_grasp_arr.reshape(-1, 1)
                final_t[:T_compose] = np.where(
                    grasp_t_mask, new_t, old_t).astype(np.float32)

                # Override static-stretch frames with median pose_t. Run
                # over the FULL clip range, not just T_compose — pose_t
                # is available for every frame regardless of pose.npy length.
                state_pf_pkl = info.get('state_per_frame')
                if state_pf_pkl is not None and len(state_pf_pkl) >= T_full:
                    state_arr_full = np.asarray(state_pf_pkl)[:T_full]
                    if state_arr_full.dtype.kind in ('U', 'S', 'O'):
                        strict_static = np.array(
                            [str(s) == 'static' for s in state_arr_full],
                            dtype=bool)
                    else:
                        strict_static = (state_arr_full == 0)  # STATE_STATIC=0
                    # Find contiguous static stretches over full range
                    n_static_overridden = 0
                    i = 0
                    while i < T_full:
                        if strict_static[i]:
                            j = i
                            while j + 1 < T_full and strict_static[j + 1]:
                                j += 1
                            # Stretch [i, j]: prefer PCA-MAD pts centroid
                            # (matches viser wireframe). Fall back to pose_t
                            # if pts missing/degenerate.
                            pca_centers = []
                            pose_t_fallback = []
                            for t in range(i, j + 1):
                                if t >= len(fd):
                                    continue
                                ff = fd[t]
                                sd = ff.get('sam3_obj_data') or {}
                                od_ = sd.get(oid)
                                if od_ is None:
                                    od_ = sd.get(int(oid)) if not isinstance(oid, int) else None
                                if not isinstance(od_, dict):
                                    continue
                                # Compute pts inline (no rehydrate)
                                pts_field = _compute_pts_inline(
                                    od_, ff, dp_focal, cx_img, cy_img,
                                    frame_key=(fav_dir.name, t))
                                c = _pts_pca_center(pts_field)
                                if c is not None and np.all(np.isfinite(c)):
                                    pca_centers.append(c.astype(np.float32))
                                pt = od_.get('pose_t')
                                if pt is not None:
                                    pt = np.asarray(pt, dtype=np.float32)
                                    if pt.shape == (3,) and np.all(np.isfinite(pt)):
                                        pose_t_fallback.append(pt)
                            if pca_centers:
                                stretch_med = np.median(
                                    np.stack(pca_centers), axis=0).astype(np.float32)
                                final_t[i:j + 1] = stretch_med
                                n_static_overridden += (j - i + 1)
                            elif pose_t_fallback:
                                stretch_med = np.median(
                                    np.stack(pose_t_fallback), axis=0).astype(np.float32)
                                final_t[i:j + 1] = stretch_med
                                n_static_overridden += (j - i + 1)
                            i = j + 1
                        else:
                            i += 1
                    if n_static_overridden > 0:
                        per_oid_log[-1] += f"|static_obb={n_static_overridden}"

                T_seq_new[:T_full, :3, 3] = final_t
            info["T_seq"] = T_seq_new
        n_baked += 1

    if n_baked == 0:
        reason = "no oids baked"
        if per_oid_log:
            reason += " [" + "+".join(per_oid_log) + "]"
        return "skip", reason, {"missing": n_missing, "skipped": n_skipped}

    if not dry_run:
        prov = data.get("fp_pose_bake") or {}
        prov_history = prov.get("history") or []
        prov_history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fp_root": str(fp_root),
            "toggles": toggles,
            "n_oid_baked": n_baked,
            "n_oid_missing": n_missing,
            "r_change_deg_mean": float(np.mean(r_change_means)) if r_change_means else 0.0,
            "r_change_deg_max": float(np.max(r_change_maxes)) if r_change_maxes else 0.0,
        })
        data["fp_pose_bake"] = {"history": prov_history}
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return "ok", "+".join(per_oid_log), {
        "n_baked": n_baked, "n_missing": n_missing, "n_skipped": n_skipped,
        "r_change_mean": float(np.mean(r_change_means)) if r_change_means else 0.0,
        "r_change_max": float(np.max(r_change_maxes)) if r_change_maxes else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp_root", type=Path, default=DEFAULT_FP_ROOT,
                    help="FoundationPose-plus-plus/testcase root")
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--force", action="store_true")
    # Toggle defaults: matches viser_multiclip "all on" except do_obs_anchor=False
    # (we want to KEEP the EI translation, so obs_anchor — which overwrites t —
    # stays off; do_ei_t (which we keep on) does the same thing via use_ei_translation
    # but is harmless since we discard composed t anyway).
    ap.add_argument("--no-unflip",       action="store_true")
    ap.add_argument("--no-pca",          action="store_true")
    ap.add_argument("--no-hand-rigid",   action="store_true")
    ap.add_argument("--no-state-lock",   action="store_true")
    ap.add_argument("--no-smooth",       action="store_true")
    ap.add_argument("--hand-rigid-mode",
                    choices=["rotation_only", "full_se3"],
                    default="rotation_only",
                    help="rotation_only (legacy): hand_rigid_grasp_lock only "
                         "rewrites R, translation stays from observation. "
                         "full_se3: per-segment chordal-mean R + median t in "
                         "explicit hand body frame, propagated rigidly so the "
                         "object has zero relative motion to the hand within "
                         "a grasp segment.")
    args = ap.parse_args()

    # Default toggles tuned for "R-only bake": skip stages whose output is
    # the composed translation (we discard composed t and keep the
    # existing EI smoothed t already in T_seq). Saves ~30-50% per-clip
    # bake time since obs_anchor is the most expensive stage.
    full_se3 = (args.hand_rigid_mode == "full_se3")
    toggles = dict(
        do_unflip          = not args.no_unflip,
        do_pca             = not args.no_pca,
        do_hand_rigid      = (not args.no_hand_rigid) and not full_se3,
        do_full_se3_bind   = full_se3,
        do_obs_anchor      = False,
        do_ei_t            = False,
        do_state_lock      = not args.no_state_lock,
        # In full_se3 mode, obb_priority would override grasp-segment R from
        # observation PCA (defeats the rigid lock), and smooth_se3 would
        # SavGol-perturb the locked T_seq away from the canonical. Boundary
        # SLERP ramp inside full_se3_bind already handles segment edges.
        do_obb_priority    = not full_se3,
        do_smooth          = (not args.no_smooth) and not full_se3,
    )

    if not args.fp_root.is_dir():
        print(f"error: --fp_root {args.fp_root} not a dir", file=sys.stderr)
        sys.exit(1)

    allow = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    favs = sorted(p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"bake_fp_pose: {len(favs)} clips  dry_run={args.dry_run}  toggles={toggles}")
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        try:
            status, action, stats = _process_clip(
                fav, args.fp_root,
                dry_run=args.dry_run, skip_if_done=args.skip_if_done,
                force=args.force, toggles=toggles)
        except Exception as e:
            status, action, stats = "fail", f"{type(e).__name__}:{e}", {}
        dt = time.time() - t_clip
        if status == "ok":
            n_ok += 1
            n_baked = stats.get("n_baked", 0)
            n_missing = stats.get("n_missing", 0)
            r_mean = stats.get("r_change_mean", 0)
            r_max = stats.get("r_change_max", 0)
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}+ {fav.name:<48}"
                  f"  baked={n_baked:>2} miss={n_missing:>2}  "
                  f"ΔR mean={r_mean:>4.0f}° max={r_max:>4.0f}°  "
                  f"({dt:.1f}s)  [{action}]")
        elif status == "skip":
            n_skip += 1
            print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({action})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({action})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")


if __name__ == "__main__":
    main()
