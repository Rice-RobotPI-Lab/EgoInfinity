"""
Refresh the *hand* portion of `pipeline_result.pkl.gz` in place without
re-running depth (MoGe-2), gravity, SAM2/SAM3/SAM3D, or 6DoF pose tracking.

Use after editing the post-WiLoR pipeline (smoothing windows, infiller
short-gap threshold, biomech, etc.) when the cached depth/objects from
a previous full run are still trustworthy.

Per clip:
  1. Load pkl, decode depth_png -> float32 depth maps (kept in memory only;
     pkl's depth_png bytes are preserved unchanged on write-back).
  2. Re-run Phase B: HandDetector + HandReconstructor on
     favorites/<id>/frames/*.jpg.
  3. Re-run Phase B post-processing: per-track handedness majority vote,
     overlapping-bbox dedup, track-level filtering (short/outlier).
  4. Re-run Phase C (hand half):
       align_hand_to_depth_multiscale  (using cached depth, dp_focal, cx, cy)
       MotionInfiller.fill_missing_frames
       apply_biomech_constraints
       smooth_translations(window=5)   median per-axis per-track
       reject_spikes
       smooth_joints_savgol(window=7, polyorder=2)
       smooth_mano_params(window=7, polyorder=2)  -> smoothed verts via MANO fwd
  5. Build per-frame joints_3d_pred / joints_2d_pred / vertices_3d /
     hand_is_right / hand_meta and overwrite those five fields in
     data['frame_data'][i].  Everything else (img_rgb / flow_rgb /
     depth_png / obj_data / sam3_obj_data / pose_track_info / etc.) is
     left exactly as loaded.
  6. Re-pickle.

Skipped: depth stabilization (cached depth is the post-stabilize result;
re-running would need raw MoGe-2 output, which isn't kept).  Optical
flow masks are unused outside of stabilization, so we also skip
build_optical_flow_masks.

Usage:
    python -m tools.rerun.refresh_hands                       # all 105
    python -m tools.rerun.refresh_hands --only ID1,ID2        # subset
    python -m tools.rerun.refresh_hands --dry-run             # process but don't write
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_BROWSER_DIR = REPO_ROOT  # favorites cache root
FAVORITES_DIR = Path(os.environ.get(
    "ACTION100M_CACHE", str(_BROWSER_DIR / "cache"))) / "favorites"


# ── helpers ──────────────────────────────────────────────────────────

def _decode_depth_png(png_bytes):
    """Mirror scripts/pipeline_utils.decode_depth_png locally."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    d_mm = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if d_mm is None:
        return None
    return d_mm.astype(np.float32) / 1000.0


def _bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _load_frames_bgr(fav_dir: Path, n_expected: int):
    frames_dir = fav_dir / "frames"
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if len(paths) < n_expected:
        raise RuntimeError(
            f"frames mismatch: have {len(paths)} files, pkl has {n_expected}")
    frames_bgr = []
    for p in paths[:n_expected]:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"failed to decode {p}")
        frames_bgr.append(img)
    return frames_bgr


# ── per-clip refresh ─────────────────────────────────────────────────

def _refresh_one(
    fav_dir: Path,
    detector,
    reconstructor,
    infiller_ckpt: str,
    dry_run: bool = False,
):
    from egoinfinity.pipeline.depth_align import align_hand_to_depth_multiscale
    from egoinfinity.pipeline.depth_stabilize import (
        smooth_translations, smooth_joints_savgol, reject_spikes)
    from egoinfinity.pipeline.motion_infiller import MotionInfiller
    from egoinfinity.pipeline.biomech_constraints import apply_biomech_constraints
    from egoinfinity.pipeline.mano_smoothing import smooth_mano_params

    pkl = fav_dir / "pipeline_result.pkl.gz"
    if not pkl.is_file():
        return "skip", "no pkl"
    try:
        with gzip.open(pkl, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return "fail", f"pkl load: {e}"

    fdata = data.get("frame_data") or []
    if not fdata:
        return "skip", "empty frame_data"
    total = len(fdata)

    dp_focal = float(data.get("dp_focal") or 0.0)
    cx = float(data.get("cx") or 0.0)
    cy = float(data.get("cy") or 0.0)
    if dp_focal <= 0 or cx <= 0 or cy <= 0:
        return "fail", f"bad intrinsics dp_focal={dp_focal} cx={cx} cy={cy}"

    # depth: keep depth_png on disk, decode locally for compute only
    depth_maps = []
    for fd in fdata:
        png = fd.get("depth_png")
        if png is None:
            return "fail", "frame missing depth_png"
        d = _decode_depth_png(png)
        if d is None:
            return "fail", "depth decode failed"
        depth_maps.append(d)

    # load frames
    try:
        frames_bgr = _load_frames_bgr(fav_dir, total)
    except Exception as e:
        return "fail", str(e)

    # MANO is a sub-module of WiLoR. Phase C++ / smoothing later moves it to
    # CPU float32; force it back to cuda half before this clip's WiLoR forward
    # so we're robust to whatever the previous clip left behind.
    reconstructor.model.mano.cuda().half()

    # ── Phase B: detector + reconstructor ────────────────────────
    detector.reset_tracking()
    hand_results_per_frame = []
    for img_bgr in frames_bgr:
        dets = detector.detect(img_bgr)
        hands = reconstructor.reconstruct(img_bgr, dets, focal_length=dp_focal)
        hand_results_per_frame.append(hands)

    mano_model = reconstructor.model.mano

    # ── Phase B post-process: handedness majority vote ──────────
    track_hand_votes = defaultdict(list)
    for frame_hands in hand_results_per_frame:
        for h in frame_hands:
            track_hand_votes[h.track_id].append(h.is_right)
    for tid, votes in track_hand_votes.items():
        majority = Counter(votes).most_common(1)[0][0]
        for frame_hands in hand_results_per_frame:
            for h in frame_hands:
                if h.track_id == tid and h.is_right != majority:
                    h.is_right = majority
                    h.joints_3d[:, 0] *= -1
                    h.joints_3d_rel[:, 0] *= -1
                    h.vertices[:, 0] *= -1
                    h.cam_t[0] *= -1

    # ── Phase B post-process: dedup overlapping bboxes ──────────
    for i in range(total):
        hands = hand_results_per_frame[i]
        if len(hands) <= 1:
            continue
        keep, removed = [], set()
        for ai in range(len(hands)):
            if ai in removed:
                continue
            best = ai
            for bi in range(ai + 1, len(hands)):
                if bi in removed:
                    continue
                if _bbox_iou(hands[ai].bbox, hands[bi].bbox) > 0.3:
                    if hands[bi].confidence > hands[best].confidence:
                        removed.add(best); best = bi
                    else:
                        removed.add(bi)
            keep.append(best)
        if removed:
            hand_results_per_frame[i] = [hands[k] for k in sorted(set(keep))]

    # ── Phase B post-process: track-level filtering ─────────────
    track_stats = defaultdict(lambda: {
        'frames': [], 'wrists': [], 'bbox_areas': [], 'depths': [], 'is_right': None})
    for i, frame_hands in enumerate(hand_results_per_frame):
        for h in frame_hands:
            ts = track_stats[h.track_id]
            ts['frames'].append(i)
            ts['wrists'].append(h.cam_t.copy())
            ts['depths'].append(h.cam_t[2])
            bw, bh = h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1]
            ts['bbox_areas'].append(bw * bh)
            ts['is_right'] = h.is_right

    dominant = {}
    for side in [True, False]:
        side_tracks = [(tid, ts) for tid, ts in track_stats.items()
                       if ts['is_right'] == side]
        if side_tracks:
            dominant[side] = max(side_tracks,
                                 key=lambda x: len(x[1]['frames']))[0]

    tracks_to_remove = set()
    for tid, ts in track_stats.items():
        side = ts['is_right']
        n_frames_t = len(ts['frames'])
        if n_frames_t < 5:
            tracks_to_remove.add(tid)
            continue
        if side in dominant and dominant[side] != tid:
            dom_ts = track_stats[dominant[side]]
            dom_wrist = np.median(dom_ts['wrists'], axis=0)
            my_wrist = np.median(ts['wrists'], axis=0)
            dist = np.linalg.norm(my_wrist - dom_wrist)
            if dist > 0.5 and n_frames_t < len(dom_ts['frames']) * 0.3:
                tracks_to_remove.add(tid)
                continue
        if side in dominant and dominant[side] != tid:
            dom_ts = track_stats[dominant[side]]
            dom_depths = np.array(dom_ts['depths'])
            dom_areas = np.array(dom_ts['bbox_areas'])
            dom_valid = dom_depths > 0.1
            my_depths = np.array(ts['depths'])
            my_areas = np.array(ts['bbox_areas'])
            my_valid = my_depths > 0.1
            if dom_valid.sum() > 0 and my_valid.sum() > 0:
                dom_size = np.median(dom_areas[dom_valid] / dom_depths[dom_valid] ** 2)
                my_size = np.median(my_areas[my_valid] / my_depths[my_valid] ** 2)
                ratio = my_size / max(dom_size, 1e-6)
                if (ratio > 3.0 or ratio < 0.33) and n_frames_t < len(dom_ts['frames']) * 0.3:
                    tracks_to_remove.add(tid)
    if tracks_to_remove:
        for i in range(total):
            hand_results_per_frame[i] = [
                h for h in hand_results_per_frame[i]
                if h.track_id not in tracks_to_remove]

    # ── Phase C (hand half): depth align ─────────────────────────
    # depth_maps here are the cached *already-stabilized* depths.
    aligned_cam_ts = []
    for i in range(total):
        hands = hand_results_per_frame[i]
        frame_ts = []
        for h in hands:
            t_a = align_hand_to_depth_multiscale(
                h.joints_3d_rel, h.joints_2d, depth_maps[i],
                dp_focal, cx, cy, h.cam_t, h.scaled_focal)
            frame_ts.append(t_a)
        aligned_cam_ts.append(frame_ts)

    # ── Phase C+: motion infiller ────────────────────────────────
    # MANO model needs CPU float for infiller's MANO forward path.
    mano_model.cpu().float()
    if os.path.isfile(infiller_ckpt):
        import torch as _torch
        infiller = MotionInfiller(infiller_ckpt, device='cuda',
                                  mano_model=mano_model)
        hand_results_per_frame = infiller.fill_missing_frames(
            hand_results_per_frame, total, aligned_cam_ts=aligned_cam_ts)
        del infiller
        _torch.cuda.empty_cache()

    # Pad aligned_cam_ts to match post-infill hand counts (None for filled slots).
    for i in range(total):
        while len(aligned_cam_ts[i]) < len(hand_results_per_frame[i]):
            aligned_cam_ts[i].append(None)

    # ── Phase C++: biomech constraints ───────────────────────────
    apply_biomech_constraints(hand_results_per_frame, mano_model=mano_model)

    # ── Phase C: smooth cam_t, joints, MANO params ───────────────
    track_cam_ts = defaultdict(dict)
    for i in range(total):
        for hi, h in enumerate(hand_results_per_frame[i]):
            ali = aligned_cam_ts[i][hi] if hi < len(aligned_cam_ts[i]) else None
            if ali is not None:
                track_cam_ts[h.track_id][i] = ali
            elif h.confidence == 0.0:
                track_cam_ts[h.track_id][i] = h.cam_t

    smoothed_map = {}
    for tid, fd_map in track_cam_ts.items():
        sf = sorted(fd_map.keys())
        seq = [fd_map[f] for f in sf]
        sm = smooth_translations(seq, window=5)
        for fi, f in enumerate(sf):
            smoothed_map[(f, tid)] = sm[fi]

    track_joints = defaultdict(dict)
    for i in range(total):
        for h in hand_results_per_frame[i]:
            ct = smoothed_map.get((i, h.track_id))
            if ct is not None:
                track_joints[h.track_id][i] = h.joints_3d_rel + ct

    reject_spikes(track_joints, threshold_factor=3.0)
    smoothed_joints = smooth_joints_savgol(track_joints, window=7, polyorder=2)
    smoothed_joints_map = {}
    for tid, fd_map in smoothed_joints.items():
        for f, j3d in fd_map.items():
            smoothed_joints_map[(f, tid)] = j3d

    smoothed_verts_map = smooth_mano_params(
        hand_results_per_frame, smoothed_map, mano_model, total,
        smoothed_joints_map=smoothed_joints_map,
        window=7, polyorder=2)

    # ── Build per-frame output fields (match main pipeline shape) ──
    n_detected = n_infilled = 0
    for i in range(total):
        hands = hand_results_per_frame[i]
        j3d_pred, j2d_pred, verts_3d = [], [], []
        hand_is_right, hand_meta = [], []
        for hi_idx, h in enumerate(hands):
            ct_aligned = aligned_cam_ts[i][hi_idx] if hi_idx < len(aligned_cam_ts[i]) else h.cam_t
            key = (i, h.track_id)
            ct_smooth = smoothed_map.get(key, ct_aligned)
            if ct_smooth is None:
                ct_smooth = ct_aligned

            j3d_raw = h.joints_3d_rel + ct_smooth
            j3d = smoothed_joints_map.get(key, j3d_raw)
            j3d_pred.append(j3d)

            j2d = getattr(h, 'joints_2d', None)
            if j2d is None:
                j2d = np.full((21, 2), np.nan, dtype=np.float32)
            else:
                j2d = np.asarray(j2d, dtype=np.float32)
            j2d_pred.append(j2d)

            v_smooth = smoothed_verts_map.get(key)
            if v_smooth is not None:
                verts_3d.append(v_smooth)
            else:
                verts_rel = h.vertices - h.cam_t
                delta = j3d[0] - j3d_raw[0]
                verts_3d.append(verts_rel + ct_smooth + delta)

            hand_is_right.append(h.is_right)
            source = 'detected' if h.confidence > 0 else 'infilled'
            if source == 'detected':
                n_detected += 1
            else:
                n_infilled += 1
            hand_meta.append({
                'source': source,
                'confidence': float(h.confidence),
                'track_id': int(h.track_id),
                'cam_t_raw': np.asarray(h.cam_t, dtype=np.float32).copy(),
                'cam_t_smooth': (np.asarray(ct_smooth).copy()
                                 if ct_smooth is not None else None),
            })

        # Overwrite in place, leave every other key untouched.
        fdata[i]['joints_3d_pred'] = j3d_pred
        fdata[i]['joints_2d_pred'] = j2d_pred
        fdata[i]['vertices_3d'] = verts_3d
        fdata[i]['hand_is_right'] = hand_is_right
        fdata[i]['hand_meta'] = hand_meta

    # Update mano_faces if missing (some older caches)
    if data.get('mano_faces') is None:
        data['mano_faces'] = reconstructor.model.mano.faces.astype(np.int32).copy()

    if dry_run:
        return "ok", (f"dry-run | detected={n_detected} infilled={n_infilled}")

    # ── Write back (preserve depth_png + bg_template_png + everything else) ──
    tmp = pkl.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl)

    return "ok", (f"detected={n_detected} infilled={n_infilled}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="comma-separated favorite ids (e.g. --only=ID1,ID2)")
    p.add_argument("--dry-run", action="store_true",
                   help="run pipeline but don't write back")
    p.add_argument("--infiller-checkpoint", default=None,
                   help="override path to infiller.pt")
    args = p.parse_args()

    favs = sorted(d for d in FAVORITES_DIR.iterdir()
                  if d.is_dir() and (d / "pipeline_result.pkl.gz").is_file()
                  and (d / "frames").is_dir()
                  and any((d / "frames").glob("frame_*.jpg")))
    if args.only:
        only = set(s.strip() for s in args.only.split(",") if s.strip())
        favs = [d for d in favs if d.name in only]

    if not favs:
        print("[refresh_hands] no favorites match", flush=True)
        return

    print(f"[refresh_hands] loading WiLoR / hand detector...", flush=True)
    # PyTorch 2.6+ default weights_only=True breaks legacy WiLoR/ultralytics ckpts.
    import torch
    torch.serialization.add_safe_globals([])
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    from egoinfinity.pipeline.hand_detector import HandDetector
    from egoinfinity.pipeline.hand_reconstructor import HandReconstructor
    from egoinfinity.pipeline.config import INFILLER_CHECKPOINT
    detector = HandDetector()
    reconstructor = HandReconstructor()
    ckpt = args.infiller_checkpoint or INFILLER_CHECKPOINT
    if not os.path.isfile(ckpt):
        print(f"[refresh_hands] WARNING: infiller ckpt not found at {ckpt}; "
              f"missing frames will not be filled.", flush=True)
    print(f"[refresh_hands] models loaded; processing {len(favs)} clip(s)",
          flush=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0_all = time.time()
    for i, d in enumerate(favs, 1):
        t0 = time.time()
        try:
            st, msg = _refresh_one(
                d, detector, reconstructor, ckpt, dry_run=args.dry_run)
        except Exception as e:
            st, msg = "fail", f"{type(e).__name__}: {e}"
        counts[st] += 1
        dt = time.time() - t0
        marker = {"ok": "+", "skip": "·", "fail": "x"}[st]
        print(f"  [{i:3d}/{len(favs)}] {marker} {d.name:48s}  "
              f"{dt:6.1f}s  {msg}", flush=True)

    dt = time.time() - t0_all
    print(f"\nDone in {dt/60:.1f} min — "
          f"ok: {counts['ok']}  skip: {counts['skip']}  fail: {counts['fail']}",
          flush=True)


if __name__ == "__main__":
    main()
