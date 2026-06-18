"""Detect SAM3 spurious detections — duplicate / background objects that
the text-prompted detector returned but are NOT actually being interacted
with in the scene.

Concrete example: a "blender" prompt picks up the main blender on the
counter PLUS a second blender visible far in the background.  The second
one is never touched and never moves — it's a duplicate that clutters
the 3D viewer.

Filter logic — flag if ANY rule fires:

  Rule (a) "far + static":  object is genuinely far from hand zone AND
    never moves on screen.
      d3d > d3d_threshold (default 0.5 m beyond hand-activity bbox)
      AND  motion2d < motion2d_threshold (default 10 px)

  Rule (b) "same-prompt duplicate":  same SAM3 prompt produces >=3
    detected instances → keep top-2 by motion (the "real" ones being
    interacted with), flag the rest as background duplicates.
      prompt occurs >= dup_min_count times (default 3)
      AND  this oid's motion < (motion of #2-ranked instance)
      AND  d3d > d3d_dup_threshold (default 0.2 m)
      AND  motion < motion_dup_max (default 25 px — guard against
                                    weakly-active items being mis-flagged)

  Rule (b) catches the common SAM3 failure where the same text prompt
  picks up multiple visually-similar objects in the scene (e.g. 5
  "whiskey bottle" detections — 2 being poured, 3 sitting in background).
  Rule (a) alone misses these because background dupes are NEAR the
  active instances (same shelf etc.) and their d3d is < 0.5 m.

Edge cases:
  - No hand data in clip: skip filter entirely (every oid marked "ok").
  - Object has no T_seq or empty masks: marked "no_data", not flagged.

Modes:
  soft   write spurious_flag/spurious_reason to pose_track_info; data
         retained.  Frontend / downstream can choose to hide flagged
         oids.  Default — safe, reversible.
  medium also remove oid from pose_track_info + sam3_mesh_info so viser
         export skips them.  PLY files preserved on disk.
  hard   also delete PLY (not implemented yet — preserve data integrity).

Usage
=====
  python -m egoinfinity.pipeline.post_tracking.spurious_filter --dry-run
  python -m egoinfinity.pipeline.post_tracking.spurious_filter --only=<clip>
  python -m egoinfinity.pipeline.post_tracking.spurious_filter --mode soft
  python -m egoinfinity.pipeline.post_tracking.spurious_filter --d3d-threshold 0.3 --motion2d-threshold 15
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


def _collect_hand_wrist_3d(fdata):
    """Returns (N, 3) array of all wrist world positions across all frames + hands.
    Empty if no hand data."""
    pts = []
    for fd in fdata:
        j3_list = fd.get("joints_3d_pred") or []
        for j in j3_list:
            j = np.asarray(j, dtype=np.float32)
            if j.shape != (21, 3):
                continue
            wrist = j[0]
            if np.all(np.isfinite(wrist)):
                pts.append(wrist)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(pts).astype(np.float32)


def _distance_to_bbox(pt, bb_lo, bb_hi):
    """Euclidean distance from a 3D point to a 3D axis-aligned bbox.
    Zero if point is inside."""
    d = np.maximum.reduce([bb_lo - pt, np.zeros(3), pt - bb_hi])
    return float(np.linalg.norm(d))


def _mask_bbox_center(mp_bytes, ms):
    """Return (cx, cy) bbox-midpoint of unpacked SAM2 mask. None if empty."""
    if mp_bytes is None or ms is None:
        return None
    H, W = int(ms[0]), int(ms[1])
    flat = np.unpackbits(np.asarray(mp_bytes, dtype=np.uint8))[: H * W]
    m = flat.astype(bool).reshape(H, W)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return (0.5 * (float(xs.min()) + float(xs.max())),
            0.5 * (float(ys.min()) + float(ys.max())))


def _object_world_center(pti_info, fdata, oid, dp_focal, cx, cy):
    """Median object world center across frames.
    Prefer per-frame OBB pose_t (more frames available); fallback to T_seq translation."""
    centers = []
    # Try OBB pose_t per frame (always available when mask exists)
    for fd in fdata:
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        pt = od.get("pose_t")
        if pt is None:
            continue
        pt = np.asarray(pt, dtype=np.float32)
        if pt.shape == (3,) and np.all(np.isfinite(pt)):
            centers.append(pt)
    if centers:
        return np.median(np.stack(centers), axis=0)
    # Fallback: T_seq translation
    T_seq = pti_info.get("T_seq")
    if T_seq is not None:
        T_seq = np.asarray(T_seq, dtype=np.float32)
        if T_seq.ndim == 3 and T_seq.shape[1:] == (4, 4):
            valid = np.isfinite(T_seq[:, :3, 3]).all(axis=1)
            if valid.any():
                return np.median(T_seq[valid, :3, 3], axis=0)
    return None


def _mask_motion_px(fdata, oid):
    """Total p10-p90 span of mask bbox-center 2D trajectory.  Returns 0 if no masks."""
    centers = []
    for fd in fdata:
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        c = _mask_bbox_center(od.get("mask_packed"), od.get("mask_shape"))
        if c is not None:
            centers.append(c)
    if len(centers) < 4:
        return 0.0
    arr = np.asarray(centers, dtype=np.float32)
    p10 = np.percentile(arr, 10, axis=0)
    p90 = np.percentile(arr, 90, axis=0)
    return float(np.linalg.norm(p90 - p10))


def _process_clip(fav_dir, d3d_threshold, motion2d_threshold,
                  hand_pad_m, mode, dry_run,
                  dup_min_count=3, d3d_dup_threshold=0.2, motion_dup_max=25.0):
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", []

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fdata = data.get("frame_data") or []
    pti = data.get("pose_track_info") or {}
    if not fdata or not pti:
        return "skip", "no frame_data or pose_track_info", []

    dp_focal = float(data.get("dp_focal", 0))
    cx = float(data.get("cx", 0))
    cy = float(data.get("cy", 0))

    # Build hand activity zone
    hand_pts = _collect_hand_wrist_3d(fdata)
    has_hand = len(hand_pts) >= 5
    if has_hand:
        hand_lo = hand_pts.min(axis=0) - hand_pad_m
        hand_hi = hand_pts.max(axis=0) + hand_pad_m
    else:
        hand_lo = hand_hi = None

    mapping = data.get("sam3_prompt_mapping") or []
    rows = []
    for oid in sorted(pti.keys()):
        if not isinstance(oid, int):
            continue
        info = pti[oid]
        if not isinstance(info, dict):
            continue
        prompt = (mapping[oid].get("prompt") if oid < len(mapping)
                  and isinstance(mapping[oid], dict) else "?")
        center = _object_world_center(info, fdata, oid, dp_focal, cx, cy)
        if center is None or not has_hand:
            row = {
                "oid": oid, "prompt": prompt,
                "d3d_m": None, "motion2d_px": None,
                "spurious": False,
                "reason": "no_hand_data" if not has_hand else "no_object_center",
            }
            rows.append(row)
            continue
        d3d = _distance_to_bbox(center, hand_lo, hand_hi)
        motion = _mask_motion_px(fdata, oid)
        rows.append({
            "oid": oid, "prompt": prompt,
            "d3d_m": d3d, "motion2d_px": motion,
            "spurious": False, "reason": "ok",
        })

    # Rule (a): far + static
    for r in rows:
        if r["d3d_m"] is None: continue
        if r["d3d_m"] > d3d_threshold and r["motion2d_px"] < motion2d_threshold:
            r["spurious"] = True
            r["reason"] = "far_static"

    # Rule (b): same-prompt duplicate (background dupes of an active object).
    # When the SAME prompt yields >= dup_min_count detections, keep top-2 by
    # motion (assumed real, being interacted with), flag the rest if they are
    # also somewhat outside hand zone.
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in rows:
        if r["d3d_m"] is None: continue
        by_prompt[r["prompt"]].append(r)
    for prompt, group in by_prompt.items():
        if len(group) < dup_min_count: continue
        motions = sorted([r["motion2d_px"] for r in group])
        keep_motion_floor = motions[-2]   # 2nd-highest motion in this prompt group
        for r in group:
            if (r["motion2d_px"] < keep_motion_floor
                    and r["d3d_m"] > d3d_dup_threshold
                    and r["motion2d_px"] < motion_dup_max
                    and not r["spurious"]):
                r["spurious"] = True
                r["reason"] = "prompt_dup_background"

    n_spurious = sum(1 for r in rows if r["spurious"])

    if not dry_run and n_spurious > 0 and mode != "report":
        # Write back: always set flags in pose_track_info
        for r in rows:
            oid = r["oid"]
            if oid in pti and isinstance(pti[oid], dict):
                pti[oid]["spurious_flag"] = r["spurious"]
                pti[oid]["spurious_reason"] = r["reason"]
                if r["d3d_m"] is not None:
                    pti[oid]["spurious_d3d_m"] = r["d3d_m"]
                if r["motion2d_px"] is not None:
                    pti[oid]["spurious_motion2d_px"] = r["motion2d_px"]

        if mode == "medium":
            # Remove flagged oids from three places:
            #   1. pose_track_info[oid]           — D-track 6DoF + state
            #   2. sam3_mesh_info[oid]            — SAM3D mesh metadata
            #   3. frame_data[t].sam3_obj_data[oid] — per-frame SAM2 mask,
            #      OBB corners, pose_t, point cloud source.  Critical:
            #      viser export reads this to render point cloud + OBB
            #      box, so dropping (1) + (2) alone leaves visible
            #      leftover "ghost" outlines in the 3D viewer.
            mi = data.get("sam3_mesh_info") or {}
            spurious_oids = {r["oid"] for r in rows if r["spurious"]}
            for oid in spurious_oids:
                pti.pop(oid, None)
                mi.pop(oid, None)
            for fd in fdata:
                sd = fd.get("sam3_obj_data")
                if isinstance(sd, dict):
                    for oid in spurious_oids:
                        sd.pop(oid, None)
                        sd.pop(int(oid), None)

        # Provenance
        prov = data.get("spurious_filter_refresh") or {}
        history = prov.get("history") or []
        history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": mode,
            "d3d_threshold": d3d_threshold,
            "motion2d_threshold": motion2d_threshold,
            "hand_pad_m": hand_pad_m,
            "n_spurious": n_spurious,
            "n_oids": len(rows),
            "details": [
                {"oid": r["oid"], "prompt": r["prompt"],
                 "d3d_m": r["d3d_m"], "motion2d_px": r["motion2d_px"]}
                for r in rows if r["spurious"]
            ],
        })
        data["spurious_filter_refresh"] = {"history": history}

        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    status = "ok" if n_spurious > 0 else "skip"
    summary = f"flag {n_spurious}/{len(rows)} spurious"
    return status, summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    ap.add_argument("--d3d-threshold", type=float, default=0.5,
                    help="3D distance (m) from hand zone beyond which object is FAR (default 0.5)")
    ap.add_argument("--motion2d-threshold", type=float, default=10.0,
                    help="2D mask bbox-center p10-p90 span (px) under which object is STATIC (default 10)")
    ap.add_argument("--hand-pad-m", type=float, default=0.20,
                    help="pad hand activity bbox by this on each side (m, default 0.20)")
    ap.add_argument("--dup-min-count", type=int, default=3,
                    help="rule (b): prompt must appear >= N times to trigger dup detection (default 3)")
    ap.add_argument("--d3d-dup-threshold", type=float, default=0.2,
                    help="rule (b): also require d3d > this m (default 0.2)")
    ap.add_argument("--motion-dup-max", type=float, default=25.0,
                    help="rule (b): only flag if motion < this px (default 25)")
    ap.add_argument("--mode", choices=["soft", "medium", "report"], default="soft",
                    help="soft: write flag only.  medium: also remove oid from pti+mi.  report: dry-run-like even without --dry-run.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    allow = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    favs = sorted(p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_spurious_filter: {len(favs)} clips, dry_run={args.dry_run}, "
          f"mode={args.mode}, d3d>{args.d3d_threshold}m AND motion2d<{args.motion2d_threshold}px")
    n_ok = n_skip = n_fail = 0
    total_flag = 0
    total_oids = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        status, summary, rows = _process_clip(
            fav,
            d3d_threshold=args.d3d_threshold,
            motion2d_threshold=args.motion2d_threshold,
            hand_pad_m=args.hand_pad_m,
            mode=args.mode,
            dry_run=args.dry_run,
            dup_min_count=args.dup_min_count,
            d3d_dup_threshold=args.d3d_dup_threshold,
            motion_dup_max=args.motion_dup_max,
        )
        dt = time.time() - t_clip
        total_oids += len(rows)
        n_flagged = sum(1 for r in rows if r["spurious"])
        total_flag += n_flagged
        if status == "ok":
            n_ok += 1
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}+ {fav.name:<50}  {summary}  ({dt:.1f}s)")
            for r in rows:
                if r["spurious"]:
                    print(f"      oid {r['oid']:>1} {r['prompt']:>22}  d3d={r['d3d_m']:.2f}m  motion2d={r['motion2d_px']:.1f}px  → SPURIOUS [{r['reason']}]")
        elif status == "skip":
            n_skip += 1
            # Quiet skip unless few clips
            if len(favs) <= 10:
                print(f"[{i:>3}/{len(favs)}] - {fav.name}  ({summary})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({summary})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — "
          f"clips with flags: {n_ok}  no_flags: {n_skip}  failed: {n_fail}  "
          f"flagged oids: {total_flag}/{total_oids}")


if __name__ == "__main__":
    main()
