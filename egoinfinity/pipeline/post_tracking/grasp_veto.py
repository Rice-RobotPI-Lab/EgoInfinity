"""Post-hoc grasp false-positive veto using the 2D-centroid static signal.

Reads ``pose_track_info`` from each clip's ``pipeline_result.pkl.gz``, finds
candidate grasp segments per hand from the stored ``wrist_l_per_frame`` /
``wrist_r_per_frame`` (these came from ``_estimate_state_per_frame`` during
``refresh_pose_tracking``).  For each candidate segment, checks whether the
object was confidently static (low 2D centroid displacement) for the entire
segment.  If ≥``static_frac_thr`` fraction of segment frames are static,
the segment is vetoed.

Why this is useful
==================
Current grasp detector ``detect_grasp_fingertip_persistent`` is pure
geometry (fingertip ≤ 6 cm of obs cloud, persistent ≥ 8 frames).  Triggers
false positives when a hand brushes / rests near a static object.

For the **depth-alignment use case** (modify object depth to align with the
hand grasp point), a false positive is catastrophic — would pull a static
object toward the hand.  Precision matters more than recall, since
held-still objects (the case the original detector tried to preserve)
don't need depth correction anyway (they're already at their resting
depth).  So vetoing FPs is the right trade-off.

What this tool touches
======================
ONLY status-classification fields:
  - ``wrist_l_per_frame`` / ``wrist_r_per_frame`` / ``wrist_used_per_frame``
  - ``state_per_frame``, ``grasp_hand_per_frame``, ``state_counts``
  - ``is_moving_per_frame`` (legacy)

Does NOT touch:
  - ``T_seq`` (object 6DoF pose) — already committed by main pipeline
  - ``pose_R`` / ``pose_t`` / ``obb_corners`` in ``sam3_obj_data``
  - Hand fields (joints, vertices, depth), mesh, masks

Result: the colour dot above each object in the viser viewer reflects the
new state; the object's actual pose trajectory is unchanged.  For the FP
frames, the object continues to track the wrist (the original bug) — fixing
that requires re-running the main pipeline.  This tool is the cheap first
step.

Usage
=====
    python -m egoinfinity.pipeline.post_tracking.grasp_veto --dry-run
    python -m egoinfinity.pipeline.post_tracking.grasp_veto --only=CLIP_ID
    python -m egoinfinity.pipeline.post_tracking.grasp_veto
    python -m egoinfinity.pipeline.post_tracking.grasp_veto --static-frac-thr 0.85 --min-seg-len 3
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"


# Defaults match _estimate_state_per_frame (Schmitt trigger on centroid disp)
DEFAULT_LOW_PX = 2.0
DEFAULT_HIGH_PX = 4.0
DEFAULT_STATIC_FRAC_THR = 0.85
DEFAULT_MIN_SEG_LEN = 3


# ── helpers ───────────────────────────────────────────────────────────
def _hysteresis(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Schmitt trigger — same as in egoinfinity/pipeline/post_tracking/pose_tracking.py."""
    T = len(x)
    out = np.zeros(T, dtype=bool)
    on = False
    for t in range(T):
        if on:
            if x[t] < low:
                on = False
        else:
            if x[t] > high:
                on = True
        out[t] = on
    return out


def _find_segments(b: np.ndarray, min_len: int = 1):
    """Yield (s, e) inclusive segments where ``b`` is True, of length ≥ min_len."""
    T = len(b)
    i = 0
    segs = []
    while i < T:
        if b[i]:
            j = i
            while j < T and b[j]:
                j += 1
            if (j - i) >= min_len:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def _veto_per_hand(wrist_pf: np.ndarray, confident_static: np.ndarray,
                   static_frac_thr: float, min_seg_len: int,
                   obj_flow_per_frame: 'np.ndarray | None' = None,
                   flow_static_thr_px: float = 1.0,
                   flow_moving_frac_thr: float = 0.20,
                   contact_per_frame: 'np.ndarray | None' = None,
                   contact_keep_frac_thr: float = 0.60,
                   global_span_px: float = 0.0,
                   contact_keep_span_min_px: float = 50.0):
    """Return (vetoed_wrist_pf, list_of_vetoed_segments).

    Each vetoed segment recorded as (s, e, static_frac, why, frac_moving)
    where ``why`` is "centroid" / "flow" / "both" indicating which signal
    triggered.

    Caller has already guaranteed ``is_static_global=False`` for this object
    (the globally-static branch in _process_object handles that separately).
    So we know the object DID move at SOME point in the clip. The question
    per-segment is whether THIS particular segment is a real grasp.

    Veto fires when EITHER:
      - centroid_2D static fraction ≥ static_frac_thr (existing behavior)
      - fraction of per-frame obj flow magnitudes > flow_static_thr_px is
        below flow_moving_frac_thr — fewer than 20% of frames showed real
        translation flow

    Contact-based short-circuit (overrides both above):
      - If ``contact_per_frame`` is provided and `close=True for ≥
        contact_keep_frac_thr (60%) of the segment frames`, the segment is
        kept regardless. Reasoning: object globally moves AND hand maintains
        contact for most of the segment ⇒ real grasp, not FP. This catches
        "hold mostly still + briefly manipulate" grasps (bottle held + tilted
        to pour, screwdriver held + small adjustments) where centroid AND
        flow over the whole segment look static because the manipulation is
        only a sub-region.

    Why fraction instead of median for the flow gate: a real grasp can be
    "held still then actively manipulated". Median over the whole segment is
    dominated by the still frames (median ≈ 0.03 px/f even when 25% of
    frames have flow up to 41 px/f), which falsely tagged the bottle as
    static. Fraction-of-moving-frames recovers any segment with a sub-region
    of real translation flow.
    """
    out = wrist_pf.copy()
    vetoed = []
    for (s, e) in _find_segments(wrist_pf, min_len=min_seg_len):
        n_seg = e - s + 1
        n_static = int(confident_static[s:e + 1].sum())
        frac = n_static / max(n_seg, 1)
        centroid_says_static = frac >= static_frac_thr

        flow_says_static = False
        frac_moving = float('nan')
        if obj_flow_per_frame is not None:
            seg_flow = obj_flow_per_frame[s:e + 1]
            valid = np.isfinite(seg_flow)
            if valid.sum() >= max(3, n_seg // 3):
                frames_moving = int((seg_flow[valid] > flow_static_thr_px).sum())
                frac_moving = frames_moving / int(valid.sum())
                flow_says_static = frac_moving < flow_moving_frac_thr

        # Contact-based short-circuit: globally CLEARLY-moving object (span
        # well above mask-jitter floor) + sustained hand contact ⇒ real grasp
        # regardless of per-segment motion stats. The extra span gate keeps
        # mask-jittery near-static objects (e.g. door panel where hand merely
        # rests on it, span 17.5 px ≈ 2× the 9.6 px static threshold) out of
        # this branch — those fall through to the standard centroid/flow veto.
        keep_via_contact = False
        if (contact_per_frame is not None
                and global_span_px >= contact_keep_span_min_px):
            seg_contact = contact_per_frame[s:e + 1]
            frac_contact = float(seg_contact.sum()) / max(n_seg, 1)
            if frac_contact >= contact_keep_frac_thr:
                keep_via_contact = True

        if (centroid_says_static or flow_says_static) and not keep_via_contact:
            out[s:e + 1] = False
            why = (("centroid" if centroid_says_static else "")
                   + ("+flow" if flow_says_static and centroid_says_static else
                      ("flow" if flow_says_static else "")))
            vetoed.append((int(s), int(e), float(frac), why,
                            float(frac_moving) if not np.isnan(frac_moving) else None))
    return out, vetoed


def _recompose_state(wrist_l: np.ndarray, wrist_r: np.ndarray,
                     is_moving_2d: np.ndarray, is_static_global: bool):
    """Replicate the state composition from _estimate_state_per_frame:539-560."""
    T = len(wrist_l)
    states = []
    grasp_hand = []
    if is_static_global:
        return (["static"] * T, [None] * T)
    for t in range(T):
        if wrist_l[t] and wrist_r[t]:
            states.append("grasped_both"); grasp_hand.append("both")
        elif wrist_l[t]:
            states.append("grasped_l"); grasp_hand.append("L")
        elif wrist_r[t]:
            states.append("grasped_r"); grasp_hand.append("R")
        elif is_moving_2d[t]:
            states.append("moving"); grasp_hand.append(None)
        else:
            states.append("static"); grasp_hand.append(None)
    return states, grasp_hand


def _compute_obj_flow_per_frame(info, fdata, oid, mano_faces, K, H, W,
                                 min_eff_px: int = 500,
                                 erode_px: int = 5):
    """For each frame, MEDIAN optical flow magnitude inside the eroded
    ``obj_mask & ~hand_mask``.  Returns (T,) float32 with NaN where flow
    can't be trusted (mask too small, fully occluded, etc.).

    Three defences against the FP-on-static-object failure mode:

    1. ``min_eff_px``: if effective mask < 500 px (heavy hand occlusion
       reducing mask to a sliver), flow is dominated by boundary
       artifacts → emit NaN and let veto skip this frame.
    2. ``erode_px``: erode the effective mask by 5 px to drop the
       hand-boundary "halo" where MEMFOF picks up artifactual motion.
    3. ``median`` instead of ``mean``: robust to a few high-flow
       outliers from edge pixels.

    Uses ``frame_data[t]['flow_mag_png']`` (written by
    tools.rerun.refresh_optical_flow) — full-resolution MEMFOF magnitude.
    """
    import cv2
    T = len(fdata)
    out = np.full(T, np.nan, dtype=np.float32)
    if mano_faces is not None:
        try:
            from egoinfinity.pipeline.pose_tracker.utils import render_hand_mask
            mano_faces_np = np.asarray(mano_faces)
        except Exception:
            mano_faces_np = None
    else:
        mano_faces_np = None
    erode_kernel = np.ones((erode_px, erode_px), np.uint8) if erode_px > 0 else None
    for t in range(T):
        fd = fdata[t]
        flow_buf = fd.get('flow_mag_png')
        if flow_buf is None:
            continue
        arr = np.frombuffer(flow_buf, np.uint8)
        flow_mag = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if flow_mag is None:
            continue
        flow_mag = flow_mag.astype(np.float32) / 100.0
        sd = fd.get('sam3_obj_data') or {}
        od = sd.get(oid) or sd.get(int(oid)) or sd.get(str(oid))
        if not isinstance(od, dict):
            continue
        mp = od.get('mask_packed')
        ms = od.get('mask_shape')
        if mp is None or ms is None:
            continue
        h, w = int(ms[0]), int(ms[1])
        bits = np.unpackbits(np.asarray(mp, np.uint8))[: h * w]
        obj_m = bits.astype(bool).reshape(h, w)
        if not obj_m.any():
            continue
        # Subtract hand mask
        eff_mask = obj_m
        if mano_faces_np is not None and K is not None and fd.get('vertices_3d'):
            verts = [v for v in fd['vertices_3d'] if v is not None and len(v) > 0]
            if verts:
                try:
                    hand_m = render_hand_mask(verts, mano_faces_np, K, h, w, dilate_px=3)
                    eff_mask = obj_m & ~hand_m
                except Exception:
                    pass
        # (1) Size gate — too small means dominated by hand-boundary artifacts
        if int(eff_mask.sum()) < min_eff_px:
            continue   # NaN
        # (2) Erode by erode_px to remove hand-boundary halo
        if erode_kernel is not None:
            eroded = cv2.erode(eff_mask.astype(np.uint8), erode_kernel,
                                iterations=1).astype(bool)
            if int(eroded.sum()) >= 100:
                eff_mask = eroded
            # else: erode left too few pixels; keep original eff_mask
        # Resize flow to mask shape if needed
        if flow_mag.shape != (h, w):
            flow_mag = cv2.resize(flow_mag, (w, h), interpolation=cv2.INTER_LINEAR)
        # (3) Median (not mean) — robust to a few hot edge pixels
        out[t] = float(np.median(flow_mag[eff_mask]))
    return out


def _process_object(info: dict, *, low_px: float, high_px: float,
                    static_frac_thr: float, min_seg_len: int,
                    obj_flow_per_frame=None,
                    flow_static_thr_px: float = 1.0):
    """Apply veto on one pose_track_info[oid] entry.

    Mutates ``info`` in place when not dry_run upstream. Returns stats dict.
    """
    disp = info.get("centroid_2d_disp_per_frame")
    wl = info.get("wrist_l_per_frame")
    wr = info.get("wrist_r_per_frame")
    is_static_global = bool(info.get("is_static_global", False))

    if disp is None or wl is None or wr is None:
        return {"status": "skip", "reason": "missing centroid_2d_disp / wrist_l / wrist_r"}

    T = len(wl)
    if T == 0:
        return {"status": "skip", "reason": "empty"}

    disp_arr = np.asarray(disp, dtype=np.float32)
    wl_arr = np.asarray(wl, dtype=bool)
    wr_arr = np.asarray(wr, dtype=bool)

    # Centroid-based is_moving (legacy signal).
    is_moving_centroid = _hysteresis(disp_arr, low=low_px, high=high_px)

    # Per-frame flow-based is_moving — when available, REQUIRE both centroid
    # AND flow to indicate motion for a frame to be considered "moving".
    # This rejects mask-jitter-induced centroid spikes on truly static
    # objects (the same root cause that produced the FP grasps).
    #
    # When flow is NaN (mask too small / heavy hand occlusion, can't trust
    # flow), default to NOT MOVING.  Rationale: centroid 2D jitters during
    # occlusion (the original FP cause); falling back to centroid alone
    # reintroduces those FPs.  Carry-forward from previous frame also
    # propagates True from a single real moving frame across subsequent
    # NaN frames.  The conservative "NaN → not moving" matches the spirit
    # of veto: only declare moving when we have positive evidence (centroid
    # AND flow both say moving).  For real moving objects, the visible
    # mask is usually large enough that flow is computable (not NaN).
    T = len(disp_arr)
    if obj_flow_per_frame is not None and len(obj_flow_per_frame) == T:
        flow_arr = np.asarray(obj_flow_per_frame, dtype=np.float64)
        flow_finite = np.isfinite(flow_arr)
        flow_says_moving = np.where(
            flow_finite,
            flow_arr > flow_static_thr_px,
            False,   # NaN → conservatively "not moving"
        )
        is_moving_2d = is_moving_centroid & flow_says_moving
    else:
        is_moving_2d = is_moving_centroid
    confident_static = ~is_moving_2d

    if is_static_global:
        # Whole clip is globally static.  ``_estimate_state_per_frame`` only
        # overrides the STATE LABEL ("static"), it does NOT zero out
        # wrist_l/r — those still carry the raw fingertip-persistent True
        # values from grasp detection.  Downstream tracking (_track_phase_d)
        # reads wrist_l/r as the WRIST gate, so leaving raw True values
        # produces the bug "state shows static but object moves with hand".
        #
        # Fix: zero out wrist_l/r entirely for globally-static objects.
        # "Globally static" by definition means the object doesn't move in
        # the clip → it cannot be a real grasp segment (FP by construction).
        n_before = int(wl_arr.sum() + wr_arr.sum())
        wl_new = np.zeros_like(wl_arr)
        wr_new = np.zeros_like(wr_arr)
        # Treat the wipe as a single virtual veto segment for diagnostics.
        vetoed_l = []
        vetoed_r = []
        if wl_arr.any():
            vetoed_l = [(0, len(wl_arr) - 1, 1.0)]
        if wr_arr.any():
            vetoed_r = [(0, len(wr_arr) - 1, 1.0)]
    else:
        # close_per_frame (when present) + global_span_px let the veto keep
        # "hold mostly still + briefly manipulate" grasps on clearly-moving
        # objects (bottle pour, screwdriver use) while still vetoing
        # near-static borderline objects (door panel resting under a hand).
        _close_pf = info.get("close_per_frame")
        contact_arr = (np.asarray(_close_pf, dtype=bool)
                       if _close_pf is not None and len(_close_pf) == T
                       else None)
        span_px = float(info.get("global_span_px", 0.0) or 0.0)
        wl_new, vetoed_l = _veto_per_hand(wl_arr, confident_static,
                                           static_frac_thr, min_seg_len,
                                           obj_flow_per_frame=obj_flow_per_frame,
                                           flow_static_thr_px=flow_static_thr_px,
                                           contact_per_frame=contact_arr,
                                           global_span_px=span_px)
        wr_new, vetoed_r = _veto_per_hand(wr_arr, confident_static,
                                           static_frac_thr, min_seg_len,
                                           obj_flow_per_frame=obj_flow_per_frame,
                                           flow_static_thr_px=flow_static_thr_px,
                                           contact_per_frame=contact_arr,
                                           global_span_px=span_px)

    n_before = int(wl_arr.sum() + wr_arr.sum())
    n_after = int(wl_new.sum() + wr_new.sum())

    states, grasp_hand = _recompose_state(
        wl_new, wr_new, is_moving_2d, is_static_global)
    wrist_used_new = wl_new | wr_new
    is_moving_legacy = is_moving_2d & ~wrist_used_new

    cnt = Counter(states)
    state_counts = {
        "static": cnt.get("static", 0),
        "grasped_l": cnt.get("grasped_l", 0),
        "grasped_r": cnt.get("grasped_r", 0),
        "grasped_both": cnt.get("grasped_both", 0),
        "moving": cnt.get("moving", 0),
    }

    return {
        "status": "ok",
        "vetoed_l": vetoed_l,
        "vetoed_r": vetoed_r,
        "n_grasp_before": n_before,
        "n_grasp_after": n_after,
        "is_static_global": False,
        "_new": {  # to apply when not dry_run
            "wrist_l_per_frame": [bool(v) for v in wl_new],
            "wrist_r_per_frame": [bool(v) for v in wr_new],
            "wrist_used_per_frame": [bool(v) for v in wrist_used_new],
            "is_moving_per_frame": [bool(v) for v in is_moving_legacy],
            "state_per_frame": states,
            "grasp_hand_per_frame": grasp_hand,
            "state_counts": state_counts,
        },
    }


# ── orchestration ─────────────────────────────────────────────────────
def _process_clip(fav_dir: Path, *, low_px: float, high_px: float,
                  static_frac_thr: float, min_seg_len: int,
                  dry_run: bool, skip_if_done: bool, force: bool,
                  flow_static_thr_px: float = 1.0):
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Idempotency: skip clips already processed with this tool
    if skip_if_done and not force:
        prov = data.get("grasp_veto_refresh") or {}
        if (prov.get("history") or []):
            return "skip", "already vetoed", {}

    pti = data.get("pose_track_info") or {}
    if not pti:
        return "skip", "no pose_track_info", {}

    fdata = data.get("frame_data") or []
    mano_faces = data.get("mano_faces")
    dp_focal = float(data.get("dp_focal", 0.0))
    # Image dims from first available mask
    H = W = None
    for fd in fdata:
        sd = fd.get("sam3_obj_data") or {}
        for od in sd.values():
            if isinstance(od, dict) and od.get("mask_shape") is not None:
                ms = od["mask_shape"]
                H, W = int(ms[0]), int(ms[1]); break
        if H is not None: break
    K = None
    if dp_focal > 0 and H is not None:
        cx = float(data.get("cx", W / 2.0))
        cy = float(data.get("cy", H / 2.0))
        K = np.array([[dp_focal, 0, cx], [0, dp_focal, cy], [0, 0, 1]], dtype=np.float64)

    # Check if any frame has flow data
    has_flow = any('flow_mag_png' in fd for fd in fdata[:5])

    summary = {}
    total_before = 0
    total_after = 0
    total_vetoed_segs = 0
    # Snapshot old is_moving_per_frame BEFORE _process_object can mutate info
    old_is_moving = {oid: list(info.get("is_moving_per_frame", []))
                      for oid, info in pti.items() if isinstance(info, dict)}

    for oid, info in list(pti.items()):
        if not isinstance(info, dict):
            continue
        # Per-oid obj flow (mean inside obj_mask & ~hand_mask per frame)
        obj_flow = None
        if has_flow and H is not None and K is not None:
            try:
                obj_flow = _compute_obj_flow_per_frame(info, fdata, oid,
                                                       mano_faces, K, H, W)
            except Exception:
                obj_flow = None
        res = _process_object(info, low_px=low_px, high_px=high_px,
                              static_frac_thr=static_frac_thr,
                              min_seg_len=min_seg_len,
                              obj_flow_per_frame=obj_flow,
                              flow_static_thr_px=flow_static_thr_px)
        if res["status"] != "ok":
            summary[oid] = res
            continue
        summary[oid] = res
        total_before += res.get("n_grasp_before", 0)
        total_after += res.get("n_grasp_after", 0)
        total_vetoed_segs += len(res.get("vetoed_l", [])) + len(res.get("vetoed_r", []))

        if not dry_run and "_new" in res:
            new_fields = res["_new"]
            for k, v in new_fields.items():
                info[k] = v

    delta = total_before - total_after
    action_summary = (f"obj={len(pti)}  grasp_frames {total_before}→{total_after} "
                      f"(−{delta}, segs vetoed={total_vetoed_segs})")

    # Check whether is_moving_per_frame changed for any oid (the flow-aware
    # path can null out FP moving frames even when no grasps are vetoed).
    # Compare against the pre-mutation snapshot.
    is_moving_changed = False
    for oid_k, s in summary.items():
        if s.get("status") != "ok": continue
        new_im = (s.get("_new") or {}).get("is_moving_per_frame")
        if new_im is None: continue
        old_im = old_is_moving.get(oid_k, [])
        if list(old_im) != list(new_im):
            is_moving_changed = True
            break

    if delta == 0 and total_vetoed_segs == 0 and not is_moving_changed:
        # Nothing changed: don't bump provenance / file mtime
        return "skip", "nothing to veto", summary

    if not dry_run:
        prov = data.get("grasp_veto_refresh") or {}
        history = prov.get("history") or []
        history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "low_px": low_px, "high_px": high_px,
            "static_frac_thr": static_frac_thr,
            "min_seg_len": min_seg_len,
            "objs": {
                oid: {
                    "n_before": int(s.get("n_grasp_before", 0)),
                    "n_after": int(s.get("n_grasp_after", 0)),
                    "n_segs_vetoed": len(s.get("vetoed_l", [])) + len(s.get("vetoed_r", [])),
                }
                for oid, s in summary.items() if s.get("status") == "ok"
            },
        })
        data["grasp_veto_refresh"] = {"history": history}

        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)

    return "ok", action_summary, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="comma-separated clip ids (use --only=-ID for leading-dash)")
    ap.add_argument("--low-px", type=float, default=DEFAULT_LOW_PX,
                    help=f"Schmitt-trigger LOW threshold for centroid disp (px), "
                         f"default {DEFAULT_LOW_PX}")
    ap.add_argument("--high-px", type=float, default=DEFAULT_HIGH_PX,
                    help=f"Schmitt-trigger HIGH threshold for centroid disp (px), "
                         f"default {DEFAULT_HIGH_PX}")
    ap.add_argument("--static-frac-thr", type=float, default=DEFAULT_STATIC_FRAC_THR,
                    help=f"min fraction of segment frames that must be static to "
                         f"veto (default {DEFAULT_STATIC_FRAC_THR})")
    ap.add_argument("--min-seg-len", type=int, default=DEFAULT_MIN_SEG_LEN,
                    help=f"only evaluate candidate grasp segments of length ≥ this "
                         f"(default {DEFAULT_MIN_SEG_LEN})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute veto stats without writing")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips with prior grasp_veto_refresh history "
                         "(avoid repeated application — second pass is a no-op "
                         "on already-vetoed segments but bumps mtime needlessly)")
    ap.add_argument("--force", action="store_true",
                    help="Ignore prior provenance; re-evaluate from current pkl state")
    ap.add_argument("--flow-static-thr-px", type=float, default=1.0,
                    help="If MEMFOF optical flow median inside obj_mask ∩ ~hand_mask "
                         "is below this (px/frame), veto the segment regardless of "
                         "centroid_2D check (default 1.0). Requires flow_mag_png "
                         "fields written by tools.rerun.refresh_optical_flow.")
    args = ap.parse_args()

    allow = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    favs = sorted(p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_grasp_veto: {len(favs)} clips  dry_run={args.dry_run}  "
          f"low={args.low_px}px high={args.high_px}px  thr={args.static_frac_thr:.2f}  "
          f"min_seg={args.min_seg_len}")
    n_ok = n_skip = n_fail = 0
    n_clips_with_change = 0
    total_before_all = 0
    total_after_all = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        try:
            status, action, summary = _process_clip(
                fav, low_px=args.low_px, high_px=args.high_px,
                static_frac_thr=args.static_frac_thr,
                min_seg_len=args.min_seg_len,
                dry_run=args.dry_run,
                skip_if_done=args.skip_if_done,
                force=args.force,
                flow_static_thr_px=args.flow_static_thr_px)
        except Exception as e:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  CRASH ({type(e).__name__}: {e})")
            continue
        if status == "ok":
            n_ok += 1
            n_clips_with_change += 1
            for s in summary.values():
                if isinstance(s, dict) and s.get("status") == "ok":
                    total_before_all += s.get("n_grasp_before", 0)
                    total_after_all += s.get("n_grasp_after", 0)
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}"
                  f"+ {fav.name:<50}  {action}")
        elif status == "skip":
            n_skip += 1
            # Quiet on "nothing to veto" — that's the common case
            if action != "nothing to veto":
                print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({action})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({action})")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min — "
          f"ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")
    print(f"Clips with at least one veto: {n_clips_with_change}")
    print(f"Total grasp frames {total_before_all} → {total_after_all} "
          f"(removed {total_before_all - total_after_all})")


if __name__ == "__main__":
    main()
