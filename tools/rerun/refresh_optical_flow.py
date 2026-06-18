"""Compute MEMFOF optical flow per frame and store full-resolution
magnitude in the pkl (mask-independent).

For each clip and each frame t, runs MEMFOF on (frame[t], frame[t+1])
and saves the magnitude `sqrt(dx^2 + dy^2)` as a uint16 PNG into
``frame_data[t]['flow_mag_png']``.  Last frame copies frame[T-1]'s
magnitude.  Scale: 0.01 px/unit (max representable ≈ 655 px/frame).

Why per-pixel magnitude instead of per-mask flow:
  - Stays valid through future SAM3 / mask re-curation
  - Reusable by veto, detection, R-lock, etc.

Why uint16 PNG:
  - ~80-200 KB per frame (PNG compresses sparse-motion data well)
  - Lossless within 0.01 px precision
  - Same convention as `depth_png`

Idempotency: provenance under ``data['optical_flow_refresh']['history']``.

Usage::

    python -m tools.rerun.refresh_optical_flow --only=<CLIP_ID>
    python -m tools.rerun.refresh_optical_flow
    python -m tools.rerun.refresh_optical_flow --force
    python -m tools.rerun.refresh_optical_flow --skip-if-done
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"

SCALE = 100.0      # 0.01 px/unit; max uint16 65535 → 655.35 px/frame
MAX_FLOW_PX = 65535 / SCALE


def _encode_flow_mag(mag: np.ndarray) -> bytes:
    """Float magnitude (H, W) px/frame → uint16 PNG bytes."""
    u16 = np.clip(mag * SCALE, 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", u16)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def decode_flow_mag(buf: bytes) -> np.ndarray:
    """Inverse of _encode_flow_mag — uint16 PNG → (H, W) float32 px/frame."""
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float32) / SCALE


def _process_clip(fav_dir: Path, engine, *, dry_run: bool,
                  skip_if_done: bool, force: bool) -> tuple[str, str, dict]:
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if skip_if_done and not force:
        prov = (data.get("optical_flow_refresh") or {}).get("history") or []
        if prov:
            return "skip", "already computed", {}

    fdata = data.get("frame_data") or []
    if len(fdata) < 2:
        return "skip", "< 2 frames", {}

    # Decode all RGB frames up-front
    frames_rgb = []
    for fd in fdata:
        buf = fd.get("img_rgb")
        if buf is None:
            return "fail", "missing img_rgb", {}
        arr = np.frombuffer(buf, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return "fail", "img_rgb decode failed", {}
        frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    T = len(frames_rgb)
    H, W = frames_rgb[0].shape[:2]

    # Compute flow for each consecutive pair
    flow_mags: list[np.ndarray] = []
    t_inf_total = 0.0
    for t in range(T - 1):
        t0 = time.time()
        flow = engine.compute_flow(frames_rgb[t], frames_rgb[t + 1])   # (H, W, 2)
        t_inf_total += time.time() - t0
        mag = np.linalg.norm(flow, axis=2).astype(np.float32)
        flow_mags.append(mag)
    # Last frame reuses second-to-last
    flow_mags.append(flow_mags[-1].copy())

    # Encode + assign back to pkl
    total_bytes = 0
    for t, mag in enumerate(flow_mags):
        png = _encode_flow_mag(mag)
        total_bytes += len(png)
        if not dry_run:
            fdata[t]["flow_mag_png"] = png

    mean_mag = float(np.mean([m.mean() for m in flow_mags]))
    max_mag = float(np.max([m.max() for m in flow_mags]))

    stats = {
        "n_frames": T,
        "mean_flow_px": mean_mag,
        "max_flow_px": max_mag,
        "infer_seconds": float(t_inf_total),
        "total_png_bytes": total_bytes,
    }

    if not dry_run:
        prov = data.get("optical_flow_refresh") or {}
        history = prov.get("history") or []
        history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "scale": SCALE,
            "engine": "MEMFOF",
            "n_frames": T,
            "infer_seconds": float(t_inf_total),
            "total_png_bytes": total_bytes,
            "mean_flow_px": mean_mag,
            "max_flow_px": max_mag,
        })
        data["optical_flow_refresh"] = {"history": history}
        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)

    msg = (f"T={T} mean={mean_mag:.2f}px max={max_mag:.1f}px "
           f"infer={t_inf_total:.1f}s png={total_bytes/1e6:.1f}MB")
    return "ok", msg, stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="comma-separated clip ids (use --only=-ID for leading-dash)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute flow but don't save pkl")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips with prior optical_flow_refresh history")
    ap.add_argument("--force", action="store_true",
                    help="Re-compute even if history exists")
    ap.add_argument("--max-side", type=int, default=960,
                    help="Downsample inputs whose max(H,W) exceeds this "
                         "(default 960; 480×854 frames pass through)")
    args = ap.parse_args()

    allow = (set(s.strip() for s in args.only.split(",") if s.strip())
             if args.only else None)
    favs = sorted(p for p in FAV.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    # Lazy-load MEMFOF once
    from egoinfinity.pipeline.pose_tracker.memfof_flow import MEMFOFFlowEngine
    print(f"refresh_optical_flow: loading MEMFOF (one-time, ~2s)...")
    engine = MEMFOFFlowEngine(max_side=args.max_side)
    engine._ensure_loaded()

    print(f"refresh_optical_flow: {len(favs)} clips  dry_run={args.dry_run}  "
          f"max_side={args.max_side}")
    n_ok = n_skip = n_fail = 0
    total_infer = 0.0
    total_png = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        try:
            status, msg, stats = _process_clip(
                fav, engine,
                dry_run=args.dry_run,
                skip_if_done=args.skip_if_done,
                force=args.force,
            )
        except Exception as e:
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  CRASH ({type(e).__name__}: {e})")
            continue
        dt = time.time() - t_clip
        if status == "ok":
            n_ok += 1
            total_infer += stats.get("infer_seconds", 0)
            total_png += stats.get("total_png_bytes", 0)
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}"
                  f"+ {fav.name:<48}  {msg}  ({dt:.1f}s)")
        elif status == "skip":
            n_skip += 1
            print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({msg})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({msg})")

    total_min = (time.time() - t0) / 60
    print(f"\nDone in {total_min:.1f} min — "
          f"ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")
    print(f"Total MEMFOF inference: {total_infer/60:.1f} min")
    print(f"Total PNG bytes written: {total_png/1e9:.2f} GB")


if __name__ == "__main__":
    main()
