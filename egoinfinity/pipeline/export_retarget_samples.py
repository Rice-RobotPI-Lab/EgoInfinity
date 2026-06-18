"""Export pipeline_result.pkl.gz -> retarget samples directory.

Produces the SamplesSequence format expected by retarget/utils/clip_io.py
(maintained by charlierkj). The retarget module reads:

  <samples_dir>/
  ├── hand_joints.bin    (T, max_h, 21, 3) float32 camera-frame keypoints
  │                      NaN-padded for absent hands
  ├── hand_meta.json     { n_frames, max_hands, joints_shape, joints_dtype,
  │                        is_right_per_frame, n_hands_per_frame,
  │                        nan_means_absent }
  ├── scene.json         { id: "<video_id>_<start>_<end>",
  │                        fps, duration,
  │                        camera: { focal, cx, cy, width, height,
  │                                  gravity_up } }
  └── depth.mp4          Colormap-rendered depth, baked from frame_data[*]
                         depth_png. Used by retarget/scripts/test.py to
                         derive (vid_h, vid_w) and as the canvas for
                         input_viz.mp4 overlay.

Joints convention: camera-frame, Z-forward (depth > 0 in front of camera),
which matches MoGe-2 / WiLoR / retarget all the way through. No
world_R_cam needed — retarget's SamplesSequence assumes camera frame
when world_R_cam is absent.

Usage::

    python -m egoinfinity.pipeline.export_retarget_samples --only=CLIP_ID
    python -m egoinfinity.pipeline.export_retarget_samples --only=CLIP_ID --skip-if-done
    python -m egoinfinity.pipeline.export_retarget_samples --only=CLIP_ID --force
    python -m egoinfinity.pipeline.export_retarget_samples --max-hands=4   # default 2

The samples dir is written to ``<fav_dir>/retarget_samples/``. The
canonical 8-stage post-tracking sequence should be done before running
this (hand joints are biomech-smoothed + infiller-filled by Phase C/C+).

Provenance: ``data['retarget_samples_export'] = {'history': [...]}``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO_ROOT / "cache"))) / "favorites"


DEFAULT_MAX_HANDS = 2  # matches dev's MAX_HANDS; retarget expects 2 by default


def _safe_list(x, default):
    if x is None:
        return default
    try:
        return [float(v) for v in x]
    except Exception:
        return default


def _hands_to_bins(frame_data: list, max_hands: int):
    """Pack joints into (T, max_h, 21, 3) NaN-padded float32 array.

    Returns (joints (T, max_h, 21, 3), meta dict).
    """
    T = len(frame_data)
    joints = np.full((T, max_hands, 21, 3), np.nan, dtype=np.float32)
    is_right = [[None] * max_hands for _ in range(T)]
    n_hands = [0] * T
    for ti, fd in enumerate(frame_data):
        j_list = fd.get("joints_3d_pred") or []
        r_list = fd.get("hand_is_right") or []
        n = min(len(j_list), max_hands)
        n_hands[ti] = n
        for hi in range(n):
            if j_list[hi] is not None:
                j = np.asarray(j_list[hi], dtype=np.float32)
                if j.shape == (21, 3):
                    joints[ti, hi] = j
            if hi < len(r_list):
                is_right[ti][hi] = bool(r_list[hi])
    meta = {
        "n_frames": T,
        "max_hands": max_hands,
        "n_hands_per_frame": n_hands,
        "is_right_per_frame": is_right,
        "joints_dtype": "float32",
        "joints_shape": [T, max_hands, 21, 3],
        "nan_means_absent": True,
    }
    return joints, meta


def _depth_to_colormap(depth_png_bytes: bytes) -> np.ndarray | None:
    """Decode uint16 depth PNG and apply turbo colormap. Returns HxWx3 uint8 RGB."""
    if not depth_png_bytes:
        return None
    arr = np.frombuffer(depth_png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    depth = img.astype(np.float32)
    if depth.ndim != 2:
        return None
    # Normalize to [0, 255] using a fixed range (avoid per-frame flicker)
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    z_max = float(np.percentile(valid, 99))
    z_min = float(np.percentile(valid, 1))
    norm = np.clip((depth - z_min) / max(z_max - z_min, 1e-6), 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _bake_depth_mp4(frame_data: list, out_path: Path, fps: float) -> int:
    """Write an mp4 visualizing depth across all frames. Returns frame count."""
    import imageio.v2 as imageio
    writer = None
    n = 0
    try:
        for fd in frame_data:
            img = _depth_to_colormap(fd.get("depth_png"))
            if img is None:
                continue
            if writer is None:
                writer = imageio.get_writer(
                    str(out_path), fps=float(fps), codec="libx264",
                    pixelformat="yuv420p", quality=8, macro_block_size=2)
            writer.append_data(img)
            n += 1
    finally:
        if writer is not None:
            writer.close()
    return n


def _frame_size(frame_data: list) -> tuple[int, int]:
    """Decode first frame's img_rgb to get (W, H)."""
    if not frame_data:
        return (854, 480)
    fd = frame_data[0]
    img_bytes = fd.get("img_rgb")
    if img_bytes:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return (img.shape[1], img.shape[0])
    # Fallback: depth_png shape
    depth_bytes = fd.get("depth_png")
    if depth_bytes:
        img = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return (img.shape[1], img.shape[0])
    return (854, 480)


def _scene_id(clip_id: str, manifest: dict | None, T: int, fps: float) -> str:
    """Build a scene id of the form "<video_id>_<start>_<end>".

    retarget's SamplesSequence parses this for video_id, start_sec, end_sec.
    Prefer manifest values when available; otherwise fall back to the clip_id.
    """
    if manifest:
        vid = manifest.get("video_uid") or manifest.get("video_uri")
        s = manifest.get("start_sec")
        e = manifest.get("end_sec")
        if vid is not None and s is not None and e is not None:
            return f"{vid}_{float(s):.1f}_{float(e):.1f}"
    return clip_id


def process_clip(fav_dir: Path, *, max_hands: int, fps: float | None,
                 force: bool, skip_if_done: bool, dry_run: bool
                 ) -> tuple[str, str, dict]:
    """Process one clip. Returns (status, action, stats)."""
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if skip_if_done and not force:
        prov = data.get("retarget_samples_export") or {}
        if prov.get("history"):
            return "skip", "already exported", {}

    frame_data = data.get("frame_data") or []
    if not frame_data:
        return "skip", "no frame_data", {}

    # Required prerequisites: joints_3d_pred + camera intrinsics + gravity
    fd0 = frame_data[0]
    if not fd0.get("joints_3d_pred"):
        return "skip", "no joints_3d_pred (run Phase B + C/C+ first)", {}

    manifest_path = fav_dir / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = None

    # Resolve fps: prefer manifest start/end (wall-clock playback), else default 15
    T = len(frame_data)
    if fps is None:
        if manifest and manifest.get("start_sec") is not None and manifest.get("end_sec") is not None:
            duration_s = float(manifest["end_sec"]) - float(manifest["start_sec"])
            if duration_s > 0:
                fps = T / duration_s
        if fps is None:
            fps = 15.0

    # Camera intrinsics
    W, H = _frame_size(frame_data)
    cam = {
        "focal": float(data.get("dp_focal", 500.0)),
        "cx": float(data.get("cx", W / 2.0)),
        "cy": float(data.get("cy", H / 2.0)),
        "width": int(W),
        "height": int(H),
        "gravity_up": _safe_list(data.get("gravity_up"), [0.0, -1.0, 0.0]),
    }

    # Hands
    joints, hand_meta = _hands_to_bins(frame_data, max_hands=max_hands)

    # Scene
    scene_id = _scene_id(fav_dir.name, manifest, T, fps)
    scene = {
        "id": scene_id,
        "fps": float(fps),
        "duration": T / float(fps),
        "n_frames": int(T),
        "camera": cam,
    }

    if dry_run:
        return "dry", "would write", {
            "T": T, "max_hands": max_hands, "fps": fps,
            "scene_id": scene_id, "W": W, "H": H,
        }

    out_dir = fav_dir / "retarget_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write 4 files
    (out_dir / "hand_joints.bin").write_bytes(joints.tobytes(order="C"))
    (out_dir / "hand_meta.json").write_text(json.dumps(hand_meta, indent=2))
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2))

    t0 = time.time()
    n_depth = _bake_depth_mp4(frame_data, out_dir / "depth.mp4", fps=fps)
    depth_sec = time.time() - t0

    stats = {
        "T": int(T),
        "max_hands": int(max_hands),
        "fps": round(float(fps), 3),
        "scene_id": scene_id,
        "W": int(W),
        "H": int(H),
        "depth_mp4_frames": int(n_depth),
        "depth_mp4_sec": round(depth_sec, 2),
    }

    # Provenance
    prov = data.get("retarget_samples_export") or {}
    history = prov.get("history") or []
    history.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": str(out_dir.relative_to(fav_dir.parent)),
        **stats,
    })
    data["retarget_samples_export"] = {"history": history}

    # Atomic write pkl (just provenance change)
    tmp = pkl_path.with_suffix(".pkl.gz.tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl_path)

    return "ok", "exported", stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity.pipeline.export_retarget_samples",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--only", default=None, help="Restrict to a single clip id.")
    ap.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS,
                    help=f"Hand-slot count in hand_joints.bin (default {DEFAULT_MAX_HANDS}).")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override fps for scene.json + depth.mp4 (default: "
                         "derived from manifest.start_sec/end_sec, else 15).")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips with retarget_samples_export.history already.")
    ap.add_argument("--force", action="store_true",
                    help="Re-export even if skip_if_done would apply.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not FAV.is_dir():
        print(f"ERROR: FAV={FAV} does not exist", file=sys.stderr)
        return 2

    if args.only:
        clips = [args.only]
    else:
        clips = sorted(d.name for d in FAV.iterdir()
                       if d.is_dir() and (d / "pipeline_result.pkl.gz").exists())

    if not clips:
        print(f"no clips found under {FAV}")
        return 0

    print(f"export_retarget_samples: {len(clips)} clips, max_hands={args.max_hands}, "
          f"skip_if_done={args.skip_if_done}, dry_run={args.dry_run}")
    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, clip in enumerate(clips, 1):
        fav_dir = FAV / clip
        if not fav_dir.is_dir():
            print(f"[{i:>3}/{len(clips)}] - {clip}  fav dir missing")
            n_skip += 1
            continue
        try:
            t0 = time.time()
            status, action, stats = process_clip(
                fav_dir,
                max_hands=args.max_hands, fps=args.fps,
                force=args.force, skip_if_done=args.skip_if_done,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[{i:>3}/{len(clips)}] x {clip}  ERROR: {e}")
            n_fail += 1
            continue
        elapsed = time.time() - t0
        if status == "skip":
            print(f"[{i:>3}/{len(clips)}] - {clip}  SKIP ({action})")
            n_skip += 1
        elif status in ("ok", "dry"):
            T = stats.get("T", 0)
            fps = stats.get("fps", 0)
            print(f"[{i:>3}/{len(clips)}] + {clip}  T={T} fps={fps} ({elapsed:.1f}s)")
            n_ok += 1
        else:
            print(f"[{i:>3}/{len(clips)}] ? {clip}  {status}: {action}")

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min - ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
