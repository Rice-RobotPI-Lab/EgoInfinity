"""Flow3R-fused depth refinement (depth backbone for the canonical pipeline).

Replaces per-frame depth_png in pipeline_result.pkl.gz with Flow3R-derived
metric depth, using MoGe-2's depth as the metric scale anchor.

Algorithm (matches dev's flow3r_experiments/batch_flow3r.py + swap step):

  1. Load pkl + decode frames' img_rgb (T, H, W, 3) + depth_png (T, H, W) m
  2. Build dynamic union mask from sam3_obj_data.mask_packed across oids
  3. Pick K <= max-frames frames (uniform sampling for VRAM budget)
  4. Flow3R forward on those K frames -> local_points + conf (scale-ambiguous)
  5. Fit global scale s = median(D_moge / D_flow3r) on background pixels
     (where Flow3R conf > 0.5 and both depths are in [0.05, 50] m)
  6. Upsample Flow3R depth * s to (T, H, W) full resolution
     (frames not sampled have conf=0 -> fall back to MoGe-2)
  7. Fuse: use Flow3R where conf > 0.05, otherwise MoGe-2
  8. Re-encode each frame's fused depth as uint16 PNG, write back to
     frame_data[t]['depth_png']
  9. Record provenance under data['flow3r_depth_refresh']['history']

Why Flow3R + MoGe-2 hybrid: MoGe-2 is metric but has per-frame depth jitter;
Flow3R is temporally stable but scale-ambiguous. Combining the two gives a
metric, temporally-stable depth video. See dev's
``flow3r_experiments/validate_flow3r_swap.py`` -- bg depth jitter
typically drops by ~50% vs MoGe-2 alone.

Usage::

    # opt-in via config (defaults.yaml has enabled: false)
    python -m egoinfinity.pipeline.flow3r_depth --only=CLIP_ID
    python -m egoinfinity.pipeline.flow3r_depth --skip-if-done
    python -m egoinfinity.pipeline.flow3r_depth --force                # re-fuse
    python -m egoinfinity.pipeline.flow3r_depth --max-frames=60        # 16 GB GPU
    python -m egoinfinity.pipeline.flow3r_depth --max-frames=0         # A100 (no cap)

Requires
--------
- Flow3R sibling repo at ``../flow3r/`` (or override via ``FLOW3R_REPO`` env)
  Install with ``pip install -e ../flow3r/``.
- HuggingFace model ``Clara211111/flow3r`` (auto-downloaded on first use).
- ~9-10 GB VRAM at max-frames=60 (A100/H100 fine at no cap).

Idempotency
-----------
Re-running rewrites depth_png from scratch. Use ``--skip-if-done`` to skip
clips where ``data['flow3r_depth_refresh']['history']`` is already set.
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


# ── Path / env setup ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO_ROOT / "cache"))) / "favorites"

# Flow3R sibling repo (clone next to EgoInfinity, default ../flow3r/)
FLOW3R_REPO = Path(os.environ.get(
    "FLOW3R_REPO", str(REPO_ROOT.parent / "flow3r")))


# ── Algorithm constants (from dev's batch_flow3r.py) ────────────────────────
PIXEL_LIMIT = 255_000     # Flow3R working res (matches gradio default)
SCALE_GATE = 0.5          # high gate for scale fitting (clean pixels only)
FALLBACK_GATE = 0.05      # low gate for "trust Flow3R vs fall back to MoGe-2"

# 0 = no cap (full K=T Flow3R). 16 GB cards: <=60. A100/H100: 0.
DEFAULT_MAX_FRAMES = int(os.environ.get("FLOW3R_MAX_FRAMES", "60"))

# Reduce CUDA fragmentation across clips
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── pkl decoders ────────────────────────────────────────────────────────────
def _load_rgb_from_pkl(data: dict) -> np.ndarray:
    """Decode all frames' img_rgb JPEG -> (T, H, W, 3) uint8 RGB."""
    fd = data["frame_data"]
    n = len(fd)
    img0 = cv2.imdecode(np.frombuffer(fd[0]["img_rgb"], np.uint8), cv2.IMREAD_COLOR)
    H, W = img0.shape[:2]
    out = np.empty((n, H, W, 3), dtype=np.uint8)
    out[0] = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    for t in range(1, n):
        im = cv2.imdecode(np.frombuffer(fd[t]["img_rgb"], np.uint8), cv2.IMREAD_COLOR)
        out[t] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return out


def _load_moge_depth_from_pkl(data: dict) -> np.ndarray:
    """Decode all frames' depth_png -> (T, H, W) float32 meters."""
    from scripts.pipeline_utils import decode_depth_png
    fd = data["frame_data"]
    T = len(fd)
    depth0 = decode_depth_png(fd[0]["depth_png"])
    H, W = depth0.shape
    out = np.empty((T, H, W), dtype=np.float32)
    out[0] = depth0
    for t in range(1, T):
        out[t] = decode_depth_png(fd[t]["depth_png"])
    return out


def _load_dynamic_union_from_pkl(data: dict) -> np.ndarray | None:
    """Build (T, H, W) bool union of all per-oid masks for dynamic regions.

    Returns None if no per-frame masks are stored in the pkl.
    """
    fd = data["frame_data"]
    T = len(fd)
    union = None
    for t, frame in enumerate(fd):
        sam3 = frame.get("sam3_obj_data")
        if not sam3:
            continue
        for oid_data in sam3.values() if isinstance(sam3, dict) else sam3:
            if not isinstance(oid_data, dict):
                continue
            mp = oid_data.get("mask_packed")
            ms = oid_data.get("mask_shape")
            if mp is None or ms is None:
                continue
            H, W = int(ms[0]), int(ms[1])
            if union is None:
                union = np.zeros((T, H, W), dtype=bool)
            n_total = H * W
            bits = np.unpackbits(np.frombuffer(mp, dtype=np.uint8))
            if bits.size < n_total:
                continue
            mask = bits[:n_total].reshape(H, W).astype(bool)
            union[t] |= mask
    return union


# ── Flow3R model + forward (lazy import; needs sibling repo on sys.path) ────
def _make_flow3r_model(device: str = "cuda"):
    if not FLOW3R_REPO.exists():
        raise RuntimeError(
            f"Flow3R sibling repo not found at {FLOW3R_REPO}. "
            f"Clone https://github.com/CVMI-Lab/Flow3R next to EgoInfinity "
            f"and `pip install -e {FLOW3R_REPO}`, or set FLOW3R_REPO env var.")
    sys.path.insert(0, str(FLOW3R_REPO))
    import torch
    from flow3r.models.flow3r import Flow3r
    from huggingface_hub import hf_hub_download

    print("[flow3r] download/locate checkpoint ...", flush=True)
    ckpt = hf_hub_download(repo_id="Clara211111/flow3r", filename="flow3r.bin")
    model = Flow3r()
    state = torch.load(ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _resize_for_flow3r(rgb: np.ndarray, pixel_limit: int):
    """Resize RGB (N, H, W, 3) to a 14-multiple resolution under pixel_limit.

    Returns (tensor of (N, 3, TH, TW), (TH, TW)).
    """
    import torch
    H0, W0 = rgb.shape[1], rgb.shape[2]
    scale = math.sqrt(pixel_limit / (W0 * H0)) if W0 * H0 > 0 else 1.0
    Wt, Ht = W0 * scale, H0 * scale
    k = round(Wt / 14); m = round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    out = np.empty((rgb.shape[0], 3, TH, TW), dtype=np.float32)
    for i, im in enumerate(rgb):
        r = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
        out[i] = r.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(out), (TH, TW)


def _run_flow3r(model, rgb_uint8: np.ndarray, device: str = "cuda"):
    """Forward Flow3R on (K, H, W, 3) RGB. Returns (depth, conf, TH, TW, dt, peak_gib).

    depth (K, TH, TW) float32 in Flow3R-units (NOT metric).
    conf  (K, TH, TW) float32 in [0, 1].
    """
    import torch
    from flow3r.utils.geometry import depth_edge

    imgs, (TH, TW) = _resize_for_flow3r(rgb_uint8, PIXEL_LIMIT)
    imgs = imgs.to(device)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = model(imgs[None])
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    local = pred["local_points"][0]                       # (N, TH, TW, 3)
    conf  = torch.sigmoid(pred["conf"][0]).squeeze(-1)    # (N, TH, TW)
    edge  = depth_edge(local[..., 2], rtol=0.03)
    conf[edge] = 0.0

    depth = local[..., 2].float().cpu().numpy()
    conf_np = conf.float().cpu().numpy()
    del pred, imgs, local, conf
    torch.cuda.empty_cache()
    return depth, conf_np, TH, TW, dt, peak


# ── Fuse + write-back ───────────────────────────────────────────────────────
def _fuse_depth(moge_depth: np.ndarray,
                flow_depth_K: np.ndarray, flow_conf_K: np.ndarray,
                dyn_mask: np.ndarray | None,
                flow3r_frame_idxs: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-clip fusion of MoGe-2 + Flow3R depth. Returns (fused_meters, stats)."""
    T, H, W = moge_depth.shape
    K, TH, TW = flow_depth_K.shape

    if dyn_mask is None:
        dyn_mask = np.zeros_like(moge_depth, dtype=bool)
    if dyn_mask.shape != moge_depth.shape:
        # Try to resize bool mask to MoGe depth resolution
        if dyn_mask.shape[0] == T:
            new = np.zeros_like(moge_depth, dtype=bool)
            for t in range(T):
                new[t] = cv2.resize(dyn_mask[t].astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST) > 0
            dyn_mask = new
        else:
            raise RuntimeError(
                f"dyn_mask frame count {dyn_mask.shape[0]} != depth T={T}")

    # MoGe + dyn at Flow3R working resolution, on sampled frames
    moge_low_K = np.empty((K, TH, TW), dtype=np.float32)
    dyn_low_K  = np.empty((K, TH, TW), dtype=bool)
    for k, t in enumerate(flow3r_frame_idxs):
        moge_low_K[k] = cv2.resize(moge_depth[t], (TW, TH), interpolation=cv2.INTER_LINEAR)
        dyn_low_K[k]  = cv2.resize(dyn_mask[t].astype(np.uint8), (TW, TH),
                                   interpolation=cv2.INTER_NEAREST) > 0
    bg_low_K = ~dyn_low_K

    # Fit global scale via background pixels
    valid = (bg_low_K
             & (flow_depth_K > 0.05) & (flow_depth_K < 50.0)
             & (moge_low_K > 0.05) & (moge_low_K < 50.0)
             & (flow_conf_K > SCALE_GATE))
    n_used = int(valid.sum())
    if n_used < 1000:
        valid = bg_low_K & (flow_depth_K > 0.05) & (flow_depth_K < 50.0) \
              & (moge_low_K > 0.05) & (moge_low_K < 50.0)
        n_used = int(valid.sum())
        if n_used < 100:
            raise RuntimeError(f"too few bg pixels to fit scale: {n_used}")
    ratio = moge_low_K[valid] / flow_depth_K[valid]
    s_global = float(np.median(ratio))

    # Per-frame scale (diag only)
    pf = []
    for k in range(K):
        v = (bg_low_K[k] & (flow_depth_K[k] > 0.05) & (flow_depth_K[k] < 50.0)
             & (moge_low_K[k] > 0.05) & (moge_low_K[k] < 50.0)
             & (flow_conf_K[k] > SCALE_GATE))
        if v.sum() < 200:
            v = (bg_low_K[k] & (flow_depth_K[k] > 0.05) & (flow_depth_K[k] < 50.0)
                 & (moge_low_K[k] > 0.05) & (moge_low_K[k] < 50.0)
                 & (flow_conf_K[k] > FALLBACK_GATE))
        if v.sum() < 200:
            pf.append(np.nan)
        else:
            pf.append(float(np.median(moge_low_K[k][v] / flow_depth_K[k][v])))
    pf_arr = np.array(pf, dtype=np.float32)

    # Upsample Flow3R to full res, time-axis sparse (unsampled t -> conf=0)
    flow_depth_full = np.zeros((T, H, W), dtype=np.float32)
    flow_conf_full  = np.zeros((T, H, W), dtype=np.float32)
    for k, t in enumerate(flow3r_frame_idxs):
        flow_depth_full[t] = cv2.resize(flow_depth_K[k] * s_global, (W, H),
                                        interpolation=cv2.INTER_LINEAR)
        flow_conf_full[t]  = cv2.resize(flow_conf_K[k], (W, H),
                                        interpolation=cv2.INTER_LINEAR)

    # MoGe-2 fallback where Flow3R conf low or invalid
    fallback_mask = (flow_conf_full < FALLBACK_GATE) | (flow_depth_full <= 0.05)
    fused = np.where(fallback_mask, moge_depth, flow_depth_full)
    fb_frac = fallback_mask.mean(axis=(1, 2)).astype(np.float32)

    stats = {
        "s_global": s_global,
        "n_bg_pixels_used": n_used,
        "per_frame_scale_mean": float(np.nanmean(pf_arr)) if np.isfinite(pf_arr).any() else float("nan"),
        "per_frame_consistency_pct": (float(np.nanstd(pf_arr) / np.nanmean(pf_arr) * 100)
                                       if np.isfinite(np.nanmean(pf_arr)) and np.nanmean(pf_arr) > 0
                                       else float("nan")),
        "moge_fallback_pct_mean": float(fb_frac.mean() * 100),
        "K_sampled_frames": K,
        "working_res": [int(TH), int(TW)],
    }
    return fused, stats


# ── Per-clip driver ─────────────────────────────────────────────────────────
def process_clip(fav_dir: Path, model, *, max_frames: int, force: bool,
                 skip_if_done: bool, dry_run: bool) -> tuple[str, str, dict]:
    """Process one clip. Returns (status, action, stats)."""
    from scripts.pipeline_utils import encode_depth_png

    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", {}

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Skip if already done (unless --force)
    if skip_if_done and not force:
        prov = data.get("flow3r_depth_refresh") or {}
        if prov.get("history"):
            return "skip", "already fused", {}

    # Decode pkl inputs
    rgb = _load_rgb_from_pkl(data)
    moge_depth = _load_moge_depth_from_pkl(data)
    if rgb.shape[0] != moge_depth.shape[0]:
        return "skip", f"rgb frames {rgb.shape[0]} != depth {moge_depth.shape[0]}", {}
    if rgb.shape[1:3] != moge_depth.shape[1:3]:
        # Resize rgb to depth res (rare; pipeline normally keeps them aligned)
        T, H, W = moge_depth.shape
        rgb = np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                        for im in rgb], axis=0)

    dyn_mask = _load_dynamic_union_from_pkl(data)
    T = moge_depth.shape[0]

    # Sample frames for Flow3R (VRAM cap)
    if max_frames <= 0 or T <= max_frames:
        flow3r_frame_idxs = np.arange(T)
    else:
        flow3r_frame_idxs = np.linspace(0, T - 1, max_frames).round().astype(int)
    rgb_for_flow3r = rgb[flow3r_frame_idxs]

    # Forward
    flow_depth_K, flow_conf_K, TH, TW, dt, peak = _run_flow3r(model, rgb_for_flow3r)

    # Fuse
    fused_depth, stats = _fuse_depth(moge_depth, flow_depth_K, flow_conf_K,
                                     dyn_mask, flow3r_frame_idxs)
    stats["forward_sec"] = round(float(dt), 2)
    stats["peak_vram_gib"] = round(float(peak), 2)
    stats["flow3r_frame_idxs"] = flow3r_frame_idxs.tolist()

    if dry_run:
        return "dry", "would write", stats

    # Re-encode + write back
    for t in range(T):
        data["frame_data"][t]["depth_png"] = encode_depth_png(fused_depth[t])

    # Provenance
    prov = data.get("flow3r_depth_refresh") or {}
    history = prov.get("history") or []
    history.append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "s_global": round(stats["s_global"], 6),
        "n_bg_pixels_used": stats["n_bg_pixels_used"],
        "moge_fallback_pct_mean": round(stats["moge_fallback_pct_mean"], 2),
        "per_frame_consistency_pct": round(stats["per_frame_consistency_pct"], 3),
        "K_sampled_frames": stats["K_sampled_frames"],
        "T_total_frames": int(T),
        "working_res": stats["working_res"],
        "max_frames_arg": int(max_frames),
        "forward_sec": stats["forward_sec"],
        "peak_vram_gib": stats["peak_vram_gib"],
    })
    data["flow3r_depth_refresh"] = {"history": history}

    # Atomic write
    tmp = pkl_path.with_suffix(".pkl.gz.tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl_path)

    return "ok", "fused", stats


# ── CLI entry ───────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity.pipeline.flow3r_depth",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--only", default=None,
                    help="Restrict to a single clip id.")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="Skip clips that already have a flow3r_depth_refresh "
                         "history entry.")
    ap.add_argument("--force", action="store_true",
                    help="Re-fuse even if skip_if_done would apply.")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help=f"VRAM cap (default {DEFAULT_MAX_FRAMES}; "
                         f"0 = no cap, recommended on A100/H100).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run forward + fuse; do not write pkl.")
    args = ap.parse_args(argv)

    if not FAV.is_dir():
        print(f"ERROR: FAV={FAV} does not exist (set ACTION100M_CACHE)",
              file=sys.stderr)
        return 2

    if args.only:
        clips = [args.only]
    else:
        clips = sorted(d.name for d in FAV.iterdir()
                       if d.is_dir() and (d / "pipeline_result.pkl.gz").exists())

    if not clips:
        print(f"no clips found under {FAV}")
        return 0

    print(f"flow3r_depth: {len(clips)} clips, max_frames={args.max_frames}, "
          f"skip_if_done={args.skip_if_done}, dry_run={args.dry_run}")

    model = None
    t_start = time.time()
    n_ok = n_skip = n_fail = 0

    for i, clip in enumerate(clips, 1):
        fav_dir = FAV / clip
        if not fav_dir.is_dir():
            print(f"[{i:>3}/{len(clips)}] - {clip}  fav dir missing")
            n_skip += 1
            continue
        try:
            # Lazy-load model on first clip that needs it
            if model is None:
                model = _make_flow3r_model()
            t0 = time.time()
            status, action, stats = process_clip(
                fav_dir, model,
                max_frames=args.max_frames,
                force=args.force,
                skip_if_done=args.skip_if_done,
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
        elif status == "ok":
            s = stats.get("s_global", 0)
            fb = stats.get("moge_fallback_pct_mean", 0)
            print(f"[{i:>3}/{len(clips)}] + {clip}  "
                  f"s={s:.3f}  fallback={fb:.1f}%  ({elapsed:.1f}s)")
            n_ok += 1
        else:
            print(f"[{i:>3}/{len(clips)}] ? {clip}  {status}: {action}")

    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f} min - ok: {n_ok}  skip: {n_skip}  fail: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
