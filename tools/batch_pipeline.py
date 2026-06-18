"""
Serial pipeline runner for the multi-host batch workflow.

For every ``favorites/<id>/`` that has:
  - ``frames/`` (Phase 2 batch_download_starred filled)
  - ``manifest.json`` with non-empty ``objects`` (Phase 3 Claude-curated)
  - NO ``pipeline_result.pkl.gz`` yet (or ``--force``)

Spawns one persistent SAM3D worker for the whole batch (~10 GB resident),
then runs the pipeline subprocess per clip. Without this worker Phase D-sam3d
would silent-skip and no SAM3D meshes would be produced.

Why batch-level (not per-clip) worker spawn:
- SAM3D cold-load is ~30-60s; with 50+ clips that's 30+ min wasted
- VRAM peak: SAM3D (~10 GB) + pipeline subprocess (~3-5 GB) = ~13-15 GB,
  fits in 16 GB but not in 12 GB. On a 12 GB card use --no-sam3d-worker.
- On A100 with `EGOINFINITY_LOW_VRAM=0`, `the pipeline` spawns the worker at
  startup; this orchestrator's `_spawn_sam3d_worker` is harmless because
  it checks for an existing socket first.

Was historically named ``batch_pipeline_4070ti.py`` until the multi-host
refactor; the rename is intentional — this driver runs identically on
both 4070 Ti (`EGOINFINITY_LOW_VRAM=1`) and A100 (`EGOINFINITY_LOW_VRAM=0`).

Usage::

    python -m tools.batch_pipeline [--limit N] [--force] [--dry-run]
                                   [--no-sam3d-worker]
                                   [--sam3d-preset fast|balanced|quality]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Worker lifecycle lives in egoinfinity/pipeline/sam3d_runner.py (S2 refactor).
# Re-export the names here for backward compatibility with any external
# caller that still does `from tools.batch_pipeline import _spawn_sam3d_worker`.
from egoinfinity.pipeline.sam3d_runner import (   # noqa: E402, F401
    SAM3D_WORKER_SOCKET,
    spawn_sam3d_worker as _spawn_sam3d_worker,
    kill_sam3d_worker as _kill_sam3d_worker,
)

# Default favorites cache location. Resolution order:
#   1. ACTION100M_CACHE env var (canonical override)
#   2. <repo>/cache/favorites/
_CACHE_ROOT = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO_ROOT / "cache")))
FAVORITES_DIR = _CACHE_ROOT / "favorites"
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "exo_pipeline.py"

PYTHON = sys.executable


def _writeback_pkl_state(fdir: Path) -> bool:
    """After a successful clip run, record the produced pkl's window in the
    manifest's pkl_* fields (pkl_start_sec / pkl_end_sec / pkl_n_frames /
    pkl_processed_at).

    Uses the manifest's effective range (trim_*_sec, fall back to
    start/end_sec) as the recorded pkl window — pipeline always processes
    the whole frames/ dir which was extracted at that window.

    Returns True on success, False if anything went wrong (non-fatal).
    """
    manifest = fdir / 'manifest.json'
    pkl = fdir / 'pipeline_result.pkl.gz'
    frames_dir = fdir / 'frames'
    if not manifest.is_file() or not pkl.is_file():
        return False
    try:
        m = json.loads(manifest.read_text())
        ts = m.get('trim_start_sec')
        te = m.get('trim_end_sec')
        if ts is None: ts = m.get('start_sec')
        if te is None: te = m.get('end_sec')
        n_frames = len(list(frames_dir.glob('*.jpg'))) if frames_dir.is_dir() else m.get('n_frames')
        m['pkl_start_sec']    = float(ts) if ts is not None else None
        m['pkl_end_sec']      = float(te) if te is not None else None
        m['pkl_n_frames']     = int(n_frames) if n_frames is not None else None
        m['pkl_processed_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        manifest.write_text(json.dumps(m, indent=2))
        return True
    except Exception as e:
        print(f"  WARN — pkl writeback failed for {fdir.name}: {type(e).__name__}: {e}", flush=True)
        return False


def _eligible(fdir: Path, force: bool) -> tuple[bool, str]:
    """Decide whether to (re)run pipeline for this favorite dir."""
    frames_dir = fdir / 'frames'
    manifest = fdir / 'manifest.json'
    pkl = fdir / 'pipeline_result.pkl.gz'
    if not frames_dir.is_dir() or not any(frames_dir.glob('*.jpg')):
        return False, 'no frames/'
    if not manifest.is_file():
        return False, 'no manifest.json'
    try:
        m = json.loads(manifest.read_text())
    except Exception:
        return False, 'manifest.json invalid'
    if not (m.get('objects') or []):
        return False, 'objects empty (run curate_objects_claude first)'
    if pkl.is_file() and not force:
        return False, 'already processed (use --force to redo)'
    return True, ''


def _run_one(fdir: Path, sam3d_preset: str, log_dir: Path,
             sam3d_socket: str | None = None) -> tuple[bool, float, str]:
    """Run pipeline for one favorite. Returns (ok, elapsed_s, log_path).

    ``sam3d_socket`` is the path of a live SAM3D worker socket (set by
    main() when the batch-level worker is running). Pipeline subprocess
    inherits it via env so ``sam3d_client.worker_available()`` returns
    True and Phase D-sam3d actually runs instead of silent-skipping.
    """
    frames_dir = fdir / 'frames'
    manifest = fdir / 'manifest.json'
    log_path = log_dir / f"{fdir.name}.log"

    # Post-pull (origin/main e129bcd+), exo_pipeline.py no longer takes
    # --enable_sam3d / --sam3d_preset; SAM3D is wired in unconditionally
    # and gated by SAM3D_WORKER_SOCKET env. The sam3d_preset arg here is
    # kept for CLI back-compat but is currently a no-op.
    _ = sam3d_preset
    cmd = [
        PYTHON, str(PIPELINE_SCRIPT),
        '--frames_dir', str(frames_dir),
        '--cache-dir', str(fdir),
        '--manifest_path', str(manifest),
        '--no-viser',
    ]
    env = os.environ.copy()
    env.setdefault('EGOINFINITY_LOW_VRAM', '1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    if sam3d_socket:
        env['SAM3D_WORKER_SOCKET'] = sam3d_socket

    t0 = time.time()
    with open(log_path, 'w') as logf:
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.write(f"# env: EGOINFINITY_LOW_VRAM={env['EGOINFINITY_LOW_VRAM']}, "
                   f"SAM3D_WORKER_SOCKET={env.get('SAM3D_WORKER_SOCKET', '(unset)')}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return proc.returncode == 0, elapsed, str(log_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most N favorites in this run')
    ap.add_argument('--force', action='store_true',
                    help='Reprocess even if pipeline_result.pkl.gz exists')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--sam3d-preset', default='fast',
                    choices=['fast', 'balanced', 'quality'])
    ap.add_argument('--no-sam3d-worker', action='store_true',
                    help='Skip the batch-level SAM3D worker spawn. '
                         'Phase D-sam3d will silent-skip (no meshes). '
                         'Use this if your card has < 13 GB VRAM.')
    ap.add_argument('--only', default=None,
                    help='Comma-separated list of clip ids (favorite dir '
                         'names) to limit processing to. Single arg avoids '
                         'argparse confusing `-`-prefixed ids with flags. '
                         'Example: `--only -8WOMg810tk_48.4_60.3,-5a3oTSdP3U_168.6_173.4`')
    ap.add_argument('--favorites-dir', default=str(FAVORITES_DIR))
    ap.add_argument('--log-dir', default=str(REPO_ROOT / 'tools' / '_batch_logs'))
    args = ap.parse_args()

    fav_dir = Path(args.favorites_dir)
    if not fav_dir.is_dir():
        print(f"favorites dir not found: {fav_dir}")
        sys.exit(1)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    only_set = (set(s.strip() for s in args.only.split(',') if s.strip())
                if args.only else None)
    work: list[tuple[Path, str]] = []
    for sub in sorted(fav_dir.iterdir()):
        if not sub.is_dir():
            continue
        if only_set is not None and sub.name not in only_set:
            continue
        ok, reason = _eligible(sub, args.force)
        if ok:
            work.append((sub, ''))
        else:
            work.append((sub, reason))

    eligible = [s for s, r in work if r == '']
    if not eligible:
        print("Nothing eligible to process.")
        for sub, reason in work:
            print(f"  · {sub.name:60s} {reason}")
        return
    if args.limit:
        eligible = eligible[:args.limit]

    print(f"Eligible: {len(eligible)} clip(s)")
    for sub, reason in work:
        marker = '+' if reason == '' else '·'
        print(f"  {marker} {sub.name:60s} {reason}")
    print()

    if args.dry_run:
        return

    # ── Spawn batch-level SAM3D worker so Phase D-sam3d actually runs ──
    sam3d_proc = None
    sam3d_socket = None
    if not args.no_sam3d_worker:
        try:
            sam3d_proc, _ready_msg = _spawn_sam3d_worker(
                SAM3D_WORKER_SOCKET, log_dir / "_sam3d_worker.log")
            sam3d_socket = SAM3D_WORKER_SOCKET
        except Exception as e:
            print(f"[sam3d] WARN — worker spawn failed: {e}", flush=True)
            print(f"[sam3d] Phase D-sam3d will silent-skip on every clip. "
                  f"Pass --no-sam3d-worker to suppress this warning.", flush=True)
            sam3d_proc = None
            sam3d_socket = None
    else:
        print("[sam3d] --no-sam3d-worker — Phase D-sam3d will silent-skip",
              flush=True)

    counts = {'ok': 0, 'fail': 0}
    times: list[float] = []
    t0_all = time.time()
    try:
        for i, fdir in enumerate(eligible, 1):
            eta = ''
            if times:
                avg = sum(times) / len(times)
                remaining = avg * (len(eligible) - i + 1)
                eta = f"  (avg {avg:.0f}s/clip, ETA {remaining/60:.1f}min)"
            print(f"[{i}/{len(eligible)}] {fdir.name}{eta}", flush=True)
            try:
                ok, elapsed, log_path = _run_one(
                    fdir, args.sam3d_preset, log_dir,
                    sam3d_socket=sam3d_socket)
            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                ok, elapsed, log_path = False, 0.0, str(e)
            times.append(elapsed)
            if ok:
                counts['ok'] += 1
                print(f"    ✓ {elapsed:.1f}s  log: {log_path}")
                _writeback_pkl_state(fdir)
            else:
                counts['fail'] += 1
                print(f"    ✗ FAILED ({elapsed:.1f}s)  log: {log_path}")
    finally:
        # Always release SAM3D VRAM, even on KeyboardInterrupt or exception
        if sam3d_proc is not None:
            print("[sam3d] terminating worker ...", flush=True)
            _kill_sam3d_worker(sam3d_proc, SAM3D_WORKER_SOCKET)

    total = time.time() - t0_all
    print()
    print(f"Done in {total/60:.1f} min — ok: {counts['ok']}  fail: {counts['fail']}")


if __name__ == "__main__":
    main()
