"""
Persistent SAM3D worker spawn / kill — proper module home.

Was previously buried inside ``tools/batch_pipeline_4070ti.py`` (now
``tools/batch_pipeline.py``); ``tools/rerun/refresh_sam3d_meshes.py`` had to
reverse-import a tool to reuse it, which is a smell.  Moving the
worker-lifecycle code into the importable package fixes that.

API::

    SAM3D_WORKER_SOCKET                       # default per-user socket path
    proc, ready_msg = spawn_sam3d_worker(
        socket_path,
        log_path,
        timeout_s=None,                       # honours $SAM3D_SPAWN_TIMEOUT (default 240)
    )
    kill_sam3d_worker(proc, socket_path)      # best-effort, idempotent

Environment knobs read inside ``spawn_sam3d_worker``:

  SAM3D_PYTHON        absolute path to the sam3d-objects env's python
                      (default: ~/miniconda3/envs/sam3d-objects/bin/python
                       or /opt/conda/envs/sam3d-objects/bin/python)
  SAM3D_REPO          absolute path to the sam-3d-objects checkout
                      (default: sibling of the EgoInfinity repo)
  SAM3D_SPAWN_TIMEOUT cold-load deadline in seconds (default 240; bump
                      on slow shared FS like Delta Lustre)
  HF_HOME             passed through to the subprocess env
  CUDA_VISIBLE_DEVICES same

This module is a pure-Python helper — no torch, no GPU access at
import time — so it's cheap to depend on from any tool.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
SAM3D_WORKER_SCRIPT = _REPO_ROOT / "scripts" / "sam3d_worker.py"

# Shared per-user socket path so workers spawned by either the orchestrator
# or a long-running curation app land on the same address.
SAM3D_WORKER_SOCKET = f"/tmp/egoinfinity_sam3d_{os.environ.get('USER', 'user')}.sock"


def _detect_sam3d_python() -> str:
    """Where's the sam3d-objects conda env's python? Honour env override."""
    p = os.environ.get("SAM3D_PYTHON", "").strip()
    if p and os.path.isfile(p):
        return p
    for cand in (
        Path.home() / "miniconda3" / "envs" / "sam3d-objects" / "bin" / "python",
        Path("/opt/conda/envs/sam3d-objects/bin/python"),
    ):
        if cand.is_file():
            return str(cand)
    return ""


def _detect_sam3d_repo() -> str:
    """Where's the sam-3d-objects source repo? Honour env override."""
    p = os.environ.get("SAM3D_REPO", "").strip()
    if p and os.path.isdir(p):
        return p
    for cand in (
        _REPO_ROOT.parent / "sam-3d-objects",
        Path.home() / "Research" / "EgoInfinity" / "sam-3d-objects",
    ):
        if cand.is_dir():
            return str(cand)
    return ""


def spawn_sam3d_worker(socket_path: str, log_path: Path,
                       timeout_s: Optional[int] = None) -> Tuple[subprocess.Popen, str]:
    """Spawn the persistent SAM3D worker; block until it writes ``.ready``.

    Returns (proc, ready_msg) on success.  Raises on failure.

    ``timeout_s`` defaults to ``$SAM3D_SPAWN_TIMEOUT`` (240s) — bump on
    slow shared filesystems (e.g. Delta Lustre) where the cold load of
    the SAM3D checkpoints exceeds the default.
    """
    if timeout_s is None:
        timeout_s = int(os.environ.get("SAM3D_SPAWN_TIMEOUT", "240"))
    sam3d_py = _detect_sam3d_python()
    sam3d_repo = _detect_sam3d_repo()
    if not sam3d_py:
        raise RuntimeError(
            "SAM3D_PYTHON not found. Set env SAM3D_PYTHON to the "
            "sam3d-objects conda env's python (e.g. "
            "$HOME/miniconda3/envs/sam3d-objects/bin/python).")
    if not sam3d_repo:
        raise RuntimeError(
            "SAM3D_REPO not found. Set env SAM3D_REPO to the sam-3d-objects "
            "repo path (e.g. ../sam-3d-objects).")
    if not SAM3D_WORKER_SCRIPT.is_file():
        raise RuntimeError(f"sam3d_worker.py missing at {SAM3D_WORKER_SCRIPT}")

    # Clean stale socket
    for p in (socket_path, socket_path + ".ready"):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

    sam3d_bin_dir = os.path.dirname(sam3d_py)
    env = {
        "PATH": f"{sam3d_bin_dir}:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "HF_HOME": os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "CONDA_PREFIX": os.path.dirname(sam3d_bin_dir),
        # Reduce CUDA memory fragmentation — SAM3D allocates / frees ~10 GB
        # per inference call on a 16 GB card; without expandable_segments the
        # second call OOMs even though enough memory is technically free.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    log_f.write(f"\n# === sam3d worker spawn @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_f.flush()

    cmd = [sam3d_py, str(SAM3D_WORKER_SCRIPT),
           "--socket", socket_path,
           "--repo", sam3d_repo,
           "--quality_default", "tier1"]
    print(f"[sam3d] spawning worker (socket={socket_path}) ...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    ready_file = socket_path + ".ready"
    for _ in range(timeout_s * 2):
        if proc.poll() is not None:
            log_f.close()
            raise RuntimeError(
                f"sam3d_worker exited early (code {proc.returncode}); "
                f"see {log_path}")
        if os.path.exists(ready_file):
            break
        time.sleep(0.5)
    if not os.path.exists(ready_file):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        log_f.close()
        raise TimeoutError(f"sam3d_worker did not become ready in {timeout_s}s")

    elapsed = time.time() - t0
    try:
        with open(ready_file) as f:
            st = json.load(f)
        msg = (f"pid={st.get('pid')}, model_load={st.get('load_time_s', 0)}s, "
               f"total spawn {elapsed:.1f}s")
    except Exception:
        msg = f"spawn {elapsed:.1f}s"
    print(f"[sam3d] worker ready — {msg}", flush=True)
    return proc, msg


def kill_sam3d_worker(proc: Optional[subprocess.Popen], socket_path: str) -> None:
    """Best-effort cleanup; safe to call multiple times.

    Terminates the process (SIGTERM, then SIGKILL after 5s) and removes
    the socket + .ready file.  Swallows all errors so it can be used in
    `finally` blocks without masking the real exception.
    """
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        except Exception:
            pass
    for p in (socket_path, socket_path + ".ready"):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass
