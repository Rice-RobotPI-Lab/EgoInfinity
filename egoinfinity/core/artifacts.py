"""ArtifactStore — per-clip artifact directory management.

Layout (Choice C3 — hybrid):

    <artifacts_dir>/<clip_id>/
    ├── manifest.json              # input description (video URI + objects + start/end)
    ├── state.json                 # per-stage runs + provenance hashes (see state.py)
    ├── input/
    │   ├── video.mp4              # symlink/copy of source
    │   └── frames/                # extract_frames output
    ├── <stage>/                   # one per Stage
    │   └── <named artifacts>
    └── pipeline_result.pkl.gz     # FINAL consolidated artifact (back-compat with dev tooling)

Each Stage writes its outputs to its own ``<stage>/`` dir. The runner asks
``ArtifactStore.exists()`` per declared output to decide if a stage is
"done"; ``hash()`` lets state.json detect external edits.

Atomic write: helpers use temp + rename so a crash mid-write doesn't leave
a half-written file pretending the stage finished.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ArtifactStore:
    """Per-clip artifact directory."""

    def __init__(self, artifacts_dir: Path | str, clip_id: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.clip_id = clip_id
        self.root = self.artifacts_dir / clip_id
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────────────

    def stage_dir(self, stage: str) -> Path:
        """Return ``<root>/<stage>/``, creating if missing."""
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: str, name: str) -> Path:
        """Resolve a named artifact within a stage. Does NOT create."""
        return self.stage_dir(stage) / name

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def state_path(self) -> Path:
        return self.root / "state.json"

    def pkl_path(self) -> Path:
        """Legacy consolidated artifact path (kept for back-compat)."""
        return self.root / "pipeline_result.pkl.gz"

    # ── Existence + hash ──────────────────────────────────────────────────

    def exists(self, stage: str, name: str) -> bool:
        return self.path(stage, name).exists()

    def hash(self, stage: str, name: str) -> str:
        """SHA1 hex of artifact file content, or empty string if missing.

        Files larger than 64 MB are hashed in 1 MB chunks. Directories
        (rare; some stages produce dirs like ``frames/``) hash a sorted
        listing of (rel_path, size, mtime_ns) — cheaper than recursing
        bytes but stable across re-runs that produce identical trees.
        """
        p = self.path(stage, name)
        if not p.exists():
            return ""
        if p.is_dir():
            return _hash_directory_meta(p)
        return _hash_file(p)

    # ── Atomic write ──────────────────────────────────────────────────────

    @contextmanager
    def open_write(self, stage: str, name: str, *, binary: bool = False) -> Iterator:
        """Atomic write context: writes go to a temp file, renamed on close.

        Usage::

            with store.open_write("depth", "depth.npz", binary=True) as f:
                np.save(f, arr)
        """
        final = self.path(stage, name)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_suffix(final.suffix + ".tmp")
        mode = "wb" if binary else "w"
        f = tmp.open(mode)
        try:
            yield f
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
        else:
            f.close()
            os.replace(tmp, final)

    def atomic_rename_into(self, stage: str, name: str, source: Path) -> Path:
        """Move ``source`` into the stage dir atomically as ``name``."""
        final = self.path(stage, name)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, final)
        return final

    def copy_into(self, stage: str, name: str, source: Path) -> Path:
        """Copy ``source`` into the stage dir as ``name`` (use for read-only sources)."""
        final = self.path(stage, name)
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, final)
        return final

    # ── Discovery ─────────────────────────────────────────────────────────

    def listdir(self, stage: str) -> list[str]:
        d = self.root / stage
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir())


# ── Hashing helpers ──────────────────────────────────────────────────────

_CHUNK = 1 << 20   # 1 MB

def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_directory_meta(path: Path) -> str:
    """Cheap stable hash over (rel_path, size, mtime_ns) of files in dir."""
    h = hashlib.sha1()
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(path).as_posix()
        st = p.stat()
        h.update(f"{rel}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()
