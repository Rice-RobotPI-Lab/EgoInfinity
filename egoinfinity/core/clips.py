"""Clip selection — expand CLI targets into a list of clips to process.

A "target" is one of:
  - an existing **clip artifact dir** (contains ``manifest.json``) -> one clip
  - a **batch root** (a dir whose immediate subdirs are clip dirs) -> many clips,
    filtered by ``--only A,B,C``
  - (a video file is handled by the caller as single new-clip mode, not here)

Returns :class:`ClipSpec` records that ``PipelineRunner.run_many`` consumes.
All resolved clips are resume-mode (``manifest=None``): they already have a
manifest.json on disk. New-clip-from-video stays single (see cli/process.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClipSpec:
    clip_id: str
    artifacts_root: Path        # parent dir that contains <clip_id>/
    manifest: dict | None = None  # None = resume an existing on-disk clip


def _is_clip_dir(p: Path) -> bool:
    return p.is_dir() and (p / "manifest.json").exists()


def resolve_clips(targets: list[str], only: str | None = None) -> list[ClipSpec]:
    """Expand targets into clip specs.

    ``only`` (comma-separated clip ids) filters the children of a *batch root*
    only; explicit clip-dir targets are always included.
    """
    only_set = {s.strip() for s in only.split(",") if s.strip()} if only else None
    specs: list[ClipSpec] = []
    seen: set[tuple[str, str]] = set()

    def _add(clip_id: str, root: Path) -> None:
        key = (str(root), clip_id)
        if key not in seen:
            seen.add(key)
            specs.append(ClipSpec(clip_id=clip_id, artifacts_root=root, manifest=None))

    for t in targets:
        p = Path(t).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"target not found: {p}")
        if _is_clip_dir(p):
            # A single existing clip dir.
            _add(p.name, p.parent)
        elif p.is_dir():
            # A batch root: immediate subdirs that look like clip dirs.
            children = [d for d in sorted(p.iterdir()) if _is_clip_dir(d)]
            if not children:
                raise ValueError(
                    f"{p} has no manifest.json and no clip subdirs "
                    f"(a clip dir needs manifest.json). Not a valid target.")
            for d in children:
                if only_set is not None and d.name not in only_set:
                    continue
                _add(d.name, p)
        else:
            # A file (e.g. a video): new-clip mode is single-clip and needs
            # --objects/--start/--end; the caller handles it, not here.
            raise ValueError(
                f"{p} is a file; new-clip-from-video is single-clip mode "
                f"(pass exactly one video, no batching).")

    if only_set is not None:
        missing = only_set - {s.clip_id for s in specs}
        if missing:
            raise ValueError(f"--only ids not found under the given root(s): {sorted(missing)}")
    return specs
