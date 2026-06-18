"""`egoinfinity import` — convert a legacy pkl into the new artifact layout.

For clips processed by dev's batch_pipeline (which produced
``favorites/<clip>/pipeline_result.pkl.gz`` + sibling files like
``sam3_meshes/``, ``frames/``), this command builds the corresponding
``<artifacts_dir>/<clip_id>/`` layout so the new pipeline can resume on
top of the existing pkl.

What gets carried over:
  - The pkl itself (symlinked or copied)
  - frames/ directory (if present)
  - sam3_meshes/ directory (if present)
  - manifest.json (if present)
  - state.json marked with all "stages" up to and including phase1 as
    done — so subsequent ``egoinfinity process`` only runs post-tracking.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from ..core.artifacts import ArtifactStore
from ..core.state import ClipState


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity import")
    ap.add_argument("source", type=Path,
                    help="Path to legacy clip dir OR pipeline_result.pkl.gz file.")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Destination artifact dir (will be created).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking (default: symlink).")
    ap.add_argument("--phase1-done", action="store_true", default=True,
                    help="Mark phase1 stage as done in state.json (default: yes).")
    args = ap.parse_args(argv)

    source = args.source.expanduser().resolve()
    if source.is_file() and source.name.endswith(".pkl.gz"):
        clip_dir = source.parent
        pkl = source
    elif source.is_dir():
        clip_dir = source
        pkl = source / "pipeline_result.pkl.gz"
        if not pkl.exists():
            print(f"ERROR: no pipeline_result.pkl.gz in {clip_dir}", file=sys.stderr)
            return 2
    else:
        print(f"ERROR: {source} not found", file=sys.stderr)
        return 2

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    clip_id = output.name

    # Carry over pkl
    target_pkl = output / "pipeline_result.pkl.gz"
    _carry(pkl, target_pkl, copy=args.copy)

    # Carry over auxiliary dirs/files if present
    for aux in ("frames", "sam3_meshes", "manifest.json"):
        src = clip_dir / aux
        if src.exists():
            _carry(src, output / aux, copy=args.copy)

    # If no manifest, synthesize a minimal one
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "clip_id": clip_id,
            "imported_from": str(clip_dir),
            "imported_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Mark phase1 done (no actual files in phase1/, but pkl exists)
    store = ArtifactStore(output.parent, clip_id)
    state = ClipState(store)
    if args.phase1_done:
        state.mark_done(
            "phase1",
            backend="default",
            args={"imported": True},
            duration_s=0.0,
            outputs={"pipeline_result.pkl.gz": store.hash("phase1", "pipeline_result.pkl.gz")
                     if (output / "phase1" / "pipeline_result.pkl.gz").exists() else ""},
        )

        # Also bridge: link the pkl from clip root → phase1/ stage dir
        stage_pkl = store.path("phase1", "pipeline_result.pkl.gz")
        if not stage_pkl.exists() and target_pkl.exists():
            stage_pkl.parent.mkdir(parents=True, exist_ok=True)
            stage_pkl.symlink_to(target_pkl.resolve())

    # Mark extract_frames done if frames/ present
    if (output / "frames").exists():
        state.mark_done(
            "extract_frames",
            backend="default",
            args={"imported": True},
            duration_s=0.0,
            outputs={"frames": store.hash("extract_frames", "frames")},
        )
        # Bridge: extract_frames/frames -> frames/
        bridge = output / "extract_frames" / "frames"
        if not bridge.exists():
            bridge.parent.mkdir(parents=True, exist_ok=True)
            bridge.symlink_to((output / "frames").resolve())

    print(f"imported {source.name} -> {output}")
    print(f"  pkl: {target_pkl}")
    print(f"  state.json marked: extract_frames + phase1 done")
    print("  next: `egoinfinity process {output}` will resume from post-tracking")
    return 0


def _carry(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())
