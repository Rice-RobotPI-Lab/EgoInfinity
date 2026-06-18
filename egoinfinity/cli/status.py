"""`egoinfinity status <artifacts_dir>` — inspect what's done / pending / stale."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.artifacts import ArtifactStore
from ..core.config import PipelineConfig
from ..core.registry import _import_stages_module, _STAGES
from ..core.state import ClipState


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity status")
    ap.add_argument("artifacts_dir", type=Path,
                    help="Path to a clip's artifact directory.")
    ap.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    args = ap.parse_args(argv)

    target = args.artifacts_dir.expanduser().resolve()
    if not (target / "state.json").exists() and not (target / "manifest.json").exists():
        print(f"ERROR: {target} doesn't look like an artifact dir.", file=sys.stderr)
        return 2

    cfg = PipelineConfig.from_yaml(args.config)
    _import_stages_module()
    store = ArtifactStore(target.parent, target.name)
    state = ClipState(store)

    if store.manifest_path().exists():
        manifest = json.loads(store.manifest_path().read_text())
        print(f"Clip: {target.name}")
        print(f"  video: {manifest.get('video_uri', '?')}")
        print(f"  start={manifest.get('start')} end={manifest.get('end')} fps={manifest.get('fps')}")
        print(f"  objects: {manifest.get('objects', [])}")
        print()

    print(f"{'stage':22} {'backend':16} {'enabled':9} {'status':10} {'duration':>10}")
    print("─" * 75)
    for s_cfg in cfg.pipeline:
        info = state.get(s_cfg.name)
        enabled = "yes" if s_cfg.enabled else "no"
        if info is None:
            status = "—"
            dur = ""
        elif info.get("ts_end") is None:
            status = "running?"
            dur = "running"
        else:
            status = "done"
            dur = f"{info.get('duration_s', 0):.1f}s"
        backend = s_cfg.backend
        print(f"{s_cfg.name:22} {backend:16} {enabled:9} {status:10} {dur:>10}")
    return 0
