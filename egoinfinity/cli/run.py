"""`egoinfinity run <stages> <target...>` — force-run a stage subset on one
or more existing clips.

``<stages>`` is one stage or a comma-separated list (DAG / declared order is
applied automatically). Each ``<target>`` is an existing clip dir or a batch
root (see :func:`egoinfinity.core.clips.resolve_clips`).

The named stages are force-run (their state is forgotten first). Each
selected stage's upstream must already be done, OR also selected, OR pulled
in with ``--with-deps``; otherwise the run errors with the missing prereq.

Examples::

    egoinfinity run bake_fp artifacts/CLIP/                 # one stage, one clip
    egoinfinity run grasp_veto,pose_track_p2 artifacts/CLIP/
    egoinfinity run refresh_sam3d clipA clipB               # cross-host SAM3D fill
    egoinfinity run pose_track_p2 ROOT/ --only A,B --with-deps
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..core.clips import resolve_clips
from ..core.config import PipelineConfig
from ..core.runner import PipelineRunner


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity run")
    ap.add_argument("stages",
                    help="Stage name, or comma-separated list "
                         "(e.g. 'pose_track_p1,grasp_veto'). See configs/defaults.yaml.")
    ap.add_argument("target", nargs="+",
                    help="One or more existing clip dirs / batch roots.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated clip ids to keep when a target is a batch root.")
    ap.add_argument("--with-deps", action="store_true",
                    help="Auto-run missing upstream stages instead of erroring.")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Stop the batch at the first failing clip "
                         "(default: keep going + report a summary).")
    ap.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    cfg = PipelineConfig.from_yaml(args.config)
    if args.set:
        cfg = cfg.with_overrides(args.set)

    stage_list = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stage_list:
        print("ERROR: no stage names given.", file=sys.stderr)
        return 2
    unknown = [s for s in stage_list if cfg.stage(s) is None]
    if unknown:
        print(f"ERROR: stage(s) not in config: {unknown}", file=sys.stderr)
        return 2

    try:
        specs = resolve_clips(args.target, args.only)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    runner = PipelineRunner(cfg)
    return runner.run_many(
        specs,
        keep_going=not args.fail_fast,
        stages=stage_list,
        with_deps=args.with_deps,
    )
