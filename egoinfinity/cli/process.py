"""`egoinfinity process` subcommand.

Two modes:
  1. New clip — pass a video path: an artifact dir is created, manifest
     written, full pipeline runs from scratch.
  2. Resume — pass an existing artifact dir: reads manifest, picks up
     from the first not-done stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..core.config import PipelineConfig
from ..core.runner import PipelineRunner


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity process")
    ap.add_argument(
        "target",
        nargs="+",
        help="One or more targets. Each is a video (new clip, single only), an "
             "existing clip artifact dir (resume), or a batch root whose subdirs "
             "are clip dirs. Multiple targets / a batch root run as a batch.",
    )
    ap.add_argument("--only", default=None,
                    help="Comma-separated clip ids to keep when a target is a "
                         "batch root (ignored for explicit clip-dir targets).")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Stop the batch at the first failing clip "
                         "(default: keep going + report a summary).")
    ap.add_argument("--objects", default="",
                    help="Comma-separated object prompts (new-clip mode).")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start time in seconds (new-clip mode).")
    ap.add_argument("--end", type=float, default=None,
                    help="End time in seconds (new-clip mode).")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--output", type=Path, default=None,
                    help="Artifact dir for the clip (new-clip mode). "
                         "Default: <artifacts_dir>/<clip_id>/.")
    ap.add_argument("--clip-id", default=None,
                    help="Override the clip-id (defaults to video stem).")
    ap.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    ap.add_argument("--set", action="append", default=[],
                    metavar="path=value",
                    help="Config override; can repeat. E.g. --set 'depth.backend=depthanything'")
    ap.add_argument("--force", default=None,
                    help="Comma-separated stage names to force-rerun.")
    ap.add_argument("--cascade", action="store_true",
                    help="With --force: also forget all downstream of forced stages.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Run all stages from scratch (resume.policy=never).")
    ap.add_argument("--robot", default=None,
                    choices=["g1", "franka", "robonaut2", "xlerobot"],
                    help="Enable retarget at the end of the pipeline. Selects "
                         "which robot's ckpt (under retarget/ckpts/<robot>.pt) "
                         "to use. Implies enabling export_retarget_samples + "
                         "retarget stages.")
    # ── Phase-1 component selection + tail-skip (modularization surface) ──
    # These map to the backend registry (EGOINFINITY_<ROLE>_BACKEND, default ==
    # the pre-modularization class) and the existing, production-tested phase
    # gates. They do NOT alter the monolithic Phase-1 control flow.
    ap.add_argument("--depth-backend", default=None,
                    help="Phase-A depth backbone (registry name; default 'moge2'). "
                         "Sets EGOINFINITY_DEPTH_BACKEND for the phase1 subprocess.")
    ap.add_argument("--hand-recon-backend", default=None,
                    help="Phase-B hand reconstructor backend (default 'wilor').")
    ap.add_argument("--no-sam3d", action="store_true",
                    help="Skip Phase D-sam3d (SAM 3D Objects mesh). For <13 GB "
                         "GPUs / multi-host hand-off. (EGOINFINITY_RUN_SAM3D=0)")
    ap.add_argument("--no-track", action="store_true",
                    help="Skip Phase D-track (6DoF pose seed). (EGOINFINITY_RUN_TRACK=0)")
    ap.add_argument("--hand-tta", action="store_true",
                    help="Enable multi-scale YOLO TTA on hand detection "
                         "(EGOINFINITY_HAND_TTA=1; ~3x detector cost).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load + override config
    cfg = PipelineConfig.from_yaml(args.config)
    if args.set:
        cfg = cfg.with_overrides(args.set)
    if args.force:
        cfg.resume.force_stages = [s.strip() for s in args.force.split(",") if s.strip()]
        cfg.resume.cascade_force = args.cascade
    if args.no_resume:
        cfg.resume.policy = "never"
    if args.robot:
        # Enable retarget pipeline tail with the requested robot backend.
        # The runner reads stage.enabled + stage.backend per-stage. We patch
        # both via the in-memory config (no YAML edit needed).
        for st in cfg.pipeline:
            if st.name == "export_retarget_samples":
                st.enabled = True
            if st.name == "retarget":
                st.enabled = True
                st.backend = args.robot

    # Backend selection → env vars consumed by the phase1 subprocess via the
    # registry (egoinfinity.pipeline.backends.get_backend). Default (flag unset)
    # leaves the env unset → registry default → identical class → unchanged.
    import os as _os
    if args.depth_backend:
        _os.environ["EGOINFINITY_DEPTH_BACKEND"] = args.depth_backend
    if args.hand_recon_backend:
        _os.environ["EGOINFINITY_HAND_RECON_BACKEND"] = args.hand_recon_backend
    # Phase-1 tail-skip + TTA → patch the phase1 stage's args (maps to the
    # production-tested EGOINFINITY_RUN_SAM3D / EGOINFINITY_RUN_TRACK / EGOINFINITY_HAND_TTA
    # gates inside egoinfinity/stages/phase1.py). No change to Phase-1 internals.
    if args.no_sam3d or args.no_track or args.hand_tta:
        for st in cfg.pipeline:
            if st.name == "phase1":
                if args.no_sam3d:
                    st.args["run_sam3d"] = False
                if args.no_track:
                    st.args["run_track"] = False
                if args.hand_tta:
                    st.args["hand_tta"] = True

    runner = PipelineRunner(cfg)
    targets = args.target

    # New-clip-from-video is single-clip mode (needs --objects/--start/--end).
    first = Path(targets[0]).expanduser()
    if len(targets) == 1 and first.is_file():
        target = first.resolve()
        clip_id = args.clip_id or target.stem
        artifacts_root = args.output.parent if args.output else cfg.artifacts_dir
        if args.output:
            clip_id = args.output.name
        objects = [s.strip() for s in args.objects.split(",") if s.strip()]
        manifest = {
            "video_uri": str(target),
            "objects": objects,
            "start": args.start,
            "end": args.end,
            "fps": args.fps,
            "clip_id": clip_id,
        }
        return runner.run_clip(
            clip_id, manifest=manifest, artifacts_root=Path(artifacts_root))

    # Otherwise: one or more existing clip dirs / batch root(s).
    from ..core.clips import resolve_clips
    try:
        specs = resolve_clips(targets, args.only)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return runner.run_many(specs, keep_going=not args.fail_fast)
