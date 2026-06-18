"""Post-tracking Stages — table-driven wrappers for the 8 canonical
algorithms in ``egoinfinity.pipeline.post_tracking``.

Each entry in ``POST_TRACK_STAGES`` becomes a Stage class registered
against the global registry. The shared ``_RefreshStage.run()``
subprocess-calls ``python -m <module>`` and synthesizes the argv from
``cli_args`` plus the per-clip ``--only=<CLIP_ID>``.

The tools mutate the per-clip ``pipeline_result.pkl.gz`` in place
(no file-level outputs), so ``is_done`` is decided by ``state.json``
alone (the default ``Stage.is_done`` returns True iff state has the
stage marked done; ``outputs=()`` means no file check).

Resume semantics (replaces the old ``tools/refresh`` CLI):

  - If pose_track_p1 ran but grasp_veto didn't, ``egoinfinity process``
    picks up at grasp_veto automatically.
  - ``--force pose_track_p1 --cascade`` re-runs pose_track_p1 plus
    everything downstream that lists it (transitively) as upstream.

All stages share the favorites-dir layout used by the existing tools:
``<artifacts_dir>/<clip_id>/{pipeline_result.pkl.gz, sam3_meshes/, ...}``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from ..core.registry import register_stage
from ..core.stage import Stage, StageContext


class _RefreshStage(Stage):
    """Generic post-tracking Stage. Subprocess-calls one algorithm module
    via ``python -m <module> <cli_args> --only=<CLIP_ID>``."""

    # Per-subclass attributes (set via the factory below).
    name: ClassVar[str] = ""
    module: ClassVar[str] = ""
    cli_args: ClassVar[tuple[str, ...]] = ()
    upstream: ClassVar[tuple[str, ...]] = ()
    outputs: ClassVar[tuple[str, ...]] = ()   # pkl is mutated in place

    def run(self, ctx: StageContext) -> None:
        clip_root = ctx.artifacts.root
        clip_id = ctx.clip_id

        # Build argv: python -m <module> <cli_args> [config kv -> --flags] --only=<CLIP>
        cmd = [sys.executable, "-m", self.module]
        cmd += list(self.cli_args)
        for k, v in ctx.stage_config.args.items():
            flag = f"--{k.replace('_', '-')}"
            if v is True:
                cmd.append(flag)
            elif v is False:
                continue
            else:
                cmd.extend([flag, str(v)])
        cmd.append(f"--only={clip_id}")

        # Legacy tools assume ``$ACTION100M_CACHE/favorites/<CLIP_ID>/{...}``.
        # The new per-clip artifact dir is flat (clip_root IS the clip dir),
        # so we build a throwaway view via symlink and point the env var at it.
        with tempfile.TemporaryDirectory(prefix="egoinfinity_post_track_view_") as tmp:
            fav = Path(tmp) / "favorites"
            fav.mkdir()
            (fav / clip_id).symlink_to(clip_root.resolve(), target_is_directory=True)

            env = os.environ.copy()
            env["ACTION100M_CACHE"] = tmp

            ctx.log.info("running: %s", " ".join(cmd))
            ret = subprocess.run(cmd, env=env)
            if ret.returncode != 0:
                raise RuntimeError(f"{self.module} failed (exit {ret.returncode})")


# ── Canonical 8-stage post-tracking table ───────────────────────────────────
#
# Order matters: it matches the documented PIPELINE.md §4.4 canonical
# sequence and the dev pkl provenance order observed across reference clips
# (depth_align BEFORE depth_smooth, per 5/28 timestamps).
#
# Each row: (stage_name, backend, module_path, cli_args, upstream)
#
# Add a new post-tracking stage by adding a row. Naming convention: stage
# name == its registry id; the generated class name is auto-derived from
# the stage name (e.g. "pose_track_p1" -> "PoseTrackP1Stage").

POST_TRACK_STAGES: tuple = (
    # name             backend                   module                                                      cli_args               upstream
    # Depth refinement (opt-in; sits between phase1 and pose_track_p1):
    ("flow3r_depth",   "default",                "egoinfinity.pipeline.flow3r_depth",                            ("--skip-if-done",),   ("phase1",)),

    # Canonical 8-stage post-tracking sequence:
    ("pose_track_p1",  "phase_d",                "egoinfinity.pipeline.post_tracking.pose_tracking",             ("--mode", "phase_d"), ("phase1",)),
    ("grasp_veto",     "default",                "egoinfinity.pipeline.post_tracking.grasp_veto",                ("--force",),          ("pose_track_p1",)),
    ("pose_track_p2",  "phase_d_preserve_veto",  "egoinfinity.pipeline.post_tracking.pose_tracking",             ("--mode", "phase_d"), ("grasp_veto",)),
    ("scale_sanity",   "default",                "egoinfinity.pipeline.post_tracking.scale_sanity",              (),                    ("phase1",)),
    # depth_align / depth_smooth are non-idempotent (running on already-aligned
    # / already-smoothed pkl compounds the operation). --skip-if-done makes
    # them check provenance and short-circuit; the runner's resume mechanism
    # plus an explicit --force at the CLI level (mapped through `force: true`
    # in stage_config.args) overrides the skip when re-tuning is intended.
    ("depth_align",    "default",                "egoinfinity.pipeline.post_tracking.depth_align",               ("--skip-if-done",),   ("pose_track_p2",)),
    ("depth_smooth",   "default",                "egoinfinity.pipeline.post_tracking.depth_smooth",              ("--skip-if-done",),   ("depth_align",)),
    ("bake_fp",        "default",                "egoinfinity.pipeline.post_tracking.bake_fp_pose",              ("--force",),          ("depth_smooth",)),
    ("spurious",       "soft",                   "egoinfinity.pipeline.post_tracking.spurious_filter",           ("--mode", "soft"),    ("pose_track_p2",)),

    # Retarget bridge: pkl -> retarget/utils/clip_io.py SamplesSequence format
    # (hand_joints.bin + hand_meta.json + scene.json + depth.mp4). Required
    # whenever --robot is requested; default skip-if-done so resumes are fast.
    ("export_retarget_samples", "default",         "egoinfinity.pipeline.export_retarget_samples",                 ("--skip-if-done",),   ("bake_fp",)),
)


def _camel(snake: str) -> str:
    """`pose_track_p1` -> `PoseTrackP1`."""
    return "".join(part.capitalize() for part in snake.split("_"))


# Generate Stage subclasses dynamically and register them.
for (_name, _backend, _module, _cli_args, _upstream) in POST_TRACK_STAGES:
    _cls = type(
        f"{_camel(_name)}Stage",
        (_RefreshStage,),
        {
            "name": _name,
            "module": _module,
            "cli_args": _cli_args,
            "upstream": _upstream,
        },
    )
    register_stage(_name, backend=_backend)(_cls)
    # Expose at module level for callers that might want explicit imports
    globals()[_cls.__name__] = _cls

del _name, _backend, _module, _cli_args, _upstream, _cls
