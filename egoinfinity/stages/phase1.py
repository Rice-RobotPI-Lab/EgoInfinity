"""Phase1Stage — wraps the monolithic Phase-1 detection+tracking pipeline.

Phase 1 (A→E) is implemented as a single subprocess invocation of
``tools.batch_pipeline`` for ONE clip. The output is the consolidated
``pipeline_result.pkl.gz``.

Why one big Stage instead of per-substage (depth, hands, sam3, sam3d, ...)?
The existing ``scripts/exo_pipeline.py`` is a monolithic script — it
spawns model workers once and threads them across substages. Splitting
that would require touching algorithm code, which violates the
"algorithm-frozen" guarantee for this refactor. So Phase 1 stays atomic;
the granular re-iteration happens at Phase-3 stages (refresh tools)
which ARE individually invokable.

Inputs:
  manifest.json                # video_uri + objects (+ optional start/end/fps)
  extract_frames/frames/*.jpg  # OR a "favorites" layout with frames/ at clip root

Outputs:
  pipeline_result.pkl.gz       # the consolidated artifact

Implementation strategy: this stage prepares a per-clip "favorites-style"
layout under the artifact dir (frames/, manifest.json) and invokes
``python -m tools.batch_pipeline --only=<clip_id> --favorites-dir <artifacts_dir>``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..core.registry import register_stage
from ..core.stage import Stage, StageContext


@register_stage("phase1")
class Phase1Stage(Stage):
    name = "phase1"
    upstream: tuple[str, ...] = ("extract_frames",)
    outputs: tuple[str, ...] = ("pipeline_result.pkl.gz",)

    def run(self, ctx: StageContext) -> None:
        # batch_pipeline expects a favorites-style layout at:
        #   <fav_dir>/<clip_id>/frames/*.jpg
        #   <fav_dir>/<clip_id>/manifest.json
        # Our artifact layout has frames at <artifact_root>/<clip_id>/extract_frames/frames/.
        # Bridge by symlinking the expected paths.
        clip_root = ctx.artifacts.root
        bridge_frames = clip_root / "frames"
        src_frames = ctx.artifacts.path("extract_frames", "frames")
        if not src_frames.exists():
            raise FileNotFoundError(f"frames missing at {src_frames}; run extract_frames first")
        # Don't touch clip_root/frames if it already resolves to the same dir as
        # extract_frames/frames. `egoinfinity import` stores the REAL frames at
        # clip_root/frames with a symlink at extract_frames/frames pointing back,
        # so blindly rmtree-ing clip_root/frames here would delete the imported
        # frames. Only (re)create the bridge when it's missing or points elsewhere.
        already = bridge_frames.exists() and src_frames.resolve() == bridge_frames.resolve()
        if not already:
            if bridge_frames.exists() or bridge_frames.is_symlink():
                bridge_frames.unlink() if bridge_frames.is_symlink() else shutil.rmtree(bridge_frames)
            bridge_frames.symlink_to(src_frames.resolve())

        # batch_pipeline relies on a favorites-style manifest with objects[].
        # The pipeline manifest already has video_uri + objects + start + end.
        # Copy/symlink them as needed; just make sure manifest is at <clip_root>/manifest.json.
        if not ctx.artifacts.manifest_path().exists():
            raise FileNotFoundError("manifest.json missing under clip artifact dir")

        # The artifact root is <artifacts_dir>/<clip_id>/. batch_pipeline
        # expects --favorites-dir to point to the PARENT (<artifacts_dir>).
        favorites_dir = clip_root.parent
        clip_id = ctx.clip_id

        # Override defaults the dev pipeline expects.
        env = os.environ.copy()
        env["ACTION100M_CACHE"] = str(favorites_dir)
        # Honor per-stage args
        if ctx.stage_config.args.get("hand_tta"):
            env["EGOINFINITY_HAND_TTA"] = "1"
        if ctx.stage_config.args.get("low_vram"):
            env["EGOINFINITY_LOW_VRAM"] = "1"
        if not ctx.stage_config.args.get("run_sam3d", True):
            env["EGOINFINITY_RUN_SAM3D"] = "0"
        if not ctx.stage_config.args.get("run_track", True):
            env["EGOINFINITY_RUN_TRACK"] = "0"

        # `--only=<id>` (not `--only <id>`): clip ids can start with '-'
        # (YouTube-derived), which argparse would otherwise treat as a flag.
        cmd = [sys.executable, "-m", "tools.batch_pipeline",
               "--favorites-dir", str(favorites_dir),
               f"--only={clip_id}", "--force"]
        if not ctx.stage_config.args.get("with_sam3d_worker", False):
            cmd.append("--no-sam3d-worker")
        sam3d_preset = ctx.stage_config.args.get("sam3d_preset")
        if sam3d_preset:
            cmd += ["--sam3d-preset", sam3d_preset]

        ctx.log.info("running: %s", " ".join(cmd))
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"batch_pipeline failed (exit {ret.returncode})")

        # batch_pipeline writes pipeline_result.pkl.gz at clip_root.
        # Verify it landed.
        out_pkl = clip_root / "pipeline_result.pkl.gz"
        if not out_pkl.exists():
            raise FileNotFoundError(f"expected output not found: {out_pkl}")

        # Mirror into the stage's own dir as well, for is_done() to find.
        target = ctx.artifacts.path(self.name, "pipeline_result.pkl.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(out_pkl.resolve())
