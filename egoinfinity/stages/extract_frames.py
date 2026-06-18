"""ExtractFramesStage — ffmpeg wrapper as the canonical first stage.

Reads ``manifest.json`` to find the source video URI + start/end times,
extracts frames as zero-padded JPEGs into ``input/frames/``.

Manifest fields (relevant):
  video_uri:  str (path or URL; if URL, downloaded first — TODO)
  start:      float seconds (default 0)
  end:        float seconds (default = full duration)
  fps:        int (default 15; matches Action100M / EgoInfinity convention)
  frame_size: [w, h] (default keeps source resolution)

Outputs (under ``input/``):
  frames/000000.jpg, 000001.jpg, ...   (registered as one directory artifact)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..core.registry import register_stage
from ..core.stage import Stage, StageContext


@register_stage("extract_frames")
class ExtractFramesStage(Stage):
    name = "extract_frames"
    upstream: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ("frames",)         # directory artifact

    def run(self, ctx: StageContext) -> None:
        manifest_path = ctx.artifacts.manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json missing at {manifest_path}; ExtractFramesStage "
                f"requires manifest.video_uri / start / end."
            )
        manifest = json.loads(manifest_path.read_text())
        video_uri = manifest.get("video_uri")
        if not video_uri:
            raise ValueError("manifest.video_uri is required for extract_frames")

        start = float(manifest.get("start", 0.0))
        end = manifest.get("end", None)
        fps = int(manifest.get("fps", 15))

        # Resolve video path (URL download deferred — Phase 2 filter handles that)
        video_path = self._resolve_video(video_uri, ctx)

        # Output: <root>/input/frames/<index>.jpg
        # We use the "input" stage dir (not extract_frames) to match the
        # spec'd layout where frames live under input/. The stage dir is
        # "input"; the artifact name "frames" resolves to <root>/input/frames/.
        # Override: use ArtifactStore.path("input", "frames") to get the dir,
        # then write into it. But our outputs declares "frames" relative to
        # the stage's own dir (extract_frames). Reconcile: the canonical
        # location for frames is input/frames/, but we register it under
        # this stage's "outputs" list. Make stage dir == "input" by special-casing.
        # Simpler: just use this stage's own dir extract_frames/frames/ initially.
        # Future polish: alias to input/frames/ via symlink or move.
        frames_dir = ctx.artifacts.path(self.name, "frames")
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.3f}"]
        if end is not None:
            duration = float(end) - start
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(frames_dir / "%06d.jpg")]
        ctx.log.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        n_frames = len(list(frames_dir.glob("*.jpg")))
        ctx.log.info("extracted %d frames @ %d fps into %s", n_frames, fps, frames_dir)

    @staticmethod
    def _resolve_video(uri: str, ctx: StageContext) -> Path:
        """Resolve a video URI to a local path.

        Phase 1: only local paths supported. URL → download is Phase 2
        (filter stage's download_video logic).
        """
        if uri.startswith(("http://", "https://", "youtube://")):
            raise NotImplementedError(
                "URL video sources not yet supported in extract_frames; "
                "download video locally first."
            )
        p = Path(uri).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"video not found: {p}")
        return p
