"""FilterStage — per-clip static-cam + hand-presence gate.

Wraps the existing ``action100m_filter/detect.py:analyze_frames`` (YOLO
hand detection + background optical-flow) to produce a single judgment:
is this clip suitable for downstream pipeline?

Inputs:
  extract_frames/frames/*.jpg     (from ExtractFramesStage)

Outputs:
  filter/result.json              { pass: bool, reason: str|null, metrics: {...} }
  filter/sample_frames/*.jpg      (a few representative frames for viz, optional)

This Stage is OPTIONAL by default (config: enabled: optional). It only
RECORDS the judgment; it does not raise / block pipeline progress. Use
``hard_gate: true`` in args to make pipeline fail when filter rejects.

For the BATCH curation workflow (filter many YouTube videos against
Action100M index db), use the standalone ``egoinfinity filter`` CLI which
wraps ``action100m_filter/main.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..core.registry import register_stage
from ..core.stage import Stage, StageContext


# Lazy imports so the package loads without action100m_filter installed deps.
def _get_analyzer():
    from action100m_filter.config import FilterConfig
    from action100m_filter.detect import analyze_frames
    return FilterConfig, analyze_frames


@register_stage("filter")
class FilterStage(Stage):
    name = "filter"
    upstream: tuple[str, ...] = ("extract_frames",)
    outputs: tuple[str, ...] = ("result.json",)

    def run(self, ctx: StageContext) -> None:
        FilterConfig, analyze_frames = _get_analyzer()

        # Sample frames from extract_frames stage output
        frames_dir = ctx.artifacts.path("extract_frames", "frames")
        if not frames_dir.exists():
            raise FileNotFoundError(
                f"FilterStage needs extract_frames output at {frames_dir}; "
                f"run extract_frames first."
            )

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no frames in {frames_dir}")

        # Sample up to N=8 evenly-spaced frames for the hand-presence judgment
        # (matches FilterConfig.n_frames default). Separate Stage-level args
        # (hard_gate) from FilterConfig-valid fields.
        from dataclasses import fields as _fields
        _cfg_fields = {f.name for f in _fields(FilterConfig)}
        cfg_args = {k: v for k, v in ctx.stage_config.args.items() if k in _cfg_fields}
        cfg = FilterConfig(**cfg_args)
        n_sample = cfg.n_frames
        step = max(1, len(frame_paths) // n_sample)
        sampled = frame_paths[::step][:n_sample]

        ctx.log.info("sampling %d/%d frames for filter analysis",
                     len(sampled), len(frame_paths))
        # action100m_filter.detect.analyze_frames expects BGR (cv2.imread default).
        frames = [cv2.imread(str(p)) for p in sampled]
        frames = [f for f in frames if f is not None]

        # Motion pairs: consecutive sampled frames; detect_shot_cuts-style.
        motion_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(frames) - 1):
            motion_pairs.append((frames[i], frames[i + 1]))

        metrics = analyze_frames(
            frames,
            detector_path=cfg.detector_path,
            hand_conf=cfg.hand_conf,
            trunc_edge_px=cfg.trunc_edge_px,
            motion_pairs=motion_pairs or None,
        )
        passed, reason = _judge(metrics, cfg)
        result = {
            "pass": passed,
            "reason": reason,
            "metrics": _coerce_numpy(metrics),
            "n_frames_sampled": len(frames),
            "n_frames_total": len(frame_paths),
        }
        out = ctx.artifacts.path(self.name, "result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
        ctx.log.info("filter: pass=%s reason=%s", passed, reason)

        # Optional hard gate
        if not passed and ctx.stage_config.args.get("hard_gate", False):
            raise RuntimeError(f"filter rejected clip: {reason}")


def _judge(metrics: dict, cfg) -> tuple[bool, str | None]:
    """Mirror action100m_filter.main._judge logic — kept local so we don't
    import the entire main.py with its sqlite and CLI baggage.
    """
    if metrics.get("hand_ratio", 0.0) < cfg.min_hand_ratio:
        return False, f"hand_ratio<{cfg.min_hand_ratio}"
    if metrics.get("trunc_ratio", 1.0) > cfg.max_trunc_ratio:
        return False, f"trunc_ratio>{cfg.max_trunc_ratio}"
    if metrics.get("bg_flow_p20", 999) > cfg.max_bg_flow:
        return False, f"bg_flow>{cfg.max_bg_flow}"
    return True, None


def _coerce_numpy(obj):
    """Convert numpy scalars to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _coerce_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
