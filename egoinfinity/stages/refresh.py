"""Refresh Stages — wrappers for the phase1-internal re-run tools in
``tools/rerun``.

Phase 1 (A->E) is monolithic, so there is no fine-grained `egoinfinity`
stage for re-running a single Phase-B/C/D-sam3d substep. The standalone
``tools/rerun/refresh_*.py`` tools fill that gap: each re-runs ONE substep
on an existing ``pipeline_result.pkl.gz``, reusing the upstream results
already in the pkl. This module exposes them as registered Stages so they
are invokable via ``egoinfinity run <stage> <clip>`` (and over many clips,
once the runner supports it).

They are NOT part of the default `process` pipeline (phase1 already
produces these outputs); they ship `enabled: false` in defaults.yaml and
are opt-in re-runs. Typical use: cross-host SAM3D fill (run phase1 with
``--no-sam3d`` on a small GPU, ship the pkl, then
``egoinfinity run refresh_sam3d <clip>`` on a 16 GB+ host). See
docs/MULTI_HOST.md.

All four reuse :class:`egoinfinity.stages.post_track._RefreshStage`, which
subprocess-calls ``python -m <module> --only=<CLIP_ID>`` and bridges the
favorites-dir layout the tools expect via a tempdir symlink + ACTION100M_CACHE.
"""
from __future__ import annotations

from ..core.registry import register_stage
from .post_track import _RefreshStage, _camel


# ── phase1-internal re-run table ─────────────────────────────────────────────
# Each row: (stage_name, module, cli_args). All have upstream=("phase1",) and
# mutate the per-clip pkl in place (outputs=()). The tools default to "all
# favorites"; --only=<CLIP_ID> (injected by _RefreshStage) scopes to one clip.
REFRESH_STAGES: tuple = (
    # name                 module                                        cli_args
    ("refresh_sam3d",        "tools.rerun.refresh_sam3d_meshes",   ()),
    ("refresh_hands",        "tools.rerun.refresh_hands",          ()),
    ("refresh_hand_scale",   "tools.rerun.refresh_hand_scale",     ()),
    ("refresh_optical_flow", "tools.rerun.refresh_optical_flow",   ()),
)


for (_name, _module, _cli_args) in REFRESH_STAGES:
    _cls = type(
        f"{_camel(_name)}Stage",
        (_RefreshStage,),
        {
            "name": _name,
            "module": _module,
            "cli_args": _cli_args,
            "upstream": ("phase1",),
        },
    )
    register_stage(_name)(_cls)
    globals()[_cls.__name__] = _cls

del _name, _module, _cli_args, _cls
