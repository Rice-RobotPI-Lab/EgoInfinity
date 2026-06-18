"""Stage base class + StageContext.

Stages are thin wrappers around algorithm code in ``egoinfinity/pipeline/``
(incl. ``post_tracking/``), ``retarget/``, ``action100m_filter/``, and ``tools/``.
A Stage:

  - declares ``upstream`` (other stages it depends on, for ordering)
  - declares ``outputs`` (named artifact files it writes into its
    ``<artifacts>/<stage>/`` dir, used to decide if "done")
  - implements ``run(ctx)`` which produces those outputs

The runner asks every Stage ``is_done(ctx)`` and skips it when true,
giving the "resume from any point" semantics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from .artifacts import ArtifactStore
from .config import PipelineConfig, StageConfig
from .state import ClipState


@dataclass
class StageContext:
    clip_id: str
    artifacts: ArtifactStore
    state: ClipState
    config: PipelineConfig
    stage_config: StageConfig
    log: logging.Logger


class Stage:
    """Base class — subclasses set ``name``, ``upstream``, ``outputs`` and impl ``run``."""

    name: ClassVar[str] = ""
    upstream: ClassVar[tuple[str, ...]] = ()
    outputs: ClassVar[tuple[str, ...]] = ()
    backend_key: ClassVar[str | None] = None      # which interface this stage satisfies (for backend swap)

    def is_done(self, ctx: StageContext) -> bool:
        """Default: state.is_done(stage) AND all named outputs exist on disk."""
        if not ctx.state.is_done(self.name):
            return False
        for o in self.outputs:
            if not ctx.artifacts.exists(self.name, o):
                return False
        return True

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError(f"{type(self).__name__}.run not implemented")
