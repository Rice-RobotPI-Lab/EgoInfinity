"""Stage + backend registry.

Stage classes register themselves at import time::

    @register_stage("depth", backend="moge2")
    class MoGe2DepthStage(Stage):
        ...

The runner resolves ``(stage_name, backend) → Stage class`` via ``get_stage()``.
``backend="default"`` is the sentinel for stages without backend choice
(e.g. ``extract_frames`` has no swap surface).
"""
from __future__ import annotations

from typing import Type

from .stage import Stage


_STAGES: dict[tuple[str, str], Type[Stage]] = {}


def register_stage(name: str, *, backend: str = "default"):
    """Class decorator to register a Stage implementation."""
    def deco(cls: Type[Stage]) -> Type[Stage]:
        key = (name, backend)
        if key in _STAGES:
            raise RuntimeError(f"stage {name!r} backend {backend!r} already registered")
        cls.name = name
        _STAGES[key] = cls
        return cls
    return deco


def get_stage(name: str, backend: str = "default") -> Type[Stage]:
    key = (name, backend)
    if key not in _STAGES:
        # Try default backend as fallback
        if backend != "default" and (name, "default") in _STAGES:
            return _STAGES[(name, "default")]
        raise KeyError(f"no Stage registered for ({name!r}, backend={backend!r})")
    return _STAGES[key]


def list_stages() -> list[tuple[str, str]]:
    return sorted(_STAGES.keys())


def _import_stages_module() -> None:
    """Lazy-import egoinfinity.stages package so all Stage classes register."""
    # Import the package; its __init__ should import all submodules to trigger
    # their @register_stage decorators.
    from egoinfinity import stages   # noqa: F401
