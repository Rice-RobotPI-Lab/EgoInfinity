"""Swappable Phase-1 backend registry.

Lets the pipeline pick a component implementation by name without the
call site hardcoding a concrete class. The DEFAULT for every role
resolves to the exact class the monolithic pipeline used before
modularization, so behavior is byte-for-byte unchanged unless a backend
is explicitly overridden via env var.

Selection (per role):
    EGOINFINITY_<ROLE>_BACKEND  env var   (e.g. EGOINFINITY_DEPTH_BACKEND=moge2)
    -> falls back to DEFAULTS[role]

Usage at the call site (scripts/exo_pipeline.py):

    from egoinfinity.pipeline.backends import get_backend
    depth_estimator = get_backend("depth")(device="cuda")   # == MoGe2Estimator(device="cuda")

Registration uses LAZY loaders (a zero-arg callable returning the class)
so importing this package does NOT import torch/MoGe/WiLoR/etc. The
heavy import happens only when get_backend() is actually called.

Adding a new backend:

    from egoinfinity.pipeline.backends import register
    def _my_depth():
        from my_pkg import MyDepth
        return MyDepth
    register("depth", "mydepth", _my_depth)
    # then: EGOINFINITY_DEPTH_BACKEND=mydepth

See egoinfinity/pipeline/interfaces/ for the Protocol each role must satisfy.
"""
from __future__ import annotations

import os
from typing import Callable, Dict

# role -> { name -> loader() -> class }
_REGISTRY: Dict[str, Dict[str, Callable[[], type]]] = {}

# role -> default backend name (resolves to the pre-modularization class)
DEFAULTS: Dict[str, str] = {
    "depth": "moge2",
    "gravity": "geocalib",
    "hand_detect": "yolo",
    "hand_recon": "wilor",
}


def register(role: str, name: str, loader: Callable[[], type]) -> None:
    """Register a backend loader. ``loader`` is a zero-arg callable that
    imports and returns the backend CLASS (not an instance)."""
    _REGISTRY.setdefault(role, {})[name] = loader


def list_backends(role: str) -> list[str]:
    return sorted(_REGISTRY.get(role, {}).keys())


def get_backend(role: str, name: str | None = None) -> type:
    """Resolve a backend class for ``role``.

    name precedence: explicit arg -> ``EGOINFINITY_<ROLE>_BACKEND`` env ->
    DEFAULTS[role]. Returns the CLASS; the caller instantiates it
    (preserving the exact constructor call the monolith used).
    """
    if name is None:
        name = os.environ.get(f"EGOINFINITY_{role.upper()}_BACKEND") or DEFAULTS.get(role)
    if name is None:
        raise KeyError(f"no default backend for role {role!r} and none given")
    table = _REGISTRY.get(role, {})
    if name not in table:
        raise KeyError(
            f"unknown backend {name!r} for role {role!r}. "
            f"Registered: {list_backends(role)}")
    return table[name]()


# ── Default registrations (lazy) ────────────────────────────────────────────
# Each loader returns the exact class the monolith instantiated, so the
# default resolution is provably identical to pre-modularization behavior.

def _moge2() -> type:
    from egoinfinity.pipeline.moge2_estimator import MoGe2Estimator
    return MoGe2Estimator


def _geocalib() -> type:
    from egoinfinity.pipeline.gravity_estimator import GravityEstimator
    return GravityEstimator


def _yolo_hand() -> type:
    from egoinfinity.pipeline.hand_detector import HandDetector
    return HandDetector


def _wilor() -> type:
    from egoinfinity.pipeline.hand_reconstructor import HandReconstructor
    return HandReconstructor


register("depth", "moge2", _moge2)
register("gravity", "geocalib", _geocalib)
register("hand_detect", "yolo", _yolo_hand)
register("hand_recon", "wilor", _wilor)


__all__ = ["register", "get_backend", "list_backends", "DEFAULTS"]
