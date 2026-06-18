"""Backend Protocol contracts for the swappable Phase-1 components.

These are ``typing.Protocol`` declarations (structural typing) — concrete
backends do NOT need to inherit from them; they only need to provide the
listed methods with compatible signatures. They serve as:

  1. Documentation of the minimum API a new backend must implement.
  2. A type-check target (mypy / IDE) for backend authors.

The current shipped backends live under ``egoinfinity/pipeline/backends/``
and are selected at runtime via ``egoinfinity.pipeline.backends.get_backend``
(env vars ``EGOINFINITY_<ROLE>_BACKEND``). The default for every role
resolves to the exact class the monolithic pipeline used before
modularization, so behavior is unchanged unless a backend is explicitly
overridden.

Roles:
  depth        -> IDepthEstimator      (Phase A)
  gravity      -> IGravityEstimator    (Phase A-grav)
  hand_detect  -> IHandDetector        (Phase B-1)
  hand_recon   -> IHandReconstructor   (Phase B-2)
  obj_track    -> IObjectTracker       (Phase D-sam3; socket-worker, see note)
  obj_mesh     -> IObjectMesh          (Phase D-sam3d; socket-worker, see note)

Note on obj_track / obj_mesh: SAM3.1 detection and SAM 3D Objects run in
separate conda envs via Unix-socket workers (scripts/sam3_worker.py,
scripts/sam3d_worker.py). They are swapped by pointing SAM3_REPO /
SAM3D_REPO at a different checkout, not by in-process class substitution.
Their Protocols are declared here for documentation only.
"""
from .depth import IDepthEstimator
from .gravity import IGravityEstimator
from .hand import IHandDetector, IHandReconstructor
from .objects import IObjectTracker, IObjectMesh

__all__ = [
    "IDepthEstimator",
    "IGravityEstimator",
    "IHandDetector",
    "IHandReconstructor",
    "IObjectTracker",
    "IObjectMesh",
]
