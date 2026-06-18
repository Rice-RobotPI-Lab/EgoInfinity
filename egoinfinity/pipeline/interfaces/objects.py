"""Object detection/tracking + mesh contracts (Phase D-sam3 / D-sam3d).

These run in separate conda envs via Unix-socket workers
(scripts/sam3_worker.py, scripts/sam3d_worker.py), so they are NOT
swapped by in-process class substitution — they are swapped by pointing
SAM3_REPO / SAM3D_REPO at a different checkout/worker. These Protocols
are documentation of the data contract only; the registry does not
construct them.

Default backends:
  obj_track -> SAM3.1 + SAM2  (egoinfinity.pipeline.sam3_client + object_tracker)
  obj_mesh  -> SAM 3D Objects (egoinfinity.pipeline.sam3d_client / sam3d_runner)
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IObjectTracker(Protocol):
    """Text-prompted detection + streaming mask track (Phase D-sam3).

    Produces, per frame per object, a mask + 3D OBB written into
    ``frame_data[t].sam3_obj_data``. The reference implementation calls
    ``sam3_client.run_sam3_detect`` (socket → sam3 worker) then a SAM2
    streaming tracker.
    """

    def detect_and_track(self, frames_rgb: List[np.ndarray],
                         prompts: List[str], *args, **kwargs):
        ...


@runtime_checkable
class IObjectMesh(Protocol):
    """Single-image 3D object reconstruction (Phase D-sam3d).

    Produces a per-object mesh (Gaussian-splat PLY) + canonical pose,
    written into ``sam3_mesh_info[oid]``. Reference implementation calls
    the SAM 3D Objects worker via ``sam3d_client`` / ``sam3d_runner``.
    """

    def reconstruct_mesh(self, image: np.ndarray, mask: np.ndarray,
                        *args, **kwargs):
        ...
