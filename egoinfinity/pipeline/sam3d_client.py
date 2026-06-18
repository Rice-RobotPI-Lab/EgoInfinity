"""
Client wrapper for SAM 3D Objects single-image reconstruction.

Talks to a persistent ``scripts/sam3d_worker.py`` (spawned by the pipeline at
startup, running in the ``sam3d-objects`` conda env) over a Unix socket.
Socket path comes from ``$SAM3D_WORKER_SOCKET``.

Usage (from pipeline subprocess)::

    from egoinfinity.pipeline.sam3d_client import reconstruct_object

    res = reconstruct_object(
        rgb=image_uint8_HxWx3,
        mask=bool_mask_HxW,
        out_ply_path="/path/obj_0.ply",
        seed=42, quality="tier1",
    )
    # res.ply_path, res.translation (3,), res.rotation_quat (4,), res.scale (float)
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Sam3dResult:
    ply_path: str
    translation: np.ndarray        # (3,) float32
    rotation_quat: np.ndarray      # (4,) float32  [w, x, y, z]
    scale: float                   # uniform
    n_points: int
    timings_ms: dict


class Sam3dWorkerUnavailable(RuntimeError):
    pass


def _resolve_socket_path() -> str:
    """Resolve the SAM3D worker socket path.

    Prefers ``$SAM3D_WORKER_SOCKET`` (set by the pipeline once worker is ready) but
    falls back to the well-known path so subprocesses spawned before the env
    var was set can still reach a now-ready worker.
    """
    sock_path = os.environ.get('SAM3D_WORKER_SOCKET', '')
    if sock_path:
        return sock_path
    user = os.environ.get('USER', 'user')
    return f'/tmp/egoinfinity_sam3d_{user}.sock'


def _prepare_request_inputs(
    rgb: np.ndarray, mask: np.ndarray, tmp_root: Optional[str] = None,
) -> tuple[str, str, str]:
    """Write rgb as JPG + mask as NPZ into a fresh tmpdir.  Returns (tmpdir, rgb_path, mask_path)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}')
    if mask.ndim != 2 or mask.shape != rgb.shape[:2]:
        raise ValueError(f'mask must be (H, W) matching rgb, got mask {mask.shape} vs rgb {rgb.shape[:2]}')
    tmpdir = tempfile.mkdtemp(prefix='sam3d_req_', dir=tmp_root)
    rgb_path = os.path.join(tmpdir, 'image.jpg')
    mask_path = os.path.join(tmpdir, 'mask.npz')
    # Lazy import so main env's cv2 doesn't always get loaded
    import cv2
    # rgb (RGB) -> BGR for imwrite
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    np.savez_compressed(mask_path, mask=mask.astype(bool))
    return tmpdir, rgb_path, mask_path


def reconstruct_object(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_ply_path: str,
    seed: int = 42,
    quality: str = "tier1",
    timeout: float = 300.0,
    tmp_root: Optional[str] = None,
    keep_tmpdir: bool = False,
) -> Sam3dResult:
    """Reconstruct one object via SAM 3D worker.

    Raises
    ------
    Sam3dWorkerUnavailable
        If ``SAM3D_WORKER_SOCKET`` env is unset or socket does not exist.
    RuntimeError
        Worker returned an error or closed unexpectedly.
    """
    sock_path = _resolve_socket_path()
    if not os.path.exists(sock_path):
        raise Sam3dWorkerUnavailable(
            f'SAM3D worker socket not available: {sock_path!r}')

    tmpdir, rgb_path, mask_path = _prepare_request_inputs(rgb, mask, tmp_root)
    try:
        req = {
            'image_rgb_path': rgb_path,
            'mask_path': mask_path,
            'out_ply_path': out_ply_path,
            'seed': seed,
            'quality': quality,
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(sock_path)
            s.sendall((json.dumps(req) + '\n').encode('utf-8'))
            buf = bytearray()
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if b'\n' in chunk:
                    break
        line = buf.decode('utf-8', errors='replace').split('\n', 1)[0]
        if not line:
            raise RuntimeError('sam3d_worker closed connection without response')
        resp = json.loads(line)
        if resp.get('type') != 'done':
            raise RuntimeError(f"sam3d_worker error: {resp.get('msg')}")
        return Sam3dResult(
            ply_path=resp['ply_path'],
            translation=np.asarray(resp['translation'], dtype=np.float32).reshape(-1),
            rotation_quat=np.asarray(resp['rotation'], dtype=np.float32).reshape(-1),
            scale=float(np.asarray(resp['scale'], dtype=np.float32).reshape(-1)[0]),
            n_points=int(resp.get('n_points', 0)),
            timings_ms=resp.get('timings_ms', {}),
        )
    finally:
        if not keep_tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def worker_available() -> bool:
    """Quick probe: is the SAM3D worker socket present?"""
    return os.path.exists(_resolve_socket_path())
