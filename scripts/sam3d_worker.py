#!/usr/bin/env python
"""
SAM 3D Objects persistent worker.

Spawned once by the pipeline at startup (analogous to sam3_worker.py).  Loads the
SAM 3D pipeline once (~20-40s warm / ~90s cold) and then answers per-object
reconstruction requests over a Unix socket.

Runs inside the ``sam3d-objects`` conda env (separate from the main ``egoinfinity``
env because of torch / transformers / custom-CUDA dependencies).

Protocol (line-delimited JSON over Unix stream socket)
-----------------------------------------------------
Startup status::
    <socket>.ready  =  {"pid": ..., "load_time_s": ..., "quality": "tier1"}

Request::
    {"image_rgb_path": "/path/frame.jpg",
     "mask_path":      "/path/mask.npz",        # np.savez key='mask', bool (H, W)
     "out_ply_path":   "/path/obj_0.ply",
     "seed":           42,                      # optional
     "quality":        "tier1"}                 # optional: "tier1" | "high"

Response (success)::
    {"type":       "done",
     "ply_path":   "/path/obj_0.ply",
     "translation": [x, y, z],                  # mesh -> camera_init, 3 float
     "rotation":    [qw, qx, qy, qz],           # quaternion
     "scale":       [s, s, s],                  # uniform 3-vec
     "n_points":    N,
     "timings_ms":  {"total": N, "decode": N}}

Response (error)::
    {"type": "error", "msg": "..."}

Shutdown: SIGTERM / SIGINT -> graceful; stale socket file cleaned up.
"""
import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback

import numpy as np
from PIL import Image


# Default to a sibling clone of the sam-3d-objects repo (same convention as
# SAM3 in config.py).  Override via env var SAM3D_REPO.
_DEFAULT_REPO = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'sam-3d-objects'))
SAM3D_REPO_DEFAULT = os.environ.get("SAM3D_REPO", _DEFAULT_REPO)


def _log(msg: str) -> None:
    print(f"[sam3d_worker] {msg}", file=sys.stderr, flush=True)


def _recv_line(conn: socket.socket, max_bytes: int = 1 << 20) -> str:
    buf = bytearray()
    while len(buf) < max_bytes:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if b'\n' in chunk:
            break
    return buf.decode('utf-8', errors='replace').split('\n', 1)[0]


def _send_line(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + '\n').encode('utf-8'))


def _tensor_to_list(x):
    """Convert torch.Tensor / ndarray to a flat Python list."""
    if hasattr(x, 'detach'):
        x = x.detach().float().cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    return [float(v) for v in arr]


def _run_reconstruction(inference, req: dict) -> dict:
    image_rgb_path = req['image_rgb_path']
    mask_path = req['mask_path']
    out_ply_path = req['out_ply_path']
    seed = int(req.get('seed', 42))
    quality = req.get('quality', 'tier1')

    if not os.path.isfile(image_rgb_path):
        return {'type': 'error', 'msg': f'image not found: {image_rgb_path}'}
    if not os.path.isfile(mask_path):
        return {'type': 'error', 'msg': f'mask not found: {mask_path}'}

    os.makedirs(os.path.dirname(out_ply_path) or '.', exist_ok=True)

    # Load inputs
    image = np.array(Image.open(image_rgb_path).convert('RGB')).astype(np.uint8)
    with np.load(mask_path) as nz:
        mask = nz['mask'].astype(bool)
    if mask.shape != image.shape[:2]:
        return {'type': 'error',
                'msg': f'mask shape {mask.shape} != image shape {image.shape[:2]}'}

    t0 = time.time()
    # Quality preset
    if quality == 'high':
        kwargs = dict(stage1_inference_steps=25, stage2_inference_steps=25,
                      use_stage1_distillation=False,
                      use_stage2_distillation=False,
                      decode_formats=["gaussian"])
        out_key = 'gs'
    else:   # tier1
        kwargs = dict(stage1_inference_steps=2, stage2_inference_steps=12,
                      use_stage1_distillation=True,
                      use_stage2_distillation=True,
                      decode_formats=["gaussian_4"])
        out_key = 'gs_4'

    _log(f"reconstruct: {os.path.basename(image_rgb_path)} mask={mask.shape} "
         f"mask_area={int(mask.sum())} quality={quality}")
    output = inference(image, mask, seed=seed, **kwargs)

    gs = output.get(out_key)
    if gs is None:
        return {'type': 'error', 'msg': f'pipeline returned no {out_key}'}
    gs.save_ply(out_ply_path)
    n_points = 0
    try:
        # Gaussian model has ._xyz (N, 3)
        n_points = int(gs._xyz.shape[0])
    except Exception:
        pass

    translation = _tensor_to_list(output.get('translation', [0.0, 0.0, 0.0]))
    rotation    = _tensor_to_list(output.get('rotation',    [1.0, 0.0, 0.0, 0.0]))
    scale       = _tensor_to_list(output.get('scale',       [1.0, 1.0, 1.0]))

    total_ms = int((time.time() - t0) * 1000)
    _log(f"  done ({total_ms}ms) -> {out_ply_path} ({n_points} pts)")
    return {
        'type': 'done',
        'ply_path': out_ply_path,
        'translation': translation,
        'rotation': rotation,
        'scale': scale,
        'n_points': n_points,
        'timings_ms': {'total': total_ms},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--repo', default=SAM3D_REPO_DEFAULT)
    ap.add_argument('--config', default='checkpoints/hf/pipeline.yaml',
                    help='Path relative to --repo')
    ap.add_argument('--compile', action='store_true',
                    help='Enable torch.compile (slower first call, faster after)')
    ap.add_argument('--quality_default', default='tier1',
                    choices=['tier1', 'high'])
    args = ap.parse_args()

    # Clean stale socket
    for p in (args.socket, args.socket + '.ready'):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

    # chdir into sam3d repo so relative checkpoint paths resolve,
    # also add notebook/ to sys.path so `from inference import Inference` works
    os.chdir(args.repo)
    sys.path.insert(0, os.path.join(args.repo, 'notebook'))

    _log(f"loading SAM 3D pipeline from {args.config} (repo={args.repo}) ...")
    t0 = time.time()
    from inference import Inference
    inference = Inference(args.config, compile=args.compile)
    load_s = time.time() - t0
    _log(f"pipeline loaded ({load_s:.1f}s)")

    # Bind socket
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(4)

    status = {'pid': os.getpid(), 'load_time_s': round(load_s, 1),
              'quality_default': args.quality_default}
    # Atomic write: parent polls .ready existence and races us if we used a
    # plain `open('w')` (file is truncated to 0 bytes before json.dump
    # writes), so write to .tmp first then rename.
    ready_path = args.socket + '.ready'
    tmp_path = ready_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(status, f)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, ready_path)
    _log(f"listening on {args.socket} (pid {os.getpid()})")

    stop = {'flag': False}
    def _sig(_s, _f):
        _log(f"signal {_s} received, shutting down")
        stop['flag'] = True
        try: srv.close()
        except Exception: pass
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    while not stop['flag']:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        try:
            line = _recv_line(conn)
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as e:
                _send_line(conn, {'type': 'error', 'msg': f'bad json: {e}'})
                continue
            try:
                resp = _run_reconstruction(inference, req)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                resp = {'type': 'error',
                        'msg': traceback.format_exc().splitlines()[-1]}
            _send_line(conn, resp)
        finally:
            try: conn.close()
            except Exception: pass

    for p in (args.socket, args.socket + '.ready'):
        try:
            if os.path.exists(p): os.unlink(p)
        except Exception: pass
    _log('exited cleanly')


if __name__ == '__main__':
    main()
