#!/usr/bin/env python
"""
SAM3 persistent worker.  Intended to be spawned **once** by the pipeline at startup
(analogous to Qwen preload) and listen on a Unix socket for detection
requests from pipeline subprocesses across multiple clips.

Life cycle
----------
1. the pipeline startup hook runs this worker once, passes --socket PATH.
2. Worker loads SAM3.1 model (~15-30s from NVMe cache) then accepts
   connections on the socket.
3. Each pipeline subprocess connects to the socket for every
   ``run_sam3_detect()`` call, sends one JSON line, reads one JSON
   line response, closes the connection.
4. Worker runs until the pipeline dies or is explicitly terminated.

Protocol (line-delimited JSON over Unix stream socket)
-----------------------------------------------------
Request::
    {"image": "/path/frame.jpg",
     "prompts": ["bowl", "knife"],
     "out_dir": "/tmp/sam3_xxx",
     "min_score": 0.4,          # optional
     "max_per_prompt": 10}      # optional
Response::
    {"type": "done", "result_file": "/tmp/sam3_xxx/result.json"}
    # or
    {"type": "error", "msg": "..."}

A small status file is written on startup::
    <socket_path>.ready   = JSON {"pid": ..., "version": ..., "load_time_s": ...}
so the pipeline can block until the worker is actually serving requests.
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
import torch
from PIL import Image


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BPE = os.path.join(
    _SCRIPT_DIR, '..', '..', 'sam3', 'sam3', 'assets', 'bpe_simple_vocab_16e6.txt.gz')
_BPE_PATH = os.path.realpath(_DEFAULT_BPE) if os.path.isfile(_DEFAULT_BPE) else None


def _log(msg: str) -> None:
    print(f"[sam3_worker] {msg}", file=sys.stderr, flush=True)


def _recv_line(conn: socket.socket, max_bytes: int = 1 << 20) -> str:
    """Read bytes from conn until newline or EOF."""
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
    data = (json.dumps(obj) + '\n').encode('utf-8')
    conn.sendall(data)


def _handle_request(processor, req: dict) -> dict:
    image_path = req['image']
    prompts = [s.strip() for s in req.get('prompts', []) if s and s.strip()]
    out_dir = req['out_dir']
    min_score = float(req.get('min_score', 0.4))
    max_per_prompt = int(req.get('max_per_prompt', 10))

    if not prompts:
        return {'type': 'error', 'msg': 'no valid prompts'}
    if not os.path.isfile(image_path):
        return {'type': 'error', 'msg': f'image not found: {image_path}'}
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    t_enc = time.time()
    state = processor.set_image(img)
    torch.cuda.synchronize()
    t_set_image = (time.time() - t_enc) * 1000

    def _to_np(x, dtype):
        if hasattr(x, 'detach'):
            return x.detach().float().cpu().numpy().astype(dtype)
        return np.asarray(x).astype(dtype)

    per_prompt = []
    t_det = time.time()
    for pi, prompt in enumerate(prompts):
        t_p = time.time()
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks_np = _to_np(out['masks'], bool)
        boxes_np = _to_np(out['boxes'], float)
        scores_np = _to_np(out['scores'], float)

        keep = np.where(scores_np >= min_score)[0]
        order = keep[np.argsort(-scores_np[keep])][: max_per_prompt]

        kept = []
        for rank, mi in enumerate(order):
            mask = masks_np[mi]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_file = f'masks_{pi}_{rank}.npz'
            np.savez_compressed(os.path.join(out_dir, mask_file), mask=mask)
            kept.append({
                'mask_file': mask_file,
                'box': [float(x) for x in boxes_np[mi].tolist()],
                'score': float(scores_np[mi]),
                'area': int(mask.sum()),
            })
        per_prompt.append({
            'prompt': prompt, 'n_raw': int(len(scores_np)),
            'n_kept': len(kept), 'results': kept,
        })
        _log(f"  [{pi}] '{prompt}' -> {len(scores_np)} raw, {len(kept)} kept ({(time.time()-t_p)*1000:.0f}ms)")

    detect_total_ms = (time.time() - t_det) * 1000
    result = {
        'image_path': image_path, 'image_size': [W, H],
        'min_score': min_score, 'per_prompt': per_prompt,
        'timings_ms': {'set_image': int(t_set_image), 'detect_total': int(detect_total_ms)},
    }
    result_file = os.path.join(out_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    _log(f"total detect {detect_total_ms:.0f}ms -> {result_file}")
    return {'type': 'done', 'result_file': result_file}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True,
                    help='Unix socket path to bind; will be (re)created.')
    ap.add_argument('--version', default='sam3.1', choices=['sam3', 'sam3.1'])
    ap.add_argument('--bpe_path', default=_BPE_PATH)
    args = ap.parse_args()

    # Clean stale socket before binding
    try:
        if os.path.exists(args.socket):
            os.unlink(args.socket)
    except Exception:
        pass

    # bf16 autocast (matches sam3 example notebooks)
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    _log(f"loading SAM3 ({args.version}) ...")
    ckpt_path = download_ckpt_from_hf(version=args.version)
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=ckpt_path)
    processor = Sam3Processor(model)
    torch.cuda.synchronize()
    load_s = time.time() - t0
    _log(f"model loaded ({load_s:.1f}s)")

    # Bind socket
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(4)

    # Write status file so the pipeline can tell the worker is ready.
    status = {'pid': os.getpid(), 'version': args.version, 'load_time_s': round(load_s, 1)}
    with open(args.socket + '.ready', 'w') as f:
        json.dump(status, f)
    _log(f"listening on {args.socket} (pid {os.getpid()})")

    # Graceful shutdown on SIGTERM / SIGINT (used by the pipeline shutdown hook)
    stop = {'flag': False}
    def _sig(_sig, _frm):
        _log(f"signal {_sig} received, shutting down")
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
                resp = _handle_request(processor, req)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                resp = {'type': 'error', 'msg': traceback.format_exc().splitlines()[-1]}
            # Don't let a broken client connection kill the worker for
            # every other request. BrokenPipe / ConnectionReset are
            # expected when a client times out and disconnects.
            try:
                _send_line(conn, resp)
            except (BrokenPipeError, ConnectionResetError, OSError) as _e:
                _log(f"  [send] client disconnected before response ({_e})")
        finally:
            try: conn.close()
            except Exception: pass

    try:
        os.unlink(args.socket)
        os.unlink(args.socket + '.ready')
    except Exception:
        pass
    _log('exited cleanly')


if __name__ == '__main__':
    main()
