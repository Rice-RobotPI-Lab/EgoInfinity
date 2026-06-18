"""
Client wrapper for SAM3 detection.

By default this module talks to a **persistent worker** (``sam3_worker.py``)
over a Unix socket.  The worker is expected to be spawned once by the pipeline at
startup (analogous to Qwen preload) so every pipeline subprocess across
multiple clips reuses the same warm SAM3 model.

Socket path is read from ``$SAM3_WORKER_SOCKET``.  If the env var is not
set or the socket is unreachable, the code falls back to the legacy
one-shot ``scripts/sam3_detect_cli.py`` subprocess path (slow,
~170 s / call, but self-contained — good for debugging).

Force one-shot mode even when the socket exists by setting
``SAM3_CLIENT_MODE=oneshot``.
"""
import json
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


from .config import SAM3_PYTHON as SAM3_PY, SAM3_CLI, SAM3_BPE_PATH


@dataclass
class Sam3Mask:
    """A single SAM3 detection result (one mask for one prompt)."""
    prompt: str
    mask: np.ndarray            # (H, W) bool
    box: List[float]            # [x1, y1, x2, y2]
    score: float
    area: int


# ---------------------------------------------------------------------------
# Unix socket client to persistent worker
# ---------------------------------------------------------------------------
def _detect_via_socket(
    image_path: str, prompts: List[str], out_dir: str,
    min_score: float, max_per_prompt: int, timeout: float,
) -> str:
    """Send one request to the persistent worker over its Unix socket.

    Returns path to result.json on success; raises on failure.
    """
    sock_path = os.environ.get('SAM3_WORKER_SOCKET', '') \
        or f"/tmp/egoinfinity_sam3_{os.environ.get('USER', 'user')}.sock"
    if not os.path.exists(sock_path):
        raise FileNotFoundError(f'SAM3 worker socket not available: {sock_path!r}')

    req = {
        'image': image_path, 'prompts': prompts, 'out_dir': out_dir,
        'min_score': min_score, 'max_per_prompt': max_per_prompt,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps(req) + '\n').encode('utf-8'))
        # read until newline
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
        raise RuntimeError('sam3_worker closed connection without response')
    resp = json.loads(line)
    if resp.get('type') != 'done':
        raise RuntimeError(f"sam3_worker error: {resp.get('msg')}")
    return resp['result_file']


# ---------------------------------------------------------------------------
# Legacy one-shot subprocess (fallback when worker socket absent)
# ---------------------------------------------------------------------------
def _worker_env() -> dict:
    sam3_bin_dir = os.path.dirname(SAM3_PY)
    return {
        'PATH': f'{sam3_bin_dir}:/usr/bin:/bin',
        'HOME': os.environ['HOME'],
        'HF_HOME': os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    }


def _detect_via_oneshot(
    image_path: str, prompts: List[str], out_dir: str,
    min_score: float, max_per_prompt: int, version: str, timeout: float,
) -> Optional[str]:
    cmd = [
        SAM3_PY, SAM3_CLI,
        '--image', image_path,
        '--prompts', ','.join(prompts),
        '--out_dir', out_dir,
        '--min_score', str(min_score),
        '--max_per_prompt', str(max_per_prompt),
        '--version', version,
        '--bpe_path', SAM3_BPE_PATH,
    ]
    proc = subprocess.run(
        cmd, env=_worker_env(), capture_output=True, text=True, timeout=timeout,
        encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        print(f"[sam3_client] one-shot failed (code {proc.returncode})")
        print(f"[sam3_client] stdout: {proc.stdout[-500:]}")
        print(f"[sam3_client] stderr: {proc.stderr[-500:]}")
        return None
    return os.path.join(out_dir, 'result.json')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_sam3_detect(
    image_path: str,
    prompts: List[str],
    min_score: float = 0.4,
    max_per_prompt: int = 10,
    version: str = "sam3.1",
    timeout: float = 600.0,
    keep_tmpdir: bool = False,
    tmp_root: Optional[str] = None,
) -> List[Sam3Mask]:
    """Run SAM3 detection on one image with a list of text prompts.

    Prefers the persistent worker over Unix socket (env var
    ``SAM3_WORKER_SOCKET``); falls back to legacy one-shot subprocess if
    the worker is unavailable or ``SAM3_CLIENT_MODE=oneshot``.

    Returns flat list of kept detections sorted by descending score; empty
    list on failure.
    """
    if not prompts:
        return []
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    tmpdir = tempfile.mkdtemp(prefix='sam3_', dir=tmp_root)
    try:
        mode = os.environ.get('SAM3_CLIENT_MODE', 'auto').lower()
        result_file: Optional[str] = None

        _wk_sock = os.environ.get('SAM3_WORKER_SOCKET', '') \
            or f"/tmp/egoinfinity_sam3_{os.environ.get('USER', 'user')}.sock"
        try_worker = (mode != 'oneshot' and os.path.exists(_wk_sock))
        if try_worker:
            try:
                result_file = _detect_via_socket(
                    image_path, prompts, tmpdir,
                    min_score, max_per_prompt, timeout)
            except Exception as e:
                print(f"[sam3_client] worker path failed ({e}), falling back to one-shot")
                result_file = None

        if result_file is None:
            result_file = _detect_via_oneshot(
                image_path, prompts, tmpdir,
                min_score, max_per_prompt, version, timeout)

        if not result_file or not os.path.isfile(result_file):
            print(f"[sam3_client] result.json missing in {tmpdir}")
            return []

        with open(result_file) as f:
            result = json.load(f)
        out: List[Sam3Mask] = []
        for pp in result.get('per_prompt', []):
            prompt = pp['prompt']
            for item in pp.get('results', []):
                mask_path = os.path.join(tmpdir, item['mask_file'])
                with np.load(mask_path) as nz:
                    mask = nz['mask'].astype(bool)
                out.append(Sam3Mask(
                    prompt=prompt, mask=mask,
                    box=list(item['box']),
                    score=float(item['score']),
                    area=int(item['area']),
                ))
        out.sort(key=lambda m: -m.score)
        return out
    finally:
        if not keep_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _is_word_subseq(a: str, b: str) -> bool:
    """True if `a`'s tokens form a contiguous word-subsequence of `b`'s.

    Used by containment merge to detect intentional "fallback simple noun"
    relationships:
      "white plate"      ⊂ "white plate of sliced turkey"  → True
      "turkey"           ⊂ "white plate of sliced turkey"  → True
      "tongs"            ⊂ "white plate of sliced turkey"  → False
    """
    aw = a.lower().split()
    bw = b.lower().split()
    if not aw or len(aw) >= len(bw):
        return False
    for i in range(len(bw) - len(aw) + 1):
        if bw[i:i+len(aw)] == aw:
            return True
    return False


def containment_merge(masks: List[Sam3Mask],
                       contain_thresh: float = 0.9) -> List[Sam3Mask]:
    """Drop smaller mask when ≥ ``contain_thresh`` of its area is contained
    by a larger mask AND their prompts are related (one is a word-subseq
    of the other).

    Designed for the "compound + simple fallback" pattern:
      ["white plate of sliced turkey", "white plate", "turkey", ...]
    SAM3 may detect both the compound and the simple. After IoU NMS they
    coexist (cross-prompt IoU ≈ 0.5 < 0.92). This pass collapses them
    into one mask (the larger compound) so SAM2 / SAM3D doesn't track
    duplicates.

    Prompt-affinity gate prevents semantically-distinct objects that
    happen to overlap (e.g. tongs sitting on a plate of turkey) from
    being absorbed.
    """
    if len(masks) <= 1:
        return masks
    # Sort largest first so containment check is one-way
    sorted_m = sorted(masks, key=lambda m: -int(m.mask.sum()))
    kept: List[Sam3Mask] = []
    for cand in sorted_m:
        cand_area = int(cand.mask.sum())
        absorbed = False
        for k in kept:
            inter = int((cand.mask & k.mask).sum())
            if inter == 0:
                continue
            cand_in_k = inter / max(cand_area, 1)
            if cand_in_k >= contain_thresh and (
                _is_word_subseq(cand.prompt, k.prompt) or
                _is_word_subseq(k.prompt, cand.prompt) or
                cand.prompt == k.prompt
            ):
                absorbed = True
                break
        if not absorbed:
            kept.append(cand)
    return kept


def nms_sam3_masks(masks: List[Sam3Mask], iou_thresh: float = 0.5,
                   max_keep: int = 5,
                   cross_prompt_iou_thresh: float = 0.92) -> List[Sam3Mask]:
    """Prompt-aware greedy NMS.

    Within the same prompt, multiple SAM3 detections compete normally:
    IoU > ``iou_thresh`` (default 0.5) → drop the lower-scoring one (kills
    near-duplicate detections returned by SAM3 for the same object).

    ACROSS prompts, only essentially identical masks get deduped:
    IoU > ``cross_prompt_iou_thresh`` (default 0.92).  This preserves the
    case where two semantically distinct objects physically overlap heavily
    in 2D — e.g. a paper towel placed ON TOP OF raw turkey, where the towel
    occludes the turkey almost entirely.  Both prompts return masks with
    high IoU, but they ARE different objects in 3D and the user typed both
    prompts on purpose, so keep them.

    Cross-prompt at 0.92 still catches the spaCy-era duplication case where
    "knife", "kitchen knife", and "blade" all return the same mask
    (IoU ≈ 0.99) — only the highest-scoring stays.
    """
    if not masks:
        return []
    masks = sorted(masks, key=lambda m: -m.score)
    kept: List[Sam3Mask] = []
    for cand in masks:
        dup = False
        for k in kept:
            inter = int((cand.mask & k.mask).sum())
            union = int((cand.mask | k.mask).sum())
            if union == 0:
                continue
            iou = inter / union
            if cand.prompt == k.prompt:
                threshold = iou_thresh
            else:
                threshold = cross_prompt_iou_thresh
            if iou > threshold:
                dup = True
                break
        if not dup:
            kept.append(cand)
        if len(kept) >= max_keep:
            break
    return kept
