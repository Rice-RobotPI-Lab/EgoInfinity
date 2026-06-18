#!/usr/bin/env python
"""
SAM3 single-image detection CLI.

Runs inside the *sam3* conda environment.  Called from the main pipeline
via subprocess.  Takes one image and one or more text prompts, runs SAM3
detection for each prompt, and writes masks + metadata to a temp dir.

Usage:
    python sam3_detect_cli.py \
        --image /path/to/frame.jpg \
        --prompts "cauliflower,knife,cutting board" \
        --out_dir /path/to/tmp/sam3_out \
        [--min_score 0.4] \
        [--version sam3.1]

Output files in --out_dir:
    result.json                       - per-prompt metadata
    masks_{prompt_idx}_{mask_idx}.npz - each mask as a bool array
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


# bpe_path must be explicit: pkg_resources on editable install returns None.
# Default: auto-detect from this script's location (assumes sam3 repo is a
# sibling of the EgoInfinity repo: .../EgoInfinity/EgoInfinity/ and .../EgoInfinity/sam3/).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BPE = os.path.join(
    _SCRIPT_DIR, '..', '..', 'sam3', 'sam3', 'assets', 'bpe_simple_vocab_16e6.txt.gz')
_BPE_PATH = os.path.realpath(_DEFAULT_BPE) if os.path.isfile(_DEFAULT_BPE) else None


def _log(msg):
    print(f"[sam3_cli] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompts", required=True,
                   help="Comma-separated list of text prompts")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--min_score", type=float, default=0.4,
                   help="Minimum detection score to keep")
    p.add_argument("--max_per_prompt", type=int, default=10,
                   help="Maximum masks kept per prompt (sorted by score)")
    p.add_argument("--version", type=str, default="sam3.1",
                   choices=["sam3", "sam3.1"],
                   help="Which SAM3 checkpoint to use")
    p.add_argument("--bpe_path", type=str, default=_BPE_PATH)
    args = p.parse_args()

    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    if not prompts:
        _log("ERROR: no valid prompts")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    # bf16 autocast is required (SAM3 example notebooks all do this)
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    _log(f"loading SAM3 ({args.version})...")
    ckpt_path = download_ckpt_from_hf(version=args.version)
    model = build_sam3_image_model(
        bpe_path=args.bpe_path, checkpoint_path=ckpt_path)
    processor = Sam3Processor(model)
    torch.cuda.synchronize()
    _log(f"model loaded ({time.time() - t0:.1f}s)")

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    _log(f"image {W}x{H}, {len(prompts)} prompts")

    t_enc = time.time()
    state = processor.set_image(img)
    torch.cuda.synchronize()
    _log(f"set_image done ({(time.time() - t_enc) * 1000:.0f}ms)")

    per_prompt = []
    t_det = time.time()
    for pi, prompt in enumerate(prompts):
        t_p = time.time()
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks = out["masks"]
        boxes = out["boxes"]
        scores = out["scores"]

        # to numpy (cast bfloat16 -> float32 first since numpy doesn't support bf16)
        def _to_np(x, dtype):
            if hasattr(x, "detach"):
                return x.detach().float().cpu().numpy().astype(dtype)
            return np.asarray(x).astype(dtype)
        masks_np = _to_np(masks, bool)
        boxes_np = _to_np(boxes, float)
        scores_np = _to_np(scores, float)

        # score filter + sort
        keep = np.where(scores_np >= args.min_score)[0]
        order = keep[np.argsort(-scores_np[keep])][: args.max_per_prompt]

        kept_items = []
        for rank, mi in enumerate(order):
            mask = masks_np[mi]
            if mask.ndim == 3:  # (1, H, W)
                mask = mask.squeeze(0)
            mask_file = f"masks_{pi}_{rank}.npz"
            np.savez_compressed(os.path.join(args.out_dir, mask_file), mask=mask)
            kept_items.append({
                "mask_file": mask_file,
                "box": [float(x) for x in boxes_np[mi].tolist()],
                "score": float(scores_np[mi]),
                "area": int(mask.sum()),
            })

        per_prompt.append({
            "prompt": prompt,
            "n_raw": int(len(scores_np)),
            "n_kept": len(kept_items),
            "results": kept_items,
        })
        _log(
            f"  [{pi}] '{prompt}' -> {len(scores_np)} raw, "
            f"{len(kept_items)} kept  ({(time.time() - t_p) * 1000:.0f}ms)")

    _log(f"total detect {(time.time() - t_det) * 1000:.0f}ms")

    result = {
        "image_path": args.image,
        "image_size": [W, H],
        "version": args.version,
        "min_score": args.min_score,
        "per_prompt": per_prompt,
        "timings_ms": {
            "model_load": int((t_det - t0) * 1000),
            "set_image": int((t_det - t_enc) * 1000),
            "detect_total": int((time.time() - t_det) * 1000),
        },
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    _log(f"wrote {args.out_dir}/result.json")


if __name__ == "__main__":
    main()
