#!/bin/bash
# Download all pretrained weights for the EgoInfinity pipeline.
# Run from repo root: bash scripts/setup_weights.sh
#
# Honors EGOINFINITY_CKPT_DIR (see egoinfinity/pipeline/config.py) — if set, weights
# go there instead of `<repo>/pretrained_models`.  This lets multiple repos
# share one cached weights tree.

set -e

MODELS_DIR="${EGOINFINITY_CKPT_DIR:-pretrained_models}"
MANO_DIR="third_party/wilor/mano_data"

mkdir -p "$MODELS_DIR"
echo "Weights destination: $MODELS_DIR"

# WiLoR source (Phase B) is fetched from upstream + patched, not vendored
# (CC-BY-NC-ND). This creates third_party/wilor/ + its mano_data/.
echo "=== 0/5: WiLoR source (clone + patch) ==="
bash "$(dirname "$0")/setup_wilor.sh"

mkdir -p "$MANO_DIR"

echo "=== 1/5: WiLoR detector ==="
if [ ! -f "$MODELS_DIR/detector.pt" ]; then
    wget -q --show-progress \
        https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
        -O "$MODELS_DIR/detector.pt"
else
    echo "  Already exists, skipping."
fi

echo "=== 2/5: WiLoR checkpoint ==="
if [ ! -f "$MODELS_DIR/wilor_final.ckpt" ]; then
    wget -q --show-progress \
        https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
        -O "$MODELS_DIR/wilor_final.ckpt"
else
    echo "  Already exists, skipping."
fi

echo "=== 3/5: MoGe-2 ==="
echo "  MoGe-2 weights are downloaded automatically from HuggingFace on first run."
echo "  No manual setup needed. (pip install git+https://github.com/microsoft/MoGe.git)"

echo "=== 4/5: SAM2 ==="
if [ ! -f "$MODELS_DIR/sam2.1_hiera_small.pt" ]; then
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
        -O "$MODELS_DIR/sam2.1_hiera_small.pt"
else
    echo "  Already exists, skipping."
fi

echo "=== 5/5: HaWoR motion infiller (Phase C+) ==="
if [ ! -f "$MODELS_DIR/infiller.pt" ]; then
    wget -q --show-progress \
        https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt \
        -O "$MODELS_DIR/infiller.pt"
else
    echo "  Already exists, skipping."
fi

echo ""
echo "=== MANO ==="
if [ -f "$MANO_DIR/MANO_RIGHT.pkl" ]; then
    echo "  MANO_RIGHT.pkl present at $MANO_DIR/"
else
    echo "  ⚠ MANO_RIGHT.pkl NOT FOUND at $MANO_DIR/"
    echo ""
    echo "  MANO is licensed for non-commercial research use; we cannot"
    echo "  redistribute the weights.  To install (HaWoR / WiLoR convention):"
    echo ""
    echo "    1. Register at https://mano.is.tue.mpg.de"
    echo "    2. Download the 'Models & Code' archive (mano_v1_2.zip)"
    echo "    3. Extract MANO_RIGHT.pkl from models/MANO_RIGHT.pkl into"
    echo "       $MANO_DIR/"
    echo ""
    echo "  Without this file, hand reconstruction (Phase B) will fail."
fi

echo ""
echo "=== Done! ==="
ls -lh "$MODELS_DIR"
