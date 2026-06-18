#!/bin/bash
# Set up WiLoR (Phase B hand reconstruction).
#
# WiLoR is licensed CC-BY-NC-ND 4.0 (NoDerivatives), so this repo does NOT
# redistribute a modified copy. Instead it ships only EgoInfinity's local
# modifications as a patch (third_party/wilor.patch) and fetches WiLoR
# verbatim from upstream at a pinned commit here, applying the patch locally.
# The result is byte-identical to the previously-vendored tree.
#
# Run from the repo root:  bash scripts/setup_wilor.sh
set -euo pipefail

WILOR_REPO="https://github.com/rolpotamias/WiLoR.git"
WILOR_COMMIT="fcb911312a38fa8badd30d9656a167485d61b8f9"   # pinned (no upstream tags)
DEST="third_party/wilor"
PATCH="third_party/wilor.patch"

if [ -f "$DEST/models/wilor.py" ]; then
    echo "WiLoR already present at $DEST/ — skipping.  (rm -rf $DEST to redo.)"
    exit 0
fi
[ -f "$PATCH" ] || { echo "ERROR: $PATCH not found — run from the repo root."; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning WiLoR @ ${WILOR_COMMIT:0:10} ..."
git clone --quiet "$WILOR_REPO" "$TMP/WiLoR"
git -C "$TMP/WiLoR" checkout --quiet "$WILOR_COMMIT"

mkdir -p "$DEST"
cp -r "$TMP/WiLoR/wilor/." "$DEST/"                          # the importable `wilor` package
mkdir -p "$DEST/mano_data"
cp "$TMP/WiLoR/mano_data/mano_mean_params.npz" "$DEST/mano_data/"
cp "$TMP/WiLoR/license.txt" "$DEST/LICENSE"                  # upstream CC-BY-NC-ND, verbatim

echo "Applying EgoInfinity patch (path discovery + dtype + pinned attention) ..."
patch -p1 -d "$DEST" < "$PATCH"

echo "WiLoR ready at $DEST/  (upstream ${WILOR_COMMIT:0:10} + EgoInfinity patch)."
