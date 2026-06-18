# Third-party code under third_party/

| Subdirectory | Upstream | License | How it gets here | Used in |
|---|---|---|---|---|
| `sam2/` | https://github.com/facebookresearch/sam2 | Apache 2.0 | **Vendored** (LICENSE + NOTICE present) | Phase D (object mask tracking) |
| `wilor/` | https://github.com/rolpotamias/WiLoR | CC-BY-NC-ND 4.0 | **Fetched at install**, not vendored — see below | Phase B (hand reconstruction) |

For non-vendored third-party dependencies (pip-installable, sibling-repo,
Docker, etc.), see [`docs/THIRD_PARTY.md`](../docs/THIRD_PARTY.md).

## WiLoR: fetched + patched at install (NOT redistributed)

WiLoR is **CC-BY-NC-ND 4.0** (NoDerivatives), so this repo does not ship a
modified copy. Instead `scripts/setup_wilor.sh` clones WiLoR verbatim from
upstream at a pinned commit and applies EgoInfinity's local modifications
from [`wilor.patch`](wilor.patch). The result is byte-identical to what the
pipeline expects. Only the patch (our own work) is tracked here; the
`third_party/wilor/` tree it produces is gitignored.

The patch covers: MANO-data path discovery (package-relative), a `.float()`
dtype guard, and pinning the ViT backbone to its manual-attention form (the
pinned upstream commit predates upstream's switch to scaled-dot-product
attention). `setup_wilor.sh` runs automatically as part of `setup_weights.sh`.

## sam2 vendoring policy

`sam2/` is vendored (Apache-2.0 permits redistribution) and pinned to a
specific upstream commit; modifications are documented in its NOTICE.
