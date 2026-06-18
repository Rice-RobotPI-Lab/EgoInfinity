# Third-party dependencies

EgoInfinity integrates several external models / libraries. They are
handled in one of four ways depending on license, size, and conflict
profile:

1. **Vendored fork** in [`third_party/`](../third_party/) — license
   permits redistribution + we maintain local modifications.
2. **Pip install from git** — small wheels, no conflict with the main
   conda env; listed in install instructions.
3. **Sibling repo + separate conda env + Unix-socket worker** — heavy
   dependencies (different Python/torch/CUDA) that would conflict;
   talk to the worker via IPC.
4. **Docker container** — third-party tools that ship as containers.

## Catalog

| Tool | Handling | Where in repo | Used by |
|---|---|---|---|
| [WiLoR](https://github.com/rolpotamias/WiLoR) | Vendored fork | `third_party/wilor/` (CC-BY-NC-ND 4.0) | Phase B-2 hand recon |
| [SAM 2](https://github.com/facebookresearch/sam2) | Vendored fork | `third_party/sam2/` (Apache 2.0) | Phase D mask tracking |
| [MoGe / MoGe-2](https://github.com/microsoft/MoGe) | Pip install from git | site-packages | Phase A metric depth |
| [GeoCalib](https://github.com/cvg/GeoCalib) | Pip install from git | site-packages | Phase A-g gravity |
| [MEMFOF](https://github.com/msu-video-group/memfof) | Pip install from git (lazy hint) | site-packages | Phase C optical flow |
| [HaWoR](https://github.com/ThunderVVV/HaWoR) | Algorithm code vendored + weights from HF | `egoinfinity/pipeline/infiller_utils/` (see NOTICE) | Phase C+ motion infiller |
| [SAM 3.1](https://huggingface.co/facebook/sam3.1) | Sibling repo + dedicated `sam3` conda env | `../sam3/` (env vars `SAM3_PYTHON`, `SAM3_REPO`) | Phase D-sam3 text-prompted detection |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | Sibling repo + dedicated `sam3d-objects` conda env | `../sam-3d-objects/` (env vars `SAM3D_PYTHON`, `SAM3D_REPO`) | Phase D-sam3d single-image 3D recon |
| [FoundationPose-plus-plus](https://github.com/teal024/FoundationPose-plus-plus) | Sibling repo + Docker container `fp_dev` | `../FoundationPose-plus-plus/` | Canonical 6DoF rotation bake (`egoinfinity/pipeline/post_tracking/bake_fp_pose.py`) |
| [YOLO (Ultralytics)](https://github.com/ultralytics/ultralytics) | Pip install (`ultralytics`) | site-packages | Phase B-1 hand detection — **AGPL-3.0** |
| [Flow3R](https://github.com/CVMI-Lab/Flow3R) | Sibling repo (clone + deps; sys.path-loaded) — **opt-in, default off** | `../flow3r/` (env var `FLOW3R_REPO`) | Optional depth refinement (`egoinfinity/pipeline/flow3r_depth.py`) — frame-stable depth fused with MoGe-2 metric anchor |

## Install instructions

### 1. Core env (pip + vendored)

```bash
conda create -n egoinfinity python=3.10 -y
conda activate egoinfinity

# PyTorch (match your CUDA — cu124 default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Repo deps (vendored wilor, sam2 are picked up automatically)
pip install -e .

# Pip-installable third-party libraries
pip install git+https://github.com/microsoft/MoGe.git
pip install git+https://github.com/cvg/GeoCalib.git
pip install git+https://github.com/msu-video-group/memfof
```

### 2. Pretrained weights (vendored + HF)

```bash
bash scripts/setup_weights.sh
```

This downloads (into `${EGOINFINITY_CKPT_DIR}`, default `<repo>/pretrained_models`):
- WiLoR hand detector (`detector.pt`) + encoder / MANO regressor (`wilor_final.ckpt`)
- SAM 2 image / video predictor checkpoint (`sam2.1_hiera_small.pt`)
- HaWoR motion infiller (`infiller.pt`, from the `ThunderVVV/HaWoR` HF repo)

(MoGe-2 weights auto-download from HF on first run.) You also need to
manually obtain MANO (separate license — see README §MANO).

### 3. SAM 3.1 (sibling env, gated weights)

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd EgoInfinity

conda create -n sam3 python=3.12 -y
SAM3_PIP=$(conda info --base)/envs/sam3/bin/pip
$SAM3_PIP install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
$SAM3_PIP install -e ../sam3
$SAM3_PIP install 'setuptools<80'

# Request access at https://huggingface.co/facebook/sam3.1 then:
$(conda info --base)/envs/sam3/bin/hf auth login

export SAM3_PYTHON=$(conda info --base)/envs/sam3/bin/python
export SAM3_REPO=$(pwd)/../sam3
```

### 4. SAM 3D Objects (sibling env)

```bash
cd ..
git clone https://github.com/facebookresearch/sam-3d-objects.git
# Follow that repo's installation guide (separate sam3d-objects conda env).
cd EgoInfinity

export SAM3D_PYTHON=/path/to/sam3d-objects/env/bin/python
export SAM3D_REPO=$(pwd)/../sam-3d-objects
```

### 5. FoundationPose-plus-plus (optional — for canonical R bake)

```bash
cd ..
git clone https://github.com/teal024/FoundationPose-plus-plus.git
cd FoundationPose-plus-plus
# Follow its Docker / install guide.
# Pipeline expects testcase/<clip>/pose.npy output before bake_fp runs.
```

If FP++ is not installed, the `bake_fp` stage is skipped; the canonical
6DoF rotation falls back to the phase_d seed (usable but less stable).

## How the pipeline calls each

| Stage | Code path | External call |
|---|---|---|
| Phase A (depth) | `egoinfinity/pipeline/moge2_estimator.py` | `from moge.model.v2 import MoGeModel` (pip) |
| Phase A-g (gravity) | `egoinfinity/pipeline/gravity_estimator.py` | `from geocalib import GeoCalib` (pip) |
| Phase B-1 (hand det) | `egoinfinity/pipeline/hand_detector.py` | `from ultralytics import YOLO` (pip) |
| Phase B-2 (hand recon) | `egoinfinity/pipeline/hand_reconstructor.py` | `from third_party.wilor.models...` (vendored) |
| Phase C (flow / stab) | `egoinfinity/pipeline/depth_stabilize.py`, `pose_tracker/memfof_flow.py` | `from memfof import MEMFOF` (pip) |
| Phase C+ (infiller) | `egoinfinity/pipeline/motion_infiller.py` | `from .infiller_utils.network import TransformerModel` (vendored algorithm; weights from HF Hub) |
| Phase D-sam3 (det + track) | `egoinfinity/pipeline/sam3_client.py` ↔ `scripts/sam3_worker.py` | Unix socket → `sam3` conda env subprocess |
| Phase D-sam3d (mesh) | `egoinfinity/pipeline/sam3d_client.py` ↔ `scripts/sam3d_worker.py` | Unix socket → `sam3d-objects` conda env subprocess |
| Phase D-track (6DoF) | `egoinfinity/pipeline/pose_tracker/*.py` | (no external; internal) |
| Bake R | `egoinfinity/pipeline/post_tracking/bake_fp_pose.py` + `egoinfinity/pipeline/post_tracking/fp_compose.py` | Reads `FoundationPose-plus-plus/testcase/<clip>/pose.npy` |

## Opt-in / disabled by default

- **Flow3R** (https://github.com/CVMI-Lab/Flow3R) — Temporally-stable depth
  refinement. The hybrid mode (MoGe-2 metric anchor + Flow3R for frame-stable
  depth) is integrated as the `flow3r_depth` stage in the canonical pipeline.
  Default: **off**. Enable via `--set 'flow3r_depth.enabled=true'` or by
  editing `configs/defaults.yaml`. Install:

  ```bash
  # Clone the repo next to EgoInfinity (no pip install needed — Flow3R
  # has no setup.py; we load it via sys.path insert from FLOW3R_REPO).
  cd ..
  git clone https://github.com/CVMI-Lab/Flow3R flow3r

  # Install Flow3R's own deps into the egoinfinity env
  cd flow3r
  /path/to/egoinfinity/bin/pip install -r requirements.txt
  cd ../EgoInfinity
  ```

  The pipeline auto-detects `../flow3r/`; override via `FLOW3R_REPO=/path`.
  HuggingFace model `Clara211111/flow3r` is auto-downloaded on first use.
  VRAM budget: ~9-10 GB at `max_frames=60` (16 GB cards). Set
  `FLOW3R_MAX_FRAMES=0` on A100/H100 for no-cap full-coverage forward.

  Algorithm (`egoinfinity/pipeline/flow3r_depth.py`):
    1. Decode pkl's per-frame RGB + MoGe-2 depth + sam3 dynamic masks
    2. Flow3R forward on K sampled frames → scale-ambiguous local_points
    3. Fit `s_global = median(D_moge / D_flow3r)` on background pixels
    4. Upsample Flow3R × s_global to full resolution
    5. Where Flow3R confidence is low, fall back to MoGe-2
    6. Re-encode fused depth back into pkl's `frame_data[t]['depth_png']`

## License compliance

- The EgoInfinity project's own source is **MIT** (see [`license.txt`](../license.txt)).
- `third_party/sam2/` is vendored and ships its full Apache-2.0 `LICENSE` +
  `NOTICE`. Its ViT backbone is from OpenMMLab/ViTPose (Apache-2.0) — that
  attribution is preserved. The HaWoR-adapted infiller code
  (`infiller_utils/`) carries a `NOTICE` (algorithm adaptation, not a
  verbatim fork); consult the HaWoR repo for its upstream terms.
- **WiLoR is CC-BY-NC-ND (no derivatives), so it is NOT redistributed.**
  `scripts/setup_wilor.sh` fetches it verbatim from upstream (pinned commit)
  and applies `third_party/wilor.patch` locally; only the patch is tracked.
  See [`third_party/README.md`](../third_party/README.md).
- **Ultralytics YOLO is AGPL-3.0** (network copyleft): if you serve the
  pipeline / viz over a network, AGPL §13 source-offer obligations attach.
- Robot model assets (`retarget/robots/`) are third-party (MuJoCo Menagerie /
  ManiSkill / NASA) — see [`retarget/robots/README.md`](../retarget/robots/README.md).
- Pip-installed deps inherit their upstream licenses unchanged. Sibling-repo /
  Docker deps are run in isolation and not redistributed by this repo.
- WiLoR + MANO non-commercial restrictions apply to any downstream use of
  the full pipeline regardless of the MIT grant.
