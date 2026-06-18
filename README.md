# EgoInfinity

**A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning**

**Authors:** Gaotian Wang¹, Kejia Ren¹, Andrew Morgan², Yiting Chen¹, Howard H. Qian¹, Podshara Chanrungmaneekul¹, Kaiyu Hang¹<br/>
¹ Rice University · ² Robotics and AI Institute

<p align="center">
  <a href="https://arxiv.org/abs/2606.17385"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.17385-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/Rice-RobotPI-Lab/egoinfinity"><img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EgoInfinity-FFD21E?style=for-the-badge&labelColor=555"></a>
  <a href="https://huggingface.co/spaces/Rice-RobotPI-Lab/EgoInfinity"><img alt="HF Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Viewer-2EA0F5?style=for-the-badge&labelColor=555"></a>
</p>

<img width="1000" height="408" alt="11" src="https://github.com/user-attachments/assets/3b51c381-6a5c-46cd-84d3-bd0c80c9e409" />


Metric 3D hand-object tracking from a single **static-camera RGB video** —
no depth sensor, no calibration required.  Outputs per-frame 21-joint hand
poses, MANO meshes, per-object 6DoF pose sequences with reconstructed 3D
mesh, and an interactive [Viser](https://github.com/nerfstudio-project/viser)
3D viewer.

| Stage | What | Backbone |
|---|---|---|
| **A** | Metric depth + focal length | [MoGe-2](https://github.com/microsoft/MoGe) (ViT-L) |
| **A-grav** | World up direction | [GeoCalib](https://github.com/cvg/GeoCalib) |
| **B** | Hand detection + MANO recon | YOLO + [WiLoR](https://github.com/rolpotamias/WiLoR) (DINOv2-L) |
| **C** | Optical flow, depth stabilization, alignment, smoothing, joint clamps | [MEMFOF](https://github.com/msu-video-group/memfof), SavGol, biomech swing-twist |
| **C+** | Missing-frame motion infiller | [HaWoR](https://github.com/ThunderVVV/HaWoR) Transformer |
| **D-sam3** | Text-prompted object detection + tracking | [SAM3.1](https://github.com/facebookresearch/sam3) + [SAM2](https://github.com/facebookresearch/sam2) |
| **D-sam3d** | Single-image 3D object reconstruction (Gaussian splat PLY) | [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) |
| **D-track** | 6DoF mesh pose tracking (FGR + ICP anchor → flow + RANSAC PnP propagation → LBFGS smoothing) | Open3D + MEMFOF + custom |

<img width="2094" height="1023" alt="image" src="https://github.com/user-attachments/assets/2b09f2b4-2633-4510-b739-94a4ffc799a4" />

## Documentation

| Doc | Covers |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | **Pipeline architecture deep-dive** — every phase (A → D-track), the canonical post-tracking sequence, PKL v2 format, data layout, hardware budget |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **The `egoinfinity/` orchestration layer** — stage / registry / runner design, resume + clip×stage selectors, and **swapping Phase-1 backbones** (depth / gravity / hand) |
| [docs/MANIFEST.md](docs/MANIFEST.md) | **`manifest.json` field reference** — the per-clip input (`video_uri`, `objects`, …) |
| [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | **Third-party deps, weights & licenses** + Action100M dataset acquisition |
| [docs/MULTI_HOST.md](docs/MULTI_HOST.md) | **Running across machines** — e.g. offload SAM 3D Objects to a bigger GPU when a 16 GB card can't fit it |

---

## Quick start (single GPU, ≥ 16 GB recommended)

```bash
# 1. Clone
git clone https://github.com/Rice-RobotPI-Lab/EgoInfinity.git
cd EgoInfinity

# 2. Conda env (single env for pipeline + retarget)
conda create -n egoinfinity python=3.10 -y
conda activate egoinfinity

# 3. PyTorch (match your CUDA — cu124 default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Repo + deps (editable install)
pip install -e .                        # pulls everything in pyproject.toml
pip install git+https://github.com/microsoft/MoGe.git
pip install git+https://github.com/cvg/GeoCalib.git

# 4b. Retarget deps (same env; skip if you don't need robot retargeting)
pip install mujoco==3.6.0 pytorch-kinematics==0.9.1 roma "PyOpenGL>=3.1.10"

# 5. Fetch WiLoR source (cloned + patched; CC-BY-NC-ND, not redistributed)
#    + download pretrained weights (WiLoR detector + ckpt, SAM2, infiller)
bash scripts/setup_weights.sh

# 6. MANO model — manual download required (see "MANO model" below)

# 7. Environment setup
cp activate.sh.example ~/egoinfinity_env.sh   # then edit paths
source ~/egoinfinity_env.sh

# 8. Run the pipeline on a clip directory (frames/ + manifest.json)
python -m egoinfinity process /path/to/clip_dir/
# add --robot g1 to also retarget to a robot (see "Retarget" below)
```

> A clip directory needs `frames/` (extracted RGB) + `manifest.json`
> (`{"video_uri": ..., "objects": ["red mug", ...]}`). See
> [docs/MANIFEST.md](docs/MANIFEST.md) for the field reference and
> [docs/PIPELINE.md](docs/PIPELINE.md) for the full architecture.

> **System dep**: WiLoR's renderer uses `pyrender` with the EGL backend.
> On Debian/Ubuntu install `libegl1 libgles2 libglvnd0` once.  Headless
> servers don't need an X display.

---

## Iterating on tracking results

After running the pipeline once on a clip, you can resume / re-run any
stage of the post-tracking sequence (anchor, depth smooth, FP++ bake,
spurious detection, etc.) via the unified pipeline entry — no separate
"refresh" CLI needed.

```bash
# Default: resume the pipeline. Skips stages already marked done in
# state.json, runs whatever's pending.
python -m egoinfinity process /path/to/artifacts/CLIP_ID/

# Force a specific stage and everything downstream.
python -m egoinfinity process /path/to/artifacts/CLIP_ID/ \
    --force pose_track_p1 --cascade

# Run one or several stages explicitly (force-run; comma-separated).
python -m egoinfinity run bake_fp /path/to/artifacts/CLIP_ID/
python -m egoinfinity run grasp_veto,pose_track_p2 /path/to/artifacts/CLIP_ID/

# Inspect status (which stages are done, which are stale).
python -m egoinfinity status /path/to/artifacts/CLIP_ID/
```

`run` force-runs the named stage(s); a stage's upstream must already be done,
else it errors (pass `--with-deps` to auto-run the missing upstream first).

The 8-stage canonical post-tracking sequence (see docs/PIPELINE.md §4.4):
`pose_track_p1 → grasp_veto → pose_track_p2 → depth_align → depth_smooth → bake_fp → scale_sanity → spurious`.

You can also re-run a **phase1-internal** substep on an existing pkl (phase1
is monolithic, so these are the only fine-grained re-runs for Phase B/C/D-sam3d):
`refresh_sam3d`, `refresh_hands`, `refresh_hand_scale`, `refresh_optical_flow`.
E.g. cross-host SAM3D fill: `egoinfinity run refresh_sam3d CLIP/` (see docs/MULTI_HOST.md).

### Batch (multiple clips)

Both `process` and `run` accept several clip dirs, or a single **batch root**
whose subdirs are clip dirs (filter with `--only`). Per-clip failures are
isolated (a summary is printed; `--fail-fast` stops at the first):

```bash
python -m egoinfinity process clipA/ clipB/                 # several clips
python -m egoinfinity process favorites/ --only A,B         # batch root, subset
python -m egoinfinity run scale_sanity favorites/           # one stage, all clips
```

The individual algorithm scripts (`egoinfinity/pipeline/post_tracking/pose_tracking.py`,
etc.) and `tools/rerun/refresh_*.py` still run standalone for back-compat,
but the pipeline runner is the recommended entry.

---

## MANO model

**MANO is licensed for non-commercial research use** ([MANO license](https://mano.is.tue.mpg.de/license.html)) — we cannot
redistribute the weights.  Follow these steps once:

1. **Register** at https://mano.is.tue.mpg.de (free, requires email + agreement).
2. After approval, download **"Models & Code"** → `mano_v1_2.zip`.
3. Extract `models/MANO_RIGHT.pkl` to `third_party/wilor/mano_data/`:
   ```bash
   unzip mano_v1_2.zip
   cp mano_v1_2/models/MANO_RIGHT.pkl third_party/wilor/mano_data/
   ```

Without this file Phase B (hand reconstruction) cannot run.  This is the
same convention as [HaWoR](https://github.com/ThunderVVV/HaWoR) and
[WiLoR](https://github.com/rolpotamias/WiLoR) upstream.

---

## Optional components

### SAM3 — text-prompted object detection

SAM3.1 lives in a **separate conda env** (different Python / PyTorch /
CUDA versions).  Weights are gated; **request access first** at
https://huggingface.co/facebook/sam3.1.

```bash
# Sibling repo (clone next to EgoInfinity/)
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd EgoInfinity

# Create the dedicated env
conda create -n sam3 python=3.12 -y
SAM3_PIP=$(conda info --base)/envs/sam3/bin/pip
$SAM3_PIP install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
$SAM3_PIP install -e ../sam3
$SAM3_PIP install 'setuptools<80'   # pkg_resources compat

# Authenticate with HuggingFace once (after access is granted)
$(conda info --base)/envs/sam3/bin/hf auth login
```

Pipeline auto-detects SAM3 at `../sam3/`.  Override via env vars:
```bash
export SAM3_PYTHON=/path/to/sam3/env/bin/python
export SAM3_REPO=/path/to/sam3-repo
```

### SAM 3D Objects — single-image 3D mesh reconstruction

```bash
cd ..
git clone https://github.com/facebookresearch/sam-3d-objects.git
# follow that repo's installation guide (separate sam3d-objects conda env)
cd EgoInfinity

export SAM3D_PYTHON=/path/to/sam3d-objects/env/bin/python
export SAM3D_REPO=/path/to/sam-3d-objects
```

The `sam3d_worker.py` is spawned automatically by the pipeline when the env
vars are set.

> SAM3 object prompts come from the `objects` list in `manifest.json` (one
> short noun phrase per target). See [docs/MANIFEST.md](docs/MANIFEST.md).

---

## Filtering a raw video corpus (optional)

To screen many videos for "static-camera + visible hands" candidates
before running the pipeline, use the visual filter over an Action100M
sqlite index:

```bash
python -m egoinfinity filter --from-db data/action100m_index.db \
    --test_n_videos 500 --viz :8899
```

It runs YOLO hand detection + static-background + optical-flow checks and
serves an optional review UI. (Filtering an arbitrary `--input-list` of
video URLs is planned but not yet wired.) See `action100m_filter/` and
[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) for the dataset acquisition flow.

---

## Retarget (optional)

After tracking, retarget the recovered hand motion to a bilateral robot:

```bash
python -m egoinfinity process /path/to/clip_dir/ --robot g1
# robots: g1 | franka | robonaut2 | xlerobot
# output: <clip>/retarget/<robot>/{trajectory.npz, robot_sim.mp4,
#                                   input_viz.mp4, metrics.npz}
```

This runs in the **same `egoinfinity` env** (needs the step-4b retarget
deps: mujoco 3.6 + pytorch-kinematics + roma + PyOpenGL≥3.1.10). Headless
rendering uses EGL automatically (`MUJOCO_GL=egl`). Pretrained policies
ship in `retarget/ckpts/<robot>.pt`. The `retarget/` subtree is maintained
independently; training (not needed for inference) additionally requires
`jax[cuda12]` + `mujoco-mjx` (see `retarget/README.md`).

---

## Repo layout

```
EgoInfinity/
├── egoinfinity/             # ★ Orchestration layer (python -m egoinfinity) ★
│   ├── cli/                 # process / run / status / filter / import
│   ├── core/                # config, artifacts, state, stage, runner, registry
│   ├── stages/              # Stage wrappers (phase1, post_track.*, retarget, …)
│   └── viz/                 # filter viz server
│
├── egoinfinity/pipeline/        # Algorithm layer (importable; called by stages/scripts)
│   ├── config.py            # Paths + hyperparameters (env-var driven)
│   ├── moge2_estimator.py   # Phase A          gravity_estimator.py  # Phase A-grav
│   ├── hand_detector.py     # Phase B-1        hand_reconstructor.py # Phase B-2
│   ├── depth_align.py / depth_stabilize.py     # Phase C
│   ├── motion_infiller.py   # Phase C+         mano_smoothing.py / biomech_constraints.py
│   ├── object_tracker.py    # Phase D / D-sam3
│   ├── sam3_client.py / sam3d_client.py        # Unix-socket → workers
│   ├── flow3r_depth.py      # opt-in Flow3R depth refinement
│   ├── export_retarget_samples.py              # pkl → retarget SamplesSequence
│   ├── pose_tracker/        # ★ Phase D-track 6DoF mesh tracking ★
│   ├── post_tracking/       # canonical 8-stage post-tracking algorithms
│   ├── interfaces/          # backend Protocol contracts
│   └── backends/            # swappable-backend registry
│
├── scripts/
│   ├── exo_pipeline.py      # Phase-1 subprocess entry (A→E)
│   ├── pipeline_utils.py    # free_gpu / image / pkl helpers
│   ├── sam3_worker.py / sam3d_worker.py        # SAM3 / SAM 3D Objects workers
│   ├── sam3_detect_cli.py   # SAM3 single-image fallback (slow path)
│   └── setup_weights.sh     # Download pretrained checkpoints
│
├── tools/                   # batch_pipeline + standalone re-run helpers
├── action100m_filter/       # Generic static-cam + hand video filter
├── retarget/                # MuJoCo / IK retargeting to robot arms (independent)
├── third_party/             # sam2 (vendored); wilor (fetched at install, see setup_wilor.sh)
├── configs/                 # defaults.yaml + wilor_model_config.yaml
├── docs/                    # PIPELINE.md, ARCHITECTURE.md, MANIFEST.md, … (see "Documentation")
├── pretrained_models/       # Gitignored; populated by setup_weights.sh
├── activate.sh.example      # Env template — copy + edit, then `source`
├── pyproject.toml / requirements.txt / license.txt
└── README.md
```

---

## Environment variables

All paths are env-driven so the repo is portable.  See
[`activate.sh.example`](activate.sh.example) for the template.

| Variable | Default | Purpose |
|---|---|---|
| `EGOINFINITY_CKPT_DIR` | `<repo>/pretrained_models` | Where WiLoR / SAM2 / infiller .pt files live |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache (MoGe-2, DINOv2, SAM3.1, MEMFOF) |
| `SAM3_PYTHON`, `SAM3_REPO` | `~/miniconda3/envs/sam3/bin/python`, `<repo>/../sam3` | SAM3 worker |
| `SAM3D_PYTHON`, `SAM3D_REPO` | `<repo>/../sam-3d-objects` | SAM 3D Objects worker |
| `YTDLP_COOKIES_FILE` | (empty) | Path to YouTube cookies (Netscape format) for Action100M downloader |
| `EGOINFINITY_RUN_SAM3D` | `1` | Set `0` on 4070 Ti — skip Phase D-sam3d (see [docs/PIPELINE.md](docs/PIPELINE.md) §4.2 multi-host note) |
| `EGOINFINITY_RUN_TRACK` | `1` | Set `0` to skip Phase D-track |
| `EGOINFINITY_LOW_VRAM` | `0` | Set `1` on small GPUs — lazy-load SAM3 / SAM3D workers to cap residency |
| `EGOINFINITY_HAND_TTA` | `0` | Set `1` for multi-scale (640/960/1280) YOLO TTA on hand detection — costs ~3× detector latency, boosts recall on fast-motion / oblique-angle clips |
| `SAM3D_SPAWN_TIMEOUT` | `240` | Bump to `600` on slow shared-FS clusters (Delta Lustre etc.) |
| `ACTION100M_CACHE` | `<repo>/cache` | Override the favorites cache root (used by `batch_pipeline` + curation tools) |
| `ACTION100M_DB` | `<repo>/data/action100m_index.db` | Override the Action100M SQLite index path (used by `action100m_filter`) |

---

## Hardware

Designed for a single A100 (40 GB) or H100 (80 GB).  When Phase D is
enabled, two persistent GPU workers add ~14 GB residency:

| Worker | Process | Resident VRAM |
|---|---|---|
| SAM3.1 | `sam3_worker.py` (sam3 env subprocess) | ~4 GB |
| SAM 3D Objects | `sam3d_worker.py` (sam3d-objects env subprocess) | ~10 GB |
| **Pipeline** (active phase) | `exo_pipeline.py` (sequential model load) | 3-5 GB peak |

Smaller GPUs are not officially supported in this revision; SAM3 / SAM3D
would need lazy load + idle-unload (planned — see `EGOINFINITY_LOW_VRAM`).

---

## License

This repo bundles or links code under several licenses:

| Component | License |
|---|---|
| Project source (Python, etc.) | **MIT** — see `license.txt` |
| WiLoR (Phase B; fetched at install, not redistributed) | [CC-BY-NC-ND](https://github.com/rolpotamias/WiLoR/blob/main/LICENSE) — research-only, no derivatives |
| SAM2 (`third_party/sam2/`) | Apache 2.0 |
| Ultralytics YOLO (`ultralytics` dep) | **AGPL-3.0** (network copyleft) |
| MANO model (manual download) | Non-commercial research only |
| Robot assets (`retarget/robots/`) | Derived from MuJoCo Menagerie / ManiSkill / NASA — see [retarget/robots/README.md](retarget/robots/README.md) |
| MoGe-2 / GeoCalib / SAM3.1 / SAM 3D Objects / MEMFOF | each upstream's license |

The project's own code is MIT, but **commercial use of this repo as a whole
is restricted** by the WiLoR (CC-BY-NC-ND) and MANO (non-commercial) terms,
and the bundled Ultralytics YOLO is AGPL-3.0 (its network-copyleft applies if
you serve the pipeline over a network). See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

---

## Citation

```bibtex
@misc{egoinfinity_2026,
  title         = {EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning},
  author        = {Wang, Gaotian and Ren, Kejia and Morgan, Andrew and Chen, Yiting and Qian, Howard H. and Chanrungmaneekul, Podshara and Hang, Kaiyu},
  year          = {2026},
  eprint        = {2606.17385},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.17385}
}
```

Plus the upstream backbones — please cite WiLoR / MoGe-2 / SAM2 / SAM3 /
HaWoR / MANO as appropriate to the components you use.

---

## Acknowledgements

Built on:
- [WiLoR](https://github.com/rolpotamias/WiLoR) — Hand localization & MANO recon (Potamias et al., CVPR 2025)
- [HaWoR](https://github.com/ThunderVVV/HaWoR) — World-space hand motion + infiller (CVPR 2025)
- [MoGe-2](https://github.com/microsoft/MoGe) — Monocular metric depth (Microsoft)
- [GeoCalib](https://github.com/cvg/GeoCalib) — Gravity / camera calibration (Phase A-grav)
- [SAM2 / SAM3 / SAM 3D Objects](https://github.com/facebookresearch/) (Meta)
- [MEMFOF](https://github.com/msu-video-group/memfof) — Optical flow (Phase C / D-track)
- [Open3D](https://github.com/isl-org/Open3D) — 6DoF pose tracking geometry (Phase D-track)
- [HaMeR](https://github.com/geopavlakos/hamer/) — Hand reconstruction backbone
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO hand detection
