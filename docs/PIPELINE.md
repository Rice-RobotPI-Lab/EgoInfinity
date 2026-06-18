# EgoInfinity Pipeline — Architecture Reference

> Module-level deep dive — pair with [README.md](README.md) for setup.

---

## 1. What the system does

Given **a single static-camera RGB video** (typically 4 s at 15 fps,
480×854 px), produce:

- per-frame **metric depth map**(MoGe-2)
- per-frame **21-joint MANO hand pose + 778-vert mesh**, both hands
- per-frame **dynamic-mask + background reconstruction**
- per-text-prompt **object SAM3.1 detection + SAM2 streaming track**
  (mask + 3D OBB + 6DoF pose)
- per-object **3D mesh** (Gaussian splat PLY, from SAM 3D Objects)
- a **gradable trust signal** + grasp / contact time series for
  downstream HOI analysis
- an interactive **viser 3D viewer** showing all of the above synced
  to a frame slider

The output bundle (`pipeline_result.pkl.gz`) is ~20-40 MB after
**format v2** compression (was 100-200 MB in v1).

The 6DoF object pose published in the public HF demo is the result of a
*hybrid* tracker: position from EI's own `phase_d` state machine (§3
D-track), orientation from FoundationPose++ composed through a 7-step
R chain. The main inference (§3) writes the first-pass tracker output;
the canonical pose seen on HF is produced by the post-tracking refresh
sequence in §4.

---

## 2. Top-level repo layout

```
EgoInfinity/
├── egoinfinity/pipeline/         # Importable Python package — all phase modules
│   ├── config.py             # Paths + constants, env-var driven
│   ├── moge2_estimator.py    # Phase A    — MoGe-2 wrapper
│   ├── gravity_estimator.py  # Phase A-g  — GeoCalib wrapper
│   ├── hand_detector.py      # Phase B-1  — YOLO bbox + handedness
│   ├── hand_reconstructor.py # Phase B-2  — WiLoR MANO recon
│   ├── depth_align.py        # Phase C-1  — multi-scale hand-depth align
│   ├── depth_stabilize.py    # Phase C-2  — flow + temporal stab
│   ├── motion_infiller.py    # Phase C+   — HaWoR Transformer
│   ├── biomech_constraints.py# Phase C++  — swing-twist clamp
│   ├── mano_smoothing.py     # Phase C′   — SavGol on MANO params
│   ├── object_tracker.py     # Phase D, D-sam3 — SAM2 streaming + OBB
│   ├── sam3_client.py        # Unix-socket → sam3 worker + NMS + containment_merge
│   ├── sam3d_client.py       # Unix-socket → sam3d worker
│   ├── prompt_extractor.py   # (deprecated) prompts now come from manifest.json `objects`
│   ├── llm_extractor.py      # (deprecated) former on-GPU Qwen path — raises; kept as failure path
│   ├── pipeline_state.py     # Per-clip phase tracker (read-only API)
│   ├── infiller_utils/       # HaWoR network + rotation + filling
│   └── pose_tracker/         # ★ Phase D-track 6DoF mesh tracking ★
│       ├── anchor.py             # FGR + ICP anchor pose
│       ├── flow_pnp.py           # Optical flow + RANSAC PnP propagation
│       ├── memfof_flow.py        # MEMFOF deep flow engine
│       ├── orientation_search.py # 24-cube symmetry rotation search
│       ├── trust_filter.py       # IoU + Chamfer + inlier ratio
│       ├── mask_completion.py    # Static-object mask repair
│       ├── grasp.py / contact.py # Contact + motion-correlated grasp
│       ├── hand_driven.py        # Mask + wrist anchor + Kabsch
│       ├── fill_optimize.py      # 7-loss LBFGS pose-seq optimizer
│       ├── smooth.py             # SavGol on T_seq
│       └── debug_viz.py
│
├── scripts/
│   ├── exo_pipeline.py       # ★ Pipeline subprocess entry + Viser server ★
│   ├── pipeline_utils.py     # free_gpu / draw / pkl rehydrate
│   ├── sam3_worker.py        # SAM3.1 persistent worker (sam3 env)
│   ├── sam3d_worker.py       # SAM 3D Objects worker (sam3d-objects env)
│   ├── sam3_detect_cli.py    # SAM3 one-shot fallback (slow path)
│   └── setup_weights.sh      # Download WiLoR / SAM2 / infiller
│
├── egoinfinity/              # ★ Orchestration layer (python -m egoinfinity) ★
│   ├── cli/                  # process / run / status / filter / import
│   ├── core/                 # config, artifacts, state, stage, runner, registry
│   ├── stages/               # Stage wrappers (phase1, post_track.*, retarget, …)
│   └── viz/                  # filter viz server (stdlib http.server)
│
├── action100m_filter/        # Static-cam + visible-hands video filter
├── tools/                    # batch_pipeline + standalone re-run helpers
├── retarget/                 # MuJoCo / IK retargeting (sub-research)
├── third_party/
│   ├── wilor/                # Trimmed WiLoR fork
│   └── sam2/                 # SAM2 streaming API (modified)
├── configs/
│   ├── defaults.yaml         # Pipeline DAG + per-stage backends
│   └── wilor_model_config.yaml
├── pretrained_models/        # gitignored; populated by setup_weights.sh
├── activate.sh.example       # Env template
└── pyproject.toml
```

(The `egoinfinity/pipeline/` tree above is the importable algorithm layer;
the `egoinfinity/` cli/core/stages/viz packages are the orchestration layer
that drives it.)

---

## 3. Pipeline phases

`python -m egoinfinity process <clip>` runs the pipeline. Phase 1 (A→E)
executes as a `scripts/exo_pipeline.py` subprocess, which sequentially
loads-and-releases each model so peak GPU is bounded:

| Phase | Module(s) | Compute | GPU peak | I/O |
|---|---|---|---|---|
| **A** Metric depth + focal | `moge2_estimator.py` | ViT-L fp16, per-frame | ~3 GB | GPU |
| **A-grav** World up | `gravity_estimator.py` | small ResNet on 3 sample frames | < 1 GB | GPU |
| **B-1** Hand bbox + handedness | `hand_detector.py` | YOLO ultralytics | < 1 GB | GPU |
| **B-2** MANO recon | `hand_reconstructor.py` | WiLoR DINOv2-L fp16, per-detection | ~2.5 GB | GPU |
| **B+** Handedness vote / dedupe / short-track filter | `exo_pipeline.py` | numpy | — | CPU |
| **C** Optical flow + dynamic mask + bg template + multi-scale depth align | `depth_stabilize.py`, `pose_tracker/memfof_flow.py`, `depth_align.py` | MEMFOF fp16 | ~2.5 GB | GPU |
| **C+** Motion infiller (gap-fill missing hand frames) | `motion_infiller.py` | ~35 M-param Transformer | < 1 GB | GPU |
| **C++** Biomech swing-twist clamp | `biomech_constraints.py` | torch CPU | — | CPU |
| **C′** Spike rejection + SavGol on joints / MANO params | `depth_stabilize.py`, `mano_smoothing.py` | scipy | — | CPU |
| **D-sam3** Text-prompt object detect + SAM2 forward/backward streaming. Post-detect: prompt-aware NMS + `containment_merge` (drops `"X"` when subsumed by `"X of Y"`), `max_keep=7`. Also writes `joints_2d_pred` (WiLoR-native pixel-space joints, no MoGe-2 focal projection). | `sam3_client.py` (→ socket) + `object_tracker.py` | SAM3 fp16 (worker) + SAM2 bf16 | 4 GB worker + 1.5 GB inline | GPU |
| **D-sam3d** Per-object 3D Gaussian-splat mesh | `sam3d_client.py` (→ socket) | SAM 3D Objects (worker) | 10 GB worker | GPU |
| **E** Object-aware mask refine + bg fill | `depth_stabilize.py` | numpy / cv2 | — | CPU |
| **D-track** 6DoF mesh pose tracking (first pass; canonical R later overwritten by FP++ bake, see §4) | `pose_tracker/*` | MEMFOF + Open3D + RANSAC + LBFGS | 1.5 GB flow / LBFGS optional | mixed |
| **Build frame_data + save** | `exo_pipeline.py` | numpy / cv2 / threadpool JPG encode | — | CPU + IO |

Total wall-clock for a 4 s / 80-frame clip on a warm A100: **~80-120 s**
(perf optimizations from the recent A-class pass shave ~10-15 s).

---

## 4. Object pose tracking — three-part model + canonical sequence

`scripts/exo_pipeline.py` (§3) writes a first-pass
`pose_track_info[oid]['T_seq']` during Phase D-track. The **canonical**
6DoF pose published on HF is then produced by a post-tracking refresh
sequence that runs *after* the main inference, fusing in
FoundationPose++. There is no single entry point; the steps in §4.4 are
the de-facto sequence the favourites maintenance flow has converged on.

This section is the reference for both **understanding the algorithm**
and **running it on new clips**.

### 4.1 Conceptual three-part model

Once `sam3_mesh_info[oid]` exists (Phase D-sam3d, §3), object tracking
decomposes into three parts:

```
              SAM-3D mesh + SAM-2 masks + MoGe depth + MANO
                              │
                              ▼
              ┌────────────────────────────────────┐
              │  A · Status detection              │
              │  per-frame state ∈ {STATIC,        │
              │   MOVING_NOT_GRASPED, GRASPED}     │
              │  + wrist/hand-driven flags         │
              └─────────┬───────────────┬──────────┘
                        │               │
              ┌─────────▼──────┐  ┌─────▼────────────────────┐
              │ B · Position t │  │ C · Orientation R        │
              │ obs centroid   │  │ R_anchor + R_obb + R_fp  │
              │ + depth_pose   │  │ fused by fp_compose 7-   │
              │ + z-smooth     │  │ step chain, gated by A   │
              │ + state lock   │  │                          │
              │ + hand-rigid   │  │                          │
              └────────┬───────┘  └────────┬─────────────────┘
                       │                   │
                       └─────────┬─────────┘
                                 ▼
                pose_track_info[oid]['T_seq']
                t  = T_seq[:, :3, 3]   (owned by B)
                R  = T_seq[:, :3, :3]  (owned by C)
```

A planned **Stage 0 · Depth Alignment** would sit upstream of A; see §4.7.

### 4.2 Mapping the three parts to actual code

The A / B / C boundaries are NOT clean module boundaries in the current
code. A and B are computed together in a single pass (`phase_d`), and
chunks of A are recomputed a second time inside `fp_compose.load_clip`.
This subsection is the source of truth for *which module owns which
signal*, so future changes preserve the right invariants.

#### Part A — Status detection

| Signal | Module | Notes |
|---|---|---|
| `is_moving_per_frame` | `pose_tracker/object_motion.py` | OR of MEMFOF mean flow in `obj_mask ∖ hand_mask` (lock 1.0 px/f, fast 3.0 px/f) and PC bbox-center displacement (lock 0.015 m/f) |
| Hysteresis + min-segment | `pose_tracker/hysteresis.py` | SavGol(7,2) on raw signal, decide at 0.5, min segment 5 frames |
| Stability gates (per frame) | `pose_tracker/object_motion.py` | mask ≥ 800 px, PC density ≥ 0.30, depth var ≤ 1e-3 m² |
| Grasp detector #1 (mask-overlap + motion) | `pose_tracker/grasp.py:detect_grasp_with_motion` | contact ≥ 30 px, hand disp ≥ 30 px, obj/hand ratio < 0.3, cos sim ≥ 0.5 |
| Grasp detector #2 (proximity + motion) | `pose_tracker/grasp.py:detect_grasp_proximity_motion` | wrist↔cloud < 0.10 m AND object moving |
| Grasp detector #3 (fingertip persistent) | `pose_tracker/grasp.py:detect_grasp_fingertip_persistent` | tips < 0.04 m, bridge ≤ 5 f, min run 8 f |
| Contact soft signal | `pose_tracker/contact.py` | hand mask dilate(3) ∩ obj mask ≥ 30 px → SavGol(7,2) → [0,1] |
| Grasp veto (false-positive removal) | `egoinfinity/pipeline/post_tracking/grasp_veto.py` | low=2.0 / high=4.0 px/f hysteresis, threshold 0.85, min_seg 3; writes `wrist_l_per_frame` / `wrist_r_per_frame` back into pkl. When `is_static_global=True` it zeros wrist_l/r entirely (a globally-static object cannot be a real grasp) — earlier versions left the raw flags in place, which produced "state=static but object follows the hand" in viser. Fixed 2026-05-24. |
| `palm_angular_per_frame` | `egoinfinity/pipeline/post_tracking/fp_compose.py:compute_palm_angular` | Per-frame Kabsch on 6 palm KP (wrist + MCP {1,5,9,13,17}); 0 on SVD non-convergence (guarded since 2026-05) |
| `state_per_frame ∈ {STATIC, MOVING_NOT_GRASPED, GRASPED}` | `egoinfinity/pipeline/post_tracking/fp_compose.py:compute_state_per_frame` | combines `is_moving` and `is_grasp` |

> **Duplication note.** Status is computed twice today. Once by
> `egoinfinity/pipeline/post_tracking/pose_tracking.py:_process_clip` (writes `state_per_frame`,
> `wrist_l_per_frame`, `wrist_r_per_frame`, `wrist_used_per_frame`,
> `is_moving_per_frame`, `close_per_frame`, `is_static_global`,
> `centroid_2d_*_per_frame`, `global_span_px`, `state_counts` into
> `pose_track_info[oid]` via `_estimate_state_per_frame`), and again by
> `egoinfinity/pipeline/post_tracking/fp_compose.py:load_clip` (re-derives `is_grasp` via
> `derive_hand_signals`, computes `palm_angular`, recomputes `motion_score`
> from obs bbox if pkl `pc_motion_xy/z` is missing/zero, builds its own
> `state_per_frame`). Keep them in sync; if you change one, mirror to the
> other or bake will diverge from phase_d's intent.
>
> **Preserve-veto rule (Pass 2 of phase_d).** `_process_clip` carries an
> existing veto-bearing state forward verbatim instead of re-estimating.
> The marker is `wrist_l_per_frame` AND `state_per_frame` already in the
> prior `pose_track_info[oid]`. Without this guard, the second phase_d
> pass would overwrite `refresh_grasp_veto`'s output. Inside the state
> machine the same vetoed `wrist_l/r_per_frame` arrays are also threaded
> back in via `existing_wrist_l_per_frame` / `existing_wrist_r_per_frame`
> so WRIST mode only triggers on vetoed grasp frames.

#### Part B — Position tracking (t)

Translation is set by `phase_d` and **never touched** by FP++ in the
default bake.

| Step | Module | Behavior |
|---|---|---|
| Primary t (every frame) | `pose_tracker/obs_obb.py`, `pose_tracker/depth_pose.py` | t = bbox center of obs cloud, where obs cloud = `unproj(depth_t × SAM-2 mask_t)`; mesh bbox-center aligned to obs bbox-center |
| Grasp / wrist override | `pose_tracker/hand_driven.py` | When `state ∈ {GRASPED, WRIST}`, rigid_wrist_binding_propagation slaves t to palm via Kabsch on 6 palm KP; hand-snap at segment start (target gap 5 mm, max 100 mm) |
| Static lock | `pose_tracker/anchor.py` + `phase_d` state machine | When `state == STATIC`, t locked to `t_anchor` (= FGR+ICP register, see Part C) |
| z-smoothing (post-hoc) | `egoinfinity/pipeline/post_tracking/depth_smooth.py` | Gaussian σ_z = 12 on `T_seq[:, 2, 3]`; same Δz also translates cached cloud `pts[:, 2]` and OBB `corners[:, 2]`; NaN-tolerant convolution; MAD ×5 spike rejection on hand reliable joints {0,5,9,13,17} |
| Sanity cap | `pose_tracker/flow_pnp.py` | Reject translation jump > 0.20 m/frame |
| Write target | `pose_track_info[oid]['T_seq'][:, :3, 3]` | Bake forces `do_obs_anchor=False` and `do_ei_t=False`, so this column is the exclusive output of Part B |

#### Part C — Orientation tracking (R)

R has three independent estimators, fused in
`egoinfinity/pipeline/post_tracking/fp_compose.py:compose_display_pose`, written via
`egoinfinity/pipeline/post_tracking/bake_fp_pose.py`.

**Three R sources:**

| Source | Where produced | Coverage |
|---|---|---|
| `R_anchor` (SAM-3D canonical) | `pose_tracker/anchor.py` (FGR + ICP at `init_frame`, with Umeyama scale check; SOR k=20, std=2.0 on obs cloud) | one frame; reference for everything else |
| `R_obb` (per-frame obs PCA) | `pose_tracker/obs_obb.py` (PCA on `unproj(depth × SAM-2 mask)` cloud, forward + backward sign sweep) | every frame whose OBB passes trust gate: λ₂/λ₁ ≤ 0.92, λ₃/λ₁ ≥ 0.015, n_pts ≥ 200 |
| `R_fp` (FoundationPose++) | optional external FP++ dependency (see [docs/THIRD_PARTY.md](THIRD_PARTY.md)); per-(clip, oid) `pose.npy`. Look-up via `egoinfinity/pipeline/post_tracking/bake_fp_pose.py:_fp_testcase_for_oid` (matches the per-oid marker; if FP++ output is absent, bake is skipped) | every frame from `init_frame` onward; NaN before |

**fp_compose 7-step R chain** (`compose_display_pose`, applied in order):

| # | Function | Role | Key constants |
|---|---|---|---|
| 1 | `align_to_ei_canonical` | Resolve FP++ 180° symmetry pick against EI R at `init_frame` | apply only if angle improves ≥ 60° |
| 2 | `remove_180_flips` | Per-frame symmetry jitter filter | jump > 60°/f, improvement > 30° |
| 3 | `pca_anchored_rotation` | Non-grasp frames: replace FP++ R with obs PCA R, sign-corrected to anchor | skips frames where OBB trust gate fails |
| 4 | `hand_rigid_grasp_lock` | Grasp frames: Kabsch on 6 palm KP; reference R from `T_anchor.R` if `init_frame ∈ segment`, else chordal mean of pre-grasp PCA Rs (snap to T_anchor if within 30°) | freeze segment if obj motion < 50 mm AND palm angular < 1.0°/f |
| 5 | `obb_priority_lock` | Override Kabsch with obs PCA when mask IoU > 0.85 AND OBB trustworthy | OBB_TRACK dR < 15°, OBB_STATIC dR < 1° |
| 6 | `state_aware_lock` | STATIC frames → `T_anchor` verbatim; snap to T_anchor if within 45° | strict `state == STATIC` |
| 7 | `smooth_se3` | Final SavGol on R | window = 9, polyorder = 3 |

CLI toggles on `egoinfinity.pipeline.post_tracking.bake_fp_pose`: `--no-pca`, `--no-hand-rigid`,
`--no-state-lock`, `--no-smooth`. The compose function also has
`do_obs_anchor` and `do_ei_t` toggles that would let R-chain touch t;
both are forced OFF in `bake_fp_pose` so Part B stays the exclusive
owner of `T_seq[:, :3, 3]`.

**Write target:** `pose_track_info[oid]['T_seq'][:, :3, :3]`. R-only
bake by default; pre-`init_frame` NaN frames keep the upstream
phase_d R.

**All-NaN FP++ fallback.** If `pose.npy` for an oid is all-NaN
(FP++ register failed entirely), compose still runs but
`bake_fp_pose` per-frame replaces NaN `new_R[t]` with `old_R[t]`
(= phase_d R). Net effect: bake is a no-op for that oid; R stays at
phase_d output. As of 2026-05-25 all 106 baked clips have valid FP++
output (the cloth-like geometry oids on `-LxWGDmwOMY` and `-Phcfypd7tE`
were re-registered successfully on the latest pass).

### 4.3 Where the three parts get interleaved in execution

The conceptual A → (B || C) flow is implemented as:

1. **A + B together** by `egoinfinity/pipeline/post_tracking/pose_tracking.py:_track_phase_d`
   in a single pass per oid. It writes both `is_*` signals (part of A)
   and a full `T_seq` (both t for B and a placeholder R) into
   `pose_track_info[oid]`.
2. **Veto pass** by `egoinfinity/pipeline/post_tracking/grasp_veto.py` post-edits A's grasp
   flags using a sharper flow gate.
3. **A + B again** in a second `phase_d` pass, picking up the vetoed
   flags via `wrist_l/r_per_frame`.
4. **C** by `egoinfinity/pipeline/post_tracking/bake_fp_pose.py` reading the FP++ side's
   `pose.npy`, building a fresh `ClipData` (which recomputes a copy of
   A internally), running `compose_display_pose`, and overwriting the
   R column of `T_seq`. t is preserved.
5. **z-smoothing for B** by `egoinfinity/pipeline/post_tracking/depth_smooth.py` (typically
   already applied earlier; idempotent via `depth_smooth_refresh.history`).
6. **Sanity passes**: `refresh_scale_sanity`, `refresh_spurious_filter`.

Step 1 + 3 is "phase_d Pass 1 + veto + Pass 2". Step 4 must run after
Step 3 because `fp_compose.load_clip` reads `is_grasp_per_frame`,
`pc_motion_*`, `anchor_t`, and `R_anchor` out of the vetoed
`pose_track_info`.

### 4.4 The canonical sequence

**FP++ rotation source (optional, external):**

The `R_fp` orientation input to bake (Part C) comes from the optional
external FoundationPose++ dependency (see [docs/THIRD_PARTY.md](THIRD_PARTY.md)).
FP++ registers each object at `init_frame` and forward-tracks, producing a
per-(clip, oid) `pose.npy`. If FP++ output is absent, `bake_fp_pose` is
skipped and the phase_d-only rotation is kept (translation is unaffected).

> **Multi-host note.** Phase D-sam3d (SAM 3D Objects, ~10 GB worker) only
> fits on a 16 GB+ card. On a <16 GB GPU host, set `EGOINFINITY_RUN_SAM3D=0`
> so `exo_pipeline.py` skips Phase D-sam3d; fill the SAM3D meshes later on a
> larger host. See [docs/MULTI_HOST.md](MULTI_HOST.md).

**Canonical post-tracking (per clip):**

The canonical 8-stage post-tracking sequence is invoked via the unified
pipeline runner (`egoinfinity process <artifacts>` — uses the resume mechanism
to pick up after `phase1` is done). The individual algorithm modules can
also be run standalone:

```
python -m egoinfinity.pipeline.post_tracking.pose_tracking    --only=<CLIPS> --mode=phase_d  # Pass 1
python -m egoinfinity.pipeline.post_tracking.grasp_veto       --only=<CLIPS> --force
python -m egoinfinity.pipeline.post_tracking.pose_tracking    --only=<CLIPS> --mode=phase_d  # Pass 2
python -m egoinfinity.pipeline.post_tracking.depth_align      --only=<CLIPS>
python -m egoinfinity.pipeline.post_tracking.depth_smooth     --only=<CLIPS>
python -m egoinfinity.pipeline.post_tracking.bake_fp_pose     --only=<CLIPS> --force
python -m egoinfinity.pipeline.post_tracking.scale_sanity     --only=<CLIPS>
python -m egoinfinity.pipeline.post_tracking.spurious_filter  --only=<CLIPS> --mode soft
```

Why Pass-1 / veto / Pass-2: `phase_d` (in
`egoinfinity/pipeline/post_tracking/pose_tracking.py:_track_phase_d`) reads
`wrist_l_per_frame` / `wrist_r_per_frame` from the *previous* pose-track
run; those flags are produced by `grasp_veto` and disqualify
false-positive grasp frames so they fall back to DEPTH_TRACKED instead of
WRIST mode. Hence: track, veto, track again.

Why bake after Pass-2: `egoinfinity/pipeline/post_tracking/fp_compose.py:load_clip` pulls
`is_grasp_per_frame`, `pc_motion_*`, `anchor_t`, and `R_anchor` out of
`pose_track_info` to drive the fp_compose state machine, so it needs the
veto-aware Pass-2 output.

### 4.5 Provenance + "is this clip canonical?" check

| pkl field | Writer | Notes |
|---|---|---|
| `pose_track_info[oid]`         | `refresh_pose_tracking`    | overwritten every refresh; state fields preserved when prior veto exists |
| `pose_track_info_meta`         | `refresh_pose_tracking`    | carries `updated_at`, `mode`, `algo_version` |
| `grasp_veto_refresh.history`   | `refresh_grasp_veto`       | per-call ts, `low/high_px`, `static_frac_thr`, per-oid `n_before/n_after/n_segs_vetoed` |
| `depth_smooth_refresh.history` | `refresh_depth_smooth`     | per-call ts |
| `scale_sanity_refresh.history` | `refresh_scale_sanity`     | per-call ts, per-oid scale override stats |
| `spurious_filter_refresh.history` | `refresh_spurious_filter` | per-call ts, mode, per-oid `spurious_flag` / `spurious_reason` |
| `fp_pose_bake.history`         | `bake_fp_pose`             | per-call ts, `n_oid_baked`, `n_oid_missing`, `r_change_deg_mean/max` |

A clip is **canonical** iff
`fp_pose_bake.history[-1].ts ≥ pose_track_info_meta.updated_at`. If the
pose-track ts is newer (e.g. a phase_d refresh ran after the last bake),
R has been overwritten by phase_d alone and a re-bake is required.

`bake_fp_pose --skip-if-done` reads the history; `--force` re-bakes.

Timestamp parsing gotcha: `fp_pose_bake.history[*].ts` uses
`±HHMM` timezone offsets (e.g. `2026-05-24T18:06:30-0500`), which Python
3.10's `datetime.fromisoformat` does NOT accept. Normalize to `±HH:MM`
before parsing.

### 4.7 What can go wrong

- **Pose_track newer than fp_bake** — someone ran `refresh_pose_tracking`
  after `bake_fp_pose`. Re-bake to recover canonical R.
- **`bake_fp_pose` SKIPs "no oids baked"** — the optional FP++ dependency
  produced no `pose.npy` for that clip, or the output is empty. The
  mesh-side oid in the EI pkl and the FP++ oid must agree;
  `egoinfinity/pipeline/post_tracking/bake_fp_pose.py:_fp_testcase_for_oid` does the lookup.
  See [docs/THIRD_PARTY.md](THIRD_PARTY.md).
- **Empty `pose_track_info_meta`** — clip never made it past Phase D-sam3
  (no SAM-3D mesh, no pose). Re-run main inference, or drop the clip.
- **Changing `manifest.objects` requires wiping stale artifacts** — the
  oid set inside the pkl is rebuilt by `batch_pipeline --force`, but the
  following per-clip caches keep old per-oid files and silently mix old +
  new data in the next run. Delete both before re-running Stage 1:
    - `favorites/<clip>/sam3_meshes/` (SAM-3D PLYs, indexed by oid)
    - `favorites/<clip>/sam3_quality.json`

### 4.8 Planned: `refresh_depth_align` (hand-object Z alignment)

**Why.** During WRIST mode the object t is slaved to the palm via Kabsch
on 6 palm KP. MANO joint depths are more reliable than MoGe-2 object
depth (the rigid kinematic chain anchored at the wrist absorbs less mono-
depth bias than a textured surface backprojection). Result: a held
object's z drifts by 1-3 cm relative to where the fingers actually meet
it, visible as the mesh "hovering above" or "sinking into" the palm in
viser. This refresh corrects that drift with no model changes — it's
pure post-processing on `T_seq[:, 2, 3]`.

**Where it slots in.** Between phase_d Pass 2 and `bake_fp_pose`:

```
phase_d Pass 1 → veto → phase_d Pass 2 → depth_align → bake → scale_sanity → spurious_filter
```

Must run before bake because bake's R-chain reads `T_seq` to drive
`hand_rigid_grasp_lock` and `state_aware_lock` decisions.

**Per-oid, per-WRIST-frame algorithm.** Δz is computed from the
**actual hand-object contact patch**, not a hard-coded palm/fingertip
formula — different grasp types (power vs. precision vs. lateral pinch)
have different contact regions, and a fixed weight would mis-target any
non-power grasp.

| Step | Behavior |
|---|---|
| 1. Hand vertices | Take all 778 MANO vertices in world frame for the active hand(s) — from `frame_data[t]['vertices_3d']` joined with `hand_is_right` and `grasp_hand_per_frame`. For `grasped_both`, run each hand independently and weight-average the per-hand Δz by contact count. |
| 2. Contact set | KDTree-query each hand vertex against the obs object cloud; mark vertices within `contact_radius_m` (default 0.03 m). If `contact_count < min_contact_verts` (default 5) → skip frame (no real contact, residual veto FP). |
| 3. `hand_z[t]` | Mean z of contact-set hand vertices in world frame. |
| 4. `object_z[t]` | Mean z of the obs-cloud points that each contact-set vertex pointed to (paired by the KDTree query). |
| 5. `raw Δz[t]` | `hand_z − object_z` (target: shift the object up/down so its contact-surface z matches the hand's). |
| 6. Segment smooth | SavGol(window=7, polyorder=2) on `Δz` within each WRIST segment to absorb contact-set membership noise. |
| 7. Edge ramp | First / last 3 frames of each segment linearly ramp 0 ↔ smoothed-Δz, so adjacent STATIC LOCK / DEPTH_TRACKED frames don't see a step change. |
| 8. Cap | `|Δz[t]| > max_delta_m` (default 0.10 m) → drop to 0 (outlier in hand mesh or obs cloud; don't pull mesh that far). |
| 9. Apply | `T_seq[t, 2, 3] += Δz[t]`. Mirror Δz onto the per-frame cached `frame_data[t]['sam3_obj_data'][oid]['pts'][:, 2]`, `pose_t[2]`, and `obb_corners[:, 2]` (same invariant as `refresh_depth_smooth`). |
| 10. Provenance | `data['depth_align_refresh'] = {'history': [{ts, params, n_frames_aligned, n_frames_no_contact, abs_dz_median_m, abs_dz_max_m, n_capped, ...}]}` for idempotence. |

**Why contact geometry, not a fixed palm/tip weight.** A power grasp's
contact set naturally includes palm + proximal phalanges, so the contact
mean is palm-dominated. A precision pinch's contact set is just the
thumb / index tips, so the contact mean is tip-dominated. The weighting
falls out of the geometry without us having to detect grasp type. Using
all 778 MANO vertices (instead of a hand-picked 13-vertex palm subset)
also avoids hard-coded MANO-topology assumptions and adapts to MANO
shape variation across actors.

**What it does NOT touch:**

- STATIC LOCK frames — lock pose comes from anchor, not from the Kabsch chain.
- DEPTH_TRACKED frames — obs cloud directly drives `T_depth_seq`, no Kabsch bias.
- Rotation — `T_seq[:, :3, :3]` is owned by bake; depth_align only modifies the z-column of t.
- `pose_track_info_meta` — keep the phase_d meta intact so the canonical check (`fp_pose_bake.ts ≥ pose_track_info_meta.updated_at`) still passes after depth_align bumps mtime.

**Invariants to preserve (same as `refresh_depth_smooth`):** any Δz
applied to `T_seq[t, 2, 3]` must also translate the cached cloud
`pts[:, 2]` and OBB `corners[:, 2]` by the same Δz. Otherwise downstream
viser export (which reuses these cached fields) shows the mesh in the
new position but the OBB / point cloud in the old.

### 4.9 Future: Depth Alignment as Stage 0

Long-term, the three patches that touch depth post-hoc —
`egoinfinity/pipeline/depth_align.py` (Phase C-1 inference-time multi-scale
hand-depth align), `egoinfinity/pipeline/post_tracking/depth_smooth.py` (σ_z = 12 Gaussian +
MAD ×5 spike rejection), and the §4.8 `refresh_depth_align` — should
fold into one **Stage 0 · Depth Alignment** module between MoGe-2
(Phase A) and Part A status detection. Until validated and stable,
keeping them separate lets each be A/B-tested independently.

The three-part model (A / B / C) stays unchanged; only the depth input
seen by Part A and the t output of Part B change. If Stage 0 changes z
after `bake_fp_pose` has run, t becomes stale and R may need re-baking;
the Stage 0 module would need its own timestamp checked by bake's
canonical-check, analogous to `fp_pose_bake.history`.

---

## 5. Cross-process worker topology

```
┌──────────────────── Login or Compute Node ────────────────────┐
│                                                                │
│  conda env: egoinfinity (main)                                     │
│   └── python -m egoinfinity process  (or scripts/exo_pipeline.py)  │
│       │                                                        │
│       ├── subprocess: scripts/exo_pipeline.py                  │
│       │     (phase-1; per clip; short-lived; explicit free_gpu)│
│       │     └── MoGe-2, WiLoR, SAM2, MEMFOF, infiller, ...     │
│       │                                                        │
│       ├── subprocess: scripts/sam3_worker.py                   │
│       │     conda env: sam3 (py3.12, torch 2.10, cu128)        │
│       │     ├── SAM3.1 (persistent, bf16, ~4 GB)               │
│       │     └── unix socket /tmp/egoinfinity_sam3_${USER}.sock │
│       │                                                        │
│       └── subprocess: scripts/sam3d_worker.py                  │
│             conda env: sam3d-objects                           │
│             ├── SAM 3D Objects (persistent, ~10 GB)            │
│             └── unix socket /tmp/egoinfinity_sam3d_${USER}.sock│
└────────────────────────────────────────────────────────────────┘
```

**Why this design**: SAM3 / SAM3D have incompatible torch / CUDA toolchains
and ship as separate envs.  Both also pay a heavy cold-load cost
(SAM3 ~17 s, SAM3D ~5 min) — so we run them as **persistent workers**
behind Unix sockets, and pipeline subprocesses talk to them via
line-delimited JSON.  Cold load is paid once per worker startup; per-clip
SAM3 latency drops from 170 s to 0.3 s.

---

## 6. Data layout

| What | Where | Size | Notes |
|---|---|---|---|
| Pretrained pt/ckpt (WiLoR detector + final, SAM2-hiera-small, infiller) | `${EGOINFINITY_CKPT_DIR}` (default `<repo>/pretrained_models/`) | ~900 MB | gitignored; downloaded by `setup_weights.sh` |
| HuggingFace cache (MoGe-2, DINOv2, SAM3.1, MEMFOF) | `${HF_HOME}` | ~17 GB | offline mode after first download |
| MANO model | `third_party/wilor/mano_data/MANO_RIGHT.pkl` | 3.8 MB | gitignored; user downloads (see README "MANO model") |
| Action100M filter index (SQLite) | `${ACTION100M_DB}` (default `<repo>/data/action100m_index.db`) | ~29 GB | gitignored; built by `action100m_filter`, rebuilt locally |
| Per-clip cache | `<repo>/cache/<clip_id>/` (override via `ACTION100M_CACHE`) | 50-500 MB | gitignored |
| `pipeline_result.pkl.gz` (v2) | same, in each clip cache | 20-40 MB | depth uint16-mm PNG + recompute pts on load |
| Per-clip favorites snapshot | `<repo>/cache/favorites/<clip_id>/` | 10-50 MB | promoted favorites |

---

## 7. PKL format v2

`pipeline_result.pkl.gz` was 100-200 MB / clip in v1.  v2 cuts that to
**20-40 MB** (8× shrink) by:

| Field | v1 | v2 |
|---|---|---|
| `frame_data[i]['depth_map']` | float32 (480, 854) ≈ 1.6 MB / frame | `depth_png` bytes — uint16 mm via PNG ≈ 130 KB / frame |
| `frame_data[i]['sam3_obj_data'][oid]['pts']` | float32 (N, 3) ≈ 30k pts × 12 B | dropped — recomputed on load from mask + depth |
| `bg_template` | float32 single frame | `bg_template_png` bytes |

Loading goes through `scripts/pipeline_utils.rehydrate_pkl(data)` which:
1. decodes PNG → float32 depth map
2. recomputes `pts` via existing `mask_to_pointcloud(mask, depth, focal, cx, cy)`

Backwards compatible — v1 caches load unchanged.  The depth quantization
budget is 0.5 mm (half of 1 mm round-to-nearest), well below MoGe-2's own
~10 mm noise.

---

## 8. Hardware

Designed for **single A100 40 GB** or **H100 80 GB**.

Resident GPU memory while idle:
- SAM3 4 GB + SAM3D 10 GB ≈ **~14 GB always-on**
- pipeline subprocess active: 3-5 GB peak (sequential)
- Total budget: ~18-19 GB — A100 40 GB is comfortable, H100 has slack for
  multiple parallel pipeline workers (planned, see "Roadmap").

GPUs with **< 16 GB**: not officially supported in this revision.  SAM3 /
SAM3D would need lazy-load + idle-unload + mutex; that work is
sketched in our internal notes but not landed.

---

## 9. YouTube cookies (Action100M downloader)

The Action100M downloader uses `yt-dlp` to download Action100M segments from
YouTube; YouTube rate-limits anonymous IPs.  Authenticate via cookies:

- Path: `${YTDLP_COOKIES_FILE}` (configured in your `activate.sh`)
- Critical cookies: `SID`, `__Secure-1PSID`, `__Secure-3PSID`, `HSID`,
  `SSID`, `APISID`, `SAPISID`, `LOGIN_INFO`
- **Permission**: `chmod 400` (read-only) to prevent yt-dlp from
  silently overwriting after a failed (401/403) response
- 1PSID validity ≈ 2 years; rotate when `Sign in to confirm you're not
  a bot` errors appear

To re-export from your laptop:
1. Open youtube.com in a logged-in browser session
2. Use a "Get cookies.txt" extension
3. Save in **Netscape format**
4. `scp` to `${YTDLP_COOKIES_FILE}` and `chmod 400`

---

## 10. Roadmap

Currently sketched, not landed:

- **Batch processing mode**: parallel `PipelineWorker` instances per GPU,
  shared SAM3 / SAM3D workers, SLURM array submission for thousands of
  clips
- **Lazy worker scheduling for ≤ 16 GB GPUs**
- **30-fps hand path / 15-fps object path**: dual-timeline support so
  WiLoR (cheap) runs at 30 fps for finer hand dynamics while expensive
  models stay at 15 fps
- **Numerical optimizations**: WiLoR / MoGe-2 batching, `torch.compile`
  for batch worker, bf16 across the board

These are independent enhancements; the current single-clip path is
considered stable.

---

## 11. References

- [README.md](README.md) — install / quick start
- `FoundationPose-plus-plus/` (sibling repo) — Stage 2 of §4
- WiLoR / MoGe-2 / SAM2 / SAM3 / HaWoR / MANO upstream repos — see
  README's Acknowledgements
