# Architecture

The new `egoinfinity/` package is a **thin orchestration layer** on top of
the existing algorithm code in `egoinfinity/pipeline/` (incl. `post_tracking/`),
`action100m_filter/`, `retarget/`, and `tools/`. No algorithm code is
modified by the orchestration layer — every Stage subprocess-invokes (or
directly calls) the existing implementation.

## Layers

```
   cli/                   argparse + dispatch
     └─ process / status / run / filter / import
   core/                  framework
     ├─ config            PipelineConfig + StageConfig (dataclass + YAML)
     ├─ artifacts         per-clip per-stage file management
     ├─ state             state.json: per-stage provenance, hashes, durations
     ├─ stage             Stage base class + StageContext
     ├─ runner            topo-sort + resume + cascade + --force
     └─ registry          @register_stage decorator + lookup
   stages/                concrete Stage subclasses
     └─ extract_frames, filter, phase1, post_track.*,
        export_retarget_samples, retarget
   viz/                   filter viz server

egoinfinity/pipeline/         algorithm layer (scripts/ + stages call into here)
   interfaces/            Protocol contracts for swappable Phase-1 roles
   backends/              registry: get_backend(role) -> class (default == the
                          monolith's original class; override via env)
```

## Stage contract

```python
class Stage:
    name: str
    upstream: tuple[str, ...]              # stage-name deps
    outputs: tuple[str, ...]               # files written under <root>/<stage>/

    def is_done(self, ctx) -> bool:
        """Default: state.is_done(name) AND all outputs exist on disk."""

    def run(self, ctx: StageContext) -> None:
        """Produce the outputs."""
```

The runner asks every Stage `is_done(ctx)` and skips it when true,
giving "resume from any point" semantics. `--force <stage>` makes the
runner call `state.forget(stage)` before checking, forcing a re-run.
`--cascade` additionally forgets every stage that lists the forced one
(transitively) as upstream.

## Artifact layout

```
<artifacts_dir>/<clip_id>/
├── manifest.json              # input description
├── state.json                 # per-stage runs (timestamps + hashes)
├── input/, frames/            # extract_frames bridge
├── extract_frames/frames/     # output of ExtractFramesStage
├── filter/result.json         # FilterStage output
├── phase1/pipeline_result.pkl.gz   # legacy consolidated artifact (symlink)
├── pipeline_result.pkl.gz     # the same pkl at root (legacy callers)
├── pose_track_p1/, grasp_veto/, pose_track_p2/, ...
│   └── (these stages mutate the pkl in place; no file outputs)
└── retarget/<robot>/trajectory.json   # RetargetStage output (consumes the root pkl directly)
```

Choice C3 (hybrid): the per-stage dirs are the resume markers; the
flat `pipeline_result.pkl.gz` at the root stays for back-compat with
existing dev tools (FP++ Docker, viser viewer, etc.).

## Resume mechanics

`egoinfinity process` walks `pipeline` in declared order:

1. For each enabled Stage, ask `is_done(ctx)`.
2. If yes → log SKIP, continue.
3. If no → mark_running in state.json, call `run(ctx)`, then
   mark_done with hashes of the produced outputs.

`--force pose_track_p1` calls `state.forget("pose_track_p1")` before
walking, so it re-runs. With `--cascade`, also forgets all stages
that list `pose_track_p1` in their `upstream`.

This replaces the old `tools/refresh --pipeline canonical` flow:
the canonical sequence is just the declared order in `configs/defaults.yaml`,
and a re-run is whichever subset of stages haven't been done yet.

## Clip × stage selectors (one engine)

Two orthogonal selectors run over one engine (`core/runner.py`):

- **Clips** — `core/clips.py:resolve_clips` expands CLI targets into a clip
  list: an existing clip dir, a batch root (its clip subdirs, `--only`-filtered),
  or (single) a video. `runner.run_many` loops them with per-clip failure
  isolation + a summary (`--fail-fast` to stop early).
- **Stages** — `egoinfinity process` runs the full enabled pipeline with
  resume; `egoinfinity run <stages> <clips>` force-runs an explicit subset
  (`runner.run_stages`) in declared order. A named stage runs regardless of
  its `enabled` flag, and its `upstream` must already be done or also selected,
  else it errors unless `--with-deps` auto-runs the missing upstream.

The phase1-internal re-run tools (`tools/rerun/refresh_*`) are registered as
`enabled: false` stages (`stages/refresh.py`) so they are selectable via `run`
without joining the default `process` flow — the fine-grained counterpart to
the monolithic phase1 (e.g. cross-host `refresh_sam3d`).

## Backend swap (implemented) + why phase1 stays monolithic

Phase-1 components (metric depth, gravity, hand detection, hand
reconstruction) are swappable via a small registry, NOT by splitting phase1
into per-substage subprocess Stages. **The default for every role is the
exact class the pipeline has always used**, so an unset env is byte-identical
to pre-modularization. `scripts/exo_pipeline.py` constructs each role via
`get_backend(role)`.

| Role | Env var | Default | Class | Contract |
|---|---|---|---|---|
| depth | `EGOINFINITY_DEPTH_BACKEND` | `moge2` | `MoGe2Estimator` | `interfaces/depth.py:IDepthEstimator` |
| gravity | `EGOINFINITY_GRAVITY_BACKEND` | `geocalib` | `GravityEstimator` | `interfaces/gravity.py:IGravityEstimator` |
| hand_detect | `EGOINFINITY_HAND_DETECT_BACKEND` | `yolo` | `HandDetector` | `interfaces/hand.py:IHandDetector` |
| hand_recon | `EGOINFINITY_HAND_RECON_BACKEND` | `wilor` | `HandReconstructor` | `interfaces/hand.py:IHandReconstructor` |

Interfaces live in `egoinfinity/pipeline/interfaces/`; the registry is
`egoinfinity/pipeline/backends/__init__.py`. (Object detection / mesh —
SAM3.1 / SAM 3D Objects — run as separate conda-env socket workers, swapped
by pointing `SAM3_REPO` / `SAM3D_REPO` at a different checkout, not via this
registry — see [THIRD_PARTY.md](THIRD_PARTY.md).)

**Use an alternate backend** (sets the env for the phase1 subprocess):
```bash
python -m egoinfinity process my_video.mp4 --objects "mug" --depth-backend mydepth
# or directly:  EGOINFINITY_DEPTH_BACKEND=mydepth python -m tools.batch_pipeline --only=CLIP
```

**Add a new backend** — implement a class satisfying the role's `Protocol`
(structural; no inheritance), register a lazy loader, then select it:
```python
from egoinfinity.pipeline.backends import register
def _mydepth():
    from my_pkg.depth import MyDepth
    return MyDepth          # return the CLASS, not an instance
register("depth", "mydepth", _mydepth)   # then: --depth-backend mydepth
```
The call site instantiates it as `get_backend('depth')(device='cuda')`, so the
`__init__` must accept those args. The default resolution is provably the
original class: `get_backend('depth') is MoGe2Estimator` holds with env unset.
Swapping a backend means **re-running phase1** (its output is persisted, so
downstream phases pick it up).

**Why phase1 is NOT split into 9 subprocess stages.** Two independent
adversarial verifiers (+ direct source trace) confirmed a per-phase
subprocess split CANNOT reproduce the monolith: Phase A's raw float32
`depth_maps` is never persisted (only a lossy uint16-mm PNG), and
`depth_maps_stable` / `frames_rgb` / `hand_results_per_frame` /
`sam3_obj_states_multi` are in-memory-only handoffs. A faithful split
would need a parallel lossless-checkpoint system AND still wouldn't be
bit-identical (cuDNN/TF32 nondeterminism). So phase1 remains a single
stage (one process, full-precision in-memory handoffs = unchanged
result). The unit of Phase-1 re-run is the whole pass; component swaps
re-run phase1 with a different backend env. The post-tracking half is
already per-stage runnable (it legitimately resumes from the pkl).

Partial Phase-1 runs for multi-host hand-off use the existing tail-skip
gates: `--no-sam3d` (skip D-sam3d) / `--no-track` (skip D-track), surfaced
on `egoinfinity process`.

## Status (2026-06-15)

| Phase | Implemented |
|---|---|
| 0 | Public repo includes all pre-June work |
| 1 | `egoinfinity/` foundation: config, artifacts, state, stage, runner |
| 2 | `filter` Stage + `egoinfinity filter` standalone CLI |
| 3 | `phase1` Stage wraps `batch_pipeline.py` |
| 4 | Post-tracking Stages: `pose_track_p1` / `grasp_veto` / `pose_track_p2` / `scale_sanity` / `depth_smooth` / `depth_align` / `bake_fp` / `spurious` |
| 4 | `egoinfinity import` CLI for legacy pkl ingestion |
| 5 | `export_retarget_samples` + `retarget` Stages (g1 / franka / robonaut2 / xlerobot backends; subprocess to `retarget/scripts/test.py`) |
| 7 | Smoke-verified: egoinfinity reproduces dev's canonical post-tracking field-by-field |

The pipeline runs end-to-end through retarget. `python -m egoinfinity
process <clip> --robot g1` produces the full chain from RGB frames to a
retargeted robot trajectory.
