# Multi-host pipeline

A guide for splitting EgoInfinity across multiple GPU hosts when a single
machine doesn't fit the full memory budget. Most release users **don't need
this** — a single 16 GB+ GPU runs the entire pipeline. Read this only if:

- Your local card has < 13 GB VRAM and can't host the SAM 3D Objects worker
- You want to spread per-stage cost across heterogeneous GPUs (e.g. 4070 Ti
  for hand/object detection on the laptop + A100 for SAM3D in a SLURM cluster)
- You're running large-batch dataset publishing flows and want to overlap
  stages across hosts

For single-host operation, just run
`python -m tools.batch_pipeline --only=CLIP` and skip this doc.

---

## 1 · Why multi-host

The pipeline's three heaviest persistent workers (resident VRAM listed,
per [README.md](../README.md) §Hardware):

| Worker | Resident |
|---|---|
| SAM3.1 (text-prompted detector) | ~4 GB |
| SAM 3D Objects (single-image 3D mesh) | ~10 GB |

plus a 3-5 GB peak from the pipeline subprocess itself. Total ~14-15 GB
(SAM3 4 + SAM3D 10 + pipeline 3-5 peak). A 40 GB A100 or 80 GB H100
absorbs this comfortably; a 16 GB card needs selective skipping; a 12 GB
card needs splitting across hosts.

The pipeline has been factored into [phases](PIPELINE.md) that don't
need to share GPU state — they communicate via the `pipeline_result.pkl.gz`
file. That makes "ship the pkl to another host, finish there, ship back"
the natural primitive.

## 2 · Standard two-host layout

This is the layout used in development (4070 Ti laptop + A100 cluster
node):

```
┌─────────────────────────────────────────────┐
│ Host A — 4070 Ti (16 GB) or smaller         │
│                                              │
│   1. Phase A..C  (MoGe-2, WiLoR, MEMFOF,    │
│                   infiller, biomech, etc.)  │
│   2. Phase D-sam3 (SAM3.1 + SAM2 streaming) │
│   3. (skip Phase D-sam3d — not enough VRAM) │
│   4. Phase D-track Pass 1                   │
└─────────────────────────────────────────────┘
                     │
                     ↓ ship via rsync/scp/etc.
                     │
┌─────────────────────────────────────────────┐
│ Host B — A100 / H100 (≥ 16 GB)              │
│                                              │
│   5. (receive favorites/<clip>/)            │
│   6. tools.rerun.refresh_sam3d_meshes              │
│      (just Phase D-sam3d, reuses Phase D-sam3 │
│       masks already in the pkl)              │
│   7. ship back via rsync/scp/etc.           │
└─────────────────────────────────────────────┘
                     │
                     ↓ ship via rsync/scp/etc.
                     │
┌─────────────────────────────────────────────┐
│ Host A — back home                          │
│                                              │
│   8. egoinfinity process <clip>             │
│      (resumes into canonical post-tracking: │
│       pose_track Pass 1 → grasp_veto →      │
│       pose_track Pass 2 → bake →            │
│       scale_sanity → spurious)              │
└─────────────────────────────────────────────┘
```

## 3 · Step-by-step (release-user version)

The release tag strips the HF-sync tools (they live under `private/tools/`
post-cleanup) since release users don't share the same private transit
dataset. The release-user equivalent is **bring-your-own transit** — any
file copy mechanism works (`rsync`, `scp`, `s3`, NFS, etc.).

Required files to ship between hosts:

```
favorites/<clip>/
├── pipeline_result.pkl.gz    # required — contains masks, depth, joints
├── manifest.json              # required — has objects[] for re-prompting
├── sam3_meshes/*.ply          # written by Host B step 6, shipped back to A
└── (optional) sam3_quality.json, pipeline_state.json
```

`pipeline_result.pkl.gz` in v2 format is 20-40 MB per 4-second clip. The
PLY meshes from SAM3D add another 5-15 MB per object. A 7-second clip with
3 objects ships in roughly 100 MB total.

### On Host A (smaller card):

```bash
# Phase 1: detect + track, but skip the SAM3D mesh step
EGOINFINITY_RUN_SAM3D=0 \
python -m tools.batch_pipeline --only=CLIP_ID --no-sam3d-worker
```

This produces `pipeline_result.pkl.gz` and lays out
`favorites/CLIP_ID/frames/`, but `sam3_meshes/` stays empty.

```bash
# Ship to Host B (example: rsync)
rsync -a favorites/CLIP_ID/ HOST_B:/path/to/EgoInfinity/cache/favorites/CLIP_ID/
```

### On Host B (A100 / H100):

```bash
cd /path/to/EgoInfinity

# Phase 2: SAM 3D Objects mesh reconstruction (refresh_sam3d stage)
python -m egoinfinity run refresh_sam3d /path/to/cache/favorites/CLIP_ID/
# (equivalently: python -m tools.rerun.refresh_sam3d_meshes --only=CLIP_ID)
```

The `refresh_sam3d` stage re-uses the SAM3 masks already in the pkl from
Host A's Phase D-sam3, so the SAM3 worker is **not** needed here — just the
SAM3D worker. See [README §Optional components](../README.md#optional-components)
for SAM3D installation. To mesh many clips at once, point `run` at the
cache root: `egoinfinity run refresh_sam3d /path/to/cache/favorites/ --only A,B`.

```bash
# Ship back to Host A
rsync -a favorites/CLIP_ID/ HOST_A:/path/to/EgoInfinity/cache/favorites/CLIP_ID/
```

### Back on Host A:

```bash
# Resume into the canonical post-tracking sequence
python -m egoinfinity process CLIP_ID
```

Done. The clip's `pose_track_info[oid].T_seq` is now the canonical 6DoF
trajectory.

## 4 · Edge cases

### Phase D-track running on either host

Phase D-track (the 6DoF pose tracker that consumes SAM3D meshes) is
relatively light — MEMFOF + Open3D + LBFGS, 1-2 GB peak. It can run on
either host. The plan above runs it on Host A (after pulling the meshes
back), which is fine; you can equally run `egoinfinity process <clip>`
on Host B before shipping back if you prefer.

### Skipping FoundationPose++ entirely

The `bake` stage in the canonical pipeline needs FP++ `pose.npy` output
from the optional external FoundationPose++ dependency (see
[docs/THIRD_PARTY.md](THIRD_PARTY.md)). If FP++ isn't installed, bake
silently skips and `pose_track_info[oid].T_seq` keeps the phase_d-only R
(translation is unaffected). The orientation will be less stable than a
baked clip but still usable.

To skip the bake stage explicitly, run the post-tracking stages
individually and omit `bake`:

```bash
egoinfinity run pose_track_p1 CLIP_ID
egoinfinity run grasp_veto    CLIP_ID
egoinfinity run pose_track_p2 CLIP_ID
egoinfinity run scale_sanity  CLIP_ID
egoinfinity run spurious      CLIP_ID
```

### Re-prompting SAM3 mid-flow

If you change `manifest.objects` after the initial Phase D-sam3 run, you
need to wipe the stale meshes and re-cascade. The full safe re-prompt
flow:

```bash
# Wipe stale per-clip artifacts
rm -rf favorites/CLIP_ID/sam3_meshes/
rm -f favorites/CLIP_ID/sam3_quality.json

# Edit manifest.json — change "objects": [...]

# Re-run Phase 1 (regenerates SAM3 masks)
python -m tools.batch_pipeline --only=CLIP_ID --force --no-sam3d-worker

# Ship to Host B, re-mesh (tools.rerun.refresh_sam3d_meshes), ship back

# Then cascade everything downstream of sam3d
python -m egoinfinity process CLIP_ID --force pose_track_p1 --cascade
```

The `--cascade` flag pulls in pose_track_p1, grasp_veto, pose_track_p2,
depth_smooth, depth_align, bake, scale_sanity, spurious automatically.
(SAM3D itself re-runs on Host B via `tools.rerun.refresh_sam3d_meshes`.)

## 5 · Provenance & idempotency

Each component writes a provenance field into the pkl (see PIPELINE.md
§4.5):

| Component | Provenance field |
|---|---|
| pose_track_p1, pose_track_p2 | `pose_track_info_meta.updated_at` |
| grasp_veto | `grasp_veto_refresh.history[-1].ts` |
| depth_smooth | `depth_smooth_refresh.history[-1].ts` |
| scale_sanity | `scale_sanity_refresh.history[-1].ts` |
| spurious | `spurious_filter_refresh.history[-1].ts` |
| bake | `fp_pose_bake.history[-1].ts` |

A clip is **canonical** iff
`fp_pose_bake.history[-1].ts ≥ pose_track_info_meta.updated_at`. If a
later refresh writes pose_track_info, the clip drops out of canonical
state until re-baked.

(Provenance-aware skip — i.e. "only run components that are stale" — is
partially handled by the runner's resume/state mechanism: `egoinfinity
process <clip>` picks up where a prior run left off rather than redoing
completed stages.)
