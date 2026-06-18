# `manifest.json` field reference

Every clip directory ships with a `manifest.json` that describes the
input video segment and the user-supplied curation knobs the pipeline
consumes. This document is the authoritative reference for what fields
matter, which are required, and how to use the manual-override knobs.

```
<artifacts_dir>/<clip_id>/
├── manifest.json          ← this file
├── pipeline_result.pkl.gz
├── frames/
└── ...
```

## Required fields

| Field | Type | Meaning |
|---|---|---|
| `video_uri` | str | Source video path or URL. Read by `extract_frames` (downloaded first if a URL). Required for the from-scratch path; not needed if you supply `frames/` yourself. |
| `objects` | list[str] | **The text prompts SAM 3.1 searches for.** Example: `["red mug", "blue spoon"]`. Order matters — `objects[0]` becomes object id `0` in `pose_track_info`. |

If you build the artifact dir manually (rather than via the curation
flow), at minimum write a `manifest.json` like:

```json
{
  "video_uri": "/path/to/clip.mp4",
  "objects": ["red mug", "blue spoon"]
}
```

Optional: `start` / `end` (seconds) to trim, `fps` (default 15). The
retarget export also reads optional `video_uid` / `start_sec` / `end_sec`
to name its output sequence (falls back to the clip-id if absent).

## Optional fields (recommended)

| Field | Type | Meaning |
|---|---|---|
| `start_sec` | float | Trim start in seconds (selects which sub-window of the source video this clip covers). |
| `end_sec` | float | Trim end in seconds. `end_sec - start_sec` is the clip duration. |
| `duration` | float | Clip duration in seconds. Redundant with `end_sec - start_sec`; the pipeline tolerates either being present. |
| `action_brief` | str | One-line action description. Useful for the LLM prompt extractor. |
| `action_detailed` | str | Multi-sentence action description. |
| `summary` | str | Free-form scene description. |
| `objects_source` | str | `"manual"` / `"claude"` / `"llm"`. Records who curated the objects list. |
| `objects_curated_at` | ISO ts | When the objects list was finalized. |
| `note` | str | Free-form human note. |

## Manual-override knobs

These are the "人工选长度 / human-in-the-loop" touchpoints. They let
you steer the pipeline without editing code.

### A. `start_sec` / `end_sec` — clip trim window

Pick the sub-window of the source video that contains the
hand-object interaction. Shorter clips are easier to track; longer
clips capture more context.

The official pipeline does not implement automatic trim selection.
Either:
- Use the `action100m_filter` tool to score candidate windows; or
- Edit `manifest.json` directly to set the bounds you want.

### B. `objects` — what SAM 3 looks for

The text prompts passed to SAM 3.1 for detection. Examples:
- `["red mug"]` — clean noun-phrase, works best
- `["red mug", "blue spoon", "wooden cutting board"]` — multiple objects, each tracked separately
- `["the mug being held"]` — natural-language prompt, may help with disambiguation

Avoid:
- Empty list → pipeline errors out at SAM 3 prompt phase
- Pronouns alone ("it", "the thing") → SAM 3 has nothing to ground

After Phase D-sam3 runs, you can inspect the masks in viser and edit
`objects` + re-run to fix mis-detections.

### C. `sam3_mesh_info[oid].init_frame_force` — manual SAM3D anchor

Stored inside the pkl, not the manifest. Set this if SAM 3D Objects
picks a bad anchor frame (e.g. the object is partially occluded at
the default anchor). To override:

```python
import pickle, gzip
with gzip.open("pipeline_result.pkl.gz", "rb") as f:
    d = pickle.load(f)
d["sam3_mesh_info"][0]["init_frame_force"] = 42   # frame index
with gzip.open("pipeline_result.pkl.gz", "wb") as f:
    pickle.dump(d, f)
```

Then re-run the SAM 3D mesh stage with `--force-rerun-changed`:

```bash
python -m egoinfinity run refresh_sam3d /path/to/artifacts/<CLIP>/
# or the standalone tool:
python -m tools.rerun.refresh_sam3d_meshes --only=<CLIP> --force-rerun-changed
```

The pipeline preserves `init_frame_force` across reruns (see
`tools/rerun/refresh_sam3d_meshes.py:451`).

### D. `sam3_mesh_info[oid].prompt` — per-object prompt override

If SAM 3.1 fails to find an object under its global prompt list, set
a per-object prompt override directly in the pkl:

```python
d["sam3_mesh_info"][0]["prompt"] = "blue ceramic mug with floral pattern"
```

Then re-run detection + downstream (the prompt is consumed during phase1's
SAM3 pass): `egoinfinity process /path/to/artifacts/<CLIP>/ --force phase1 --cascade`.

### E. `sam3_mesh_info[oid].notes` — free-form annotation

Human notes about the object. Pipeline never reads this field; it's
for human use only. Survives reruns.

## Schema validation

There is no formal JSON schema validator in the pipeline; field
presence is checked at the consumer site (e.g. `exo_pipeline.py`
expects `manifest['objects']` to be a non-empty list). Editing
`manifest.json` is the supported way to drive the pipeline by hand.

## Auto-generated fields (do not edit)

These are added by various pipeline / curation tools and should not
be hand-edited:

- `segment_id` — Action100M segment id (Action100M-specific)
- `level`, `exo_hoi`, `actor_ambiguous`, `visual_pass`, `visual_reject_reason`
- `hand_ratio`, `bg_flow`, `avg_hand_size`, `trunc_ratio`, `both_hands_ratio`
- `favorited_at`
- `n_cuts`, `video_duration`, `video_title`
