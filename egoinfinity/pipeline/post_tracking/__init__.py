"""Post-tracking algorithm library — the canonical 8-stage sequence that
runs after Phase D-track produces the initial pose_track_info.

Each module exposes a ``main(argv=None)``-style entry plus the algorithm
internals as importable functions. They are invoked from the pipeline
via subprocess (see ``egoinfinity/stages/post_track.py``); they can also be
run standalone:

    python -m egoinfinity.pipeline.post_tracking.pose_tracking --mode=phase_d --only=CLIP
    python -m egoinfinity.pipeline.post_tracking.bake_fp_pose --only=CLIP --force

Files in this package (each operates on the per-clip
``pipeline_result.pkl.gz`` in place):

  pose_tracking.py     Phase D-track refinement (state machine, T_seq seed)
  grasp_veto.py        2D-static veto of false-positive grasp segments
  scale_sanity.py      Sanity check + override SAM3D's canonical_scale
  depth_align.py       Hand-mesh Z alignment in WRIST mode frames
  depth_smooth.py      sigma_z Gaussian smoothing on hand + obj Z
  bake_fp_pose.py      Compose FoundationPose++ rotation into T_seq.R
  fp_compose.py        State machine library used by bake_fp_pose
  spurious_filter.py   Flag duplicate / spurious oid detections

History: these were previously in ``tools/refresh_*.py``. The "refresh"
naming was a relic of the per-tool CLI workflow that the unified
``egoinfinity process`` runner has now replaced; they live here because
they ARE post-tracking algorithms, not utility scripts.
"""
