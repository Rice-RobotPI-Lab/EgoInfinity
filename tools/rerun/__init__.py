"""Standalone re-run helpers for individual pipeline phases.

These operate directly on a clip's pipeline_result.pkl.gz and are NOT
part of the canonical post-tracking sequence (that lives in
egoinfinity/pipeline/post_tracking/). They are used when re-running a single
heavy Phase-1 stage on already-computed data — e.g. re-meshing SAM3D on
an A100 in the multi-host flow, or back-filling optical-flow magnitudes.

  refresh_sam3d_meshes  — Phase D-sam3d only (reuses existing SAM3 masks)
  refresh_hands         — Phase B (re-run WiLoR + infiller on cached frames)
  refresh_hand_scale    — MANO metric-scale correction
  refresh_optical_flow  — back-fill MEMFOF flow magnitude into the pkl
"""
