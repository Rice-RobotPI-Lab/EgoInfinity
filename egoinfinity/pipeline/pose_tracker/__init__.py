"""6DoF pose tracking for SAM3D meshes.

Stage-by-stage pipeline (see POSE_TRACKING_PLAN.md):
    Stage 0  Anchor frame selection
    Stage 1  Anchor 6DoF (FGR + ICP) + Umeyama scale check
    Stage 2  Optical flow + RANSAC PnP propagation
    Stage 3  Triple-signal trust filter (IoU + Chamfer + inlier ratio)
    Stage 4  Untrusted segment fill (interpolation + physics opt)
    Stage 5  Final SavGol smoothing

Currently implemented: Stage 0 + Stage 1.
"""
from .anchor import (
    select_anchor_frame,
    estimate_anchor_pose,
    AnchorResult,
)
from .flow_pnp import (
    track_6dof,
    propagate,
    make_flow_engine,
    TrackResult,
    PropagationResult,
)
from .utils import (
    build_K,
    project_points,
    backproject_mask,
    render_hand_mask,
    load_mesh_points_from_ply,
    umeyama_with_scale,
)
from .trust_filter import (
    evaluate_trust,
    render_mesh_mask,
    mask_iou,
    chamfer_partial,
    TrustResult,
)
from .contact import (
    detect_contact_per_frame,
    detect_contact_2d_aware,
    MANO_PALM_VERTICES,
)
from .mask_completion import complete_static_masks
from .grasp import (
    detect_grasp_with_motion,
    detect_grasp_proximity_motion,
    detect_grasp_fingertip_persistent,
)
from .object_motion import (
    compute_object_motion_per_frame,
    detect_static_moving_segments,
    per_frame_stability_flags,
)
from .depth_pose import (
    compute_depth_pose_per_frame,
    lock_pose_for_segment,
)
from .debug_viz import (
    is_debug_enabled,
    dump_completed_masks,
    dump_flow_field,
    flow_to_color,
)
from .fill_optimize import (
    optimize_pose_seq,
    OptimizeResult,
    DEFAULT_LAMBDAS,
)
from .smooth import smooth_se3_savgol
from .memfof_flow import (
    MEMFOFFlowEngine,
    get_global_engine as get_memfof_engine,
)
from .hand_driven import (
    hand_driven_pose_propagation,
    wrist_binding_propagation,
    mask_wrist_anchor_propagation,
    rigid_wrist_binding_propagation,
    compute_mask_wrist_anchor,
    kabsch_rigid,
    palm_keypoints,
    find_continuous_segments,
    find_last_trusted_before,
    PALM_FRAME_JOINT_INDICES,
)
from .hysteresis import hysteresis_filter
from .so3_utils import (
    so3_riemannian_median,
    slerp_so3,
    logmap_so3,
    expmap_so3,
    so3_distance,
)
from .orientation_search import (
    search_anchor_orientation,
    cube_symmetry_rotations,
    axis_aligned_quarter_rotations,
)
from .obs_obb import (
    compute_obb as compute_obs_obb,
    compute_obs_obb_per_frame,
    is_trustworthy as obs_obb_trustworthy,
    sign_correct as obs_obb_sign_correct,
    obb_corners as obs_obb_corners,
    obb_axes as obs_obb_axes,
)

__all__ = [
    'select_anchor_frame',
    'estimate_anchor_pose',
    'AnchorResult',
    'track_6dof',
    'propagate',
    'make_flow_engine',
    'TrackResult',
    'PropagationResult',
    'build_K',
    'project_points',
    'backproject_mask',
    'render_hand_mask',
    'load_mesh_points_from_ply',
    'umeyama_with_scale',
    'evaluate_trust',
    'render_mesh_mask',
    'mask_iou',
    'chamfer_partial',
    'TrustResult',
    'detect_contact_per_frame',
    'detect_contact_2d_aware',
    'MANO_PALM_VERTICES',
    'complete_static_masks',
    'detect_grasp_with_motion',
    'detect_grasp_proximity_motion',
    'detect_grasp_fingertip_persistent',
    'optimize_pose_seq',
    'OptimizeResult',
    'DEFAULT_LAMBDAS',
    'smooth_se3_savgol',
    'hand_driven_pose_propagation',
    'wrist_binding_propagation',
    'mask_wrist_anchor_propagation',
    'compute_mask_wrist_anchor',
    'kabsch_rigid',
    'palm_keypoints',
    'find_continuous_segments',
    'find_last_trusted_before',
    'PALM_FRAME_JOINT_INDICES',
    'rigid_wrist_binding_propagation',
    'hysteresis_filter',
    'so3_riemannian_median',
    'slerp_so3',
    'logmap_so3',
    'expmap_so3',
    'so3_distance',
    'search_anchor_orientation',
    'cube_symmetry_rotations',
    'axis_aligned_quarter_rotations',
    'compute_obs_obb',
    'compute_obs_obb_per_frame',
    'obs_obb_trustworthy',
    'obs_obb_sign_correct',
    'obs_obb_corners',
    'obs_obb_axes',
]
