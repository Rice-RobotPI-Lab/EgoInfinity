import numpy as np

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter on j0 (shoulder_roll) keeps each arm on its own side.
    "lateral_joint": {"index": 0, "left": (-0.55, 0.15), "right": (-0.15, 0.55)},

    # proximal_jitter: j1=shoulder_pitch (negative bias keeps arm forward),
    # j2=elbow_yaw (full range), j3=elbow_pitch (more bend variation).
    # Wrist (j4-j6) handled by wrist_limits.
    "proximal_jitter": {1: (-0.5, 0.2), 2: (-0.7, 0.7), 3: (-0.5, 0.3)},

    # Larger OU walk for broader workspace coverage.
    "ou_step":   np.array([0.040, 0.035, 0.030], dtype=np.float32),
    "ou_spring": 0.04,

    # Wrist slice [4:7]; per-side absolute limits for j4 (forearm_roll).
    # FK analysis: left palm faces outward when j4<-0.3; right when j4>0.5.
    "wrist_joints":   (4, 7),
    "wrist_limits": {
        "left":  np.array([[-0.3, 2.5], [-0.8, 0.8], [-0.6, 0.6]], dtype=np.float32),
        "right": np.array([[-2.5, 0.5], [-0.8, 0.8], [-0.6, 0.6]], dtype=np.float32),
    },
    "wrist_relative": False,

    # No additional workspace clamps.
    "workspace_bias": {},
}
