import numpy as np

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter on j0 (shoulder_roll) keeps each arm on its own side.
    # start_config has j0_left=+0.84, j0_right=-0.84 (y≈±0.21 m separation).
    "lateral_joint": {"index": 0, "left": (-0.40, 0.50), "right": (-0.50, 0.40)},

    # proximal_jitter: j1=shoulder_pitch, j2=shoulder_yaw, j3=elbow.
    # shoulder_roll (j0) handled by lateral_joint; wrist (j4-j6) handled by wrist_limits.
    "proximal_jitter": {1: (-0.30, 0.30), 2: (-0.30, 0.30), 3: (-0.25, 0.25)},

    # OU walk: slightly slower than G1 to avoid occasional fast clips.
    "ou_step":   np.array([0.016, 0.014, 0.012], dtype=np.float32),
    "ou_spring": 0.08,

    # Wrist slice [4:7]; limits relative to mean start_config wrist values.
    "wrist_joints":   (4, 7),
    "wrist_limits":   np.array([[-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8]], dtype=np.float32),
    "wrist_relative": True,

    # No additional workspace clamps — Franka joint limits are already well-scoped.
    "workspace_bias": {},
}
