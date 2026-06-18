import numpy as np

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter on j0 (Rotation / shoulder swing).
    # FK: right Rotation=0 → y=-0.123; left Rotation=0 → y=+0.143.
    # Bounds keep each EE on its own y-side of the torso.
    "lateral_joint": {"index": 0, "left": (-0.80, 0.20), "right": (-0.20, 0.80)},

    # proximal_jitter: j1=Pitch, j2=Elbow.
    # Wrist (j3, j4) handled by wrist_limits.
    "proximal_jitter": {1: (-0.40, 0.40), 2: (-0.40, 0.40)},

    # OU walk: moderate step for a compact 5-DoF arm.
    "ou_step":   np.array([0.020, 0.016, 0.012], dtype=np.float32),
    "ou_spring": 0.08,

    # Wrist slice [3:5] (j3=Wrist_Pitch, j4=Wrist_Roll); limits relative to start_config.
    "wrist_joints":   (3, 5),
    "wrist_limits":   np.array([[-0.8, 0.8], [-1.0, 1.0]], dtype=np.float32),
    "wrist_relative": True,

    # No additional workspace clamps.
    "workspace_bias": {},
}
