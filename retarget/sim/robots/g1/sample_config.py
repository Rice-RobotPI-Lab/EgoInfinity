import numpy as np

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter range (taskspace) and absolute clip (random_joint)
    # for shoulder_roll (j1), keeping each arm on its own side of the midline.
    "lateral_joint": {"index": 1, "left": (-0.20, 0.30), "right": (-0.30, 0.20)},

    # proximal_jitter: per-joint additive noise applied to the reference config.
    # j0=shoulder_pitch, j2=shoulder_yaw, j3=elbow. shoulder_roll (j1) handled by lateral_joint.
    "proximal_jitter": {0: (-0.3, 0.4), 2: (-0.3, 0.3), 3: (0.0, 0.6)},

    # Ornstein-Uhlenbeck parameters for taskspace random walk (m / step).
    "ou_step":   np.array([0.026, 0.024, 0.020], dtype=np.float32),
    "ou_spring": 0.05,

    # Wrist joint slice [ws:we] and absolute per-clip limits (rad).
    "wrist_joints":   (4, 7),
    "wrist_limits":   np.array([[-1.5, 1.5], [-0.8, 0.8], [-0.7, 0.7]], dtype=np.float32),
    "wrist_relative": False,

    # Absolute joint clamps for random_joint_traj (verified by FK to keep wrists
    # in the forward manipulation workspace).
    # j0=shoulder_pitch (forward reach), j2=shoulder_yaw, j3=elbow (always bent),
    # j4-j6=wrist (natural deviation range).
    "workspace_bias": {
        0: (-1.5, -0.2),
        2: (-0.8,  0.8),
        3: ( 0.5,  1.5),
        4: (-0.8,  0.8),
        5: (-0.6,  0.6),
        6: (-0.5,  0.5),
    },
}
