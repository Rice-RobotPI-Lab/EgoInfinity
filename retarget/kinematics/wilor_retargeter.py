"""
WiLOR keypoint-based hand retargeter.

Input : (21, 3) MANO keypoints in any consistent 3-D frame (world or camera)
Output: (n_dof,) robot finger joint angles, or None if the robot has no hands.

MANO-21 keypoint layout
-----------------------
 0  wrist
 1  thumb  CMC
 2  thumb  MCP
 3  thumb  IP
 4  thumb  tip
 5  index  MCP
 6  index  PIP
 7  index  DIP
 8  index  tip
 9  middle MCP
10  middle PIP
11  middle DIP
12  middle tip
13  ring   MCP
14  ring   PIP
15  ring   DIP
16  ring   tip
17  pinky  MCP
18  pinky  PIP
19  pinky  DIP
20  pinky  tip

Feature extraction strategy
----------------------------
For each of the 15 MANO skeleton joints we compute (flex, abd):

  flex : bend angle at the joint — 0 when straight, positive when curling.
         Computed as acos(dot(incoming_bone_dir, outgoing_bone_dir)).

  abd  : lateral abduction angle.
         Only implemented for thumb CMC (the spread/retract DoF) and the
         four finger MCPs; all others are zero.

The (15, 2) feature array shares the same index layout as
the same RobotHandConfig / JointMapping objects drive the mapping to robot DoFs.

Robots without articulated fingers
-----------------------------------
Call get_wilor_hand_retargeter(robot, side) — returns None for robots that
have no movable finger joints (e.g. AthenaZero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── robot hand configuration ──────────────────────────────────────────────────

@dataclass
class JointMapping:
    """One robot finger joint mapped from one MANO joint."""
    robot_name: str
    mano_joint: int
    dof:        str           # "flex" | "abd"
    scale:      float = 1.0
    offset:     float = 0.0
    limit:      tuple[float, float] = (-3.14159, 3.14159)
    left_sign:  float = 1.0   # set to -1 for joints that mirror on the left hand


@dataclass
class RobotHandConfig:
    """Robot finger DOF layout and their mapping from MANO joint features."""
    joints: list[JointMapping] = field(default_factory=list)

    @property
    def n_dof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> list[str]:
        return [j.robot_name for j in self.joints]


# ── MANO-21 keypoint indices ──────────────────────────────────────────────────

_W  = 0                                   # wrist
_TH = (1, 2, 3, 4)                        # thumb  CMC MCP IP tip
_ID = (5, 6, 7, 8)                        # index  MCP PIP DIP tip
_MD = (9, 10, 11, 12)                     # middle MCP PIP DIP tip
_RG = (13, 14, 15, 16)                    # ring   MCP PIP DIP tip
_PK = (17, 18, 19, 20)                    # pinky  MCP PIP DIP tip

# Triplets (parent, joint, child) for bend-angle flexion,
# one per MANO joint in MANO_JOINT_NAMES order:
#   index[mcp,pip,dip], middle[mcp,pip,dip], pinky[mcp,pip,dip],
#   ring[mcp,pip,dip], thumb[cmc,mcp,ip]
_TRIPLETS = [
    (_W,      _ID[0], _ID[1]),  # 0  index_mcp
    (_ID[0],  _ID[1], _ID[2]),  # 1  index_pip
    (_ID[1],  _ID[2], _ID[3]),  # 2  index_dip
    (_W,      _MD[0], _MD[1]),  # 3  middle_mcp
    (_MD[0],  _MD[1], _MD[2]),  # 4  middle_pip
    (_MD[1],  _MD[2], _MD[3]),  # 5  middle_dip
    (_W,      _PK[0], _PK[1]),  # 6  pinky_mcp
    (_PK[0],  _PK[1], _PK[2]),  # 7  pinky_pip
    (_PK[1],  _PK[2], _PK[3]),  # 8  pinky_dip
    (_W,      _RG[0], _RG[1]),  # 9  ring_mcp
    (_RG[0],  _RG[1], _RG[2]),  # 10 ring_pip
    (_RG[1],  _RG[2], _RG[3]),  # 11 ring_dip
    (_W,      _TH[0], _TH[1]),  # 12 thumb_cmc
    (_TH[0],  _TH[1], _TH[2]),  # 13 thumb_mcp
    (_TH[1],  _TH[2], _TH[3]),  # 14 thumb_ip
]


# ── geometry helpers ──────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _bend_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Angle at B in the A-B-C chain: 0 = straight, +π = fully reversed."""
    d1 = _norm(B - A)
    d2 = _norm(C - B)
    return float(np.arccos(np.clip(np.dot(d1, d2), -1.0, 1.0)))


def _keypoint_features(kp: np.ndarray, side: str) -> np.ndarray:
    """
    kp   : (21, 3) MANO keypoints
    side : "right" | "left"

    Returns (15, 2) feature array:
        column 0 = flexion  [rad]  positive = finger curling
        column 1 = abduction [rad] positive = finger spreading (right-hand convention)
    """
    feats = np.zeros((15, 2), dtype=np.float64)

    # ── flexion: bend angle at each joint ─────────────────────────────────
    for i, (a, b, c) in enumerate(_TRIPLETS):
        feats[i, 0] = _bend_angle(kp[a], kp[b], kp[c])

    # ── abduction ─────────────────────────────────────────────────────────
    # Palm frame: origin = wrist, normal = cross(index_mcp - wrist, pinky_mcp - wrist)
    palm_in  = _norm(kp[_ID[0]] - kp[_W])     # wrist → index MCP
    palm_out = _norm(kp[_PK[0]] - kp[_W])     # wrist → pinky MCP
    palm_n   = _norm(np.cross(palm_in, palm_out))   # palm normal (roughly out of palm)

    # Middle-finger reference direction in the palm plane (lateral = 0)
    mid_dir  = _norm(kp[_MD[0]] - kp[_W])
    lat_axis = _norm(np.cross(mid_dir, palm_n))     # points toward index side (right hand)
    if side == "left":
        lat_axis = -lat_axis

    # MCP abduction for index/middle/ring/pinky: project MCP→PIP onto lat_axis
    for mano_idx, finger in [(0, _ID), (3, _MD), (9, _RG), (6, _PK)]:
        mcp_dir = _norm(kp[finger[1]] - kp[finger[0]])
        mcp_dir_palm = _norm(mcp_dir - np.dot(mcp_dir, palm_n) * palm_n)
        # signed angle from mid_dir projected onto palm
        ref_palm = _norm(mid_dir - np.dot(mid_dir, palm_n) * palm_n)
        cos_a = np.clip(np.dot(mcp_dir_palm, ref_palm), -1.0, 1.0)
        sin_a = np.dot(np.cross(ref_palm, mcp_dir_palm), palm_n)
        feats[mano_idx, 1] = float(np.arctan2(sin_a, cos_a))

    # Thumb CMC abduction: how far thumb sticks out of the palm plane
    thumb_prox = _norm(kp[_TH[1]] - kp[_TH[0]])   # CMC → MCP direction
    feats[12, 1] = float(np.arcsin(np.clip(np.dot(thumb_prox, palm_n), -1.0, 1.0)))

    return feats


# ── main class ────────────────────────────────────────────────────────────────

class WilorHandRetargeter:
    """
    Maps WiLOR 21-keypoint hand poses to robot finger joint angles.

    Parameters
    ----------
    config : RobotHandConfig — robot finger DOF layout and MANO mapping
    side   : "right" | "left"
    """

    def __init__(self, config: RobotHandConfig, side: str):
        assert side in ("left", "right")
        self.config = config
        self.side   = side
        self._sign  = -1.0 if side == "left" else 1.0

    @property
    def n_dof(self) -> int:
        return self.config.n_dof

    @property
    def joint_names(self) -> list[str]:
        return self.config.joint_names

    def retarget(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Map a single frame of 21 keypoints to robot joint angles.

        Parameters
        ----------
        keypoints : (21, 3) MANO keypoints in any 3-D frame

        Returns
        -------
        q : (n_dof,) float32 robot joint angles [rad], clamped to joint limits
        """
        kp    = np.asarray(keypoints, dtype=np.float64)
        feats = _keypoint_features(kp, self.side)  # (15, 2)
        q     = np.zeros(self.config.n_dof, dtype=np.float64)

        for k, jm in enumerate(self.config.joints):
            dof_idx = 0 if jm.dof == "flex" else 1
            val     = feats[jm.mano_joint, dof_idx]
            if jm.dof == "abd":
                val *= self._sign * jm.left_sign
            q[k] = np.clip(val * jm.scale + jm.offset, jm.limit[0], jm.limit[1])

        return q.astype(np.float32)

    def retarget_sequence(self, keypoints_seq: np.ndarray) -> np.ndarray:
        """
        Retarget a sequence of keypoints.

        Parameters
        ----------
        keypoints_seq : (T, 21, 3)

        Returns
        -------
        Q : (T, n_dof) float32
        """
        seq = np.asarray(keypoints_seq, dtype=np.float64)
        return np.stack([self.retarget(kp) for kp in seq]).astype(np.float32)


# ── G1 config for keypoint-based retargeting ─────────────────────────────────
#
# The MANO axis-angle retargeter used signed flexion values (positive or negative).
# Keypoint bend angles are always ≥ 0 (straight = 0, fully curled ≈ π), so
# which are signed (positive or negative flexion).  Keypoint bend angles are
# always ≥ 0 (straight = 0, fully curled ≈ π).  The G1 hand is also mirrored:
#
#   Right hand joints three/four/five/six have range [0, +1.84]  → +scale
#   Left  hand joints three/four/five/six have range [-1.84, 0]  → -scale
#
#   Thumb joints mirror too: right IP range [-1.84, 0] → -scale
#                             left  IP range [0, +1.84] → +scale
#
# Using the wrong sign clamps outputs to zero and the hand never closes.

def _g1_7dof_from_keypoints(side: str) -> RobotHandConfig:
    if side == "right":
        return RobotHandConfig(joints=[
            # thumb CMC spread: abd feature, out-of-palm = positive → into palm = negative
            JointMapping("zero",  12, "abd",  scale= 1.0, limit=(-0.52, 0.52), left_sign=-1.0),
            # thumb MCP flex: right range [-1.20, 1.00], closing = negative
            JointMapping("one",   13, "flex", scale=-1.0, limit=(-1.2,  1.0)),
            # thumb IP flex:  right range [-1.84, 0.00], closing = negative
            JointMapping("two",   14, "flex", scale=-1.5, limit=(-1.84, 0.0)),
            # index+middle MCP: right range [-0.30, +1.84], closing = positive
            JointMapping("three",  0, "flex", scale= 1.0, limit=(-0.3,  1.84)),
            # index+middle PIP: right range [0.00, +1.84], closing = positive
            JointMapping("four",   1, "flex", scale= 1.2, limit=(0.0,   1.84)),
            # ring+pinky MCP:  right range [-0.30, +1.84], closing = positive
            JointMapping("five",   9, "flex", scale= 1.0, limit=(-0.3,  1.84)),
            # ring+pinky PIP:  right range [0.00, +1.84], closing = positive
            JointMapping("six",   10, "flex", scale= 1.2, limit=(0.0,   1.84)),
        ])
    else:  # left — mirror of right
        return RobotHandConfig(joints=[
            JointMapping("zero",  12, "abd",  scale=-1.0, limit=(-0.52, 0.52), left_sign=-1.0),
            # thumb MCP: left range [-1.00, 1.20], closing = positive
            JointMapping("one",   13, "flex", scale= 1.0, limit=(-1.0,  1.2)),
            # thumb IP:  left range [0.00, +1.84], closing = positive
            JointMapping("two",   14, "flex", scale= 1.5, limit=(0.0,   1.84)),
            # index+middle MCP: left range [-1.84, +0.30], closing = negative
            JointMapping("three",  0, "flex", scale=-1.0, limit=(-1.84, 0.3)),
            # index+middle PIP: left range [-1.84, 0.00], closing = negative
            JointMapping("four",   1, "flex", scale=-1.2, limit=(-1.84, 0.0)),
            # ring+pinky MCP:  left range [-1.84, +0.30], closing = negative
            JointMapping("five",   9, "flex", scale=-1.0, limit=(-1.84, 0.3)),
            # ring+pinky PIP:  left range [-1.84, 0.00], closing = negative
            JointMapping("six",   10, "flex", scale=-1.2, limit=(-1.84, 0.0)),
        ])


# ── XLeRobot jaw gripper config ──────────────────────────────────────────────
#
# Single DoF jaw: middle PIP flexion drives jaw angle continuously.
#   open hand  (pip flex ≈ 0.0 rad) → 1.70 rad (jaw fully open)
#   tight grasp (pip flex ≈ 1.5 rad) → 0.00 rad (jaw fully closed)
# Middle PIP (feats[4, 0]) provides more range than MCP for real grasps.
# Denominator 1.5 rad avoids premature closure on WiLOR's slightly-curled estimates.
# Linear map: jaw = pip_flex * (-1.7/1.5) + 1.7, clamped to [0, 1.7].

def _xlerobot_jaw_from_keypoints(side: str) -> RobotHandConfig:
    return RobotHandConfig(joints=[
        JointMapping("gripper", 4, "flex",
                     scale=-1.7 / 1.5, offset=1.7, limit=(0.0, 1.7)),
    ])


# ── Franka parallel-jaw gripper config ───────────────────────────────────────
#
# Single DoF: middle PIP flexion drives gripper width continuously.
#   open hand  (pip flex ≈ 0.0 rad) → 0.04 m (fully open)
#   tight grasp (pip flex ≈ 1.5 rad) → 0.00 m (fully closed)
# Middle PIP (feats[4, 0]) has more dynamic range than MCP during real grasps.
# Denominator 1.5 rad matches a typical tight grasp without closing prematurely
# on moderately curled fingers that WiLOR predicts even for open hands.
# Linear map: width = pip_flex * (-0.04/1.5) + 0.04, clamped to [0, 0.04].

def _franka_gripper_from_keypoints(side: str) -> RobotHandConfig:
    return RobotHandConfig(joints=[
        JointMapping("gripper", 4, "flex",
                     scale=-0.04 / 1.5, offset=0.04, limit=(0.0, 0.04)),
    ])


# ── Robonaut 2 dexterous hand retargeter ─────────────────────────────────────
#
# Joint order per hand (18 DoF), matching r2.xml declaration order:
#   index:  joint0=abduction [-0.35, 0.35], joint1=MCP [0, 1.57],
#           joint2=PIP [0, 1.57], joint3=DIP [0, 1.57]
#   middle: joint0=abduction [-0.35, 0.35], joint1=MCP [0, 1.57],
#           joint2=PIP [0, 1.57], joint3=DIP [0, 1.57]
#   little: joint0=proximal [0, 2.97], joint1=medial [0, 2.97],
#           joint2=distal [0, 2.97]
#   ring:   joint0=proximal [0, 2.97], joint1=medial [0, 2.97],
#           joint2=distal [0, 2.97]
#   thumb:  joint0=CMC spread (R:[0,1.22] L:[-1.22,0]),
#           joint1=MCP [0, 1.40], joint2=IP [0, 1.22], joint3=DIP [-0.52, 1.57]
#
# MANO features used:
#   index:  feats[0,1]=abd, feats[0,0]=MCP, feats[1,0]=PIP, feats[2,0]=DIP
#   middle: feats[3,1]=abd, feats[3,0]=MCP, feats[4,0]=PIP, feats[5,0]=DIP
#   little: feats[6,0]=MCP, feats[7,0]=PIP, feats[8,0]=DIP
#   ring:   feats[9,0]=MCP, feats[10,0]=PIP, feats[11,0]=DIP
#   thumb:  feats[12,1]=CMC-spread, feats[12,0]=CMC-flex→MCP,
#           feats[13,0]=MCP-flex→IP, feats[14,0]=IP-flex→DIP

def _r2_18dof_from_keypoints(side: str) -> RobotHandConfig:
    # Thumb CMC spread: right=[0,1.22], left=[-1.22,0].
    # feats[12,1] > 0 when thumb sticks out of palm.
    # Right: val = feats[12,1]; left: val = feats[12,1] * (-1) * left_sign.
    # left_sign=1.0 → left val = -feats[12,1], clipped to [-1.22, 0] ✓
    thumb_cmc_lim = (0.0, 1.2217) if side == "right" else (-1.2217, 0.0)
    return RobotHandConfig(joints=[
        # ── index ──────────────────────────────────────────────────────────
        JointMapping("index_abd", 0, "abd",  scale=+1.0, limit=(-0.3491, 0.3491), left_sign=-1.0),
        JointMapping("index_mcp", 0, "flex", scale=+1.0, limit=(0.0, 1.57)),
        JointMapping("index_pip", 1, "flex", scale=+1.0, limit=(0.0, 1.57)),
        JointMapping("index_dip", 2, "flex", scale=+1.0, limit=(0.0, 1.57)),
        # ── middle ─────────────────────────────────────────────────────────
        JointMapping("middle_abd", 3, "abd",  scale=+1.0, limit=(-0.3491, 0.3491), left_sign=-1.0),
        JointMapping("middle_mcp", 3, "flex", scale=+1.0, limit=(0.0, 1.57)),
        JointMapping("middle_pip", 4, "flex", scale=+1.0, limit=(0.0, 1.57)),
        JointMapping("middle_dip", 5, "flex", scale=+1.0, limit=(0.0, 1.57)),
        # ── little (pinky) ─────────────────────────────────────────────────
        JointMapping("little_prox", 6, "flex", scale=+1.0, limit=(0.0, 2.9671)),
        JointMapping("little_med",  7, "flex", scale=+1.0, limit=(0.0, 2.9671)),
        JointMapping("little_dist", 8, "flex", scale=+1.0, limit=(0.0, 2.9671)),
        # ── ring ───────────────────────────────────────────────────────────
        JointMapping("ring_prox",   9,  "flex", scale=+1.0, limit=(0.0, 2.9671)),
        JointMapping("ring_med",   10,  "flex", scale=+1.0, limit=(0.0, 2.9671)),
        JointMapping("ring_dist",  11,  "flex", scale=+1.0, limit=(0.0, 2.9671)),
        # ── thumb ──────────────────────────────────────────────────────────
        JointMapping("thumb_cmc_spread", 12, "abd",  scale=+1.0, limit=thumb_cmc_lim, left_sign=1.0),
        JointMapping("thumb_mcp",        12, "flex", scale=+1.0, limit=(0.0, 1.3963)),
        JointMapping("thumb_ip",         13, "flex", scale=+1.0, limit=(0.0, 1.2217)),
        JointMapping("thumb_dip",        14, "flex", scale=+1.0, limit=(-0.5236, 1.57)),
    ])


# ── per-robot factory ─────────────────────────────────────────────────────────

def get_wilor_hand_retargeter(
    robot: str,
    side:  str,
):
    """
    Return a WilorHandRetargeter for the given robot and side.

    Parameters
    ----------
    robot : "g1" | "franka" | "robonaut2" | "xlerobot"
    side  : "left" | "right"
    """
    if robot == "g1":
        return WilorHandRetargeter(_g1_7dof_from_keypoints(side), side)

    if robot == "franka":
        return WilorHandRetargeter(_franka_gripper_from_keypoints(side), side)

    if robot == "robonaut2":
        return WilorHandRetargeter(_r2_18dof_from_keypoints(side), side)

    if robot == "xlerobot":
        return WilorHandRetargeter(_xlerobot_jaw_from_keypoints(side), side)

    raise ValueError(f"Unknown robot '{robot}'. "
                     f"Add it to wilor_retargeter.py or to _ROBOTS_WITHOUT_HANDS.")
