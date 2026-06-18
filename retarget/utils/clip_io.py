"""
Clip I/O utilities — loading extracted clip directories and saving trajectories.

Clip directory format
---------------------
    hand_joints.bin  — (T, max_h, 21, 3) float32  camera-frame keypoints (NaN = absent)
    hand_meta.json   — n_frames, max_hands, joints_shape, is_right_per_frame
    scene.json       — camera.focal, camera.gravity_up, fps, id
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── MANO joint indices ────────────────────────────────────────────────────────

_J_WRIST  = 0
_J_INDEX  = 5    # index MCP
_J_MIDDLE = 9    # middle MCP
_J_PINKY  = 17   # pinky MCP
_INVALID  = -1.0


# ── wrist pose helpers ────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def _joints_to_wrist_pose(joints: np.ndarray, side: str) -> np.ndarray | None:
    """21 MANO keypoints (camera frame) → 7D wrist pose [pos(3) | quat_wxyz(4)].

    Returns None if joints contain the invalid sentinel.
    Frame: y=finger direction, x=lateral (side-aware), z=palm normal (x×y).
    """
    if joints[_J_WRIST, 0] == _INVALID:
        return None
    pos   = joints[_J_WRIST].copy()
    y     = _normalize(joints[_J_MIDDLE] - joints[_J_WRIST])
    x_raw = _normalize((joints[_J_PINKY] - joints[_J_INDEX])
                       if side == "left" else
                       (joints[_J_INDEX] - joints[_J_PINKY]))
    x = _normalize(x_raw - np.dot(x_raw, y) * y)
    z = _normalize(np.cross(x, y))
    x = _normalize(np.cross(y, z))
    return np.concatenate([pos, _rotmat_to_quat(np.stack([x, y, z], axis=1))]).astype(np.float32)


# ── clip reader ───────────────────────────────────────────────────────────────

class SamplesSequence:
    """Loads one extracted clip directory for retargeting inference."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path / "hand_meta.json") as f:
            meta = json.load(f)
        with open(self.path / "scene.json") as f:
            scene = json.load(f)

        shape = tuple(meta["joints_shape"])
        self.joints_world = np.fromfile(
            self.path / "hand_joints.bin", dtype=np.float32).reshape(shape)
        self.n_frames = int(meta["n_frames"])
        max_h = shape[1]

        hand_is_right = np.full((self.n_frames, max_h), -1, dtype=np.int8)
        for t, slots in enumerate(meta["is_right_per_frame"]):
            for s, val in enumerate(slots):
                if val is True:   hand_is_right[t, s] = 1
                elif val is False: hand_is_right[t, s] = 0
        hand_is_right[np.isnan(self.joints_world).any(axis=(-1, -2))] = -1
        self.hand_is_right = hand_is_right

        cam = scene["camera"]
        self.focal = float(cam["focal"])
        gravity_up = np.array(cam["gravity_up"], dtype=np.float32)
        gravity_up /= np.linalg.norm(gravity_up) + 1e-8
        self.g_cam = -gravity_up
        self.fps   = float(scene.get("fps", 20.0))

        scene_id = scene.get("id", self.path.name)
        parts = scene_id.rsplit("_", 2)
        self.video_id  = parts[0] if len(parts) == 3 else scene_id
        self.start_sec = float(parts[1]) if len(parts) == 3 else 0.0
        self.end_sec   = float(parts[2]) if len(parts) == 3 else 0.0

        self._left_traj, self._right_traj = self._build_trajectories()

    def _build_trajectories(self):
        T, max_h = self.n_frames, self.joints_world.shape[1]
        left_traj  = np.full((T, 7), np.nan, dtype=np.float32)
        right_traj = np.full((T, 7), np.nan, dtype=np.float32)
        for t in range(T):
            for slot in range(max_h):
                flag = int(self.hand_is_right[t, slot])
                if flag == -1:
                    continue
                side = "right" if flag == 1 else "left"
                pose = _joints_to_wrist_pose(self.joints_world[t, slot], side)
                if pose is not None:
                    (right_traj if side == "right" else left_traj)[t] = pose
        return left_traj, right_traj

    def _fill_gaps(self, traj: np.ndarray) -> np.ndarray | None:
        valid = ~np.isnan(traj[:, 0])
        if not valid.any():
            return None
        traj = traj.copy()
        last = traj[valid][0]
        for i in range(len(traj)):
            if valid[i]: last = traj[i]
            else:        traj[i] = last
        return traj

    def get_window(self, start: int, length: int) -> dict:
        end = min(start + length, self.n_frames)
        def _window(traj):
            w = traj[start:end].copy()
            if len(w) < length:
                w = np.concatenate([w, np.tile(w[-1:], (length - len(w), 1))])
            return self._fill_gaps(w)
        return {
            "left_traj":   _window(self._left_traj),
            "right_traj":  _window(self._right_traj),
            "g_cam":       self.g_cam.copy(),
            "focal":       self.focal,
            "video_id":    self.video_id,
            "start_sec":   self.start_sec,
            "end_sec":     self.end_sec,
            "frame_start": start,
        }


# ── trajectory saving ─────────────────────────────────────────────────────────

def save_trajectory(path, q_left, q_right, Q_lf, Q_rf,
                    l_jnames, r_jnames, f_jnames, fps, robot, clip_id):
    """Save arm (and optionally finger) joint trajectories to an .npz file."""
    data = {
        "q_left":                q_left.astype(np.float32),
        "q_right":               q_right.astype(np.float32),
        "left_arm_joint_names":  np.array(l_jnames, dtype=object),
        "right_arm_joint_names": np.array(r_jnames, dtype=object),
        "fps":     np.float32(fps),
        "robot":   np.array(robot),
        "clip_id": np.array(clip_id),
    }
    if Q_lf is not None:
        data["q_left_fingers"]          = Q_lf.astype(np.float32)
        data["left_finger_joint_names"] = np.array(
            [f"left_{n}_joint" for n in f_jnames], dtype=object)
    if Q_rf is not None:
        data["q_right_fingers"]          = Q_rf.astype(np.float32)
        data["right_finger_joint_names"] = np.array(
            [f"right_{n}_joint" for n in f_jnames], dtype=object)
    np.savez(path, **data)
