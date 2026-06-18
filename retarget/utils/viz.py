"""
Visualization utilities — video I/O, OpenCV overlays, and MuJoCo offscreen rendering.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np


# ── video I/O ─────────────────────────────────────────────────────────────────

def load_video_frames(path: Path) -> list[np.ndarray]:
    """Load all frames from a video file. Returns empty list if file is missing."""
    frames = []
    if not path.exists():
        return frames
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write a list of BGR frames to an mp4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# ── MuJoCo offscreen rendering ────────────────────────────────────────────────

def render_robot_sim(env, q_left, q_right, Q_lf, Q_rf, finger_jnames,
                     robot_cfg, height=1080, width=1440) -> list[np.ndarray]:
    """Render a bilateral arm trajectory to a list of BGR frames."""
    T = len(q_left)
    try:
        renderer = mujoco.Renderer(env.model, height=height, width=width)
    except Exception as e:
        print(f"  [warn] offscreen renderer unavailable: {e}")
        return []
    cam = mujoco.MjvCamera()
    cam.azimuth   = float(robot_cfg["cam_azimuth"])
    cam.elevation = float(robot_cfg["cam_elevation"])
    cam.distance  = float(robot_cfg["cam_distance"])
    cam.lookat[:] = robot_cfg["cam_lookat"]
    frames = []
    for t in range(T):
        env.set_arm_joints("left",  q_left[t])
        env.set_arm_joints("right", q_right[t])
        if Q_lf is not None:
            env.set_finger_joints(Q_lf[t], [f"left_{n}_joint"  for n in finger_jnames])
        if Q_rf is not None:
            env.set_finger_joints(Q_rf[t], [f"right_{n}_joint" for n in finger_jnames])
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render()[:, :, ::-1].copy())
    renderer.close()
    return frames


# ── projection helpers ────────────────────────────────────────────────────────

def project_point(pt, K):
    """Project a 3D camera-frame point to pixel coordinates. Returns None if behind camera."""
    if pt[2] <= 0:
        return None
    return (int(K["fx"] * pt[0] / pt[2] + K["cx"]),
            int(K["fy"] * pt[1] / pt[2] + K["cy"]))


# ── overlay drawing ───────────────────────────────────────────────────────────

def draw_frame(img, pos, R, K, axis_len, label, dot_color):
    """Draw a coordinate frame (RGB axes + dot + label) projected into the image."""
    origin = project_point(pos, K)
    if origin is None:
        return
    for i, (color, al) in enumerate(zip([(0,0,220),(0,220,0),(220,0,0)], ["X","Y","Z"])):
        tip = project_point(pos + axis_len * R[:, i], K)
        if tip:
            cv2.arrowedLine(img, origin, tip, color, 2, tipLength=0.25)
            cv2.putText(img, al, (tip[0]+2, tip[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.circle(img, origin, 6, dot_color, -1)
    cv2.circle(img, origin, 6, (0,0,0), 1)
    cv2.putText(img, label, (origin[0]+8, origin[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)


def draw_gravity(img, g_cam, K, anchor=(60, 60), arrow_px=50):
    """Draw a gravity direction arrow in the corner of the image."""
    dx, dy = K["fx"] * g_cam[0], K["fy"] * g_cam[1]
    n = np.sqrt(dx**2 + dy**2) + 1e-8
    tip = (anchor[0] + int(dx/n*arrow_px), anchor[1] + int(dy/n*arrow_px))
    cv2.arrowedLine(img, anchor, tip, (220,220,0), 2, tipLength=0.30)
    cv2.putText(img, "g", (tip[0]+4, tip[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,0), 1)


def draw_trail(img, positions, K, color, trail_len=20):
    """Draw a fading position trail projected into the image."""
    pts = positions[-trail_len:]
    for i, pos in enumerate(pts):
        px = project_point(pos, K)
        if px:
            a = (i + 1) / len(pts)
            cv2.circle(img, px, max(1, int(3*a)), tuple(int(ch*a) for ch in color), -1)


def draw_hand_keypoints(img, joints, K, color):
    """Draw 21-keypoint hand skeleton projected into the image."""
    _EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
              (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    px = [project_point(j, K) for j in joints]
    for p, c in _EDGES:
        if px[p] and px[c]:
            cv2.line(img, px[p], px[c], color, 1, cv2.LINE_AA)
    for i, p in enumerate(px):
        if p:
            cv2.circle(img, p, 4 if i == 0 else 2, color, -1)
