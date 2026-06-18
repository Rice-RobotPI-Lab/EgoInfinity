"""
Retarget a single extracted clip to a robot.

Pipeline
--------
  1. Run NeuralRootFrameEstimator on strided 2-second windows → (R_k, t_k) keyframes
  2. Cluster keyframes → best SE(3) anchor via IK-convergence scoring
  3. Blend each keyframe toward the anchor (torso_alpha: 0=anchor-only, 1=per-window)
  4. SLERP-interpolate blended keyframes → smooth per-frame root frame
  5. Convert wrist poses to per-frame root frame
  6. Batch IK → arm joint trajectories
  7. Interpolate failed frames, smooth, finger retarget
  8. Save trajectory.npz, input_viz.mp4, robot_sim.mp4, metrics.npz

Usage
-----
    # default robot (g1), checkpoint auto-resolved to ckpts/g1.pt
    python3 scripts/test.py /examples/--QALmP1nHtM_678.2_682.2

    # specify robot
    python3 scripts/test.py /examples/--QALmP1nHtM_678.2_682.2 --robot franka

    # custom checkpoint (e.g. your own trained model)
    python3 scripts/test.py /examples/--QALmP1nHtM_678.2_682.2 --robot franka \
        --ckpt runs/franka/best.pt

    # custom output directory
    python3 scripts/test.py /examples/--QALmP1nHtM_678.2_682.2 --robot g1 --out /results/g1/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vn_transformer import NeuralRootFrameEstimator
from models.root_opt import RootPoseOptimizer, _interpolate_failed
from models.collision import CollisionFilter
from sim.robots import ROBOT_CONFIGS as _ROBOT_CONFIGS
from kinematics.wilor_retargeter import get_wilor_hand_retargeter
from utils.clip_io import SamplesSequence, save_trajectory
from utils.viz import (load_video_frames, write_video, render_robot_sim,
                       draw_frame, draw_gravity, draw_trail, draw_hand_keypoints)
from utils.pose_utils import (estimate_root_poses, select_best_anchor,
                               blend_keyframes, interpolate_root_frames,
                               smooth_root_frames, cam_to_root_targets,
                               rescale_bilateral_separation)

_CKPTS_DIR = Path(__file__).parent.parent / "ckpts"


def _default_ckpt(robot: str) -> Path:
    return _CKPTS_DIR / f"{robot}.pt"


# ── helpers ───────────────────────────────────────────────────────────────────

def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = NeuralRootFrameEstimator(
        d_model=saved.get("d_model", 64),
        num_heads=saved.get("num_heads", 4),
        num_layers=saved.get("num_layers", 4),
        dim_feedforward=saved.get("dim_feedforward", 128),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded: {ckpt_path}  (epoch {ckpt.get('epoch','?')})")
    return model


# ── simulation preview ────────────────────────────────────────────────────────

def _preview_trajectory(env_vis, q_left, q_right, Q_lf, Q_rf, finger_jnames,
                        robot_cfg, fps: float) -> None:
    """Play the retargeted trajectory interactively in a MuJoCo viewer."""
    dt = 1.0 / fps
    cfg = robot_cfg
    with mujoco.viewer.launch_passive(env_vis.model, env_vis.data) as viewer:
        viewer.cam.azimuth   = cfg["cam_azimuth"]
        viewer.cam.elevation = cfg["cam_elevation"]
        viewer.cam.distance  = cfg["cam_distance"]
        viewer.cam.lookat[:] = cfg["cam_lookat"]
        viewer.sync()

        n_frames = len(q_left)
        t = 0
        while viewer.is_running():
            env_vis.set_arm_joints("left",  q_left[t].astype(np.float64))
            env_vis.set_arm_joints("right", q_right[t].astype(np.float64))
            if Q_lf is not None:
                env_vis.set_finger_joints(Q_lf[t], [f"left_{jn}_joint"  for jn in finger_jnames])
            if Q_rf is not None:
                env_vis.set_finger_joints(Q_rf[t], [f"right_{jn}_joint" for jn in finger_jnames])
            viewer.sync()
            time.sleep(dt)
            t = (t + 1) % n_frames

    os._exit(0)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_path  = Path(args.clip).resolve()
    robot_name = args.robot
    out_dir    = Path(args.out).resolve() if args.out else clip_path.parent / robot_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clip : {clip_path}")
    print(f"Robot: {robot_name}")
    print(f"Out  : {out_dir}")

    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt(robot_name)
    model     = _load_model(ckpt_path, device)

    seq     = SamplesSequence(clip_path)
    seq_fps = seq.fps
    seq_len = seq.n_frames
    window  = seq.get_window(0, seq_len)

    left_cam_np  = window["left_traj"]
    right_cam_np = window["right_traj"]
    g_cam        = window["g_cam"]

    if left_cam_np is None or right_cam_np is None:
        raise RuntimeError("Clip is missing both-hand tracking data.")

    vid_frames = load_video_frames(clip_path / "depth.mp4")
    vid_h, vid_w = (vid_frames[0].shape[:2] if vid_frames else (480, 640))
    K = {"fx": seq.focal, "fy": seq.focal,
         "cx": vid_w / 2.0, "cy": vid_h / 2.0}

    left_vis_pos  = left_cam_np[:,  :3].copy()
    right_vis_pos = right_cam_np[:, :3].copy()
    left_rotmats  = np.array([_quat_to_rotmat(left_cam_np[f, 3:])  for f in range(seq_len)])
    right_rotmats = np.array([_quat_to_rotmat(right_cam_np[f, 3:]) for f in range(seq_len)])

    n_slots = seq.joints_world.shape[1]
    all_left_j  = np.full((seq_len, 21, 3), np.nan, dtype=np.float32)
    all_right_j = np.full((seq_len, 21, 3), np.nan, dtype=np.float32)
    for t in range(seq_len):
        for slot in range(n_slots):
            if seq.hand_is_right[t, slot] == -1:
                continue
            jc = seq.joints_world[t, slot]
            if np.isnan(jc).any():
                continue
            if seq.hand_is_right[t, slot] == 0:
                all_left_j[t]  = jc
            else:
                all_right_j[t] = jc

    g_cam_t    = torch.tensor(g_cam, dtype=torch.float32).unsqueeze(0).to(device)
    _robot_cfg = _ROBOT_CONFIGS[robot_name]

    # Bilateral scaling
    _bts = _robot_cfg.get("bilateral_target_sep")
    if _bts is not None:
        left_cam_ik, right_cam_ik = rescale_bilateral_separation(
            left_cam_np, right_cam_np, float(_bts))
        print(f"  bilateral scale → {_bts:.3f}m")
    else:
        left_cam_ik  = left_cam_np
        right_cam_ik = right_cam_np

    # ── Step 1: estimate root poses at strided window positions ──────────────
    _stride = args.window_stride if args.window_stride is not None \
              else max(4, int(round(args.window_secs * seq_fps)))
    kf_positions, Rs, ts = estimate_root_poses(
        model, left_cam_ik, right_cam_ik, g_cam_t,
        seq_len, seq_fps, args.window_secs, _stride, device)

    # ── Step 2: build IK solver ───────────────────────────────────────────────
    _wsc           = _robot_cfg.get("workspace_center")
    _ik_robot_name = _robot_cfg["ik_robot"]
    _null_man      = 0.5 if robot_name == "franka" else 0.1
    _null_smo      = 0.5 if robot_name == "franka" else 0.2
    _tol_pos       = 0.1 if robot_name == "xlerobot" else args.tol_pos

    opt = RootPoseOptimizer(
        model, device=device,
        ik_robot=_ik_robot_name,
        workspace_center=_wsc,
        start_config=_robot_cfg["start_config"],
        tol_pos=_tol_pos,
        null_weight_man=_null_man,
        null_weight_smo=_null_smo,
    )

    # ── Step 3: cluster + score → best anchor ────────────────────────────────
    R_anchor, t_anchor, _ = select_best_anchor(
        Rs, ts, left_cam_ik, right_cam_ik, opt, _wsc,
        n_clusters=args.n_clusters)
    R_global, t_global = R_anchor, t_anchor

    print(f"  anchor t={t_anchor.round(3)}  "
          f"max_rot_dev={max(float(Rotation.from_matrix(R_anchor.T @ Rs[k]).magnitude()) for k in range(len(Rs))):.3f} rad")

    # ── Step 4: blend each window estimate toward best anchor ─────────────────
    Rs_blended, ts_blended = blend_keyframes(
        Rs, ts, args.torso_alpha, alpha_rot=args.torso_alpha_rot,
        R_anchor=R_anchor, t_anchor=t_anchor)

    # ── Step 5: interpolate to per-frame ──────────────────────────────────────
    if len(kf_positions) == seq_len:
        R_per_frame, t_per_frame = Rs_blended, ts_blended
    else:
        R_per_frame, t_per_frame = interpolate_root_frames(
            kf_positions, Rs_blended, ts_blended, seq_len)
    R_per_frame, t_per_frame = smooth_root_frames(
        R_per_frame, t_per_frame, sigma=args.torso_smooth_sigma)

    # ── Step 6: convert wrist poses to per-frame root frame ───────────────────
    left_pos, left_quat, right_pos, right_quat = cam_to_root_targets(
        left_cam_ik, right_cam_ik, R_per_frame, t_per_frame, _wsc)

    # ── Step 7: batch IK ──────────────────────────────────────────────────────
    lp_t = torch.tensor(left_pos,   dtype=torch.float32)
    lq_t = torch.tensor(left_quat,  dtype=torch.float32)
    rp_t = torch.tensor(right_pos,  dtype=torch.float32)
    rq_t = torch.tensor(right_quat, dtype=torch.float32)
    with torch.no_grad():
        q_l_t, info_l = opt.ik_left_traj.solve_batch( lp_t, lq_t)
        q_r_t, info_r = opt.ik_right_traj.solve_batch(rp_t, rq_t)
    q_left  = q_l_t.cpu().numpy()
    q_right = q_r_t.cpu().numpy()

    conv_l    = info_l["converged"].cpu().numpy().astype(bool)
    conv_r    = info_r["converged"].cpu().numpy().astype(bool)
    pos_err_l = info_l["pos_err"].cpu().numpy().astype(np.float32)
    pos_err_r = info_r["pos_err"].cpu().numpy().astype(np.float32)
    ori_err_l = info_l["ori_err"].cpu().numpy().astype(np.float32)
    ori_err_r = info_r["ori_err"].cpu().numpy().astype(np.float32)
    jlm_l     = info_l["joint_limit_margin"].cpu().numpy().astype(np.float32)
    jlm_r     = info_r["joint_limit_margin"].cpu().numpy().astype(np.float32)
    man_l     = info_l["manipulability"].cpu().numpy().astype(np.float32)
    man_r     = info_r["manipulability"].cpu().numpy().astype(np.float32)
    ik_rate   = (conv_l.sum() + conv_r.sum()) / (2 * seq_len)
    print(f"  IK L: {conv_l.sum()}/{seq_len} ({100*conv_l.mean():.1f}%)  "
          f"R: {conv_r.sum()}/{seq_len} ({100*conv_r.mean():.1f}%)  "
          f"overall: {100*ik_rate:.1f}%")

    _q_def_l = (opt.ik_left_traj.q_default.cpu().numpy()
                if opt.ik_left_traj.q_default is not None else None)
    _q_def_r = (opt.ik_right_traj.q_default.cpu().numpy()
                if opt.ik_right_traj.q_default is not None else None)
    q_left  = _interpolate_failed(q_left,  conv_l, _q_def_l)
    q_right = _interpolate_failed(q_right, conv_r, _q_def_r)

    # Self-collision post-processing
    if args.self_collision:
        _sc = CollisionFilter(_robot_cfg)
        q_left, q_right = _sc.process(q_left, q_right)

    # Joint smoothing
    if args.smooth_sigma > 0:
        _w = max(1, int(args.smooth_sigma * seq_fps))
        for _ in range(3):
            q_left  = uniform_filter1d(q_left,  size=_w, axis=0, origin=-(_w//2))
            q_right = uniform_filter1d(q_right, size=_w, axis=0, origin=-(_w//2))

    # Finger retargeting
    Q_lf = Q_rf = None
    finger_jnames: list[str] = []
    _rt = _robot_cfg["retargeter"]
    rt_l = get_wilor_hand_retargeter(_rt, "left")  if _rt else None
    rt_r = get_wilor_hand_retargeter(_rt, "right") if _rt else None
    if rt_l is not None and not np.isnan(all_left_j).all():
        valid = ~np.isnan(all_left_j).any(axis=(1, 2))
        Q_lf = np.zeros((seq_len, rt_l.n_dof), dtype=np.float32)
        for t in np.where(valid)[0]:
            Q_lf[t] = rt_l.retarget(all_left_j[t])
        last = Q_lf[np.where(valid)[0][0]] if valid.any() else None
        for t in range(seq_len):
            if valid[t]:         last = Q_lf[t]
            elif last is not None: Q_lf[t] = last
        finger_jnames = rt_l.joint_names
    if rt_r is not None and not np.isnan(all_right_j).all():
        valid = ~np.isnan(all_right_j).any(axis=(1, 2))
        Q_rf = np.zeros((seq_len, rt_r.n_dof), dtype=np.float32)
        for t in np.where(valid)[0]:
            Q_rf[t] = rt_r.retarget(all_right_j[t])
        last = Q_rf[np.where(valid)[0][0]] if valid.any() else None
        for t in range(seq_len):
            if valid[t]:         last = Q_rf[t]
            elif last is not None: Q_rf[t] = last
        finger_jnames = finger_jnames or rt_r.joint_names

    # Build env for joint names + rendering
    scene_path = _robot_cfg.get("scene_path_fingers", _robot_cfg["scene_path"])
    env_vis = _robot_cfg["env_cls"](
        mjcf_path=scene_path, start_config=_robot_cfg["start_config"])
    env_vis.reset()
    l_jnames = [mujoco.mj_id2name(env_vis.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                for jid in env_vis._joint_ids["left"]]
    r_jnames = [mujoco.mj_id2name(env_vis.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                for jid in env_vis._joint_ids["right"]]

    # ── input viz: wrist trails + anchor + per-frame root frame ──────────────
    def _get_vid_frame(idx):
        if vid_frames:
            i = min(max(0, int(round(idx / seq_fps * seq_fps))), len(vid_frames) - 1)
            return vid_frames[i].copy()
        return np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

    input_frames = []
    for t in range(seq_len):
        img = _get_vid_frame(t)
        draw_trail(img, left_vis_pos[:t+1],  K, (0, 200, 0))
        draw_trail(img, right_vis_pos[:t+1], K, (200, 100, 0))
        draw_frame(img, left_vis_pos[t],  left_rotmats[t],
                    K, args.axis_len_wrist, "L", (0, 200, 0))
        draw_frame(img, right_vis_pos[t], right_rotmats[t],
                    K, args.axis_len_wrist, "R", (200, 100, 0))
        draw_frame(img, t_global, R_global,
                    K, args.axis_len_torso * 0.7, "anchor", (0, 220, 220))
        draw_frame(img, t_per_frame[t], R_per_frame[t],
                    K, args.axis_len_torso, "root",
                    (0, 220, 220) if args.torso_alpha == 0 else (220, 220, 0))
        draw_gravity(img, g_cam, K)
        if not np.isnan(all_left_j[t]).all():
            draw_hand_keypoints(img, all_left_j[t],  K, (0, 220, 100))
        if not np.isnan(all_right_j[t]).all():
            draw_hand_keypoints(img, all_right_j[t], K, (0, 100, 220))
        cv2.putText(img, f"{t+1}/{seq_len}  {robot_name}  a={args.torso_alpha:.2f}",
                    (8, vid_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        input_frames.append(img)

    robot_frames = render_robot_sim(
        env_vis, q_left, q_right, Q_lf, Q_rf, finger_jnames, _robot_cfg)

    # ── save ──────────────────────────────────────────────────────────────────
    save_trajectory(out_dir / "trajectory.npz",
                     q_left, q_right, Q_lf, Q_rf,
                     l_jnames, r_jnames, finger_jnames,
                     seq_fps, robot_name, clip_path.name)
    np.savez(out_dir / "root_frames.npz",
             R_per_frame=R_per_frame.astype(np.float32),
             t_per_frame=t_per_frame.astype(np.float32),
             R_anchor=R_anchor.astype(np.float32),
             t_anchor=t_anchor.astype(np.float32))
    write_video(out_dir / "input_viz.mp4",  input_frames,  seq_fps)
    write_video(out_dir / "robot_sim.mp4",  robot_frames,  seq_fps)

    def _roughness(q):
        return float(np.mean(np.diff(q, axis=0) ** 2))
    np.savez(
        out_dir / "metrics.npz",
        ik_rate_l        = np.float32(conv_l.mean()),
        ik_rate_r        = np.float32(conv_r.mean()),
        ik_rate          = np.float32(ik_rate),
        pos_err_l        = pos_err_l,
        pos_err_r        = pos_err_r,
        ori_err_l        = ori_err_l,
        ori_err_r        = ori_err_r,
        pos_err_l_all    = np.float32(pos_err_l.mean()),
        pos_err_r_all    = np.float32(pos_err_r.mean()),
        ori_err_l_all    = np.float32(ori_err_l.mean()),
        ori_err_r_all    = np.float32(ori_err_r.mean()),
        pos_err_l_conv   = np.float32(pos_err_l[conv_l].mean() if conv_l.any() else np.nan),
        pos_err_r_conv   = np.float32(pos_err_r[conv_r].mean() if conv_r.any() else np.nan),
        ori_err_l_conv   = np.float32(ori_err_l[conv_l].mean() if conv_l.any() else np.nan),
        ori_err_r_conv   = np.float32(ori_err_r[conv_r].mean() if conv_r.any() else np.nan),
        jlm_l            = jlm_l,
        jlm_r            = jlm_r,
        manipulability_l = man_l,
        manipulability_r = man_r,
        roughness        = np.float32((_roughness(q_left) + _roughness(q_right)) / 2),
        torso_alpha      = np.float32(args.torso_alpha),
        n_frames         = np.int32(seq_len),
        n_keyframes      = np.int32(len(kf_positions)),
        window_secs      = np.float32(args.window_secs),
        fps              = np.float32(seq_fps),
        robot            = np.array(robot_name),
        clip             = np.array(clip_path.name),
    )

    print(f"\nResults saved to: {out_dir}")

    if not args.no_preview:
        print("Launching interactive preview (close the viewer window to exit)…")
        _preview_trajectory(env_vis, q_left, q_right, Q_lf, Q_rf, finger_jnames,
                            _robot_cfg, seq_fps)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retarget a single extracted clip to a robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument("clip",
                   help="Path to extracted clip directory "
                        "(e.g. /examples/--QALmP1nHtM_678.2_682.2). "
                        "Must contain hand tracking data produced by the extraction pipeline.")

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--robot",  default="g1",
                   choices=["g1", "franka", "robonaut2", "xlerobot"],
                   help="Target robot to retarget to.")
    p.add_argument("--ckpt",   default=None, metavar="PATH",
                   help="Path to model checkpoint. "
                        "Defaults to ckpts/<robot>.pt.")
    p.add_argument("--out",    default=None, metavar="DIR",
                   help="Directory where results are written. "
                        "Defaults to <clip_parent>/<robot>/.")

    # ── root frame estimation ─────────────────────────────────────────────────
    p.add_argument("--window_secs",        type=float, default=2.0,
                   help="Duration of each sliding window used for root frame "
                        "estimation. Longer windows give more stable estimates "
                        "at the cost of temporal resolution.")
    p.add_argument("--window_stride",      type=int,   default=None,
                   metavar="FRAMES",
                   help="Step size (in frames) between consecutive window "
                        "centres. Defaults to window_len (non-overlapping). "
                        "Set to 1 for a fully sliding window.")
    p.add_argument("--n_clusters",         type=int,   default=5,
                   help="Number of SE(3) K-medoids clusters used to find the "
                        "best root anchor from all window estimates.")
    p.add_argument("--torso_alpha",        type=float, default=0.3,
                   help="Translation blend weight between the global anchor "
                        "(0) and the per-window estimate (1). Lower values "
                        "keep the torso more static.")
    p.add_argument("--torso_alpha_rot",    type=float, default=0.7,
                   help="Rotation blend weight between the global anchor "
                        "(0) and the per-window estimate (1).")
    p.add_argument("--torso_smooth_sigma", type=float, default=10.0,
                   metavar="FRAMES",
                   help="Gaussian smoothing sigma (in frames) applied to the "
                        "per-frame root trajectory after interpolation. "
                        "Set to 0 to disable.")

    # ── IK / post-processing ──────────────────────────────────────────────────
    p.add_argument("--tol_pos",       type=float, default=0.01,
                   metavar="M",
                   help="IK position convergence tolerance in metres.")
    p.add_argument("--smooth_sigma",  type=float, default=0.1,
                   metavar="SEC",
                   help="Joint-space Gaussian smoothing window in seconds "
                        "applied after IK. Set to 0 to disable.")
    p.add_argument("--self_collision", action="store_true",
                   help="Run gradient-based self-collision post-processing "
                        "(CollisionFilter) on the final joint trajectory.")

    # ── visualisation ─────────────────────────────────────────────────────────
    p.add_argument("--axis_len_wrist", type=float, default=0.15,
                   metavar="M",
                   help="Length of the wrist coordinate-frame axes drawn in "
                        "input_viz.mp4.")
    p.add_argument("--axis_len_torso", type=float, default=0.30,
                   metavar="M",
                   help="Length of the root coordinate-frame axes drawn in "
                        "input_viz.mp4.")
    p.add_argument("--no-preview", dest="no_preview", action="store_true",
                   help="Skip the interactive MuJoCo viewer playback after saving results.")

    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
