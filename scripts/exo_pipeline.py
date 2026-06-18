#!/usr/bin/env python3
"""
Standalone exo pipeline for Exo Browser.
Takes a directory of JPEG frames, runs WiLoR + MoGe-2 + depth stabilization,
launches viser 3D viewer. No HO-Cap metadata, no GT, no ego camera.

Progress is printed to stdout for the parent process to parse.
"""
import os, sys, time, logging, argparse, glob, threading
import numpy as np
import cv2
# torch is imported lazily in main() — only when not in --load-cache mode.
# This saves ~11s on viser launches that just visualize a cached pkl.
import viser

logging.basicConfig(level=logging.INFO, format='%(message)s')

REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from scripts.pipeline_utils import (
    FINGER_COLORS_BGR, FRUSTUM_SCALE, OBB_EDGES,
    free_gpu, bone_colors_bgr, make_line_segments_3d,
    draw_skeleton_2d, depth_to_colormap, depth_to_pointcloud,
)
from egoinfinity.pipeline.config import HAND_EDGES


def progress(msg):
    """Print progress for parent process to parse."""
    print(f"PROGRESS| {msg}", flush=True)


def _decode_image_maybe(val):
    """Return a (H, W, 3) RGB ndarray for `val`, accepting any of:
        - None → returns None
        - bytes (JPG-encoded) → decoded RGB ndarray
        - ndarray → passthrough (assumed already RGB)
    Used for backward compat with old pkls that stored raw arrays.
    """
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        arr = np.frombuffer(val, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return val


def save_cache(cache_dir, state):
    """Save frame_data + global state to disk in compact format v2.

    Compact v2 (pkl format version 2):
      - depth_map  -> 'depth_png' bytes (uint16 mm, PNG)        ~9x smaller
      - bg_template -> 'bg_template_png' bytes                  ~7x smaller
      - sam3_obj_data[oid] drops 'pts' (regenerable from mask + depth)
    Reading is handled by `pipeline_utils.rehydrate_pkl` (also called from
    `load_cache` below); legacy v1 caches still load.
    """
    import pickle, gzip
    from scripts.pipeline_utils import encode_depth_png, PKL_FORMAT_VERSION
    cache_path = os.path.join(cache_dir, 'pipeline_result.pkl.gz')
    tmp_path = cache_path + '.tmp'

    def _strip_pts(sam3):
        """Return a shallow-copy of sam3_obj_data with 'pts' removed per object."""
        if not sam3:
            return sam3
        out = {}
        for oid, entry in sam3.items():
            if isinstance(entry, dict):
                out[oid] = {k: v for k, v in entry.items() if k != 'pts'}
            else:
                out[oid] = entry
        return out

    data = {
        '_pkl_format_version': PKL_FORMAT_VERSION,
        'frame_data': [{
            'img_rgb': fd['img_rgb'],          # JPG bytes
            'flow_rgb': fd.get('flow_rgb'),    # JPG bytes or None
            # depth_rgb regenerated from depth_map on load (saves space)
            'joints_3d_pred': fd['joints_3d_pred'],
            'joints_2d_pred': fd.get('joints_2d_pred', []),
            'vertices_3d': fd['vertices_3d'],
            'hand_is_right': fd['hand_is_right'],
            'hand_meta': fd.get('hand_meta', []),
            # depth as uint16-mm PNG; raw float32 dropped
            'depth_png': encode_depth_png(fd['depth_map']),
            'obj_data': fd['obj_data'],
            'sam3_obj_data': _strip_pts(fd.get('sam3_obj_data', {})),
        } for fd in state['frame_data']],
        'dp_focal': state['dp_focal'],
        'mano_faces': state.get('mano_faces'),
        # bg_template as uint16-mm PNG; raw float32 dropped
        'bg_template_png': encode_depth_png(state.get('bg_template')),
        'bg_rgb': state.get('bg_rgb'),
        'cx': state.get('cx'),
        'cy': state.get('cy'),
        'sam3_prompts': state.get('sam3_prompts', []),
        'sam3_prompt_mapping': state.get('sam3_prompt_mapping', []),
        'sam3_mesh_info': state.get('sam3_mesh_info', {}),
        'pose_track_info': state.get('pose_track_info', {}),
        'gravity_up': state.get('gravity_up'),
        'gravity_roll_deg': state.get('gravity_roll_deg'),
        'gravity_pitch_deg': state.get('gravity_pitch_deg'),
    }
    # compresslevel=3 trades ~30% larger pkl for ~3-5x faster write
    # (level 9 default was ~2-4s on 180MB; level 3 ~0.5-1s).
    # Decompressed pickle content is identical; existing gzip-9 caches still read.
    with gzip.open(tmp_path, 'wb', compresslevel=3) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    progress(f"Cache saved: {cache_path} ({size_mb:.1f} MB)")


def load_cache(cache_dir):
    """Load cached frame_data + global state from disk.

    Handles both compact v2 (depth_png + dropped pts) and legacy v1 caches —
    `rehydrate_pkl` decodes depth and recomputes pts to canonical in-memory
    shape so downstream code sees float32 depth_map and pts as before.
    """
    import pickle, gzip
    from scripts.pipeline_utils import rehydrate_pkl
    cache_path = os.path.join(cache_dir, 'pipeline_result.pkl.gz')
    progress(f"Loading cache: {cache_path}")
    with gzip.open(cache_path, 'rb') as f:
        data = pickle.load(f)
    rehydrate_pkl(data)
    # Regenerate depth_rgb from (now-canonical) depth_map
    for fd in data['frame_data']:
        fd['depth_rgb'] = cv2.cvtColor(depth_to_colormap(fd['depth_map']), cv2.COLOR_BGR2RGB)
        fd['metrics'] = {}
    progress(f"Cache loaded: {len(data['frame_data'])} frames")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='(deprecated) Max frames to process. Use --start_frame/--end_frame instead.')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='First frame index to process (inclusive)')
    parser.add_argument('--end_frame', type=int, default=-1,
                        help='Last frame index to process (inclusive, -1 = all)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--segment-id', type=int, default=None,
                        help='Segment ID (used by viser GUI to call /api/favorite/<id>)')
    parser.add_argument('--app-port', type=int, default=28421,
                        help='Curation app HTTP port (optional, for favorite-state callbacks)')
    parser.add_argument('--obj_point', type=float, nargs=2, default=None,
                        help='(legacy) Single object point: x y')
    parser.add_argument('--obj_points', type=str, default=None,
                        help='(legacy) JSON list of object points: [[x1,y1],[x2,y2],...]')
    parser.add_argument('--obj_prompts', type=str, default=None,
                        help='JSON list of {type,data} prompts for object tracking')
    parser.add_argument('--obj_frame', type=int, default=0,
                        help='Frame index where obj_points were selected')
    parser.add_argument('--sam3_prompts', type=str, default=None,
                        help='Comma-separated text prompts for SAM3 detection '
                             '(e.g. "cauliflower,knife,cutting board")')
    parser.add_argument('--manifest_path', type=str, default=None,
                        help='Path to favorites/<id>/manifest.json. When set, '
                             'sam3_prompts is read from manifest.objects '
                             '(overrides --sam3_prompts). Used by the 4070 Ti S '
                             'batch workflow where Claude curates objects offline '
                             'into manifest.json (replacing the on-GPU Qwen path).')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to save/load pipeline_result.pkl.gz')
    parser.add_argument('--no-viser', action='store_true',
                        help='Exit after processing (no viser server)')
    parser.add_argument('--load-cache', action='store_true',
                        help='Skip processing, load cached result and launch viser')
    parser.add_argument('--no-infiller', action='store_true',
                        help='Disable HaWoR motion infiller for missing hand frames')
    parser.add_argument('--infiller-checkpoint', type=str, default=None,
                        help='Path to infiller .pt checkpoint (default: pretrained_models/infiller.pt)')
    args = parser.parse_args()

    # 4070 Ti S batch path: read SAM3 prompts from a curated manifest.json
    # (Claude-filled `objects` list) instead of relying on the on-GPU Qwen
    # extractor or Browser CLI. Overrides --sam3_prompts when both given.
    if args.manifest_path:
        try:
            import json as _json
            with open(args.manifest_path) as _mf:
                _manifest = _json.load(_mf)
            _objs = _manifest.get('objects') or []
            if _objs:
                args.sam3_prompts = ','.join(_objs)
                print(f"[manifest] loaded {len(_objs)} objects from "
                      f"{args.manifest_path}: {_objs}", flush=True)
            else:
                print(f"[manifest] {args.manifest_path} has empty 'objects' "
                      f"— Phase D-sam3 will be skipped", flush=True)
        except Exception as _e:
            print(f"[manifest] failed to read {args.manifest_path}: {_e}",
                  flush=True)

    # Lazy torch import: --load-cache only renders pkl, no model forward runs.
    # Skipping torch import here saves ~11s on every viser launch.
    if not args.load_cache:
        import torch
        import torch.serialization
        # PyTorch 2.6+ defaults weights_only=True, but older ultralytics/WiLoR checkpoints need pickle
        torch.serialization.add_safe_globals([])  # ensure module is initialized
        _orig_torch_load = torch.load
        torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})

    # Parse object prompts: new format [{type:'point'|'bbox', data:[...]}, ...],
    # with fallback to legacy --obj_points and --obj_point
    import json as _json
    obj_prompts_list = []  # list of {'type': 'point'|'bbox', 'data': [...]}
    if args.obj_prompts is not None:
        obj_prompts_list = _json.loads(args.obj_prompts)
    elif args.obj_points is not None:
        pts = _json.loads(args.obj_points)
        obj_prompts_list = [{'type': 'point', 'data': p} for p in pts]
    elif args.obj_point is not None:
        obj_prompts_list = [{'type': 'point', 'data': list(args.obj_point)}]
    has_object = len(obj_prompts_list) > 0
    obj_frame_idx = args.obj_frame

    image_files = sorted(glob.glob(os.path.join(args.frames_dir, '*.jpg')))
    assert len(image_files) > 0, f"No JPG frames in {args.frames_dir}"

    # Determine frame range
    n_all = len(image_files)
    if args.max_frames is not None:
        # Legacy --max_frames support
        sf = 0
        ef = min(n_all - 1, args.max_frames - 1)
    else:
        sf = max(0, min(args.start_frame, n_all - 1))
        ef = (n_all - 1) if args.end_frame < 0 else min(args.end_frame, n_all - 1)
    image_files = image_files[sf:ef + 1]
    total = len(image_files)

    first_img_bgr = cv2.imread(image_files[0])
    H, W = first_img_bgr.shape[:2]
    first_img_rgb = cv2.cvtColor(first_img_bgr, cv2.COLOR_BGR2RGB)
    cam_aspect = W / H

    # ── No-viser mode: use a random port to avoid conflict with running viser ──
    if args.no_viser:
        import random
        args.port = random.randint(19000, 29000)

    # ── Load cache mode ──────────────────────────────────────────
    if args.load_cache:
        assert args.cache_dir, "--cache-dir required with --load-cache"
        cached = load_cache(args.cache_dir)
        # Backfill hand_meta for old caches that don't have it
        for fd in cached['frame_data']:
            if 'hand_meta' not in fd:
                fd['hand_meta'] = [
                    {'source': 'unknown', 'confidence': -1,
                     'track_id': -1, 'cam_t_raw': None, 'cam_t_smooth': None}
                    for _ in fd.get('joints_3d_pred', [])]
            if 'sam3_obj_data' not in fd:
                fd['sam3_obj_data'] = {}

        state = {
            'frame_data': cached['frame_data'],
            'dp_focal': cached['dp_focal'],
            'mano_faces': cached.get('mano_faces'),
            'bg_template': cached.get('bg_template'),
            'bg_rgb': cached.get('bg_rgb'),
            'cx': cached.get('cx', W / 2.0),
            'cy': cached.get('cy', H / 2.0),
            'sam3_prompts': cached.get('sam3_prompts', []),
            'sam3_prompt_mapping': cached.get('sam3_prompt_mapping', []),
            'sam3_mesh_info': cached.get('sam3_mesh_info', {}),
            'pose_track_info': cached.get('pose_track_info', {}),
            'gravity_up': cached.get('gravity_up'),
            'gravity_roll_deg': cached.get('gravity_roll_deg'),
            'gravity_pitch_deg': cached.get('gravity_pitch_deg'),
            'processing': False,
            'ready': True,
        }
        total = len(state['frame_data'])
        # Fall through to viser setup below

    # ── Launch Viser ──────────────────────────────────────────────
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("-y")  # default; overridden by apply_gravity below
    progress(f"Viser ready on port {args.port} | {total} frames (range {sf}-{ef} of {n_all})")

    if not args.load_cache:
        state = {
            'frame_data': [],
            'dp_focal': 500.0,
            'sam3_prompts': [],
            'sam3_prompt_mapping': [],
            'processing': False,
            'ready': False,
        }

    # ── GUI ────────────────────────────────────────────────────────
    with server.gui.add_folder("Controls"):
        progress_md = server.gui.add_markdown(f"Processing {total} frames...")
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(total - 1, 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        ego_view_btn = server.gui.add_button("👁 Ego View")

    @ego_view_btn.on_click
    def _on_ego_view(_):
        # Hand-derived ego-style camera:
        #   position = H + 0.5 * up - 0.3 * forward_horiz
        #   look_at  = H
        # H = mean wrist+MCP position over all frames/hands
        # forward = mean(MCP_centroid - wrist), projected onto plane ⟂ up
        gravity_up = state.get('gravity_up')
        if gravity_up is None:
            gravity_up = np.array([0.0, -1.0, 0.0])
        u = gravity_up / (np.linalg.norm(gravity_up) + 1e-9)

        MCP_IDX = [1, 5, 9, 13, 17]
        positions, fwds = [], []
        for fd in state.get('frame_data', []):
            for j3d in fd.get('joints_3d_pred', []):
                if j3d is None or len(j3d) < 18:
                    continue
                j3d = np.asarray(j3d)
                wrist = j3d[0]
                mcp_c = j3d[MCP_IDX].mean(axis=0)
                positions.append(wrist)
                positions.append(mcp_c)
                d = mcp_c - wrist
                n = np.linalg.norm(d)
                if n > 1e-6:
                    fwds.append(d / n)
        if not positions or not fwds:
            print("[viser] Ego View: no hand data available")
            return

        H = np.mean(np.array(positions), axis=0)
        d = np.mean(np.array(fwds), axis=0)
        d_h = d - np.dot(d, u) * u
        n = np.linalg.norm(d_h)
        if n < 1e-4:
            d_h = d  # fallback: hand essentially vertical, skip projection
            n = np.linalg.norm(d_h) + 1e-9
        d_h = d_h / n

        C = H + 0.6 * u - 0.3 * d_h
        L = H + 0.3 * u
        cam_pos = (float(C[0]), float(C[1]), float(C[2]))
        look_at = (float(L[0]), float(L[1]), float(L[2]))
        up_dir  = (float(u[0]), float(u[1]), float(u[2]))
        for cli in server.get_clients().values():
            try:
                cli.camera.position = cam_pos
                cli.camera.look_at = look_at
                cli.camera.up_direction = up_dir
            except Exception as e:
                print(f"[viser] Ego View: client update failed: {e}")

    MAX_OBJECTS = 5
    OBJ_COLORS_RGB = [
        np.array([255, 100, 50], dtype=np.uint8),   # orange-red
        np.array([50, 200, 100], dtype=np.uint8),    # green
        np.array([100, 150, 255], dtype=np.uint8),   # blue
        np.array([255, 200, 50], dtype=np.uint8),    # yellow
        np.array([200, 100, 255], dtype=np.uint8),   # purple
    ]
    SAM3_OBJ_COLORS_RGB = [
        np.array([  0, 200, 255], dtype=np.uint8),   # cyan
        np.array([255, 200,   0], dtype=np.uint8),   # amber
        np.array([255,   0, 200], dtype=np.uint8),   # magenta
        np.array([ 80, 255,   0], dtype=np.uint8),   # chartreuse
        np.array([255, 100,   0], dtype=np.uint8),   # bright orange
    ]

    display_folder = server.gui.add_folder("Display", expand_by_default=False)
    with display_folder:
        show_mesh = server.gui.add_checkbox("Hand Mesh", initial_value=True)
        show_pred = server.gui.add_checkbox("Hand Skeleton", initial_value=False)
        # MANUAL_DETECT_DISABLED: manual object OBB + pointcloud hidden
        show_object = server.gui.add_checkbox("Object OBB (disabled)", initial_value=False)
        show_obj_pc = server.gui.add_checkbox("Object Pointcloud (disabled)", initial_value=False)
        show_sam3_obj_pc = server.gui.add_checkbox("SAM3 Auto Pointcloud", initial_value=True)
        show_sam3d_mesh = server.gui.add_checkbox("SAM3D Mesh (6DoF)", initial_value=True)
        show_state_dot = server.gui.add_checkbox(
            "State Dot (above mesh)", initial_value=True)
        show_sam3d_init = server.gui.add_checkbox("SAM3D Init Pose (debug)", initial_value=False)
        show_sam3_obb = server.gui.add_checkbox("SAM3 Object OBB", initial_value=True)
        show_obs_obb = server.gui.add_checkbox("Obs Cloud OBB (PCA)", initial_value=False)
        show_trajectories = server.gui.add_checkbox("Trajectories", initial_value=True)
        traj_length = server.gui.add_slider("Trail Length", min=5, max=100, step=5, initial_value=30)
        show_debug_colors = server.gui.add_checkbox("Debug: Source Colors", initial_value=False)
        show_depth_pc = server.gui.add_checkbox("Scene Pointcloud", initial_value=False)
        show_depth_overlay = server.gui.add_checkbox("Depth Overlay", initial_value=False)
        show_flow_overlay = server.gui.add_checkbox("Flow Overlay (MEMFOF)", initial_value=False)
        flow_alpha = server.gui.add_slider("Flow Alpha", min=0.0, max=1.0, step=0.05, initial_value=0.6)
        pc_step = server.gui.add_slider("PC Step", min=2, max=16, step=1, initial_value=6)

    # ── Favorites folder (only when running in --load-cache view mode) ──
    if args.segment_id is not None:
        with server.gui.add_folder("Favorites"):
            fav_note = server.gui.add_text("Note", initial_value="")
            fav_save_btn = server.gui.add_button("⭐ Save to Favorites")
            fav_status_md = server.gui.add_markdown(
                f"_segment_id={args.segment_id}_")

            @fav_save_btn.on_click
            def _on_fav_click(_):
                import urllib.request, urllib.parse, json as _json
                try:
                    fav_status_md.content = "_saving..._"
                    qs = urllib.parse.urlencode({'note': fav_note.value or ''})
                    req = urllib.request.Request(
                        f"http://localhost:{args.app_port}/api/favorite/"
                        f"{args.segment_id}?{qs}",
                        method='POST',
                    )
                    with urllib.request.urlopen(req, timeout=120) as r:
                        resp = _json.loads(r.read().decode('utf-8'))
                    if resp.get('ok'):
                        fav_status_md.content = (
                            f"✅ saved to `{resp['path']}`  \n"
                            f"size: **{resp['size_mb']} MB**  \n"
                            f"copied: {', '.join(resp['copied'])}"
                        )
                    else:
                        fav_status_md.content = f"❌ error: {resp.get('error', '?')}"
                except Exception as e:
                    fav_status_md.content = f"❌ exception: {e}"

    # ── Scene placeholders ─────────────────────────────────────────
    # Use (20, 2, 3) so the buffer size matches HAND_EDGES; otherwise viser
    # locks the line_segments node at 1 segment from the dummy init and
    # subsequent (20, 2, 3) updates only render the first edge.
    dummy_seg = np.zeros((20, 2, 3), dtype=np.float32)
    dummy_seg_c = np.zeros((20, 2, 3), dtype=np.uint8)
    dummy_pts = np.zeros((1, 3), dtype=np.float32)
    dummy_pts_c = np.zeros((1, 3), dtype=np.uint8)
    MAX_HANDS = 4

    MESH_COLORS = {True: (100, 200, 255), False: (255, 180, 100)}  # right=blue, left=orange
    _dummy_verts = np.zeros((3, 3), dtype=np.float32)
    _dummy_faces = np.array([[0, 1, 2]], dtype=np.int32)

    for hi in range(MAX_HANDS):
        server.scene.add_line_segments(f"/pred/h{hi}/bones", points=dummy_seg, colors=dummy_seg_c, line_width=2.0)
        server.scene.add_mesh_simple(f"/pred/h{hi}/mesh", vertices=_dummy_verts, faces=_dummy_faces,
                                     color=(100, 200, 255), opacity=0.6, side="double", visible=False)
    server.scene.add_point_cloud("/depth_pc", points=dummy_pts, colors=dummy_pts_c, point_size=0.003)
    obb_dummy = np.zeros((12, 2, 3), dtype=np.float32)
    obb_dummy_c = np.zeros((12, 2, 3), dtype=np.uint8)
    for oi in range(MAX_OBJECTS):
        server.scene.add_point_cloud(f"/obj_pc/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.005)
        server.scene.add_line_segments(f"/obj_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=3.0)
        server.scene.add_point_cloud(f"/traj/obj/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
    for oi in range(MAX_OBJECTS):
        server.scene.add_point_cloud(f"/sam3_obj_pc/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.005)
    # SAM3D mesh placeholders: one frame per object + a point cloud child
    # The frame's wxyz/position is updated per-video-frame to follow the OBB
    for oi in range(MAX_OBJECTS):
        server.scene.add_frame(f"/sam3d_mesh/{oi}", wxyz=(1, 0, 0, 0),
                                position=(0, 0, 0), show_axes=False)
        server.scene.add_point_cloud(
            f"/sam3d_mesh/{oi}/pts",
            points=dummy_pts, colors=dummy_pts_c, point_size=0.003)
        # State indicator dot — single big point above the mesh bbox.
        # Color encodes mode tag: WRIST=red, MOVING=blue, STATIC=gray.
        server.scene.add_point_cloud(
            f"/state_dot/{oi}",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.array([[180, 180, 180]], dtype=np.uint8),
            point_size=0.025, point_shape="circle", visible=False)
        server.scene.add_line_segments(
            f"/sam3_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=2.0)
        # Obs cloud OBB wireframe (12 edges) + 3 principal-axis lines
        server.scene.add_line_segments(
            f"/obs_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=2.0)
        obs_axes_dummy = np.zeros((3, 2, 3), dtype=np.float32)
        obs_axes_dummy_c = np.zeros((3, 2, 3), dtype=np.uint8)
        server.scene.add_line_segments(
            f"/obs_obb/{oi}/axes", points=obs_axes_dummy,
            colors=obs_axes_dummy_c, line_width=4.0)
    server.scene.add_point_cloud("/traj/right", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
    server.scene.add_point_cloud("/traj/left", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")

    # Per-finger colors for pred skeleton
    _finger_rgb = [np.array([b, g, r], dtype=np.uint8) for (b, g, r) in FINGER_COLORS_BGR]
    pred_seg_c = np.array([[c, c] for c in _finger_rgb for _ in range(4)], dtype=np.uint8)

    server.scene.add_camera_frustum("/camera", fov=1.0, aspect=cam_aspect, scale=FRUSTUM_SCALE, image=first_img_rgb)

    info_folder = server.gui.add_folder("Status", expand_by_default=False)
    with info_folder:
        metrics_md = server.gui.add_markdown("")
        contact_md = server.gui.add_markdown("")
        grasp_md = server.gui.add_markdown("")


    # ── SAM3D mesh loading + OBB registration ───────────────────────
    def _load_gs_ply(path):
        """Parse a SAM3D gaussian splat PLY, returning (xyz_Nx3, rgb_Nx3_uint8).

        The SAM3D `save_ply` output is binary LE with properties
        x/y/z/nx/ny/nz/f_dc_0/f_dc_1/f_dc_2/opacity/scale_0-2/rot_0-3.
        Color = SH DC coefficient -> RGB via 0.28209 * f_dc + 0.5.
        """
        with open(path, 'rb') as f:
            props = []
            n_vertex = 0
            while True:
                line = f.readline().decode('utf-8', errors='replace').strip()
                if line.startswith('element vertex'):
                    n_vertex = int(line.split()[2])
                elif line.startswith('property float'):
                    props.append(line.split()[2])
                elif line == 'end_header':
                    break
            dtype = np.dtype([(p, '<f4') for p in props])
            data = np.fromfile(f, dtype=dtype, count=n_vertex)
        xyz = np.stack([data['x'], data['y'], data['z']], -1).astype(np.float32)
        C = 0.28209479177387814
        if 'f_dc_0' in data.dtype.names:
            dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], -1)
            rgb = np.clip(dc * C + 0.5, 0, 1)
        else:
            rgb = np.full((n_vertex, 3), 0.7)
        rgb_u8 = (rgb * 255).astype(np.uint8)
        return xyz, rgb_u8

    def _quat_wxyz_to_mat(q):
        """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
        w, x, y, z = q
        n = (w*w + x*x + y*y + z*z) ** 0.5
        if n == 0:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w/n, x/n, y/n, z/n
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _mat_to_quat_wxyz(R):
        """3x3 rotation matrix to (w, x, y, z)."""
        t = np.trace(R)
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        if R[1,1] > R[2,2]:
            s = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

    # Mesh→OBB fitting strategy:
    #   1. Apply SAM3D's canonical_rotation: canonical → SAM3D-internal camera.
    #   2. Fix axes: SAM3D uses PyTorch3D convention (+Y up, +Z out of screen),
    #      we use OpenCV (+Y down, +Z into screen).  Multiply by diag(1,-1,-1)
    #      to flip both.
    #   3. Center mesh at origin (drop SAM3D canonical_translation — its
    #      internal scale is different from our MoGe-2 scale).
    #   4. Uniform scale (preserve aspect ratio): max(obb_extent) / max(mesh_extent).
    #      Per-axis scaling was distorting spherical objects into ellipsoids.
    #   5. Store points in a "local-to-OBB-at-init" frame; per-video-frame the
    #      /sam3d_mesh/{oid} frame gets transform (R_obb_i @ R_obb0.T, t_obb_i)
    #      so the mesh follows OBB rotation relative to init.
    sam3d_mesh_local_pts = {}   # oid -> (pts_local (N,3), colors (N,3) uint8)
    sam3d_mesh_init_R = {}      # oid -> R_obb0 for PCA-OBB delta mode, OR None for tracker mode
    sam3d_init_handles = {}     # oid -> viser scene-node handle for the SAM3D init debug cloud

    # SAM3D (PyTorch3D) camera → our OpenCV camera.  SAM3D's pipeline brings
    # the MoGe pointmap into PyTorch3D camera via look_at_view_transform(
    # eye=(0,0,-1), at=(0,0,0), up=(0,-1,0)) which yields R = diag(-1,-1,1)
    # (flips X and Y).  The inverse = diag(-1,-1,1) again is what we apply to
    # SAM3D output points/poses to get OpenCV frame.
    R_P3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])

    def _register_sam3d_meshes():
        info = state.get('sam3_mesh_info') or {}
        if not info:
            return
        pose_track = state.get('pose_track_info') or {}
        frames = state.get('frame_data') or []
        for oid, meta in info.items():
            oid_int = int(oid)
            ply_path = meta.get('ply_path')
            if not ply_path:
                continue
            # Resolve relative ply_path against the cache-dir we were
            # invoked with.  Favorited clips store relative paths like
            # "sam3_meshes/obj_3.ply" which only resolve when cwd ==
            # cache-dir; --load-cache from a different cwd (e.g. an
            # orchestrator spawning from REPO_ROOT) used to silently skip
            # the mesh.  Always prefer the cache-dir-anchored path; fall
            # back to the raw value for legacy absolute paths.
            if not os.path.isabs(ply_path):
                resolved = os.path.join(args.cache_dir, ply_path)
                if os.path.isfile(resolved):
                    ply_path = resolved
            if not os.path.isfile(ply_path):
                print(f"[viser] sam3d mesh not found: {ply_path} "
                      f"(cache-dir: {args.cache_dir})")
                continue
            try:
                pts_canon, rgb = _load_gs_ply(ply_path)
            except Exception as e:
                print(f"[viser] failed to load {ply_path}: {e}")
                continue

            tinfo = pose_track.get(oid_int) or pose_track.get(oid) or {}
            tracker_ok = tinfo.get('tracking_status') == 'ok'

            # Always register the SAM3D-reported init pose under /sam3d_init/{oid}
            # for visual debugging.  This is a static (frame-independent) point
            # cloud at the pose SAM3D returned for the init_frame:
            #   pts_in_opencv = (canonical_scale * canonical_rot @ pts + canonical_t)
            #                   then flip P3D->OpenCV via diag(1,-1,-1).
            try:
                q_canon_dbg = np.asarray(meta.get('canonical_rotation_quat')
                                         or [1, 0, 0, 0], dtype=np.float64)
                t_canon_dbg = np.asarray(meta.get('canonical_translation')
                                         or [0, 0, 0], dtype=np.float64)
                s_canon_dbg = float(meta.get('canonical_scale') or 1.0)
                # NOTE: SAM3D composes as Transform3d.scale(s).rotate(R).translate(t),
                # and PyTorch3D's Transform3d.rotate(R) applies row-form `p @ R`
                # (NOT `p @ R.T` like the column-form convention).  So we use the
                # same: pts @ R, NOT pts @ R.T.
                R_canon_dbg = _quat_wxyz_to_mat(q_canon_dbg)
                pts_init = (pts_canon.astype(np.float64) * s_canon_dbg) @ R_canon_dbg \
                           + t_canon_dbg
                pts_init = (pts_init @ R_P3D_TO_OPENCV.T).astype(np.float32)
                sam3d_init_handles[oid_int] = server.scene.add_point_cloud(
                    f"/sam3d_init/{oid_int}",
                    points=pts_init, colors=rgb,
                    point_size=0.003, visible=False)
            except Exception as _e:
                print(f"[viser] obj {oid_int}: failed to register SAM3D init pose: {_e}")

            if tracker_ok:
                # Tracker path: per-frame pose_R / pose_t IS mesh→camera.
                # Store mesh in raw canonical frame (with any scale correction
                # applied during tracking), flag init_R = None to signal that
                # update_frame should use pose directly (no delta).
                scale_corr = float(tinfo.get('scale_correction', 1.0))
                pts_local = (pts_canon.astype(np.float32) * scale_corr)
                sam3d_mesh_local_pts[oid_int] = (pts_local, rgb)
                sam3d_mesh_init_R[oid_int] = None
                server.scene.add_point_cloud(
                    f"/sam3d_mesh/{oid_int}/pts",
                    points=pts_local, colors=rgb, point_size=0.003)
                print(f"[viser] obj {oid_int}: tracker mode, "
                      f"{len(pts_canon)} pts, scale_corr={scale_corr:.3f}")
                continue

            # Fallback: PCA-OBB delta mode (legacy).
            init_frame_idx = int(meta.get('init_frame', 0))
            sod = None
            if 0 <= init_frame_idx < len(frames):
                sod = frames[init_frame_idx].get('sam3_obj_data', {}).get(oid)
            if sod is None or sod.get('pose_R') is None or sod.get('obb_extent') is None:
                print(f"[viser] obj {oid_int}: no OBB at init frame {init_frame_idx}, skip mesh")
                continue
            R_obb0 = np.asarray(sod['pose_R'], dtype=np.float64)
            obb_extent = np.asarray(sod['obb_extent'], dtype=np.float64)

            q_canon = np.asarray(meta.get('canonical_rotation_quat') or [1, 0, 0, 0],
                                 dtype=np.float64)
            R_canon = _quat_wxyz_to_mat(q_canon)
            # See note above: PyTorch3D Transform3d.rotate(R) applies `p @ R`.
            pts_sam3d_cam = pts_canon.astype(np.float64) @ R_canon
            pts_cam = pts_sam3d_cam @ R_P3D_TO_OPENCV.T
            pts_cam -= pts_cam.mean(axis=0)
            mesh_lo, mesh_hi = pts_cam.min(axis=0), pts_cam.max(axis=0)
            mesh_size = float(np.max(mesh_hi - mesh_lo))
            obb_size = float(np.max(obb_extent))
            if mesh_size > 1e-6:
                pts_cam *= (obb_size / mesh_size)
            sam3d_mesh_local_pts[oid_int] = (pts_cam.astype(np.float32), rgb)
            sam3d_mesh_init_R[oid_int] = R_obb0
            server.scene.add_point_cloud(
                f"/sam3d_mesh/{oid_int}/pts",
                points=pts_cam.astype(np.float32), colors=rgb, point_size=0.003)
            print(f"[viser] obj {oid_int}: PCA-OBB delta mode (tracker unavailable)")

    # ── World origin axes ────────────────────────────────────────────
    def setup_world_origin():
        """Apply gravity up direction + draw coordinate axes and ground grid.

        Origin = median wrist position shifted 0.75m downward along gravity.
        Axes aligned to gravity-defined coordinate frame.
        """
        from scipy.spatial.transform import Rotation

        gravity_up = state.get('gravity_up')
        if gravity_up is None:
            gravity_up = np.array([0.0, -1.0, 0.0])
        gravity_up = gravity_up / (np.linalg.norm(gravity_up) + 1e-8)

        # Set Viser scene up direction from gravity
        server.scene.set_up_direction((float(gravity_up[0]), float(gravity_up[1]), float(gravity_up[2])))

        fd_list = state.get('frame_data', [])
        if not fd_list:
            return

        # Collect all wrist positions across all frames
        all_wrists = []
        for fd in fd_list:
            for j3d in fd.get('joints_3d_pred', []):
                if j3d is not None and len(j3d) > 0:
                    all_wrists.append(j3d[0])
        if not all_wrists:
            return

        hand_center = np.median(np.array(all_wrists), axis=0)

        # Origin = 0.75m below hand center (along gravity direction)
        origin = hand_center - gravity_up * 0.75

        # Build right-handed coordinate frame: Y=up, X=right, Z=forward
        up = gravity_up
        ref = np.array([0.0, 0.0, 1.0]) if abs(np.dot(up, [0, 0, 1])) < 0.9 else np.array([1.0, 0.0, 0.0])
        right = np.cross(up, ref)
        right /= np.linalg.norm(right)
        forward = np.cross(right, up)

        # Rotation matrix: columns = [X=right, Y=up, Z=forward]
        R = np.column_stack([right, up, forward])
        xyzw = Rotation.from_matrix(R).as_quat()
        wxyz = (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))

        origin_pos = (float(origin[0]), float(origin[1]), float(origin[2]))

        server.scene.add_frame("/world_origin",
            position=origin_pos,
            wxyz=wxyz,
            axes_length=0.3,
            axes_radius=0.008,
            show_axes=True)

        server.scene.add_grid("/ground_grid",
            position=origin_pos,
            wxyz=wxyz,
            width=1.5, height=1.5,
            cell_size=0.1, cell_color=(80, 80, 80),
            section_size=0.5, section_color=(120, 120, 120),
            plane="xz")

        # Initial client view: position = -gravity_up * 0.4, look_at = (0,0,0.2).
        cam_pos = tuple((-gravity_up * 0.4).tolist())
        look_at = (0.0, 0.0, 0.2)
        up_dir = (float(gravity_up[0]), float(gravity_up[1]), float(gravity_up[2]))

        @server.on_client_connect
        def _init_client_view(client):
            try:
                client.camera.position = cam_pos
                client.camera.look_at = look_at
                client.camera.up_direction = up_dir
            except Exception as e:
                print(f"[viser] init camera failed: {e}")

    # ── update_frame ───────────────────────────────────────────────
    def update_frame(idx):
        fd_list = state['frame_data']
        if idx < 0 or idx >= len(fd_list):
            return
        fd = fd_list[idx]
        dp_focal = state['dp_focal']

        # Camera frustum image: priority depth > flow blend > rgb.
        # Lazy-decode JPG bytes (when present) into RGB ndarray.
        img = _decode_image_maybe(fd.get('img_rgb'))
        if show_depth_overlay.value:
            d = _decode_image_maybe(fd.get('depth_rgb'))
            if d is not None:
                img = d
        elif show_flow_overlay.value:
            flow_rgb = _decode_image_maybe(fd.get('flow_rgb'))
            if flow_rgb is not None:
                alpha = float(flow_alpha.value)
                if img is not None and img.shape == flow_rgb.shape:
                    img = (img.astype(np.float32) * (1 - alpha)
                            + flow_rgb.astype(np.float32) * alpha
                           ).clip(0, 255).astype(np.uint8)
                else:
                    img = flow_rgb
        fov = float(2 * np.arctan(H / 2 / dp_focal))
        server.scene.add_camera_frustum("/camera", fov=fov, aspect=cam_aspect, scale=FRUSTUM_SCALE, image=img)

        # Pred hand skeletons
        n_hands = len(fd['joints_3d_pred'])
        n_skel = n_hands if show_pred.value else 0
        for hi in range(MAX_HANDS):
            if hi < n_skel:
                segs = make_line_segments_3d(fd['joints_3d_pred'][hi])
                server.scene.add_line_segments(f"/pred/h{hi}/bones", points=segs, colors=pred_seg_c, line_width=3.0)
            else:
                server.scene.add_line_segments(f"/pred/h{hi}/bones", points=dummy_seg, colors=dummy_seg_c, line_width=3.0)

        # Hand meshes
        mano_f = state.get('mano_faces')
        n_mesh = n_hands if (show_mesh.value and mano_f is not None) else 0
        for hi in range(MAX_HANDS):
            if hi < n_mesh and fd['vertices_3d'][hi] is not None:
                is_right = fd['hand_is_right'][hi]
                color = MESH_COLORS[is_right]
                server.scene.add_mesh_simple(
                    f"/pred/h{hi}/mesh", vertices=fd['vertices_3d'][hi].astype(np.float32),
                    faces=mano_f, color=color, opacity=0.6, side="double", flat_shading=False, visible=True)
            else:
                server.scene.add_mesh_simple(
                    f"/pred/h{hi}/mesh", vertices=_dummy_verts, faces=_dummy_faces,
                    color=(100, 200, 255), opacity=0.6, side="double", visible=False)

        # Object OBB + point cloud (multi-object)
        obj_data = fd.get('obj_data', {})  # {oid: {pts, corners}}
        for oi in range(MAX_OBJECTS):
            od = obj_data.get(oi)
            color = OBJ_COLORS_RGB[oi % len(OBJ_COLORS_RGB)]
            if show_object.value and od and od.get('corners') is not None:
                corners = od['corners']
                segs = np.array([[corners[e[0]], corners[e[1]]] for e in OBB_EDGES], dtype=np.float32)
                seg_colors = np.tile(color, (12, 2, 1))
                server.scene.add_line_segments(f"/obj_obb/{oi}", points=segs, colors=seg_colors, line_width=3.0)
            else:
                server.scene.add_line_segments(f"/obj_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=3.0)
            if show_obj_pc.value and od and od.get('pts') is not None and len(od['pts']) > 0:
                obj_colors = np.tile(color, (len(od['pts']), 1))
                server.scene.add_point_cloud(f"/obj_pc/{oi}", points=od['pts'], colors=obj_colors, point_size=0.005)
            else:
                server.scene.add_point_cloud(f"/obj_pc/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.005)

        # SAM3-detected object point clouds (text-prompt method)
        sam3_obj_data = fd.get('sam3_obj_data', {})
        for oi in range(MAX_OBJECTS):
            sod = sam3_obj_data.get(oi)
            scolor = SAM3_OBJ_COLORS_RGB[oi % len(SAM3_OBJ_COLORS_RGB)]
            if show_sam3_obj_pc.value and sod and sod.get('pts') is not None and len(sod['pts']) > 0:
                sc = np.tile(scolor, (len(sod['pts']), 1))
                server.scene.add_point_cloud(f"/sam3_obj_pc/{oi}", points=sod['pts'], colors=sc, point_size=0.005)
            else:
                server.scene.add_point_cloud(f"/sam3_obj_pc/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.005)

        # SAM3D init-pose debug visibility (static cloud, just toggle)
        for h in sam3d_init_handles.values():
            try:
                h.visible = bool(show_sam3d_init.value)
            except Exception:
                pass

        # SAM3D mesh 6DoF update + SAM3 OBB wireframe
        _pti_all = state.get('pose_track_info') or {}
        for oi in range(MAX_OBJECTS):
            sod = sam3_obj_data.get(oi)
            # Per-frame OBB (attached mesh coords in local OBB frame)
            if sod is not None and sod.get('pose_R') is not None:
                # Prefer pose_track_info[oid].T_seq[idx] when present —
                # this is the refined pose written by
                # egoinfinity/pipeline/post_tracking/pose_tracking.py and is what the static
                # recording.viser / HF Space playback uses.  Falls back
                # to the original SAM3 OBB pose for legacy clips.
                R_i = None
                t_i = None
                _tinfo = _pti_all.get(oi) or _pti_all.get(str(oi)) or {}
                _T_seq = _tinfo.get('T_seq')
                if _T_seq is not None:
                    try:
                        _T_arr = np.asarray(_T_seq)
                        if (_T_arr.ndim == 3 and _T_arr.shape[1:] == (4, 4)
                                and 0 <= idx < len(_T_arr)):
                            _T_t = _T_arr[idx]
                            if np.all(np.isfinite(_T_t)):
                                R_i = _T_t[:3, :3].astype(np.float64)
                                t_i = _T_t[:3, 3].astype(np.float64)
                    except Exception:
                        R_i = None
                if R_i is None:
                    R_i = np.asarray(sod['pose_R'])
                    t_i = np.asarray(sod['pose_t'])
                # Two rendering modes:
                # (a) Tracker mode (init_R == None): pose_R/t is mesh→camera,
                #     mesh pts are stored in raw canonical frame.  Drive the
                #     scene frame with (R_i, t_i) directly.
                # (b) PCA-OBB delta mode (init_R is a rotation matrix): mesh
                #     pts are pre-transformed into the init-frame OBB body
                #     coords.  Drive the scene frame with (R_i @ R_init.T, t_i)
                #     so the mesh rotates relative to init frame only.
                R_init = sam3d_mesh_init_R.get(oi)
                if R_init is None:
                    R_frame = R_i
                else:
                    R_frame = R_i @ R_init.T
                q_wxyz = _mat_to_quat_wxyz(R_frame)
                # Frame transform drives any /sam3d_mesh/{oi}/* children
                server.scene.add_frame(
                    f"/sam3d_mesh/{oi}", wxyz=tuple(q_wxyz.tolist()),
                    position=tuple(t_i.tolist()), show_axes=False,
                    visible=show_sam3d_mesh.value and (oi in sam3d_mesh_local_pts))
                # OBB wireframe
                if show_sam3_obb.value and sod.get('obb_corners') is not None:
                    corners = np.asarray(sod['obb_corners'])
                    segs = np.array([[corners[e[0]], corners[e[1]]]
                                     for e in OBB_EDGES], dtype=np.float32)
                    seg_c = np.tile(SAM3_OBJ_COLORS_RGB[oi % len(SAM3_OBJ_COLORS_RGB)], (12, 2, 1))
                    server.scene.add_line_segments(
                        f"/sam3_obb/{oi}", points=segs, colors=seg_c, line_width=2.0)
                else:
                    server.scene.add_line_segments(
                        f"/sam3_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=2.0)
            else:
                # No OBB this frame -> hide mesh, hide obb
                server.scene.add_frame(f"/sam3d_mesh/{oi}", wxyz=(1, 0, 0, 0),
                                       position=(0, 0, 0), show_axes=False,
                                       visible=False)
                server.scene.add_line_segments(
                    f"/sam3_obb/{oi}", points=obb_dummy, colors=obb_dummy_c, line_width=2.0)

            # ── State indicator dot (above mesh bbox along world up) ──
            if (show_state_dot.value
                    and sod is not None and sod.get('pose_R') is not None
                    and oi in sam3d_mesh_local_pts):
                # Reuse the refined T_seq pose (if any) just like the
                # mesh frame block above, so the dot stays glued on top
                # of the mesh regardless of pose_track_info refinements.
                R_i_dot = None
                t_i_dot = None
                _tinfo_dot = _pti_all.get(oi) or _pti_all.get(str(oi)) or {}
                _T_seq_dot = _tinfo_dot.get('T_seq')
                if _T_seq_dot is not None:
                    try:
                        _T_arr2 = np.asarray(_T_seq_dot)
                        if (_T_arr2.ndim == 3 and _T_arr2.shape[1:] == (4, 4)
                                and 0 <= idx < len(_T_arr2)):
                            _T_t2 = _T_arr2[idx]
                            if np.all(np.isfinite(_T_t2)):
                                R_i_dot = _T_t2[:3, :3].astype(np.float64)
                                t_i_dot = _T_t2[:3, 3].astype(np.float64)
                    except Exception:
                        R_i_dot = None
                if R_i_dot is None:
                    R_i_dot = np.asarray(sod['pose_R'], dtype=np.float64)
                    t_i_dot = np.asarray(sod['pose_t'], dtype=np.float64)
                R_i = R_i_dot
                t_i = t_i_dot
                pts_local, _ = sam3d_mesh_local_pts[oi]
                # mesh local center + half-extent along world up
                mesh_center_local = np.asarray(pts_local, dtype=np.float64).mean(axis=0)
                lo = np.asarray(pts_local, dtype=np.float64).min(axis=0)
                hi = np.asarray(pts_local, dtype=np.float64).max(axis=0)
                half_extent = float(np.max(hi - lo)) * 0.5
                # World position: rotate-translate mesh center, then offset along world up
                R_init_oi = sam3d_mesh_init_R.get(oi)
                R_w = R_i if R_init_oi is None else (R_i @ R_init_oi.T)
                mesh_center_world = R_w @ mesh_center_local + t_i
                g = state.get('gravity_up')
                if g is None:
                    g = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                else:
                    g = np.asarray(g, dtype=np.float64)
                    g = g / (np.linalg.norm(g) + 1e-8)
                dot_pos = mesh_center_world + g * (half_extent + 0.06)
                # Color from per-frame state flags
                pti = (state.get('pose_track_info') or {}).get(oi, {}) or {}
                wrist_pf = pti.get('wrist_used_per_frame') or []
                moving_pf = pti.get('is_moving_per_frame') or []
                if wrist_pf and 0 <= idx < len(wrist_pf) and bool(wrist_pf[idx]):
                    rgb = (230, 90, 60)        # WRIST — red/orange
                elif moving_pf and 0 <= idx < len(moving_pf) and bool(moving_pf[idx]):
                    rgb = (60, 130, 230)       # DEPTH_TRACKED — blue
                else:
                    rgb = (180, 180, 180)      # STATIC — gray
                server.scene.add_point_cloud(
                    f"/state_dot/{oi}",
                    points=np.asarray(dot_pos, dtype=np.float32).reshape(1, 3),
                    colors=np.asarray(rgb, dtype=np.uint8).reshape(1, 3),
                    point_size=0.025, point_shape="circle", visible=True)
            else:
                server.scene.add_point_cloud(
                    f"/state_dot/{oi}",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.array([[0, 0, 0]], dtype=np.uint8),
                    point_size=0.025, point_shape="circle", visible=False)

            # Obs cloud OBB (PCA) — independent of mesh OBB above.  Yellow when
            # trustworthy, dim grey when not (still rendered for debugging).
            obs_obb_seqs = (state.get('pose_track_info') or {}).get(oi, {}) \
                            .get('obs_obb_per_frame') or []
            obb_t = obs_obb_seqs[idx] if 0 <= idx < len(obs_obb_seqs) else None
            if show_obs_obb.value and obb_t is not None:
                R_obb = np.asarray(obb_t['R'], dtype=np.float64)
                center = np.asarray(obb_t['center'], dtype=np.float64)
                extents = np.asarray(obb_t['extents'], dtype=np.float64)
                trust = bool(obb_t.get('trustworthy'))
                # 8 corners
                signs = np.array([
                    [-1,-1,-1],[ 1,-1,-1],[-1, 1,-1],[ 1, 1,-1],
                    [-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],[ 1, 1, 1]],
                    dtype=np.float64)
                corners = (signs * extents * 0.5) @ R_obb.T + center
                segs = np.array([[corners[e[0]], corners[e[1]]]
                                 for e in OBB_EDGES], dtype=np.float32)
                if trust:
                    base_c = np.array([255, 230, 60], dtype=np.uint8)   # yellow
                else:
                    base_c = np.array([100, 100, 100], dtype=np.uint8)  # grey
                seg_c = np.tile(base_c, (12, 2, 1))
                server.scene.add_line_segments(
                    f"/obs_obb/{oi}", points=segs, colors=seg_c, line_width=2.0)
                # 3 principal-axis lines (R/G/B = 1st/2nd/3rd component)
                axis_colors = np.array([
                    [255, 60,  60],   # PC1 — red
                    [ 60,255,  60],   # PC2 — green
                    [ 60, 60, 255],   # PC3 — blue
                ], dtype=np.uint8)
                half = extents * 0.5
                axes = np.zeros((3, 2, 3), dtype=np.float32)
                axes_c = np.zeros((3, 2, 3), dtype=np.uint8)
                for ai in range(3):
                    axes[ai, 0] = center
                    axes[ai, 1] = center + R_obb[:, ai] * half[ai]
                    axes_c[ai] = axis_colors[ai]
                server.scene.add_line_segments(
                    f"/obs_obb/{oi}/axes", points=axes,
                    colors=axes_c, line_width=4.0)
            else:
                server.scene.add_line_segments(
                    f"/obs_obb/{oi}", points=obb_dummy,
                    colors=obb_dummy_c, line_width=2.0)
                server.scene.add_line_segments(
                    f"/obs_obb/{oi}/axes",
                    points=np.zeros((3, 2, 3), dtype=np.float32),
                    colors=np.zeros((3, 2, 3), dtype=np.uint8),
                    line_width=4.0)


        # Trajectories (wrist + object centroid trails)
        def _densify(pts_list, n_interp=4):
            """Cubic-spline interpolation for smooth, dense trail."""
            if len(pts_list) < 2:
                return np.array(pts_list, dtype=np.float32) if pts_list else np.zeros((0, 3), dtype=np.float32)
            from scipy.interpolate import CubicSpline
            pts = np.array(pts_list, dtype=np.float64)
            t_knots = np.arange(len(pts))
            t_dense = np.linspace(0, len(pts) - 1, max((len(pts) - 1) * n_interp + 1, 2))
            cs = CubicSpline(t_knots, pts)
            return cs(t_dense).astype(np.float32)

        def _trail_colors(n, base_rgb):
            """Color gradient: older points dimmer, newest bright."""
            if n == 0:
                return np.zeros((0, 3), dtype=np.uint8)
            alpha = np.linspace(0.3, 1.0, n)[:, None]
            return (alpha * base_rgb).astype(np.uint8)

        if show_trajectories.value:
            trail = int(traj_length.value)
            start_i = max(0, idx - trail)
            right_pts, left_pts = [], []
            right_src, left_src = [], []  # source labels for debug coloring
            obj_traj_map = {oi: [] for oi in range(MAX_OBJECTS)}
            for ti in range(start_i, idx + 1):
                if ti >= len(fd_list):
                    break
                tfd = fd_list[ti]
                meta_list = tfd.get('hand_meta', [])
                for hi, j3d in enumerate(tfd['joints_3d_pred']):
                    src = meta_list[hi]['source'] if hi < len(meta_list) else 'detected'
                    if tfd['hand_is_right'][hi]:
                        right_pts.append(j3d[0])
                        right_src.append(src)
                    else:
                        left_pts.append(j3d[0])
                        left_src.append(src)
                for oi, od in tfd.get('obj_data', {}).items():
                    if od and od.get('corners') is not None:
                        obj_traj_map.setdefault(oi, []).append(od['corners'].mean(axis=0))

            # Debug colors: green=detected, yellow=infilled
            _SRC_COLORS = {'detected': np.array([0, 220, 0]),
                           'infilled': np.array([255, 200, 0]),
                           'unknown': np.array([180, 180, 180])}
            use_debug = show_debug_colors.value

            if right_pts:
                if use_debug:
                    rp = np.array(right_pts, dtype=np.float32)
                    rc = np.array([_SRC_COLORS.get(s, [100,200,255]) for s in right_src], dtype=np.uint8)
                else:
                    rp = _densify(right_pts)
                    rc = _trail_colors(len(rp), np.array([100, 200, 255]))
                server.scene.add_point_cloud("/traj/right", points=rp,
                    colors=rc, point_size=0.010 if use_debug else 0.008, point_shape="circle")
            else:
                server.scene.add_point_cloud("/traj/right", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
            if left_pts:
                if use_debug:
                    lp = np.array(left_pts, dtype=np.float32)
                    lc = np.array([_SRC_COLORS.get(s, [255,180,100]) for s in left_src], dtype=np.uint8)
                else:
                    lp = _densify(left_pts)
                    lc = _trail_colors(len(lp), np.array([255, 180, 100]))
                server.scene.add_point_cloud("/traj/left", points=lp,
                    colors=lc, point_size=0.010 if use_debug else 0.008, point_shape="circle")
            else:
                server.scene.add_point_cloud("/traj/left", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
            for oi in range(MAX_OBJECTS):
                pts = obj_traj_map.get(oi, [])
                color = OBJ_COLORS_RGB[oi % len(OBJ_COLORS_RGB)]
                if pts:
                    op = _densify(pts)
                    server.scene.add_point_cloud(f"/traj/obj/{oi}", points=op,
                        colors=_trail_colors(len(op), color), point_size=0.008, point_shape="circle")
                else:
                    server.scene.add_point_cloud(f"/traj/obj/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
        else:
            server.scene.add_point_cloud("/traj/right", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
            server.scene.add_point_cloud("/traj/left", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")
            for oi in range(MAX_OBJECTS):
                server.scene.add_point_cloud(f"/traj/obj/{oi}", points=dummy_pts, colors=dummy_pts_c, point_size=0.008, point_shape="circle")

        # Background point cloud
        if show_depth_pc.value:
            bg_tpl = state.get('bg_template')
            bg_rgb_img = state.get('bg_rgb')
            if bg_tpl is not None:
                step = max(2, int(pc_step.value))
                _cx, _cy = state.get('cx', W / 2.0), state.get('cy', H / 2.0)
                pts, cols = depth_to_pointcloud(bg_tpl, dp_focal, _cx, _cy, step=step, img_rgb=bg_rgb_img)
                if len(pts) > 0:
                    server.scene.add_point_cloud("/depth_pc", points=pts, colors=cols, point_size=0.003)
                else:
                    server.scene.add_point_cloud("/depth_pc", points=dummy_pts, colors=dummy_pts_c, point_size=0.003)
            else:
                server.scene.add_point_cloud("/depth_pc", points=dummy_pts, colors=dummy_pts_c, point_size=0.003)
        else:
            server.scene.add_point_cloud("/depth_pc", points=dummy_pts, colors=dummy_pts_c, point_size=0.003)

        # Build status string with hand source info
        _status_parts = [f"Frame {idx} / {len(fd_list)}"]
        for hm in fd.get('hand_meta', []):
            side = "R" if fd['hand_is_right'][fd.get('hand_meta', []).index(hm)] else "L"
            _status_parts.append(f"{side}:{hm['source']}(c={hm['confidence']:.2f})")
        metrics_md.content = " | ".join(_status_parts)

        # Per-frame contact + grasp display (per-object × [left, right])
        pose_track = state.get('pose_track_info') or {}
        if pose_track:
            sam3_mapping = state.get('sam3_prompt_mapping') or []

            def _g(v):
                return "⬤" if v > 0.5 else ("◐" if v > 0.2 else "·")

            def _prompt_for(oid):
                try:
                    return sam3_mapping[int(oid)].get('prompt', '?')[:18]
                except Exception:
                    return "?"

            contact_lines = ["**Contact (L/R)** _raw mask overlap_"]
            grasp_lines = ["**Grasp (L/R)** _motion-correlated_"]
            for oid, ti in sorted(pose_track.items()):
                p = _prompt_for(oid)
                cs = ti.get('contact_soft')
                if cs is not None and idx < len(cs):
                    cl, cr = cs[idx][0], cs[idx][1]
                    contact_lines.append(
                        f"obj{oid} {p:<18s}: L {_g(cl)} {cl:.2f}  R {_g(cr)} {cr:.2f}")
                gs = ti.get('grasp_soft')
                if gs is not None and idx < len(gs):
                    gl, gr = gs[idx][0], gs[idx][1]
                    # Per-frame mode tag: WRIST > STATIC > MICRO > TRACKED
                    wrist_pf = ti.get('wrist_used_per_frame') or []
                    moving_pf = ti.get('is_moving_per_frame') or []
                    fast_pf = ti.get('is_fast_per_frame') or []
                    if idx < len(wrist_pf) and wrist_pf[idx]:
                        mode_tag = ' 🤝[W]'           # WRIST this frame
                    elif idx < len(moving_pf) and not moving_pf[idx]:
                        mode_tag = ' [s]'              # DEPTH_STATIC (locked)
                    elif idx < len(fast_pf) and not fast_pf[idx]:
                        mode_tag = ' [m]'              # DEPTH_MICRO (per-frame, slow)
                    else:
                        mode_tag = ' [d]'              # DEPTH_TRACKED (per-frame, fast)
                    grasp_lines.append(
                        f"obj{oid} {p:<18s}: L {_g(gl)} {gl:.2f}  R {_g(gr)} {gr:.2f}{mode_tag}")
            contact_md.content = "  \n".join(contact_lines)
            grasp_md.content = "  \n".join(grasp_lines)
        else:
            contact_md.content = ""
            grasp_md.content = ""

    # ── Processing (runs immediately) ──────────────────────────────
    def run_processing():
        state['processing'] = True
        b_colors = bone_colors_bgr()

        # ── Phase A: MoGe-2 Depth ────────────────────────────────
        progress("Phase A: Loading MoGe-2...")
        # Backend selected via registry; default 'moge2' -> MoGe2Estimator
        # (identical to the pre-modularization direct instantiation).
        # Override with EGOINFINITY_DEPTH_BACKEND=<name>.
        from egoinfinity.pipeline.backends import get_backend
        depth_estimator = get_backend('depth')(device='cuda')

        depth_maps = []
        est_focals = []
        frames_rgb = []
        t0 = time.time()
        for i in range(total):
            img_bgr = cv2.imread(image_files[i])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(img_rgb)
            result = depth_estimator.estimate(img_bgr)
            depth_maps.append(result.depth)
            est_focals.append(result.focal_length_px)
            fps = (i + 1) / (time.time() - t0)
            progress(f"Phase A: Depth [{i+1}/{total}] {fps:.1f}fps")

        del depth_estimator
        free_gpu()

        dp_focal = float(np.median(est_focals))
        state['dp_focal'] = dp_focal
        progress(f"Phase A done | focal={dp_focal:.0f}")

        # ── Phase A-grav: Gravity estimation (GeoCalib) ──────────
        progress("Phase A-grav: Loading GeoCalib...")
        from egoinfinity.pipeline.backends import get_backend
        from egoinfinity.pipeline.config import GEOCALIB_SAMPLE_FRAMES

        grav_estimator = get_backend('gravity')(device='cuda')

        # Sample a few frames (static camera → gravity is constant)
        n_samples = min(GEOCALIB_SAMPLE_FRAMES, total)
        if n_samples == 1:
            sample_idxs = [0]
        elif n_samples == 2:
            sample_idxs = [0, total - 1]
        else:
            sample_idxs = [0, total // 2, total - 1]

        # Reuse frames_rgb decoded in Phase A; avoid disk re-read.
        sample_imgs = [cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR) for idx in sample_idxs]
        grav_result = grav_estimator.estimate_robust(sample_imgs)

        progress(f"Phase A-grav done | roll={grav_result.roll_deg:.1f}° pitch={grav_result.pitch_deg:.1f}° "
                 f"up=[{grav_result.up_direction[0]:.3f}, {grav_result.up_direction[1]:.3f}, {grav_result.up_direction[2]:.3f}]")
        progress(f"Phase A-grav focal comparison | MoGe-2={dp_focal:.0f}px  GeoCalib={grav_result.focal_length_px:.0f}px")

        state['gravity_up'] = grav_result.up_direction   # (3,) unit vec, up in camera frame
        state['gravity_roll_deg'] = grav_result.roll_deg
        state['gravity_pitch_deg'] = grav_result.pitch_deg

        del grav_estimator
        free_gpu()

        # ── Phase B: WiLoR (with metric focal from MoGe-2) ───────
        progress("Phase B: Loading hand model...")
        from egoinfinity.pipeline.backends import get_backend

        detector = get_backend('hand_detect')()
        reconstructor = get_backend('hand_recon')()

        # MANO faces (fixed topology, extracted once)
        mano_faces = reconstructor.model.mano.faces.astype(np.int32).copy()
        state['mano_faces'] = mano_faces

        hand_results_per_frame = []
        t0 = time.time()
        for i in range(total):
            # Reuse frames_rgb decoded in Phase A; avoid disk re-read.
            img_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
            dets = detector.detect(img_bgr)
            hands = reconstructor.reconstruct(img_bgr, dets, focal_length=dp_focal)
            hand_results_per_frame.append(hands)

            fps = (i + 1) / (time.time() - t0)
            progress(f"Phase B: Hand [{i+1}/{total}] {fps:.1f}fps | {len(hands)} hands")

        # Keep MANO model for infiller, release the rest
        mano_model = reconstructor.model.mano
        mano_model.cpu().float()  # WiLoR runs in half; MANO forward needs float32
        del detector, reconstructor
        free_gpu()

        # Fix handedness: (1) per-track majority vote, (2) remove duplicate detections
        from collections import Counter, defaultdict

        # Step 1: Per-track majority vote
        track_hand_votes = defaultdict(list)
        for frame_hands in hand_results_per_frame:
            for h in frame_hands:
                track_hand_votes[h.track_id].append(h.is_right)
        n_fixed = 0
        for tid, votes in track_hand_votes.items():
            majority = Counter(votes).most_common(1)[0][0]
            for frame_hands in hand_results_per_frame:
                for h in frame_hands:
                    if h.track_id == tid and h.is_right != majority:
                        h.is_right = majority
                        h.joints_3d[:, 0] *= -1
                        h.joints_3d_rel[:, 0] *= -1
                        h.vertices[:, 0] *= -1
                        h.cam_t[0] *= -1
                        n_fixed += 1

        # Step 2: Remove duplicate hands (overlapping bbox, same handedness after vote)
        def _bbox_iou(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            return inter / max(area_a + area_b - inter, 1e-6)

        n_removed = 0
        for i in range(total):
            hands = hand_results_per_frame[i]
            if len(hands) <= 1:
                continue
            keep = []
            removed = set()
            for ai in range(len(hands)):
                if ai in removed:
                    continue
                best = ai
                for bi in range(ai + 1, len(hands)):
                    if bi in removed:
                        continue
                    if _bbox_iou(hands[ai].bbox, hands[bi].bbox) > 0.3:
                        # Overlapping: keep higher confidence
                        if hands[bi].confidence > hands[best].confidence:
                            removed.add(best)
                            best = bi
                        else:
                            removed.add(bi)
                keep.append(best)
            if removed:
                hand_results_per_frame[i] = [hands[k] for k in sorted(set(keep))]
                n_removed += len(removed)

        if n_fixed + n_removed > 0:
            progress(f"Phase B: Fixed {n_fixed} handedness, removed {n_removed} duplicates")

        # Step 3: Track-level filtering — remove short/outlier tracks (misdetections)
        # Collect per-track stats
        track_stats = defaultdict(lambda: {'frames': [], 'wrists': [], 'bbox_areas': [],
                                            'depths': [], 'is_right': None})
        for i, frame_hands in enumerate(hand_results_per_frame):
            for h in frame_hands:
                ts = track_stats[h.track_id]
                ts['frames'].append(i)
                ts['wrists'].append(h.cam_t.copy())
                ts['depths'].append(h.cam_t[2])
                bw, bh = h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1]
                ts['bbox_areas'].append(bw * bh)
                ts['is_right'] = h.is_right

        # Find the dominant track per handedness (longest track = real hand)
        dominant = {}  # {True: tid, False: tid}
        for side in [True, False]:
            side_tracks = [(tid, ts) for tid, ts in track_stats.items()
                           if ts['is_right'] == side]
            if side_tracks:
                dominant[side] = max(side_tracks, key=lambda x: len(x[1]['frames']))[0]

        # Identify tracks to remove
        tracks_to_remove = set()
        for tid, ts in track_stats.items():
            side = ts['is_right']
            n_frames = len(ts['frames'])

            # 3a: Short tracks (< 5 frames) — likely misdetection
            if n_frames < 5:
                tracks_to_remove.add(tid)
                continue

            # 3b: Position outlier relative to dominant track
            if side in dominant and dominant[side] != tid:
                dom_ts = track_stats[dominant[side]]
                dom_wrist = np.median(dom_ts['wrists'], axis=0)
                my_wrist = np.median(ts['wrists'], axis=0)
                dist = np.linalg.norm(my_wrist - dom_wrist)
                if dist > 0.5 and n_frames < len(dom_ts['frames']) * 0.3:
                    tracks_to_remove.add(tid)
                    continue

            # 3c: Hand size outlier (bbox_area / depth^2 should be consistent)
            if side in dominant and dominant[side] != tid:
                dom_ts = track_stats[dominant[side]]
                dom_depths = np.array(dom_ts['depths'])
                dom_areas = np.array(dom_ts['bbox_areas'])
                dom_valid = dom_depths > 0.1
                my_depths = np.array(ts['depths'])
                my_areas = np.array(ts['bbox_areas'])
                my_valid = my_depths > 0.1
                if dom_valid.sum() > 0 and my_valid.sum() > 0:
                    dom_size = np.median(dom_areas[dom_valid] / dom_depths[dom_valid] ** 2)
                    my_size = np.median(my_areas[my_valid] / my_depths[my_valid] ** 2)
                    ratio = my_size / max(dom_size, 1e-6)
                    if (ratio > 3.0 or ratio < 0.33) and n_frames < len(dom_ts['frames']) * 0.3:
                        tracks_to_remove.add(tid)

        # Remove bad tracks from hand_results_per_frame
        n_track_removed = 0
        if tracks_to_remove:
            for i in range(total):
                before = len(hand_results_per_frame[i])
                hand_results_per_frame[i] = [
                    h for h in hand_results_per_frame[i]
                    if h.track_id not in tracks_to_remove]
                n_track_removed += before - len(hand_results_per_frame[i])

        if n_track_removed > 0:
            progress(f"Phase B: Removed {n_track_removed} detections from "
                     f"{len(tracks_to_remove)} bad tracks")
        progress("Phase B done")

        # ── Phase C: Depth stabilization + hand align ─────────────
        progress("Phase C: Optical flow segmentation...")
        from egoinfinity.pipeline.depth_stabilize import (
            stabilize_depth_sequence, smooth_translations, smooth_joints_savgol,
            reject_spikes, compute_background_rgb, refine_dynamic_masks,
            compute_background_template, build_optical_flow_masks)
        from egoinfinity.pipeline.depth_align import align_hand_to_depth_multiscale
        from collections import defaultdict

        # Optical flow dynamic masks (static camera → background flow ≈ 0)
        t0_flow = time.time()
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        from egoinfinity.pipeline.pose_tracker.debug_viz import is_debug_enabled
        # flow_rgbs are no longer rendered here — they're built at the end of
        # Phase D-track from pair_mag_list + obj_masks + obj_motion_pf so the
        # visualisation can include per-object motion overlays.  See the
        # `state['frame_data'][i]['flow_rgb']` second-pass write-back below.
        flow_masks, pair_mag_list = build_optical_flow_masks(
            frames_gray, magnitude_threshold=2.0, temporal_window=3, dilate_px=7,
            debug_cache_dir=(args.cache_dir if is_debug_enabled() and args.cache_dir
                             else None),
            debug_rgb_frames=frames_rgb if is_debug_enabled() else None,
            return_flow_rgbs=False,
            return_pair_mag=True,         # used by Phase D-track + flow_rgb overlay
        )
        flow_rgbs_per_frame = None         # populated post Phase D-track
        del frames_gray
        progress(f"Phase C: Optical flow done ({time.time() - t0_flow:.1f}s)")

        progress("Phase C: Depth stabilization...")
        hand_bboxes = [[h.bbox for h in hands] for hands in hand_results_per_frame]
        depth_maps_stable, bg_template, dynamic_masks = stabilize_depth_sequence(
            depth_maps, hand_bboxes, return_template=True, flow_masks=flow_masks)

        cx, cy = W / 2.0, H / 2.0
        state['cx'] = cx
        state['cy'] = cy
        aligned_cam_ts = []
        for i in range(total):
            hands = hand_results_per_frame[i]
            frame_ts = []
            for h in hands:
                t_a = align_hand_to_depth_multiscale(
                    h.joints_3d_rel, h.joints_2d, depth_maps_stable[i],
                    dp_focal, cx, cy, h.cam_t, h.scaled_focal)
                frame_ts.append(t_a)
            aligned_cam_ts.append(frame_ts)

        # ── Phase C+: Motion infiller (fill missing hand frames) ─────
        if not args.no_infiller:
            import os as _os
            from egoinfinity.pipeline.motion_infiller import MotionInfiller
            from egoinfinity.pipeline.config import INFILLER_CHECKPOINT
            ckpt_path = args.infiller_checkpoint or INFILLER_CHECKPOINT
            if _os.path.isfile(ckpt_path):
                progress("Phase C+: Loading motion infiller...")
                t0_infill = time.time()
                infiller = MotionInfiller(ckpt_path, device='cuda',
                                         mano_model=mano_model)
                hand_results_per_frame = infiller.fill_missing_frames(
                    hand_results_per_frame, total,
                    aligned_cam_ts=aligned_cam_ts)
                del infiller
                torch.cuda.empty_cache()
                progress(f"Phase C+: Infiller done ({time.time() - t0_infill:.1f}s)")
            else:
                progress(f"Phase C+: Infiller checkpoint not found at {ckpt_path}, skipping")

        # Pad aligned_cam_ts to match the post-infill hand counts. Filled
        # frames have h.cam_t already in the depth-aligned space (the infiller
        # interpolated aligned values), so the track_cam_ts loop below uses
        # h.cam_t for those slots.
        for i in range(total):
            while len(aligned_cam_ts[i]) < len(hand_results_per_frame[i]):
                aligned_cam_ts[i].append(None)

        # ── Phase C++: Biomechanical constraints ─────────────────────
        from egoinfinity.pipeline.biomech_constraints import apply_biomech_constraints
        n_clamped = apply_biomech_constraints(hand_results_per_frame, mano_model=mano_model)
        if n_clamped > 0:
            progress(f"Phase C++: Biomechanical constraints applied ({n_clamped} hands clamped)")

        track_cam_ts = defaultdict(dict)
        for i in range(total):
            for hi, h in enumerate(hand_results_per_frame[i]):
                ali = aligned_cam_ts[i][hi] if hi < len(aligned_cam_ts[i]) else None
                if ali is not None:
                    track_cam_ts[h.track_id][i] = ali
                elif h.confidence == 0.0:
                    # Infiller-filled frame: h.cam_t is already in the
                    # depth-aligned space (interpolated from neighbors' aligned
                    # cam_t inside the infiller).
                    track_cam_ts[h.track_id][i] = h.cam_t

        smoothed_map = {}
        for tid, fd in track_cam_ts.items():
            sf = sorted(fd.keys())
            seq = [fd[f] for f in sf]
            sm = smooth_translations(seq, window=5)
            for fi, f in enumerate(sf):
                smoothed_map[(f, tid)] = sm[fi]

        # Build per-track joints_3d with smoothed cam_t, reject spikes, then SavGol
        track_joints = defaultdict(dict)
        for i in range(total):
            for hi, h in enumerate(hand_results_per_frame[i]):
                ct = smoothed_map.get((i, h.track_id))
                if ct is not None:
                    track_joints[h.track_id][i] = h.joints_3d_rel + ct

        n_spikes = reject_spikes(track_joints, threshold_factor=3.0)
        if n_spikes > 0:
            progress(f"Phase C: Rejected {n_spikes} joint spikes")

        smoothed_joints = smooth_joints_savgol(track_joints, window=7, polyorder=2)
        # Flatten into (frame, track_id) -> (21,3) map
        smoothed_joints_map = {}
        for tid, fd in smoothed_joints.items():
            for f, j3d in fd.items():
                smoothed_joints_map[(f, tid)] = j3d

        # Smooth MANO params (global_orient + hand_pose) per track, then
        # recompute vertices via MANO forward for consistent mesh.
        from egoinfinity.pipeline.mano_smoothing import smooth_mano_params
        smoothed_verts_map = smooth_mano_params(
            hand_results_per_frame, smoothed_map, mano_model, total,
            smoothed_joints_map=smoothed_joints_map,
            window=7, polyorder=2)
        del mano_model

        progress("Phase C done (joints + MANO params smoothed)")

        # ── Phase D: SAM2 object tracking (multi-object) ───────────
        # obj_states_multi[frame_idx] = {obj_id: ObjectState or None}
        # MANUAL_DETECT_DISABLED: manual obj_prompts path kept for reference
        # but force-disabled. All detection goes through Phase D-sam3
        # (SAM3 text prompts).
        n_objects = len(obj_prompts_list)
        obj_states_multi = [{} for _ in range(total)]
        has_object = False  # MANUAL_DETECT_DISABLED
        if has_object:
            progress(f"Phase D: Loading SAM2 ({n_objects} object(s))...")
            from egoinfinity.pipeline.object_tracker import (
                ObjectTracker, mask_to_pointcloud, compute_obb, obb_corners, ObjectState)

            tracker = ObjectTracker()
            init_idx = min(obj_frame_idx, total - 1)
            progress(f"Phase D: SAM2 loaded, init on frame {init_idx}...")

            # Register all object prompts on init frame (point or bbox)
            for oid, prompt in enumerate(obj_prompts_list):
                ptype = prompt.get('type', 'point')
                pdata = prompt['data']
                kwargs = {'bbox': pdata} if ptype == 'bbox' else {'point': pdata}
                init_mask = tracker.init_track(frames_rgb[init_idx], obj_id=oid, **kwargs)
                pts0 = mask_to_pointcloud(init_mask, depth_maps_stable[init_idx], dp_focal, cx, cy, step=2)
                obb0 = compute_obb(pts0) if len(pts0) >= 10 else None
                smoother = tracker.smoothers.get(oid)
                if obb0 is not None:
                    c, a, e = obb0
                    sm_c, sm_r, sm_e = smoother.update(c, a, e)
                    obj_states_multi[init_idx][oid] = ObjectState(
                        mask=init_mask, center_3d=sm_c, extent_3d=sm_e,
                        rotation_3d=sm_r, corners_3d=obb_corners(sm_c, sm_r, sm_e),
                        n_points=len(pts0))
                else:
                    obj_states_multi[init_idx][oid] = ObjectState(
                        mask=init_mask, center_3d=np.zeros(3), extent_3d=np.zeros(3),
                        rotation_3d=np.eye(3), corners_3d=np.zeros((8, 3)), n_points=len(pts0))

            t0 = time.time()
            done = 1
            total_steps = total * 2 - 1  # forward + backward
            # Track forward from init_idx
            for i in range(init_idx + 1, total):
                results = tracker.track_frame(frames_rgb[i], depth_maps_stable[i], dp_focal, cx, cy, pc_step=2)
                obj_states_multi[i] = results
                done += 1
                fps = done / max(time.time() - t0, 0.001)
                progress(f"Phase D: SAM2 [{done}/{total_steps}] {fps:.1f}fps")
            # Track backward from init_idx
            if init_idx > 0:
                tracker.reset()
                for oid, prompt in enumerate(obj_prompts_list):
                    ptype = prompt.get('type', 'point')
                    pdata = prompt['data']
                    kwargs = {'bbox': pdata} if ptype == 'bbox' else {'point': pdata}
                    tracker.init_track(frames_rgb[init_idx], obj_id=oid, **kwargs)
                for i in range(init_idx - 1, -1, -1):
                    results = tracker.track_frame(frames_rgb[i], depth_maps_stable[i], dp_focal, cx, cy, pc_step=2)
                    obj_states_multi[i] = results
                    done += 1
                    fps = done / max(time.time() - t0, 0.001)
                    progress(f"Phase D: SAM2 [{done}/{total_steps}] {fps:.1f}fps")

            # Bidirectional OBB smoothing per object
            from egoinfinity.pipeline.object_tracker import smooth_obb_bidirectional
            for oid in range(n_objects):
                per_obj = [frame_dict.get(oid) for frame_dict in obj_states_multi]
                smooth_obb_bidirectional(per_obj, alpha=0.75)
                for i, s in enumerate(per_obj):
                    if s is not None:
                        obj_states_multi[i][oid] = s

            del tracker
            free_gpu()
            progress("Phase D done")
        else:
            progress("Phase D skipped (no object)")

        # ── Phase D-sam3: SAM3 text-prompted detection + SAM2 tracking ──
        # Parse SAM3 prompts (may be comma-separated list or empty)
        sam3_prompts = []
        if args.sam3_prompts:
            sam3_prompts = [p.strip() for p in args.sam3_prompts.split(",") if p.strip()]

        sam3_obj_states_multi = [{} for _ in range(total)]
        state['sam3_prompts'] = sam3_prompts  # save for viser/cache display

        if sam3_prompts:
            progress(f"Phase D-sam3: {len(sam3_prompts)} prompt(s): {sam3_prompts}")
            # Wipe any pre-existing sam3_meshes/*.ply from a prior pipeline
            # run.  Phase D-sam3 may produce a different oid count / numbering
            # (NMS, containment_merge, post-SAM2 dedup), so stale PLY files
            # from a previous run with old oid indices would either:
            #   - get coincidentally reused under wrong identity, or
            #   - become orphans uploaded to HF private cache as junk
            # Safer to start from a clean slate; A100's refresh_sam3d_meshes
            # regenerates whatever is needed.  Cheap (~few MB on local disk).
            if args.cache_dir:
                _mesh_dir = os.path.join(args.cache_dir, 'sam3_meshes')
                if os.path.isdir(_mesh_dir):
                    _n_wiped = 0
                    for _f in os.listdir(_mesh_dir):
                        if _f.endswith('.ply'):
                            try:
                                os.unlink(os.path.join(_mesh_dir, _f))
                                _n_wiped += 1
                            except OSError:
                                pass
                    if _n_wiped:
                        progress(f"Phase D-sam3: cleared {_n_wiped} stale "
                                 f"sam3_meshes/*.ply from prior run")
            from egoinfinity.pipeline.sam3_client import run_sam3_detect, nms_sam3_masks, containment_merge
            from egoinfinity.pipeline.object_tracker import ObjectTracker as _OT, ObjectState as _OS

            # Pick init frame: score every frame by (a) how static the scene
            # is and (b) how little of the image the hands occupy.  Highest
            # score = the cleanest "rest" view available.  Among frames within
            # 95% of the max score, pick the earliest — this biases toward the
            # pre-grasp rest pose when one exists, and degrades gracefully:
            #   • clip with pre-grasp rest    → score≈1.0 at frame 0–N → picked
            #   • clip held until put-down    → only post-grasp rest scores
            #     high → picked there (still untouched canonical)
            #   • all-grasp clip              → no static frame; least-busy
            #     frame still wins.  Pose tracking is rigid-hand-bound on
            #     every frame anyway, so canonical orientation matters less
            #     here (mesh quality is the only thing that can suffer).
            _H, _W = frames_rgb[0].shape[:2]
            _total_px = float(_H * _W)
            _static_ratio = np.zeros(total, dtype=np.float32)
            for _i in range(total):
                _fm = flow_masks[_i] if _i < len(flow_masks) else None
                if _fm is None or _fm.size == 0:
                    _static_ratio[_i] = 1.0
                else:
                    _static_ratio[_i] = 1.0 - float(_fm.sum()) / _total_px
            _hand_area = np.zeros(total, dtype=np.float32)
            for _i in range(total):
                for _h in (hand_results_per_frame[_i] or []):
                    if _h.bbox is None:
                        continue
                    _x0, _y0, _x1, _y1 = [float(_v) for _v in _h.bbox]
                    _hand_area[_i] += max(0.0, _x1 - _x0) * max(0.0, _y1 - _y0)
            _hand_area_norm = _hand_area / max(_total_px, 1.0)
            _hand_score = 1.0 - np.clip(_hand_area_norm * 4.0, 0.0, 1.0)
            _score = _static_ratio * _hand_score
            _max_score = float(_score.max()) if _score.size else 0.0
            if _max_score > 0.0:
                # Pick earliest within 95% of max.  When max is high (≥0.85
                # static), this lands on the rest segment.  When max is low
                # (busy clip), it still grabs the calmest cluster.
                _near_best = np.where(_score >= 0.95 * _max_score)[0]
                sam3_init_frame = int(_near_best[0])
            else:
                # Degenerate (no usable signal at all): fall back to old logic.
                _hand_frames = [i for i in range(total)
                                if hand_results_per_frame[i]]
                sam3_init_frame = (
                    _hand_frames[len(_hand_frames) // 2] if _hand_frames
                    else total // 2
                )
            progress(
                f"Phase D-sam3: init frame {sam3_init_frame} "
                f"(static={_static_ratio[sam3_init_frame]:.2f}, "
                f"hand_area={_hand_area_norm[sam3_init_frame]:.3f}, "
                f"score={_score[sam3_init_frame]:.2f}/{_max_score:.2f})"
            )

            # Multi-candidate init frame: run SAM-3 on N evenly-spaced frames
            # plus the heuristic pick, choose the candidate that detects the
            # most distinct prompts (mean score as tiebreaker). Single-frame
            # init silently loses prompts whose target object is not visible
            # at the chosen instant (occluded by hand, off-screen, not yet
            # introduced into the scene, etc.).
            import tempfile as _tempfile
            import shutil as _shutil
            SAM3_INIT_N_CANDIDATES = 8
            n_cand = max(1, min(SAM3_INIT_N_CANDIDATES, total))
            cand_idx = sorted(set(
                np.linspace(0, total - 1, n_cand, dtype=int).tolist()
                + [int(sam3_init_frame)]
            ))
            progress(f"Phase D-sam3: trying {len(cand_idx)} candidate init "
                     f"frames {cand_idx} (heuristic pick: {sam3_init_frame})")
            _shared_tmp = _tempfile.mkdtemp(prefix="sam3_init_")
            best = None  # ((n_uniq, mean_score), frame_idx, detections)
            try:
                for ci in cand_idx:
                    _frame_path = os.path.join(_shared_tmp, f"frame_{ci:06d}.jpg")
                    cv2.imwrite(_frame_path,
                                cv2.cvtColor(frames_rgb[ci], cv2.COLOR_RGB2BGR))
                    t_c = time.time()
                    try:
                        cand_results = run_sam3_detect(
                            image_path=_frame_path,
                            prompts=sam3_prompts,
                            min_score=0.4,
                            max_per_prompt=5,
                            version="sam3.1",
                        )
                    except Exception as _e:
                        progress(f"  cand frame {ci}: FAIL ({_e})")
                        continue
                    n_uniq = len(set(r.prompt for r in cand_results))
                    mean_score = (sum(r.score for r in cand_results)
                                  / max(len(cand_results), 1)) if cand_results else 0.0
                    progress(
                        f"  cand frame {ci}: {n_uniq}/{len(sam3_prompts)} "
                        f"unique prompts, {len(cand_results)} dets, "
                        f"mean_score={mean_score:.3f} "
                        f"({time.time() - t_c:.1f}s)"
                    )
                    score_tuple = (n_uniq, mean_score)
                    if best is None or score_tuple > best[0]:
                        best = (score_tuple, ci, cand_results)
                if best is not None:
                    sam3_init_frame = best[1]
                    sam3_results = best[2]
                    progress(
                        f"Phase D-sam3: chose init frame {sam3_init_frame} "
                        f"({best[0][0]} unique prompts, "
                        f"mean_score={best[0][1]:.3f}, "
                        f"{len(sam3_results)} raw detections)"
                    )
                else:
                    sam3_results = []
                    progress("Phase D-sam3: all candidates failed to detect")
            finally:
                _shutil.rmtree(_shared_tmp, ignore_errors=True)

            # NMS across prompts (dedupe overlapping masks from multiple prompts)
            # max_keep=7 to fit clips with multiple compound + simple fallbacks
            # (e.g., "plate of turkey", "plate", "turkey", "bowl of stuffing", ...).
            sam3_kept = nms_sam3_masks(sam3_results, iou_thresh=0.5, max_keep=7)
            # Containment merge: drop "white plate" when subsumed by
            # "white plate of sliced turkey" (compound + simple fallback pattern).
            n_before = len(sam3_kept)
            sam3_kept = containment_merge(sam3_kept, contain_thresh=0.9)
            if len(sam3_kept) < n_before:
                progress(f"Phase D-sam3: containment merge {n_before} → {len(sam3_kept)}")
            progress(f"Phase D-sam3: {len(sam3_kept)} after NMS")
            for i, m in enumerate(sam3_kept):
                progress(f"  sam3_obj_{i}: prompt='{m.prompt}' score={m.score:.3f} "
                         f"area={m.area}")

            if sam3_kept:
                # Feed SAM3 bboxes to SAM2 camera tracker for temporal propagation
                sam3_tracker = _OT()

                for oid, m in enumerate(sam3_kept):
                    # Use bbox prompt (more stable than mask prompt)
                    sam3_tracker.init_track(
                        frames_rgb[sam3_init_frame],
                        bbox=list(m.box), obj_id=oid)
                    # Store SAM3's own mask at init frame (higher quality than SAM2 re-seg)
                    sam3_obj_states_multi[sam3_init_frame][oid] = _OS(
                        mask=m.mask, center_3d=np.zeros(3), extent_3d=np.zeros(3),
                        rotation_3d=np.eye(3), corners_3d=np.zeros((8, 3)),
                        n_points=int(m.mask.sum()))

                t0_s3 = time.time()
                done_s3 = 0
                s3_total_steps = total - 1

                # Forward
                for i in range(sam3_init_frame + 1, total):
                    results = sam3_tracker.track_frame(frames_rgb[i])
                    sam3_obj_states_multi[i] = results
                    done_s3 += 1
                    fps = done_s3 / max(time.time() - t0_s3, 0.001)
                    progress(f"Phase D-sam3: SAM2-track [{done_s3}/{s3_total_steps}] {fps:.1f}fps")

                # Backward
                if sam3_init_frame > 0:
                    sam3_tracker.reset()
                    for oid, m in enumerate(sam3_kept):
                        sam3_tracker.init_track(
                            frames_rgb[sam3_init_frame],
                            bbox=list(m.box), obj_id=oid)
                    for i in range(sam3_init_frame - 1, -1, -1):
                        results = sam3_tracker.track_frame(frames_rgb[i])
                        sam3_obj_states_multi[i] = results
                        done_s3 += 1
                        fps = done_s3 / max(time.time() - t0_s3, 0.001)
                        progress(f"Phase D-sam3: SAM2-track [{done_s3}/{s3_total_steps}] {fps:.1f}fps")

                del sam3_tracker
                free_gpu()

                # ── Post-SAM2 trajectory dedup ─────────────────────
                # The cross-prompt NMS at SAM3 raw output (mask-IoU > 0.92,
                # see sam3_client.py:nms_sam3_masks) can't catch the case
                # where two prompts produce masks of different "tightness"
                # for the same physical object (e.g. "white bowl" gets the
                # bowl interior at 2300px while "bowl of vegetables" gets
                # the whole bowl at 3800px → raw IoU=0.6, but bbox-IoU=0.95).
                # SAM2 then propagates both bboxes; they converge to the
                # same physical object and produce per-frame masks with
                # IoU ≈ 1.0 across the whole video.  Catch and drop the
                # lower-scoring oid after propagation, when the trajectory
                # statistics are unambiguous.
                _DEDUP_IOU = float(os.environ.get(
                    'EGOINFINITY_SAM3_DEDUP_IOU', '0.80'))
                _DEDUP_AREA_RATIO = float(os.environ.get(
                    'EGOINFINITY_SAM3_DEDUP_AREA_RATIO', '0.70'))

                def _mean_iou_and_area_ratio(oid_a, oid_b):
                    """Mean per-frame mask-IoU + mean area ratio across all
                    frames where both oids have non-empty masks."""
                    inters = 0
                    unions = 0
                    a_areas = []
                    b_areas = []
                    for fi in range(total):
                        sa = sam3_obj_states_multi[fi].get(oid_a)
                        sb = sam3_obj_states_multi[fi].get(oid_b)
                        if sa is None or sb is None: continue
                        if sa.mask is None or sb.mask is None: continue
                        ia = int((sa.mask & sb.mask).sum())
                        ua = int((sa.mask | sb.mask).sum())
                        if ua == 0: continue
                        inters += ia
                        unions += ua
                        a_areas.append(int(sa.mask.sum()))
                        b_areas.append(int(sb.mask.sum()))
                    if unions == 0 or not a_areas:
                        return 0.0, 0.0
                    miou = inters / unions
                    mean_a = sum(a_areas) / len(a_areas)
                    mean_b = sum(b_areas) / len(b_areas)
                    aratio = min(mean_a, mean_b) / max(mean_a, mean_b, 1.0)
                    return miou, aratio

                # Greedy: sort by SAM3 score desc, keep each oid unless it
                # duplicates an already-kept higher-scoring one.
                order = sorted(range(len(sam3_kept)),
                               key=lambda i: -sam3_kept[i].score)
                dropped = []  # list of (loser_oid, winner_oid, miou, aratio)
                kept_set = []
                for oid in order:
                    is_dup = False
                    for kept_oid in kept_set:
                        miou, aratio = _mean_iou_and_area_ratio(oid, kept_oid)
                        if miou >= _DEDUP_IOU and aratio >= _DEDUP_AREA_RATIO:
                            is_dup = True
                            dropped.append((oid, kept_oid, miou, aratio))
                            break
                    if not is_dup:
                        kept_set.append(oid)

                if dropped:
                    progress(f"Phase D-sam3 dedup: dropping {len(dropped)} duplicate(s) "
                             f"(IoU>={_DEDUP_IOU}, area_ratio>={_DEDUP_AREA_RATIO})")
                    for loser, winner, miou, aratio in dropped:
                        progress(f"  oid {loser} ('{sam3_kept[loser].prompt}') "
                                 f"≈ oid {winner} ('{sam3_kept[winner].prompt}') "
                                 f"meanIoU={miou:.3f} areaRatio={aratio:.3f}")
                    # Remove dropped oids from sam3_obj_states_multi.
                    # Re-index surviving oids to a contiguous [0..N) range
                    # so downstream (sam3d_client, viser, etc.) sees a
                    # clean numbering.
                    survivor_old_ids = sorted(kept_set, key=lambda i: i)
                    # NOTE: we sort by ORIGINAL oid to preserve the order
                    # the user sees in sam3_prompt_mapping for any clip
                    # where no dedup fires.
                    new_sam3_kept = [sam3_kept[i] for i in survivor_old_ids]
                    old_to_new = {old: new
                                  for new, old in enumerate(survivor_old_ids)}
                    new_states_multi = [{} for _ in range(total)]
                    for fi in range(total):
                        for old_id, st in sam3_obj_states_multi[fi].items():
                            if old_id in old_to_new:
                                new_states_multi[fi][old_to_new[old_id]] = st
                    sam3_kept = new_sam3_kept
                    sam3_obj_states_multi = new_states_multi
                    progress(f"Phase D-sam3 dedup: {len(sam3_kept)} oids remain "
                             f"(was {len(sam3_kept) + len(dropped)})")

                progress(f"Phase D-sam3 done ({len(sam3_kept)} objects tracked)")
                # Save the prompt->object mapping for UI display
                state['sam3_prompt_mapping'] = [
                    {'prompt': m.prompt, 'score': m.score} for m in sam3_kept]
            else:
                progress("Phase D-sam3: no masks passed NMS")
                state['sam3_prompt_mapping'] = []
        else:
            progress("Phase D-sam3 skipped (no --sam3_prompts)")
            state['sam3_prompt_mapping'] = []

        # Multi-host sentinel: record D-sam3 completion so A100's
        # `refresh_sam3d_meshes --scan` knows this clip is ready for
        # SAM3D.  Skipped when no prompts (sam3 didn't run at all).
        if sam3_prompts and args.cache_dir:
            try:
                from egoinfinity.pipeline import pipeline_state as _ps
                _ps.mark_done(args.cache_dir, "D-sam3")
            except Exception as _e:
                progress(f"Phase D-sam3 state mark failed (non-fatal): {_e}")

        # ── Phase D-sam3d: single-image 3D reconstruction per object ──
        # For each SAM3-detected object, call SAM 3D Objects on the init
        # frame's (RGB + mask) to produce a 3D Gaussian Splat (.ply) saved
        # to disk.  Per-frame pose comes from PCA OBB in the frame-build
        # loop below.  Non-fatal: if the SAM3D worker is absent, skip.
        # Explicit env-var gate ON TOP OF the worker_available() check —
        # lets a host opt out of D-sam3d even when its socket happens to
        # be live (e.g. on a 4070 Ti that shouldn't be running SAM3D).
        _RUN_SAM3D = os.environ.get('EGOINFINITY_RUN_SAM3D', '1').strip() not in (
            '', '0', 'false', 'False', 'no', 'No')
        sam3_mesh_info = {}
        if not _RUN_SAM3D:
            progress("Phase D-sam3d skipped (EGOINFINITY_RUN_SAM3D=0)")
        try:
            from egoinfinity.pipeline.sam3d_client import reconstruct_object, worker_available
            if _RUN_SAM3D and sam3_prompts and sam3_obj_states_multi[sam3_init_frame] \
                    and worker_available():
                mesh_out_dir = os.path.join(args.cache_dir, 'sam3_meshes') \
                    if args.cache_dir else None
                if mesh_out_dir:
                    os.makedirs(mesh_out_dir, exist_ok=True)

                init_frame_rgb = frames_rgb[sam3_init_frame]
                for oid, obj_state in list(
                        sam3_obj_states_multi[sam3_init_frame].items()):
                    if obj_state is None or obj_state.mask is None \
                            or not obj_state.mask.any():
                        continue
                    if mesh_out_dir is None:
                        continue   # no cache dir, can't persist PLY
                    ply_path = os.path.join(mesh_out_dir, f'obj_{oid}.ply')
                    progress(f"Phase D-sam3d: reconstructing obj {oid} "
                             f"(mask area {int(obj_state.mask.sum())})...")
                    try:
                        t0_r = time.time()
                        res = reconstruct_object(
                            rgb=init_frame_rgb,
                            mask=obj_state.mask,
                            out_ply_path=ply_path,
                            seed=42, quality='tier1',
                            timeout=300.0)
                        sam3_mesh_info[int(oid)] = {
                            'ply_path': ply_path,
                            'init_frame': int(sam3_init_frame),
                            'canonical_translation': res.translation.tolist(),
                            'canonical_rotation_quat': res.rotation_quat.tolist(),
                            'canonical_scale': float(res.scale),
                            'n_points': int(res.n_points),
                        }
                        progress(f"  obj {oid}: {res.n_points} pts, "
                                 f"{time.time()-t0_r:.1f}s")
                    except Exception as _e:
                        progress(f"Phase D-sam3d: obj {oid} FAILED ({_e})")
                progress(f"Phase D-sam3d done ({len(sam3_mesh_info)} meshes)")
            elif _RUN_SAM3D and sam3_prompts and not worker_available():
                progress("Phase D-sam3d: worker unavailable, skipping mesh reconstruction")
        except Exception as _e:
            progress(f"Phase D-sam3d: skipped ({_e})")
        state['sam3_mesh_info'] = sam3_mesh_info
        if _RUN_SAM3D and sam3_mesh_info and args.cache_dir:
            try:
                from egoinfinity.pipeline import pipeline_state as _ps
                _ps.mark_done(args.cache_dir, "D-sam3d")
            except Exception as _e:
                progress(f"Phase D-sam3d state mark failed (non-fatal): {_e}")

        # ── Phase E: Refine masks + background ────────────────────
        progress("Phase E: Refining masks...")
        # Merge all object masks (manual + auto-trajectory + sam3) for background subtraction
        sam2_masks = [None] * total
        for i in range(total):
            combined = None
            for s in obj_states_multi[i].values():
                if s is not None and s.mask is not None:
                    combined = s.mask if combined is None else (combined | s.mask)
            for s in sam3_obj_states_multi[i].values():
                if s is not None and s.mask is not None:
                    combined = s.mask if combined is None else (combined | s.mask)
            sam2_masks[i] = combined
        if all(m is None for m in sam2_masks):
            sam2_masks = None

        dynamic_masks = refine_dynamic_masks(
            dynamic_masks, depth_maps_stable,
            object_masks=sam2_masks, depth_variance_threshold=3.0, dilate_px=5)

        bg_template = compute_background_template(depth_maps_stable, dynamic_masks)
        bg_rgb = compute_background_rgb(frames_rgb, dynamic_masks)
        state['bg_template'] = bg_template
        state['bg_rgb'] = bg_rgb
        progress("Phase E done")

        # ── Build frame_data ──────────────────────────────────────
        # JPG encoding runs in a background thread pool while the rest of the
        # pipeline continues (Phase D-track in particular is slow), so by the
        # time save_cache is called, encoding is essentially free.
        # cv2.imencode releases the GIL during compression -> threads scale.
        from concurrent.futures import ThreadPoolExecutor
        _jpg_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpg-enc")
        _jpg_pending = []  # list of (frame_idx, key, future)

        def _jpg_encode_rgb(rgb_arr, q):
            if rgb_arr is None:
                return None
            bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return bytes(buf) if ok else None

        progress("Building frames...")
        for i in range(total):
            # Reuse frames_rgb decoded in Phase A; avoid disk re-read.
            img_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
            hands = hand_results_per_frame[i]
            depth_map = depth_maps_stable[i]

            dc = depth_to_colormap(depth_map)
            img_ov = img_bgr.copy()
            depth_ov = cv2.addWeighted(img_bgr, 0.6, dc, 0.4, 0)
            for h in hands:
                draw_skeleton_2d(img_ov, h.joints_2d, b_colors, (255, 200, 0), 2, 3)

            # Object mask overlay (multi-object with different colors)
            for oid, os_i in obj_states_multi[i].items():
                if os_i is not None and os_i.mask is not None:
                    omask = os_i.mask
                    oc = OBJ_COLORS_RGB[oid % len(OBJ_COLORS_RGB)].astype(np.float64)
                    # BGR for overlay
                    oc_bgr = oc[::-1]
                    img_ov[omask] = (img_ov[omask] * 0.6 + oc_bgr * 0.4).astype(np.uint8)
                    depth_ov[omask] = (depth_ov[omask] * 0.6 + oc_bgr * 0.4).astype(np.uint8)

            j3d_pred = []
            j2d_pred = []
            verts_3d = []
            hand_is_right = []
            hand_meta = []
            for hi_idx, h in enumerate(hands):
                ct_aligned = aligned_cam_ts[i][hi_idx] if hi_idx < len(aligned_cam_ts[i]) else h.cam_t
                key = (i, h.track_id)
                ct_smooth = smoothed_map.get(key, ct_aligned)
                if ct_smooth is None:
                    ct_smooth = ct_aligned

                # Use SavGol-smoothed joints if available
                j3d_raw = h.joints_3d_rel + ct_smooth
                j3d = smoothed_joints_map.get(key, j3d_raw)
                j3d_pred.append(j3d)

                # WiLoR-native 2D joints — pixel-space, MoGe-2-independent.
                # Stored verbatim for the frontend skeleton overlay so it
                # can skip 3D→2D projection (which depends on dp_focal and
                # therefore on MoGe-2's per-frame focal estimate).
                j2d = getattr(h, 'joints_2d', None)
                if j2d is None:
                    j2d = np.full((21, 2), np.nan, dtype=np.float32)
                else:
                    j2d = np.asarray(j2d, dtype=np.float32)
                j2d_pred.append(j2d)

                # Use MANO-smoothed vertices if available, else fallback to wrist delta
                v_smooth = smoothed_verts_map.get(key)
                if v_smooth is not None:
                    verts_3d.append(v_smooth)
                else:
                    verts_rel = h.vertices - h.cam_t
                    delta = j3d[0] - j3d_raw[0]
                    verts_3d.append(verts_rel + ct_smooth + delta)
                hand_is_right.append(h.is_right)

                # Debug metadata
                if h.confidence > 0:
                    source = 'detected'
                else:
                    source = 'infilled'
                hand_meta.append({
                    'source': source,
                    'confidence': h.confidence,
                    'track_id': h.track_id,
                    'cam_t_raw': h.cam_t.copy(),
                    'cam_t_smooth': ct_smooth.copy() if ct_smooth is not None else None,
                })

            # Multi-object point clouds and OBB corners
            obj_data = {}
            from egoinfinity.pipeline.object_tracker import mask_to_pointcloud, filter_outliers_sor
            for oid, os_i in obj_states_multi[i].items():
                if os_i is not None and os_i.n_points > 0:
                    pts = mask_to_pointcloud(os_i.mask, depth_map, dp_focal, cx, cy, step=2)
                    if len(pts) > 20:
                        pts = filter_outliers_sor(pts, k=20, std_ratio=2.0)
                    corners = os_i.corners_3d if np.any(os_i.corners_3d) else None
                    obj_data[oid] = {'pts': pts, 'corners': corners}

            # SAM3-detected object point clouds + per-frame PCA OBB pose
            # OBB is used by viser to attach the SAM3D-reconstructed mesh
            # (if present in state['sam3_mesh_info']) and rigidly follow the
            # object across frames.
            sam3_obj_data = {}
            from egoinfinity.pipeline.object_tracker import compute_obb_from_mask
            for oid, os_i in sam3_obj_states_multi[i].items():
                if os_i is None or os_i.mask is None or not os_i.mask.any():
                    continue
                pts = mask_to_pointcloud(os_i.mask, depth_map, dp_focal, cx, cy, step=2)
                if len(pts) > 20:
                    pts = filter_outliers_sor(pts, k=20, std_ratio=2.0)
                if len(pts) == 0:
                    continue
                entry = {'pts': pts}
                # Per-frame OBB pose (may be None for tiny / degenerate masks)
                obb = compute_obb_from_mask(
                    os_i.mask, depth_map, dp_focal, cx, cy, step=2)
                if obb is not None:
                    entry['pose_R']      = obb.R.astype(np.float32)
                    entry['pose_t']      = obb.center_3d.astype(np.float32)
                    entry['obb_corners'] = obb.corners_3d.astype(np.float32)
                    entry['obb_extent']  = obb.extent.astype(np.float32)
                # Save the raw 2D mask (packbits-compressed) so the pose
                # tracker can run offline against the cached pkl without
                # rerunning the full pipeline.  ~50 KB per mask after
                # packing; ~8 MB extra per clip with 2 objects × 82 frames.
                entry['mask_packed'] = np.packbits(os_i.mask, axis=None)
                entry['mask_shape']  = np.asarray(os_i.mask.shape, dtype=np.int32)
                sam3_obj_data[oid] = entry

            # JPG-encode the two viz images that don't depend on Phase D-track
            # output.  flow_rgb is encoded later (after obj_motion_pf is
            # computed) so the visualisation can include per-object overlays.
            img_rgb_arr = cv2.cvtColor(img_ov, cv2.COLOR_BGR2RGB)
            depth_rgb_arr = cv2.cvtColor(depth_ov, cv2.COLOR_BGR2RGB)
            f_img   = _jpg_pool.submit(_jpg_encode_rgb, img_rgb_arr,   90)
            f_depth = _jpg_pool.submit(_jpg_encode_rgb, depth_rgb_arr, 85)
            _jpg_pending.append((i, 'img_rgb',   f_img))
            _jpg_pending.append((i, 'depth_rgb', f_depth))
            state['frame_data'].append({
                'img_rgb':   None,   # patched below from _jpg_pending futures
                'depth_rgb': None,
                'flow_rgb':  None,
                'joints_3d_pred': j3d_pred,
                'joints_2d_pred': j2d_pred,
                'vertices_3d': verts_3d,
                'hand_is_right': hand_is_right,
                'hand_meta': hand_meta,
                'depth_map': depth_map,
                'obj_data': obj_data,
                'sam3_obj_data': sam3_obj_data,
                'metrics': {},
            })

        # ── Phase D-track: 6DoF pose tracking per SAM3D mesh ──────
        # Runs AFTER state['frame_data'] is populated so we can read vertices_3d
        # and sam3_obj_data.  Replaces per-frame PCA OBB (jittery, obs-driven)
        # with a mesh-to-camera pose via anchor FGR+ICP then optical-flow +
        # RANSAC-PnP propagation.  Falls back to PCA OBB fields on failure.
        # Explicit env gate — 4070 Ti can set EGOINFINITY_RUN_TRACK=0 when
        # D-track is being deferred until A100 has filled in SAM3D meshes.
        _RUN_TRACK = os.environ.get('EGOINFINITY_RUN_TRACK', '1').strip() not in (
            '', '0', 'false', 'False', 'no', 'No')
        pose_track_info = {}
        if not _RUN_TRACK:
            progress("Phase D-track skipped (EGOINFINITY_RUN_TRACK=0)")
        if _RUN_TRACK and sam3_mesh_info:
            progress("Phase D-track: 6DoF mesh tracking...")
            try:
                from egoinfinity.pipeline.pose_tracker import (
                    select_anchor_frame, estimate_anchor_pose, track_6dof,
                    build_K, render_hand_mask, load_mesh_points_from_ply,
                )
                K_mat = build_K(dp_focal, cx, cy)

                progress("Phase D-track: rendering hand masks...")
                hand_masks_allframes = [None] * total
                mano_f = state.get('mano_faces')
                for i in range(total):
                    verts = state['frame_data'][i].get('vertices_3d') or []
                    verts = [v for v in verts if v is not None and len(v) > 0]
                    if verts and mano_f is not None:
                        hand_masks_allframes[i] = render_hand_mask(
                            verts, mano_f, K_mat, H, W, dilate_px=3)

                for oid, meta in sam3_mesh_info.items():
                    ply = meta.get('ply_path')
                    if not ply or not os.path.isfile(ply):
                        continue
                    obj_masks = [None] * total
                    for i in range(total):
                        st = sam3_obj_states_multi[i].get(oid)
                        if st is not None and st.mask is not None:
                            obj_masks[i] = st.mask

                    anchor_t = select_anchor_frame(obj_masks, hand_masks_allframes)
                    if anchor_t < 0:
                        pose_track_info[oid] = {'tracking_status': 'fail',
                                                 'reason': 'no_anchor'}
                        progress(f"  obj {oid}: no valid anchor frame, "
                                 f"keeping PCA OBB fallback")
                        continue

                    progress(f"  obj {oid}: anchor={anchor_t}, FGR+ICP...")
                    mesh_xyz, _ = load_mesh_points_from_ply(ply)

                    # Build T_init from SAM3D's reported single-frame pose.
                    # SAM3D applies (Transform3d).scale(s).rotate(R).translate(t)
                    # in row-form (p @ R), then we flip P3D → OpenCV via diag(-1,-1,1).
                    # In column form: T_init_sim mesh_canonical → OpenCV camera is
                    #   [[s * F @ R^T, F @ t], [0, 1]]
                    # where F = diag(-1, -1, 1) and R is the column-form rotation
                    # from quaternion_to_matrix.
                    T_init_sim = None
                    try:
                        q = meta.get('canonical_rotation_quat')
                        t_p3d = meta.get('canonical_translation')
                        s_p3d = meta.get('canonical_scale')
                        if q is not None and t_p3d is not None and s_p3d is not None:
                            from egoinfinity.pipeline.pose_tracker.utils import (
                                _quat_wxyz_to_mat as _q2m,
                            )
                    except Exception:
                        _q2m = None
                    if _q2m is None:
                        # Inline conversion (avoid importing private helper if absent)
                        def _q2m(q4):
                            w, x, y, z = q4
                            n = (w*w + x*x + y*y + z*z) ** 0.5
                            if n == 0:
                                return np.eye(3)
                            w, x, y, z = w/n, x/n, y/n, z/n
                            return np.array([
                                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
                            ])
                    q = meta.get('canonical_rotation_quat')
                    t_p3d_arr = meta.get('canonical_translation')
                    s_p3d = meta.get('canonical_scale')
                    if q is not None and t_p3d_arr is not None and s_p3d is not None:
                        F = np.diag([-1.0, -1.0, 1.0])
                        R_col = _q2m(np.asarray(q, dtype=np.float64))
                        # PyTorch3D rotates as p @ R_col (row form), so column-form
                        # rotation = R_col.T.  Then apply flip on the left.
                        R_total = float(s_p3d) * (F @ R_col.T)
                        t_total = F @ np.asarray(t_p3d_arr, dtype=np.float64)
                        T_init_sim = np.eye(4)
                        T_init_sim[:3, :3] = R_total
                        T_init_sim[:3, 3] = t_total

                    # 2026-05-01: When SAM3D provides T_init_sim, skip FGR/ICP
                    # entirely — anchor R = SAM3D canonical, t = align mesh mean
                    # to obs cloud mean.  PCA-axis2 + chamfer sanity check still
                    # runs (shrink-only).  Falls back to legacy ICP path only
                    # when there is no SAM3D prior at all.
                    if T_init_sim is not None:
                        from egoinfinity.pipeline.pose_tracker.anchor import (
                            build_anchor_from_prior,
                        )
                        ares = build_anchor_from_prior(
                            anchor_idx=anchor_t,
                            T_init_sim=T_init_sim,
                            mesh_pts_canonical=mesh_xyz,
                            depth=depth_maps_stable[anchor_t],
                            mask=obj_masks[anchor_t],
                            hand_mask=hand_masks_allframes[anchor_t],
                            K=K_mat,
                        )
                        progress(
                            f"  obj {oid}: anchor from SAM3D prior "
                            f"(scale={ares.scale_correction:.3f}, "
                            f"n_obs={ares.n_obs_pts}, "
                            f"chamfer={ares.nn_mean_err_m*1000:.0f}mm)")
                    else:
                        # Legacy path — ICP for objects without SAM3D prior.
                        ares = estimate_anchor_pose(
                            anchor_idx=anchor_t,
                            mesh_pts_canonical=mesh_xyz,
                            depth=depth_maps_stable[anchor_t],
                            mask=obj_masks[anchor_t],
                            hand_mask=hand_masks_allframes[anchor_t],
                            K=K_mat,
                            T_init_sim=None,
                            world_up=state.get('gravity_up'),
                        )
                        if ares.fitness < 0.5:
                            pose_track_info[oid] = {
                                'tracking_status': 'fail',
                                'reason': 'anchor_low_fitness_no_prior',
                                'anchor_frame': int(anchor_t),
                                'anchor_fitness': float(ares.fitness),
                                'note': ares.note,
                            }
                            progress(f"  obj {oid}: anchor FAIL "
                                     f"(fitness={ares.fitness:.3f}, no SAM3D prior), "
                                     f"keeping PCA OBB")
                            continue

                    if abs(ares.scale_correction - 1.0) > 0.02:
                        mesh_xyz = mesh_xyz * ares.scale_correction

                    # Translation-only anchor refinement: align mesh centroid to
                    # observed point-cloud centroid at the anchor frame.  Catches
                    # the constant SAM3D ↔ MoGe-2 frame offset that ICP can't fix
                    # under low fitness.  Robust because centroid alignment doesn't
                    # depend on having clean correspondence pairs.
                    try:
                        from egoinfinity.pipeline.pose_tracker.anchor import _make_anchor_pointcloud
                        obs_cloud = _make_anchor_pointcloud(
                            depth_maps_stable[anchor_t], obj_masks[anchor_t],
                            hand_masks_allframes[anchor_t], K_mat)
                        if len(obs_cloud) >= 50:
                            R_a, t_a = ares.T[:3, :3], ares.T[:3, 3]
                            mesh_in_cam = mesh_xyz @ R_a.T + t_a
                            mesh_c = mesh_in_cam.mean(0)
                            obs_c = obs_cloud.mean(0)
                            delta_t = obs_c - mesh_c
                            d_norm = float(np.linalg.norm(delta_t))
                            # Only apply if delta is non-trivial but plausible.
                            # > 50cm is suspicious (could be wrong-object alignment);
                            # < 5mm is noise.
                            if 0.005 < d_norm < 0.50:
                                ares.T[:3, 3] = t_a + delta_t
                                progress(f"  obj {oid}: anchor t-refine Δ={d_norm*100:.1f}cm")
                    except Exception:
                        pass

                    # ── Layer A: try-multi-orientation anchor refinement ──
                    # SAM3D's canonical orientation is arbitrary for symmetric /
                    # elongated objects.  Search the 24 cube-symmetry rotations
                    # and pick the one whose rendered mask best matches SAM2.
                    _orient_diag = {}
                    try:
                        from egoinfinity.pipeline.pose_tracker import (
                            search_anchor_orientation,
                        )
                        T_best, iou_best, idx_best, _orient_diag = \
                            search_anchor_orientation(
                                T_anchor_init=ares.T,
                                mesh_pts=mesh_xyz,
                                mask_anchor=obj_masks[anchor_t],
                                hand_mask_anchor=hand_masks_allframes[anchor_t],
                                depth_anchor=depth_maps_stable[anchor_t],
                                K=K_mat,
                                rotation_set="cube",
                            )
                        if _orient_diag.get("replaced"):
                            ares.T = T_best
                            progress(
                                f"  obj {oid}: orientation search: "
                                f"IoU {_orient_diag['iou_base']:.3f} → "
                                f"{iou_best:.3f} (cand #{idx_best})")
                        else:
                            progress(
                                f"  obj {oid}: orientation search: kept base "
                                f"(IoU {_orient_diag['iou_base']:.3f})")
                    except Exception as _eo:
                        progress(f"  obj {oid}: orientation search failed: {_eo}")

                    progress(f"  obj {oid}: propagating ({total} frames)...")
                    tr = track_6dof(
                        mesh_pts=mesh_xyz, anchor_idx=anchor_t, T_anchor=ares.T,
                        frames_rgb=frames_rgb, masks=obj_masks,
                        hand_masks=hand_masks_allframes, K=K_mat)

                    # Centroid-tracking fallback for un-propagated frames.
                    # When PnP fails (small / heavily-occluded objects like a
                    # held knife), gap-fill just freezes the pose between
                    # neighbouring known frames.  For those frames, replace the
                    # interpolated translation with the SAM2 mask centroid back-
                    # projected via the per-frame depth.  Keeps the rotation
                    # from gap-fill SLERP.  Object follows the mask.
                    fx_, fy_ = K_mat[0,0], K_mat[1,1]
                    cx_, cy_ = K_mat[0,2], K_mat[1,2]
                    n_centroid = 0
                    for i in range(total):
                        if tr.propagated[i]:
                            continue
                        m_i = obj_masks[i]
                        if m_i is None or not m_i.any():
                            continue
                        h_i = hand_masks_allframes[i]
                        eff = m_i & ~h_i if h_i is not None else m_i
                        if not eff.any():
                            eff = m_i
                        ys, xs = np.where(eff)
                        d_i = depth_maps_stable[i]
                        z_pix = d_i[ys, xs]
                        valid = (z_pix > 1e-3) & np.isfinite(z_pix)
                        if valid.sum() < 5:
                            continue
                        # median is robust to depth jumps at edges
                        u_med = float(np.median(xs[valid]))
                        v_med = float(np.median(ys[valid]))
                        z_med = float(np.median(z_pix[valid]))
                        x_3d = (u_med - cx_) * z_med / fx_
                        y_3d = (v_med - cy_) * z_med / fy_
                        tr.T_seq[i][:3, 3] = np.array([x_3d, y_3d, z_med])
                        n_centroid += 1
                    if n_centroid > 0:
                        progress(f"  obj {oid}: centroid-fallback applied to "
                                 f"{n_centroid} un-propagated frames")

                    # ── Stage 3: Trust filter (per-frame validation) ──
                    # Three signals: mask IoU, partial Chamfer, PnP inlier ratio.
                    # Diagnostic only — does not modify pose.  Used by Stage 4.
                    try:
                        from egoinfinity.pipeline.pose_tracker import evaluate_trust
                        trust_res = evaluate_trust(
                            T_seq=tr.T_seq,
                            mesh_pts=mesh_xyz,
                            masks=obj_masks,
                            hand_masks=hand_masks_allframes,
                            depth_maps=depth_maps_stable,
                            K=K_mat,
                            inlier_ratios=tr.inlier_ratios,
                            propagated=tr.propagated,
                        )
                        progress(f"  obj {oid}: trust {trust_res.n_trusted}/{total} "
                                 f"(mean iou={np.nanmean(trust_res.iou):.2f} "
                                 f"chamfer={1000*np.nanmedian([c for c in trust_res.chamfer_m if np.isfinite(c)] or [0]):.1f}mm)")
                        _trust_data = {
                            'trust': trust_res.trust,
                            'trust_rate': trust_res.trust_rate,
                            'trust_diag': {
                                'iou': trust_res.iou,
                                'chamfer_m': trust_res.chamfer_m,
                                'inlier_ratio': trust_res.inlier_ratio,
                            },
                        }
                    except Exception as _e:
                        progress(f"  obj {oid}: trust eval failed ({_e})")
                        _trust_data = {}
                        trust_res = None

                    # ── Contact detection (2D mask overlap ∪ 3D hand-mesh distance) ──
                    _opt_data = {}
                    if trust_res is not None:
                        try:
                            from egoinfinity.pipeline.pose_tracker import (
                                detect_contact_2d_aware, detect_contact_per_frame,
                                MANO_PALM_VERTICES,
                                optimize_pose_seq,
                                complete_static_masks, detect_grasp_with_motion,
                            )
                            # Per-frame hand verts (world frame)
                            hverts_pf = [
                                state['frame_data'][i].get('vertices_3d') or []
                                for i in range(total)
                            ]
                            hright_pf = [
                                state['frame_data'][i].get('hand_is_right') or []
                                for i in range(total)
                            ]
                            joints_pf_for_grasp = [
                                state['frame_data'][i].get('joints_3d_pred') or []
                                for i in range(total)
                            ]
                            mano_f = state.get('mano_faces')

                            # ── Mask completion for static objects ──
                            # Heals SAM2 masks where the hand occludes part of a
                            # *static* object — keeps centroid stable for grasp
                            # detection and contact 2D overlap.  Non-static
                            # objects pass through unchanged.
                            obj_masks_completed, mc_is_static, mc_diag = \
                                complete_static_masks(obj_masks, hand_masks_allframes)
                            progress(
                                f"  obj {oid}: mask_completion static={mc_is_static} "
                                f"avg_iou={mc_diag.get('avg_iou', 0):.2f} "
                                f"(healed {sum(1 for i,m in enumerate(obj_masks_completed) if m is not None and obj_masks[i] is not None and m is not obj_masks[i])} frames)")

                            # Debug viz: dump raw / completed masks (env-gated)
                            try:
                                from egoinfinity.pipeline.pose_tracker import (
                                    is_debug_enabled, dump_completed_masks)
                                if is_debug_enabled() and args.cache_dir:
                                    rgb_pf = [_decode_image_maybe(
                                        state['frame_data'][i].get('img_rgb'))
                                              for i in range(total)]
                                    dump_completed_masks(
                                        cache_dir=args.cache_dir,
                                        oid=oid,
                                        raw_masks=obj_masks,
                                        completed_masks=obj_masks_completed,
                                        rgb_frames=rgb_pf,
                                        suffix="static" if mc_is_static else "dynamic",
                                    )
                            except Exception as _viz_e:
                                progress(f"  obj {oid}: mask viz dump failed: {_viz_e}")

                            # ── Diagnostic only: 2D + 3D contact (display in GUI, not used for grasp) ──
                            contact_2d = detect_contact_2d_aware(
                                obj_mask_per_frame=obj_masks_completed,
                                hand_verts_per_frame=hverts_pf,
                                hand_is_right_per_frame=hright_pf,
                                mano_faces=mano_f,
                                K=K_mat, H=H, W=W,
                                overlap_px_threshold=30,
                            )
                            mesh_world_pf = []
                            for i in range(total):
                                Ti = tr.T_seq[i]
                                if Ti is None:
                                    mesh_world_pf.append(np.zeros((0, 3), dtype=np.float64))
                                else:
                                    Ti = np.asarray(Ti, dtype=np.float64)
                                    mesh_world_pf.append(
                                        mesh_xyz @ Ti[:3, :3].T + Ti[:3, 3])
                            contact_3d = detect_contact_per_frame(
                                hand_verts_per_frame=hverts_pf,
                                hand_is_right_per_frame=hright_pf,
                                mesh_world_per_frame=mesh_world_pf,
                                threshold_m=0.025,
                            )
                            contact_soft_raw = np.maximum(contact_2d, contact_3d).astype(np.float32)
                            n_2d = int((contact_2d.max(axis=1) > 0.5).sum())
                            n_3d = int((contact_3d.max(axis=1) > 0.5).sum())

                            from egoinfinity.pipeline.pose_tracker import (
                                compute_object_motion_per_frame,
                                detect_static_moving_segments,
                                per_frame_stability_flags,
                                compute_depth_pose_per_frame,
                                lock_pose_for_segment,
                                compute_obs_obb_per_frame,
                                hysteresis_filter,
                                rigid_wrist_binding_propagation,
                                find_continuous_segments,
                                detect_grasp_fingertip_persistent,
                            )
                            from egoinfinity.pipeline.pose_tracker.object_motion import (
                                compute_pc_motion_per_frame,
                            )
                            # Object motion signals are still useful for the
                            # is_moving / state classifier downstream, but no
                            # longer gate grasp detection itself.
                            obj_motion_pf_pre = compute_object_motion_per_frame(
                                pair_mag_list, obj_masks_completed, hand_masks_allframes,
                            )
                            pc_motion_pf_pre = compute_pc_motion_per_frame(trust_res.obs_clouds)

                            # ── Grasp = sustained fingertip↔obs proximity ──
                            # No motion gating, no curl check.  Pure geometry,
                            # offline morphological segment filtering:
                            #   close[t] = (min fingertip→obs dist ≤ 6cm)
                            #   bridge internal 0-runs ≤ 10 frames (≈0.67s)
                            #   drop 1-runs < 8 frames (≈0.53s)
                            #
                            # Build a per-frame cloud specifically for grasp
                            # detection from the **raw SAM2 mask** (NOT the
                            # completed mask, NOT trust_res.obs_clouds):
                            #   • trust_res.obs_clouds subtracts hand pixels
                            #     (right for ICP, wrong for grasp — fingers
                            #     contact exactly the surface that gets
                            #     removed).
                            #   • obj_masks_completed substitutes a static
                            #     median silhouette for occluded frames; back-
                            #     projecting that with per-frame stab depth
                            #     produces a cloud at unpredictable 3D
                            #     locations and decouples from the object's
                            #     actual current position.  See the CRITICAL
                            #     warning at the top of mask_completion.py:
                            #     "the completed pixels correspond to the
                            #     hand's depth, not the object's."
                            # Raw obj_masks (= SAM2-tracked, no hand subtract,
                            # no completion) gives a cloud that follows the
                            # object's actual current position.
                            from egoinfinity.pipeline.object_tracker import mask_to_pointcloud
                            grasp_obs_clouds = []
                            for _i in range(total):
                                _m = obj_masks[_i]
                                if _m is None or not _m.any():
                                    grasp_obs_clouds.append(None)
                                    continue
                                _pts = mask_to_pointcloud(
                                    _m, depth_maps_stable[_i],
                                    dp_focal, cx, cy, step=2,
                                )
                                grasp_obs_clouds.append(_pts if len(_pts) >= 30 else None)

                            contact_soft = detect_grasp_fingertip_persistent(
                                obs_clouds=grasp_obs_clouds,
                                joints_per_frame=joints_pf_for_grasp,
                                hand_is_right_per_frame=hright_pf,
                                fingertip_threshold_m=0.06,
                                bridge_gap_frames=10,
                                min_grasp_frames=8,
                            )
                            grasp_soft = contact_soft   # alias kept for downstream / pkl
                            n_contact = int((contact_soft_raw.max(axis=1) > 0.5).sum())
                            n_grasp = int((grasp_soft.max(axis=1) > 0.5).sum())
                            grasp_rate = n_grasp / max(total, 1)
                            progress(
                                f"  obj {oid}: contact_2d={n_2d}/{total} "
                                f"contact_3d={n_3d}/{total} "
                                f"grasp(fingertip persistent)={n_grasp}/{total} ({grasp_rate:.0%})")
                            from egoinfinity.pipeline.pose_tracker.object_motion import (
                                compute_pc_motion_per_frame,
                            )
                            # contact_soft (= grasp_soft) and contact_soft_raw (= 2D∪3D) already set above.
                            # Diagnostics from trust filter (kept for pkl)
                            chamfer_vals = [c for c in trust_res.chamfer_m if np.isfinite(c)]
                            mean_chamfer_m = (float(np.nanmedian(chamfer_vals))
                                              if chamfer_vals else float('inf'))
                            iou_vals = [v for v in trust_res.iou if np.isfinite(v)]
                            mean_iou_val = float(np.nanmean(iou_vals)) if iou_vals else 0.0

                            # ── 2026-05-01 reroute: hysteresis + rigid binding ──
                            # 1. Per-frame stability flags + hysteresis on every
                            #    binary signal that drives mode selection so
                            #    single-frame edge noise can't flip state.
                            mask_small_raw, pc_unstable_raw = per_frame_stability_flags(
                                obj_masks, trust_res.obs_clouds,
                            )
                            # mask_area + density signals for hysteresis (reverse direction)
                            mask_area_pf = np.array(
                                [int(m.sum()) if (m is not None and m.any()) else 0
                                 for m in obj_masks], dtype=np.float32)
                            density_pf = np.array([
                                (len(o) / max(int(m.sum()), 1))
                                if (o is not None and m is not None and m.any())
                                else 0.0
                                for o, m in zip(trust_res.obs_clouds, obj_masks)
                            ], dtype=np.float32)
                            # Reverse signals: small/unstable is True when value LOW.
                            mask_small_pf  = ~hysteresis_filter(mask_area_pf, low=600,  high=1200)
                            pc_unstable_pf = ~hysteresis_filter(density_pf,   low=0.25, high=0.35)
                            pc_stable_pf   = ~(mask_small_pf | pc_unstable_pf)

                            # 2. Motion signals — already computed earlier for grasp detection.
                            #    Apply hysteresis dead-zones here for clean state borders.
                            obj_motion_pf = obj_motion_pf_pre
                            pc_motion_pf  = pc_motion_pf_pre
                            flow_moving_pf = hysteresis_filter(obj_motion_pf, low=1.0,  high=3.0)
                            pc_moving_pf   = hysteresis_filter(pc_motion_pf,  low=0.010, high=0.020)
                            is_moving_raw  = flow_moving_pf | pc_moving_pf

                            # 2b. Per-object rest-depth override.
                            #     bg_template is often 0 at the object's
                            #     pixel location (the object itself blocked
                            #     bg from being sampled), so we can't rely
                            #     on it for "is the object at its rest
                            #     position?" comparisons.  Instead, build a
                            #     per-pixel rest depth template from the
                            #     object's OWN clean frames (frames where
                            #     hand barely overlaps the obj mask) and
                            #     compare current depth against that.
                            #
                            #     For mc_is_static=True objects: rest_depth
                            #     captures the object's resting depth profile.
                            #     For occluded frames, pixels NOT covered by
                            #     hand should still show rest_depth → high
                            #     match → object is still at rest.
                            #
                            #     For mc_is_static=False objects (truly held):
                            #     skip the override; rely on raw flow / pc
                            #     motion signals.
                            DEPTH_TOL = 0.05
                            OCCL_THR = 0.05
                            bg_match_pf = np.zeros(total, dtype=np.float32)
                            if mc_is_static:
                                clean_frames = []
                                for _i in range(total):
                                    _m = obj_masks[_i]
                                    if _m is None or not _m.any():
                                        continue
                                    _h = (hand_masks_allframes[_i]
                                          if (hand_masks_allframes is not None
                                              and _i < len(hand_masks_allframes))
                                          else None)
                                    if _h is None:
                                        clean_frames.append(_i)
                                        continue
                                    _ovr = float((_m & _h).sum()) / float(max(_m.sum(), 1))
                                    if _ovr < OCCL_THR:
                                        clean_frames.append(_i)

                                if len(clean_frames) >= 3:
                                    _stk = []
                                    for _cf in clean_frames:
                                        _m = obj_masks[_cf]
                                        _d_masked = np.where(
                                            _m, depth_maps_stable[_cf], np.nan
                                        ).astype(np.float32)
                                        _stk.append(_d_masked)
                                    _stk = np.stack(_stk, axis=0)
                                    rest_depth = np.nanmedian(_stk, axis=0)
                                    rest_valid = ~np.isnan(rest_depth)
                                    del _stk
                                    for _i in range(total):
                                        _m = obj_masks_completed[_i]
                                        if _m is None or not _m.any():
                                            continue
                                        _vm = _m & rest_valid & (depth_maps_stable[_i] > 0.05)
                                        if not _vm.any():
                                            continue
                                        _diff = np.abs(
                                            depth_maps_stable[_i][_vm] - rest_depth[_vm]
                                        )
                                        bg_match_pf[_i] = float((_diff < DEPTH_TOL).mean())
                            bg_static_pf = bg_match_pf >= 0.6
                            is_moving_pf = is_moving_raw & ~bg_static_pf

                            # is_fast_pf still useful for GUI diagnostic
                            _, _, _, is_fast_pf = detect_static_moving_segments(
                                obj_motion_pf, pc_motion_per_frame=pc_motion_pf)
                            # Re-derive segs from the hysteresis is_moving
                            moving_segs = find_continuous_segments(is_moving_pf, min_len=3)
                            static_segs = find_continuous_segments(~is_moving_pf, min_len=3)

                            # 3. R_anchor (SAM3D canonical, kept stable)
                            if T_init_sim is not None:
                                R_canon = T_init_sim[:3, :3]
                                _s_baked = float(np.cbrt(max(abs(np.linalg.det(R_canon)), 1e-12)))
                                R_canon = R_canon / _s_baked
                                _U, _, _Vt = np.linalg.svd(R_canon)
                                R_canon = _U @ _Vt
                                if np.linalg.det(R_canon) < 0:
                                    _U[:, -1] *= -1
                                    R_canon = _U @ _Vt
                            else:
                                R_canon = ares.T[:3, :3]
                            # 3b. Per-frame obs cloud OBB (PCA), sign-corrected
                            #     against anchor frame.  Now wired into compute_depth_pose
                            #     with consecutive-trust gating + SLERP smoothing.
                            obs_obb_pf = compute_obs_obb_per_frame(
                                trust_res.obs_clouds, anchor_idx=int(ares.frame_idx))
                            R_obb_anchor = None
                            anchor_idx = int(ares.frame_idx)
                            if 0 <= anchor_idx < total:
                                _obb_a = obs_obb_pf[anchor_idx]
                                if _obb_a is not None and _obb_a.get('trustworthy'):
                                    R_obb_anchor = np.asarray(_obb_a['R'], dtype=np.float64)

                            # 4. Per-frame T_depth: t = obs bbox-center,
                            #    R = R_anchor + obs PCA SLERP (when trustworthy ≥ 3 consec).
                            T_depth_seq = compute_depth_pose_per_frame(
                                R_anchor=R_canon, mesh_pts=mesh_xyz,
                                obs_clouds=trust_res.obs_clouds,
                                fallback_pose_seq=tr.T_seq,
                                obs_obb_per_frame=obs_obb_pf,
                                R_obb_anchor=R_obb_anchor,
                                slerp_alpha=0.3,
                                require_trust_consecutive=3,
                            )
                            # 5. Per-frame T_wrist (rigid binding: R_obj = R_palm @ R_rel)
                            joints_pf_for_wrist = [
                                state['frame_data'][i].get('joints_3d_pred') or []
                                for i in range(total)
                            ]
                            T_wrist_seq, hd_diag = rigid_wrist_binding_propagation(
                                T_seq_in=tr.T_seq,
                                mesh_pts=mesh_xyz,
                                R_obj_anchor=R_canon,
                                obj_masks=obj_masks,
                                hand_masks=hand_masks_allframes,
                                obs_clouds=trust_res.obs_clouds,
                                pc_stable_per_frame=np.zeros(total, dtype=bool),  # 5-1 reset: post-grasp, use hand only for t (obs unreliable while held)
                                joints_per_frame=joints_pf_for_wrist,
                                hand_is_right_per_frame=hright_pf,
                                contact_soft=contact_soft,
                                K=K_mat,
                                min_seg_len=3,
                                soft_threshold=0.3,
                            )

                            # 6. Per-frame switch
                            #    WRIST = grasp (no longer gated by mask_small/pc_unstable)
                            #    The t-source decision (obs bbox vs palm offset)
                            #    happens INSIDE rigid_wrist_binding_propagation.
                            grasp_strong_pf = hysteresis_filter(
                                contact_soft.max(axis=1), low=0.2, high=0.4)
                            wrist_used_pf = grasp_strong_pf
                            T_final_seq = list(tr.T_seq)
                            n_wrist = n_static = n_tracked = 0
                            for i in range(total):
                                if wrist_used_pf[i]:
                                    T_final_seq[i] = np.asarray(T_wrist_seq[i],
                                                                 dtype=np.float64)
                                    n_wrist += 1
                                elif not is_moving_pf[i]:
                                    # placeholder; segment-locked below
                                    T_final_seq[i] = None
                                    n_static += 1
                                else:
                                    T_final_seq[i] = np.asarray(T_depth_seq[i],
                                                                 dtype=np.float64)
                                    n_tracked += 1
                            # 7. Static segments → lock pose.
                            #   • If a static segment is adjacent to a grasp
                            #     segment, inherit T_wrist at the grasp
                            #     boundary so the pose continues from where
                            #     the hand left it.  This avoids ICP basin
                            #     flips (e.g. a near-cylindrical bottle ends
                            #     up rotated 180° around its long axis when
                            #     released — ICP fits the obs cloud equally
                            #     well in either orientation).
                            #   • Truly static throughout (no adjacent grasp)
                            #     → fall back to T_depth median.
                            for (s, e) in static_segs:
                                idxs_static = [t for t in range(s, e+1) if not wrist_used_pf[t]]
                                if not idxs_static:
                                    continue
                                _s_idx = idxs_static[0]
                                _e_idx = idxs_static[-1]
                                lock_T = None
                                if _s_idx > 0 and wrist_used_pf[_s_idx - 1]:
                                    lock_T = np.asarray(
                                        T_wrist_seq[_s_idx - 1], dtype=np.float64
                                    ).copy()
                                elif _e_idx < total - 1 and wrist_used_pf[_e_idx + 1]:
                                    lock_T = np.asarray(
                                        T_wrist_seq[_e_idx + 1], dtype=np.float64
                                    ).copy()
                                if lock_T is None:
                                    seg_T_depth = [T_depth_seq[t] for t in idxs_static]
                                    lock_T = lock_pose_for_segment(
                                        seg_T_depth, 0, len(seg_T_depth) - 1, R_canon)
                                for t in idxs_static:
                                    T_final_seq[t] = lock_T.copy()
                            # 8. Any remaining None (shouldn't happen) → fallback to T_depth
                            for i in range(total):
                                if T_final_seq[i] is None:
                                    T_final_seq[i] = np.asarray(T_depth_seq[i],
                                                                 dtype=np.float64)
                            for i in range(total):
                                tr.T_seq[i] = T_final_seq[i]

                            path = (
                                "WRIST" if n_wrist > max(n_static, n_tracked)
                                else "DEPTH_STATIC" if n_static > n_tracked
                                else "DEPTH_TRACKED"
                            )
                            progress(
                                f"  obj {oid}: per-frame split → "
                                f"WRIST={n_wrist} STATIC={n_static} TRACKED={n_tracked} "
                                f"(majority={path}, motion median={float(np.median(obj_motion_pf)):.2f}px)")
                            is_held = (n_wrist + n_tracked) > 0   # any non-static frames
                            opt_res = None    # LBFGS retired in favour of explicit per-frame logic

                            # ── Stage 5: SavGol post-smooth (kill depth jitter) ──
                            from egoinfinity.pipeline.pose_tracker import smooth_se3_savgol
                            T_smooth = smooth_se3_savgol(tr.T_seq, win=11, poly=3)
                            for i in range(total):
                                tr.T_seq[i] = T_smooth[i]
                            progress(f"  obj {oid}: SavGol smoothed (win=11)")
                            _opt_data = {
                                'contact_soft': contact_soft_raw.tolist(),
                                'grasp_soft': grasp_soft.tolist(),
                                'is_held': bool(is_held),
                                'path': path,                                  # WRIST | DEPTH_STATIC | DEPTH_TRACKED (majority)
                                'contact_rate': float(n_contact / max(total, 1)),
                                'grasp_rate': float(grasp_rate),
                                'median_chamfer_m': float(mean_chamfer_m) if np.isfinite(mean_chamfer_m) else None,
                                'mean_iou_val': float(mean_iou_val),
                                'mask_completion_static': bool(mc_is_static),
                                'mask_completion_avg_iou': float(mc_diag.get('avg_iou', 0.0)),
                                'object_motion_per_frame': obj_motion_pf.tolist(),
                                'pc_motion_per_frame': pc_motion_pf.tolist(),
                                'is_moving_per_frame': is_moving_pf.astype(bool).tolist(),
                                'is_fast_per_frame': is_fast_pf.astype(bool).tolist(),
                                'obs_obb_trustworthy_per_frame': [
                                    bool(o is not None and o.get('trustworthy'))
                                    for o in obs_obb_pf
                                ],
                                # OBB geometry per frame (None when obs cloud
                                # too small).  Stored as plain lists for pkl.
                                'obs_obb_per_frame': [
                                    None if o is None else {
                                        'R': o['R'].tolist(),
                                        'center': o['center'].tolist(),
                                        'extents': o['extents'].tolist(),
                                        'eigvals': o['eigvals'].tolist(),
                                        'trustworthy': bool(o.get('trustworthy')),
                                    }
                                    for o in obs_obb_pf
                                ],
                                'mask_small_per_frame': mask_small_pf.astype(bool).tolist(),
                                'pc_unstable_per_frame': pc_unstable_pf.astype(bool).tolist(),
                                'wrist_used_per_frame': wrist_used_pf.astype(bool).tolist(),
                                'n_frames_wrist': int(n_wrist),
                                'n_frames_static': int(n_static),
                                'n_frames_tracked': int(n_tracked),
                            }
                        except Exception as _eo:
                            import traceback as _tb
                            progress(f"  obj {oid}: opt failed ({_eo})")
                            _tb.print_exc()
                            _opt_data = {}

                    bmin = mesh_xyz.min(0); bmax = mesh_xyz.max(0)
                    bcen = (bmin + bmax) / 2
                    bext = bmax - bmin
                    signs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], np.float32)
                    corners_canon = bcen + signs * bext / 2

                    for i in range(total):
                        sod = state['frame_data'][i].get('sam3_obj_data', {}).get(oid)
                        if sod is None:
                            continue
                        T_i = tr.T_seq[i]
                        R_i = T_i[:3, :3]
                        t_i = T_i[:3, 3]
                        sod['pose_R'] = R_i.astype(np.float32)
                        sod['pose_t'] = t_i.astype(np.float32)
                        sod['obb_corners'] = (corners_canon @ R_i.T + t_i).astype(np.float32)
                        sod['obb_extent'] = bext.astype(np.float32)

                    n_prop = int(sum(tr.propagated))
                    mean_ir = float(np.mean([ir for ir, p in zip(
                        tr.inlier_ratios, tr.propagated) if p])) if n_prop > 0 else 0.0
                    pose_track_info[oid] = {
                        'tracking_status': 'ok',
                        'anchor_frame': int(anchor_t),
                        'anchor_fitness': float(ares.fitness),
                        'anchor_rmse_mm': float(ares.inlier_rmse * 1000),
                        'scale_correction': float(ares.scale_correction),
                        'n_propagated': n_prop,
                        'mean_inlier_ratio': mean_ir,
                        'mesh_bbox_extent': bext.tolist(),
                        'mesh_bbox_center': bcen.tolist(),
                        **_trust_data,
                        **_opt_data,
                    }
                    progress(f"  obj {oid}: {n_prop}/{total} propagated, "
                             f"mean_inlier_ratio={mean_ir:.2f}")
                progress(f"Phase D-track done "
                         f"({sum(1 for v in pose_track_info.values() if v.get('tracking_status')=='ok')} "
                         f"tracked, {sum(1 for v in pose_track_info.values() if v.get('tracking_status')=='fail')} fallback)")
            except Exception as _e:
                import traceback
                progress(f"Phase D-track CRASHED: {_e}")
                traceback.print_exc()
        state['pose_track_info'] = pose_track_info
        if _RUN_TRACK and pose_track_info and args.cache_dir:
            try:
                from egoinfinity.pipeline import pipeline_state as _ps
                _ps.mark_done(args.cache_dir, "D-track")
            except Exception as _e:
                progress(f"Phase D-track state mark failed (non-fatal): {_e}")

        # ── Render flow_rgb with per-object motion overlay ──────────
        # Done AFTER Phase D-track so obj_motion_pf is available.  Each
        # frame's flow_rgb is a grayscale magnitude map (clip-global max
        # normalisation, no direction hue) with each detected object's
        # mask outlined in a colour that encodes its obj_motion_pf scalar
        # (grey = static, yellow = mid, red = moving).
        if pair_mag_list and total > 0:
            try:
                from egoinfinity.pipeline.pose_tracker.flow_viz import (
                    render_flow_with_motion_overlay,
                )
                _percentiles = [
                    float(np.percentile(m, 99)) for m in pair_mag_list
                    if m is not None and m.size > 0
                ]
                _global_max = max(_percentiles) + 1e-6 if _percentiles else 1.0
                _obj_motion_by_oid = {
                    oid: np.asarray(info.get('object_motion_per_frame', []),
                                    dtype=np.float32)
                    for oid, info in pose_track_info.items()
                    if info.get('object_motion_per_frame')
                }
                for i in range(total):
                    m_i = pair_mag_list[i] if i < len(pair_mag_list) else None
                    if m_i is None:
                        continue
                    overlays = []
                    sod_i = state['frame_data'][i].get('sam3_obj_data') or {}
                    for oid, sod in sod_i.items():
                        if not isinstance(sod, dict) or 'mask_packed' not in sod:
                            continue
                        H_, W_ = (int(x) for x in sod['mask_shape'])
                        mask = (np.unpackbits(sod['mask_packed'])[:H_*W_]
                                .reshape(H_, W_).astype(bool))
                        mot = _obj_motion_by_oid.get(oid)
                        v = float(mot[i]) if (mot is not None and i < len(mot)) else 0.0
                        overlays.append((mask, v))
                    flow_bgr = render_flow_with_motion_overlay(
                        m_i, _global_max, overlays, contour_thickness=2)
                    flow_rgb_arr = cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB)
                    _jpg_pending.append((
                        i, 'flow_rgb',
                        _jpg_pool.submit(_jpg_encode_rgb, flow_rgb_arr, 85),
                    ))
                progress("flow_rgb overlay rendered (magnitude + obj motion contours)")
            except Exception as _e:
                progress(f"flow_rgb overlay render failed: {_e}")

        frame_slider.max = total - 1
        progress_md.content = f"Done | {total} frames | focal={dp_focal:.0f}"
        state['ready'] = True
        setup_world_origin()
        _register_sam3d_meshes()
        update_frame(0)

        # Resolve background JPG encoding futures before persistence /
        # downstream consumers read frame_data['img_rgb'] etc.
        # Most futures complete during Phase D-track, so this is near-instant.
        if _jpg_pending:
            t0_jpg = time.time()
            for fi, key, fut in _jpg_pending:
                state['frame_data'][fi][key] = fut.result()
            _jpg_pool.shutdown(wait=True)
            progress(f"JPG encode resolved ({time.time() - t0_jpg:.2f}s wait)")

        # Save cache to disk
        if args.cache_dir:
            save_cache(args.cache_dir, state)

        progress("Processing complete")

    if args.load_cache:
        # Already loaded — just update viser GUI
        frame_slider.max = total - 1
        progress_md.content = f"Done | {total} frames | focal={state['dp_focal']:.0f}"
        setup_world_origin()
        _register_sam3d_meshes()
        update_frame(0)
        progress("Processing complete")
    elif args.no_viser:
        # Process, save cache, then exit
        run_processing()
        logging.info("Processing done, exiting (--no-viser)")
        server.stop()
        return
    else:
        # Start processing in background thread (viser serves while processing)
        threading.Thread(target=run_processing, daemon=True).start()

    # ── Callbacks ──────────────────────────────────────────────────
    for ctrl in [frame_slider, show_mesh, show_pred, show_object, show_obj_pc, show_sam3_obj_pc, show_sam3d_mesh, show_sam3d_init, show_sam3_obb, show_trajectories, traj_length, show_debug_colors, show_depth_pc, show_depth_overlay, show_flow_overlay, flow_alpha, pc_step]:
        ctrl.on_update(lambda _: update_frame(frame_slider.value) if state['ready'] else None)

    playing = [False]

    @play_btn.on_click
    def _on_play(_):
        playing[0] = not playing[0]
        # While playing, force-collapse Display + Status. When paused, leave
        # whatever the user has open — they may want to inspect signals.
        if playing[0]:
            try:
                display_folder.expand_by_default = False
                info_folder.expand_by_default = False
            except Exception as e:
                print(f"[viser] folder collapse on play failed: {e}")

    try:
        while True:
            if playing[0] and state['ready']:
                nxt = (frame_slider.value + 1) % total
                frame_slider.value = nxt
                update_frame(nxt)
                time.sleep(1.0 / 15.0)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info("\nExiting.")


if __name__ == '__main__':
    main()
