"""Video streaming: yt-dlp download + ffmpeg local frame extraction."""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Use imageio_ffmpeg's bundled binary if system ffmpeg not found
try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    _FFPROBE = str(Path(_FFMPEG).parent / "ffprobe")
    if not Path(_FFPROBE).exists():
        _FFPROBE = "ffprobe"
except ImportError:
    _FFMPEG = "ffmpeg"
    _FFPROBE = "ffprobe"

# yt-dlp path: prefer conda env copy
import shutil
_YTDLP = shutil.which("yt-dlp") or "yt-dlp"

# Ensure ~/.deno/bin is in PATH for yt-dlp JS challenge solving
_deno_bin = os.path.expanduser("~/.deno/bin")
if os.path.isdir(_deno_bin) and _deno_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _deno_bin + os.pathsep + os.environ.get("PATH", "")


def download_video(video_id: str, timeout: int = 300) -> str | None:
    """Download lowest quality video to a temp file via yt-dlp.

    Returns path to temp file, or None on failure.
    Caller is responsible for deleting the file.
    """
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, f"a100m_{video_id}.mp4")

    # Skip if already downloaded (e.g. retry)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path

    cmd = [
        _YTDLP,
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", "worst[height>=360][ext=mp4]/worst[ext=mp4]/best[ext=mp4]/worst/best",
        "--js-runtimes", "deno",
        "--cookies-from-browser", "firefox",
        "--no-warnings",
        "-q",
        "-o", out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_path
        # Clean up failed download
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    except subprocess.TimeoutExpired:
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    except Exception:
        return None


def _probe_dimensions(path: str, max_height: int, timeout: int = 10) -> tuple[int, int] | None:
    """Use ffprobe to get source frame dimensions, then compute scale output."""
    cmd = [
        _FFPROBE,
        "-v", "error",
        "-i", path,
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        info = json.loads(r.stdout)
        streams = info.get("streams", [])
        if not streams:
            return None
        w = streams[0]["width"]
        h = streams[0]["height"]
        if h > max_height:
            new_h = max_height
            new_w = int(w * max_height / h)
            new_w = new_w + (new_w % 2)  # ensure even
        else:
            new_w, new_h = w, h
        return new_w, new_h
    except Exception:
        return None


def sample_frames_local(
    video_path: str,
    start: float,
    end: float,
    n_frames: int = 5,
    max_height: int = 360,
    timeout: int = 10,
    _cached_dims: dict | None = None,
) -> list[np.ndarray]:
    """Extract n_frames from a LOCAL video file. Fast local seeking.

    Returns list of BGR numpy arrays.
    """
    duration = end - start
    if duration <= 0:
        return []

    # Get dimensions (cache across calls for same video)
    if _cached_dims and "dims" in _cached_dims:
        width, height = _cached_dims["dims"]
    else:
        dims = _probe_dimensions(video_path, max_height, timeout=timeout)
        if dims is None:
            width, height = 640, 360
        else:
            width, height = dims
        if _cached_dims is not None:
            _cached_dims["dims"] = (width, height)

    target_fps = max(1.0, n_frames / duration * 1.2)

    cmd = [
        _FFMPEG,
        "-ss", f"{start:.2f}",
        "-t", f"{duration:.2f}",
        "-i", video_path,
        "-vf", f"fps={target_fps:.2f},scale=-2:{max_height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0 or not r.stdout:
        return []

    raw = r.stdout
    frame_bytes = width * height * 3
    if len(raw) < frame_bytes:
        # Try common widths
        for w_try in [640, 480, 426, 320]:
            fb = w_try * max_height * 3
            if len(raw) >= fb and len(raw) % fb == 0:
                width, height = w_try, max_height
                frame_bytes = fb
                break
        else:
            return []

    n_total = len(raw) // frame_bytes
    frames = []
    for i in range(n_total):
        buf = raw[i * frame_bytes : (i + 1) * frame_bytes]
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        frames.append(frame)

    if len(frames) > n_frames:
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return frames


def sample_motion_pairs(
    video_path: str,
    start: float,
    end: float,
    n_positions: int = 3,
    pair_gap: float = 0.1,
    max_height: int = 360,
    timeout: int = 10,
    _cached_dims: dict | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract pairs of closely-spaced frames for background motion detection.

    Samples n_positions evenly across the segment; at each position extracts
    two frames separated by pair_gap seconds (~0.1s).  Short gap ensures
    optical-flow matching works reliably.

    Returns list of (gray1, gray2) uint8 pairs.
    """
    import cv2 as _cv2

    duration = end - start
    if duration <= 0:
        return []

    # Get dimensions (cache across calls for same video)
    if _cached_dims and "dims" in _cached_dims:
        width, height = _cached_dims["dims"]
    else:
        dims = _probe_dimensions(video_path, max_height, timeout=timeout)
        if dims is None:
            width, height = 640, 360
        else:
            width, height = dims
        if _cached_dims is not None:
            _cached_dims["dims"] = (width, height)

    # Pick timestamps: n_positions evenly spaced, each producing 2 frames
    timestamps: list[float] = []
    for i in range(n_positions):
        t = start + duration * (i + 1) / (n_positions + 1)
        t = min(t, end - pair_gap - 0.01)  # ensure second frame is within segment
        timestamps.append(t)
        timestamps.append(t + pair_gap)

    # Extract all frames in one ffmpeg call using select filter
    # Build select expression: pick the frame closest to each timestamp
    select_expr = "+".join(
        f"between(t,{t-0.02:.3f},{t+0.02:.3f})" for t in timestamps
    )
    cmd = [
        _FFMPEG,
        "-ss", f"{timestamps[0] - 0.05:.2f}",
        "-t", f"{timestamps[-1] - timestamps[0] + 0.2:.2f}",
        "-i", video_path,
        "-vf", f"select='{select_expr}',scale=-2:{max_height}",
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0 or not r.stdout:
        return []

    raw = r.stdout
    frame_bytes = width * height * 3

    # Try to recover actual width if mismatch
    if len(raw) < frame_bytes:
        for w_try in [640, 480, 426, 320]:
            fb = w_try * max_height * 3
            if len(raw) >= fb and len(raw) % fb == 0:
                width, height = w_try, max_height
                frame_bytes = fb
                break
        else:
            return []

    n_total = len(raw) // frame_bytes
    all_frames = []
    for i in range(n_total):
        buf = raw[i * frame_bytes : (i + 1) * frame_bytes]
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        all_frames.append(frame)

    # Pair them up
    pairs = []
    for i in range(0, len(all_frames) - 1, 2):
        g1 = _cv2.cvtColor(all_frames[i], _cv2.COLOR_BGR2GRAY)
        g2 = _cv2.cvtColor(all_frames[i + 1], _cv2.COLOR_BGR2GRAY)
        pairs.append((g1, g2))

    return pairs


def detect_shot_cuts(
    video_path: str,
    start: float,
    end: float,
    threshold: float = 0.4,
    timeout: int = 15,
) -> list[float]:
    """Detect shot boundaries using ffmpeg's scene change filter.

    Runs ffmpeg with ``select='gt(scene,threshold)',showinfo`` which detects
    hard cuts frame-by-frame during decode — no frames are piped to Python.

    Returns list of absolute timestamps where cuts occur.
    """
    import re

    duration = end - start
    if duration <= 0:
        return []

    cmd = [
        _FFMPEG,
        "-ss", f"{start:.2f}",
        "-t", f"{duration:.2f}",
        "-i", video_path,
        "-an",
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []

    # Parse showinfo output regardless of returncode — ffmpeg may return
    # non-zero for non-critical reasons but the scene detection is still valid.
    # Any match with pts_time > 0.1s is a real cut (skip first-frame artifact).
    times = re.findall(r"pts_time:([\d.]+)", r.stderr)
    cut_rel = [float(t) for t in times if float(t) > 0.1]
    return [start + t for t in cut_rel]
