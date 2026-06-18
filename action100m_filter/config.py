"""Visual filter configuration."""
import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_db_path() -> str:
    """Resolve default DB path.

    Resolution order:
      1. ACTION100M_DB env var (canonical override)
      2. <repo>/data/action100m_index.db (release-user friendly default)
    """
    env = os.environ.get("ACTION100M_DB")
    if env:
        return env
    return str(_repo_root() / "data" / "action100m_index.db")


@dataclass
class FilterConfig:
    # Database
    db_path: str = ""

    # Hand detector
    detector_path: str = ""             # YOLO hand detector .pt
    hand_conf: float = 0.3              # YOLO confidence threshold
    trunc_edge_px: int = 5              # bbox within N px of border = truncated

    # Judgment thresholds
    min_hand_ratio: float = 0.75        # >=75% frames must have a hand
    min_hand_size: float = 0.005        # hand bbox >= 0.5% of frame
    max_hand_size: float = 0.40         # hand bbox <= 40% of frame
    max_trunc_ratio: float = 0.5        # <=50% frames with truncated hands
    max_bg_flow: float = 2.0            # optical flow px/frame (P20 over 4x4 grid)

    # Shot cut detection
    scene_threshold: float = 0.4        # ffmpeg scene filter threshold
    min_subseg_duration: float = 5.0    # longest no-cut sub-segment must be >= 5s

    # ffmpeg / yt-dlp
    max_height: int = 360
    n_frames: int = 8
    ffmpeg_timeout: int = 20
    ytdlp_timeout: int = 30

    # Concurrency
    n_workers: int = 1

    # Test mode
    test_n_videos: int = 500

    def __post_init__(self):
        if not self.db_path:
            self.db_path = _default_db_path()
        if not self.detector_path:
            self.detector_path = str(_repo_root() / "pretrained_models" / "detector.pt")
