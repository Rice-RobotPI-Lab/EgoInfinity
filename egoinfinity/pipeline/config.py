"""
Pipeline configuration — all paths and hyperparameters in one place.

All paths are relative to the EgoInfinity repo root.
"""
import os

# ── Repo root ─────────────────────────────────────────────────────
REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ── Pretrained model directory ───────────────────────────────────
# Override via env var EGOINFINITY_CKPT_DIR (e.g. point at a shared NVMe path);
# default = `<repo>/pretrained_models` populated by scripts/setup_weights.sh.
CKPT_DIR = os.environ.get(
    'EGOINFINITY_CKPT_DIR',
    os.path.join(REPO_ROOT, 'pretrained_models'),
)

# ── WiLoR ─────────────────────────────────────────────────────────
WILOR_CHECKPOINT = os.path.join(CKPT_DIR, 'wilor_final.ckpt')
DETECTOR_PATH    = os.path.join(CKPT_DIR, 'detector.pt')
MANO_DIR         = os.path.join(REPO_ROOT, 'third_party', 'wilor', 'mano_data')

# WiLoR model config (small YAML, version-controlled in configs/).
# Fallback to legacy in-CKPT_DIR location for setups that haven't migrated.
_WILOR_CFG_NEW = os.path.join(REPO_ROOT, 'configs', 'wilor_model_config.yaml')
_WILOR_CFG_OLD = os.path.join(CKPT_DIR, 'model_config.yaml')
WILOR_CFG = _WILOR_CFG_NEW if os.path.isfile(_WILOR_CFG_NEW) else _WILOR_CFG_OLD

# ── Detection ─────────────────────────────────────────────────────
DETECTOR_CONF   = 0.3      # YOLO confidence threshold
RESCALE_FACTOR  = 2.0      # bbox rescale for WiLoR crop

# ── MoGe-2 ───────────────────────────────────────────────────────
MOGE2_MODEL = 'Ruicheng/moge-2-vitl'

# ── SAM2 ──────────────────────────────────────────────────────────
SAM2_DIR = os.path.join(REPO_ROOT, 'third_party', 'sam2')
SAM2_CFG = 'sam2.1/sam2.1_hiera_s.yaml'
SAM2_CHECKPOINT = os.path.join(CKPT_DIR, 'sam2.1_hiera_small.pt')

# ── MANO joint indices ───────────────────────────────────────────
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
WRIST_INDEX = 0

# ── Hand skeleton edges (MANO 21-joint) ──────────────────────────
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
]

# ── Motion Infiller (HaWoR) ──────────────────────────────────────
INFILLER_CHECKPOINT = os.path.join(CKPT_DIR, 'infiller.pt')

# ── SAM3 (text-prompted detection, runs in separate conda env) ───
#   SAM3_PYTHON : Python binary inside the sam3 conda env
#   SAM3_REPO   : Root of the cloned sam3 repo (for bpe_path etc.)
#   Override via environment variables if your layout differs.
SAM3_PYTHON = os.environ.get(
    'SAM3_PYTHON',
    os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'sam3', 'bin', 'python'))
SAM3_REPO = os.environ.get(
    'SAM3_REPO',
    os.path.realpath(os.path.join(REPO_ROOT, '..', 'sam3')))
SAM3_CLI = os.path.join(REPO_ROOT, 'scripts', 'sam3_detect_cli.py')
SAM3_BPE_PATH = os.path.join(SAM3_REPO, 'sam3', 'assets', 'bpe_simple_vocab_16e6.txt.gz')

# ── GeoCalib (gravity estimation) ────────────────────────────────
GEOCALIB_WEIGHTS = 'pinhole'       # 'pinhole' or 'distorted'
GEOCALIB_SAMPLE_FRAMES = 3         # number of frames to sample for robust estimate

# ── Ego camera estimation ────────────────────────────────────────
EGO_D_BACK       = 0.35   # distance behind workspace (meters)
EGO_D_UP         = 0.40   # distance above workspace (meters)
EGO_FOV_MARGIN   = 1.3    # FOV multiplier
EGO_HALF_SHOULDER = 0.18  # half shoulder width (meters)
