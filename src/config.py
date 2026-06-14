"""Single source of truth for the shared pipeline.

Keeping clip sampling, resolution and normalization here (and importing them
everywhere) is what guarantees every model sees IDENTICAL inputs -- the basis
for a fair accuracy-vs-efficiency comparison.
"""
from pathlib import Path

# --- shared clip sampling (identical for every model => fair comparison) ---
NUM_FRAMES = 16        # frames sampled uniformly per clip (VideoMAE-base requires 16)
IMG_SIZE = 224         # spatial resolution fed to every model
MEAN = [0.485, 0.456, 0.406]   # ImageNet normalization (matches Kinetics-pretrained backbones)
STD = [0.229, 0.224, 0.225]

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

# number of action classes per dataset
DATASETS = {
    "ucf101": {"classes": 101},
    "hmdb51": {"classes": 51},
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
