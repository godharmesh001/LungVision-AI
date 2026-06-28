"""
Project : Lung Cancer AI
Module  : Configuration
Author  : Dharmesh Kumar

Purpose:
Central configuration file used across the entire project.
"""

from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DATASET = Path("/Users/dharmeshkumar/Downloads/archive (5)")

DATASET_DIR = PROJECT_ROOT / "dataset"

REPORTS_DIR = PROJECT_ROOT / "reports"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# =====================================================
# DATASET SETTINGS
# =====================================================

CLASSES = [
    "adenocarcinoma",
    "benign",
    "squamous_cell_carcinoma"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_SIZE = (224, 224)

# =====================================================
# CLASS LABELS
# =====================================================

CLASS_TO_INDEX = {
    "adenocarcinoma": 0,
    "benign": 1,
    "squamous_cell_carcinoma": 2,
}

INDEX_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_INDEX.items()
}

# =====================================================
# TRAINING SETTINGS
# =====================================================

BATCH_SIZE = 32

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MODEL_NAME = "resnet50"

NUM_CLASSES = 3

SAVE_BEST_ONLY = True