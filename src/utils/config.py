"""
Project : LungVision-AI
Module  : Configuration

Purpose:
Central configuration used across the entire project.
Works on both local machines and Google Colab.
"""

from pathlib import Path
import os

# =====================================================
# ENVIRONMENT DETECTION
# =====================================================

IS_COLAB = "COLAB_GPU" in os.environ

# =====================================================
# PROJECT ROOT
# =====================================================

if IS_COLAB:
    PROJECT_ROOT = Path("/content/LungVision-AI")
    DRIVE_ROOT = Path("/content/drive/MyDrive/LungVisionAI")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DRIVE_ROOT = PROJECT_ROOT

# =====================================================
# DATASET
# =====================================================

DATASET_DIR = DRIVE_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

CHECKPOINTS_DIR = DRIVE_ROOT / "checkpoints"

OUTPUTS_DIR = DRIVE_ROOT / "outputs"

REPORTS_DIR = DRIVE_ROOT / "reports"

LOGS_DIR = DRIVE_ROOT / "logs"

TENSORBOARD_DIR = DRIVE_ROOT / "tensorboard"

EXPERIMENTS_DIR = DRIVE_ROOT / "experiments"

# =====================================================
# DATASET SETTINGS
# =====================================================

CLASSES = [
    "adenocarcinoma",
    "benign",
    "squamous_cell_carcinoma",
]

CLASS_TO_INDEX = {
    "adenocarcinoma": 0,
    "benign": 1,
    "squamous_cell_carcinoma": 2,
}

INDEX_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_INDEX.items()
}

NUM_CLASSES = len(CLASSES)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_SIZE = (224, 224)

# =====================================================
# TRAINING SETTINGS
# =====================================================

EARLY_STOPPING_PATIENCE = 5

MIN_LEARNING_RATE = 1e-6

LR_FACTOR = 0.5

LR_PATIENCE = 2

BATCH_SIZE = 32

# Development configuration
NUM_EPOCHS = 5

# Final baseline
# NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

SAVE_BEST_ONLY = True

# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "resnet50"

PRETRAINED = True