"""
Project : LungVision-AI
Module  : Checkpoint Manager

Purpose:
Save and load model checkpoints.
"""

from pathlib import Path
import torch

from src.utils.config import CHECKPOINTS_DIR


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    filename="best_model.pth",
):
    """
    Save training checkpoint.
    """

    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(
        checkpoint,
        CHECKPOINTS_DIR / filename,
    )

    print(f"✅ Checkpoint saved: {filename}")


def load_checkpoint(
    model,
    optimizer,
    filename="best_model.pth",
):
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(
        CHECKPOINTS_DIR / filename,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint