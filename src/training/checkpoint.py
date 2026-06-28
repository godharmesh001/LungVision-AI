"""
Project : LungVision-AI
Module  : Checkpoint Manager
"""

from pathlib import Path
import torch
import src.utils.config as config


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
    filename,
):
    config.CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "model_name": config.MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    save_path = Path(config.CHECKPOINTS_DIR) / filename

    torch.save(
        checkpoint,
        save_path,
    )

    print(f"✓ Checkpoint saved -> {save_path}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    filename,
):
    checkpoint_path = Path(config.CHECKPOINTS_DIR) / filename

    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    print(f"✓ Loaded checkpoint: {checkpoint_path}")

    return checkpoint
