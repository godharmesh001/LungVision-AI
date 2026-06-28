"""
Project : LungVision-AI
Module  : Checkpoint Manager
"""

import torch
import src.utils.config as config


def save_checkpoint(
    model,
    optimizer,
    epoch,
    accuracy,
    filename="best_model.pth",
):

    config.CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "accuracy": accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        config.CHECKPOINTS_DIR / filename,
    )

    print(f"✓ Checkpoint saved -> {filename}")


def load_checkpoint(
    model,
    optimizer,
    filename="best_model.pth",
):

    checkpoint = torch.load(
        config.CHECKPOINTS_DIR / filename,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        checkpoint["epoch"],
        checkpoint["accuracy"],
    )