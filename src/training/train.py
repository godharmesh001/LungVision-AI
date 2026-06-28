"""
Project : LungVision-AI
Module  : Training Entry Point

Purpose:
Main entry point for training baseline models.
"""

import torch

import src.utils.config as config

from src.utils.device import get_device
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer


def main():
    """Main training entry point."""

    print("=" * 60)
    print("LungVision-AI Training")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE,
    )

    # Model
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
    )

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    print("\nPipeline initialized successfully.\n")

    # ==============================
    # Training Loop
    # ==============================

        # ==============================
    # Training Loop
    # ==============================

    for epoch in range(config.NUM_EPOCHS):

        print("=" * 60)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("=" * 60)

        train_loss, train_acc = trainer.train_one_epoch(
            train_loader
        )

        val_loss, val_acc = trainer.validate_one_epoch(
            val_loader
        )

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc * 100:.2f}%")

        print()

        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Training Finished Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

