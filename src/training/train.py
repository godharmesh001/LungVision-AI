"""
Project : LungVision-AI
Module  : Training Entry Point
"""

import torch

import src.utils.config as config

from src.utils.device import get_device
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.optimizers.optimizer_factory import create_optimizer
from src.training.trainer import Trainer
from src.training.csv_logger import CSVLogger
from src.training.scheduler import create_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def main():

    print("=" * 60)
    print("LungVision-AI Training")
    print("=" * 60)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        model_name=config.MODEL_NAME,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = create_scheduler(optimizer)

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    logger = CSVLogger()

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE
    )

    start_epoch = 0
    best_accuracy = 0.0

    # --------------------------------------------------
    # Resume Training
    # --------------------------------------------------

    if config.RESUME_TRAINING:

        checkpoint = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=config.LAST_CHECKPOINT_NAME,
        )

        if checkpoint is not None:

            start_epoch = checkpoint["epoch"]
            best_accuracy = checkpoint["best_accuracy"]

            print()
            print("✓ Resuming Training")
            print(f"Starting Epoch : {start_epoch + 1}")
            print(f"Best Accuracy  : {best_accuracy*100:.2f}%")
            print()

    print("\nPipeline initialized successfully.\n")

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(start_epoch, config.NUM_EPOCHS):

        print("=" * 60)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        print("=" * 60)

        train_loss, train_acc = trainer.train_one_epoch(
            train_loader
        )

        val_loss, val_acc = trainer.validate_one_epoch(
            val_loader
        )

        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        scheduler.step(val_acc)

        # ----------------------------------------------
        # Save Best Model
        # ----------------------------------------------

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
                filename=config.BEST_CHECKPOINT_NAME,
            )

            print("✅ Best model updated.")

        # ----------------------------------------------
        # Save Last Checkpoint
        # ----------------------------------------------

        if config.CHECKPOINT_EVERY_EPOCH:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
                filename=config.LAST_CHECKPOINT_NAME,
            )

        # ----------------------------------------------
        # Early Stopping
        # ----------------------------------------------

        if early_stopping(val_acc):

            print("\n🛑 Early stopping triggered.")
            break

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc*100:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc*100:.2f}%")
        print(f"Best Acc   : {best_accuracy*100:.2f}%")
        print(f"LR         : {current_lr:.7f}")
        print()

    print("=" * 60)
    print("Training Finished Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()