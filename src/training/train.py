"""
Project : LungVision-AI
Module  : Training Entry Point
"""

import torch

import src.utils.config as config

from src.utils.device import get_device
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.csv_logger import CSVLogger
from src.training.scheduler import create_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import save_checkpoint


def main():

    print("=" * 60)
    print("LungVision-AI Training")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    # Model
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
    )

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = create_scheduler(optimizer)

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE
    )

    # CSV Logger
    logger = CSVLogger()

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    best_accuracy = 0.0

    print("\nPipeline initialized successfully.\n")

    for epoch in range(config.NUM_EPOCHS):

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

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                accuracy=val_acc,
                filename="best_model.pth",
            )

            print("✅ Best model updated.")

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