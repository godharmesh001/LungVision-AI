"""
Project : LungVision-AI
Module  : CSV Logger

Purpose:
Logs training metrics into a CSV file.
"""

import csv
from pathlib import Path

import src.utils.config as config


class CSVLogger:

    def __init__(self, filename="training_log.csv"):

        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.filepath = config.LOGS_DIR / filename

        if not self.filepath.exists():

            with open(self.filepath, "w", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Epoch",
                    "Train Loss",
                    "Train Accuracy",
                    "Validation Loss",
                    "Validation Accuracy",
                    "Learning Rate",
                ])

    def log(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        lr,
    ):

        with open(self.filepath, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                lr,
            ])