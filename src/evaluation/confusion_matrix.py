"""
Project : LungVision-AI
Module  : Confusion Matrix

Purpose:
Generate and save a confusion matrix figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_dir,
):
    """
    Generate and save confusion matrix.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        save_dir / "confusion_matrix.png",
        dpi=300,
    )

    plt.close(fig)
