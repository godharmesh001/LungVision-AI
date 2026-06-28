"""
Project : LungVision-AI
Module  : ROC Curve

Purpose:
Generate multiclass ROC curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def save_roc_curve(
    y_true,
    y_score,
    class_names,
    save_dir,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = label_binarize(
        y_true,
        classes=range(len(class_names)),
    )

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(
            y_true[:, i],
            y_score[:, i],
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_dir / "roc_curve.png",
        dpi=300,
    )

    plt.close()
