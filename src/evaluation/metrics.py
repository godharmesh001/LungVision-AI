"""
Project : LungVision-AI
Module  : Evaluation Metrics

Purpose:
Provides reusable evaluation metrics for classification models.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "report": classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        ),
    }

    return results
