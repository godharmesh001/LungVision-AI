"""
Project : LungVision-AI
Module  : Model Evaluation

Purpose:
Evaluate a trained model on the test dataset.
"""

import torch

import src.utils.config as config

from src.utils.device import get_device
from src.data.dataloader import create_dataloaders
from src.models.model_factory import create_model
from src.training.checkpoint import load_checkpoint

from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import save_confusion_matrix


def main():

    print("=" * 60)
    print("LungVision-AI Evaluation")
    print("=" * 60)

    device = get_device()

    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False,
    )

    checkpoint = load_checkpoint(
        model=model,
        filename=config.BEST_CHECKPOINT_NAME,
    )

    if checkpoint is None:
        raise FileNotFoundError(
            "No trained checkpoint found."
        )

    epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_accuracy"]

    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from Epoch {epoch}")
    print(f"Best Validation Accuracy : {best_acc*100:.2f}%")

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    results = compute_metrics(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall   : {results['recall']*100:.2f}%")
    print(f"F1 Score : {results['f1']*100:.2f}%")

    print("\nClassification Report\n")
    print(results["report"])

    save_confusion_matrix(
        y_true,
        y_pred,
        config.CLASSES,
        config.REPORTS_DIR,
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        config.REPORTS_DIR / "classification_report.txt",
        "w",
    ) as f:

        f.write(results["report"])

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()
