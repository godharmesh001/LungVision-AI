"""
Project : LungVision-AI
Module  : Model Factory

Purpose:
Centralized model creation for all supported architectures.
"""

from src.models.resnet50 import create_model as create_resnet50


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
):
    """
    Create and return the requested model.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.

    Returns:
        torch.nn.Module
    """

    model_name = model_name.lower()

    if model_name == "resnet50":
        return create_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unsupported model: {model_name}")