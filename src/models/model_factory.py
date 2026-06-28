"""
Project : LungVision-AI
Module  : Model Factory

Purpose:
Centralized factory for all supported model architectures.
"""

from src.models.resnet50 import create_model as create_resnet50
from src.models.vit import VisionTransformer


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
):
    """
    Create and return the requested model.
    """

    model_name = model_name.lower()

    # -------------------------------------------------
    # ResNet50
    # -------------------------------------------------
    if model_name == "resnet50":
        return create_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    # -------------------------------------------------
    # Vision Transformer
    # -------------------------------------------------
    elif model_name == "vit_base_patch16_224":
        return VisionTransformer(
            model_name="vit_base_patch16_224",
            num_classes=num_classes,
            pretrained=pretrained,
        )

    # -------------------------------------------------
    # Unsupported
    # -------------------------------------------------
    raise ValueError(
        f"Unsupported model architecture: {model_name}"
    )
