"""
Project : LungVision-AI
Module  : Model Factory
"""

from src.models.resnet50 import create_model as create_resnet50
from src.models.vit import VisionTransformer
from src.models.dual_encoder import DualEncoder


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
):

    model_name = model_name.lower()

    # --------------------------------------------------
    # ResNet50
    # --------------------------------------------------
    if model_name == "resnet50":

        return create_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    # --------------------------------------------------
    # Vision Transformer
    # --------------------------------------------------
    elif model_name == "vit_base_patch16_224":

        return VisionTransformer(
            model_name="vit_base_patch16_224",
            num_classes=num_classes,
            pretrained=pretrained,
        )

    # --------------------------------------------------
    # Dual Encoder
    # --------------------------------------------------
    elif model_name == "dual_encoder":

        return DualEncoder(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unsupported model architecture: {model_name}"
    )
