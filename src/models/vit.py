"""
Project : LungVision-AI
Module  : Vision Transformer (ViT)

Purpose:
Provides a ViT-Base backbone for image classification.
"""

import timm
import torch.nn as nn


class VisionTransformer(nn.Module):
    """
    Vision Transformer wrapper using timm.
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        model_name="vit_base_patch16_224",
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
