"""
Project : LungVision-AI
Module  : CNN Backbone

Purpose:
Extract local visual features for the dual-encoder architecture.
"""

import torch.nn as nn
import timm


class CNNBackbone(nn.Module):
    """
    CNN feature extractor using timm.
    """

    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4],
        )

    def forward(self, x):
        """
        Returns:
            Final feature map:
            (B, C, H, W)
        """
        features = self.backbone(x)

        return features[-1]
