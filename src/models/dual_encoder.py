"""
Project : LungVision-AI
Module  : Dual Encoder Network

Purpose:
Fuse CNN local features and MAE global features for
lung cancer histopathology classification.
"""

import torch.nn as nn

from src.models.cnn.backbone import CNNBackbone
from src.models.mae.model import MAEBackbone
from src.models.fusion.feature_projection import FeatureProjection
from src.models.fusion.attention_fusion import TransformerFusionBlock
from src.models.fusion.classifier import ClassificationHead


class DualEncoder(nn.Module):

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
    ):
        super().__init__()

        self.cnn = CNNBackbone(
            pretrained=pretrained,
        )

        self.mae = MAEBackbone()

        self.projection = FeatureProjection()

        self.fusion = TransformerFusionBlock()

        self.classifier = ClassificationHead(
            num_classes=num_classes,
        )

    def forward(self, x):

        # Local features
        cnn = self.cnn(x)

        cnn = self.projection(cnn)

        # Global features
        mae = self.mae(x)

        # Cross-attention fusion
        fused, _ = self.fusion(
            cnn,
            mae,
        )

        # Classification
        logits = self.classifier(fused)

        return logits
