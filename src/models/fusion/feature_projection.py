"""
Project : LungVision-AI
Module  : Feature Projection

Purpose:
Project CNN feature maps into transformer token embeddings.
"""

import torch.nn as nn


class FeatureProjection(nn.Module):

    def __init__(
        self,
        in_channels=2048,
        embed_dim=768,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=1,
        )

    def forward(self, x):

        x = self.projection(x)

        b, c, h, w = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x
