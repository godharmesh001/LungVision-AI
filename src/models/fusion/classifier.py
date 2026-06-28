"""
Project : LungVision-AI
Module  : Classification Head

Purpose:
Convert fused token embeddings into class predictions.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):

        # x : (B, Tokens, Embedding)

        x = x.transpose(1, 2)

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.classifier(x)
