"""
Project : LungVision-AI
Module  : Cross Attention

Purpose:
Fuse CNN and MAE features using Multi-Head Cross Attention.
"""

import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        cnn_features,
        mae_features,
    ):
        """
        Query  : CNN
        Key    : MAE
        Value  : MAE
        """

        fused, attention_weights = self.attention(
            query=cnn_features,
            key=mae_features,
            value=mae_features,
        )

        fused = self.norm(fused + cnn_features)

        return fused, attention_weights
