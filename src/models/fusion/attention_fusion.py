"""
Project : LungVision-AI
Module  : Transformer Fusion Block

Purpose:
Fuse CNN and MAE features using Cross Attention followed by
a Transformer-style Feed Forward Network.
"""

import torch.nn as nn

from src.models.fusion.cross_attention import CrossAttention


class TransformerFusionBlock(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        hidden = embed_dim * mlp_ratio

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        cnn_tokens,
        mae_tokens,
    ):

        fused, attention = self.cross_attention(
            cnn_tokens,
            mae_tokens,
        )

        output = self.norm(
            fused + self.ffn(fused)
        )

        return output, attention
