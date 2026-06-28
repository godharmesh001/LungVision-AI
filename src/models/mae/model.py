"""
Project : LungVision-AI
Module  : MAE Backbone
"""

import torch
import torch.nn as nn
from transformers import ViTMAEModel


class MAEBackbone(nn.Module):

    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        freeze=True,
    ):
        super().__init__()

        self.freeze = freeze

        self.model = ViTMAEModel.from_pretrained(
            model_name
        )

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):

        if self.freeze:

            with torch.no_grad():

                outputs = self.model(
                    pixel_values=x
                )

        else:

            outputs = self.model(
                pixel_values=x
            )

        return outputs.last_hidden_state[:, 1:, :]
