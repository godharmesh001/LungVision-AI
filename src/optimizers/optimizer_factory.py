"""
Project : LungVision-AI
Module  : Optimizer Factory
"""

import torch


def create_optimizer(model, model_name, learning_rate, weight_decay):

    model_name = model_name.lower()

    trainable_params = filter(
        lambda p: p.requires_grad,
        model.parameters(),
    )

    if model_name == "resnet50":
        return torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif model_name in [
        "vit_base_patch16_224",
        "dual_encoder",
    ]:
        return torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer for {model_name}"
    )
