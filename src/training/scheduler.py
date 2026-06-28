"""
Project : LungVision-AI
Module  : Learning Rate Scheduler
"""

import torch


def create_scheduler(optimizer):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    return scheduler