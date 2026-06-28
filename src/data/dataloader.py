"""
Project : LungVision-AI
Module  : DataLoader

Purpose:
Creates reusable PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

import src.utils.config as config

from src.data.dataset import LungCancerDataset
from src.data.preprocessing import (
    get_train_transforms,
    get_test_transforms,
)


def create_dataloaders(batch_size=config.BATCH_SIZE):

    train_dataset = LungCancerDataset(
        dataset_dir=config.TRAIN_DIR,
        transform=get_train_transforms(),
    )

    val_dataset = LungCancerDataset(
        dataset_dir=config.VAL_DIR,
        transform=get_test_transforms(),
    )

    test_dataset = LungCancerDataset(
        dataset_dir=config.TEST_DIR,
        transform=get_test_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader