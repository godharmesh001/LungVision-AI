"""
Project : Lung Cancer AI
Module  : DataLoader

Purpose:
Create reusable PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

from src.data.dataset import LungCancerDataset
from src.data.preprocessing import (
    get_train_transforms,
    get_test_transforms,
)


def create_dataloaders(batch_size=32):

    train_dataset = LungCancerDataset(
        dataset_dir="dataset/train",
        transform=get_train_transforms(),
    )

    val_dataset = LungCancerDataset(
        dataset_dir="dataset/val",
        transform=get_test_transforms(),
    )

    test_dataset = LungCancerDataset(
        dataset_dir="dataset/test",
        transform=get_test_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader