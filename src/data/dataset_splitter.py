"""
Project : Lung Cancer AI
Module  : Dataset Splitter
Author  : Dharmesh Kumar

Purpose:
Split the original dataset into
Train / Validation / Test
using symbolic links.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import os

from src.utils.config import (
    ORIGINAL_DATASET,
    DATASET_DIR,
    CLASSES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def create_symlink(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    os.symlink(source.resolve(), destination)


def split_class(class_name):

    source_folder = ORIGINAL_DATASET / class_name

    images = sorted(source_folder.glob("*"))

    train_images, temp_images = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_images, test_images = train_test_split(
        temp_images,
        train_size=val_size,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split_name, split_images in splits.items():

        destination = DATASET_DIR / split_name / class_name

        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:

            create_symlink(
                image,
                destination / image.name,
            )

    print(
        f"{class_name:<30}"
        f" Train:{len(train_images):4}"
        f"  Val:{len(val_images):4}"
        f"  Test:{len(test_images):4}"
    )


def main():

    print("=" * 60)
    print("DATASET SPLITTER")
    print("=" * 60)

    for cls in CLASSES:
        split_class(cls)

    print("\nDataset successfully prepared.")


if __name__ == "__main__":
    main()