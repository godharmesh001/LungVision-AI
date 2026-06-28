"""
Project : Lung Cancer AI
Module  : Dataset
Author  : Dharmesh Kumar

Purpose:
PyTorch Dataset class for loading
lung histopathology images.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.utils.config import CLASS_TO_INDEX


class LungCancerDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):

        self.dataset_dir = Path(dataset_dir)
        self.transform = transform

        self.samples = []

        for class_name, label in CLASS_TO_INDEX.items():

            class_folder = self.dataset_dir / class_name

            for image_path in sorted(class_folder.glob("*")):

                self.samples.append(
                    (
                        image_path,
                        label,
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label