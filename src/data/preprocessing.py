"""
Project : Lung Cancer AI
Module  : Image Preprocessing
Author  : Dharmesh Kumar

Purpose:
Reusable preprocessing pipelines for
training, validation and testing.
"""

from torchvision import transforms

from src.utils.config import IMAGE_SIZE


# =====================================================
# ImageNet Statistics
# =====================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================
# Training Pipeline
# =====================================================

def get_train_transforms():

    return transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

    ])


# =====================================================
# Validation / Test Pipeline
# =====================================================

def get_test_transforms():

    return transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

    ])