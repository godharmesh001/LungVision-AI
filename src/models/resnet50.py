"""
Project : Lung Cancer AI
Module  : ResNet50 Baseline (timm)

Purpose:
Baseline CNN model using timm.
"""

import timm


def create_model(num_classes=3, pretrained=True):
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


if __name__ == "__main__":

    model = create_model()

    print("=" * 60)
    print("ResNet50 Model Created Successfully")
    print("=" * 60)

    print(model)

    print("\nClassifier:")
    print(model.get_classifier())