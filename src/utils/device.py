"""
Project : Lung Cancer AI
Module  : Device Manager

Purpose:
Automatically select the best available device.
"""

import torch


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")


if __name__ == "__main__":

    device = get_device()

    print("=" * 50)
    print("Device Detection")
    print("=" * 50)

    print("Selected Device :", device)

    print("\nPyTorch Version :", torch.__version__)

    if device.type == "mps":
        print("\n✅ Apple Metal GPU is available!")

    elif device.type == "cuda":
        print("\n✅ NVIDIA CUDA GPU is available!")

    else:
        print("\n⚠️ Running on CPU.")