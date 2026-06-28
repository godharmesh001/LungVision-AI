import torch

from src.training.metrics import (
    AverageMeter,
    calculate_accuracy,
)

meter = AverageMeter()

meter.update(2.0, 2)
meter.update(4.0, 2)

print("=" * 50)
print("AverageMeter Test")
print("=" * 50)

print("Average:", meter.avg)

outputs = torch.tensor([
    [0.1, 0.8, 0.1],
    [0.7, 0.2, 0.1],
])

labels = torch.tensor([1, 0])

acc = calculate_accuracy(outputs, labels)

print("Accuracy:", acc)