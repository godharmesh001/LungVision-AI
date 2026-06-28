"""
Project : Lung Cancer AI
Module  : Metrics

Purpose:
Utility functions for computing
training and validation metrics.
"""

import torch


class AverageMeter:
    """
    Computes and stores the average value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, labels):

    predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy