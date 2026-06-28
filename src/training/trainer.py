"""
Project : Lung Cancer AI
Module  : Trainer

Purpose:
Generic trainer for any PyTorch classification model.
"""

import torch

from tqdm import tqdm

from src.training.metrics import (
    AverageMeter,
    calculate_accuracy,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, dataloader):

        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        progress = tqdm(dataloader)

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy, images.size(0))

            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc_meter.avg:.4f}",
            )

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate_one_epoch(self, dataloader):

        self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            accuracy = calculate_accuracy(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy, images.size(0))

        return loss_meter.avg, acc_meter.avg