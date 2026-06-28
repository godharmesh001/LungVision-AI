import torch
import torch.nn as nn
import torch.optim as optim

from src.models.resnet50 import create_model
from src.data.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.utils.device import get_device

device = get_device()

train_loader, _, _ = create_dataloaders(batch_size=8)

model = create_model()

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

trainer = Trainer(
    model,
    optimizer,
    criterion,
    device,
)

print("=" * 60)
print("TRAINER SMOKE TEST")
print("=" * 60)

loss, accuracy = trainer.train_one_epoch(train_loader)

print()

print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")