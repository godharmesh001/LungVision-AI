import torch
import torch.optim as optim

from src.models.resnet50 import create_model
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

model = create_model(pretrained=False)

optimizer = optim.Adam(model.parameters())

print("=" * 50)
print("Checkpoint Test")
print("=" * 50)

save_checkpoint(
    model,
    optimizer,
    epoch=5,
    accuracy=0.95,
)

epoch, accuracy = load_checkpoint(
    model,
    optimizer,
)

print("Loaded Epoch:", epoch)
print("Loaded Accuracy:", accuracy)