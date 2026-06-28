from src.data.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders()

print("=" * 50)
print("DATALOADER TEST")
print("=" * 50)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

images, labels = next(iter(train_loader))

print()
print("Image Batch Shape :", images.shape)
print("Label Batch Shape :", labels.shape)