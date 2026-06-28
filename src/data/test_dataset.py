from src.data.dataset import LungCancerDataset
from src.data.preprocessing import get_train_transforms

dataset = LungCancerDataset(
    dataset_dir="dataset/train",
    transform=get_train_transforms(),
)

print("=" * 50)
print("Dataset Test")
print("=" * 50)

print("Total samples:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)