from src.data.preprocessing import (
    get_train_transforms,
    get_test_transforms,
)

print("=" * 50)
print("TRAIN TRANSFORMS")
print("=" * 50)
print(get_train_transforms())

print()

print("=" * 50)
print("TEST TRANSFORMS")
print("=" * 50)
print(get_test_transforms())