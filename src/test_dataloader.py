from torch.utils.data import DataLoader
from create_dataset import BreaKHisDataset
from transforms import get_transforms

csv_path = "outputs/breakhis_metadata_splits.csv"

train_transform, val_test_transform = get_transforms()

train_dataset = BreaKHisDataset(csv_path, split="train", transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Test one batch
for images, labels in train_loader:
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    break