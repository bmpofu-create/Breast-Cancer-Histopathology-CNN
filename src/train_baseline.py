import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

# -----------------------
# Settings
# -----------------------
CSV_PATH = "outputs/breakhis_metadata_splits.csv"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Data
# -----------------------
train_transform, val_test_transform = get_transforms()

train_dataset = BreaKHisDataset(CSV_PATH, split="train", transform=train_transform)
val_dataset = BreaKHisDataset(CSV_PATH, split="val", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# -----------------------
# Model
# -----------------------
model = BaselineCNN().to(device)

# Class balancing
# label 0 = benign, label 1 = malignant

train_labels = train_dataset.df["label"]

num_benign = (train_labels == 0).sum()
num_malignant = (train_labels == 1).sum()

pos_weight_value = num_benign / num_malignant

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

print("Benign training images:", num_benign)
print("Malignant training images:", num_malignant)
print("Positive class weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # Validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss/len(train_loader):.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss/len(val_loader):.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

# Save model
torch.save(model.state_dict(), "models/baseline_cnn_17052026.pth")
print("Model saved to models/baseline_cnn_17052026.pth")