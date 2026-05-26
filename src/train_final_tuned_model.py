import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

CSV_PATH = "outputs/breakhis_metadata_splits.csv"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

MODEL_SAVE_PATH = "models/final_tuned_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform, val_test_transform = get_transforms()

train_dataset = BreaKHisDataset(
    CSV_PATH,
    split="train",
    transform=train_transform
)

val_dataset = BreaKHisDataset(
    CSV_PATH,
    split="val",
    transform=val_test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

model = BaselineCNN(dropout_rate=DROPOUT_RATE).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
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

        if batch_idx % 25 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    model.eval()

    val_loss = 0.0
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
    avg_val_loss = val_loss / len(val_loader)

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Acc:  {train_acc:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("New best final tuned model saved.")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")