import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

print("Starting evaluation...")

CSV_PATH = "outputs/breakhis_metadata_splits.csv"
MODEL_PATH="models/final_tuned_cnn.pth"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading transforms...")
_, val_test_transform = get_transforms()

print("Loading test dataset...")
test_dataset = BreaKHisDataset(
    CSV_PATH,
    split="test",
    transform=val_test_transform
)

print("Test images:", len(test_dataset))

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Loading model...")
model = BaselineCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Beginning prediction...")

all_labels = []
all_preds = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):

        if batch_idx % 20 == 0:
            print(f"Processing batch {batch_idx}/{len(test_loader)}")

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        preds = (probs >= 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("Predictions complete")

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

print("\nTest Results")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print(
    "Weighted F1:",
    f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )
)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=["Benign","Malignant"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nEvaluation finished.")