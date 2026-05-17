import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

# -----------------------
# Settings
# -----------------------
CSV_PATH = "outputs/breakhis_metadata_splits.csv"
MODEL_PATH = "models/baseline_cnn.pth"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Dataset
# -----------------------
_, val_test_transform = get_transforms()

test_dataset = BreaKHisDataset(
    CSV_PATH,
    split="test",
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test images:", len(test_dataset))

# -----------------------
# Load model
# -----------------------
model = BaselineCNN().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

# -----------------------
# Prediction
# -----------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        preds = (probs >= 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# -----------------------
# Metrics
# -----------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
weighted_f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted"
)

print("\nTest Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Weighted F1:", weighted_f1)

# -----------------------
# Save metrics CSV
# -----------------------
results_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        weighted_f1
    ]
})

results_df.to_csv(
    "outputs/baseline_results.csv",
    index=False
)

print("\nSaved metrics to outputs/baseline_results.csv")

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax)

plt.title("Baseline CNN Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved confusion matrix to outputs/confusion_matrix.png")

plt.show()

# -----------------------
# Classification report
# -----------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=["Benign", "Malignant"],
    digits=4
)

print("\nClassification Report:")
print(report)

import os
output_dir = "outputs"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"Saved classification report to {output_dir}/classification_report.txt")