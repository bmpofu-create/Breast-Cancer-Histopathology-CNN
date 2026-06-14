import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

CSV_PATH = "outputs/breakhis_metadata_splits.csv"
MODEL_PATH = "models/baseline_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

_, val_test_transform = get_transforms()

test_dataset = BreaKHisDataset(
    CSV_PATH,
    split="test",
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

model = BaselineCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

all_labels = []
all_probs = []

print("Collecting probabilities...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(
            outputs
        ).cpu().numpy().flatten()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

import pandas as pd

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

results = []

print("\nThreshold Results")
print("=" * 60)

for t in thresholds:

    preds = (all_probs >= t).astype(int)

    acc = accuracy_score(
        all_labels,
        preds
    )

    precision = precision_score(
        all_labels,
        preds
    )

    recall = recall_score(
        all_labels,
        preds
    )

    f1 = f1_score(
        all_labels,
        preds,
        average="weighted"
    )

    print(f"\nThreshold={t}")
    print(f"Accuracy={acc:.4f}")
    print(f"Precision={precision:.4f}")
    print(f"Recall={recall:.4f}")
    print(f"Weighted F1={f1:.4f}")

    results.append({
        "Threshold": t,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Weighted_F1": f1
    })

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/threshold_results.csv",
    index=False
)

print("\nSaved threshold results:")
print("outputs/threshold_results.csv")