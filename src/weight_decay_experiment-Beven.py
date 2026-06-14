import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from dataset import BreaKHisDataset
from transforms import get_transforms
from model import BaselineCNN

CSV_PATH = "outputs/breakhis_metadata_splits.csv"

BATCH_SIZE = 16
EPOCHS = 10

weight_decays = [
    0,
    1e-5,
    1e-4,
    1e-3
]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

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

results=[]

for wd in weight_decays:

    print("\n"+"="*50)
    print(f"Training Weight Decay={wd}")
    print("="*50)

    model=BaselineCNN().to(device)

    criterion=nn.BCEWithLogitsLoss()

    optimizer=optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=wd
    )

    best_val_acc=0

    for epoch in range(EPOCHS):

        model.train()

        train_correct=0
        train_total=0

        for images,labels in train_loader:

            images=images.to(device)

            labels=labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs=model(images)

            loss=criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            probs=torch.sigmoid(outputs)

            preds=(probs>=0.5).float()

            train_correct+=(preds==labels).sum().item()

            train_total+=labels.size(0)

        train_acc=train_correct/train_total

        model.eval()

        val_correct=0
        val_total=0

        with torch.no_grad():

            for images,labels in val_loader:

                images=images.to(device)

                labels=labels.float().unsqueeze(1).to(device)

                outputs=model(images)

                probs=torch.sigmoid(outputs)

                preds=(probs>=0.5).float()

                val_correct+=(preds==labels).sum().item()

                val_total+=labels.size(0)

        val_acc=val_correct/val_total

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Train:{train_acc:.4f} "
            f"Val:{val_acc:.4f}"
        )

        if val_acc>best_val_acc:
            best_val_acc=val_acc

    results.append({
        "WeightDecay":wd,
        "BestValAcc":best_val_acc
    })

results_df=pd.DataFrame(results)

results_df.to_csv(
    "outputs/weight_decay_results.csv",
    index=False
)

print("\nSaved:")
print("outputs/weight_decay_results.csv")
print(results_df)