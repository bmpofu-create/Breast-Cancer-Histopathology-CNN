from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from paths import BREAKHIS_ROOT

class BreaKHisDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None):
        self.df = pd.read_csv(csv_file)

        # Select only one split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["path"])
        if not image_path.is_file():
            image_path = BREAKHIS_ROOT / image_path
        label = row["label"]

        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label