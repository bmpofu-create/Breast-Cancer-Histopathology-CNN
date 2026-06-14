import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

METADATA_FILE = Path("outputs/breakhis_metadata.csv")
OUTPUT_FILE = Path("outputs/breakhis_metadata_splits.csv")

df = pd.read_csv(METADATA_FILE)

# One row per patient
patients = df.groupby("patient_id").first().reset_index()

# First split: 70% train, 30% temp
train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=42,
    stratify=patients["label"]
)

# Second split: 15% validation, 15% test
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
    stratify=temp_patients["label"]
)

df["split"] = "unknown"

df.loc[df["patient_id"].isin(train_patients["patient_id"]), "split"] = "train"
df.loc[df["patient_id"].isin(val_patients["patient_id"]), "split"] = "val"
df.loc[df["patient_id"].isin(test_patients["patient_id"]), "split"] = "test"

df.to_csv(OUTPUT_FILE, index=False)

print("Saved split file to:", OUTPUT_FILE)
print("\nImage counts by split:")
print(df["split"].value_counts())

print("\nClass counts by split:")
print(pd.crosstab(df["split"], df["label_name"]))

print("\nPatient counts by split:")
print(df.groupby("split")["patient_id"].nunique())