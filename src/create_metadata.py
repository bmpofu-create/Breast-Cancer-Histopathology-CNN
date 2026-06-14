from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\beven\MTU\BreastK_Research\BreaKHis_v1\BreaKHis_v1\histology_slides\breast")
OUTPUT_FILE = Path("outputs/breakhis_metadata.csv")

images = sorted(DATA_DIR.rglob("*.png"))

records = []

for img_path in images:
    filename = img_path.name
    stem = img_path.stem

    # Example filename:
    # SOB_B_A-14-22549AB-40-001.png
    parts = stem.split("-")

    # Safety check
    if len(parts) < 4:
        print(f"Skipping unexpected filename: {filename}")
        continue

    # Label
    if "_B_" in filename:
        label = 0
        label_name = "benign"
    elif "_M_" in filename:
        label = 1
        label_name = "malignant"
    else:
        print(f"Skipping file with unknown label: {filename}")
        continue

    # Subtype
    # SOB_B_A-14-22549AB-40-001
    subtype = stem.split("_")[2].split("-")[0]

    # Patient ID
    # Usually: 14-22549AB
    patient_id = parts[1] + "-" + parts[2]

    # Magnification
    magnification = parts[-2]

    records.append({
        "path": str(img_path),
        "filename": filename,
        "label": label,
        "label_name": label_name,
        "subtype": subtype,
        "patient_id": patient_id,
        "magnification": magnification
    })

df = pd.DataFrame(records)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Metadata saved to:", OUTPUT_FILE)
print("Total images:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nClass counts:")
print(df["label_name"].value_counts())

print("\nMagnification counts:")
print(df["magnification"].value_counts())

print("\nUnique patients:")
print(df["patient_id"].nunique())

print("\nImages per patient:")
print(df["patient_id"].value_counts().sort_index())