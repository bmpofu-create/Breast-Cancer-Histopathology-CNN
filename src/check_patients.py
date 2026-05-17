import pandas as pd

df = pd.read_csv("outputs/breakhis_metadata.csv")

print("Total images:", len(df))
print("Unique patient IDs:", df["patient_id"].nunique())

print("\nPatient IDs:")
print(sorted(df["patient_id"].unique()))

print("\nImages per patient:")
print(df["patient_id"].value_counts().sort_index())