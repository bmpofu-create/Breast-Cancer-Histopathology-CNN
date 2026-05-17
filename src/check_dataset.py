import os

data_dir = r"C:\Users\beven\MTU\BreastK_Research\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"

count = 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".png"):
            count += 1

print("Total images:", count)