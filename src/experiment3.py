import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoProcessor, AutoModel

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Pathology Language-Image Pretraining (PLIP) model descriptor
PLIP_MODEL_NAME = "vinid/plip"
MAGNIFICATION_TIERS = ["40X", "100X", "200X", "400X"]

# Root directory pointing to the top level of your extracted dataset
DATASET_ROOT = "C:/Users/beven/MTU/Projects/BreaKHis_v1" 

# -------------------------------------------------------------------------
# 2. INITIALIZE PLIP MODEL & PROCESSOR
# -------------------------------------------------------------------------
print("Loading PLIP foundation model from Hugging Face Hub...")
processor = AutoProcessor.from_pretrained(PLIP_MODEL_NAME)
model = AutoModel.from_pretrained(PLIP_MODEL_NAME).to(DEVICE)
# Isolate the vision transformer backbone
image_encoder = model.get_image_features

# -------------------------------------------------------------------------
# 3. HELPER FUNCTION: EXTRACT & L2-NORMALIZE EMBEDDINGS
# -------------------------------------------------------------------------
def extract_embeddings(image_paths, batch_size=32):
    """
    Extracts, L2-normalizes, and returns the latent features from PLIP.
    Ensures outputs are flattened to 2D arrays for dot-product compatibility.
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"\nError loading image {path}: {e}")
                continue

        if not batch_images:
            continue

        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = image_encoder(**inputs)
            
            # Safely unpack raw embedding tensor from HF wrapper object
            features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            
            # FIX: If the tensor is 3D (batch_size, sequence_length, embedding_dim),
            # perform Global Average Pooling over the sequence dimension (dim 1) to compress it to 2D
            if len(features.shape) == 3:
                features = torch.mean(features, dim=1)
            
            # Apply L2 normalization for clean cosine similarity via dot product
            l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)
            normalized_features = features / l2_norm
            all_embeddings.append(normalized_features.cpu().numpy())
            
    return np.vstack(all_embeddings)

# -------------------------------------------------------------------------
# 4. EVALUATION LOOP ACROSS THE NESTED STRUCTURE
# -------------------------------------------------------------------------
results_summary = {}

# Recursively crawl all paths beneath root folder up front
print("\nScanning dataset directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)
print(f"Total localized image files: {len(all_png_files)}")

for mag in MAGNIFICATION_TIERS:
    print(f"\n{"="*50}\nProcessing Magnification Tier: {mag}\n{"="*50}")
    
    benign_paths = []
    malignant_paths = []
    
    for path in all_png_files:
        # Standardize slashes for cross-platform robustness (Windows path handling)
        normalized_path = path.replace("\\", "/")
        path_parts = normalized_path.lower().split("/")
        
        # Check if target magnification tier folder is in the path fragments
        if mag.lower() in path_parts:
            # Safely assign binary ground truth based on high-level folder labels
            if "benign" in path_parts:
                benign_paths.append(path)
            elif "malignant" in path_parts:
                malignant_paths.append(path)

    all_paths = benign_paths + malignant_paths
    labels = ["Benign"] * len(benign_paths) + ["Malignant"] * len(malignant_paths)
    
    if len(all_paths) == 0:
        print(f"Skipping {mag}: No images isolated for this zoom tier.")
        continue
        
    print(f"Found {len(benign_paths)} Benign samples and {len(malignant_paths)} Malignant samples.")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print("Re-embedding targets through PLIP feature-extractor...")
    X = extract_embeddings(all_paths, batch_size=32)
    
    # Stratified split to preserve representation boundaries
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Computing dot-product similarity matrix...")
    # Because embeddings are L2-normalized, Cosine Similarity simplifies to a dot product matrix
    cosine_similarity = np.dot(X_test, X_train.T)
    
    # Maximum similarity alignment index retrieval (1-Nearest Neighbor style matching)
    closest_match_indices = np.argmax(cosine_similarity, axis=1)
    y_pred = y_train[closest_match_indices]
    
    accuracy = np.mean(y_pred == y_test) * 100
    results_summary[mag] = accuracy
    print(f"Results for {mag} Tier -> Classification Accuracy: {accuracy:.2f}%")

# -------------------------------------------------------------------------
# 5. EXPERIMENT SUMMARY & SAVING RESULTS
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("FINAL EXPERIMENT SUMMARY (PLIP BACKBONE)")
print("="*50)
for mag, acc in results_summary.items():
    print(f"Magnification {mag}: {acc:.2f}%")
print("="*50)

TXT_REPORT_PATH = "breakhis_plip_results.txt"
CSV_REPORT_PATH = "breakhis_plip_results.csv"

# 1. Save as a clean, structured Text Report (Markdown table format)
with open(TXT_REPORT_PATH, "w") as txt_file:
    txt_file.write("==================================================\n")
    txt_file.write("BreaKHis Classification Results - PLIP Backbone\n")
    txt_file.write("==================================================\n")
    txt_file.write("| Magnification Tier | Classification Accuracy |\n")
    txt_file.write("|--------------------|-------------------------|\n")
    for mag, acc in results_summary.items():
        txt_file.write(f"| {mag:<18} | {acc:.2f}% |\n")
    txt_file.write("==================================================\n")

print(f"Text report successfully saved to: {os.path.abspath(TXT_REPORT_PATH)}")

# 2. Save as a raw CSV file for easy plotting/data framing
with open(CSV_REPORT_PATH, "w") as csv_file:
    csv_file.write("magnification,accuracy\n")
    for mag, acc in results_summary.items():
        csv_file.write(f"{mag},{acc:.4f}\n")

print(f"CSV spreadssheet successfully saved to: {os.path.abspath(CSV_REPORT_PATH)}")
print("="*50)