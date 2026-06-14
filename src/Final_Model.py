import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoProcessor, AutoModel

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"
MAGNIFICATION_TIERS = ["40X", "100X", "200X", "400X"]
DATASET_ROOT = "C:/Users/beven/MTU/Projects/BreaKHis_v1" 

# -------------------------------------------------------------------------
# 2. INITIALIZE PLIP MODEL & PROCESSOR
# -------------------------------------------------------------------------
print("Loading PLIP foundation model from Hugging Face Hub...")
processor = AutoProcessor.from_pretrained(PLIP_MODEL_NAME)
model = AutoModel.from_pretrained(PLIP_MODEL_NAME).to(DEVICE)
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
            features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            
            # Compress spatial visual dimensions down to a 2D context array
            if len(features.shape) == 3:
                features = torch.mean(features, dim=1)
            
            # Apply L2 normalization to enable exact matrix dot-product cosine similarity
            l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)
            normalized_features = features / l2_norm
            all_embeddings.append(normalized_features.cpu().numpy())
            
    return np.vstack(all_embeddings)

# -------------------------------------------------------------------------
# 4. EVALUATION LOOP ACROSS SUBTYPES & TIERS
# -------------------------------------------------------------------------
binary_results = {}
subtype_results = {}
trained_references = {}

print("\nScanning dataset directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)
print(f"Total localized image files: {len(all_png_files)}")

for mag in MAGNIFICATION_TIERS:
    print(f"\n{'='*50}\nProcessing Magnification Tier: {mag}\n{'='*50}")
    
    valid_paths = []
    binary_labels = []
    subtype_labels = []
    
    for path in all_png_files:
        normalized_path = path.replace("\\", "/")
        path_parts = normalized_path.lower().split("/")
        
        if mag.lower() in path_parts:
            if "benign" in path_parts or "malignant" in path_parts:
                bi_label = "Benign" if "benign" in path_parts else "Malignant"
                
                try:
                    anchor_idx = path_parts.index(bi_label.lower())
                    # Skip 'sob' folder if it immediately follows benign/malignant
                    if path_parts[anchor_idx + 1] == "sob":
                        sub_label = path_parts[anchor_idx + 2]
                    else:
                        sub_label = path_parts[anchor_idx + 1]
                    
                    valid_paths.append(path)
                    binary_labels.append(bi_label)
                    subtype_labels.append(sub_label.upper()) # uniform casing
                except (ValueError, IndexError):
                    continue

    if len(valid_paths) == 0:
        print(f"Skipping {mag}: No images isolated for this zoom tier.")
        continue
        
    print(f"Found {len(valid_paths)} total samples.")
    
    # Label array integer conversions
    le_binary = LabelEncoder()
    y_binary = le_binary.fit_transform(binary_labels)
    
    le_subtype = LabelEncoder()
    y_subtype = le_subtype.fit_transform(subtype_labels)
    print(f"Detected {len(le_subtype.classes_)} unique subtypes: {list(le_subtype.classes_)}")

    print("Re-embedding targets through PLIP feature-extractor...")
    X = extract_embeddings(valid_paths, batch_size=32)
    
    # Split using the subtype array as a stratification guide
    indices = np.arange(len(X))
    idx_train, idx_test, _, _ = train_test_split(
        indices, y_subtype, test_size=0.3, random_state=42, stratify=y_subtype
    )
    
    X_train, X_test = X[idx_train], X[idx_test]
    y_bin_train, y_bin_test = y_binary[idx_train], y_binary[idx_test]
    y_sub_train, y_sub_test = y_subtype[idx_train], y_subtype[idx_test]
    
    # Keep track of training references
    trained_references[mag] = {
        "X_train": X_train,
        "y_bin_train": y_bin_train,
        "y_sub_train": y_sub_train,
        "le_binary": le_binary,
        "le_subtype": le_subtype
    }
    
    print("Computing dot-product similarity matrix...")
    cosine_similarity = np.dot(X_test, X_train.T)
    closest_match_indices = np.argmax(cosine_similarity, axis=1)
    
    # Evaluate Overarching Binary Diagnostics
    y_bin_pred = y_bin_train[closest_match_indices]
    bin_accuracy = accuracy_score(y_bin_test, y_bin_pred) * 100
    binary_results[mag] = bin_accuracy
    
    # Evaluate Fine-Grained Tissue Subtypes
    y_sub_pred = y_sub_train[closest_match_indices]
    sub_accuracy = accuracy_score(y_sub_test, y_sub_pred) * 100
    subtype_results[mag] = sub_accuracy
    
    print(f"\n--- {mag} Evaluation Report ---")
    print(f"Overarching Class (Binary) Accuracy: {bin_accuracy:.2f}%")
    print(f"Fine Subtype Class Accuracy: {sub_accuracy:.2f}%\n")
    
    print(f"[{mag}] Detailed Subtype Classification Report:")
    print(classification_report(y_sub_test, y_sub_pred, target_names=le_subtype.classes_, zero_division=0))

# -------------------------------------------------------------------------
# 5. EXPERIMENT SUMMARY & SAVING RESULTS
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("FINAL EXPERIMENT SUMMARY (PLIP DUAL-CLASSIFICATION)")
print("="*50)
for mag in binary_results.keys():
    print(f"Magnification {mag:<5} -> Binary: {binary_results[mag]:.2f}% | Subtype: {subtype_results[mag]:.2f}%")
print("="*50)

TXT_REPORT_PATH = "breakhis_plip_subtypes_results.txt"
CSV_REPORT_PATH = "breakhis_plip_subtypes_results.csv"

with open(TXT_REPORT_PATH, "w") as txt_file:
    txt_file.write("============================================================\n")
    txt_file.write("BreaKHis Multi-Class Classification Results - PLIP Backbone\n")
    txt_file.write("============================================================\n")
    txt_file.write("| Magnification Tier | Binary Accuracy | Subtype Accuracy |\n")
    txt_file.write("|--------------------|-----------------|------------------|\n")
    for mag in binary_results.keys():
        bin_str = f"{binary_results[mag]:.2f}%"
        sub_str = f"{subtype_results[mag]:.2f}%"
        txt_file.write(f"| {mag:<18} | {bin_str:<15} | {sub_str:<16} |\n")
    txt_file.write("============================================================\n")

with open(CSV_REPORT_PATH, "w") as csv_file:
    csv_file.write("magnification,binary_accuracy,subtype_accuracy\n")
    for mag in binary_results.keys():
        csv_file.write(f"{mag},{binary_results[mag]:.4f},{subtype_results[mag]:.4f}\n")

print(f"Text report successfully saved to: {os.path.abspath(TXT_REPORT_PATH)}")
print(f"CSV spreadsheet successfully saved to: {os.path.abspath(CSV_REPORT_PATH)}")
print("="*50)

# -------------------------------------------------------------------------
# 6. TESTING PIPELINE FOR NEW IMAGES
# -------------------------------------------------------------------------
def predict_image(image_path, magnification_tier):
    if magnification_tier not in trained_references:
        print(f"Error: No training reference available for magnification tier '{magnification_tier}'.")
        return None, None

    try:
        test_embedding = extract_embeddings([image_path], batch_size=1)
    except Exception as e:
        print(f"Failed to process image: {e}")
        return None, None

    ref = trained_references[magnification_tier]
    X_train = ref["X_train"]
    y_bin_train = ref["y_bin_train"]
    y_sub_train = ref["y_sub_train"]
    le_binary = ref["le_binary"]
    le_subtype = ref["le_subtype"]

    similarity = np.dot(test_embedding, X_train.T)
    closest_idx = np.argmax(similarity, axis=1)[0]

    pred_bin_encoded = y_bin_train[closest_idx]
    pred_sub_encoded = y_sub_train[closest_idx]

    predicted_class = le_binary.inverse_transform([pred_bin_encoded])[0]
    predicted_subtype = le_subtype.inverse_transform([pred_sub_encoded])[0]

    return predicted_class, predicted_subtype

# Interactive Tester Loop
print("\n" + "═"*50)
print("   INTERACTIVE PLIP EXPERIMENT TESTER")
print("═"*50)

while True:
    test_path = input("\nEnter the path to a histopathology image to classify (or type 'exit' to quit): ").strip()
    if test_path.lower() == 'exit':
        print("Exiting. Happy modeling!")
        break
        
    if not os.path.exists(test_path):
        print("File path does not exist. Please check your pathing and try again.")
        continue

    print(f"Available magnification tiers: {list(trained_references.keys())}")
    test_mag = input("Enter the magnification tier for this image (e.g., 40X, 100X, 200X, 400X): ").strip().upper()

    if test_mag not in trained_references:
        print(f"Invalid tier. Please choose from: {list(trained_references.keys())}")
        continue

    print("\nRunning inference...")
    pred_class, pred_subtype = predict_image(test_path, test_mag)

    if pred_class and pred_subtype:
        print("\n" + "─"*40)
        print(f"   IMAGE:   {os.path.basename(test_path)}")
        print(f"   TIER:    {test_mag}")
        print(f"   CLASS:   {pred_class}")
        print(f"   SUBTYPE: {pred_subtype}")
        print("─"*40)