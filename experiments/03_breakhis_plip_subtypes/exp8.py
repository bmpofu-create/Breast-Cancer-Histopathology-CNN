import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoProcessor, AutoModel

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"
MAGNIFICATION_TIERS = ["40X", "100X", "200X", "400X"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "subtype_1nn")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_ROOT = os.path.join(REPO_ROOT, "BreaKHis_v1", "BreaKHis_v1")
OUTPUT_DIR = RESULTS_DIR

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 4. EVALUATION LOOP ACROSS SUBTYPES & TIERS (PATIENT-LEVEL SPLIT)
# -------------------------------------------------------------------------
binary_results = {}
subtype_results = {}
trained_references = {}
classification_reports_log = []  # Holds text blocks for the final file write
detailed_test_records = []       # Global log for row-by-row prediction artifacts

print("\nScanning dataset directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)
print(f"Total localized image files: {len(all_png_files)}")

for mag in MAGNIFICATION_TIERS:
    print(f"\n{'='*50}\nProcessing Magnification Tier: {mag}\n{'='*50}")
    
    valid_paths = []
    binary_labels = []
    subtype_labels = []
    patient_ids = []  
    
    for path in all_png_files:
        normalized_path = path.replace("\\", "/")
        path_parts = normalized_path.lower().split("/")
        
        if mag.lower() in path_parts:
            if "benign" in path_parts or "malignant" in path_parts:
                bi_label = "Benign" if "benign" in path_parts else "Malignant"
                
                try:
                    anchor_idx = path_parts.index(bi_label.lower())
                    if path_parts[anchor_idx + 1] == "sob":
                        sub_label = path_parts[anchor_idx + 2]
                        patient_id = path_parts[anchor_idx + 3]
                    else:
                        sub_label = path_parts[anchor_idx + 1]
                        patient_id = path_parts[anchor_idx + 2]
                    
                    valid_paths.append(path)
                    binary_labels.append(bi_label)
                    subtype_labels.append(sub_label.upper()) 
                    patient_ids.append(patient_id.upper())
                except (ValueError, IndexError):
                    continue

    if len(valid_paths) == 0:
        print(f"Skipping {mag}: No images isolated for this zoom tier.")
        continue
        
    print(f"Found {len(valid_paths)} total samples.")
    
    le_binary = LabelEncoder()
    y_binary_all = le_binary.fit_transform(binary_labels)
    
    le_subtype = LabelEncoder()
    y_subtype_all = le_subtype.fit_transform(subtype_labels)
    print(f"Detected {len(le_subtype.classes_)} unique subtypes: {list(le_subtype.classes_)}")

    unique_patients = list(set(patient_ids))
    patient_to_subtype = {}
    for p_id, sub_lbl in zip(patient_ids, y_subtype_all):
        patient_to_subtype[p_id] = sub_lbl
        
    unique_patient_subtypes = [patient_to_subtype[p] for p in unique_patients]
    
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42, stratify=unique_patient_subtypes
    )
    
    print(f"Patient split summary -> Total unique patients: {len(unique_patients)} | Train Patients: {len(train_patients)} | Test Patients: {len(test_patients)}")

    idx_train = [i for i, p_id in enumerate(patient_ids) if p_id in train_patients]
    idx_test = [i for i, p_id in enumerate(patient_ids) if p_id in test_patients]

    print("Re-embedding targets through PLIP feature-extractor...")
    X_all = extract_embeddings(valid_paths, batch_size=32)
    
    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_bin_train, y_bin_test = y_binary_all[idx_train], y_binary_all[idx_test]
    y_sub_train, y_sub_test = y_subtype_all[idx_train], y_subtype_all[idx_test]
    
    # Save raw numpy matrices for downstream custom workloads
    np.save(os.path.join(OUTPUT_DIR, f"{mag}_X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"{mag}_X_test.npy"), X_test)

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
    
    # Collect data entries to write into a comprehensive run breakdown CSV
    test_paths_subset = [valid_paths[idx] for idx in idx_test]
    test_patients_subset = [patient_ids[idx] for idx in idx_test]
    
    for idx, (true_b, pred_b, true_s, pred_s, p_id, f_path) in enumerate(zip(
        y_bin_test, y_bin_pred, y_sub_test, y_sub_pred, test_patients_subset, test_paths_subset
    )):
        detailed_test_records.append({
            "magnification": mag,
            "patient_id": p_id,
            "file_name": os.path.basename(f_path),
            "true_binary": le_binary.inverse_transform([true_b])[0],
            "pred_binary": le_binary.inverse_transform([pred_b])[0],
            "true_subtype": le_subtype.inverse_transform([true_s])[0],
            "pred_subtype": le_subtype.inverse_transform([pred_s])[0],
        })

    # Generate and save Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Binary Matrix Plot
    cm_bin = confusion_matrix(y_bin_test, y_bin_pred)
    disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=le_binary.classes_)
    disp_bin.plot(ax=axes[0], cmap=plt.cm.Blues, values_format='d')
    axes[0].set_title(f"Binary Confusion Matrix ({mag})")
    
    # Subtype Matrix Plot
    cm_sub = confusion_matrix(y_sub_test, y_sub_pred)
    disp_sub = ConfusionMatrixDisplay(confusion_matrix=cm_sub, display_labels=le_subtype.classes_)
    disp_sub.plot(ax=axes[1], cmap=plt.cm.Oranges, values_format='d')
    axes[1].set_title(f"Subtype Confusion Matrix ({mag})")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mag}_confusion_matrices.png"), dpi=150)
    plt.close()

    # Capture the Classification Report text block dynamically
    report_text = classification_report(y_sub_test, y_sub_pred, target_names=le_subtype.classes_, zero_division=0)
    
    # Structure text blocks for output log file
    formatted_report_block = f"{'='*60}\n[{mag}] Detailed Subtype Classification Report\n{'='*60}\n{report_text}\n\n"
    classification_reports_log.append(formatted_report_block)

    print(f"\n--- {mag} Evaluation Report ---")
    print(f"Overarching Class (Binary) Accuracy: {bin_accuracy:.2f}%")
    print(f"Fine Subtype Class Accuracy: {sub_accuracy:.2f}%\n")
    print(formatted_report_block)

# -------------------------------------------------------------------------
# 5. EXPERIMENT SUMMARY & SAVING RESULTS
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("FINAL EXPERIMENT SUMMARY (PLIP DUAL-CLASSIFICATION)")
print("="*50)
for mag in binary_results.keys():
    print(f"Magnification {mag:<5} -> Binary: {binary_results[mag]:.2f}% | Subtype: {subtype_results[mag]:.2f}%")
print("="*50)

TXT_REPORT_PATH = os.path.join(OUTPUT_DIR, "breakhis_plip_subtypes_results.txt")
CSV_REPORT_PATH = os.path.join(OUTPUT_DIR, "breakhis_plip_subtypes_results.csv")
DETAILED_CSV_PATH = os.path.join(OUTPUT_DIR, "breakhis_detailed_predictions.csv")
REPORTS_TEXT_PATH = os.path.join(OUTPUT_DIR, "breakhis_classification_reports.txt")

# Write out precision, recall, f1-score report blocks
with open(REPORTS_TEXT_PATH, "w") as reports_file:
    for block in classification_reports_log:
        reports_file.write(block)

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

# Write individual inference transactions out for auditing
with open(DETAILED_CSV_PATH, "w") as detailed_file:
    detailed_file.write("magnification,patient_id,file_name,true_binary,pred_binary,true_subtype,pred_subtype\n")
    for row in detailed_test_records:
        detailed_file.write(f"{row['magnification']},{row['patient_id']},{row['file_name']},{row['true_binary']},{row['pred_binary']},{row['true_subtype']},{row['pred_subtype']}\n")

print(f"All processing outputs structured within folder: {os.path.abspath(OUTPUT_DIR)}")
print(f"↳ Saved Classification Reports (Precision/Recall blocks): {os.path.basename(REPORTS_TEXT_PATH)}")
print(f"↳ Performance summaries: {os.path.basename(TXT_REPORT_PATH)} & {os.path.basename(CSV_REPORT_PATH)}")
print(f"↳ Granular item-level logs: {os.path.basename(DETAILED_CSV_PATH)}")
print(f"↳ Confusion Matrices: [Mag]_confusion_matrices.png")
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