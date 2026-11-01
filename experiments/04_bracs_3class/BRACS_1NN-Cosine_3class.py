import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoProcessor, AutoModel

# =========================================================================
# 0. DEEP DUAL-STREAM LOGGER
# =========================================================================
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_ROOT = os.path.join(REPO_ROOT, "BRACS_ROI")
LOG_PATH = os.path.join(RESULTS_DIR, "bracs_3class_knn_log.txt")
sys.stdout = DualLogger(LOG_PATH)

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"

# Map folder naming markers to the identical 3-Class clinical groupings
CLINICAL_MAPPING = {
    "0_n":   "Benign",
    "1_pb":  "Benign",
    "2_udh": "Benign",
    "3_fea": "Atypical",
    "4_adh": "Atypical",
    "5_dcis": "Malignant",
    "6_ic":   "Malignant"
}

# =========================================================================
# 2. PLIP BACKBONE INITIALIZATION
# =========================================================================
print("Loading PLIP foundation model from Hugging Face Hub...")
processor = AutoProcessor.from_pretrained(PLIP_MODEL_NAME)
model = AutoModel.from_pretrained(PLIP_MODEL_NAME).to(DEVICE)
model.eval()
image_encoder = model.get_image_features

# =========================================================================
# 3. FEATURE EXTRACTION ENGINE
# =========================================================================
def extract_embeddings(image_paths, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features", file=sys.stdout):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
            outputs = image_encoder(**inputs)
            features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            if len(features.shape) == 3:
                features = torch.mean(features, dim=1)
            all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings)

# =========================================================================
# 4. PATIENT AND CLASS STRUCTURAL DATA PARSING
# =========================================================================
print("\nScanning BRACS ROI directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)

valid_paths, clinical_labels, patient_ids = [], [], []

for path in all_png_files:
    normalized_path = path.replace("\\", "/")
    path_parts = normalized_path.lower().split("/")
    
    matched_class = [key for key in CLINICAL_MAPPING.keys() if key in path_parts]
    if matched_class:
        class_key = matched_class[0]
        filename = path_parts[-1]
        
        # Pull core Patient identifier (e.g., BRACS_1152)
        if "bracs_" in filename:
            tokens = filename.split("_")
            idx = tokens.index("bracs")
            patient_id = f"BRACS_{tokens[idx+1]}"
            
            valid_paths.append(path)
            clinical_labels.append(CLINICAL_MAPPING[class_key])
            patient_ids.append(patient_id)

print(f"Isolated {len(valid_paths)} total ROIs from {len(set(patient_ids))} unique patients.")

valid_paths = np.array(valid_paths)
patient_ids = np.array(patient_ids)

print("Extracting global frozen PLIP features for all ROIs...")
X_embeddings = extract_embeddings(valid_paths, batch_size=32)

# =========================================================================
# 5. CORE WORKFLOW: 5-FOLD PATIENT-SAFE CROSS VALIDATION (1NN-COSINE)
# =========================================================================
print(f"\n\n{'='*50}\nRUNNING EXPERIMENT: 3-CLASS 1NN-COSINE BASELINE\n{'='*50}")

le = LabelEncoder()
y = le.fit_transform(clinical_labels)

gkf = GroupKFold(n_splits=5)
fold_accuracies = []

all_fold_trues = []
all_fold_preds = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_embeddings, y, groups=patient_ids)):
    print(f"  -> Executing Patient-Isolated Fold {fold+1}/5...")
    
    X_train, y_train = X_embeddings[train_idx], y[train_idx]
    X_test, y_test = X_embeddings[test_idx], y[test_idx]
    
    # 1-Nearest Neighbor with explicit Cosine Distance Metric
    knn_model = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_model.fit(X_train, y_train)
    
    # Predict fold out-of-sample data
    fold_preds_raw = knn_model.predict(X_test)
    
    fold_acc = accuracy_score(y_test, fold_preds_raw) * 100
    fold_accuracies.append(fold_acc)
    
    all_fold_trues.extend(y_test)
    all_fold_preds.extend(fold_preds_raw)

mean_cv_acc = np.mean(fold_accuracies)

# =========================================================================
# 6. REPORT GENERATION
# =========================================================================
print("\n" + "="*60)
print("FINAL BRACS 3-CLASS 1NN-COSINE BASELINE REPORT")
print("="*60)
print(f"Average Cross-Validated Accuracy: {mean_cv_acc:.2f}%")
print("="*60)
print("\nDetailed Out-of-Fold Classification Report:")
print(classification_report(all_fold_trues, all_fold_preds, target_names=le.classes_))

# Save structured baseline results to a comparative CSV
df_res = pd.DataFrame([{"Experiment": "3_Class_1NN_Cosine", "Accuracy": mean_cv_acc}])
df_res.to_csv(os.path.join(RESULTS_DIR, "bracs_3class_knn_summary.csv"), index=False)

print(f"\n[SUCCESS] Execution log perfectly written to: {os.path.abspath(LOG_PATH)}")