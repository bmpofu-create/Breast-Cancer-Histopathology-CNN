import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoProcessor, AutoModel

# =========================================================================
# 0. DEEP DUAL-STREAM LOGGER (Saves absolutely everything to text file)
# =========================================================================
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Forces system buffer to write instantly to disk

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Direct ALL standard stdout streams into the automatic file tracking module
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_ROOT = os.path.join(REPO_ROOT, "BRACS_ROI")
LOG_PATH = os.path.join(RESULTS_DIR, "bracs_7class_subtyping_log.txt")
sys.stdout = DualLogger(LOG_PATH)

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"
# Absolute or relative path to your BRACS ROI dataset directory

BATCH_SIZE = 64
EPOCHS = 25  # Slightly increased to allow fine-grained class boundaries to stabilize
LEARNING_RATE = 1e-3

# Mapping raw folder names strictly to the 7 fine-grained histological subtypes
SUBTYPE_MAPPING = {
    "0_n":   "Normal",
    "1_pb":  "Pathological_Benign",
    "2_udh": "Usual_Ductal_Hyperplasia",
    "3_fea": "Flat_Epithelial_Atypia",
    "4_adh": "Atypical_Ductal_Hyperplasia",
    "5_dcis": "Ductal_Carcinoma_in_Situ",
    "6_ic":   "Invasive_Carcinoma"
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
# 3. CORE ARCHITECTURAL MODULES
# =========================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.embeddings[idx], self.labels[idx]

class PLIP7ClassAdapter(nn.Module):
    """Trainable adapter head scaled up for 7 fine-grained classes with increased capacity."""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=7, dropout_rate=0.3):
        super(PLIP7ClassAdapter, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.network(x)

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

valid_paths, subtype_labels, patient_ids = [], [], []

for path in all_png_files:
    normalized_path = path.replace("\\", "/")
    path_parts = normalized_path.lower().split("/")
    
    matched_class = [key for key in SUBTYPE_MAPPING.keys() if key in path_parts]
    if matched_class:
        class_key = matched_class[0]
        filename = path_parts[-1]
        
        if "bracs_" in filename:
            tokens = filename.split("_")
            idx = tokens.index("bracs")
            patient_id = f"BRACS_{tokens[idx+1]}"
            
            valid_paths.append(path)
            subtype_labels.append(SUBTYPE_MAPPING[class_key])
            patient_ids.append(patient_id)

print(f"Isolated {len(valid_paths)} total ROIs from {len(set(patient_ids))} unique patients.")

valid_paths = np.array(valid_paths)
patient_ids = np.array(patient_ids)

print("Extracting global frozen PLIP features for all ROIs...")
X_embeddings = extract_embeddings(valid_paths, batch_size=32)

# =========================================================================
# 5. CORE WORKFLOW: 5-FOLD PATIENT-SAFE CROSS VALIDATION (7-CLASS)
# =========================================================================
print(f"\n\n{'='*50}\nRUNNING EXPERIMENT: 7-CLASS FINE-GRAINED SUBTYPING\n{'='*50}")

le = LabelEncoder()
y = le.fit_transform(subtype_labels)

gkf = GroupKFold(n_splits=5)
fold_accuracies = []

all_fold_trues = []
all_fold_preds = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_embeddings, y, groups=patient_ids)):
    print(f"  -> Executing Patient-Isolated Fold {fold+1}/5...")
    
    train_dataset = EmbeddingDataset(X_embeddings[train_idx], y[train_idx])
    test_dataset = EmbeddingDataset(X_embeddings[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Using hidden_dim=256 to accommodate the increased class space
    adapter_model = PLIP7ClassAdapter(input_dim=768, hidden_dim=256, num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(adapter_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        adapter_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(adapter_model(inputs), targets)
            loss.backward()
            optimizer.step()
            
    adapter_model.eval()
    fold_preds, fold_trues = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = adapter_model(inputs.to(DEVICE))
            preds = torch.argmax(outputs, dim=1)
            fold_preds.extend(preds.cpu().numpy())
            fold_trues.extend(targets.numpy())
            
    fold_acc = accuracy_score(fold_trues, fold_preds) * 100
    fold_accuracies.append(fold_acc)
    
    all_fold_trues.extend(fold_trues)
    all_fold_preds.extend(fold_preds)

mean_cv_acc = np.mean(fold_accuracies)

# =========================================================================
# 6. REPORT GENERATION (Both prints to screen and appends to the log file)
# =========================================================================
print("\n" + "="*60)
print("FINAL BRACS 7-CLASS FINE-GRAINED SUBTYPING REPORT")
print("="*60)
print(f"Average Cross-Validated Accuracy: {mean_cv_acc:.2f}%")
print("="*60)
print("\nDetailed Out-of-Fold Classification Report:")
print(classification_report(all_fold_trues, all_fold_preds, target_names=le.classes_))

# Save structural metrics summary to CSV
df_res = pd.DataFrame([{"Experiment": "7_Class_Subtyping", "Accuracy": mean_cv_acc}])
df_res.to_csv(os.path.join(RESULTS_DIR, "bracs_7class_subtyping_summary.csv"), index=False)

print(f"\n[SUCCESS] Execution log perfectly written to: {os.path.abspath(LOG_PATH)}")