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
from sklearn.metrics import accuracy_score
from transformers import AutoProcessor, AutoModel

# =========================================================================
# 0. DUAL-STREAM LOGGER (Saves Terminal Output to Text Automatically)
# =========================================================================
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Force write immediately to disk

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Start logging everything to 'experiment_terminal_log.txt'
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "adapter")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_ROOT = os.path.join(REPO_ROOT, "BreaKHis_v1", "BreaKHis_v1")
LOG_PATH = os.path.join(RESULTS_DIR, "experiment_terminal_log.txt")
sys.stdout = DualLogger(LOG_PATH)

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"
MAGNIFICATION_TIERS = ["40X", "100X", "200X", "400X"]
# Absolute path to your BreaKHis project folder

# Hyperparameters for the trainable adapter head
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

# =========================================================================
# 2. INITIALIZE PLIP MODEL & PROCESSOR (FROZEN BACKBONE)
# =========================================================================
print("Loading PLIP foundation model from Hugging Face Hub...")
processor = AutoProcessor.from_pretrained(PLIP_MODEL_NAME)
model = AutoModel.from_pretrained(PLIP_MODEL_NAME).to(DEVICE)
model.eval()  # Freeze backbone weights
image_encoder = model.get_image_features

# =========================================================================
# 3. PYTORCH DATASET & TINY ADAPTER HEAD DEFINITION
# =========================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class PLIPBinaryAdapter(nn.Module):
    """
    The tiny trainable adapter network requested by Sujan.
    Matches the native 768-dimensional visual space of PLIP.
    """
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2, dropout_rate=0.3):
        super(PLIPBinaryAdapter, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# =========================================================================
# 4. EMBEDDING EXTRACTION ENGINE
# =========================================================================
def extract_embeddings(image_paths, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features", file=sys.stdout):
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
            outputs = image_encoder(**inputs)
            features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            
            if len(features.shape) == 3:
                features = torch.mean(features, dim=1)
                
            all_embeddings.append(features.cpu().numpy())
                
    return np.vstack(all_embeddings)

# =========================================================================
# 5. CORE WORKFLOW: TRAIN & EVALUATE WITH PATIENT STRUCTURAL GROUPING
# =========================================================================
adapter_results = {}
trained_models = {}

print("\nScanning dataset directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)
print(f"Total localized image files: {len(all_png_files)}")

for mag in MAGNIFICATION_TIERS:
    print(f"\n{'='*50}\nProcessing Magnification Tier: {mag}\n{'='*50}")
    
    valid_paths = []
    binary_labels = []
    patient_ids = []
    
    # Structural parsing mapping to your exact folder system
    for path in all_png_files:
        normalized_path = path.replace("\\", "/")
        path_parts = normalized_path.lower().split("/")
        
        if mag.lower() in path_parts:
            if "benign" in path_parts or "malignant" in path_parts:
                bi_label = "Benign" if "benign" in path_parts else "Malignant"
                
                try:
                    # Dynamically isolate the 'sob_x_x_patient' folder token 
                    target_idx = [i for i, part in enumerate(path_parts) if "sob_" in part][0]
                    patient_group = path_parts[target_idx]
                    
                    valid_paths.append(path)
                    binary_labels.append(bi_label)
                    patient_ids.append(patient_group)
                except IndexError:
                    continue

    if len(valid_paths) == 0:
        print(f"Skipping {mag}: No images isolated for this zoom tier.")
        continue
        
    print(f"Found {len(valid_paths)} total patches from {len(set(patient_ids))} unique patients.")
    
    le_binary = LabelEncoder()
    y_binary = le_binary.fit_transform(binary_labels)
    patient_ids = np.array(patient_ids)
    valid_paths = np.array(valid_paths)

    print("Extracting frozen PLIP features...")
    X = extract_embeddings(valid_paths, batch_size=32)
    
    # 5-Fold Patient-Safe Cross-Validation
    gkf = GroupKFold(n_splits=5)
    fold_accuracies = []
    
    # Containers to collect cross-validated sample outputs
    all_fold_paths = []
    all_fold_patients = []
    all_fold_trues = []
    all_fold_preds = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_binary, groups=patient_ids)):
        print(f"  -> Executing Patient-Isolated Fold {fold+1}/5...")
        
        train_dataset = EmbeddingDataset(X[train_idx], y_binary[train_idx])
        test_dataset = EmbeddingDataset(X[test_idx], y_binary[test_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Instantiate independent custom adapter head per validation fold step
        adapter_model = PLIPBinaryAdapter(input_dim=768, hidden_dim=128).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(adapter_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Training local adapter layer weights
        for epoch in range(EPOCHS):
            adapter_model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(adapter_model(inputs), targets)
                loss.backward()
                optimizer.step()
                
        # Evaluating fold stability
        adapter_model.eval()
        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = adapter_model(inputs.to(DEVICE))
                preds = torch.argmax(outputs, dim=1)
                fold_preds.extend(preds.cpu().numpy())
                fold_trues.extend(targets.numpy())
                
        fold_acc = accuracy_score(fold_trues, fold_preds) * 100
        fold_accuracies.append(fold_acc)
        
        # Collect test metadata for saving out to the CSV sheet
        all_fold_paths.extend(valid_paths[test_idx])
        all_fold_patients.extend(patient_ids[test_idx])
        all_fold_trues.extend(fold_trues)
        all_fold_preds.extend(fold_preds)
        
    mean_cv_acc = np.mean(fold_accuracies)
    adapter_results[mag] = mean_cv_acc
    print(f"\n[{mag}] Final Average Cross-Validated Accuracy: {mean_cv_acc:.2f}%\n")

    # Save detailed patch classification outputs for this specific magnification
    print(f"Saving detailed patch predictions for tier {mag}...")
    detailed_df = pd.DataFrame({
        "Image_Path": all_fold_paths,
        "Patient_ID": all_fold_patients,
        "True_Label": le_binary.inverse_transform(all_fold_trues),
        "Predicted_Label": le_binary.inverse_transform(all_fold_preds),
        "Correct": np.array(all_fold_trues) == np.array(all_fold_preds)
    })
    detailed_csv_name = os.path.join(RESULTS_DIR, f"breakhis_plip_adapter_details_{mag}.csv")
    detailed_df.to_csv(detailed_csv_name, index=False)
    print(f"  -> Saved to {os.path.abspath(detailed_csv_name)}")

    # Train a full-tier production head for your interactive tester loop
    print(f"Finalizing full-tier production head for inference testing...")
    full_dataset = EmbeddingDataset(X, y_binary)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = PLIPBinaryAdapter(input_dim=768, hidden_dim=128).to(DEVICE)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        final_model.train()
        for inputs, targets in full_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            final_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(final_model(inputs), targets)
            loss.backward()
            final_optimizer.step()
            
    trained_models[mag] = {
        "model": final_model,
        "le_binary": le_binary
    }

# =========================================================================
# 6. OUTPUT SUMMARY LOGS
# =========================================================================
print("\n" + "="*60)
print("FINAL EXPERIMENT REPORT (PLIP + TRAINABLE ADAPTER HEAD)")
print("="*60)
for mag in adapter_results.keys():
    print(f"Magnification Tier {mag:<5} | Patient-Safe Cross Validation: {adapter_results[mag]:.2f}%")
print("="*60)

CSV_REPORT_PATH = os.path.join(RESULTS_DIR, "breakhis_plip_adapter_results.csv")
with open(CSV_REPORT_PATH, "w") as csv_file:
    csv_file.write("magnification,binary_adapter_accuracy\n")
    for mag in adapter_results.keys():
        csv_file.write(f"{mag},{adapter_results[mag]:.4f}\n")
print(f"Metrics summary spreadsheet written to: {os.path.abspath(CSV_REPORT_PATH)}")
print(f"Full pipeline log completely recorded in: {os.path.abspath(LOG_PATH)}")

# =========================================================================
# 7. INTERACTIVE ADAPTER TESTER LOOP
# =========================================================================
print("\n" + "═"*50)
print("   INTERACTIVE PLIP ADAPTER TESTER")
print("═"*50)

while True:
    test_path = input("\nEnter image path to classify (or type 'exit'): ").strip()
    if test_path.lower() == 'exit':
        break
        
    if not os.path.exists(test_path):
        print("Path invalid. Try again.")
        continue

    test_mag = input("Enter zoom tier (40X, 100X, 200X, 400X): ").strip().upper()
    if test_mag not in trained_models:
        print("No active model for that magnification.")
        continue

    print("Running forward pass through PLIP + Trainable Adapter...")
    try:
        raw_feat = extract_embeddings([test_path], batch_size=1)
        feat_tensor = torch.tensor(raw_feat, dtype=torch.float32).to(DEVICE)
        
        meta = trained_models[test_mag]
        meta["model"].eval()
        
        with torch.no_grad():
            output_logits = meta["model"](feat_tensor)
            pred_idx = torch.argmax(output_logits, dim=1).cpu().numpy()[0]
            
        predicted_class = meta["le_binary"].inverse_transform([pred_idx])[0]
        
        print("\n" + "─"*40)
        print(f"   IMAGE:   {os.path.basename(test_path)}")
        print(f"   ADAPTER BINARY PREDICTION: {predicted_class}")
        print("─"*40)
    except Exception as e:
        print(f"Inference run failed: {e}")