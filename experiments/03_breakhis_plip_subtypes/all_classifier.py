import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoProcessor, AutoModel

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PLIP_MODEL_NAME = "vinid/plip"
MAGNIFICATION_TIERS = ["40X", "100X", "200X", "400X"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "classifier_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)
DATASET_ROOT = os.path.join(REPO_ROOT, "BreaKHis_v1", "BreaKHis_v1")

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
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                continue

        if not batch_images:
            continue

        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = image_encoder(**inputs)
            features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
            if len(features.shape) == 3:
                features = torch.mean(features, dim=1)
            
            l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)
            normalized_features = features / l2_norm
            all_embeddings.append(normalized_features.cpu().numpy())
            
    return np.vstack(all_embeddings)

# -------------------------------------------------------------------------
# 4. CUSTOM SMALL MLP CLASSIFIER ARCHITECTURE
# -------------------------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SmallMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train_mlp(X_train, y_train, num_classes, epochs=25, batch_size=32):
    input_dim = X_train.shape[1]
    mlp_model = SmallMLP(input_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    mlp_model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = mlp_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return mlp_model

# -------------------------------------------------------------------------
# 5. CORE EVALUATION ENGINE
# -------------------------------------------------------------------------
results_registry = {}

print("\nScanning dataset directory tree...")
search_pattern = os.path.join(DATASET_ROOT, "**", "*.png")
all_png_files = glob.glob(search_pattern, recursive=True)
print(f"Total localized image files: {len(all_png_files)}")

for mag in MAGNIFICATION_TIERS:
    print(f"\n{'='*50}\nProcessing Magnification Tier: {mag}\n{'='*50}")
    
    valid_paths, binary_labels, subtype_labels, patient_ids = [], [], [], []
    
    for path in all_png_files:
        normalized_path = path.replace("\\", "/")
        path_parts = normalized_path.lower().split("/")
        
        if mag.lower() in path_parts and ("benign" in path_parts or "malignant" in path_parts):
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
        
    le_binary = LabelEncoder().fit(binary_labels)
    le_subtype = LabelEncoder().fit(subtype_labels)
    y_binary_all = le_binary.transform(binary_labels)
    y_subtype_all = le_subtype.transform(subtype_labels)

    # Patient Split Strategy
    unique_patients = list(set(patient_ids))
    patient_to_subtype = {p_id: sub_lbl for p_id, sub_lbl in zip(patient_ids, y_subtype_all)}
    unique_patient_subtypes = [patient_to_subtype[p] for p in unique_patients]
    
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42, stratify=unique_patient_subtypes
    )
    
    idx_train = [i for i, p_id in enumerate(patient_ids) if p_id in train_patients]
    idx_test = [i for i, p_id in enumerate(patient_ids) if p_id in test_patients]

    print("Extracting feature representations...")
    X_all = extract_embeddings(valid_paths, batch_size=32)
    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_bin_train, y_bin_test = y_binary_all[idx_train], y_binary_all[idx_test]
    y_sub_train, y_sub_test = y_subtype_all[idx_train], y_subtype_all[idx_test]
    
    # Establish Patient-Level Ground Truth Maps
    test_image_patients = [patient_ids[i] for i in idx_test]
    patient_true_bin, patient_true_sub = {}, {}
    for p, b, s in zip(test_image_patients, y_bin_test, y_sub_test):
        patient_true_bin[p] = b
        patient_true_sub[p] = s
    ordered_test_patients = list(patient_true_bin.keys())
    y_true_bin_patient = [patient_true_bin[p] for p in ordered_test_patients]
    y_true_sub_patient = [patient_true_sub[p] for p in ordered_test_patients]

    # Model Dictionary Initializations
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear SVM": LinearSVC(max_iter=2000, C=0.1, dual=False)
    }
    classifiers["Logistic Regression"].fit(X_train, y_sub_train)
    classifiers["Linear SVM"].fit(X_train, y_sub_train)
    mlp_estimator = train_mlp(X_train, y_sub_train, num_classes=len(le_subtype.classes_))
    
    results_registry[mag] = {
        "metadata": {"total_p": len(unique_patients), "train_p": len(train_patients), "test_p": len(test_patients)},
        "models": {}
    }
    
    # Evaluate across all 4 targeted setups
    for name in ["Cosine Similarity", "Logistic Regression", "Linear SVM", "Small MLP Classifier"]:
        if name == "Cosine Similarity":
            sim = np.dot(X_test, X_train.T)
            img_preds = y_sub_train[np.argmax(sim, axis=1)]
        elif name == "Small MLP Classifier":
            mlp_estimator.eval()
            with torch.no_grad():
                logits = mlp_estimator(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
                img_preds = torch.argmax(logits, dim=1).cpu().numpy()
        else:
            img_preds = classifiers[name].predict(X_test)
            
        # Convert image-level predictions to PATIENT-LEVEL via Majority Vote
        votes = {p: [] for p in ordered_test_patients}
        for p, pred in zip(test_image_patients, img_preds):
            votes[p].append(pred)
            
        y_pred_sub_patient = [Counter(votes[p]).most_common(1)[0][0] for p in ordered_test_patients]
        pred_sub_names = le_subtype.inverse_transform(y_pred_sub_patient)
        y_pred_bin_patient = [le_binary.transform(["Benign" if "B" in s_name or "BENIGN" in s_name.upper() else "Malignant"])[0] for s_name in pred_sub_names]
        
        # Calculate Validation Metrics
        bin_acc = accuracy_score(y_true_bin_patient, y_pred_bin_patient) * 100
        sub_acc = accuracy_score(y_true_sub_patient, y_pred_sub_patient) * 100
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true_sub_patient, y_pred_sub_patient, average='macro', zero_division=0)
        cm = confusion_matrix(y_true_sub_patient, y_pred_sub_patient)
        
        results_registry[mag]["models"][name] = {
            "bin_acc": bin_acc, "sub_acc": sub_acc, "p_mac": p_mac, "r_mac": r_mac, "f1_mac": f1_mac, "cm": cm
        }

# -------------------------------------------------------------------------
# 6. COMPREHENSIVE REFACTORING AND DUMP ENGINE
# -------------------------------------------------------------------------
TXT_REPORT_PATH = os.path.join(RESULTS_DIR, "breakhis_comprehensive_patient_metrics.txt")
CSV_REPORT_PATH = os.path.join(RESULTS_DIR, "breakhis_comprehensive_patient_metrics.csv")

with open(TXT_REPORT_PATH, "w", encoding="utf-8") as txt:
    txt.write("=========================================================================\n")
    txt.write("       BreaKHis Multi-Classifier Strict Patient-Level Verification       \n")
    txt.write("=========================================================================\n\n")
    
    for mag in results_registry.keys():
        meta = results_registry[mag]["metadata"]
        txt.write(f"\n{'='*80}\nMAGNIFICATION LEVEL: {mag}\n{'='*80}\n")
        txt.write(f"Cohort Split Details: Unique Patients Total = {meta['total_p']} | Train = {meta['train_p']} | Test = {meta['test_p']}\n\n")
        
        for model_name, metrics in results_registry[mag]["models"].items():
            txt.write(f"--- Classifier: {model_name} ---\n")
            txt.write(f"  └─ Binary Accuracy   : {metrics['bin_acc']:.2f}%\n")
            txt.write(f"  └─ Subtype Accuracy  : {metrics['sub_acc']:.2f}%\n")
            txt.write(f"  └─ Macro Precision   : {metrics['p_mac']:.4f}\n")
            txt.write(f"  └─ Macro Recall      : {metrics['r_mac']:.4f}\n")
            txt.write(f"  └─ Macro F1-Score    : {metrics['f1_mac']:.4f}\n")
            txt.write(f"  └─ Patient Confusion Matrix:\n{np.array2string(metrics['cm'])}\n\n")

with open(CSV_REPORT_PATH, "w") as csv:
    csv.write("magnification,classifier,binary_accuracy,subtype_accuracy,macro_precision,macro_recall,macro_f1_score\n")
    for mag in results_registry.keys():
        for model_name, metrics in results_registry[mag]["models"].items():
            csv.write(f"{mag},{model_name},{metrics['bin_acc']:.2f},{metrics['sub_acc']:.2f},{metrics['p_mac']:.4f},{metrics['r_mac']:.4f},{metrics['f1_mac']:.4f}\n")

print(f"\nExecution finished successfully.\nStructured text files logged: {os.path.abspath(TXT_REPORT_PATH)}\nSpreadsheets logged: {os.path.abspath(CSV_REPORT_PATH)}")