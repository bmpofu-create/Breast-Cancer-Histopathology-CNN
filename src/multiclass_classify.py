"""
BRACS 7-class subtype classifier.

The completed experiments pick this setup over the alternatives:
frozen PLIP image embeddings, then a small adapter
(768 -> 256 -> 7) trained with patient-isolated cross-validation.
That adapter reached 65.19% accuracy. Cosine 1-nearest-neighbor on the
same embeddings reached 55.31%. BreakHis 8-class subtyping stayed near
31-47% under patient-level splits, so this program classifies the
7 BRACS subtypes instead.

The three subtypes the adapter separates best, by out-of-fold F1, are
invasive carcinoma (0.90), flat epithelial atypia (0.74), and ductal
carcinoma in situ (0.73).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "BRACS_ROI"
MODEL_DIR = ROOT / "models"
CACHE_PATH = MODEL_DIR / "bracs_plip_cache.npz"
BUNDLE_PATH = MODEL_DIR / "bracs_multiclass.pt"
SUMMARY_PATH = MODEL_DIR / "bracs_multiclass_summary.json"

PLIP_MODEL_NAME = "vinid/plip"
HIDDEN_DIM = 256
DROPOUT = 0.3
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EMBED_BATCH_SIZE = 16
N_SPLITS = 5
NEIGHBORS = 8
SEED = 42

# Folder names used by the BRACS ROI release, matched case-insensitively.
SUBTYPE_FOLDERS = {
    "0_n": "Normal",
    "1_pb": "Pathological_Benign",
    "2_udh": "Usual_Ductal_Hyperplasia",
    "3_fea": "Flat_Epithelial_Atypia",
    "4_adh": "Atypical_Ductal_Hyperplasia",
    "5_dcis": "Ductal_Carcinoma_in_Situ",
    "6_ic": "Invasive_Carcinoma",
}

# Alphabetical, matching sklearn LabelEncoder on the names above.
CLASSES = [
    "Atypical_Ductal_Hyperplasia",
    "Ductal_Carcinoma_in_Situ",
    "Flat_Epithelial_Atypia",
    "Invasive_Carcinoma",
    "Normal",
    "Pathological_Benign",
    "Usual_Ductal_Hyperplasia",
]

DISPLAY_NAMES = {
    "Atypical_Ductal_Hyperplasia": "Atypical ductal hyperplasia",
    "Ductal_Carcinoma_in_Situ": "Ductal carcinoma in situ",
    "Flat_Epithelial_Atypia": "Flat epithelial atypia",
    "Invasive_Carcinoma": "Invasive carcinoma",
    "Normal": "Normal",
    "Pathological_Benign": "Pathological benign",
    "Usual_Ductal_Hyperplasia": "Usual ductal hyperplasia",
}

# Out-of-fold F1 from bracs_7class_subtyping_log.txt (PLIP adapter).
EXPERIMENT_F1 = {
    "Invasive_Carcinoma": 0.90,
    "Flat_Epithelial_Atypia": 0.74,
    "Ductal_Carcinoma_in_Situ": 0.73,
    "Normal": 0.68,
    "Pathological_Benign": 0.60,
    "Usual_Ductal_Hyperplasia": 0.42,
    "Atypical_Ductal_Hyperplasia": 0.39,
}

BEST_SUBCLASSES = [
    "Invasive_Carcinoma",
    "Flat_Epithelial_Atypia",
    "Ductal_Carcinoma_in_Situ",
]

CLASS_DEFINITIONS = {
    "Invasive_Carcinoma": (
        "Invasive carcinoma means malignant epithelial cells have grown beyond "
        "the duct or lobule into the surrounding stroma."
    ),
    "Flat_Epithelial_Atypia": (
        "Flat epithelial atypia is a low-grade change in which dilated acini "
        "are lined by one or a few layers of monomorphic columnar cells."
    ),
    "Ductal_Carcinoma_in_Situ": (
        "Ductal carcinoma in situ is a neoplastic epithelial proliferation that "
        "stays inside the duct."
    ),
    "Normal": (
        "Normal denotes breast tissue without a diagnostic proliferative or "
        "atypical lesion."
    ),
    "Pathological_Benign": (
        "Pathological benign groups non-atypical benign lesions, such as "
        "adenosis or fibroadenoma."
    ),
    "Usual_Ductal_Hyperplasia": (
        "Usual ductal hyperplasia is a mixed benign epithelial proliferation "
        "filling a duct, without the uniform atypia of atypical ductal "
        "hyperplasia or ductal carcinoma in situ."
    ),
    "Atypical_Ductal_Hyperplasia": (
        "Atypical ductal hyperplasia is a small low-grade atypical intraductal "
        "proliferation that falls short of ductal carcinoma in situ."
    ),
}


class SubtypeAdapter(nn.Module):
    """Same head as the 7-class experiment: one hidden layer, then 7 logits."""

    def __init__(self, input_dim: int, num_classes: int = 7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def scan_rois(dataset_root: Path = DATASET_ROOT) -> list[dict]:
    """Collect ROI paths, subtype labels, and patient ids."""
    records = []
    for path in sorted(dataset_root.rglob("*.png")):
        parts = [part.lower() for part in path.parts]
        folder = next((key for key in SUBTYPE_FOLDERS if key in parts), None)
        if folder is None:
            continue
        tokens = path.stem.split("_")
        if len(tokens) < 2 or tokens[0].upper() != "BRACS":
            continue
        records.append(
            {
                "path": str(path),
                "label": SUBTYPE_FOLDERS[folder],
                "patient": f"BRACS_{tokens[1]}",
            }
        )
    if not records:
        raise FileNotFoundError(f"No BRACS ROI images found under {dataset_root}")
    return records


def _embedding_from_output(outputs: torch.Tensor | object) -> torch.Tensor:
    """Match the experiment: mean-pool the vision token sequence to 768-d."""
    if torch.is_tensor(outputs):
        features = outputs
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        features = outputs.image_embeds
    else:
        features = outputs[0]
    if features.ndim == 3:
        features = features.mean(dim=1)
    return features


def load_plip(model_name: str = PLIP_MODEL_NAME, device: torch.device | None = None):
    from transformers import AutoModel, AutoProcessor

    if device is None:
        device = torch.device("cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model.get_image_features, device


def extract_embeddings(
    records: list[dict],
    processor,
    image_encoder,
    device: torch.device,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    paths = [record["path"] for record in records]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = _embedding_from_output(image_encoder(**inputs))
            chunks.append(features.cpu().numpy().astype(np.float32))
            done = min(start + batch_size, len(paths))
            print(f"  embedded {done}/{len(paths)}", flush=True)
    return np.vstack(chunks)


def load_or_extract_embeddings(records: list[dict]) -> np.ndarray:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = np.array([record["path"] for record in records])
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=False)
        if len(cached["paths"]) == len(paths) and np.array_equal(cached["paths"], paths):
            print(f"Using cached embeddings from {CACHE_PATH}")
            return cached["embeddings"]
        print("Embedding cache does not match the current ROI list. Re-extracting.")

    print("Loading PLIP and extracting embeddings...")
    processor, image_encoder, device = load_plip()
    embeddings = extract_embeddings(records, processor, image_encoder, device)
    np.savez_compressed(
        CACHE_PATH,
        embeddings=embeddings,
        paths=paths,
        labels=np.array([record["label"] for record in records]),
        patients=np.array([record["patient"] for record in records]),
    )
    print(f"Saved embeddings {embeddings.shape} to {CACHE_PATH}")
    return embeddings


def _train_adapter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> SubtypeAdapter:
    _set_seed(seed)
    device = torch.device("cpu")
    model = SubtypeAdapter(input_dim=x_train.shape[1], num_classes=len(CLASSES)).to(device)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    drop_last = len(dataset) % BATCH_SIZE == 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _predict_logits(model: SubtypeAdapter, embeddings: np.ndarray) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(embeddings, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    chunks = []
    with torch.inference_mode():
        for (inputs,) in loader:
            chunks.append(model(inputs).cpu().numpy())
    return np.vstack(chunks)


def cross_validate(embeddings: np.ndarray, labels: np.ndarray, patients: np.ndarray) -> dict:
    """Patient-isolated 5-fold evaluation, matching the 7-class experiment."""
    y = np.array([CLASSES.index(label) for label in labels], dtype=np.int64)
    oof = np.zeros(len(y), dtype=np.int64)
    fold_accuracies = []
    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(embeddings, y, groups=patients)):
        model = _train_adapter(embeddings[train_idx], y[train_idx], seed=SEED + fold)
        pred = _predict_logits(model, embeddings[test_idx]).argmax(axis=1)
        oof[test_idx] = pred
        fold_acc = accuracy_score(y[test_idx], pred)
        fold_accuracies.append(float(fold_acc))
        print(f"  fold {fold + 1}/{N_SPLITS}: accuracy {fold_acc * 100:.2f}%")

    report = classification_report(
        y,
        oof,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=2,
        zero_division=0,
    )
    per_class_f1 = f1_score(y, oof, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    f1_map = {name: float(score) for name, score in zip(CLASSES, per_class_f1)}
    accuracy = float(accuracy_score(y, oof))
    print("\nPatient-isolated out-of-fold report")
    print(report)
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    return {
        "accuracy": accuracy,
        "fold_accuracies": fold_accuracies,
        "f1": f1_map,
        "report": report,
    }


def fit_and_save(embeddings: np.ndarray, labels: np.ndarray, patients: np.ndarray, cv: dict) -> None:
    y = np.array([CLASSES.index(label) for label in labels], dtype=np.int64)
    model = _train_adapter(embeddings, y, seed=SEED)
    bundle = {
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "input_dim": int(embeddings.shape[1]),
        "plip_model": PLIP_MODEL_NAME,
        "reference_embeddings": embeddings.astype(np.float32),
        "reference_labels": np.array(labels),
        "cv_accuracy": cv["accuracy"],
        "cv_f1": cv["f1"],
        "experiment_f1": EXPERIMENT_F1,
        "best_subclasses": BEST_SUBCLASSES,
        "n_reference": int(len(labels)),
    }
    torch.save(bundle, BUNDLE_PATH)
    summary = {
        "cv_accuracy": cv["accuracy"],
        "fold_accuracies": cv["fold_accuracies"],
        "cv_f1": cv["f1"],
        "experiment_f1": EXPERIMENT_F1,
        "best_subclasses": BEST_SUBCLASSES,
        "n_rois": int(len(labels)),
        "n_patients": int(len(set(patients.tolist()))),
        "bundle": str(BUNDLE_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (MODEL_DIR / "bracs_multiclass_cv_report.txt").write_text(cv["report"], encoding="utf-8")
    print(f"Saved classifier to {BUNDLE_PATH}")


def train() -> None:
    records = scan_rois()
    patients = np.array([record["patient"] for record in records])
    labels = np.array([record["label"] for record in records])
    print(f"Found {len(records)} ROIs from {len(set(patients.tolist()))} patients.")
    embeddings = load_or_extract_embeddings(records)
    if embeddings.shape[1] != 768:
        print(f"Feature dimension is {embeddings.shape[1]} (the original experiment used 768).")
    print("Running patient-isolated 5-fold evaluation...")
    cv = cross_validate(embeddings, labels, patients)
    print("Fitting the deployed adapter on all ROIs...")
    fit_and_save(embeddings, labels, patients, cv)


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train it with: python src/multiclass_classify.py --train"
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def explain(
    probabilities: np.ndarray,
    embedding: np.ndarray,
    bundle: dict,
) -> dict:
    classes = list(bundle["classes"])
    labels = np.array([str(label) for label in bundle["reference_labels"]])
    order = np.argsort(probabilities)[::-1]
    predicted = classes[int(order[0])]
    runner_up = classes[int(order[1])]
    confidence = float(probabilities[order[0]])
    runner_confidence = float(probabilities[order[1]])

    query = embedding.astype(np.float32).reshape(1, -1)
    query_norm = _l2_normalize(query)[0]
    reference_norm = _l2_normalize(bundle["reference_embeddings"].astype(np.float32))
    similarities = reference_norm @ query_norm

    # A copied training image would otherwise cite itself as the evidence.
    candidate = similarities.copy()
    candidate[candidate > 0.999] = -1.0
    neighbor_index = np.argsort(candidate)[::-1][:NEIGHBORS]
    neighbor_labels = labels[neighbor_index]
    neighbor_sims = candidate[neighbor_index]

    prototype_scores = {}
    for name in classes:
        members = reference_norm[labels == name]
        centroid = members.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        prototype_scores[name] = float(np.dot(query_norm, centroid))
    prototype_rank = sorted(prototype_scores, key=prototype_scores.get, reverse=True)

    experiment_f1 = bundle.get("experiment_f1", EXPERIMENT_F1)
    best = list(bundle.get("best_subclasses", BEST_SUBCLASSES))
    reason = _reason_text(
        predicted=predicted,
        runner_up=runner_up,
        confidence=confidence,
        runner_confidence=runner_confidence,
        prototype_scores=prototype_scores,
        prototype_rank=prototype_rank,
        neighbor_labels=neighbor_labels,
        neighbor_sims=neighbor_sims,
        experiment_f1=experiment_f1,
        best=best,
    )
    return {
        "predicted_class": predicted,
        "display_name": DISPLAY_NAMES[predicted],
        "confidence": confidence,
        "runner_up": runner_up,
        "runner_up_name": DISPLAY_NAMES[runner_up],
        "runner_up_confidence": runner_confidence,
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(classes)},
        "prototype_scores": prototype_scores,
        "neighbors": [
            {
                "label": str(label),
                "display_name": DISPLAY_NAMES[str(label)],
                "similarity": float(score),
            }
            for label, score in zip(neighbor_labels, neighbor_sims)
            if score >= 0
        ],
        "reason": reason,
        "is_best_subclass": predicted in best,
        "experiment_f1": float(experiment_f1[predicted]),
    }


def _reason_text(
    predicted: str,
    runner_up: str,
    confidence: float,
    runner_confidence: float,
    prototype_scores: dict[str, float],
    prototype_rank: list[str],
    neighbor_labels: np.ndarray,
    neighbor_sims: np.ndarray,
    experiment_f1: dict,
    best: list[str],
) -> str:
    pred_name = DISPLAY_NAMES[predicted]
    second_name = DISPLAY_NAMES[runner_up]
    margin = confidence - runner_confidence
    decision = (
        f"Classified as {pred_name} because the adapter assigns that subtype "
        f"{confidence * 100:.1f}% probability, "
        f"{margin * 100:.1f} percentage points ahead of {second_name} "
        f"({runner_confidence * 100:.1f}%)."
    )

    nearest_proto = prototype_rank[0]
    next_proto = prototype_rank[1]
    prototype_gap = prototype_scores[nearest_proto] - prototype_scores[next_proto]
    if nearest_proto == predicted and prototype_gap >= 0.01:
        prototype = (
            f"The PLIP embedding agrees: it is closest to the {pred_name} prototype "
            f"(cosine {prototype_scores[nearest_proto]:.2f}), ahead of "
            f"{DISPLAY_NAMES[next_proto]} ({prototype_scores[next_proto]:.2f})."
        )
    elif nearest_proto == predicted:
        prototype = (
            f"The {pred_name} prototype is ranked first, but only narrowly "
            f"(cosine {prototype_scores[nearest_proto]:.3f} versus "
            f"{prototype_scores[next_proto]:.3f} for {DISPLAY_NAMES[next_proto]})."
        )
    else:
        prototype = (
            f"The nearest class prototype is {DISPLAY_NAMES[nearest_proto]} "
            f"(cosine {prototype_scores[nearest_proto]:.2f}), while the {pred_name} "
            f"prototype is at cosine {prototype_scores[predicted]:.2f}. "
            f"The reported label follows the adapter, which outperformed nearest-neighbor "
            f"search in the patient-isolated experiments (65.19% versus 55.31% accuracy)."
        )

    valid = [(label, float(score)) for label, score in zip(neighbor_labels, neighbor_sims) if score >= 0]
    agree = [item for item in valid if item[0] == predicted]
    if not valid:
        neighbors = "No distinct reference region was available for comparison."
    elif not agree:
        neighbors = (
            f"None of the {len(valid)} closest reference regions are {pred_name}. "
            f"The closest is {DISPLAY_NAMES[valid[0][0]]} at cosine {valid[0][1]:.2f}."
        )
    else:
        neighbors = (
            f"{len(agree)} of the {len(valid)} closest reference regions are {pred_name}. "
            f"The nearest matching region has cosine similarity {agree[0][1]:.2f}."
        )

    definition = (
        f"{CLASS_DEFINITIONS[predicted]} That sentence is the meaning of the class, "
        f"not a structure measured in this image."
    )
    f1 = float(experiment_f1[predicted])
    if predicted in best:
        reliability = (
            f"{pred_name} is one of the three subtypes these experiments classify best "
            f"(patient-isolated F1 {f1:.2f})."
        )
    else:
        reliability = (
            f"{pred_name} is harder to separate than invasive carcinoma, flat epithelial "
            f"atypia, and ductal carcinoma in situ (patient-isolated F1 {f1:.2f}), "
            f"so this call is less dependable."
        )
    return " ".join([decision, prototype, neighbors, definition, reliability])


def embed_image(image: Image.Image, processor, image_encoder, device: torch.device) -> np.ndarray:
    rgb = image.convert("RGB")
    inputs = processor(images=[rgb], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = _embedding_from_output(image_encoder(**inputs))
    return features.cpu().numpy().astype(np.float32)


def classify_image(image: Image.Image, bundle: dict, processor, image_encoder, device) -> dict:
    embedding = embed_image(image, processor, image_encoder, device)
    model = SubtypeAdapter(input_dim=bundle["input_dim"], num_classes=len(bundle["classes"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(embedding, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
    return explain(probabilities, embedding[0], bundle)


def classify_path(image_path: Path) -> dict:
    bundle = load_bundle()
    processor, image_encoder, device = load_plip(bundle["plip_model"])
    with Image.open(image_path) as image:
        result = classify_image(image, bundle, processor, image_encoder, device)
    result["image"] = str(image_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the BRACS 7-class subtype classifier.")
    parser.add_argument("--train", action="store_true", help="Extract PLIP features, evaluate, and save the adapter.")
    parser.add_argument("--image", type=str, help="Classify one image and print the class and the reason.")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.image:
        result = classify_path(Path(args.image))
        print(result["display_name"])
        print(f"confidence: {result['confidence']:.1%}")
        print(result["reason"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
