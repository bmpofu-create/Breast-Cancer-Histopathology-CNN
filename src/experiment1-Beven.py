import os
import json
from pathlib import Path
from datetime import datetime
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. INITIALIZATION & AUTHENTICATION ---
load_dotenv()
client = OpenAI()

# Connect to your existing ChromaDB database
chroma_client = chromadb.PersistentClient(path="breakhis_vector_db")
collection = chroma_client.get_collection(name="breakhis_dataset")


# --- 2. HARDWARE ACCELERATION & FEATURE EXTRACTOR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).flatten().tolist()
    return embedding


# --- 3. CORE RAG PIPELINE FUNCTION ---
def run_image_rag(query_image_path, magnification, top_k=3, generate_report=False):
    """
    Modified to make generate_report optional. 
    When calculating overall accuracy, skipping LLM report generation saves API costs and time.
    """
    query_vector = get_image_embedding(query_image_path)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"magnification": str(magnification)}
    )
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    if not metadatas:
        return None

    # [Classification Step] Majority Vote Logic
    classes_retrieved = [m['class'].lower() for m in metadatas] # Lowercase for safe comparison
    final_classification = max(set(classes_retrieved), key=classes_retrieved.count)
    
    report_content = "LLM Report Generation Skipped during evaluation."
    
    # Only hit the OpenAI API if explicitly requested (saves credits during accuracy testing)
    if generate_report:
        references_text = ""
        for i, meta in enumerate(metadatas):
            references_text += f"- Reference Match {i+1}: Source File: {meta['filename']} | Confirmed Category: {meta['class']} | Pathological Subtype: {meta['subtype']} (Distance Score: {distances[i]:.4f})\n"

        prompt = f"""
        You are an expert AI Pathology Assistant specializing in histopathology.
        A new breast biopsy tissue image taken at {magnification}X magnification has been processed by our visual vector retrieval system.
        The algorithm classified this sample as: **{final_classification}**
        The system retrieved the top {top_k} matches:
        {references_text}
        Please generate a structured 'Histopathology Reference & Verification Report'.
        """
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        report_content = response.choices[0].message.content
        
    return {
        "classification_category": final_classification,
        "references": metadatas,
        "report": report_content
    }


# === 🚀 BULK EVALUATION & ACCURACY PIPELINE ===
# Define the root path to your evaluation dataset. 
# Structuring it like: test_dataset/benign/ and test_dataset/malignant/ makes validation easy.
TEST_DATASET_DIR = Path("C:/Users/beven/MTU/Projects/test_dataset") 
SAMPLE_ZOOM = "200" s
OUTPUT_DIR = Path("C:/Users/beven/MTU/Projects/output_reports")

if __name__ == "__main__":
    print("🏁 Evaluation Suite Started...")
    
    if not TEST_DATASET_DIR.exists():
        print(f"⚠️ Please place your testing dataset at: {TEST_DATASET_DIR}")
        print("💡 Hint: Organize it into subfolders 'benign' and 'malignant' containing your test .jpg images.")
    else:
        # Supported extensions
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        test_images = [p for p in TEST_DATASET_DIR.rglob('*') if p.suffix.lower() in valid_extensions]
        
        if not test_images:
            print(f"❌ No test images found in {TEST_DATASET_DIR}")
            exit()
            
        print(f"📷 Found {len(test_images)} images. Testing at zoom {SAMPLE_ZOOM}X via RAG voting mechanism...")
        
        # Performance trackers
        metrics = {
            "total": 0,
            "correct": 0,
            "benign_total": 0,
            "benign_correct": 0,
            "malignant_total": 0,
            "malignant_correct": 0
        }
        
        detailed_results = []

        for idx, img_path in enumerate(test_images, start=1):
            # 1. Determine Ground Truth from directory structure or filename
            # Checks if 'benign' or 'malignant' is in the path structure
            path_str = str(img_path).lower()
            if "benign" in path_str:
                ground_truth = "benign"
                metrics["benign_total"] += 1
            elif "malignant" in path_str:
                ground_truth = "malignant"
                metrics["malignant_total"] += 1
            else:
                # If structure isn't folder-based, skip or default
                print(f"⚠️ Skipping {img_path.name}: Cannot infer true class label from path/filename string.")
                continue
            
            metrics["total"] += 1
            
            # 2. Run RAG Pipeline (Set generate_report=False to skip LLM cost during mass testing)
            output = run_image_rag(img_path, magnification=SAMPLE_ZOOM, generate_report=False)
            
            if output is None:
                print(f"[{idx}/{len(test_images)}] ❌ {img_path.name} -> No matches found in Vector DB.")
                predicted_class = "unknown"
                is_correct = False
            else:
                predicted_class = output["classification_category"].lower()
                is_correct = (predicted_class == ground_truth)
                
                if is_correct:
                    metrics["correct"] += 1
                    if ground_truth == "benign":
                        metrics["benign_correct"] += 1
                    else:
                        metrics["malignant_correct"] += 1

            status_marker = "✅" if is_correct else "❌"
            print(f"[{idx}/{len(test_images)}] {status_marker} File: {img_path.name} | Truth: {ground_truth.upper()} | Predicted: {predicted_class.upper()}")
            
            detailed_results.append({
                "filename": img_path.name,
                "ground_truth": ground_truth,
                "predicted": predicted_class,
                "correct": is_correct
            })

        # --- 4. CALCULATE FINAL CLASSIFICATION METRICS ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        total_acc = (metrics["correct"] / metrics["total"]) * 100 if metrics["total"] > 0 else 0
        benign_acc = (metrics["benign_correct"] / metrics["benign_total"]) * 100 if metrics["benign_total"] > 0 else 0
        malignant_acc = (metrics["malignant_correct"] / metrics["malignant_total"]) * 100 if metrics["malignant_total"] > 0 else 0

        summary_report = f"""==================================================================
📊 FINAL HISTOPATHOLOGY RAG CLASSIFICATION METRICS
==================================================================
Evaluation Timestamp : {timestamp}
Magnification Filter : {SAMPLE_ZOOM}X
Total Images Tested  : {metrics['total']}
Overall Accuracy     : {total_acc:.2f}% ({metrics['correct']}/{metrics['total']})

Breakdown by Category:
------------------------------------------------------------------
🟢 Benign Accuracy   : {benign_acc:.2f}% ({metrics['benign_correct']}/{metrics['benign_total']})
🔴 Malignant Accuracy: {malignant_acc:.2f}% ({metrics['malignant_correct']}/{metrics['malignant_total']})
=================================================================="""

        print(f"\n{summary_report}")
        
        # Save evaluation metrics summary & detailed payload log
        with open(OUTPUT_DIR / f"evaluation_metrics_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(summary_report)
            
        evaluation_payload = {
            "summary_metrics": {
                "overall_accuracy_percent": total_acc,
                "benign_accuracy_percent": benign_acc,
                "malignant_accuracy_percent": malignant_acc,
                "raw_counts": metrics
            },
            "detailed_runs": detailed_results
        }
        
        with open(OUTPUT_DIR / f"evaluation_run_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=4, ensure_ascii=False)
            
        print(f"\n🎉 RESULTS SAVED. Metrics saved to output folder: {OUTPUT_DIR}")