import os
import json
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModel

# --- 1. INITIALIZATION & AUTHENTICATION ---
load_dotenv()
client = OpenAI()

# Connect to your existing ChromaDB database
# IMPORTANT: If this DB was built using ResNet18, you must re-index it using PLIP embeddings!
chroma_client = chromadb.PersistentClient(path="breakhis_vector_db")
collection = chroma_client.get_collection(name="breakhis_dataset")


# --- 2. MEDICAL MODEL ACCELERATION & FEATURE EXTRACTOR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🧬 Loading medical-specific vision backbone (PLIP)...")
MODEL_NAME = "vinid/plip"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_image_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get the PLIP output object
        outputs = model.get_image_features(**inputs)
        
        # Extract the raw tensor hidden layer from the wrapper object
        image_features = outputs.image_embeds
        
        # Normalize the vector for accurate cosine similarity inside ChromaDB
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = image_features.flatten().tolist()
        
    return embedding


# --- 3. CORE RAG PIPELINE FUNCTION ---
def run_image_rag(query_image_path, magnification, top_k=3, generate_report=False):
    query_vector = get_image_embedding(query_image_path)
    
    # Query ChromaDB matching specifically on the targeted zoom filter
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"magnification": str(magnification)}
    )
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    if not metadatas:
        return None

    # Majority Vote Logic
    classes_retrieved = [m['class'].lower() for m in metadatas] 
    final_classification = max(set(classes_retrieved), key=classes_retrieved.count)
    
    report_content = "LLM Report Generation Skipped during evaluation."
    
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


# === 🚀 MULTI-MAGNIFICATION EVALUATION PIPELINE ===
TEST_DATASET_DIR = Path("C:/Users/beven/MTU/Projects/test_dataset") 
OUTPUT_DIR = Path("C:/Users/beven/MTU/Projects/output_reports")

# Iterating across all standard BreakHis magnification metrics tiers
MAGNIFICATION_LEVELS = ["40", "100", "200", "400"]

if __name__ == "__main__":
    print("🏁 Multi-Magnification Evaluation Suite Started...")
    
    if not TEST_DATASET_DIR.exists():
        print(f"⚠️ Please place your testing dataset root folder at: {TEST_DATASET_DIR}")
        print("💡 Expected Structure:\n test_dataset/40/benign/\n test_dataset/40/malignant/\n test_dataset/100/benign/... etc.")
        exit()
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_magnification_metrics = {}
    detailed_global_log = []

    # Outer loop parses through every magnification category tier
    for mag in MAGNIFICATION_LEVELS:
        print(f"\n==================================================================")
        print(f"🔬 STARTING EVALUATION FOR MAGNIFICATION: {mag}X")
        print(f"==================================================================")
        
        # Strategy A: Check if images reside inside structured directory blocks (e.g., /test_dataset/200/benign/)
        mag_images = [
            p for p in TEST_DATASET_DIR.rglob('*') 
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and f"{os.sep}{mag}{os.sep}" in str(p)
        ]
        
        # Strategy B Fallback: Look for magnification markers inside string names directly (e.g., sob_b_a_14-22549ab-200-001.png)
        if not mag_images:
            mag_images = [
                p for p in TEST_DATASET_DIR.rglob('*') 
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and f"_{mag}_" in p.name
            ]

        if not mag_images:
            print(f"⚠️ No matching validation files found for {mag}X zoom. Skipping execution...")
            continue
            
        print(f"📷 Found {len(mag_images)} images matching {mag}X zoom factors.")
        
        metrics = {
            "total": 0,
            "correct": 0,
            "benign_total": 0,
            "benign_correct": 0,
            "malignant_total": 0,
            "malignant_correct": 0
        }

        for idx, img_path in enumerate(mag_images, start=1):
            path_str = str(img_path).lower()
            if "benign" in path_str:
                ground_truth = "benign"
                metrics["benign_total"] += 1
            elif "malignant" in path_str:
                ground_truth = "malignant"
                metrics["malignant_total"] += 1
            else:
                continue # Skip stray operating system files (like Desktop.ini)
            
            metrics["total"] += 1
            
            # Execute pipeline tracking calculation weights only
            output = run_image_rag(img_path, magnification=mag, generate_report=False)
            
            if output is None:
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
            # Periodic print statements to prevent terminal buffer overload across huge batch loops
            if idx % 50 == 0 or idx == len(mag_images): 
                print(f"   [{idx}/{len(mag_images)}] {status_marker} Last Processed: {img_path.name} | Truth: {ground_truth.upper()} | Predicted: {predicted_class.upper()}")
            
            detailed_global_log.append({
                "magnification": mag,
                "filename": img_path.name,
                "ground_truth": ground_truth,
                "predicted": predicted_class,
                "correct": is_correct
            })

        # Process final validation mathematics per magnification cycle
        total_acc = (metrics["correct"] / metrics["total"]) * 100 if metrics["total"] > 0 else 0
        benign_acc = (metrics["benign_correct"] / metrics["benign_total"]) * 100 if metrics["benign_total"] > 0 else 0
        malignant_acc = (metrics["malignant_correct"] / metrics["malignant_total"]) * 100 if metrics["malignant_total"] > 0 else 0
        
        all_magnification_metrics[mag] = {
            "overall_accuracy_percent": total_acc,
            "benign_accuracy_percent": benign_acc,
            "malignant_accuracy_percent": malignant_acc,
            "counts": metrics
        }

    # --- 4. GENERATE COMPREHENSIVE MULTI-ZOOM SUMMARY REPORT ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_report = f"""==================================================================
📊 FINAL COMPREHENSIVE HISTOPATHOLOGY RAG CLASSIFICATION METRICS
==================================================================
Evaluation Timestamp : {timestamp}
Model Embedding Type : {MODEL_NAME.upper()}
==================================================================\n"""

    for mag, m_data in all_magnification_metrics.items():
        c = m_data["counts"]
        summary_report += f"""🔍 MAGNIFICATION LEVEL: {mag}X
------------------------------------------------------------------
Total Images Evaluated: {c['total']}
Overall Accuracy      : {m_data['overall_accuracy_percent']:.2f}% ({c['correct']}/{c['total']})
🟢 Benign Accuracy     : {m_data['benign_accuracy_percent']:.2f}% ({c['benign_correct']}/{c['benign_total']})
🔴 Malignant Accuracy  : {m_data['malignant_accuracy_percent']:.2f}% ({c['malignant_correct']}/{c['malignant_total']})
------------------------------------------------------------------\n"""

    summary_report += "=================================================================="
    print(f"\n{summary_report}")
    
    # Save validation metadata dumps
    with open(OUTPUT_DIR / f"global_magnification_metrics_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(summary_report)
        
    evaluation_payload = {
        "summary_per_magnification": all_magnification_metrics,
        "detailed_runs": detailed_global_log
    }
    
    with open(OUTPUT_DIR / f"global_evaluation_run_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 GLOBAL EXPERIMENT COMPLETE. Metrics written to output folder: {OUTPUT_DIR}")