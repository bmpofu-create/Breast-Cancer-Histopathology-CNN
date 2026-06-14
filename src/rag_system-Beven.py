import os
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import chromadb
from openai import OpenAI  # Swapped Google GenAI for OpenAI
from dotenv import load_dotenv

# --- 1. INITIALIZATION & AUTHENTICATION ---
# Load Environment Variables BEFORE initializing the OpenAI client
load_dotenv()

# Initialize OpenAI Client (automatically reads OPENAI_API_KEY from environment)
client = OpenAI()

# Connect to your existing ChromaDB database
chroma_client = chromadb.PersistentClient(path="breakhis_vector_db")
collection = chroma_client.get_collection(name="breakhis_dataset")


# --- 2. HARDWARE ACCELERATION & FEATURE EXTRACTOR ---
# Automatically utilize NVIDIA GPU if configured, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
# Remove the final fully connected classification head to extract pure visual features
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model = model.to(device)
model.eval()

# Standard PyTorch normalization required for ImageNet pre-trained models
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
def run_image_rag(query_image_path, magnification, top_k=3):
    print(f"\n[1/3] Processing query image and searching Vector DB...")
    query_vector = get_image_embedding(query_image_path)
    
    # Retrieve closest visual reference matches filtering by target zoom magnification
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"magnification": str(magnification)}
    )
    
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    if not metadatas:
        print(f"❌ No matching reference records found in the database for {magnification}X magnification.")
        return None

    # [2/3] Classification Category Step (Majority Vote Logic)
    classes_retrieved = [m['class'] for m in metadatas]
    # Pick the most common class among matches (Benign vs Malignant)
    final_classification = max(set(classes_retrieved), key=classes_retrieved.count)
    
    print(f"📊 Algorithmic Classification: {final_classification.upper()}")
    
    # Format reference data for the prompt context window
    references_text = ""
    for i, meta in enumerate(metadatas):
        references_text += f"- Reference Match {i+1}: Source File: {meta['filename']} | Confirmed Category: {meta['class']} | Pathological Subtype: {meta['subtype']} (Distance Score: {distances[i]:.4f})\n"

    # [3/3] LLM Report Generation Step
    print(f"[3/3] Generating Clinical Reference Report via OpenAI (gpt-4o)...")
    
    prompt = f"""
    You are an expert AI Pathology Assistant specializing in histopathology.
    A new breast biopsy tissue image taken at {magnification}X magnification has been processed by our visual vector retrieval system.
    
    The algorithm classified this sample as: **{final_classification}**
    
    The system retrieved the top {top_k} most morphologically similar historical cases from the BreakHis database to support this decision:
    {references_text}
    
    Please generate a structured 'Histopathology Reference & Verification Report'. Include:
    1. **Diagnostic Summary**: State the calculated classification and context.
    2. **Morphological Evidence**: Detail typical visual tissue markers associated with the retrieved subtypes (e.g., if Ductal Carcinoma or Adenosis are present in references, mention what microscopic structures to double check).
    3. **Database References**: Neatly list the reference filenames provided above so the pathologist can look them up.
    4. **Differential Diagnostic Recommendation**: A brief guidance note on verifying this classification.
    
    Write professionally, using appropriate medical terminology.
    """
    
    # OpenAI Chat Completions API implementation
    response = client.chat.completions.create(
        model='gpt-4o',  # Using flagship gpt-4o for complex medical text reasoning
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2  # Lower temperature keeps the clinical responses structured and accurate
    )
    
    return {
        "classification_category": final_classification,
        "references": metadatas,
        "report": response.choices[0].message.content
    }


# === 🚀 EXECUTE THE RAG SYSTEM ===
unknown_sample = "C:/Users/beven/MTU/Projects/test.jpg" 
sample_zoom = "200" # Must match image magnification (40, 100, 200, or 400)

if __name__ == "__main__":
    if not os.path.exists(unknown_sample):
        print(f"⚠️ Please place a valid test image at: {unknown_sample}")
    else:
        output = run_image_rag(unknown_sample, magnification=sample_zoom)
        
        if output:
            print("\n==================================================================")
            print("🔬 HISTOPATHOLOGY RAG SYSTEM OUTPUT")
            print("==================================================================")
            print(f"FINAL CLASSIFICATION: {output['classification_category']}")
            print("\nREPORT DETAILS:")
            print(output['report'])

import json
from datetime import datetime

# === 🚀 EXECUTE THE RAG SYSTEM ===
unknown_sample = "C:/Users/beven/MTU/Projects/test.jpg" 
sample_zoom = "200" 
output_dir = Path("C:/Users/beven/MTU/Projects/output_reports")

if __name__ == "__main__":
    print("🏁 Execution started...")
    
    if not os.path.exists(unknown_sample):
        print(f"⚠️ Please place a valid test image at: {unknown_sample}")
    else:
        print(f"📷 Found test image. Running RAG pipeline for zoom {sample_zoom}X...")
        output = run_image_rag(unknown_sample, magnification=sample_zoom)
        
        print("🔍 Checking pipeline output...")
        if output is None:
            print("❌ Pipeline returned None. Check if ChromaDB has data matching this magnification!")
        else:
            print("📝 Report generated successfully. Starting the save process...")
            
            # Force creating the directory right here
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Verified directory exists at: {output_dir}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"pathology_report_{timestamp}"
            
            # --- SAVE TEXT ---
            txt_filepath = output_dir / f"{base_filename}.txt"
            print(f"💾 Attempting to write text file to: {txt_filepath}")
            with open(txt_filepath, "w", encoding="utf-8") as txt_file:
                txt_file.write(output['report'])
            print("✅ Text file write complete.")
            
            # --- SAVE JSON ---
            json_filepath = output_dir / f"{base_filename}.json"
            print(f"💾 Attempting to write JSON file to: {json_filepath}")
            with open(json_filepath, "w", encoding="utf-8") as json_file:
                json.dump(output, json_file, indent=4, ensure_ascii=False)
            print("✅ JSON file write complete.")

            print("\n🎉 ALL TASKS COMPLETE. Check your output folder!")