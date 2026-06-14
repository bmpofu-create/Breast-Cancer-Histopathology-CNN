import chromadb
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Connect to your newly populated DB
chroma_client = chromadb.PersistentClient(path="breakhis_vector_db")
collection = chroma_client.get_collection(name="breakhis_dataset")

# 2. Setup the exact same feature extractor
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def query_similar_cases(image_path, target_magnification, top_k=2):
    # Process test image
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        query_vector = model(tensor).flatten().tolist()
        
    # Query database and filter by magnification level
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"magnification": str(target_magnification)}
    )
    return results
    

# === 🚀 RUN A TEST ===
# Replace this with the path to a test image you held out from the database
test_image = "C:/Users/beven/MTU/Projects/test3.png" 
test_mag = "200" # Put the zoom level of your test image here (40, 100, 200, 400)

try:
    matches = query_similar_cases(test_image, target_magnification=test_mag, top_k=3)
    
    print(f"\n🔍 --- Top 3 Reference Matches for {test_mag}X Query ---")
    for idx, meta in enumerate(matches['metadatas'][0]):
        distance = matches['distances'][0][idx]
        print(f"Match {idx+1}:")
        print(f"  • File: {meta['filename']}")
        print(f"  • Diagnosis: {meta['class']} ({meta['subtype']})")
        print(f"  • Similarity Distance: {distance:.4f}\n") # Lower distance = closer match
        
except Exception as e:
    print(f"Make sure you point 'test_image' to a real file path! Error: {e}")