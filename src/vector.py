import os
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import chromadb

# 1. Test Base Directory
dataset_root = Path("C:/Users/beven/MTU/Projects/BreaKHis_v1")

print("=== 🔍 DIRECTORY DIAGNOSTIC ===")
if not dataset_root.exists():
    print(f"❌ Error: The path '{dataset_root}' is completely invalid or typed wrong.")
    print("Available contents of C:/Users/beven/MTU/Projects/:", os.listdir("C:/Users/beven/MTU/Projects/"))
    exit()
else:
    print(f"✅ Found root folder: {dataset_root}")
    # Show what is directly inside this folder
    print("Contents inside BreaKHis_v1:", os.listdir(dataset_root))

# 2. Look for images using standard os.walk (bypasses pathlib rglob quirks on Windows)
print("\n🔍 Scanning with standard os.walk...")
all_images = []
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = Path(root) / file
            all_images.append(full_path)

print(f"📄 Found a total of {len(all_images)} image files across all subfolders.")

if len(all_images) == 0:
    print("❌ No images found. Your BreakHis archive might still be zipped, or it extracted into a deeper nested folder structure.")
    exit()

# 3. Setup Vector DB & Model
print("\n📦 Setting up ChromaDB and ResNet Feature Extractor...")
chroma_client = chromadb.PersistentClient(path="breakhis_vector_db")
collection = chroma_client.get_or_create_collection(
    name="breakhis_dataset", 
    metadata={"hnsw:space": "cosine"}
)

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

def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).flatten().tolist()
    return embedding

# 4. Fallback Metadata Parser (Reads filenames directly if folder names match weird patterns)
id_counter = 0
print("\n🚀 Starting DB Ingestion...")

for img_path in all_images:
    path_str = str(img_path).lower()
    file_name = img_path.name
    
    # Extract Class
    tumor_class = "Malignant" if "malignant" in path_str or "_m_" in file_name.lower() else "Benign"
    
    # Extract Magnification Factor
    magnification = "Unknown"
    for mag in ["40", "100", "200", "400"]:
        if f"_{mag}_" in file_name or f"-{mag}-" in file_name or f"/{mag}x/" in path_str or f"\\{mag}x\\" in path_str:
            magnification = mag
            break

    # Extract Subtype
    tumor_subtype = "Unknown"
    subtypes = {"adenosis": "Adenosis", "fibroadenoma": "Fibroadenoma", "phyllodes": "Phyllodes_Tumor", 
                "tubular": "Tubular_Adenoma", "ductal": "Ductal_Carcinoma", "lobular": "Lobular_Carcinoma", 
                "mucinous": "Mucinous_Carcinoma", "papillary": "Papillary_Carcinoma"}
    for key, value in subtypes.items():
        if key in path_str or f"_{key[:2].upper()}_" in file_name:
            tumor_subtype = value
            break

    try:
        vector = extract_embedding(img_path)
        
        collection.add(
            embeddings=[vector],
            metadatas=[{
                "filename": file_name,
                "class": tumor_class,
                "subtype": tumor_subtype,
                "magnification": magnification
            }],
            ids=[f"id_{id_counter}"]
        )
        id_counter += 1
        
        if id_counter % 50 == 0:
            print(f" Indexed {id_counter} images...")
            
    except Exception as e:
        print(f"Error reading image {file_name}: {e}")

print(f"\n🎉 Success! Added {id_counter} images to your Vector Database.")