# %% [markdown]
# CRoss encoder re rank

# %%
import faiss
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import re
from sentence_transformers import CrossEncoder

print("Loading re-ranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Paths
INDEX_PATH = "backend/index_store/faiss.index"
ID_PATH = "backend/index_store/id_mapping.csv"
DATA_PATH = "data/processed/amazon_clean.parquet"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load ID mapping
id_mapping = pd.read_csv(ID_PATH).values.flatten()

# Load metadata
df = pd.read_parquet(DATA_PATH)


def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    # Check if 'outputs' is an object; if so, extract the tensor
    # BaseModelOutputWithPooling stores the vector in 'pooler_output'
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs.pooler_output

    emb = outputs / outputs.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    
    # Apply the same extraction logic here
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs.pooler_output

    emb = outputs / outputs.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)
#############################################################################
def parse_query(query):
    filters = {}

    # Detect "under $X"
    match = re.search(r'under \$?(\d+)', query.lower())
    if match:
        filters["max_price"] = float(match.group(1))

    # Detect "above X stars"
    match = re.search(r'above (\d(\.\d)?)', query.lower())
    if match:
        filters["min_stars"] = float(match.group(1))

    # Detect "cheaper"
    if "cheaper" in query.lower():
        filters["sort_by"] = "price_asc"

    if "best rated" in query.lower():
        filters["sort_by"] = "stars_desc"

    return filters
#####################################################################

## modified search function with re-ranking and secondary sorting
def search(embedding, query_text=None, top_k=200):

    distances, indices = index.search(embedding, top_k)

    filters = parse_query(query_text) if query_text else {}

    candidates = []

    for idx in indices[0]:
        product_id = id_mapping[idx]
        product = df[df["asin"] == product_id]

        if product.empty:
            continue

        product = product.iloc[0]

        # Apply filters
        if "max_price" in filters and product["price"] > filters["max_price"]:
            continue

        if "min_stars" in filters and product["stars"] < filters["min_stars"]:
            continue

        candidates.append({
            "asin": product_id,
            "title": product["title"],
            "price": product["price"],
            "stars": product["stars"],
            "category": product.get("category_name", "N/A")
        })

    # 🔥 RERANKING STEP
    if query_text and len(candidates) > 0:
        pairs = [(query_text, c["title"]) for c in candidates]
        scores = reranker.predict(pairs)

        for i in range(len(candidates)):
            candidates[i]["rerank_score"] = scores[i]

        candidates = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

    # Optional secondary sorting logic
    if "sort_by" in filters:
        if filters["sort_by"] == "price_asc":
            candidates = sorted(candidates, key=lambda x: x["price"])
        if filters["sort_by"] == "stars_desc":
            candidates = sorted(candidates, key=lambda x: x["stars"], reverse=True)

    return candidates[:10]

    
    

while(True):
    if __name__ == "__main__":
        mode = input("Search by (1) Text or (2) Image? ")

        if mode == "1":
            query = input("Enter text query: ")
            emb = encode_text(query)

        elif mode == "2":
            path = input("Enter image path: ")
            emb = encode_image(path)

        else:
            print("Invalid option")
            exit()

        results = results = search(emb, query_text=query)

        print("\nTop Results:\n")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['title']}")
            print(f"   Price: ${r['price']} | Stars: {r['stars']} | Category: {r['category']}")
            print("-" * 50)


