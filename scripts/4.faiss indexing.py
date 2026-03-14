# %%
import os
import numpy as np
import pandas as pd
import faiss
from glob import glob
from tqdm import tqdm

EMBEDDING_DIR = "embeddings"
INDEX_DIR = "backend/index_store"
os.makedirs(INDEX_DIR, exist_ok=True)

print("Collecting chunk files...")

image_files = sorted(glob(f"{EMBEDDING_DIR}/image_emb_chunk_*.npy"))
text_files = sorted(glob(f"{EMBEDDING_DIR}/text_emb_chunk_*.npy"))
id_files = sorted(glob(f"{EMBEDDING_DIR}/ids_chunk_*.csv"))

assert len(image_files) == len(text_files) == len(id_files)

all_ids = []
all_embeddings = []

print("Merging chunks safely...")

for img_f, txt_f, id_f in tqdm(zip(image_files, text_files, id_files), total=len(image_files)):
    img_emb = np.load(img_f)
    txt_emb = np.load(txt_f)
    ids = pd.read_csv(id_f).values.flatten()

    # Combine image + text embeddings (average)
    combined = (img_emb + txt_emb) / 2.0

    # Normalize again
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / norms

    # Convert to float16 to reduce memory
    combined = combined.astype(np.float16)

    all_embeddings.append(combined)
    all_ids.extend(ids)

# Stack everything
print("Stacking embeddings...")
all_embeddings = np.vstack(all_embeddings)

print("Final shape:", all_embeddings.shape)

# Save merged embeddings
np.save(f"{INDEX_DIR}/final_embeddings.npy", all_embeddings)

pd.Series(all_ids).to_csv(
    f"{INDEX_DIR}/id_mapping.csv",
    index=False
)

print("Building FAISS index...")

dimension = all_embeddings.shape[1]

# Use Inner Product (cosine similarity because normalized)
index = faiss.IndexFlatIP(dimension)

# FAISS needs float32
index.add(all_embeddings.astype(np.float32))

faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

print("FAISS index built and saved.")


