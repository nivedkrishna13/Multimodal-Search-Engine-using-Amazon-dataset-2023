# %%
import os
import torch
import requests
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import pyarrow.parquet as pq
# Paths
DATA_PATH = "data/processed/amazon_clean.parquet"
OUTPUT_DIR = "embeddings"
BATCH_SIZE = 64
CHUNK_SIZE = 500  # rows read at once (memory safe)

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None


def process_chunk(df_chunk, chunk_id):
    image_embeddings = []
    text_embeddings = []
    valid_ids = []

    for start in tqdm(range(0, len(df_chunk), BATCH_SIZE)):
        batch = df_chunk.iloc[start:start + BATCH_SIZE]

        images = []
        texts = []
        ids = []

        for _, row in batch.iterrows():
            img = download_image(row["imgUrl"])
            if img is None:
                continue
            images.append(img)
            texts.append(row["title"])
            ids.append(row["asin"])

            if len(images) == 0:
                continue

            try:
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,    # Crucial fix
                    max_length=77       # CLIP limit
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # These are the projected latent vectors
                    img_emb = outputs.image_embeds
                    txt_emb = outputs.text_embeds

                    # L2 Normalization for Cosine Similarity
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            except Exception as e:
                print(f"Error processing batch in chunk {chunk_id}: {e}")
                continue
        image_embeddings.append(img_emb.cpu().numpy())
        text_embeddings.append(txt_emb.cpu().numpy())
        valid_ids.extend(ids)

        torch.cuda.empty_cache()

    if len(image_embeddings) == 0:
        return

    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    np.save(f"{OUTPUT_DIR}/image_emb_chunk_{chunk_id}.npy", image_embeddings)
    np.save(f"{OUTPUT_DIR}/text_emb_chunk_{chunk_id}.npy", text_embeddings)

    pd.Series(valid_ids).to_csv(
        f"{OUTPUT_DIR}/ids_chunk_{chunk_id}.csv",
        index=False
    )

    print(f"Saved chunk {chunk_id}")



def main():
    print("Reading dataset in chunks...")
    
    # Create a ParquetFile object
    parquet_file = pq.ParquetFile(DATA_PATH)
    
    # Iterate through row groups (Parquet's internal "chunks")
    for i in range(parquet_file.num_row_groups):
        print(f"\nProcessing chunk (row group) {i}")
        
        # Read one row group into a pandas DataFrame
        df_chunk = parquet_file.read_row_group(i).to_pandas()
        
        process_chunk(df_chunk, i)
if __name__ == "__main__":
    main()


