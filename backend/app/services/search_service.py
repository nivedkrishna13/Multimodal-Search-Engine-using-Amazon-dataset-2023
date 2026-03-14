import os

import faiss
import pandas as pd
# Use the full path from the root of your project
from backend.app.services.embedding_service import encode_text
from backend.app.services.rerank_service import rerank
from backend.app.services.query_parser import parse_query
from backend.app.services.embedding_service import encode_image


import os
from fastapi import UploadFile, File
INDEX_PATH = "backend/index_store/faiss.index"
ID_PATH = "backend/index_store/id_mapping.csv"
DATA_PATH = "backend/app/services/data/processed/amazon_clean.parquet"

index = faiss.read_index(INDEX_PATH)
id_mapping = pd.read_csv(ID_PATH).values.flatten()
df = pd.read_parquet(DATA_PATH)

product_lookup = df.set_index("asin").to_dict("index")

def search_text(query_text, top_k=50):

    embedding = encode_text(query_text)
    distances, indices = index.search(embedding, top_k)

    filters = parse_query(query_text) if query_text else {}
    results = []

    for idx in indices[0]:

        product_id = id_mapping[idx]
        product = product_lookup.get(product_id)

        if not product:
            continue

        if "max_price" in filters and product["price"] > filters["max_price"]:
            continue

        if "min_stars" in filters and product["stars"] < filters["min_stars"]:
            continue

        results.append({
            "title": product["title"],
            "price": product["price"],
            "stars": product["stars"],
            "image": product["imgUrl"],
            "category": product.get("category_name", "N/A")
        })

    if filters.get("sort_by") == "price_asc":
        results.sort(key=lambda x: x["price"])

    if filters.get("sort_by") == "stars_desc":
        results.sort(key=lambda x: x["stars"], reverse=True)

    return results[:10]

def search_image(query_image, top_k=50):

    embedding = encode_image(query_image)
    distances, indices = index.search(embedding, top_k)

    results = []

    for idx in indices[0]:

        product_id = id_mapping[idx]
        product = product_lookup.get(product_id)

        if not product:
            continue

        results.append({
            "title": product["title"],
            "price": product["price"],
            "stars": product["stars"],
            "image": product["imgUrl"],
            "category": product.get("category_name", "N/A")
        })

    # if filters.get("sort_by") == "price_asc":
    #     results.sort(key=lambda x: x["price"])

    # if filters.get("sort_by") == "stars_desc":
    #     results.sort(key=lambda x: x["stars"], reverse=True)

    return results[:10]