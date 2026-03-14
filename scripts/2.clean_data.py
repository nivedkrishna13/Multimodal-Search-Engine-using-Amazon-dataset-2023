# %%
import pandas as pd
import os
import numpy as np
RAW_PRODUCTS_PATH = "Amazon Products Dataset 2023 (1.4M Products)/amazon_products.csv"
RAW_CATEGORIES_PATH = "Amazon Products Dataset 2023 (1.4M Products)/amazon_categories.csv"
OUTPUT_PATH = "data/processed/amazon_clean.parquet"


def load_data():
    print("Loading datasets...")
    products = pd.read_csv(RAW_PRODUCTS_PATH)
    categories = pd.read_csv(RAW_CATEGORIES_PATH)
    print("Products columns:", products.columns.tolist())
    print("Categories columns:", categories.columns.tolist())
    return products, categories


def basic_cleaning(products: pd.DataFrame) -> pd.DataFrame:
    print("Initial shape:", products.shape)

    # Drop rows with missing critical fields
    products = products.dropna(subset=["title", "imgUrl", "price"])

    # Remove invalid price
    products = products[products["price"] > 0]

    # Optional: remove products with no rating
    products = products[products["stars"] > 0]

    # Remove duplicates
    products = products.drop_duplicates(subset=["asin"])

    print("After cleaning:", products.shape)

    return products


def merge_categories(products: pd.DataFrame, categories: pd.DataFrame):
    print("Merging categories...")
    categories = categories.rename(columns={"id": "category_id"})
    merged = products.merge(categories, on="category_id", how="left")
    return merged


def feature_engineering(df: pd.DataFrame):
    print("Feature engineering...")

    # Discount percentage
    df["discount_percent"] = (
        (df["listPrice"] - df["price"]) / df["listPrice"]
    ).fillna(0)

    # Log price (helps in ranking later)
    df["log_price"] = df["price"].apply(lambda x: np.log1p(x))

    return df


def save_data(df: pd.DataFrame):
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")


def main():
    products, categories = load_data()
    products = basic_cleaning(products)
    merged = merge_categories(products, categories)
    final_df = feature_engineering(merged)
    print(final_df.head())
    save_data(final_df)


if __name__ == "__main__":
    main()

# %%
import pandas as pd

df = pd.read_parquet("data/processed/amazon_clean.parquet")
print(df.head())
print(df.describe())


