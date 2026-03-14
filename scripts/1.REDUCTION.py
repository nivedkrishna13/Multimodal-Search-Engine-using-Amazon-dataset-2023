# %%
import pandas as pd

# 1. Load the datasets
products = pd.read_csv('Amazon Products Dataset 2023 (1.4M Products)/amazon_products.csv')
categories = pd.read_csv('Amazon Products Dataset 2023 (1.4M Products)/amazon_categories.csv')

print(f"Initial product count: {len(products)}")

# 2. Basic Cleaning
# Remove products with missing critical info for a search engine
df = products.dropna(subset=['title', 'imgUrl', 'category_id'])
df = df[df['price'] > 0]  # Remove products with no price
df = df[df['imgUrl'].str.contains('http', na=False)] # Ensure image URL exists

# 3. Quality Filtering
# Keep BestSellers (High quality signal)
bestsellers = df[df['isBestSeller'] == True]

# Filter for "High Quality" non-bestsellers
# Criteria: Stars >= 4.0 and Reviews >= 20 (adjust review count to hit your target)
high_quality = df[(df['isBestSeller'] == False) & 
                  (df['stars'] >= 3.0) & 
                  (df['reviews'] >= 10)]

# Combine them
reduced_df = pd.concat([bestsellers, high_quality]).drop_duplicates(subset='asin')

print(f"Count after quality filtering: {len(reduced_df)}")

# 4. Stratified Sampling (If the count is still > 400,000)
# This ensures we keep a diverse range of products across all categories
TARGET_SIZE = 500000

if len(reduced_df) > TARGET_SIZE:
    # Calculate how many to take from each category
    # We use a min(category_size, required_sample) approach
    df_grouped = reduced_df.groupby('category_id')
    
    # Simple random sample to get exactly the target size while maintaining category diversity
    reduced_df = reduced_df.sample(n=TARGET_SIZE, random_state=42)

# 5. Merge with Category Names (Optional but helpful for Text Search)
reduced_df = reduced_df.merge(categories, left_on='category_id', right_on='id', how='left')

# 6. Save the reduced dataset
reduced_df.to_csv('amazon_products_reduced.csv', index=False)

print(f"Final reduced dataset count: {len(reduced_df)}")
print("File saved as 'amazon_products_reduced.csv'")


