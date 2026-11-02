import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sesuaikan nama folder/file sesuai yang ada di laptop Anda saat ini:
# Contoh jika file berada di folder data dengan nama cleaned_data.parquet:
DATA_INPUT_PARQUET = os.path.join(BASE_DIR, "data", "cleaned_data.parquet")

# ATAU jika berada di folder notebooks:
# DATA_INPUT_PARQUET = os.path.join(BASE_DIR, "notebooks", "cleaned_data.parquet")

MODEL_INDEXER_DIR = os.path.join(BASE_DIR, "models", "string_indexer_pipeline")
MODEL_ALS_DIR = os.path.join(BASE_DIR, "models", "als_recommender_model")

ALS_PARAMS = {
    "maxIter": 10,
    "regParam": 0.1,
    "rank": 20,
    "userCol": "user_idx",
    "itemCol": "item_idx",
    "ratingCol": "purchase_count",
    "implicitPrefs": True,
    "alpha": 40.0,
    "coldStartStrategy": "drop",
    "seed": 42
}
