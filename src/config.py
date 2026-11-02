import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_INPUT_PARQUET = os.path.join(BASE_DIR, "data", "cleaned_hm_data.parquet")

# Model Paths
MODEL_INDEXER_DIR = os.path.join(BASE_DIR, "models", "string_indexer_pipeline")
MODEL_ALS_DIR = os.path.join(BASE_DIR, "models", "als_recommender_model")

# ALS Hyperparameters
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
