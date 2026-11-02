from pyspark.sql import SparkSession
from src.config import DATA_INPUT_PARQUET, MODEL_INDEXER_DIR, MODEL_ALS_DIR, ALS_PARAMS
from src.preprocessing import create_implicit_ratings, fit_and_transform_indexers
from src.model import train_als_model, evaluate_model, get_top_recommendations


def init_spark() -> SparkSession:
    return SparkSession.builder \
        .appName("HM_Recommendation_Pipeline") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()


def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("[1/5] Memuat dataset bersih...")
    df_cleaned = spark.read.parquet(DATA_INPUT_PARQUET)

    print("[2/5] Membangun implicit rating & indexing...")
    interactions = create_implicit_ratings(df_cleaned)
    indexed_df, _ = fit_and_transform_indexers(interactions, model_save_path=MODEL_INDEXER_DIR)

    print("[3/5] Membagi data train & test...")
    train_data, test_data = indexed_df.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    test_data.cache()

    print("[4/5] Melatih model ALS...")
    als_model = train_als_model(train_data, ALS_PARAMS, save_path=MODEL_ALS_DIR)

    print("[5/5] Evaluasi & Inferensi...")
    rmse = evaluate_model(als_model, test_data)
    print(f">> Hasil Evaluasi RMSE: {rmse:.4f}")

    print(">> Sampel Top 5 Rekomendasi:")
    recs = get_top_recommendations(als_model, num_items=5)
    recs.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
