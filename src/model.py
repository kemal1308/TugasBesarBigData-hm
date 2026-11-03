import os
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame


def train_als_model(train_df: DataFrame, params: dict, save_path: str = None) -> ALSModel:
    """
    Melatih model Collaborative Filtering menggunakan algoritma ALS (PySpark MLlib).
    """
    als = ALS(
        maxIter=params.get("maxIter", 10),
        regParam=params.get("regParam", 0.1),
        rank=params.get("rank", 20),
        userCol=params.get("userCol", "user_idx"),
        itemCol=params.get("itemCol", "item_idx"),
        ratingCol=params.get("ratingCol", "purchase_count"),
        implicitPrefs=params.get("implicitPrefs", True),
        alpha=params.get("alpha", 40.0),
        coldStartStrategy=params.get("coldStartStrategy", "drop"),
        seed=params.get("seed", 42)
    )

    print(">> Proses fitting model ALS sedang berjalan...")
    als_model = als.fit(train_df)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        als_model.write().overwrite().save(save_path)
        print(f">> Model ALS berhasil disimpan ke: {save_path}")

    return als_model


def evaluate_model(als_model: ALSModel, test_df: DataFrame) -> float:
    """
    Mengevaluasi performa model ALS pada data uji menggunakan metrik RMSE.
    """
    predictions = als_model.transform(test_df)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="purchase_count",
        predictionCol="prediction"
    )

    # Filter nilai NaN jika coldStartStrategy tidak menghapusnya secara otomatis
    clean_predictions = predictions.filter(~predictions["prediction"].isNull())
    rmse = evaluator.evaluate(clean_predictions)
    return rmse


def get_top_recommendations(als_model: ALSModel, num_items: int = 5) -> DataFrame:
    """
    Menghasilkan rekomendasi Top-N produk untuk seluruh pengguna.
    """
    return als_model.recommendForAllUsers(num_items)
