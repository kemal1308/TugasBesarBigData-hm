from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator


def train_als_model(train_df: DataFrame, params: dict, save_path: str = None) -> ALSModel:
    """
    Melatih model rekomendasi ALS berbasis konfigurasi hyperparameter.
    """
    als = ALS(**params)
    als_model = als.fit(train_df)

    if save_path:
        als_model.write().overwrite().save(save_path)

    return als_model


def evaluate_model(als_model: ALSModel, test_df: DataFrame) -> float:
    """
    Mengevaluasi model pada data uji menggunakan RegressionEvaluator (RMSE).
    """
    predictions = als_model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="purchase_count",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse


def get_top_recommendations(als_model: ALSModel, num_items: int = 5) -> DataFrame:
    """
    Menghasilkan Top-N rekomendasi item untuk seluruh pengguna.
    """
    return als_model.recommendForAllUsers(num_items)
