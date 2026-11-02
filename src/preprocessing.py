import os

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer


def create_implicit_ratings(df: DataFrame) -> DataFrame:
    """
    Menghitung frekuensi transaksi per pasangan customer_id dan article_id
    sebagai proxy implicit feedback.
    """
    interaction_df = df.groupBy("customer_id", "article_id") \
        .agg(F.count("t_dat").alias("purchase_count")) \
        .filter(F.col("purchase_count") > 0)
    return interaction_df


def fit_and_transform_indexers(df: DataFrame, model_save_path: str = None) -> tuple[DataFrame, PipelineModel]:
    """
    Mengubah customer_id dan article_id ke tipe integer menggunakan StringIndexer.
    """
    customer_indexer = StringIndexer(
        inputCol="customer_id", 
        outputCol="user_idx", 
        handleInvalid="skip"
    )
    article_indexer = StringIndexer(
        inputCol="article_id", 
        outputCol="item_idx", 
        handleInvalid="skip"
    )

    indexer_pipeline = Pipeline(stages=[customer_indexer, article_indexer])
    indexer_model = indexer_pipeline.fit(df)
    transformed_df = indexer_model.transform(df)

    # Casting ke tipe integer dan float untuk kompatibilitas Spark ALS
    final_df = transformed_df \
        .withColumn("user_idx", F.col("user_idx").cast("integer")) \
        .withColumn("item_idx", F.col("item_idx").cast("integer")) \
        .withColumn("purchase_count", F.col("purchase_count").cast("float"))

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        indexer_model.write().overwrite().save(model_save_path)

    return final_df, indexer_model
