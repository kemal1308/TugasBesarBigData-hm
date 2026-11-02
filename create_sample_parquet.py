import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

spark = SparkSession.builder \
    .appName("Generate_Valid_Parquet") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Pastikan path mengarah ke file CSV transactions_train Anda
csv_path = "data/transactions_train.csv"

# Ambil sampel kecil (misal 2% - 5%) agar eksekusi cepat dan hemat RAM
print("Membaca dan memproses sampel transaksi...")
df_trans = spark.read.option("header", "true").csv(csv_path).sample(fraction=0.05, seed=42)

# Bersihkan dan ambil kolom esensial
df_clean = df_trans \
    .withColumn("price", F.col("price").cast(DoubleType())) \
    .filter(F.col("customer_id").isNotNull() & F.col("article_id").isNotNull()) \
    .select("t_dat", "customer_id", "article_id", "price")

# Simpan dalam format Parquet asli
output_path = "data/cleaned_hm_data.parquet"
df_clean.write.mode("overwrite").parquet(output_path)

print(f"File Parquet valid berhasil disimpan di: {output_path}")
spark.stop()
