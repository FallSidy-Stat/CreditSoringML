import os, pathlib
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys
from dotenv import load_dotenv
load_dotenv()

def ingest_kaggle_to_spark(dataset: str = "laotse/credit-risk-dataset", raw_dir="data/raw", parquet_path="data/processed/credit_data.parquet"):
    try:
        logger.info(f"Starting ingestion for {dataset}")
        # 1. Kaggle download
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        pathlib.Path(raw_dir).mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset, path=raw_dir, unzip=True)
        logger.info(f"Downloaded to {raw_dir}: {os.listdir(raw_dir)}")

        # 2. Load with Spark
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("CreditScoringIngestion") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .getOrCreate()

        # Find csv
        csv_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV found after kaggle download")
        csv_path = csv_files[0]
        logger.info(f"Reading CSV: {csv_path}")

        df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)
        logger.info(f"Spark DataFrame count: {df.count()}, columns: {df.columns}")

        # Store as parquet for lakehouse
        pathlib.Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(parquet_path)
        logger.info(f"Saved parquet to {parquet_path}")

        # Also sample to processed csv for notebooks
        df.limit(10000).toPandas().to_csv("data/processed/sample.csv", index=False)

        spark.stop()
        return parquet_path

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    ingest_kaggle_to_spark()
