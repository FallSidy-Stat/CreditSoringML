from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log1p, isnan, count
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from src.utils.logger import logger

def get_spark():   
    return SparkSession.builder \
        .appName("CreditTransformationPurePySpark") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def transform_full_pyspark(parquet_path="data/processed/credit_data.parquet", output_path="data/processed/transformed_parquet"):    
    spark = get_spark()
    df = spark.read.parquet(parquet_path)
    logger.info(f"Input: {df.count()} rows")
    # 1. Cleaning 100% Spark, pas de toPandas
    df = df.filter((col("person_age") >= 18) & (col("person_age") <= 100))
    df = df.filter(col("person_income") > 0)

    # Clip + log transform
    df = df.withColumn("person_income_clipped", when(col("person_income") > 1_000_000, 1_000_000).otherwise(col("person_income")))
    df = df.withColumn("log_income", log1p(col("person_income_clipped")))
    df = df.withColumn("log_loan_amnt", log1p(col("loan_amnt")))

    # 2. Drop duplicates - Spark
    df = df.dropDuplicates()

    # 3. Imputation avec pyspark.ml (pas sklearn)
    numeric_cols = [c for c in df.columns if c in ["person_income_clipped", "person_emp_length", "loan_int_rate", "cb_person_cred_hist_length"]]
    if numeric_cols:
        imputer = Imputer(inputCols=numeric_cols, outputCols=[c+"_imputed" for c in numeric_cols], strategy="median")
        df = imputer.fit(df).transform(df)

    df.write.mode("overwrite").parquet(output_path)
    logger.info(f"Transformed saved to {output_path} count={df.count()}")
    spark.stop()
    return output_path

if __name__ == "__main__":
    transform_full_pyspark()