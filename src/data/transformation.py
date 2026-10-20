from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, log1p
import pandas as pd

def transform_spark_to_pandas(parquet_path="data/processed/credit_data.parquet"):
    spark = SparkSession.builder.appName("Transformation").getOrCreate()
    df_spark = spark.read.parquet(parquet_path)

    # Basic cleaning in Spark
    df_spark = df_spark.filter(col("person_age") >= 18)
    df_spark = df_spark.filter(col("person_income") > 0)
    # Clip outliers
    df_spark = df_spark.withColumn("person_income", when(col("person_income") > 1_000_000, 1_000_000).otherwise(col("person_income")))

    # Convert to pandas for sklearn
    df = df_spark.toPandas()
    spark.stop()

    # Pandas cleaning
    df = df.drop_duplicates()
    # Map target if string
    if df['loan_status'].dtype == 'object':
        df['loan_status'] = df['loan_status'].map({'Fully Paid':0, 'Charged Off':1, 0:0, 1:1}).fillna(df['loan_status'])
    
    df.to_csv("data/processed/cleaned.csv", index=False)
    return df
