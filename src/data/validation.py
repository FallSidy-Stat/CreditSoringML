from pyspark.sql import SparkSession
from src.utils.logger import logger

REQUIRED_COLS = ["person_age", "person_income", "loan_amnt", "loan_status"]
BUSINESS_RULES = {
    "person_age": (18, 100),
    "person_income": (0, 10_000_000),
    "loan_amnt": (0, 2_000_000)
}

def validate(parquet_path="data/processed/credit_data.parquet"):
    spark = SparkSession.builder.appName("Validation").getOrCreate()
    df = spark.read.parquet(parquet_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, (min_v, max_v) in BUSINESS_RULES.items():
        if col in df.columns:
            bad = df.filter((df[col] < min_v) | (df[col] > max_v)).count()
            logger.info(f"Validation {col}: {bad} rows out of bounds [{min_v},{max_v}]")
            if bad > df.count()*0.05:
                logger.warning(f"Too many invalid {col}: {bad}")

    nulls = {c: df.filter(df[c].isNull()).count() for c in df.columns}
    logger.info(f"Nulls: {nulls}")
    spark.stop()
    return True

if __name__ == "__main__":
    validate()
