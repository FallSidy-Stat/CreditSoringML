import argparse
from src.data.ingestion import ingest_kaggle_to_spark
from src.data.validation import validate
from src.data.transformation import transform_spark_to_pandas
from src.models.train import train
from src.models.evaluate import evaluate
from src.utils.logger import logger

def run_pipeline(smoke=False):
    logger.info("=== Credit Scoring Training Pipeline START ===")
    if not smoke:
        parquet = ingest_kaggle_to_spark()
        validate(parquet)
        transform_spark_to_pandas(parquet)
    else:
        logger.info("Smoke mode - skipping ingestion")
    # If cleaned.csv doesn't exist in smoke mode, create dummy
    import os, pandas as pd, numpy as np
    if not os.path.exists("data/processed/cleaned.csv"):
        logger.info("Creating dummy cleaned.csv for smoke test")
        df = pd.DataFrame({
            'person_age': np.random.randint(20,60,500),
            'person_income': np.random.randint(20000,100000,500),
            'loan_amnt': np.random.randint(1000,30000,500),
            'loan_int_rate': np.random.uniform(5,20,500),
            'loan_percent_income': np.random.uniform(0.05,0.5,500),
            'person_home_ownership': np.random.choice(['RENT','OWN','MORTGAGE'],500),
            'loan_intent': np.random.choice(['PERSONAL','EDUCATION','MEDICAL','VENTURE'],500),
            'loan_status': np.random.choice([0,1],500,p=[0.8,0.2])
        })
        df.to_csv("data/processed/cleaned.csv", index=False)

    best_model, auc, name = train()
    metrics = evaluate()
    logger.info(f"Pipeline END - Best: {name} AUC: {auc}")
    logger.info(f"Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_pipeline(smoke=args.smoke)
