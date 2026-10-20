from src.models.predict import CreditScoringPredictor
import pandas as pd

def batch_inference(csv_path="data/external/new_applicants.csv"):
    predictor = CreditScoringPredictor()
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        res = predictor.predict(row.to_dict())
        results.append(res)
    out = pd.concat([df, pd.DataFrame(results)], axis=1)
    out.to_csv("data/processed/batch_predictions.csv", index=False)
    return out

if __name__ == "__main__":
    batch_inference()
