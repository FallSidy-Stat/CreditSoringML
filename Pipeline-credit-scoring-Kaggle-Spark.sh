# Option A (simple)
PYTHONPATH=. python src/data/ingestion.py

# Option B (recommandée, pro)
python -m src.data.ingestion
python -m src.data.validation
python -m src.data.transformation_pyspark
python -m src.pipelines.training_pipeline