from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
MODEL_PATH = MODEL_DIR / "electricity_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

TARGET_COLUMN = "consumption_kwh"
TIMESTAMP_COLUMN = "timestamp"
RANDOM_STATE = 42
N_ESTIMATORS = 150
