import json
from pathlib import Path

def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def metrics_explanation():
    return {
        "MAE": "Mean Absolute Error: average absolute difference between predicted and actual kWh.",
        "MSE": "Mean Squared Error: average squared prediction error; larger errors receive more weight.",
        "RMSE": "Root Mean Squared Error: error in kWh, in the same unit as the target.",
        "R²": "R-squared: proportion of variation explained by the model; it can be negative.",
        "Baseline MAE": "Error from a simple mean-prediction baseline, used for comparison."
    }
