import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from .feature_engineering import create_supervised_data, FEATURE_COLUMNS
from config import MODEL_PATH, RANDOM_STATE, N_ESTIMATORS

def train_model(hourly_df):
    X, y, timestamps = create_supervised_data(hourly_df)
    if len(X) < 100:
        raise ValueError("Not enough rows after feature engineering. Provide a longer dataset (ideally 6+ months).")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "Chronological 80% train / 20% test",
        "model": "RandomForestRegressor"
    }
    test_results = {
        "timestamp": timestamps.iloc[split:].reset_index(drop=True),
        "actual_kwh": y_test.reset_index(drop=True),
        "predicted_kwh": predictions
    }
    bundle = {"model": model, "feature_columns": FEATURE_COLUMNS}
    joblib.dump(bundle, MODEL_PATH)
    return bundle, metrics, test_results
