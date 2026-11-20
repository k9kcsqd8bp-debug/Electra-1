import pandas as pd
from .feature_engineering import make_feature_row

def forecast_hours(model_bundle, hourly_df, hours=24):
    model = model_bundle["model"]
    history = hourly_df.sort_values("timestamp")["consumption_kwh"].astype(float).tolist()
    last_timestamp = pd.Timestamp(hourly_df["timestamp"].max())
    results = []
    for step in range(1, int(hours) + 1):
        next_timestamp = last_timestamp + pd.Timedelta(hours=step)
        features = make_feature_row(history, next_timestamp)
        value = float(model.predict(features)[0])
        value = max(0.0, value)
        history.append(value)
        results.append({"timestamp": next_timestamp, "predicted_kwh": value})
    return pd.DataFrame(results)
