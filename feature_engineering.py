import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_mean_24", "rolling_mean_168"
]

def make_feature_row(history, timestamp):
    """Create one feature row from observed/predicted hourly history."""
    s = pd.Series(np.asarray(history, dtype=float))
    if len(s) < 168:
        raise ValueError("At least 168 hours of history are required for forecasting.")
    ts = pd.Timestamp(timestamp)
    row = {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "day_of_month": ts.day,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "lag_1": float(s.iloc[-1]),
        "lag_2": float(s.iloc[-2]),
        "lag_3": float(s.iloc[-3]),
        "lag_24": float(s.iloc[-24]),
        "lag_48": float(s.iloc[-48]),
        "lag_168": float(s.iloc[-168]),
        "rolling_mean_24": float(s.iloc[-24:].mean()),
        "rolling_mean_168": float(s.iloc[-168:].mean()),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

def create_supervised_data(df):
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(data["timestamp"])
    y = data["consumption_kwh"].astype(float)
    X = pd.DataFrame({
        "hour": ts.dt.hour,
        "day_of_week": ts.dt.dayofweek,
        "day_of_month": ts.dt.day,
        "month": ts.dt.month,
        "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
        "lag_1": y.shift(1),
        "lag_2": y.shift(2),
        "lag_3": y.shift(3),
        "lag_24": y.shift(24),
        "lag_48": y.shift(48),
        "lag_168": y.shift(168),
        "rolling_mean_24": y.shift(1).rolling(24).mean(),
        "rolling_mean_168": y.shift(1).rolling(168).mean(),
    })
    valid = X.notna().all(axis=1)
    return X.loc[valid, FEATURE_COLUMNS], y.loc[valid], ts.loc[valid]
