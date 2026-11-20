import pandas as pd

def prepare_hourly_data(df):
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["consumption_kwh"] = pd.to_numeric(data["consumption_kwh"], errors="coerce")
    data = data.dropna(subset=["timestamp", "consumption_kwh"])
    data = data[data["consumption_kwh"] >= 0]
    data = data.sort_values("timestamp").drop_duplicates("timestamp")
    # Standardise to a continuous hourly series; gaps are time-interpolated and flagged in the report.
    data = data.set_index("timestamp").resample("h").mean()
    missing_before = int(data["consumption_kwh"].isna().sum())
    data["consumption_kwh"] = data["consumption_kwh"].interpolate(method="time", limit=6)
    data = data.dropna().reset_index()
    return data, missing_before
