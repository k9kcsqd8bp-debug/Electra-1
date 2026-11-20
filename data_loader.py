from pathlib import Path
import pandas as pd

def _normalise_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_electricity_data(file_or_path):
    """
    Accepts:
      1) ELECTRA format: timestamp, consumption_kwh
      2) UCI household power CSV: Date, Time, Global_active_power
      3) Similar files with a datetime column and a power/energy column.
    """
    if hasattr(file_or_path, "read"):
        try:
            df = pd.read_csv(file_or_path, sep=None, engine="python")
        except Exception:
            file_or_path.seek(0)
            df = pd.read_csv(file_or_path)
    else:
        path = Path(file_or_path)
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)

    df = _normalise_column_names(df)

    # UCI Individual Household Electric Power Consumption dataset
    if {"date", "time", "global_active_power"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            dayfirst=True, errors="coerce"
        )
        power = pd.to_numeric(df["global_active_power"], errors="coerce")
        df = pd.DataFrame({"timestamp": df["timestamp"], "power_kw": power})
        df = df.dropna()
        # Mean kW over an hour is numerically equivalent to kWh for a one-hour interval.
        hourly = (df.set_index("timestamp")["power_kw"]
                    .resample("h").mean()
                    .dropna()
                    .rename("consumption_kwh")
                    .reset_index())
        return _validate(hourly)

    # Find timestamp-like column
    timestamp_candidates = ["timestamp", "datetime", "date_time", "date"]
    timestamp_col = next((c for c in timestamp_candidates if c in df.columns), None)
    if timestamp_col is None:
        raise ValueError("Could not find a timestamp column. Use 'timestamp' or the UCI Date and Time columns.")

    # Find target/power column
    target_candidates = ["consumption_kwh", "energy_kwh", "kwh", "global_active_power", "power_kw", "consumption", "usage"]
    target_col = next((c for c in target_candidates if c in df.columns and c != timestamp_col), None)
    if target_col is None:
        raise ValueError("Could not find an electricity usage column (for example, consumption_kwh or Global_active_power).")

    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    values = pd.to_numeric(df[target_col], errors="coerce")
    clean = pd.DataFrame({"timestamp": timestamps, "value": values}).dropna()

    if target_col in ["global_active_power", "power_kw"]:
        # For hourly data, mean kW over the hour gives hourly kWh.
        clean = (clean.set_index("timestamp")["value"].resample("h").mean()
                    .dropna().rename("consumption_kwh").reset_index())
    else:
        clean = clean.rename(columns={"value": "consumption_kwh"})
    return _validate(clean)

def _validate(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "consumption_kwh"])
    df = df[df["consumption_kwh"] >= 0]
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    if len(df) < 200:
        raise ValueError("The dataset has too few valid rows. At least 200 hourly observations are recommended.")
    return df.reset_index(drop=True)
