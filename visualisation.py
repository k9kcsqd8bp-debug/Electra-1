import matplotlib.pyplot as plt

def consumption_plot(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["consumption_kwh"], linewidth=0.8)
    ax.set_title("Hourly electricity consumption")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (kWh)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def forecast_plot(history_df, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    recent = history_df.tail(168)
    ax.plot(recent["timestamp"], recent["consumption_kwh"], label="Historical", linewidth=1.2)
    ax.plot(forecast_df["timestamp"], forecast_df["predicted_kwh"], label="Forecast", linewidth=1.5)
    ax.set_title("Recent usage and predicted consumption")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
