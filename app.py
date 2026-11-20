import streamlit as st
import pandas as pd
from config import SAMPLE_DATA_PATH, METRICS_PATH, PREDICTIONS_PATH
from src.data_loader import load_electricity_data
from src.data_preprocessing import prepare_hourly_data
from src.model_training import train_model
from src.prediction import forecast_hours
from src.bill_calculator import estimate_cost
from src.visualisation import consumption_plot, forecast_plot
from src.model_evaluation import save_metrics, metrics_explanation

st.set_page_config(page_title="ELECTRA | Electricity Analytics", page_icon="⚡", layout="wide")
st.title("⚡ ELECTRA")
st.subheader("AI-Based Electricity Consumption Prediction and Energy Analytics System")
st.caption("A student prototype using time-series data and machine learning. Forecasts are estimates, not guaranteed readings.")

with st.sidebar:
    st.header("Project controls")
    uploaded = st.file_uploader("Upload electricity CSV", type=["csv"])
    st.caption("Accepted formats: timestamp + consumption_kwh, or UCI Date/Time/Global_active_power.")
    use_sample = st.checkbox("Use included demonstration dataset", value=uploaded is None)
    forecast_period = st.selectbox("Forecast horizon", ["Next 24 hours", "Next 7 days"])
    tariff = st.number_input("Electricity tariff (₹ per kWh)", min_value=0.0, value=6.0, step=0.5)

@st.cache_data(show_spinner=False)
def get_data(file_bytes, use_demo):
    if file_bytes is not None:
        import io
        raw = io.BytesIO(file_bytes)
        return load_electricity_data(raw), "Uploaded dataset"
    if use_demo:
        return load_electricity_data(SAMPLE_DATA_PATH), "Simulated demonstration dataset"
    return None, "No dataset selected"

try:
    data, data_source = get_data(uploaded.getvalue() if uploaded else None, use_sample)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if data is None:
    st.info("Upload a CSV or select the included demonstration dataset from the sidebar.")
    st.stop()

hourly, missing_count = prepare_hourly_data(data)
if len(hourly) < 200:
    st.error("At least 200 hourly observations are needed. A longer dataset is recommended.")
    st.stop()

st.info(f"Data source: **{data_source}** · {len(hourly):,} hourly records · {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")
if data_source.startswith("Simulated"):
    st.warning("This is simulated data created for demonstration. Do not present it as real household measurements.")
if missing_count:
    st.warning(f"{missing_count:,} missing hourly values were detected before interpolation. Gaps longer than 6 hours are removed.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Dataset Explorer", "Model Training & Evaluation", "Forecast", "Bill Estimator"])

with tab1:
    st.markdown("### Project overview")
    st.write("ELECTRA analyses historical hourly electricity usage, learns patterns from time-related and lag features, and estimates future consumption.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average hourly use", f"{hourly['consumption_kwh'].mean():.3f} kWh")
    c2.metric("Peak hourly use", f"{hourly['consumption_kwh'].max():.3f} kWh")
    c3.metric("Total recorded use", f"{hourly['consumption_kwh'].sum():,.1f} kWh")
    st.pyplot(consumption_plot(hourly), use_container_width=True)
    st.markdown("**AI method:** supervised regression using a Random Forest model. No camera, image classification, or computer vision is used.")

with tab2:
    st.markdown("### Dataset explorer")
    st.dataframe(hourly.tail(100), use_container_width=True)
    st.write("Descriptive statistics")
    st.dataframe(hourly["consumption_kwh"].describe().to_frame("consumption_kwh"))
    st.download_button("Download cleaned hourly data", hourly.to_csv(index=False).encode("utf-8"), "electra_cleaned_hourly_data.csv", "text/csv")

with tab3:
    st.markdown("### Train and evaluate the model")
    st.write("The dataset is split chronologically: the earliest 80% is used for training and the latest 20% for testing. This avoids randomly mixing future observations into the training set.")
    if st.button("Train / retrain Random Forest model", type="primary"):
        with st.spinner("Training model and calculating evaluation metrics..."):
            try:
                bundle, metrics, test_results = train_model(hourly)
                st.session_state["electra_bundle"] = bundle
                st.session_state["electra_metrics"] = metrics
                st.session_state["electra_test"] = test_results
                save_metrics(metrics, METRICS_PATH)
                pd.DataFrame(test_results).to_csv(PREDICTIONS_PATH, index=False)
                st.success("Training completed.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    if "electra_metrics" in st.session_state:
        m = st.session_state["electra_metrics"]
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{m['mae']:.4f} kWh")
        b.metric("RMSE", f"{m['rmse']:.4f} kWh")
        c.metric("R²", f"{m['r2']:.4f}")
        d.metric("Baseline MAE", f"{m['baseline_mae']:.4f} kWh")
        st.write(f"Training rows: {m['train_rows']:,} · Test rows: {m['test_rows']:,} · {m['split']}")
        st.caption("Regression models do not have classification accuracy. MAE, RMSE and R² are the appropriate evaluation measures here.")
        with st.expander("What do these metrics mean?"):
            for label, explanation in metrics_explanation().items():
                st.write(f"**{label}:** {explanation}")
        test_df = pd.DataFrame(st.session_state["electra_test"])
        st.line_chart(test_df.set_index("timestamp")[["actual_kwh", "predicted_kwh"]])
        st.download_button("Download test predictions", test_df.to_csv(index=False).encode("utf-8"), "electra_test_predictions.csv", "text/csv")
    else:
        st.info("Train the model to display evaluation metrics.")

with tab4:
    st.markdown("### Future electricity consumption")
    if "electra_bundle" not in st.session_state:
        st.warning("Train the model in the Model Training & Evaluation tab first.")
    else:
        hours = 24 if forecast_period == "Next 24 hours" else 168
        forecast = forecast_hours(st.session_state["electra_bundle"], hourly, hours)
        st.pyplot(forecast_plot(hourly, forecast), use_container_width=True)
        total = float(forecast["predicted_kwh"].sum())
        st.metric(f"Predicted energy over {hours} hours", f"{total:.2f} kWh")
        st.dataframe(forecast, use_container_width=True)
        st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"), "electra_forecast.csv", "text/csv")
        st.session_state["electra_forecast"] = forecast

with tab5:
    st.markdown("### Estimated electricity cost")
    st.write("This is a simple estimate based on energy use × the tariff entered in the sidebar. Actual bills may include slabs, fixed charges, taxes, duties and adjustments.")
    if "electra_forecast" in st.session_state:
        forecast_energy = float(st.session_state["electra_forecast"]["predicted_kwh"].sum())
        cost = estimate_cost(forecast_energy, tariff)
        st.metric("Estimated cost for selected forecast period", f"₹{cost:,.2f}")
        st.caption(f"Based on {forecast_energy:.2f} kWh × ₹{tariff:.2f}/kWh.")
    else:
        st.info("Generate a forecast first to estimate its electricity cost.")
    st.divider()
    custom_energy = st.number_input("Or enter energy consumption (kWh)", min_value=0.0, value=100.0, step=1.0)
    st.metric("Custom cost estimate", f"₹{estimate_cost(custom_energy, tariff):,.2f}")

with st.expander("About the project, assumptions and limitations"):
    st.write("""
    - Input is converted to an hourly energy series. For the UCI source, Global_active_power is measured in kW; its hourly mean is used as an hourly kWh estimate.
    - Features include calendar information, previous-hour/day/week usage and rolling averages.
    - Forecasts are recursive: each predicted hour is used as part of the history for the next hour. Errors can accumulate over longer horizons.
    - Random Forest is a baseline student-project model, not a utility-grade forecasting system.
    - The demonstration dataset is synthetic. Use a real, appropriately permitted dataset for final experiments and clearly identify its source.
    - The cost estimator is illustrative and is not an official electricity bill calculator.
    """)
