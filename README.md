# ELECTRA — AI-Based Electricity Consumption Prediction and Energy Analytics System

A Class 12 AI project prototype that uses structured time-series data and a Random Forest regression model to analyse and forecast hourly electricity consumption. **It does not use computer vision, cameras, or image classification.**

## 1. Features
- CSV upload and data validation
- Support for the UCI Individual Household Electric Power Consumption dataset
- Hourly data preparation and missing-value handling
- Exploratory analysis and visualisation
- Time, calendar, lag and rolling-average features
- Chronological 80/20 train/test split
- Random Forest regression with a simple mean baseline
- MAE, MSE, RMSE and R² evaluation
- Recursive 24-hour or 7-day forecast
- User-entered tariff-based cost estimate
- CSV downloads for cleaned data, test predictions and forecast

## 2. Dataset
Suggested real dataset: UCI Machine Learning Repository — Individual Household Electric Power Consumption:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The UCI file uses `Date`, `Time` and `Global_active_power`. The application converts `Global_active_power` (kW) into hourly average power; for a one-hour interval, the numeric hourly mean in kW is used as an hourly kWh estimate.

The included `data/sample_data.csv` is **simulated demonstration data**, not real household readings. Do not report its results as real-world findings. Replace it with a genuine dataset for final evaluation where possible.

## 3. Setup
Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 4. CSV formats
Preferred format:
```csv
timestamp,consumption_kwh
2025-01-01 00:00:00,0.42
2025-01-01 01:00:00,0.38
```

The app also accepts the UCI format containing `Date`, `Time`, and `Global_active_power`.

For a custom dataset, use one row per hour if possible. Use a long enough time range (at least 6 months is preferable; 200 valid hourly rows is the minimum technical check).

## 5. Methodology
1. Load and validate the data.
2. Convert the readings to a sorted hourly series.
3. Interpolate short gaps (up to six hours); discard remaining missing rows.
4. Create calendar features and lag/rolling features.
5. Train on the first 80% of time and test on the final 20%.
6. Compare Random Forest with a mean baseline.
7. Forecast recursively for the selected future horizon.

## 6. Evaluation
- **MAE:** mean absolute prediction error, in kWh.
- **MSE:** mean squared error.
- **RMSE:** square root of MSE, in kWh.
- **R²:** coefficient of determination; it may be negative.
- **Baseline MAE:** mean absolute error from a simple mean predictor.

This is a regression task, so do not describe R² or any regression metric as classification accuracy. Record the metrics produced by your own run; do not invent or pre-fill results.

## 7. Limitations and responsible use
- A forecast is an estimate, not a guarantee.
- Recursive forecasts may accumulate error, especially over seven days.
- Random Forest does not automatically understand unusual events, holidays, appliance changes or weather unless such information is included in the data.
- The tariff calculator excludes tariff slabs, fixed charges, taxes, duties, subsidies and other utility-specific adjustments.
- Use only data you have permission to use. Avoid collecting names, addresses, account numbers or other unnecessary personal information.
- Clearly label synthetic/demo data and distinguish it from real observations.

## 8. Suggested six-member team roles
Adapt the teacher-provided role designations to the new project only after the teacher approves the change from the original waste-segregation project title.

1. **Team Leader & Project Coordinator:** coordinates the team, identifies the electricity-consumption problem, plans activities and monitors progress.
2. **Research Analyst:** researches energy-use patterns, electricity forecasting and existing AI approaches.
3. **Data Collection Manager:** finds a suitable public electricity dataset and records its source, licence and relevant columns.
4. **Data Processing & Annotation Manager:** checks missing values, duplicates, timestamps, units and data quality; prepares the hourly dataset.
5. **AI Model Developer:** develops the feature-engineering and Random Forest regression pipeline and records model settings.
6. **Testing, Documentation & Presentation Manager:** tests the app, records genuine results and limitations, and prepares the report and presentation.

## 9. Logbook evidence to collect
Use your school's supplied IBM-CBSE AI Project Logbook and complete its sections in the order given by the teacher. Keep evidence of:
- problem definition and supporting research
- target users and genuine user feedback
- brainstorming and reasons for selecting this solution
- system design and user workflow
- dataset source, variables, permissions and privacy considerations
- data preparation and feature engineering
- prototype screenshots and training configuration
- test cases, observations and improvements
- team meeting minutes and each member's actual contribution
- individual reflections and the project video link

Do not invent interviews, user feedback, testing outcomes, model scores or meeting minutes. Add dates and screenshots from the work your team actually completes.

## 10. Troubleshooting
- If training fails for insufficient rows, use a longer dataset.
- If the CSV is not recognised, rename columns to `timestamp` and `consumption_kwh`.
- If the app cannot import modules, run Streamlit from the ELECTRA project root.
- If packages are missing, activate the virtual environment and rerun `pip install -r requirements.txt`.
