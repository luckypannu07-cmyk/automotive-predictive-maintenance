import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import requests
from mlflow.tracking import MlflowClient
from src.load_data import load_training_data
from src.build_features import add_rul
from src.drift_monitor import detect_drift, plot_drift

st.set_page_config(layout="wide")
st.title("🚗 Automotive Predictive Maintenance Platform")

client = MlflowClient()

# ---------------------------------------
# MODEL COMPARISON SECTION
# ---------------------------------------

st.header("📊 Model Comparison")

experiment = client.get_experiment_by_name("predictive_maintenance")

if experiment:
    runs = client.search_runs(experiment.experiment_id)

    data = []
    for run in runs:
        data.append({
            "Run ID": run.info.run_id,
            "RMSE": run.data.metrics.get("RMSE"),
            "MAE": run.data.metrics.get("MAE"),
            "R2": run.data.metrics.get("R2")
        })

    df_runs = pd.DataFrame(data)
    st.dataframe(df_runs)

    if not df_runs.empty:
        best = df_runs.sort_values("RMSE").iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Best RMSE", round(best["RMSE"], 2))
        col2.metric("Best MAE", round(best["MAE"], 2))
        col3.metric("Best R2", round(best["R2"], 2))

        latest_run = runs[0]

        st.subheader("📂 Artifacts (Visualizations)")
        artifacts = client.list_artifacts(latest_run.info.run_id)

        for artifact in artifacts:
            if artifact.path.endswith(".png"):
                local_path = client.download_artifacts(
                    latest_run.info.run_id,
                    artifact.path
                )
                st.image(local_path)

# ---------------------------------------
# DRIFT MONITORING SECTION
# ---------------------------------------

st.header("📈 Drift Monitoring")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "train_FD001.txt")

st.write("Drift dataset path:", data_path)

try:
    df = load_training_data(data_path)
    df = add_rul(df)

    reference_df = df.drop(["engine_id", "cycle", "RUL"], axis=1)
    new_df = reference_df.sample(500, random_state=42)  # simulate new data

    drift_results = detect_drift(reference_df, new_df)

    drift_df = pd.DataFrame({
        "Feature": drift_results.keys(),
        "P-Value": drift_results.values()
    })

    st.dataframe(drift_df)

    feature_selected = st.selectbox(
        "Select Feature to Visualize Drift",
        list(reference_df.columns)
    )

    fig = plot_drift(reference_df, new_df, feature_selected)
    st.pyplot(fig)

except:
    st.warning("Dataset not found for drift monitoring.")

# ---------------------------------------
# LIVE PREDICTION SECTION
# ---------------------------------------

st.header("🔮 Live Prediction")

sensor_input = {}

col1, col2 = st.columns(2)

for i in range(1, 22):
    if i <= 11:
        sensor_input[f"sensor_{i}"] = col1.number_input(
            f"Sensor {i}",
            value=round(np.random.normal(0, 1), 3)
        )
    else:
        sensor_input[f"sensor_{i}"] = col2.number_input(
            f"Sensor {i}",
            value=round(np.random.normal(0, 1), 3)
        )

# Add operation settings automatically
sensor_input["op_setting_1"] = 0.0
sensor_input["op_setting_2"] = 0.0
sensor_input["op_setting_3"] = 0.0

if st.button("Predict Remaining Useful Life"):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=sensor_input
        )
        result = response.json()
        st.success(f"Predicted RUL: {result['prediction']:.2f}")
    except:
        st.error("API not running. Start FastAPI server first.")
