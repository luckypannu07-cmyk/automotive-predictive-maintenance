from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

app = FastAPI()

# ---------------------------------------
# Load Latest Model Automatically
# ---------------------------------------

client = MlflowClient()

experiment = client.get_experiment_by_name("predictive_maintenance")

if experiment is None:
    raise Exception("No experiment found. Train model first.")

runs = client.search_runs(experiment.experiment_id)

if len(runs) == 0:
    raise Exception("No runs found. Train model first.")

latest_run_id = runs[0].info.run_id

model_uri = f"runs:/{latest_run_id}/model"

model = mlflow.pyfunc.load_model(model_uri)

# ---------------------------------------
# Prediction Endpoint
# ---------------------------------------

@app.post("/predict")
def predict(data: dict):

    # Expected feature columns (must match training)
    required_columns = (
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.DataFrame([data])

    # Add missing columns as 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[required_columns]

    prediction = model.predict(df)

    return {"prediction": float(prediction[0])}


