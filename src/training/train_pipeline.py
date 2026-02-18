import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_training(df):

    # -------------------------
    # Prepare Data
    # -------------------------
    X = df.drop(["engine_id", "cycle", "RUL"], axis=1)
    y = df["RUL"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Define Model
    # -------------------------
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )

    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run():

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # -------------------------
        # Metrics
        # -------------------------
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # -------------------------
        # True vs Predicted Plot
        # -------------------------
        plt.figure()
        plt.scatter(y_val, preds)
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title("True vs Predicted")
        plt.tight_layout()
        plt.savefig("true_vs_pred.png")
        mlflow.log_artifact("true_vs_pred.png")
        plt.close()

        # -------------------------
        # Residual Plot
        # -------------------------
        residuals = y_val - preds

        plt.figure()
        plt.hist(residuals, bins=30)
        plt.title("Residual Distribution")
        plt.tight_layout()
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")
        plt.close()

        # -------------------------
        # SHAP Feature Importance
        # -------------------------
        X_sample = X_val.sample(200, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        # Save model
        mlflow.sklearn.log_model(model, "model")

    print("✅ Model training completed successfully!")
