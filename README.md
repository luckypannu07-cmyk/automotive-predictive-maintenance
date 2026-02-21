Automotive Predictive Maintenance Platform End-to-End Machine
Learning System for Remaining Useful Life (RUL) Prediction

OVERVIEW This project implements a production-oriented Predictive
Maintenance system designed for automotive and industrial applications.

The system predicts the Remaining Useful Life (RUL) of engines using
multivariate sensor data and provides:

-   Model training & evaluation
-   Experiment tracking with MLflow
-   SHAP-based interpretability
-   Drift monitoring
-   REST API deployment (FastAPI)
-   Interactive monitoring dashboard (Streamlit)

The architecture simulates a real-world MLOps pipeline suitable for
automotive environments such as BMW, Porsche, and Mercedes-Benz.

PROBLEM STATEMENT In modern automotive manufacturing, unexpected
component failure leads to: - Production downtime - Increased
maintenance costs - Safety risks - Reduced operational efficiency

This system predicts engine degradation using sensor data to enable
proactive maintenance decisions.

SYSTEM ARCHITECTURE Data → Feature Engineering → Model Training → MLflow
Tracking ↓ Model Deployment (FastAPI) ↓ Monitoring Dashboard (Streamlit)
↓ Drift Detection + Live Prediction

TECHNOLOGY STACK - Python - Pandas / NumPy - Scikit-learn - MLflow
(Experiment Tracking) - SHAP (Model Explainability) - FastAPI
(Production API) - Streamlit (Monitoring Dashboard) - Matplotlib

FEATURES

Model Training - Random Forest Regressor for RUL prediction - RMSE, MAE,
R² evaluation metrics - Model artifact logging - Experiment tracking

Model Interpretability - SHAP feature importance analysis - Residual
distribution visualization - True vs Predicted performance analysis

Drift Monitoring - Kolmogorov-Smirnov statistical drift detection -
Feature-level drift visualization - Early anomaly detection

Production API - RESTful prediction endpoint - Dynamic feature
validation - Automatic handling of missing inputs

Monitoring Dashboard - Model comparison view - Best model selection -
Drift visualization - Live prediction interface

PROJECT STRUCTURE

Autonomous-ML-System/ │ ├── data/raw/ ├── src/ │ ├── load_data.py │ ├──
build_features.py │ ├── train_pipeline.py │ ├── drift_monitor.py │ ├──
api_server.py │ └── dashboard_app.py │ ├── main.py └── requirements.txt

HOW TO RUN

1)  Train Model python main.py

2)  Start MLflow mlflow ui

3)  Start API Server uvicorn src.api_server:app –reload

4)  Launch Dashboard streamlit run src/dashboard_app.py

DATASET NASA Turbofan Engine Degradation Simulation Dataset Used for
modeling engine lifecycle and failure prediction.

ENGINEERING HIGHLIGHTS - Modular and scalable architecture -
Reproducible experiment tracking - Integrated model monitoring -
Simulated production deployment workflow - Built with
production-readiness mindset

FUTURE IMPROVEMENTS - Model Registry with production/staging promotion -
Automated retraining trigger - CI/CD integration - Containerization -
Real-time streaming ingestion

AUTHOR Data Scientist & Machine Learning Engineer Focused on building
scalable, production-grade AI systems for industrial and automotive
applications.
