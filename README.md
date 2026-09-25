# Cancer Risk Prediction App

An interactive Streamlit web application that predicts cancer risk based on patient clinical and lifestyle data, using and comparing **4 trained machine learning models**.

## How It Works

The app takes patient inputs (clinical and lifestyle factors) and returns a cancer risk prediction using a pre-trained scikit-learn / XGBoost model. Users can interactively switch between 4 different ML models and instantly compare their performance metrics before generating a prediction.

## Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 94.60% | 0.92 | 0.90 | 0.91 |
| **KNN** | 92.80% | 0.95 | 0.79 | 0.86 |
| **XGBoost** | 91.74% | 0.98 | 0.73 | 0.84 |
| **Decision Tree** | 87.88% | 0.85 | 0.71 | 0.77 |

## Features

- 🔀 **Multi-model comparison** — switch between Logistic Regression, Decision Tree, KNN, and XGBoost in real time
- 📊 **Live performance metrics** — Accuracy, Precision, Recall, and F1-Score shown as interactive donut gauges for the selected model
- 🎯 **Risk prediction** — patient details generate an instant risk probability, visualized as a donut chart
- 🎨 Clean, professional, responsive UI

## Tech Stack

- Python
- Streamlit (web interface)
- scikit-learn (Logistic Regression, Decision Tree, KNN)
- XGBoost
- Plotly (interactive charts)
- pandas, joblib

## Files

- `cancer_risk_prediction.py` — Streamlit app (multi-model UI)
- `model_logistic.pkl` — trained Logistic Regression model
- `model_decision_tree.pkl` — trained Decision Tree model
- `model_knn.pkl` — trained KNN model
- `model_xgboost.pkl` — trained XGBoost model
- `model_metrics.pkl` — saved performance metrics for all models
- `scaler.pkl` — fitted StandardScaler for feature scaling
- `requirements.txt` — dependencies

## Live Demo

[Live App](https://arif-cancer-risk-prediction.streamlit.app/)

## Disclaimer

This tool is built for educational/demo purposes as part of a data analytics training program. It is not a substitute for professional medical advice or diagnosis.
