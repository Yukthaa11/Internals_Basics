import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_PATH = "data/training_data.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop("lap_time_seconds", axis=1)
y = df["lap_time_seconds"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow
mlflow.set_experiment("swimsync-lap-time-seconds")

models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []
best_model = None
best_rmse = float("inf")
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        # Log params
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Save model
        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

# Save best model
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

# Save output JSON
output = {
    "experiment_name": "swimsync-lap-time-seconds",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": round(best_rmse, 4)
}

with open(f"{RESULTS_DIR}/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 complete.")
print(json.dumps(output, indent=4))