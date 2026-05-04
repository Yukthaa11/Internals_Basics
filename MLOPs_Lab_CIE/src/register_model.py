import json
import mlflow
from mlflow.tracking import MlflowClient

RESULT_PATH = "results/step2_s6.json"
MODEL_NAME = "swimsync-lap-time-seconds-predictor"

client = MlflowClient()

# Get experiment
experiment = mlflow.get_experiment_by_name("swimsync-lap-time-seconds")

# Get all runs
runs = mlflow.search_runs(experiment.experiment_id)

# Sort by RMSE (lowest is best)
best_run = runs.sort_values("metrics.rmse").iloc[0]

run_id = best_run.run_id
rmse = best_run["metrics.rmse"]

# This gives model name (SVR or RandomForest)
model_name = best_run["tags.mlflow.runName"]

# Register model
model_uri = f"runs:/{run_id}/{model_name}"
registered_model = mlflow.register_model(model_uri, MODEL_NAME)

# Output JSON
output = {
    "registered_model_name": MODEL_NAME,
    "version": registered_model.version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": round(rmse, 4)
}

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 complete.")
print(json.dumps(output, indent=4))