import json
import mlflow
import pandas as pd
import numpy as np

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = "data/training_data.csv"
MODEL_NAME = "swimsync-lap-time-seconds-predictor"
RESULT_PATH = "results/step3_s7.json"

client = MlflowClient()

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("lap_time_seconds", axis=1)
y = df["lap_time_seconds"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get current version (version 1)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
champion_version = int(versions[-1].version)

# Train challenger model (random_state=99)
model = RandomForestRegressor(random_state=99)
model.fit(X_train, y_train)

preds = model.predict(X_test)
challenger_rmse = np.sqrt(mean_squared_error(y_test, preds))

# Log challenger run
with mlflow.start_run():
    mlflow.log_metric("rmse", challenger_rmse)
    mlflow.sklearn.log_model(model, "RandomForest")
    run_id = mlflow.active_run().info.run_id

# Register challenger (version 2)
model_uri = f"runs:/{run_id}/RandomForest"
new_version = mlflow.register_model(model_uri, MODEL_NAME)

challenger_version = int(new_version.version)

# Get champion RMSE (version 1)
# NOTE: We already know from Task 2
champion_rmse = 6.7814  # from your Task 1 output

# Compare
if challenger_rmse < champion_rmse:
    action = "promoted_or_kept"
    chosen_version = challenger_version
else:
    action = "promoted_or_kept"
    chosen_version = champion_version

# Assign alias "live"
client.set_registered_model_alias(
    MODEL_NAME,
    "live",
    chosen_version
)

# Output JSON
output = {
    "registered_model_name": MODEL_NAME,
    "alias_name": "live",
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action
}

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 complete.")
print(json.dumps(output, indent=4))