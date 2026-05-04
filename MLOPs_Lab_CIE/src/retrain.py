import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

TRAIN_PATH = "data/training_data.csv"
NEW_PATH = "data/new_data.csv"
RESULT_PATH = "results/step4_s8.json"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
new_df = pd.read_csv(NEW_PATH)

# Combine
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Original split (IMPORTANT: same test set)
X = train_df.drop("lap_time_seconds", axis=1)
y = train_df["lap_time_seconds"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Champion model (same type as Task 1 winner)
champion = RandomForestRegressor(random_state=42)
champion.fit(X_train, y_train)

champion_rmse = np.sqrt(mean_squared_error(y_test, champion.predict(X_test)))

# Retrain on combined data
X_comb = combined_df.drop("lap_time_seconds", axis=1)
y_comb = combined_df["lap_time_seconds"]

retrained = RandomForestRegressor(random_state=42)
retrained.fit(X_comb, y_comb)

retrained_rmse = np.sqrt(mean_squared_error(y_test, retrained.predict(X_test)))

# Compare
improvement = champion_rmse - retrained_rmse

action = "promoted" if improvement >= 0.5 else "kept_champion"

# Output JSON
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": round(champion_rmse, 4),
    "retrained_rmse": round(retrained_rmse, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 complete.")
print(json.dumps(output, indent=4))