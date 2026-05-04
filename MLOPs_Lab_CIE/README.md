# MLOps Lab CIE – SwimSync

## ?? Problem Statement

You are an MLOps Engineer at SwimSync. SwimSync provides performance analytics for competitive swimmers.

The objective is to predict lap time (in seconds) using stroke and biomechanics data.

### Features:
- stroke_rate (30–80)
- drag_coefficient (0.01–0.1)
- turn_time_ms (400–900)
- pool_length_m (25–50)

### Target:
- lap_time_seconds

A second dataset (`data/new_data.csv`) contains recent records with shifted distributions, used for retraining and monitoring model performance.

## Tasks Completed

1. Experiment Tracking & Model Comparison
   - Models: SVR, RandomForest
   - Best Model: RandomForest

2. Model Versioning
   - Registered model using MLflow
   - Version tracking implemented

3. Model Promotion
   - Challenger model trained
   - Alias "live" assigned

4. Retraining Pipeline
   - Combined datasets
   - Retrained model
   - Promotion based on RMSE improvement

## Folder Structure

MLOPs_Lab_CIE/
- data/
- src/
- models/
- results/
- requirements.txt

## Output Files

- step1_s1.json
- step2_s6.json
- step3_s7.json
- step4_s8.json