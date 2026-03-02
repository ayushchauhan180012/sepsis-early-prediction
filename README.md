# Sepsis Early Prediction (6-Hour Ahead)

## Problem
To Predict sepsis 6 hours before clinical onset using ICU time-series data.

## Dataset
PhysioNet Sepsis Dataset (training_setA).

## Target
FutureSepsis:
Label = 1 if sepsis occurs within next 6 hours.

## Approach
- Patient-wise train/test split
- Missing handling:
  - Vitals forward fill
  - Labs kept as NaN
  - Missing indicators added
- Feature Engineering:
  - Baseline (raw vitals)
  - Rolling mean (tested)
  - Delta (6-hour change)

## Current Best Baseline Result
Model: XGBoost  
ROC-AUC: 0.7223  
PR-AUC: 0.0385  

## Future Work
- Add short-term delta1 features
- Volatility features
- Logistic regression baseline
- Hyperparameter tuning
