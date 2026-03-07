# Early Sepsis Prediction (6-Hour Ahead)

A machine learning system for **early detection of sepsis using ICU time-series data**.

The model predicts the **risk of sepsis up to 6 hours before clinical onset**, allowing earlier medical intervention and reducing mortality risk.

This project focuses on **building a clinically realistic alert system** that balances early detection with reduced false alarms.

---

# Problem

Sepsis is a life-threatening medical condition caused by the body’s extreme response to infection.  
It can rapidly lead to **organ failure and death** if not treated early.

Studies show that **every hour of delayed treatment significantly increases mortality**.

Therefore, **predicting sepsis hours before onset is critical for ICU care**.

---

# Project Goal

Build a machine learning model that:

• Predicts **sepsis 6 hours before onset**  
• Uses **ICU physiological time-series data**  
• Reduces **false alerts (alert fatigue)**  
• Produces **clinically interpretable risk trajectories**

---

# Dataset

PhysioNet Sepsis Challenge Dataset

Contains hourly ICU patient records including:

• Vital signs  
• Laboratory values  
• Demographic features  

Example features:
HR, O2Sat, Temp, SBP, MAP, Resp
Lactate, WBC, Creatinine, Platelets


Characteristics:

• Time-series ICU data  
• Highly imbalanced (~2% sepsis cases)  
• Missing clinical measurements  

---

# Methodology

## 1. Patient-Wise Train/Test Split

To avoid **data leakage**, patients are split rather than rows.
Train patients : 80%
Test patients : 20%


---

# 2. Missing Data Strategy

Clinical datasets contain large amounts of missing values.

Handling strategy:

• **Forward fill vitals per patient**
• **Median imputation (train statistics)**
• **Missing indicators for laboratory values**

---

# 3. Temporal Feature Engineering

Instead of static values, the model learns **physiological trends over time**.

Features created:

### Change Features
- Delta6 → change over last 6 hours
- Delta1 → short-term momentum

### Volatility Features
- Rolling 6-hour standard deviation

### Patient Baseline Deviation
- Current value - patient's early baseline

### Laboratory Monitoring Signals
- Recent test indicators


These capture **clinical deterioration patterns**.

---

# Model

Model used:

**HistGradientBoostingClassifier**

Reasons:

• Handles missing values well  
• Strong performance on tabular clinical data  
• Efficient for large datasets  

---

# Alert System Design

Instead of predicting sepsis directly, the model outputs **hourly risk probabilities**.

Alerts are generated using:

### Risk Threshold
- Alert if probability ≥ 0.045


### Temporal Persistence
- Alert only if high-risk persists for 2 consecutive hours


### Cooldown Logic
- Avoid repeated alerts within a short time window


This reduces **false alarms and alert fatigue** in ICU environments.

---

# Results

## Model Performance


ROC-AUC ≈ 0.756
PR-AUC ≈ 0.040


## Patient-Level Alert Performance
- Precision ≈ 62%
- Recall ≈ 42%


Interpretation:

• When the model raises an alert, **~62% are true sepsis cases**  
• The model detects **~42% of septic patients early**

This provides **clinically useful early warnings while limiting false alarms**.

---

# Visualization

### ROC Curve

![ROC Curve](notebooks/results/roc_curve.png)

---

### Sepsis Risk Trajectory (Septic Patient)

![Septic Patient](notebooks/results/septic_patient_trajectory.png)

The model gradually increases risk **before clinical sepsis onset**.

---

### Risk Trajectory (Non-Septic Patient)

![Non-Septic Patient](notebooks/results/non_septic_patient_trajectory.png)

Risk remains **consistently low**, indicating stable condition.

---

### Precision-Recall Tradeoff

![Tradeoff](notebooks/results/threshold_tradeoff.png)

Shows the balance between early detection and false alarms.

---

# Key Achievements

• Built **early sepsis prediction model (6 hours ahead)**  
• Designed **clinically realistic alert system**  
• Reduced **false alarm rate via persistence logic**  
• Achieved **high precision for medical alerts (~62%)**  
• Implemented **temporal clinical feature engineering**

---

# Future Improvements

Potential upgrades:

• Deep learning models (LSTM / Transformer for time series)  
• Conformal prediction for uncertainty detection  
• Multi-hospital generalization testing  
• Real-time ICU deployment simulation

---

# Tech Stack

Python

Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost


---

# Repository Structure


sepsis-early-prediction
│
├── notebooks
│ ├── early_sepsis_alert_system.ipynb
│ └── results
│ ├── roc_curve.png
│ ├── septic_patient_trajectory.png
│ ├── non_septic_patient_trajectory.png
│ └── threshold_tradeoff.png
│
└── README.md


---

# Author

Ayush Chauhan

Computer Science & Engineering (AI/ML)

Project: Early Sepsis Prediction System
