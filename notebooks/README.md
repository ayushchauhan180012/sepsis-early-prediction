# Notebooks – Sepsis Early Prediction Project

This directory contains development and system notebooks
for the ICU early sepsis detection project.

---

## Available Notebooks

### 1️⃣ early_sepsis_alert_system.ipynb  
Final deployment-oriented alert system.
Includes:
- Patient-wise train/test split
- Temporal feature engineering (delta, momentum, volatility)
- HistGradientBoosting model
- Threshold optimization
- 2-hour persistence alert logic
- Patient-level evaluation

This is the main system notebook.

---

### 2️⃣ Baseline Experiments (Archived)

- Baseline v1 – Raw physiological features  
- Baseline v2 – Rolling mean features  
- Baseline v3 – Delta features  

These notebooks document model progression
before deployment tuning.

---


### early_sepsis_alert_system.ipynb
# Early Sepsis Alert System (6-Hour Ahead Prediction)

## Overview

Sepsis is a life-threatening condition that requires early detection.
However, traditional rule-based systems suffer from high false alarm rates,
leading to alert fatigue in ICUs.

This project develops a **machine learning-based early warning system**
designed to balance early detection with practical alert reliability.

---

## Dataset

- PhysioNet Sepsis Dataset
- ICU time-series physiological data
- 6-hour ahead prediction target (FutureSepsis)

Train-test split was performed **patient-wise** to prevent data leakage.

---

## Methodology

### Feature Engineering

- Vital signs forward-filled per patient
- Lab missing indicators added
- Temporal dynamics captured using:
  - 6-hour change (delta6)
  - 1-hour momentum (delta1)
  - 6-hour volatility (rolling std)

### Model

- HistGradientBoostingClassifier
- Handles nonlinear interactions in ICU time-series data

---

## Alert System Design

Raw probability predictions are converted into practical alerts.

Final alert rule:

- Risk probability ≥ 0.04
- Sustained for 2 consecutive hours

This persistence filter reduces noisy spikes and false alarms.

---

## Final Performance (Patient-Level Evaluation)

- Precision ≈ 49%
- Recall ≈ 46%

This means:
Nearly 1 in 2 alerts correspond to true early sepsis cases.

---

## Key Insight

Improving deployment logic (alert persistence)
was more impactful than aggressive feature stacking.

Reducing alert fatigue is critical for real-world ICU adoption.

---

## Future Improvements

- Real-time streaming simulation
- Calibration analysis
- Threshold tuning per ICU policy
- External validation dataset

---
