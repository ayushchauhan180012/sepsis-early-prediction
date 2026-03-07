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
- Shock Index (HR / SBP)
- Baseline Vital Deviation
- Lab Test Presence Indicators

- **Alert System Includes**
- Probability threshold filteringe
- 2-hour persistence alert logic
- Alert cooldown mechanism
- Patient-level evaluation
- The alert system includes:


This is the main system notebook.

---

### 2️⃣ Baseline Experiments (Archived)

- Baseline v1 – Raw physiological features  
- Baseline v2 – Rolling mean features  
- Baseline v3 – Delta features  

These notebooks document model progression
before deployment tuning.

---

## Recommended Entry Point

Start with:

`early_sepsis_alert_system.ipynb`
