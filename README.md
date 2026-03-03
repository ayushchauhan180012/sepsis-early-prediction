# 🧠 Early Sepsis Alert System (6-Hour Ahead Prediction)

A deployment-oriented ICU early warning system for predicting sepsis
6 hours before clinical onset using time-series physiological data.

---

## 🚨 Problem

Sepsis is life-threatening and requires early detection.  
However, naive rule-based alert systems often produce excessive false alarms,
leading to ICU alert fatigue.

This project aims to design a **clinically realistic early warning system**
that balances early detection with practical reliability.

---

## 📊 Dataset

- PhysioNet Sepsis Dataset (Training Set A)
- ICU time-series data
- Patient-wise train/test split (to prevent data leakage)

Target:

**FutureSepsis = 1 if sepsis occurs within next 6 hours**

---

## ⚙️ Methodology

### 1️⃣ Data Handling
- Patient-wise splitting
- Forward fill for vitals
- Missing indicators for labs
- Median fill using train statistics only

### 2️⃣ Temporal Feature Engineering
- 6-hour physiological change (delta6)
- 1-hour short-term momentum (delta1)
- 6-hour volatility (rolling standard deviation)

These features capture directional instability rather than static thresholds.

### 3️⃣ Model
HistGradientBoostingClassifier  
Selected due to strong nonlinear performance and stability.

---

## 📈 Model Performance

### ROC–AUC ≈ 0.75  
### PR–AUC ≈ 0.04  

Given the extreme class imbalance, focus shifts toward deployment metrics.

---

## 🚦 Alert System Design (Deployment Logic)

Raw probabilities are converted into actionable ICU alerts.

Final rule:

- Risk probability ≥ 0.04
- Sustained for 2 consecutive hours

This persistence filter significantly reduces noise-driven false alarms.

---

## 🎯 Final Patient-Level Performance

| Metric     | Value |
|------------|--------|
| Precision  | ≈ 49%  |
| Recall     | ≈ 46%  |

Nearly 1 in 2 alerts correspond to true early sepsis cases.

---

## 📊 Visual Results

See:
`notebooks/results/`

Includes:
- ROC curve
- Septic trajectory
- Non-septic trajectory
- Threshold selection tradeoff

---

## 🔬 Key Insight

Improving deployment logic (alert persistence)
was more impactful than aggressive feature stacking.

Alert stability > raw ROC optimization.

---

## 🚀 Future Improvements

- Calibration & probability smoothing
- External validation
- Real-time streaming simulation
- Threshold customization per ICU policy
