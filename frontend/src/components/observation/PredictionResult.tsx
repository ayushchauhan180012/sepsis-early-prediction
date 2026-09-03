import type { PredictionResponse } from "../../api/types";
import "./ObservationForm.css";

interface PredictionResultProps {
  prediction: PredictionResponse | null;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  if (!prediction) {
    return null;
  }

  const { patient_id, iculos, raw_probability, filtered_probability, high_risk, alert } = prediction;

  const riskLevel = high_risk ? "High Risk" : "Low Risk";
  const riskClass = high_risk ? "risk-high" : "risk-low";
  const alertClass = alert ? "alert-active" : "alert-inactive";

  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <section className="prediction-result">
      <h2>Prediction Result</h2>

      <div className="result-grid">
        <div className="result-item">
          <span className="result-label">Patient ID</span>
          <span className="result-value">{patient_id}</span>
        </div>
        <div className="result-item">
          <span className="result-label">ICULOS (Hour)</span>
          <span className="result-value">{iculos}</span>
        </div>
        <div className="result-item">
          <span className="result-label">Raw Risk Score</span>
          <span className="result-value">{formatScore(raw_probability)}%</span>
        </div>
        <div className="result-item">
          <span className="result-label">Filtered Risk Score</span>
          <span className="result-value">{formatScore(filtered_probability)}%</span>
        </div>
        <div className="result-item">
          <span className="result-label">Risk Classification</span>
          <span className={`result-value ${riskClass}`}>{riskLevel}</span>
        </div>
        <div className="result-item">
          <span className="result-label">Alert Status</span>
          <span className={`result-value ${alertClass}`}>{alert ? "Alert Triggered" : "No Alert"}</span>
        </div>
      </div>

      <div className="result-note">
        <p>
          <strong>Note:</strong> Scores represent model risk estimates (0–1 scale), not clinically calibrated
          probabilities. The filtered score applies the alert engine's temporal smoothing; the raw score is the
          direct model output.
        </p>
      </div>
    </section>
  );
}