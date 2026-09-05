import { RiskChart } from "./RiskChart";
import type { RiskChartDatum } from "./RiskChart";
import type { TrajectoryPoint, PeakRisk } from "../../api/types";
import "./RiskTrajectory.css";

interface RiskTrajectoryProps {
  trajectory: TrajectoryPoint[];
  peakRisk: PeakRisk | null;
  alertThreshold?: number;
  uncertaintyBand?: { lower: number; upper: number };
}

const ALERT_THRESHOLD = 0.045;
const UNCERTAINTY_LOWER = 0.035;
const UNCERTAINTY_UPPER = 0.055;

export function RiskTrajectory({
  trajectory,
  peakRisk,
  alertThreshold = ALERT_THRESHOLD,
  uncertaintyBand = { lower: UNCERTAINTY_LOWER, upper: UNCERTAINTY_UPPER },
}: RiskTrajectoryProps) {
  if (trajectory.length === 0) {
    return (
      <div className="risk-trajectory">
        <h2>Risk Trajectory</h2>
        <div className="empty-state">
          <p>No trajectory data available yet.</p>
          <p className="empty-hint">Submit observations to build the patient's risk history.</p>
        </div>
      </div>
    );
  }

  const chartData: RiskChartDatum[] = trajectory.map((point) => ({
    iculos: point.iculos,
    raw: point.raw_probability,
    filtered: point.filtered_probability,
  }));

  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <div className="risk-trajectory">
      <h2>Risk Trajectory</h2>

      <div className="chart-container">
        <RiskChart
          data={chartData}
          alertThreshold={alertThreshold}
          uncertaintyBand={uncertaintyBand}
        />
      </div>

      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color raw"></span>
          <span>Raw Risk Score (model output)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color filtered"></span>
          <span>Filtered Risk Score (after uncertainty filter)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color threshold"></span>
          <span>Alert Threshold (≥ 4.5% filtered risk, 2 consecutive hours)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color uncertainty"></span>
          <span>Uncertainty Band (3.5% – 5.5% raw risk → filtered to 0%)</span>
        </div>
      </div>

      {peakRisk && (
        <div className="peak-risk-inline">
          <span className="peak-label">Peak Risk:</span>
          <span className="peak-value">{formatScore(peakRisk.peak_risk)}%</span>
          <span className="peak-time">at ICULOS {peakRisk.iculos}</span>
        </div>
      )}
    </div>
  );
}