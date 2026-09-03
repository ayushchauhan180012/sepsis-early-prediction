import type { PeakRisk } from "../../api/types";
import "./PeakRiskBadge.css";

interface PeakRiskBadgeProps {
  peakRisk: PeakRisk | null;
}

const formatScore = (score: number) => (score * 100).toFixed(1);

export function PeakRiskBadge({ peakRisk }: PeakRiskBadgeProps) {
  if (!peakRisk) {
    return (
      <div className="peak-risk-badge empty">
        <span className="badge-label">Peak Risk</span>
        <span className="badge-value empty-value">—</span>
        <span className="badge-hint">No data available</span>
      </div>
    );
  }

  const riskPercent = formatScore(peakRisk.peak_risk);
  const riskClass = peakRisk.peak_risk >= 0.045 ? "high" : "low";

  return (
    <div className="peak-risk-badge">
      <span className="badge-label">Peak Risk</span>
      <span className={`badge-value ${riskClass}`}>{riskPercent}%</span>
      <span className="badge-time">at ICULOS {peakRisk.iculos}</span>
    </div>
  );
}