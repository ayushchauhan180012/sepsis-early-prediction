import type { PeakRisk } from "../../api/types";
import { RiskChart } from "../risk";
import type { RiskChartDatum } from "../risk";
import type { AlertRun } from "./reference";

interface RiskPanelProps {
  data: RiskChartDatum[];
  alertRuns: AlertRun[];
  peakRisk: PeakRisk | null;
  showRawRisk: boolean;
  showUncertaintyBand: boolean;
  showAlertBands: boolean;
  syncId: string;
  height?: number;
}

const formatScore = (score: number) => (score * 100).toFixed(1);

export function RiskPanel({
  data,
  alertRuns,
  peakRisk,
  showRawRisk,
  showUncertaintyBand,
  showAlertBands,
  syncId,
  height = 180,
}: RiskPanelProps) {
  const hasRiskData = data.length > 0;

  return (
    <section className="timeline-panel">
      <div className="timeline-panel-header">
        <h3 className="timeline-panel-title">Risk</h3>
        {peakRisk && (
          <span className="timeline-peak">
            Peak risk <strong>{formatScore(peakRisk.peak_risk)}%</strong> at ICULOS{" "}
            {peakRisk.iculos}
          </span>
        )}
      </div>

      {hasRiskData ? (
        <>
          <RiskChart
            data={data}
            showRawRisk={showRawRisk}
            showUncertaintyBand={showUncertaintyBand}
            alertRuns={alertRuns}
            showAlertBands={showAlertBands}
            syncId={syncId}
            height={height}
            showLegend={false}
            showXAxisLabel={false}
            showYAxisLabel={false}
          />
          <div className="timeline-chart-legend">
            <div className="timeline-legend-item">
              <span className="timeline-legend-line timeline-legend-raw"></span>
              <span>Raw risk</span>
            </div>
            <div className="timeline-legend-item">
              <span className="timeline-legend-line timeline-legend-filtered"></span>
              <span>Filtered risk</span>
            </div>
            <div className="timeline-legend-item">
              <span className="timeline-legend-line timeline-legend-threshold"></span>
              <span>Alert threshold</span>
            </div>
            <div className="timeline-legend-item">
              <span className="timeline-legend-band timeline-legend-alertband"></span>
              <span>Alert episode</span>
            </div>
          </div>
        </>
      ) : (
        <div className="timeline-empty-hint">
          <p>No risk scores recorded for this patient yet.</p>
          <p className="timeline-empty-hint-sub">
            Submit observations to build the risk trajectory.
          </p>
        </div>
      )}
    </section>
  );
}