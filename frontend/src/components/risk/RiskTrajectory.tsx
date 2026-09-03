import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from "recharts";
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

  const chartData = trajectory.map((point) => ({
    iculos: point.iculos,
    raw: point.raw_probability,
    filtered: point.filtered_probability,
    highRisk: point.high_risk,
    alert: point.alert,
  }));

  const maxIculas = Math.max(...chartData.map((d) => d.iculos));
  const maxProbability = Math.max(
    ...chartData.map((d) => Math.max(d.raw, d.filtered)),
    alertThreshold,
    uncertaintyBand.upper
  );

  const yDomainMax = Math.min(Math.max(maxProbability * 1.2, 0.1), 1);

  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <div className="risk-trajectory">
      <h2>Risk Trajectory</h2>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#dee2e6" vertical={false} />
            <XAxis
              dataKey="iculos"
              type="number"
              domain={[1, maxIculas || 1]}
              tick={{ fontSize: 12, fill: "#6c757d" }}
              axisLine={{ stroke: "#dee2e6" }}
              tickLine={{ stroke: "#dee2e6" }}
              label={{ value: "ICULOS (ICU Hour)", position: "bottom", offset: 30, fontSize: 13, fill: "#212529" }}
              allowDecimals={false}
            />
            <YAxis
              type="number"
              domain={[0, yDomainMax]}
              tick={{ fontSize: 12, fill: "#6c757d" }}
              axisLine={{ stroke: "#dee2e6" }}
              tickLine={{ stroke: "#dee2e6" }}
              label={{ value: "Risk Score", angle: -90, position: "left", offset: 20, fontSize: 13, fill: "#212529" }}
              tickFormatter={(value) => (value * 100).toFixed(0) + "%"}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#fff",
                border: "1px solid #dee2e6",
                borderRadius: "8px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
              }}
              labelFormatter={(iculos) => `ICULOS: ${iculos}`}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(value: any, name: any) => {
                if (value === undefined || value === null) return ["—", name ?? "Value"];
                const numValue = typeof value === "number" ? value : Number(value);
                const label = name === "raw" ? "Raw Risk" : "Filtered Risk";
                return [`${formatScore(numValue)}%`, label];
              }}
            />
            <Legend
              wrapperStyle={{ paddingTop: 10 }}
              iconSize={12}
              formatter={(value) => value}
            />

            {trajectory.length > 1 && (
              <>
                <ReferenceLine
                  y={alertThreshold}
                  stroke="#dc3545"
                  strokeDasharray="5 5"
                  strokeWidth={1.5}
                  label={{
                    value: `Alert Threshold (${formatScore(alertThreshold)}%)`,
                    position: "right",
                    fill: "#dc3545",
                    fontSize: 11,
                    fontWeight: 500,
                    offset: 5,
                  }}
                />
                <ReferenceLine
                  y={uncertaintyBand.lower}
                  stroke="#ffc107"
                  strokeDasharray="2 4"
                  strokeWidth={1}
                  label={{
                    value: `Uncertainty Lower (${formatScore(uncertaintyBand.lower)}%)`,
                    position: "right",
                    fill: "#ffc107",
                    fontSize: 10,
                    offset: 5,
                  }}
                />
                <ReferenceLine
                  y={uncertaintyBand.upper}
                  stroke="#ffc107"
                  strokeDasharray="2 4"
                  strokeWidth={1}
                  label={{
                    value: `Uncertainty Upper (${formatScore(uncertaintyBand.upper)}%)`,
                    position: "right",
                    fill: "#ffc107",
                    fontSize: 10,
                    offset: 5,
                  }}
                />
              </>
            )}

            <Line
              type="monotone"
              dataKey="raw"
              stroke="#0d6efd"
              strokeWidth={2}
              dot={trajectory.length <= 1}
              activeDot={{ r: 6, strokeWidth: 2 }}
              name="Raw Risk"
              connectNulls={true}
            />
            <Line
              type="monotone"
              dataKey="filtered"
              stroke="#198754"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={trajectory.length <= 1}
              activeDot={{ r: 6, strokeWidth: 2 }}
              name="Filtered Risk"
              connectNulls={true}
            />
          </LineChart>
        </ResponsiveContainer>
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