// ---------------------------------------------------------------------------
// RiskChart — the shared risk-line chart body.
//
// Extracted from RiskTrajectory so the M11 Clinical Timeline risk panel can
// reuse the exact same rendering (threshold line, uncertainty band, colors,
// tooltip). All additions are OPTIONAL props with defaults that reproduce the
// original RiskTrajectory output byte-for-byte when not overridden.
// ---------------------------------------------------------------------------

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AlertRun } from "../timeline/reference";
import { TOOLTIP_CONTENT_STYLE } from "../timeline/reference";

export interface RiskChartDatum {
  iculos: number;
  raw: number | null;
  filtered: number | null;
}

interface RiskChartProps {
  data: RiskChartDatum[];
  alertThreshold?: number;
  uncertaintyBand?: { lower: number; upper: number };
  /** Show the raw-risk line (model output). Defaults to true. */
  showRawRisk?: boolean;
  /** Show the uncertainty-band reference lines. Defaults to true. */
  showUncertaintyBand?: boolean;
  /** Contiguous alert=true episodes to shade. Defaults to none. */
  alertRuns?: AlertRun[];
  /** Render alert-run shading. Defaults to false. */
  showAlertBands?: boolean;
  /** Recharts legend inside the chart. Defaults to true. */
  showLegend?: boolean;
  /** Bottom "ICULOS (ICU Hour)" axis label. Defaults to true. */
  showXAxisLabel?: boolean;
  /** Rotated "Risk Score" Y-axis label (expands the left gutter). Defaults to true. */
  showYAxisLabel?: boolean;
  /** Shared chart sync id for synchronized hover across panels. */
  syncId?: string;
  height?: number;
}

const ALERT_THRESHOLD = 0.045;
const UNCERTAINTY_LOWER = 0.035;
const UNCERTAINTY_UPPER = 0.055;

export function RiskChart({
  data,
  alertThreshold = ALERT_THRESHOLD,
  uncertaintyBand = { lower: UNCERTAINTY_LOWER, upper: UNCERTAINTY_UPPER },
  showRawRisk = true,
  showUncertaintyBand = true,
  alertRuns = [],
  showAlertBands = false,
  showLegend = true,
  showXAxisLabel = true,
  showYAxisLabel = true,
  syncId,
  height = 360,
}: RiskChartProps) {
  const maxIculos = Math.max(...data.map((d) => d.iculos));
  const maxProbability = Math.max(
    ...data.map((d) => Math.max(d.raw ?? 0, d.filtered ?? 0)),
    alertThreshold,
    uncertaintyBand.upper
  );
  const yDomainMax = Math.min(Math.max(maxProbability * 1.2, 0.1), 1);
  const formatScore = (score: number) => (score * 100).toFixed(1);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={data}
        margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
        syncId={syncId}
        syncMethod="index"
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#dee2e6" vertical={false} />
        <XAxis
          dataKey="iculos"
          type="number"
          domain={[1, maxIculos || 1]}
          tick={{ fontSize: 12, fill: "#6c757d" }}
          axisLine={{ stroke: "#dee2e6" }}
          tickLine={{ stroke: "#dee2e6" }}
          label={
            showXAxisLabel
              ? { value: "ICULOS (ICU Hour)", position: "bottom", offset: 30, fontSize: 13, fill: "#212529" }
              : undefined
          }
          allowDecimals={false}
        />
        <YAxis
          type="number"
          domain={[0, yDomainMax]}
          tick={{ fontSize: 12, fill: "#6c757d" }}
          axisLine={{ stroke: "#dee2e6" }}
          tickLine={{ stroke: "#dee2e6" }}
          width={showYAxisLabel ? undefined : 40}
          label={
            showYAxisLabel
              ? { value: "Risk Score", angle: -90, position: "left", offset: 20, fontSize: 13, fill: "#212529" }
              : undefined
          }
          tickFormatter={(value) => (value * 100).toFixed(0) + "%"}
        />
        <Tooltip
          contentStyle={TOOLTIP_CONTENT_STYLE}
          labelFormatter={(iculos) => `ICULOS: ${iculos}`}
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter={(value: any, name: any) => {
            if (value === undefined || value === null) return ["—", name ?? "Value"];
            const numValue = typeof value === "number" ? value : Number(value);
            const label = name === "raw" ? "Raw Risk" : "Filtered Risk";
            return [`${formatScore(numValue)}%`, label];
          }}
        />
        {showLegend && (
          <Legend
            wrapperStyle={{ paddingTop: 10 }}
            iconSize={12}
            formatter={(value) => value}
          />
        )}

        {showAlertBands &&
          alertRuns.map((run) => (
            <ReferenceArea
              key={`${run.startIculos}-${run.endIculos}`}
              x1={run.startIculos - 0.45}
              x2={run.endIculos + 0.45}
              y1={0}
              y2={yDomainMax}
              fill="#dc3545"
              fillOpacity={0.12}
              stroke="none"
              ifOverflow="extendDomain"
            />
          ))}

        {data.length > 1 && (
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
            {showUncertaintyBand && (
              <>
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
          </>
        )}

        {showRawRisk && (
          <Line
            type="monotone"
            dataKey="raw"
            stroke="#0d6efd"
            strokeWidth={2}
            dot={data.length <= 1}
            activeDot={{ r: 6, strokeWidth: 2 }}
            name="Raw Risk"
            connectNulls={true}
          />
        )}
        <Line
          type="monotone"
          dataKey="filtered"
          stroke="#198754"
          strokeWidth={2}
          strokeDasharray="4 4"
          dot={data.length <= 1}
          activeDot={{ r: 6, strokeWidth: 2 }}
          name="Filtered Risk"
          connectNulls={true}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}