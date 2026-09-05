import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";
import { computeSeriesDomain, formatSeriesValue } from "./timelineData";
import type { MergedRow } from "./timelineData";
import {
  LAB_META,
  TOOLTIP_CONTENT_STYLE,
  formatNormalRange,
} from "./reference";
import type { LabKey } from "./reference";

interface LabsPanelProps {
  rows: MergedRow[];
  labs: Record<LabKey, boolean>;
  xDomain: [number, number];
  syncId: string;
}

export function LabsPanel({ rows, labs, xDomain, syncId }: LabsPanelProps) {
  const activeKeys = useMemo(
    () => Object.keys(LAB_META).filter((key) => labs[key as LabKey]) as LabKey[],
    [labs]
  );

  const availableKeys = useMemo(
    () =>
      activeKeys.filter((key) => rows.some((row) => row[key] !== null)),
    [activeKeys, rows]
  );

  if (availableKeys.length === 0) {
    const anyLabRecorded = rows.some((row) =>
      Object.keys(LAB_META).some((key) => row[key as LabKey] !== null)
    );
    return (
      <section className="timeline-panel">
        <h3 className="timeline-panel-title">Labs</h3>
        <div className="timeline-empty-hint">
          {activeKeys.length === 0 ? (
            <>
              <p>No labs selected.</p>
              <p className="timeline-empty-hint-sub">
                Choose a lab in the selector above to show its trend.
              </p>
            </>
          ) : anyLabRecorded ? (
            <>
              <p>None of the selected labs have recorded results yet.</p>
              <p className="timeline-empty-hint-sub">
                Labs are sparse — results appear at the hours a test was run.
              </p>
            </>
          ) : (
            <>
              <p>No lab results recorded for this patient yet.</p>
              <p className="timeline-empty-hint-sub">
                Submit observations with lab values to plot laboratory trends.
              </p>
            </>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className="timeline-panel">
      <h3 className="timeline-panel-title">Labs</h3>

      {availableKeys.map((key) => {
        const meta = LAB_META[key];
        const domain = computeSeriesDomain(rows, [key], [meta.normal]);

        return (
          <div className="timeline-sub-panel" key={key}>
            <div className="timeline-sub-panel-title-row">
              <h4 className="timeline-sub-panel-title">{meta.label}</h4>
              <span className="timeline-sub-panel-ref">
                {formatNormalRange(meta)} — visual reference only
              </span>
            </div>
            <div className="timeline-sub-chart">
              <ResponsiveContainer width="100%" height={110}>
                <LineChart
                  data={rows}
                  margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                  syncId={syncId}
                  syncMethod="index"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#dee2e6" vertical={false} />
                  <XAxis
                    dataKey="iculos"
                    type="number"
                    domain={xDomain}
                    tick={{ fontSize: 12, fill: "#6c757d" }}
                    axisLine={{ stroke: "#dee2e6" }}
                    tickLine={{ stroke: "#dee2e6" }}
                    allowDecimals={false}
                  />
                  <YAxis
                    type="number"
                    domain={domain}
                    width={40}
                    tick={{ fontSize: 12, fill: "#6c757d" }}
                    axisLine={{ stroke: "#dee2e6" }}
                    tickLine={{ stroke: "#dee2e6" }}
                    tickFormatter={(value) => formatSeriesValue(value as number)}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_CONTENT_STYLE}
                    labelFormatter={(label) => `ICULOS ${label}`}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: any, name: any) => {
                      const labKey = name as LabKey;
                      const meta = LAB_META[labKey];
                      const label = meta?.label ?? name;
                      if (value === undefined || value === null || value === "No result") {
                        return ["No result", label];
                      }
                      return [`${formatSeriesValue(Number(value))} ${meta?.unit ?? ""}`, label];
                    }}
                  />

                  <ReferenceArea
                    x1={xDomain[0]}
                    x2={xDomain[1]}
                    y1={meta.normal.low}
                    y2={meta.normal.high}
                    fill={meta.color}
                    fillOpacity={0.08}
                    stroke="none"
                    ifOverflow="extendDomain"
                  />

                  <Line
                    type="monotone"
                    dataKey={key}
                    stroke={meta.color}
                    strokeWidth={2}
                    dot={{ r: 3, strokeWidth: 0 }}
                    activeDot={{ r: 5, strokeWidth: 2 }}
                    name={key}
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </section>
  );
}