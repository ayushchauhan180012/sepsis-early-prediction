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
import { TOOLTIP_CONTENT_STYLE, VITAL_GROUPS, VITAL_META } from "./reference";
import type { VitalKey } from "./reference";

interface VitalsPanelProps {
  rows: MergedRow[];
  vitals: Record<VitalKey, boolean>;
  xDomain: [number, number];
  syncId: string;
}

export function VitalsPanel({ rows, vitals, xDomain, syncId }: VitalsPanelProps) {
  const activeGroups = useMemo(
    () =>
      VITAL_GROUPS.map((group) => ({
        ...group,
        activeKeys: group.keys.filter((key) => vitals[key]),
      })).filter((group) => group.activeKeys.length > 0),
    [vitals]
  );

  const hasAnyVitalData = useMemo(
    () => rows.some((row) => VITAL_GROUPS.some((g) => g.keys.some((k) => row[k] !== null))),
    [rows]
  );

  if (activeGroups.length === 0) {
    return (
      <section className="timeline-panel">
        <h3 className="timeline-panel-title">Vitals</h3>
        <div className="timeline-empty-hint">
          <p>No vitals selected.</p>
          <p className="timeline-empty-hint-sub">
            Choose a vital in the selector above to show its hourly trend.
          </p>
        </div>
      </section>
    );
  }

  if (!hasAnyVitalData) {
    return (
      <section className="timeline-panel">
        <h3 className="timeline-panel-title">Vitals</h3>
        <div className="timeline-empty-hint">
          <p>No vital observations recorded for this patient yet.</p>
          <p className="timeline-empty-hint-sub">
            Submit observations to plot hourly vital trends.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="timeline-panel">
      <h3 className="timeline-panel-title">Vitals</h3>

      {activeGroups.map((group) => {
        const domain = computeSeriesDomain(
          rows,
          group.activeKeys,
          group.activeKeys.map((key) => VITAL_META[key].normal)
        );

        return (
          <div className="timeline-sub-panel" key={group.id}>
            <h4 className="timeline-sub-panel-title">{group.label}</h4>
            <div className="timeline-sub-chart">
              <ResponsiveContainer width="100%" height={140}>
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
                      const key = name as VitalKey;
                      const meta = VITAL_META[key];
                      if (value === undefined || value === null) return ["—", meta?.label ?? name];
                      return [`${formatSeriesValue(Number(value))} ${meta?.unit ?? ""}`, meta?.label ?? name];
                    }}
                  />

                  {group.activeKeys.map((key) => {
                    const meta = VITAL_META[key];
                    return (
                      <ReferenceArea
                        key={key}
                        x1={xDomain[0]}
                        x2={xDomain[1]}
                        y1={meta.normal.low}
                        y2={meta.normal.high}
                        fill={meta.color}
                        fillOpacity={0.08}
                        stroke="none"
                        ifOverflow="extendDomain"
                      />
                    );
                  })}

                  {group.activeKeys.map((key) => {
                    const meta = VITAL_META[key];
                    return (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={meta.color}
                        strokeWidth={2}
                        dot={rows.length <= 24 ? { r: 2, strokeWidth: 0 } : false}
                        activeDot={{ r: 5, strokeWidth: 2 }}
                        name={key}
                        connectNulls={true}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </section>
  );
}