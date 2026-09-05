// ---------------------------------------------------------------------------
// M11 Clinical Timeline — pure data derivation helpers.
//
// All functions are pure and never mutate their inputs. They merge the risk
// trajectory and raw observation history client-side, keyed by ICULOS. ICULOS
// is the sole timeline axis; `received_at` is not used as an X coordinate.
// ---------------------------------------------------------------------------

import type { ObservationRecord, TrajectoryPoint } from "../../api/types";
import type { AlertRun, LabKey, VitalKey } from "./reference";

/** Maximum number of points rendered per panel. Only the rendered
 *  representation is decimated; API/persisted data is never modified. */
export const MAX_RENDER_POINTS = 500;

/** One merged, per-ICULOS timeline row combining risk and raw vitals/labs. */
export interface MergedRow {
  iculos: number;
  raw: number | null;
  filtered: number | null;
  highRisk: boolean | null;
  alert: boolean | null;
  hr: number | null;
  o2sat: number | null;
  sbp: number | null;
  map: number | null;
  resp: number | null;
  temp: number | null;
  lactate: number | null;
  wbc: number | null;
  creatinine: number | null;
  platelets: number | null;
}

type NumericRowKey = VitalKey | LabKey;

/** Merge trajectory + observations into a single ICULOS-keyed, ascending array. */
export function buildTimelineRows(
  trajectory: TrajectoryPoint[],
  observations: ObservationRecord[],
): MergedRow[] {
  const obsByIculos = new Map<number, ObservationRecord>();
  for (const obs of observations) {
    obsByIculos.set(obs.iculos, obs);
  }
  const trajByIculos = new Map<number, TrajectoryPoint>();
  for (const point of trajectory) {
    trajByIculos.set(point.iculos, point);
  }

  const iculosSet = new Set<number>();
  for (const obs of observations) {
    iculosSet.add(obs.iculos);
  }
  for (const point of trajectory) {
    iculosSet.add(point.iculos);
  }

  const sorted = Array.from(iculosSet).sort((a, b) => a - b);
  return sorted.map((iculos) => {
    const obs = obsByIculos.get(iculos);
    const point = trajByIculos.get(iculos);
    return {
      iculos,
      raw: point?.raw_probability ?? null,
      filtered: point?.filtered_probability ?? null,
      highRisk: point?.high_risk ?? null,
      alert: point?.alert ?? null,
      hr: obs?.hr ?? null,
      o2sat: obs?.o2sat ?? null,
      sbp: obs?.sbp ?? null,
      map: obs?.map ?? null,
      resp: obs?.resp ?? null,
      temp: obs?.temp ?? null,
      lactate: obs?.lactate ?? null,
      wbc: obs?.wbc ?? null,
      creatinine: obs?.creatinine ?? null,
      platelets: obs?.platelets ?? null,
    };
  });
}

/** Extract alert episodes ONLY from contiguous trajectory points with
 *  `alert === true`. No threshold-based or model-logic inference here. */
export function extractAlertRuns(trajectory: TrajectoryPoint[]): AlertRun[] {
  const runs: AlertRun[] = [];
  let startIculos: number | null = null;
  let prevIculos: number | null = null;

  for (const point of trajectory) {
    if (point.alert) {
      if (startIculos === null) {
        startIculos = point.iculos;
      } else if (prevIculos !== null && point.iculos !== prevIculos + 1) {
        runs.push({
          startIculos,
          endIculos: prevIculos,
          durationHours: prevIculos - startIculos + 1,
        });
        startIculos = point.iculos;
      }
      prevIculos = point.iculos;
    } else if (startIculos !== null && prevIculos !== null) {
      runs.push({
        startIculos,
        endIculos: prevIculos,
        durationHours: prevIculos - startIculos + 1,
      });
      startIculos = null;
      prevIculos = null;
    }
  }

  if (startIculos !== null && prevIculos !== null) {
    runs.push({
      startIculos,
      endIculos: prevIculos,
      durationHours: prevIculos - startIculos + 1,
    });
  }

  return runs;
}

/** Render-only decimation. Keeps the most recent point so live timelines
 *  always end at the latest ICULOS. Never touches the source arrays. */
export function decimateRows(rows: MergedRow[], maxPoints: number = MAX_RENDER_POINTS): MergedRow[] {
  if (rows.length <= maxPoints) {
    return rows;
  }
  const step = Math.ceil(rows.length / maxPoints);
  const result: MergedRow[] = [];
  for (let i = 0; i < rows.length; i += step) {
    const row = rows[i];
    if (row !== undefined) {
      result.push(row);
    }
  }
  const last = rows[rows.length - 1];
  if (last !== undefined && result[result.length - 1] !== last) {
    result.push(last);
  }
  return result;
}

/** Shared X domain (ICULOS) across all panels. Sole timeline axis. */
export function computeXDomain(rows: MergedRow[]): [number, number] {
  if (rows.length === 0) {
    return [1, 1];
  }
  const min = rows[0]?.iculos ?? 1;
  const max = rows[rows.length - 1]?.iculos ?? 1;
  if (min === max) {
    return [Math.max(1, min - 1), max + 1];
  }
  return [min, max];
}

/** Y domain for a set of series, including any display reference ranges. */
export function computeSeriesDomain(
  rows: MergedRow[],
  keys: ReadonlyArray<NumericRowKey>,
  ranges?: ReadonlyArray<{ low: number; high: number }>,
): [number, number] {
  let min = Infinity;
  let max = -Infinity;

  for (const key of keys) {
    for (const row of rows) {
      const value = row[key];
      if (value !== null) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
  }

  if (ranges) {
    for (const range of ranges) {
      if (range.low < min) min = range.low;
      if (range.high > max) max = range.high;
    }
  }

  if (min === Infinity || max === -Infinity) {
    return [0, 1];
  }

  const pad = Math.max((max - min) * 0.15, 1);
  return [Math.max(0, min - pad), max + pad];
}

export function formatSeriesValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}