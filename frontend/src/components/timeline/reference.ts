// ---------------------------------------------------------------------------
// M11 Clinical Timeline — shared series metadata and timeline types.
//
// The reference ranges defined here are VISUAL DISPLAY REFERENCES ONLY. They
// have no effect on model inference, risk filtering, alerting, thresholds, or
// any backend behavior. They are documented as such in the UI and never fed
// back into prediction or alert logic.
// ---------------------------------------------------------------------------

import type { CSSProperties } from "react";

export type VitalKey = "hr" | "o2sat" | "sbp" | "map" | "resp" | "temp";
export type LabKey = "lactate" | "wbc" | "creatinine" | "platelets";
export type OverlayKey = "rawRisk" | "uncertaintyBand" | "alertBands";

export interface NormalRange {
  low: number;
  high: number;
}

export interface SeriesMeta {
  label: string;
  color: string;
  unit: string;
  normal: NormalRange;
}

/** Display-only typical adult reference ranges for plotting. */
export const VITAL_META: Record<VitalKey, SeriesMeta> = {
  hr: { label: "HR", color: "#0d6efd", unit: "bpm", normal: { low: 60, high: 100 } },
  o2sat: { label: "O2Sat", color: "#0dcaf0", unit: "%", normal: { low: 95, high: 100 } },
  sbp: { label: "SBP", color: "#6f42c1", unit: "mmHg", normal: { low: 90, high: 120 } },
  map: { label: "MAP", color: "#fd7e14", unit: "mmHg", normal: { low: 70, high: 100 } },
  resp: { label: "Resp", color: "#198754", unit: "/min", normal: { low: 12, high: 20 } },
  temp: { label: "Temp", color: "#d63384", unit: "°C", normal: { low: 36.0, high: 37.5 } },
};

/** Display-only typical adult reference ranges for plotting. Labs are sparse. */
export const LAB_META: Record<LabKey, SeriesMeta> = {
  lactate: { label: "Lactate", color: "#dc3545", unit: "mmol/L", normal: { low: 0.5, high: 2.2 } },
  wbc: { label: "WBC", color: "#0d6efd", unit: "×10⁹/L", normal: { low: 4.5, high: 11.0 } },
  creatinine: { label: "Creatinine", color: "#6f42c1", unit: "mg/dL", normal: { low: 0.6, high: 1.2 } },
  platelets: { label: "Platelets", color: "#198754", unit: "×10⁹/L", normal: { low: 150, high: 450 } },
};

/** Vitals are grouped by physiologic subsystem so each sub-panel keeps a legible Y scale. */
export interface VitalGroup {
  id: string;
  label: string;
  keys: VitalKey[];
}

export const VITAL_GROUPS: VitalGroup[] = [
  { id: "cardio", label: "Cardio", keys: ["hr", "sbp", "map"] },
  { id: "resp", label: "Resp", keys: ["o2sat", "resp"] },
  { id: "temp", label: "Temp", keys: ["temp"] },
];

export interface TimelineSelection {
  vitals: Record<VitalKey, boolean>;
  labs: Record<LabKey, boolean>;
  overlays: Record<OverlayKey, boolean>;
}

export const DEFAULT_SELECTION: TimelineSelection = {
  vitals: { hr: true, o2sat: false, sbp: false, map: true, resp: false, temp: true },
  labs: { lactate: true, wbc: true, creatinine: false, platelets: false },
  overlays: { rawRisk: true, uncertaintyBand: true, alertBands: true },
};

export const OVERLAY_LABELS: Record<OverlayKey, string> = {
  rawRisk: "Raw risk",
  uncertaintyBand: "Uncertainty band",
  alertBands: "Alert bands",
};

/** A maximal contiguous run of trajectory points where `alert === true`. */
export interface AlertRun {
  startIculos: number;
  endIculos: number;
  durationHours: number;
}

export const REFERENCE_RANGE_NOTE =
  "Shaded bands and reference notes show typical adult reference ranges for visual context only. They do not affect predictions, filtering, or alerts.";

export const LAB_SPARSITY_NOTE =
  "Lab results are plotted only at the hours a result was recorded. Gaps mean no result that hour — lab values are never carried forward or interpolated.";

export const SYNC_ID = "clinical-timeline";

export const TOOLTIP_CONTENT_STYLE: CSSProperties = {
  backgroundColor: "#fff",
  border: "1px solid #dee2e6",
  borderRadius: "8px",
  boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
};

export function formatNormalRange(meta: SeriesMeta): string {
  const { low, high } = meta.normal;
  return `Ref ${low}–${high} ${meta.unit}`;
}