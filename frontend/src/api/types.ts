// ---------------------------------------------------------------------------
// TypeScript types for the Early Sepsis Alert System backend API.
//
// Derived from actual backend source:
//   - Backend/Services/validation.py  (Pydantic models: Health, PredictionResponse)
//   - Backend/app.py                 (FastAPI routes + response shapes)
//   - Backend/Database/operations.py (analytics query return shapes)
// ---------------------------------------------------------------------------

// ── POST /predict request body (Pydantic: Health) ──────────────────────────

/** Required core vital signs. */
export interface VitalSigns {
  HR: number; // heart rate, 20–250, non-zero
  O2Sat: number; // oxygen saturation, 0–100
  SBP: number; // systolic blood pressure, 40–300, non-zero
  MAP: number; // mean arterial pressure, 20–250, non-zero
  Resp: number; // respiration rate, 0–80, non-zero
  Temp: number; // body temperature, 25–45 (realistic 30–43)
}

/** Optional lab values — null when not yet tested. */
export interface LabValues {
  Lactate: number | null; // >= 0
  WBC: number | null; // >= 0
  Creatinine: number | null; // >= 0
  Platelets: number | null; // >= 0
}

/** Full observation payload for POST /predict. */
export interface Observation extends VitalSigns, LabValues {
  Age: number; // 0–120
  ICULOS: number; // >= 1
  PatientID: string;
}

// ── POST /predict response (Pydantic: PredictionResponse) ──────────────────

export interface PredictionResponse {
  patient_id: string;
  iculos: number; // >= 1
  raw_probability: number; // 0–1
  filtered_probability: number; // 0–1
  high_risk: boolean;
  alert: boolean;
}

// ── GET /health response ───────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok";
  model_loaded: true;
}

// ── GET /health/ready response ─────────────────────────────────────────────

export type ComponentStatus = "ok" | "degraded";

export interface ReadinessChecks {
  model: ComponentStatus;
  database: ComponentStatus;
}

export interface ReadinessResponse {
  status: "ok" | "degraded";
  checks: ReadinessChecks;
}

// ── GET /patients/{id}/trajectory response ─────────────────────────────────

export interface TrajectoryPoint {
  iculos: number;
  raw_probability: number;
  filtered_probability: number;
  high_risk: boolean;
  alert: boolean;
}

export interface PeakRisk {
  peak_risk: number;
  iculos: number;
}

export interface TrajectoryResponse {
  patient_id: string;
  trajectory: TrajectoryPoint[];
  peak_risk: PeakRisk | null;
}

// ── GET /patients/{id}/alerts response ─────────────────────────────────────

export interface AlertSummary {
  total_alerts: number;
  total_alert_hours: number;
  first_alert_iculos: number;
  last_alert_iculos: number;
  max_peak_risk: number;
}

export interface AlertsResponse {
  patient_id: string;
  alert_summary: AlertSummary | null;
}

// ── Error types ────────────────────────────────────────────────────────────

/** Thrown for non-2xx HTTP responses. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  readonly requestId: string | null;

  constructor(status: number, body: unknown, requestId: string | null) {
    super(`API error ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.requestId = requestId;
  }
}
