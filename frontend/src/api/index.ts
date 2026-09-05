// ---------------------------------------------------------------------------
// Public API surface for the frontend API layer.
// ---------------------------------------------------------------------------

export { apiFetch } from "./client";
export { getHealth, getReadiness } from "./health";
export { submitObservation } from "./predict";
export {
  getPatientTrajectory,
  getPatientAlerts,
  getPatientObservations,
} from "./patients";

export type {
  VitalSigns,
  LabValues,
  Observation,
  PredictionResponse,
  HealthResponse,
  ComponentStatus,
  ReadinessChecks,
  ReadinessResponse,
  TrajectoryPoint,
  PeakRisk,
  TrajectoryResponse,
  AlertSummary,
  AlertsResponse,
  ObservationRecord,
  ObservationsResponse,
} from "./types";
export { ApiError } from "./types";
