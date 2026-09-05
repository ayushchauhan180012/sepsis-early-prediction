// ---------------------------------------------------------------------------
// Patient endpoint functions.
// ---------------------------------------------------------------------------

import { apiFetch } from "./client";
import type {
  TrajectoryResponse,
  AlertsResponse,
  ObservationsResponse,
} from "./types";

/** GET /patients/{patientId}/trajectory — full risk trajectory for a patient. */
export function getPatientTrajectory(patientId: string): Promise<TrajectoryResponse> {
  return apiFetch<TrajectoryResponse>(
    `/patients/${encodeURIComponent(patientId)}/trajectory`,
  );
}

/** GET /patients/{patientId}/alerts — aggregated alert statistics. */
export function getPatientAlerts(patientId: string): Promise<AlertsResponse> {
  return apiFetch<AlertsResponse>(
    `/patients/${encodeURIComponent(patientId)}/alerts`,
  );
}

/** GET /patients/{patientId}/observations — full raw observation history. */
export function getPatientObservations(
  patientId: string,
): Promise<ObservationsResponse> {
  return apiFetch<ObservationsResponse>(
    `/patients/${encodeURIComponent(patientId)}/observations`,
  );
}
