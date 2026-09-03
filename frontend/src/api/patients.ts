// ---------------------------------------------------------------------------
// Patient endpoint functions.
// ---------------------------------------------------------------------------

import { apiFetch } from "./client";
import type { TrajectoryResponse, AlertsResponse } from "./types";

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
