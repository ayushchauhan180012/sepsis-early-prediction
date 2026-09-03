// ---------------------------------------------------------------------------
// Health / readiness endpoint functions.
// ---------------------------------------------------------------------------

import { apiFetch } from "./client";
import type { HealthResponse, ReadinessResponse } from "./types";

/** GET /health — liveness check (model loaded). */
export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

/** GET /health/ready — full readiness check (model + database). */
export function getReadiness(): Promise<ReadinessResponse> {
  return apiFetch<ReadinessResponse>("/health/ready");
}
