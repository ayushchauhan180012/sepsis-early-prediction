// ---------------------------------------------------------------------------
// Prediction endpoint function.
// ---------------------------------------------------------------------------

import { apiFetch } from "./client";
import type { Observation, PredictionResponse } from "./types";

/** POST /predict — submit one hourly observation and receive a risk prediction. */
export function submitObservation(obs: Observation): Promise<PredictionResponse> {
  return apiFetch<PredictionResponse>("/predict", { method: "POST", json: obs });
}
