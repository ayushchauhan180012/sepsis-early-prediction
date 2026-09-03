import { useState, useCallback } from "react";
import { ObservationForm } from "./components/observation/ObservationForm";
import { PredictionResult } from "./components/observation/PredictionResult";
import { RiskTrajectory, PeakRiskBadge } from "./components/risk";
import { getPatientTrajectory } from "./api/patients";
import type { PredictionResponse, TrajectoryResponse, ApiError } from "./api/types";

function Dashboard() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [trajectory, setTrajectory] = useState<TrajectoryResponse | null>(null);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const [isLoadingTrajectory, setIsLoadingTrajectory] = useState(false);

  const fetchTrajectory = useCallback(async (patientId: string) => {
    setIsLoadingTrajectory(true);
    setTrajectoryError(null);
    try {
      const data = await getPatientTrajectory(patientId);
      setTrajectory(data);
    } catch (err) {
      if (err instanceof Error && "status" in err) {
        const apiError = err as ApiError;
        if (apiError.status === 404) {
          setTrajectoryError("No trajectory found for this patient.");
        } else if (apiError.status >= 500) {
          setTrajectoryError("Server error while fetching trajectory. Try refreshing.");
        } else {
          setTrajectoryError(`Failed to fetch trajectory (${apiError.status}).`);
        }
      } else if (err instanceof TypeError) {
        setTrajectoryError("Network error. Is the backend running?");
      } else {
        setTrajectoryError("An unexpected error occurred while fetching trajectory.");
      }
      setTrajectory(null);
    } finally {
      setIsLoadingTrajectory(false);
    }
  }, []);

  const handlePrediction = useCallback(
    (response: PredictionResponse) => {
      setPrediction(response);
      fetchTrajectory(response.patient_id);
    },
    [fetchTrajectory]
  );

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Early Sepsis Alert System</h1>
        <p className="app-subtitle">
          6-hour ahead sepsis risk prediction for ICU patients
        </p>
      </header>
      <section className="app-content">
        <section className="card observation-form-card">
          <ObservationForm onPrediction={handlePrediction} />
        </section>

        <PredictionResult prediction={prediction} />

        {prediction && (
          <section className="card risk-dashboard-card">
            <div className="risk-dashboard-header">
              <h2>Risk Dashboard</h2>
              {isLoadingTrajectory && <span className="loading-indicator">Loading trajectory…</span>}
            </div>

            {trajectoryError && (
              <div className="trajectory-error">
                <p>{trajectoryError}</p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => fetchTrajectory(prediction.patient_id)}
                >
                  Retry
                </button>
              </div>
            )}

            <div className="risk-dashboard-grid">
              <div className="risk-trajectory-wrapper">
                <RiskTrajectory
                  trajectory={trajectory?.trajectory ?? []}
                  peakRisk={trajectory?.peak_risk ?? null}
                />
              </div>
              <div className="peak-risk-wrapper">
                <PeakRiskBadge peakRisk={trajectory?.peak_risk ?? null} />
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

export default function App() {
  return <Dashboard />;
}