import { useState, useRef, useCallback } from "react";
import { ObservationForm } from "./components/observation/ObservationForm";
import { PredictionResult } from "./components/observation/PredictionResult";
import { RiskTrajectory, PeakRiskBadge } from "./components/risk";
import { AlertSummary } from "./components/alerts";
import { PatientLoader, PatientHistory } from "./components/patient";
import { SystemStatus } from "./components/status";
import {
  getPatientTrajectory,
  getPatientAlerts,
  getPatientObservations,
} from "./api/patients";
import type {
  PredictionResponse,
  TrajectoryResponse,
  AlertsResponse,
  ObservationsResponse,
  ApiError,
} from "./api/types";

function translateTrajectoryError(err: unknown): string {
  if (err instanceof Error && "status" in err) {
    const apiError = err as ApiError;
    if (apiError.status === 404) {
      return "No trajectory found for this patient.";
    }
    if (apiError.status >= 500) {
      return "Server error while fetching trajectory. Try refreshing.";
    }
    return `Failed to fetch trajectory (${apiError.status}).`;
  }
  if (err instanceof TypeError) {
    return "Network error. Is the backend running?";
  }
  return "An unexpected error occurred while fetching trajectory.";
}

function translateAlertError(err: unknown): string {
  if (err instanceof Error && "status" in err) {
    const apiError = err as ApiError;
    if (apiError.status === 404) {
      return "No alert data found for this patient.";
    }
    if (apiError.status >= 500) {
      return "Server error while fetching alert data. Try refreshing.";
    }
    return `Failed to fetch alert data (${apiError.status}).`;
  }
  if (err instanceof TypeError) {
    return "Network error. Is the backend running?";
  }
  return "An unexpected error occurred while fetching alert data.";
}

function translateObservationsError(err: unknown): string {
  if (err instanceof Error && "status" in err) {
    const apiError = err as ApiError;
    if (apiError.status === 404) {
      return "No observation history found for this patient.";
    }
    if (apiError.status >= 500) {
      return "Server error while fetching observation history. Try refreshing.";
    }
    return `Failed to fetch observation history (${apiError.status}).`;
  }
  if (err instanceof TypeError) {
    return "Network error. Is the backend running?";
  }
  return "An unexpected error occurred while fetching observation history.";
}

function Dashboard() {
  const [activePatientId, setActivePatientId] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [recallActive, setRecallActive] = useState(false);
  const [trajectory, setTrajectory] = useState<TrajectoryResponse | null>(null);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const [isLoadingTrajectory, setIsLoadingTrajectory] = useState(false);

  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [alertError, setAlertError] = useState<string | null>(null);
  const [isLoadingAlerts, setIsLoadingAlerts] = useState(false);

  const [observations, setObservations] = useState<ObservationsResponse | null>(null);
  const [observationsError, setObservationsError] = useState<string | null>(null);
  const [isLoadingObservations, setIsLoadingObservations] = useState(false);

  const loadTokenRef = useRef(0);

  const loadPatient = useCallback((patientId: string) => {
    const token = ++loadTokenRef.current;

    setActivePatientId(patientId);
    setTrajectory(null);
    setTrajectoryError(null);
    setIsLoadingTrajectory(true);
    setAlerts(null);
    setAlertError(null);
    setIsLoadingAlerts(true);
    setObservations(null);
    setObservationsError(null);
    setIsLoadingObservations(true);

    getPatientTrajectory(patientId)
      .then((data) => {
        if (loadTokenRef.current === token) {
          setTrajectory(data);
        }
      })
      .catch((err: unknown) => {
        if (loadTokenRef.current === token) {
          setTrajectoryError(translateTrajectoryError(err));
        }
      })
      .finally(() => {
        if (loadTokenRef.current === token) {
          setIsLoadingTrajectory(false);
        }
      });

    getPatientAlerts(patientId)
      .then((data) => {
        if (loadTokenRef.current === token) {
          setAlerts(data);
        }
      })
      .catch((err: unknown) => {
        if (loadTokenRef.current === token) {
          setAlertError(translateAlertError(err));
        }
      })
      .finally(() => {
        if (loadTokenRef.current === token) {
          setIsLoadingAlerts(false);
        }
      });

    getPatientObservations(patientId)
      .then((data) => {
        if (loadTokenRef.current === token) {
          setObservations(data);
        }
      })
      .catch((err: unknown) => {
        if (loadTokenRef.current === token) {
          setObservationsError(translateObservationsError(err));
        }
      })
      .finally(() => {
        if (loadTokenRef.current === token) {
          setIsLoadingObservations(false);
        }
      });
  }, []);

  const handlePrediction = useCallback(
    (response: PredictionResponse) => {
      setRecallActive(false);
      setPrediction(response);
      loadPatient(response.patient_id);
    },
    [loadPatient]
  );

  const handlePatientLoad = useCallback(
    (patientId: string) => {
      setRecallActive(true);
      setPrediction(null);
      loadPatient(patientId);
    },
    [loadPatient]
  );

  const retryTrajectory = useCallback(() => {
    if (activePatientId) {
      loadPatient(activePatientId);
    }
  }, [activePatientId, loadPatient]);

  const retryAlerts = useCallback(() => {
    if (activePatientId) {
      loadPatient(activePatientId);
    }
  }, [activePatientId, loadPatient]);

  const retryObservations = useCallback(() => {
    if (activePatientId) {
      loadPatient(activePatientId);
    }
  }, [activePatientId, loadPatient]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="app-header-text">
          <h1>Early Sepsis Alert System</h1>
          <p className="app-subtitle">
            6-hour ahead sepsis risk prediction for ICU patients
          </p>
        </div>
        <SystemStatus />
      </header>

      <section className="app-content">
        <section className="patient-control-section">
          <PatientLoader
            onLoad={handlePatientLoad}
            loading={isLoadingTrajectory || isLoadingAlerts}
          />
        </section>

        <section className="card observation-form-card">
          <ObservationForm onPrediction={handlePrediction} />
        </section>

        <PredictionResult prediction={prediction} />

        {activePatientId && (
          <section className="card risk-dashboard-card">
            <div className="risk-dashboard-header">
              <div className="risk-dashboard-title">
                <h2>Risk Dashboard</h2>
                <span className="risk-dashboard-patient">
                  Patient: {activePatientId}
                </span>
              </div>
              {(isLoadingTrajectory || isLoadingAlerts) && (
                <span className="loading-indicator">
                  Loading{isLoadingTrajectory ? " trajectory" : ""}
                  {isLoadingTrajectory && isLoadingAlerts ? " & " : ""}
                  {isLoadingAlerts ? " alerts" : ""}…
                </span>
              )}
            </div>

            {trajectoryError && (
              <div className="trajectory-error">
                <p>{trajectoryError}</p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={retryTrajectory}
                >
                  Retry
                </button>
              </div>
            )}

            {recallActive &&
              !isLoadingTrajectory &&
              !trajectoryError &&
              trajectory &&
              trajectory.trajectory.length === 0 && (
                <div className="trajectory-notfound">
                  <p>
                    No observation data found for patient{" "}
                    <strong>{activePatientId}</strong>. This patient ID may not
                    exist in the system yet.
                  </p>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={retryTrajectory}
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

            <div className="alert-summary-wrapper">
              <AlertSummary
                alerts={alerts}
                isLoading={isLoadingAlerts}
                error={alertError}
                onRetry={retryAlerts}
              />
            </div>

            <div className="alert-summary-wrapper">
              <PatientHistory
                observations={observations?.observations ?? null}
                isLoading={isLoadingObservations}
                error={observationsError}
                onRetry={retryObservations}
              />
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
