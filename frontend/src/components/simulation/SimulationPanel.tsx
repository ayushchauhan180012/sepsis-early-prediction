import { useCallback, useEffect, useRef, useState } from "react";
import { submitObservation } from "../../api/predict";
import { ApiError } from "../../api/types";
import type { Observation, PredictionResponse } from "../../api/types";
import {
  PatientSimulator,
  SCENARIO_IDS,
  SCENARIO_LABELS,
} from "../../sim/scenarios";
import type { ScenarioId } from "../../sim/scenarios";
import "./SimulationPanel.css";

interface SimulationPanelProps {
  activePatientId: string | null;
  nextIculos: number;
  onPrefill: (obs: Observation) => void;
  onPrediction: (response: PredictionResponse) => void;
}

const INTERVAL_OPTIONS = [
  { value: 5000, label: "5s per hour" },
  { value: 10000, label: "10s per hour" },
  { value: 15000, label: "15s per hour" },
  { value: 30000, label: "30s per hour" },
  { value: 60000, label: "60s per hour" },
];

const DEFAULT_INTERVAL_MS = 10000;
const DEFAULT_AGE = 60;

function translateSimError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 409) {
      const detail = (err.body as { detail?: string } | null)?.detail;
      return detail
        ? `Simulation stopped: ${detail}`
        : "Simulation stopped: ICULOS conflict — the hour already exists or is out of order for this patient.";
    }
    if (err.status === 422) {
      return "Simulation stopped: the generated observation failed validation.";
    }
    if (err.status >= 500) {
      return "Simulation stopped: server error while submitting the simulated observation.";
    }
    return `Simulation stopped: request failed (${err.status}).`;
  }
  if (err instanceof TypeError) {
    return "Simulation stopped: network error. Is the backend running?";
  }
  return "Simulation stopped: an unexpected error occurred.";
}

export function SimulationPanel({
  activePatientId,
  nextIculos,
  onPrefill,
  onPrediction,
}: SimulationPanelProps) {
  const [patientId, setPatientId] = useState("");
  const [scenario, setScenario] = useState<ScenarioId>("sepsis");
  const [age, setAge] = useState(DEFAULT_AGE);
  const [intervalMs, setIntervalMs] = useState(DEFAULT_INTERVAL_MS);
  const [monitoring, setMonitoring] = useState(false);
  const [inFlight, setInFlight] = useState(false);
  const [currentIculos, setCurrentIculos] = useState<number | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const enabledRef = useRef(false);
  const inFlightRef = useRef(false);
  const runIdRef = useRef(0);
  const simRef = useRef<PatientSimulator | null>(null);
  const timerRef = useRef<number | null>(null);
  const monitoringPatientRef = useRef<string | null>(null);
  const intervalMsRef = useRef(intervalMs);
  const onPredictionRef = useRef(onPrediction);
  const tickRef = useRef<(runId: number) => void>(() => {});

  intervalMsRef.current = intervalMs;
  onPredictionRef.current = onPrediction;

  const stopMonitoring = useCallback(() => {
    runIdRef.current += 1;
    enabledRef.current = false;
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    inFlightRef.current = false;
    monitoringPatientRef.current = null;
    setInFlight(false);
    setMonitoring(false);
    simRef.current = null;
  }, []);

  const scheduleNext = useCallback(() => {
    if (!enabledRef.current) return;
    timerRef.current = window.setTimeout(() => {
      void tickRef.current(runIdRef.current);
    }, intervalMsRef.current);
  }, []);

  const tick = useCallback(
    async (runId: number) => {
      if (!enabledRef.current || inFlightRef.current || runId !== runIdRef.current) {
        return;
      }
      const sim = simRef.current;
      if (!sim) return;

      inFlightRef.current = true;
      setInFlight(true);

      const obs = sim.generateNext();

      try {
        const response = await submitObservation(obs);
        if (!enabledRef.current || runId !== runIdRef.current) return;
        setError(null);
        setInfo(null);
        setCurrentIculos(obs.ICULOS);
        onPredictionRef.current(response);
        scheduleNext();
      } catch (err) {
        if (!enabledRef.current || runId !== runIdRef.current) return;
        enabledRef.current = false;
        if (timerRef.current !== null) {
          window.clearTimeout(timerRef.current);
          timerRef.current = null;
        }
        monitoringPatientRef.current = null;
        setMonitoring(false);
        simRef.current = null;
        setError(translateSimError(err));
      } finally {
        if (runId === runIdRef.current) {
          inFlightRef.current = false;
          setInFlight(false);
        }
      }
    },
    [scheduleNext]
  );

  tickRef.current = tick;

  const startMonitoring = useCallback(() => {
    stopMonitoring();
    const runId = runIdRef.current;
    const trimmedPatientId = patientId.trim();
    if (!trimmedPatientId) {
      setError("Enter a patient ID to start monitoring.");
      return;
    }
    const startIculos =
      trimmedPatientId === (activePatientId ?? "") ? nextIculos : 1;
    simRef.current = new PatientSimulator({
      patientId: trimmedPatientId,
      age,
      scenario,
      startIculos,
    });
    monitoringPatientRef.current = trimmedPatientId;
    enabledRef.current = true;
    setError(null);
    setInfo(null);
    setCurrentIculos(null);
    setMonitoring(true);
    void tick(runId);
  }, [stopMonitoring, patientId, activePatientId, nextIculos, age, scenario, tick]);

  const handleSimulateHour = useCallback(() => {
    const trimmedPatientId = patientId.trim();
    if (!trimmedPatientId) {
      setError("Enter a patient ID to generate an observation.");
      return;
    }
    const startIculos =
      trimmedPatientId === (activePatientId ?? "") ? nextIculos : 1;
    const sim = new PatientSimulator({
      patientId: trimmedPatientId,
      age,
      scenario,
      startIculos,
    });
    const obs = sim.generateNext();
    setError(null);
    setInfo(
      `Observation for ICULOS ${obs.ICULOS} generated for ${obs.PatientID}. Review, edit, and submit manually.`
    );
    onPrefill(obs);
  }, [patientId, activePatientId, nextIculos, age, scenario, onPrefill]);

  useEffect(() => {
    const nextPatientId = activePatientId ?? "";
    const monitoringTarget = monitoringPatientRef.current;
    if (monitoringTarget !== null && nextPatientId !== "" && nextPatientId !== monitoringTarget) {
      stopMonitoring();
    }
    setPatientId(nextPatientId);
    setInfo(null);
  }, [activePatientId, stopMonitoring]);

  useEffect(() => {
    return () => {
      runIdRef.current += 1;
      enabledRef.current = false;
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []);

  const trimmedPatientId = patientId.trim();
  const matchesActive =
    trimmedPatientId !== "" && trimmedPatientId === (activePatientId ?? "");
  const resolvedStartIculos = matchesActive ? nextIculos : 1;

  return (
    <section className="simulation-panel">
      <h2>Simulation</h2>

      <div className="sim-controls">
        <label className="sim-field">
          <span className="sim-label">Patient ID</span>
          <input
            type="text"
            value={patientId}
            onChange={(e) => {
              setPatientId(e.target.value);
              if (error) setError(null);
            }}
            placeholder="e.g., patient_001"
            disabled={monitoring}
            aria-label="Simulation patient ID"
          />
        </label>

        <label className="sim-field">
          <span className="sim-label">Scenario</span>
          <select
            value={scenario}
            onChange={(e) => {
              const next = e.target.value as ScenarioId;
              if (next === scenario) return;
              stopMonitoring();
              setScenario(next);
            }}
            aria-label="Simulation scenario"
          >
            {SCENARIO_IDS.map((id) => (
              <option key={id} value={id}>
                {SCENARIO_LABELS[id]}
              </option>
            ))}
          </select>
        </label>

        <label className="sim-field">
          <span className="sim-label">Age</span>
          <input
            type="number"
            value={age}
            min={0}
            max={120}
            step={1}
            disabled={monitoring}
            onChange={(e) => {
              const value = Number(e.target.value);
              if (!Number.isNaN(value)) setAge(value);
            }}
            aria-label="Simulation patient age"
          />
        </label>

        <label className="sim-field">
          <span className="sim-label">Interval</span>
          <select
            value={intervalMs}
            onChange={(e) => setIntervalMs(Number(e.target.value))}
            disabled={monitoring}
            aria-label="Simulation interval"
          >
            {INTERVAL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="sim-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleSimulateHour}
          disabled={monitoring || inFlight}
        >
          Simulate Hour
        </button>
        {monitoring ? (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={stopMonitoring}
            disabled={inFlight}
          >
            Stop Monitoring
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={startMonitoring}
          >
            Start Monitoring
          </button>
        )}
      </div>

      <div className="sim-status">
        <span className="sim-status-item">
          <span className="sim-status-label">Status:</span>
          {monitoring ? (
            <span className="sim-status-live">
              <span className="sim-status-dot" aria-hidden="true" />
              Monitoring
            </span>
          ) : (
            <span className="sim-status-idle">Stopped</span>
          )}
        </span>
        <span className="sim-status-item">
          <span className="sim-status-label">Scenario:</span>
          {SCENARIO_LABELS[scenario]}
        </span>
        <span className="sim-status-item">
          <span className="sim-status-label">
            {monitoring ? "Current ICULOS:" : "Next ICULOS:"}
          </span>
          {monitoring ? currentIculos ?? "—" : resolvedStartIculos}
        </span>
        {inFlight && (
          <span className="sim-status-item sim-status-inflight">
            Submitting hour…
          </span>
        )}
      </div>

      {info && <div className="sim-info">{info}</div>}
      {error && <div className="sim-error">{error}</div>}
    </section>
  );
}