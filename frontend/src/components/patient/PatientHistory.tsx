import type { ObservationRecord } from "../../api/types";
import "./PatientHistory.css";

interface PatientHistoryProps {
  observations: ObservationRecord[] | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
}

const formatValue = (value: number | null): string =>
  value === null ? "—" : String(value);

const formatTime = (iso: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString();
};

export function PatientHistory({
  observations,
  isLoading,
  error,
  onRetry,
}: PatientHistoryProps) {
  if (isLoading) {
    return (
      <div className="patient-history">
        <h2>Observation History</h2>
        <div className="patient-history-loading">Loading observation history…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="patient-history">
        <h2>Observation History</h2>
        <div className="patient-history-error">
          <p>{error}</p>
          <button type="button" className="btn btn-secondary" onClick={onRetry}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!observations || observations.length === 0) {
    return (
      <div className="patient-history">
        <h2>Observation History</h2>
        <div className="patient-history-empty">
          <p>No observations recorded for this patient yet.</p>
          <p className="patient-history-empty-hint">
            Submit an observation to begin building the patient's history.
          </p>
        </div>
      </div>
    );
  }

  const latest = observations[observations.length - 1]!;

  return (
    <div className="patient-history">
      <h2>Observation History</h2>

      <section className="patient-history-latest">
        <h3>Latest Observation</h3>
        <div className="patient-history-latest-grid">
          <div className="patient-history-item">
            <span className="patient-history-label">ICULOS</span>
            <span className="patient-history-value">{latest.iculos}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">HR</span>
            <span className="patient-history-value">{formatValue(latest.hr)}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">O2Sat</span>
            <span className="patient-history-value">{formatValue(latest.o2sat)}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">SBP</span>
            <span className="patient-history-value">{formatValue(latest.sbp)}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">MAP</span>
            <span className="patient-history-value">{formatValue(latest.map)}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">Resp</span>
            <span className="patient-history-value">{formatValue(latest.resp)}</span>
          </div>
          <div className="patient-history-item">
            <span className="patient-history-label">Temp</span>
            <span className="patient-history-value">{formatValue(latest.temp)}</span>
          </div>
        </div>
      </section>

      <section className="patient-history-table-section">
        <h3>All Observations</h3>
        <div className="patient-history-table-scroll">
          <table className="patient-history-table">
            <thead>
              <tr>
                <th>ICULOS</th>
                <th>HR</th>
                <th>O2Sat</th>
                <th>SBP</th>
                <th>MAP</th>
                <th>Resp</th>
                <th>Temp</th>
                <th>Lactate</th>
                <th>WBC</th>
                <th>Creatinine</th>
                <th>Platelets</th>
                <th>Received</th>
              </tr>
            </thead>
            <tbody>
              {observations.map((obs) => (
                <tr key={obs.iculos}>
                  <td className="patient-history-cell-iculos">{obs.iculos}</td>
                  <td>{formatValue(obs.hr)}</td>
                  <td>{formatValue(obs.o2sat)}</td>
                  <td>{formatValue(obs.sbp)}</td>
                  <td>{formatValue(obs.map)}</td>
                  <td>{formatValue(obs.resp)}</td>
                  <td>{formatValue(obs.temp)}</td>
                  <td>{formatValue(obs.lactate)}</td>
                  <td>{formatValue(obs.wbc)}</td>
                  <td>{formatValue(obs.creatinine)}</td>
                  <td>{formatValue(obs.platelets)}</td>
                  <td className="patient-history-cell-time">{formatTime(obs.received_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
