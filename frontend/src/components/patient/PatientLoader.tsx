import { useState } from "react";
import "./PatientLoader.css";

interface PatientLoaderProps {
  onLoad: (patientId: string) => void;
  loading: boolean;
}

export function PatientLoader({ onLoad, loading }: PatientLoaderProps) {
  const [patientId, setPatientId] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = patientId.trim();
    if (!trimmed) {
      setError("Patient ID is required");
      return;
    }
    setError(null);
    onLoad(trimmed);
  };

  return (
    <form className="patient-loader" onSubmit={handleSubmit}>
      <span className="patient-loader-label">Load Patient</span>
      <input
        type="text"
        value={patientId}
        onChange={(e) => {
          setPatientId(e.target.value);
          if (error) setError(null);
        }}
        placeholder="e.g., patient_001"
        className={error ? "error" : ""}
        disabled={loading}
        aria-label="Patient ID"
      />
      <button type="submit" className="btn btn-secondary" disabled={loading || !patientId.trim()}>
        {loading ? "Loading…" : "Load Patient"}
      </button>
      {error && <span className="patient-loader-error">{error}</span>}
    </form>
  );
}
