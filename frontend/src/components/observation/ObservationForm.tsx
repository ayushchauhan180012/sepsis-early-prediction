import { forwardRef, useImperativeHandle, useState } from "react";
import { submitObservation } from "../../api/predict";
import type { Observation, PredictionResponse, ApiError } from "../../api/types";
import "./ObservationForm.css";

interface FormErrors {
  [key: string]: string;
}

export interface ObservationFormHandle {
  prefill: (obs: Observation) => void;
}

interface ObservationFormProps {
  onPrediction: (response: PredictionResponse) => void;
}

export const ObservationForm = forwardRef<ObservationFormHandle, ObservationFormProps>(
  function ObservationForm({ onPrediction }, ref) {
  const [formData, setFormData] = useState<Observation>({
    PatientID: "",
    Age: 0,
    ICULOS: 1,
    HR: 0,
    O2Sat: 0,
    SBP: 0,
    MAP: 0,
    Resp: 0,
    Temp: 0,
    Lactate: null,
    WBC: null,
    Creatinine: null,
    Platelets: null,
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const validateField = (name: string, value: string | number | null): string | null => {
    const numValue = value === "" || value === null ? null : Number(value);

    switch (name) {
      case "PatientID":
        if (!value || (value as string).trim() === "") {
          return "Patient ID is required";
        }
        break;
      case "Age":
        if (numValue === null || isNaN(numValue)) return "Age is required";
        if (numValue < 0 || numValue > 120) return "Age must be between 0 and 120";
        break;
      case "ICULOS":
        if (numValue === null || isNaN(numValue)) return "ICULOS is required";
        if (numValue < 1) return "ICULOS must be >= 1";
        break;
      case "HR":
        if (numValue === null || isNaN(numValue)) return "Heart Rate is required";
        if (numValue < 20 || numValue > 250) return "HR must be between 20 and 250";
        if (numValue === 0) return "Heart Rate cannot be zero";
        break;
      case "O2Sat":
        if (numValue === null || isNaN(numValue)) return "O2Sat is required";
        if (numValue < 0 || numValue > 100) return "O2Sat must be between 0 and 100";
        break;
      case "SBP":
        if (numValue === null || isNaN(numValue)) return "SBP is required";
        if (numValue < 40 || numValue > 300) return "SBP must be between 40 and 300";
        if (numValue === 0) return "SBP cannot be zero";
        break;
      case "MAP":
        if (numValue === null || isNaN(numValue)) return "MAP is required";
        if (numValue < 20 || numValue > 250) return "MAP must be between 20 and 250";
        if (numValue === 0) return "MAP cannot be zero";
        break;
      case "Resp":
        if (numValue === null || isNaN(numValue)) return "Respiration Rate is required";
        if (numValue < 0 || numValue > 80) return "Resp must be between 0 and 80";
        if (numValue === 0) return "Respiration Rate cannot be zero";
        break;
      case "Temp":
        if (numValue === null || isNaN(numValue)) return "Temperature is required";
        if (numValue < 25 || numValue > 45) return "Temp must be between 25 and 45";
        if (numValue < 30 || numValue > 43) return "Temperature outside realistic human range (30–43)";
        break;
      case "Lactate":
      case "WBC":
      case "Creatinine":
      case "Platelets":
        if (value !== "" && value !== null && (numValue === null || isNaN(numValue) || numValue < 0)) {
          return "Must be a non-negative number";
        }
        break;
    }
    return null;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: string | number | null = value;

    if (type === "number") {
      parsedValue = value === "" ? null : Number(value);
    }

    if (name === "PatientID") {
      // Changing the patient breaks the previous patient's continue-state:
      // a new patient starts a fresh ICULOS sequence (hour 1).
      setFormData((prev) => ({
        ...prev,
        PatientID: String(value),
        ICULOS: 1,
      }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: parsedValue }));
    }

    const error = validateField(name, parsedValue);
    setErrors((prev) => ({
      ...prev,
      [name]: error || "",
    }));
  };

  const validateAll = (): boolean => {
    const newErrors: FormErrors = {};
    let hasErrors = false;

    (Object.keys(formData) as Array<keyof Observation>).forEach((key) => {
      const error = validateField(key, formData[key]);
      if (error) {
        newErrors[key] = error;
        hasErrors = true;
      }
    });

    setErrors(newErrors);
    return !hasErrors;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);

    if (!validateAll()) {
      return;
    }

    setIsSubmitting(true);

    try {
      const observation: Observation = {
        PatientID: formData.PatientID.trim(),
        Age: Number(formData.Age),
        ICULOS: Number(formData.ICULOS),
        HR: Number(formData.HR),
        O2Sat: Number(formData.O2Sat),
        SBP: Number(formData.SBP),
        MAP: Number(formData.MAP),
        Resp: Number(formData.Resp),
        Temp: Number(formData.Temp),
        Lactate: formData.Lactate === null ? null : Number(formData.Lactate),
        WBC: formData.WBC === null ? null : Number(formData.WBC),
        Creatinine: formData.Creatinine === null ? null : Number(formData.Creatinine),
        Platelets: formData.Platelets === null ? null : Number(formData.Platelets),
      };

      const response = await submitObservation(observation);
      onPrediction(response);
      // Continue-patient: carry the actual submitted inputs forward and
      // advance to the next ICU hour. This is a pre-fill only — no automatic
      // submission happens here.
      setFormData({
        ...observation,
        ICULOS: Number(observation.ICULOS) + 1,
      });
      setErrors({});
      setSubmitError(null);
    } catch (err) {
      if (err instanceof Error && "status" in err) {
        const apiError = err as ApiError;
        if (apiError.status === 409) {
          const detail = apiError.body as { detail?: string } | null;
          setSubmitError(detail?.detail || "ICULOS conflict: the observation hour already exists or is out of order for this patient.");
        } else if (apiError.status === 422) {
          const detail = apiError.body as { detail?: string } | null;
          setSubmitError(detail?.detail || "Invalid observation data. Check the fields and try again.");
        } else if (apiError.status >= 500) {
          setSubmitError("Server error. Please try again later.");
        } else {
          setSubmitError(`Request failed (${apiError.status}). Please check your input.`);
        }
      } else if (err instanceof TypeError) {
        setSubmitError("Network error. Is the backend running?");
      } else {
        setSubmitError("An unexpected error occurred.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  useImperativeHandle(
    ref,
    () => ({
      prefill: (obs: Observation) => {
        setFormData(obs);
        setErrors({});
        setSubmitError(null);
      },
    }),
    []
  );

  const renderField = (
    label: string,
    name: keyof Observation,
    type: "text" | "number" = "number",
    min?: number,
    max?: number,
    step?: number,
    placeholder?: string
  ) => {
    const error = errors[name];
    const value = formData[name];
    const displayValue = value === null || value === "" ? "" : value;

    return (
      <div className="form-field">
        <label htmlFor={name}>{label}</label>
        <input
          id={name}
          name={name}
          type={type}
          value={displayValue}
          onChange={handleChange}
          min={min}
          max={max}
          step={step}
          placeholder={placeholder}
          className={error ? "error" : ""}
          disabled={isSubmitting}
          required={!["Lactate", "WBC", "Creatinine", "Platelets"].includes(name)}
        />
        {error && <span className="field-error">{error}</span>}
      </div>
    );
  };

  return (
    <form className="observation-form" onSubmit={handleSubmit}>
      <section className="form-section">
        <h2>Patient Information</h2>
        <div className="form-row">
          {renderField("Patient ID", "PatientID", "text", undefined, undefined, undefined, "e.g., patient_001")}
          {renderField("Age", "Age", "number", 0, 120, 1)}
          {renderField("ICULOS (ICU Hour)", "ICULOS", "number", 1, undefined, 1)}
        </div>
      </section>

      <section className="form-section">
        <h2>Vital Signs</h2>
        <div className="form-row">
          {renderField("Heart Rate (HR)", "HR", "number", 20, 250, 1)}
          {renderField("O₂ Saturation (O2Sat)", "O2Sat", "number", 0, 100, 1)}
          {renderField("Systolic BP (SBP)", "SBP", "number", 40, 300, 1)}
          {renderField("Mean Arterial Pressure (MAP)", "MAP", "number", 20, 250, 1)}
          {renderField("Respiration Rate (Resp)", "Resp", "number", 0, 80, 1)}
          {renderField("Temperature (Temp, °C)", "Temp", "number", 25, 45, 0.1)}
        </div>
      </section>

      <section className="form-section">
        <h2>Laboratory Values (Optional)</h2>
        <p className="section-hint">Leave blank if not yet tested. Enter numeric values when available.</p>
        <div className="form-row">
          {renderField("Lactate (mmol/L)", "Lactate", "number", 0, undefined, 0.1, "Optional")}
          {renderField("WBC (×10³/µL)", "WBC", "number", 0, undefined, 0.1, "Optional")}
          {renderField("Creatinine (mg/dL)", "Creatinine", "number", 0, undefined, 0.01, "Optional")}
          {renderField("Platelets (×10³/µL)", "Platelets", "number", 0, undefined, 1, "Optional")}
        </div>
      </section>

      {submitError && <div className="submit-error">{submitError}</div>}

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
          {isSubmitting ? "Submitting…" : "Submit Observation"}
        </button>
      </div>
    </form>
  );
  });