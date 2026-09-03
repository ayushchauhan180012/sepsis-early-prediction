import { useState } from "react";
import { ObservationForm } from "./components/observation/ObservationForm";
import { PredictionResult } from "./components/observation/PredictionResult";
import type { PredictionResponse } from "./api/types";

function Dashboard() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

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
          <ObservationForm onPrediction={setPrediction} />
        </section>
        <PredictionResult prediction={prediction} />
      </section>
    </main>
  );
}

export default function App() {
  return <Dashboard />;
}
