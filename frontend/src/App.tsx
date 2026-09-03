import { Routes, Route } from "react-router-dom";

function Dashboard() {
  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Early Sepsis Alert System</h1>
        <p className="app-subtitle">
          6-hour ahead sepsis risk prediction for ICU patients
        </p>
      </header>
      <section className="app-content">
        <div className="placeholder-card">
          <p>Dashboard will be implemented in a future milestone.</p>
        </div>
      </section>
    </main>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
    </Routes>
  );
}
