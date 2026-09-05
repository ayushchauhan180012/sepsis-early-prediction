import { useEffect, useMemo, useRef, useState } from "react";
import type { ObservationRecord, PeakRisk, TrajectoryPoint } from "../../api/types";
import { MetricSelector } from "./MetricSelector";
import { LabsPanel } from "./LabsPanel";
import { RiskPanel } from "./RiskPanel";
import { VitalsPanel } from "./VitalsPanel";
import {
  buildTimelineRows,
  computeXDomain,
  decimateRows,
  extractAlertRuns,
  MAX_RENDER_POINTS,
} from "./timelineData";
import {
  DEFAULT_SELECTION,
  LAB_SPARSITY_NOTE,
  REFERENCE_RANGE_NOTE,
  SYNC_ID,
} from "./reference";
import type { LabKey, OverlayKey, TimelineSelection, VitalKey } from "./reference";
import "./ClinicalTimeline.css";

interface ClinicalTimelineProps {
  activePatientId: string;
  trajectory: TrajectoryPoint[] | null;
  observations: ObservationRecord[] | null;
  peakRisk: PeakRisk | null;
  isLoadingTrajectory: boolean;
  isLoadingObservations: boolean;
  trajectoryError: string | null;
  observationsError: string | null;
  onRetry: () => void;
}

interface RetainedData {
  patientId: string;
  trajectory: TrajectoryPoint[];
  observations: ObservationRecord[];
}

export function ClinicalTimeline({
  activePatientId,
  trajectory,
  observations,
  peakRisk,
  isLoadingTrajectory,
  isLoadingObservations,
  trajectoryError,
  observationsError,
  onRetry,
}: ClinicalTimelineProps) {
  const [selection, setSelection] = useState<TimelineSelection>(DEFAULT_SELECTION);

  const trajectoryPoints = trajectory ?? [];
  const observationRecords = observations ?? [];

  // Retain the last non-empty dataset per patient so live-monitoring refreshes
  // (M10 ticks) extend the timeline instead of flashing an empty state. Keyed by
  // patient so switching patients always shows the fresh target's state.
  const hasFreshData =
    trajectoryPoints.length > 0 || observationRecords.length > 0;
  const retainedRef = useRef<RetainedData | null>(null);

  useEffect(() => {
    if (!hasFreshData) return;
    retainedRef.current = {
      patientId: activePatientId,
      trajectory: trajectoryPoints,
      observations: observationRecords,
    };
  }, [activePatientId, hasFreshData, trajectoryPoints, observationRecords]);

  const effectiveTrajectory = useMemo(() => {
    if (trajectoryPoints.length > 0) return trajectoryPoints;
    const retained = retainedRef.current;
    if (retained && retained.patientId === activePatientId) {
      return retained.trajectory;
    }
    return trajectoryPoints;
  }, [activePatientId, trajectoryPoints]);

  const effectiveObservations = useMemo(() => {
    if (observationRecords.length > 0) return observationRecords;
    const retained = retainedRef.current;
    if (retained && retained.patientId === activePatientId) {
      return retained.observations;
    }
    return observationRecords;
  }, [activePatientId, observationRecords]);

  const rows = useMemo(
    () => buildTimelineRows(effectiveTrajectory, effectiveObservations),
    [effectiveTrajectory, effectiveObservations]
  );

  const alertRuns = useMemo(() => extractAlertRuns(effectiveTrajectory), [effectiveTrajectory]);

  const renderedRows = useMemo(() => decimateRows(rows, MAX_RENDER_POINTS), [rows]);

  const xDomain = useMemo(() => computeXDomain(renderedRows), [renderedRows]);

  const riskChartData = useMemo(
    () =>
      renderedRows.map((row) => ({
        iculos: row.iculos,
        raw: row.raw,
        filtered: row.filtered,
      })),
    [renderedRows]
  );

  const anyLoading = isLoadingTrajectory || isLoadingObservations;
  const hasAnyData = effectiveTrajectory.length > 0 || effectiveObservations.length > 0;
  const hasError = trajectoryError !== null || observationsError !== null;
  const showInitialLoading = anyLoading && !hasAnyData && !hasError;
  const globalEmpty = !hasAnyData && !showInitialLoading && !hasError;

  const toggleVital = (key: VitalKey) =>
    setSelection((prev) => ({
      ...prev,
      vitals: { ...prev.vitals, [key]: !prev.vitals[key] },
    }));

  const toggleLab = (key: LabKey) =>
    setSelection((prev) => ({
      ...prev,
      labs: { ...prev.labs, [key]: !prev.labs[key] },
    }));

  const toggleOverlay = (key: OverlayKey) =>
    setSelection((prev) => ({
      ...prev,
      overlays: { ...prev.overlays, [key]: !prev.overlays[key] },
    }));

  return (
    <section className="clinical-timeline">
      <div className="clinical-timeline-header">
        <h2 className="clinical-timeline-heading">Clinical Timeline</h2>
        <p className="clinical-timeline-subtitle">
          Risk, vital, and lab trends aligned by ICU hour (ICULOS). Hover any panel
          to inspect the same hour across all panels.
        </p>
      </div>

      <MetricSelector
        vitals={selection.vitals}
        labs={selection.labs}
        overlays={selection.overlays}
        onToggleVital={toggleVital}
        onToggleLab={toggleLab}
        onToggleOverlay={toggleOverlay}
      />

      {showInitialLoading && (
        <div className="timeline-loading">Loading clinical timeline…</div>
      )}

      {hasError && (
        <div className="timeline-errors">
          {trajectoryError && (
            <div className="timeline-error-banner">
              <p>{trajectoryError}</p>
              <button type="button" className="btn btn-secondary" onClick={onRetry}>
                Retry
              </button>
            </div>
          )}
          {observationsError && (
            <div className="timeline-error-banner">
              <p>{observationsError}</p>
              <button type="button" className="btn btn-secondary" onClick={onRetry}>
                Retry
              </button>
            </div>
          )}
        </div>
      )}

      {globalEmpty ? (
        <div className="clinical-timeline-empty">
          <p>No clinical data recorded for this patient yet.</p>
          <p className="timeline-empty-hint-sub">
            Submit an observation to build the patient's risk and observation
            timeline.
          </p>
        </div>
      ) : (
        !showInitialLoading && (
          <>
            <RiskPanel
              data={riskChartData}
              alertRuns={alertRuns}
              peakRisk={peakRisk}
              showRawRisk={selection.overlays.rawRisk}
              showUncertaintyBand={selection.overlays.uncertaintyBand}
              showAlertBands={selection.overlays.alertBands}
              syncId={SYNC_ID}
            />
            <VitalsPanel
              rows={renderedRows}
              vitals={selection.vitals}
              xDomain={xDomain}
              syncId={SYNC_ID}
            />
            <LabsPanel
              rows={renderedRows}
              labs={selection.labs}
              xDomain={xDomain}
              syncId={SYNC_ID}
            />
          </>
        )
      )}

      <p className="clinical-timeline-notes">
        X axis: ICULOS (ICU hour), shared across all panels. {REFERENCE_RANGE_NOTE}{" "}
        {LAB_SPARSITY_NOTE}
      </p>
    </section>
  );
}