import {
  LAB_META,
  OVERLAY_LABELS,
  VITAL_GROUPS,
  VITAL_META,
} from "./reference";
import type { LabKey, OverlayKey, VitalKey } from "./reference";

interface MetricSelectorProps {
  vitals: Record<VitalKey, boolean>;
  labs: Record<LabKey, boolean>;
  overlays: Record<OverlayKey, boolean>;
  onToggleVital: (key: VitalKey) => void;
  onToggleLab: (key: LabKey) => void;
  onToggleOverlay: (key: OverlayKey) => void;
}

function Toggle({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: () => void;
}) {
  return (
    <label className="metric-toggle">
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span>{label}</span>
    </label>
  );
}

export function MetricSelector({
  vitals,
  labs,
  overlays,
  onToggleVital,
  onToggleLab,
  onToggleOverlay,
}: MetricSelectorProps) {
  return (
    <div className="metric-selector" role="group" aria-label="Timeline metric selector">
      <div className="metric-selector-row">
        <span className="metric-selector-label">Vitals</span>
        {VITAL_GROUPS.flatMap((group) => group.keys).map((key) => (
          <Toggle
            key={key}
            checked={vitals[key]}
            label={VITAL_META[key].label}
            onChange={() => onToggleVital(key)}
          />
        ))}
      </div>
      <div className="metric-selector-row">
        <span className="metric-selector-label">Labs</span>
        {Object.keys(LAB_META).map((key) => {
          const labKey = key as LabKey;
          return (
            <Toggle
              key={labKey}
              checked={labs[labKey]}
              label={LAB_META[labKey].label}
              onChange={() => onToggleLab(labKey)}
            />
          );
        })}
      </div>
      <div className="metric-selector-row">
        <span className="metric-selector-label">Overlays</span>
        {Object.keys(OVERLAY_LABELS).map((key) => {
          const overlayKey = key as OverlayKey;
          return (
            <Toggle
              key={overlayKey}
              checked={overlays[overlayKey]}
              label={OVERLAY_LABELS[overlayKey]}
              onChange={() => onToggleOverlay(overlayKey)}
            />
          );
        })}
      </div>
    </div>
  );
}