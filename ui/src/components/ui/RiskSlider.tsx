import type { RiskLevel } from "../../api/types";

const LEVELS: { value: RiskLevel; label: string }[] = [
  { value: "conservative", label: "Conservative" },
  { value: "moderate", label: "Moderate" },
  { value: "aggressive", label: "Aggressive" },
];

/** Three stops on a green-to-red track; the chosen word is bold. `onCommit` fires on release (pointer up, key up, blur), not on every step of a drag. */
export function RiskSlider({ id, value, onChange, onCommit, disabled = false }: { id: string; value: RiskLevel; onChange: (value: RiskLevel) => void; onCommit: (value: RiskLevel) => void; disabled?: boolean }) {
  const index = Math.max(0, LEVELS.findIndex((l) => l.value === value));
  const commit = (target: HTMLInputElement) => onCommit(LEVELS[Number(target.value)]?.value ?? value);
  return (
    <div className="risk">
      <input
        id={id}
        type="range"
        min={0}
        max={2}
        step={1}
        value={index}
        disabled={disabled}
        aria-valuetext={LEVELS[index]?.label}
        onChange={(e) => onChange(LEVELS[Number(e.target.value)]?.value ?? value)}
        onPointerUp={(e) => commit(e.currentTarget)}
        onKeyUp={(e) => commit(e.currentTarget)}
        onBlur={(e) => commit(e.currentTarget)}
      />
      <div className="ticks" aria-hidden="true">
        {LEVELS.map((l) => (
          <span key={l.value} className={l.value === value ? "on" : undefined}>
            {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
