// A small group of toggle chips (a filter): one is on, the others are buttons with aria-pressed. Used by the outcome wall and the cadence control.
import type { ReactNode } from "react";

export interface ChipOption<T extends string> {
  value: T;
  label: ReactNode;
  count?: number;
  title?: string;
}

export function ChipGroup<T extends string>({ label, options, value, onChange, disabled = false, className = "" }: { label: string; options: ChipOption<T>[]; value: T; onChange: (v: T) => void; disabled?: boolean; className?: string }) {
  return (
    <div className={`chips ${className}`.trim()} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" className={`chip${o.value === value ? " on" : ""}`} aria-pressed={o.value === value} disabled={disabled} title={o.title} onClick={() => onChange(o.value)}>
          {o.label}
          {o.count !== undefined && <span className="chip-n num">{o.count}</span>}
        </button>
      ))}
    </div>
  );
}
