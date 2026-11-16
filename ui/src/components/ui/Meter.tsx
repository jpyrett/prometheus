import type { ReactNode } from "react";

/** One bar of the recommendation panel: a label, a track filled `fill` percent, and the value. */
export function Meter({ label, small, fill, value, tone = "info", dim = false, title }: { label: ReactNode; small?: ReactNode; fill: number; value: ReactNode; tone?: "info" | "critic" | "neg"; dim?: boolean; title?: string }) {
  const clamped = Math.max(0, Math.min(100, fill));
  return (
    <div className={`bar${dim ? " dim" : ""}`} title={title}>
      <span className="k">
        {label} {small !== undefined && <small>{small}</small>}
      </span>
      <span className="track">
        <i className={tone === "critic" ? "c" : tone === "neg" ? "z" : undefined} style={{ width: `${clamped}%` }} />
      </span>
      <span className="v">{value}</span>
    </div>
  );
}
