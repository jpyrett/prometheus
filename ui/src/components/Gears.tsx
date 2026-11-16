// Two gears turning against each other: the indeterminate indicator of a run (spec 5.3 (3)). Still when idle or finished.
export function Gears({ spinning, className = "gears" }: { spinning: boolean; className?: string }) {
  return (
    <svg className={`${className}${spinning ? " on" : ""}`} viewBox="0 0 46 40" fill="none" aria-hidden="true">
      <g className="g1">
        <circle cx="15" cy="23" r="9.5" stroke="currentColor" strokeWidth="5" strokeDasharray="3.73 3.73" />
        <circle cx="15" cy="23" r="8" stroke="currentColor" strokeWidth="3" />
        <circle cx="15" cy="23" r="2.4" fill="currentColor" />
      </g>
      <g className="g2">
        <circle cx="33" cy="14" r="6.6" stroke="currentColor" strokeWidth="4" strokeDasharray="3.46 3.46" />
        <circle cx="33" cy="14" r="5.4" stroke="currentColor" strokeWidth="2.4" />
        <circle cx="33" cy="14" r="1.7" fill="currentColor" />
      </g>
    </svg>
  );
}

/** The small single gear of the "Running…" button. */
export function SmallGear() {
  return (
    <svg className="small-gear" width="16" height="16" viewBox="0 0 46 40" fill="none" aria-hidden="true">
      <circle cx="23" cy="20" r="13" stroke="currentColor" strokeWidth="8" strokeDasharray="5.1 5.1" />
      <circle cx="23" cy="20" r="11" stroke="currentColor" strokeWidth="5" />
    </svg>
  );
}
