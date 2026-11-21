// Axes of the Proving Ground's charts: hairline grid, tick labels in muted text ink (never the series' colour). Layout only.
import { useId } from "react";
import type { Scale, TimeTick } from "./scale";

/** An id that is safe inside url(#...) (useId's colons are not). */
export function useSvgId(prefix = "c"): string {
  return `${prefix}${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
}

export interface YAxisProps {
  scale: Scale;
  ticks: number[];
  /** The left edge of the plot, in pixels: the labels end 8 px before it. */
  x: number;
  /** The right edge: the grid runs to it. */
  right: number;
  format: (value: number) => string;
  grid?: boolean;
  /** A tick that draws a stronger line (zero). */
  emphasize?: number;
}

export function YAxis({ scale, ticks, x, right, format, grid = true, emphasize }: YAxisProps) {
  return (
    <g className="ax ax-y" aria-hidden="true">
      {ticks.map((t) => {
        const y = scale(t);
        return (
          <g key={t}>
            {grid && <line className={emphasize === t ? "grid strong" : "grid"} x1={x} x2={right} y1={y} y2={y} />}
            <text className="tick-label" x={x - 8} y={y} dy="0.32em" textAnchor="end">
              {format(t)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export interface XAxisProps {
  scale: Scale;
  ticks: TimeTick[];
  y: number;
  /** When set, the ticks also draw a faint vertical grid line up to this y. */
  gridTo?: number;
}

export function XAxis({ scale, ticks, y, gridTo }: XAxisProps) {
  return (
    <g className="ax ax-x" aria-hidden="true">
      {ticks.map((t) => {
        const x = scale(t.day);
        return (
          <g key={t.day}>
            {gridTo !== undefined && <line className="grid vert" x1={x} x2={x} y1={gridTo} y2={y} />}
            <line className="tick-line" x1={x} x2={x} y1={y} y2={y + 4} />
            {t.label && (
              <text className={t.major ? "tick-label major" : "tick-label"} x={x} y={y + 16} textAnchor="middle">
                {t.label}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}
