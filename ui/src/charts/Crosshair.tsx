// The crosshair of a line chart: a vertical hairline that snaps to the nearest data position, and the control that moves it.
// The control is an HTML slider laid over the plot (pointer move and arrow keys), so a keyboard reader gets the same readout (`aria-valuetext`) that the pointer's tooltip shows.
import { useCallback, useState, type KeyboardEvent, type PointerEvent } from "react";
import { nearestIndex } from "./scale";

export interface ScrubOptions {
  /** The pixel x of each data position, ascending, in the plot's own coordinates (the chart's, not the slider's). */
  xs: number[];
  /** Where the slider's left edge is in those coordinates. */
  left: number;
  /** Where its top edge is (only a custom `nearest` uses it). */
  top?: number;
  /** Choose the position by the pointer's place (a scatter's nearest dot) instead of its x alone. */
  nearest?: (x: number, y: number) => number;
  /** What the slider says for a position ("Aug 10: Desk $10,412"). */
  describe: (index: number) => string;
  label: string;
}

export function useScrub({ xs, left, top = 0, nearest, describe, label }: ScrubOptions) {
  const [index, setIndex] = useState<number | null>(null);
  const count = xs.length;
  const clampTo = useCallback((i: number) => Math.max(0, Math.min(count - 1, i)), [count]);

  const bind = {
    role: "slider" as const,
    tabIndex: 0,
    "aria-label": label,
    "aria-orientation": "horizontal" as const,
    "aria-valuemin": 0,
    "aria-valuemax": Math.max(0, count - 1),
    "aria-valuenow": index ?? 0,
    "aria-valuetext": index === null ? "no day chosen" : describe(index),
    onPointerMove: (e: PointerEvent<HTMLElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const i = nearest ? nearest(e.clientX - rect.left + left, e.clientY - rect.top + top) : nearestIndex(xs, e.clientX - rect.left + left);
      if (i >= 0) setIndex(i);
    },
    onPointerLeave: () => setIndex(null),
    onFocus: () => setIndex((i) => i ?? count - 1),
    onBlur: () => setIndex(null),
    onKeyDown: (e: KeyboardEvent<HTMLElement>) => {
      const at = index ?? count - 1;
      const to = e.key === "ArrowLeft" ? at - 1 : e.key === "ArrowRight" ? at + 1 : e.key === "Home" ? 0 : e.key === "End" ? count - 1 : e.key === "PageUp" ? at - 10 : e.key === "PageDown" ? at + 10 : null;
      if (to === null) return;
      e.preventDefault();
      setIndex(clampTo(to));
    },
  };
  return { index: count === 0 ? null : index === null ? null : clampTo(index), bind };
}

/** The vertical hairline and one dot per series at a position. */
export function Crosshair({ x, top, bottom, dots }: { x: number; top: number; bottom: number; dots?: { y: number; color: string }[] }) {
  return (
    <g className="crosshair" aria-hidden="true">
      <line className="cross-line" x1={x} x2={x} y1={top} y2={bottom} />
      {dots?.map((d, i) => (
        <circle key={i} className="cross-dot" cx={x} cy={d.y} r={4} fill={d.color} />
      ))}
    </g>
  );
}
