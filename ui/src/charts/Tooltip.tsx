// The tooltip of a chart: one box with the values that matter, values first and names second (dataviz), placed beside the pointer and flipped
// at the right edge. It only ever ENHANCES: every value in it is also in the chart's table twin, and the keyboard slider reads the same text. Text is inserted as text, never as HTML.
import type { ReactNode } from "react";

export interface TooltipProps {
  /** The anchor in the plot's own pixels. */
  x: number;
  y: number;
  /** The plot's width, to flip the box to the left of the anchor near the right edge. */
  plotWidth: number;
  title?: ReactNode;
  /** "side" (the default): beside the anchor, flipped at the right edge. "below": centred under the anchor (a tile of the research strip). */
  placement?: "side" | "below";
  children: ReactNode;
}

export function Tooltip({ x, y, plotWidth, title, placement = "side", children }: TooltipProps) {
  const flip = x > plotWidth * 0.58;
  return (
    <div className={`chart-tip ${placement === "below" ? "below" : flip ? "left" : "right"}`} style={{ left: x, top: Math.max(4, y) }} aria-hidden="true">
      {title !== undefined && <div className="tip-title">{title}</div>}
      {children}
    </div>
  );
}

/** One row of a tooltip: a short key in the series' colour, the value (strong) and the name (quiet). */
export function TipRow({ color, value, name, dashed = false }: { color?: string; value: ReactNode; name: ReactNode; dashed?: boolean }) {
  return (
    <div className="tip-row">
      <span className="tip-key" style={color ? { background: dashed ? "transparent" : color, borderTop: dashed ? `2px dashed ${color}` : undefined } : { visibility: "hidden" }} />
      <b className="num">{value}</b>
      <span className="muted">{name}</span>
    </div>
  );
}
