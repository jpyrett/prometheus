// The frame every chart of the Proving Ground sits in (spec 13): a legend row, the plot, and a "Show as table" twin.
// The data layer is an <svg role="img" aria-label="..."> carrying the chart's headline numbers; whatever can be operated (a crosshair slider, a range handle) is a SEPARATE element on top of it,
// so the picture is read as one image and the controls stay reachable. The table twin is the same data as a `.tbl` (and the tests' handle).
import { useId, useState, type ReactNode } from "react";
import { useWidth } from "./useSize";

export type Cell = string | number | ReactNode;

export interface TableSpec {
  caption: string;
  columns: string[];
  rows: Cell[][];
  /** Columns that hold numbers (right aligned, monospace). */
  numeric?: number[];
}

export interface FrameDrawing {
  /** The picture's own content (inside the <svg>). */
  svg: ReactNode;
  /** Elements over the plot (the crosshair's slider, the handles, the tooltip): HTML, positioned in the plot's own pixels. */
  overlay?: ReactNode;
}

export interface ChartFrameProps {
  /** The accessible name of the picture: the chart's headline numbers in words. */
  label: string;
  table: TableSpec;
  height: number | ((width: number) => number);
  draw: (width: number, height: number) => FrameDrawing;
  /** A legend (>= 2 series always has one). */
  legend?: ReactNode;
  className?: string;
  /** Extra controls on the legend row (a filter, a skip button). */
  toolbar?: ReactNode;
  /** The width used where the browser has no layout (tests). */
  fallbackWidth?: number;
  /** A line above the plot (what is going on, in words). */
  status?: ReactNode;
  /** A small chart inside a card (the outcome wall): no legend row, the table toggle is a small button over the plot. */
  compact?: boolean;
}

export function ChartFrame({ label, table, height, draw, legend, className = "", toolbar, fallbackWidth = 760, status, compact = false }: ChartFrameProps) {
  const [ref, width] = useWidth<HTMLDivElement>(fallbackWidth);
  const [asTable, setAsTable] = useState(false);
  const tableId = useId();
  const h = typeof height === "function" ? height(width) : height;
  return (
    <figure className={`chart ${compact ? "compact " : ""}${className}`.trim()}>
      <div className="chart-bar">
        {legend && <div className="chart-legend">{legend}</div>}
        <div className="chart-tools">
          {toolbar}
          <button type="button" className="linkbtn" aria-pressed={asTable} aria-controls={tableId} onClick={() => setAsTable((v) => !v)}>
            {asTable ? "Show chart" : compact ? "Table" : "Show as table"}
          </button>
        </div>
      </div>
      {status && <div className="chart-status">{status}</div>}
      <div ref={ref} className="chart-plot" id={tableId} style={asTable ? undefined : { height: h }}>
        {asTable ? (
          <div className="chart-table scroll" tabIndex={0} role="group" aria-label={`${table.caption}, as a table`}>
            <ChartTable spec={table} />
          </div>
        ) : (
          <Drawn draw={draw} width={width} height={h} label={label} />
        )}
      </div>
    </figure>
  );
}

/** The drawing is its own component, so a chart's `draw` may call hooks (the crosshair's slider) without the table toggle changing the parent's hooks. */
function Drawn({ draw, width, height, label }: { draw: ChartFrameProps["draw"]; width: number; height: number; label: string }) {
  const drawing = draw(width, height);
  return (
    <>
      <svg className="chart-svg" role="img" aria-label={label} width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {drawing.svg}
      </svg>
      {drawing.overlay}
    </>
  );
}

export function ChartTable({ spec }: { spec: TableSpec }) {
  const numeric = new Set(spec.numeric ?? []);
  return (
    <table className="tbl">
      <caption className="sr-only">{spec.caption}</caption>
      <thead>
        <tr>
          {spec.columns.map((c, i) => (
            <th key={c} className={numeric.has(i) ? "r" : undefined}>
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {spec.rows.map((row, r) => (
          <tr key={r}>
            {row.map((cell, i) => (
              <td key={i} className={numeric.has(i) ? "r num" : undefined}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
        {spec.rows.length === 0 && (
          <tr>
            <td colSpan={spec.columns.length} className="muted">
              nothing to show yet
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

/** A legend entry: a key (a line, a dot, a swatch) in the series' colour and the name in text ink (text never wears the data colour). */
export function Key({ kind = "line", color, dashed = false, children }: { kind?: "line" | "dot" | "swatch" | "triangle" | "ring"; color: string; dashed?: boolean; children: ReactNode }) {
  return (
    <span className="key">
      <svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true">
        {kind === "line" && <line x1="1" x2="15" y1="5" y2="5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeDasharray={dashed ? "3 3" : undefined} />}
        {kind === "dot" && <circle cx="8" cy="5" r="3.5" fill={color} />}
        {kind === "ring" && <circle cx="8" cy="5" r="3.2" fill="none" stroke={color} strokeWidth="1.6" />}
        {kind === "swatch" && <rect x="2" y="1" width="12" height="8" rx="2" fill={color} />}
        {kind === "triangle" && <path d="M8 1 L12.5 9 L3.5 9 Z" fill={color} />}
      </svg>
      {children}
    </span>
  );
}
