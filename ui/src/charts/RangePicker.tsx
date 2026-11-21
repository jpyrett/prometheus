// The coverage range picker (spec 13.2, the launch pad): a strip over the last 18 months with one row per kind of stored data (bars, news, press releases, filings) drawn from `/coverage` (a thicker, stronger
// bar where more is stored, thin where thin), the range outside what can be tested hatched, and two draggable handles on top. A handle snaps to a pickable day, can never leave the usable range and can never
// cross the other. The handles are sliders (arrow keys one day, Page keys a week, Home and End to the edge). The decision days of the chosen cadence appear as ticks inside the range.
import { useMemo, useRef } from "react";
import type { Coverage } from "../proving/types";
import { useSvgId, XAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { clamp, dayNumber, isoOfDay, linear, moveHandle, nearestIndex, shortDate, stepHandle, timeTicks, weekdaysBetween, type DayRange } from "./scale";

const ROWS = [
  { key: "bars", label: "Day bars" },
  { key: "news", label: "News" },
  { key: "press_releases", label: "Press releases" },
  { key: "filings", label: "Filings" },
] as const;

export interface RangePickerProps {
  coverage: Coverage;
  range: { start: string; end: string };
  onChange: (range: { start: string; end: string }) => void;
  /** The decision days the cadence picks, drawn as ticks. */
  decisionDays: string[];
  disabled?: boolean;
  fallbackWidth?: number;
}

/** The months of the coverage as rows, oldest first (the server sends an object keyed by "2026-10"). */
export function monthRows(coverage: Coverage): { month: string; bars: number; news: number; press_releases: number; filings: number }[] {
  return Object.entries(coverage.months)
    .map(([month, c]) => ({ month, bars: c.bars ?? 0, news: c.news ?? 0, press_releases: c.press_releases ?? 0, filings: c.filings ?? 0 }))
    .sort((a, b) => a.month.localeCompare(b.month));
}

/** The days a handle may land on: the server's trading days inside the usable range when it sends them, else the weekdays of the range. */
export function pickableDays(coverage: Coverage): number[] {
  if (!coverage.usable?.first_day || !coverage.usable.last_day) return [];
  const first = dayNumber(coverage.usable.first_day);
  const last = dayNumber(coverage.usable.last_day);
  if (Number.isNaN(first) || Number.isNaN(last)) return [];
  const sent = (coverage.trading_days ?? []).map(dayNumber).filter((d) => !Number.isNaN(d) && d >= first && d <= last);
  return sent.length >= 2 ? sent.sort((a, b) => a - b) : weekdaysBetween(first, last);
}

/** A quick range (1M, 3M, 6M, Max): the last `months` months of the usable range, snapped to pickable days; null months is all of it. */
export function quickRange(days: number[], months: number | null): { start: string; end: string } | null {
  if (days.length < 2) return null;
  const end = days[days.length - 1]!;
  if (months === null) return { start: isoOfDay(days[0]!), end: isoOfDay(end) };
  const d = new Date(end * 86_400_000);
  const target = Math.round(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() - months, d.getUTCDate()) / 86_400_000);
  const start = days[Math.min(days.length - 2, Math.max(0, nearestIndex(days, target)))]!;
  return { start: isoOfDay(start), end: isoOfDay(end) };
}

const MONTH_DAYS = (month: string) => {
  const first = dayNumber(`${month}-01`);
  const [y, m] = month.split("-").map(Number);
  const next = Math.round(Date.UTC(y!, m!, 1) / 86_400_000);
  return [first, next] as const;
};

export function RangePicker({ coverage, range, onChange, decisionDays, disabled = false, fallbackWidth = 760 }: RangePickerProps) {
  const id = useSvgId("rp");
  const days = useMemo(() => pickableDays(coverage), [coverage]);
  const current: DayRange = { start: dayNumber(range.start), end: dayNumber(range.end) };
  const plot = useRef<HTMLDivElement | null>(null);

  const months = useMemo(() => monthRows(coverage), [coverage]);
  const table: TableSpec = {
    caption: "What is stored, by month",
    columns: ["Month", "Day bars", "News", "Press releases", "Filings"],
    numeric: [1, 2, 3, 4],
    rows: months.map((m) => [m.month, m.bars, m.news, m.press_releases, m.filings]),
  };
  const usable = coverage.usable?.first_day && coverage.usable.last_day ? { first: coverage.usable.first_day, last: coverage.usable.last_day } : null;
  const label = usable
    ? `Coverage of the stored data, with the test range ${range.start} to ${range.end}. The range can be moved inside ${usable.first} to ${usable.last}.`
    : "Coverage of the stored data: nothing is stored yet.";

  const commit = (next: DayRange) => {
    if (next.start === current.start && next.end === current.end) return;
    onChange({ start: isoOfDay(next.start), end: isoOfDay(next.end) });
  };

  return (
    <ChartFrame
      className="rangepicker"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => 22 + ROWS.length * 22 + 44}
      legend={
        <>
          <Key kind="swatch" color="var(--info)">
            Stored data (more is thicker)
          </Key>
          <Key kind="swatch" color="var(--text)">
            Chosen range
          </Key>
          <Key kind="dot" color="var(--text-2)">
            Decision days
          </Key>
        </>
      }
      draw={(width) => {
        const m = { left: 106, right: 18, top: 24 };
        const innerW = Math.max(80, width - m.left - m.right);
        const rowsTop = m.top;
        const rowH = 22;
        const rowsBottom = rowsTop + ROWS.length * rowH;
        const axisY = rowsBottom + 6;
        const hi = (usable ? dayNumber(coverage.last_bar_day ?? usable.last) : dayNumber(isoOfDay(Math.floor(Date.now() / 86_400_000)))) + 7;
        // the strip covers the last 18 months, and starts at the first month that has anything stored when that is earlier, so no cell is cut off at the edge
        const firstMonth = months.length ? dayNumber(`${months[0]!.month}-01`) : hi - 548;
        const lo = Math.min(hi - 548, usable ? dayNumber(usable.first) - 7 : hi - 548, firstMonth);
        const x = linear([lo, hi], [m.left, m.left + innerW]);
        const first = days[0] ?? lo;
        const last = days[days.length - 1] ?? hi;
        const maxOf = (k: (typeof ROWS)[number]["key"]) => Math.max(1, ...months.map((mo) => mo[k]));
        const ticks = timeTicks(lo, hi, width < 640 ? 5 : 10);
        const sx = x(clamp(current.start, lo, hi));
        const ex = x(clamp(current.end, lo, hi));

        const svg = (
          <>
            <defs>
              <pattern id={`${id}h`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--border-strong)" strokeWidth="1.5" />
              </pattern>
            </defs>
            <rect className="rp-track" x={m.left} y={rowsTop} width={innerW} height={rowsBottom - rowsTop} rx={6} />
            {/* what cannot be tested: before the first usable day and after the last */}
            {x(first) > m.left && <rect x={m.left} y={rowsTop} width={x(first) - m.left} height={rowsBottom - rowsTop} fill={`url(#${id}h)`} opacity={0.7} />}
            {x(last) < m.left + innerW && <rect x={x(last)} y={rowsTop} width={m.left + innerW - x(last)} height={rowsBottom - rowsTop} fill={`url(#${id}h)`} opacity={0.7} />}
            {ROWS.map((r, i) => {
              const cy = rowsTop + rowH * i + rowH / 2;
              return (
                <g key={r.key}>
                  <text className="row-label" x={m.left - 10} y={cy} dy="0.32em" textAnchor="end" aria-hidden="true">
                    {r.label}
                  </text>
                  {i > 0 && <line className="grid" x1={m.left} x2={m.left + innerW} y1={rowsTop + rowH * i} y2={rowsTop + rowH * i} />}
                  {months
                    .filter((mo) => mo[r.key] > 0)
                    .map((mo) => {
                      const [a, b] = MONTH_DAYS(mo.month);
                      const share = Math.sqrt(mo[r.key] / maxOf(r.key));
                      const h = 2 + share * (rowH - 8);
                      return <rect key={mo.month} className="rp-cell" x={x(a) + 1} y={cy - h / 2} width={Math.max(1, x(b) - x(a) - 2)} height={h} rx={2} opacity={0.5 + share * 0.45} />;
                    })}
                </g>
              );
            })}
            <rect className="rp-range" x={sx} y={rowsTop - 4} width={Math.max(0, ex - sx)} height={rowsBottom - rowsTop + 8} rx={6} />
            {decisionDays.map((d) => {
              const dx = x(dayNumber(d));
              return <line key={d} className="rp-decision" x1={dx} x2={dx} y1={rowsTop - 4} y2={rowsTop + 3} />;
            })}
            <XAxis scale={x} ticks={ticks} y={axisY} />
            <text className="rp-date" x={sx - 8} y={rowsTop - 10} textAnchor={sx - m.left > 52 ? "end" : "start"} aria-hidden="true">
              {shortDate(range.start, true)}
            </text>
            <text className="rp-date" x={ex + 8} y={rowsTop - 10} textAnchor={m.left + innerW - ex > 70 ? "start" : "end"} aria-hidden="true">
              {shortDate(range.end, true)}
            </text>
          </>
        );

        // dragging: a handle follows the pointer to a pickable day, and a press on the track moves the nearer handle there
        const dayAt = (clientX: number) => {
          const rect = plot.current?.getBoundingClientRect();
          return x.invert(clientX - (rect?.left ?? 0));
        };
        const drag = (which: "start" | "end") => ({
          onPointerDown: (e: React.PointerEvent<HTMLElement>) => {
            if (disabled) return;
            e.currentTarget.setPointerCapture?.(e.pointerId);
            e.currentTarget.focus();
          },
          onPointerMove: (e: React.PointerEvent<HTMLElement>) => {
            if (disabled || !e.currentTarget.hasPointerCapture?.(e.pointerId)) return;
            commit(moveHandle(current, which, dayAt(e.clientX), days));
          },
          onKeyDown: (e: React.KeyboardEvent<HTMLElement>) => {
            if (disabled) return;
            const step = e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "PageDown" ? -5 : e.key === "PageUp" ? 5 : e.key === "Home" ? -100000 : e.key === "End" ? 100000 : 0;
            if (step === 0) return;
            e.preventDefault();
            commit(stepHandle(current, which, step, days));
          },
        });
        const handle = (which: "start" | "end", px: number) => {
          const value = which === "start" ? current.start : current.end;
          return (
            <div
              key={which}
              role="slider"
              tabIndex={disabled ? -1 : 0}
              className={`rp-handle ${which}`}
              style={{ left: px, top: rowsTop - 6, height: rowsBottom - rowsTop + 12 }}
              aria-label={which === "start" ? "Start of the test range" : "End of the test range"}
              aria-valuemin={0}
              aria-valuemax={Math.max(0, days.length - 1)}
              aria-valuenow={Math.max(0, nearestIndex(days, value))}
              aria-valuetext={isoOfDay(value)}
              aria-disabled={disabled || undefined}
              {...drag(which)}
            >
              <i />
            </div>
          );
        };
        const overlay = (
          <div ref={plot} className="rp-overlay">
            <div
              className="rp-hit"
              style={{ left: m.left, top: rowsTop, width: innerW, height: rowsBottom - rowsTop }}
              onPointerDown={(e) => {
                if (disabled || e.target !== e.currentTarget) return;
                const d = dayAt(e.clientX);
                const which = Math.abs(d - current.start) <= Math.abs(d - current.end) ? "start" : "end";
                commit(moveHandle(current, which, d, days));
              }}
            />
            {days.length >= 2 && handle("start", sx)}
            {days.length >= 2 && handle("end", ex)}
            <span className="sr-only" role="status">
              {range.start} to {range.end}
            </span>
          </div>
        );
        return { svg, overlay };
      }}
    />
  );
}
