// The plan chart of an outcome card (spec 13.2 (3)): the price path from the entry as a line against SPY's (blue, thin: SPY's return since the entry laid on the entry price so both start at one point),
// the entry zone as a band, the stop as a red line, the target as a green line, the time stop as a vertical mark, the exit as a dot. A trade the desk did not make (P8) is drawn in grey and dashed.
import { useMemo } from "react";
import { money, pct, price4 } from "../format";
import type { RecOutcome } from "../proving/types";
import { useSvgId } from "./Axis";
import { ChartFrame, type TableSpec } from "./ChartFrame";
import { Crosshair, useScrub } from "./Crosshair";
import { areaPath, linePath } from "./path";
import { dayNumber, extent, linear, shortDate } from "./scale";
import { TipRow, Tooltip } from "./Tooltip";

const EXIT_COLOR: Record<string, string> = { target: "var(--pos)", stop: "var(--neg)", time_stop: "var(--neutral)" };
const EXIT_WORD: Record<string, string> = { target: "target", stop: "stop", time_stop: "time stop" };

/** Why an outcome has no path, in words (the Trader's own code for a limit never reached). */
export function noPathWords(o: RecOutcome): string {
  if (o.result === "not_filled") {
    const why: Record<string, string> = { limit_not_reached: "the price never reached the limit", price_outside_limits: "the open was outside the entry zone", price_unavailable: "no price was available" };
    return `Not filled: ${o.reject_code ? (why[o.reject_code] ?? o.reject_code.replace(/_/g, " ")) : "no fill"}.`;
  }
  return `No data: ${o.notes[0] ?? "no stored bars after the decision day"}.`;
}

export function PlanChart({ outcome, fallbackWidth = 236 }: { outcome: RecOutcome; fallbackWidth?: number }) {
  const id = useSvgId("pc");
  const o = outcome;
  const hypothetical = o.hypothetical;
  const entry = o.entry;
  const path = o.path;

  const table: TableSpec = useMemo(
    () => ({
      caption: `${o.symbol} ${o.decision_day}: price path against the plan`,
      columns: ["Date", "Close", "Return", "SPY"],
      numeric: [1, 2, 3],
      rows: path.map((p) => [p.date, price4(p.close), pct(p.return_pct), pct(p.benchmark_return_pct)]),
    }),
    [o.symbol, o.decision_day, path],
  );

  if (!entry || path.length === 0) {
    return (
      <div className="pc-empty" role="img" aria-label={`${o.symbol}: ${noPathWords(o)}`}>
        {noPathWords(o)}
      </div>
    );
  }

  const last = path[path.length - 1]!;
  const label = `${o.symbol} plan chart. Entered ${shortDate(entry.date, true)} at ${money(entry.price)}${o.plan.stop ? `, stop ${money(o.plan.stop)}` : ""}${o.plan.target ? `, target ${money(o.plan.target)}` : ""}. ${
    o.exit ? `Exited at the ${EXIT_WORD[o.exit.reason]} on ${shortDate(o.exit.date, true)} at ${money(o.exit.price)}` : `Still open, last close ${money(last.close)} on ${shortDate(last.date, true)}`
  }${o.return_pct !== null ? `, return ${pct(o.return_pct)}` : ""}${o.excess_return_pct !== null ? `, ${pct(o.excess_return_pct)} against SPY` : ""}.${hypothetical ? " Not traded: a share-equivalent followed by the plan." : ""}`;

  return (
    <ChartFrame
      compact
      className={`planchart${hypothetical ? " hypo" : ""}`}
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={96}
      draw={(width, height) => {
        const m = { left: 4, right: 46, top: 18, bottom: 16 };
        const innerW = Math.max(40, width - m.left - m.right);
        const t0 = dayNumber(entry.date);
        const timeStop = o.plan.time_stop_date ? dayNumber(o.plan.time_stop_date) : null;
        const tLast = dayNumber(last.date);
        // the chart runs from the entry to the time stop; when the time stop is far beyond what has been followed the path would be a sliver, so the axis stops a little past the path and the time stop is said at the edge
        const farStop = timeStop !== null && !Number.isNaN(timeStop) ? timeStop : tLast;
        const reach = tLast + Math.max(5, tLast - t0);
        const t1 = Math.max(Math.min(Math.max(tLast, farStop), reach), t0 + 1);
        const timeStopBeyond = timeStop !== null && !Number.isNaN(timeStop) && timeStop > t1;
        const x = linear([t0, t1], [m.left, m.left + innerW]);

        const bench = path.map((p) => (p.benchmark_return_pct === null ? null : entry.price * (1 + p.benchmark_return_pct)));
        const range = extent([...path.map((p) => p.close), ...bench, o.plan.stop, o.plan.target, o.plan.entry_low, o.plan.entry_high, entry.price, o.exit?.price]) ?? [entry.price, entry.price];
        const spread = range[1] - range[0] || entry.price * 0.02;
        const y = linear([range[0] - spread * 0.1, range[1] + spread * 0.1], [height - m.bottom, m.top]);

        const closeXY = path.map((p) => [x(dayNumber(p.date)), y(p.close)] as const);
        const benchXY = path.flatMap((p, i) => (bench[i] === null ? [] : [[x(dayNumber(p.date)), y(bench[i]!)] as const]));
        const scrubXs = path.map((p) => x(dayNumber(p.date)));
        const describe = (i: number) => `${shortDate(path[i]!.date, true)}: close ${price4(path[i]!.close)}, return ${pct(path[i]!.return_pct)}`;
        const scrub = useScrub({ xs: scrubXs, left: m.left, describe, label: `Move along the ${o.symbol} price path` });
        const at = scrub.index === null ? null : path[scrub.index]!;
        const plotRight = m.left + innerW;

        const svg = (
          <>
            {o.plan.entry_low !== null && o.plan.entry_high !== null && <rect className="pc-zone" x={m.left} width={innerW} y={y(o.plan.entry_high)} height={Math.max(1.5, y(o.plan.entry_low) - y(o.plan.entry_high))} />}
            {o.plan.stop !== null && (
              <>
                <line className="pc-stop" x1={m.left} x2={plotRight} y1={y(o.plan.stop)} y2={y(o.plan.stop)} />
                <text className="pc-label" x={plotRight + 4} y={y(o.plan.stop)} dy="0.32em">
                  stop
                </text>
              </>
            )}
            {o.plan.target !== null && (
              <>
                <line className="pc-target" x1={m.left} x2={plotRight} y1={y(o.plan.target)} y2={y(o.plan.target)} />
                <text className="pc-label" x={plotRight + 4} y={y(o.plan.target)} dy="0.32em">
                  target
                </text>
              </>
            )}
            {timeStop !== null && !Number.isNaN(timeStop) && !timeStopBeyond && (
              <>
                <line className="pc-time" x1={x(timeStop)} x2={x(timeStop)} y1={m.top - 2} y2={height - m.bottom} />
                <text className="pc-label" x={x(timeStop)} y={height - 4} textAnchor="end">
                  time stop
                </text>
              </>
            )}
            {timeStopBeyond && o.plan.time_stop_date && (
              <text className="pc-label" x={plotRight} y={height - 4} textAnchor="end">
                time stop {shortDate(o.plan.time_stop_date)} ›
              </text>
            )}
            {benchXY.length > 1 && <path className="pc-bench" d={linePath(benchXY)} />}
            {closeXY.length > 1 && <path className="pc-wash" d={areaPath(closeXY, height - m.bottom)} fill={`url(#${id})`} />}
            <defs>
              <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="currentColor" stopOpacity="0.14" />
                <stop offset="1" stopColor="currentColor" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path className="pc-line" d={linePath(closeXY)} />
            <circle className="pc-entry" cx={x(t0)} cy={y(entry.price)} r={3} />
            {o.exit && <circle className="pc-exit" cx={x(dayNumber(o.exit.date))} cy={y(o.exit.price)} r={4.5} fill={hypothetical ? "var(--neutral)" : (EXIT_COLOR[o.exit.reason] ?? "var(--neutral)")} />}
            {!o.exit && <circle className="pc-now" cx={x(tLast)} cy={y(last.close)} r={3.5} />}
            {scrub.index !== null && at && <Crosshair x={x(dayNumber(at.date))} top={m.top} bottom={height - m.bottom} dots={[{ y: y(at.close), color: "var(--text)" }]} />}
            <text className="pc-label" x={m.left} y={height - 4}>
              {shortDate(entry.date)}
            </text>
          </>
        );
        const overlay = (
          <>
            <div className="chart-probe" style={{ left: m.left, top: 0, width: innerW, height: height }} {...scrub.bind} />
            {at && (
              <Tooltip x={x(dayNumber(at.date))} y={y(at.close) - 8} plotWidth={width} title={shortDate(at.date, true)}>
                <TipRow color="var(--text)" value={price4(at.close)} name="close" />
                <TipRow value={pct(at.return_pct)} name="since the entry" />
                {at.benchmark_return_pct !== null && <TipRow color="var(--info)" value={pct(at.benchmark_return_pct)} name="SPY" />}
              </Tooltip>
            )}
          </>
        );
        return { svg, overlay };
      }}
    />
  );
}
