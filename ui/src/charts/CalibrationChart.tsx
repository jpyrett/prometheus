// "Was the Advisor's score right?" (spec 13.2 (4)): one dot per outcome, x = the Advisor's score, y = the excess return over the benchmark, coloured by result (won green, lost red, open grey; a ring is a stock
// the desk did not buy), and the buckets of `by_score` as a stepped line with each bucket's n. A bucket under the minimum sample is drawn dashed and grey: no colour of success or failure (P9).
import { useMemo } from "react";
import { pct, points } from "../format";
import type { RecOutcome, ScoreStats } from "../proving/types";
import { XAxis, YAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { Crosshair, useScrub } from "./Crosshair";
import { stepPath } from "./path";
import { extent, linear, niceDomain, niceTicks } from "./scale";
import { TipRow, Tooltip } from "./Tooltip";

const RESULT_COLOR: Record<string, string> = { won: "var(--pos)", lost: "var(--neg)", open: "var(--neutral)" };

export interface CalibrationProps {
  outcomes: RecOutcome[];
  buckets: ScoreStats[];
  benchmarkName?: string;
  minSample: number;
  fallbackWidth?: number;
}

export function CalibrationChart({ outcomes, buckets, benchmarkName = "SPY", minSample, fallbackWidth = 460 }: CalibrationProps) {
  // the dots: an outcome with a score and an excess return (what by_score is made of), in score order so the keyboard walks left to right
  const dots = useMemo(
    () =>
      outcomes
        .filter((o) => o.advisor?.score !== null && o.advisor?.score !== undefined && o.excess_return_pct !== null && (o.result === "won" || o.result === "lost" || o.result === "open"))
        .sort((a, b) => a.advisor!.score! - b.advisor!.score! || a.decision_day.localeCompare(b.decision_day)),
    [outcomes],
  );
  const shown = buckets.filter((b) => b.n > 0);

  const table: TableSpec = {
    caption: "The Advisor's score against the excess return, by bucket",
    columns: ["Score bucket", "Outcomes", "Mean excess", "Win rate", "Sample"],
    numeric: [1, 2, 3],
    rows: buckets.map((b) => [`${b.low}–${Math.min(b.high - 1, 100)}`, b.n, b.n ? points(b.mean_excess_pct) : "—", b.win_rate === null ? "—" : `${Math.round(b.win_rate * 100)}%`, b.n === 0 ? "none" : b.small_sample ? `small sample (under ${minSample})` : "enough"]),
  };
  const label = `Calibration chart. ${dots.length} outcomes by Advisor score against excess return over ${benchmarkName}. ${shown.map((b) => `Scores ${b.low} to ${Math.min(b.high - 1, 100)}: ${b.n} outcomes, mean ${points(b.mean_excess_pct)}${b.small_sample ? ", small sample" : ""}`).join(". ")}.`;

  return (
    <ChartFrame
      className="calibration"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={316}
      legend={
        <>
          <Key kind="dot" color="var(--pos)">
            Won
          </Key>
          <Key kind="dot" color="var(--neg)">
            Lost
          </Key>
          <Key kind="dot" color="var(--neutral)">
            Open
          </Key>
          <Key kind="ring" color="var(--text-2)">
            Not bought
          </Key>
          <Key color="var(--text)">Bucket mean</Key>
        </>
      }
      draw={(width, height) => {
        const m = { left: 52, right: 14, top: 12, bottom: 62 };
        const innerW = Math.max(60, width - m.left - m.right);
        const scores = extent([...dots.map((d) => d.advisor!.score), ...shown.map((b) => b.low), ...shown.map((b) => Math.min(b.high, 100))]) ?? [40, 100];
        const x = linear([Math.max(0, Math.floor(scores[0] / 10) * 10), 100], [m.left, m.left + innerW]);
        const ys = extent([0, ...dots.map((d) => d.excess_return_pct), ...shown.map((b) => b.mean_excess_pct)]) ?? [-0.05, 0.05];
        const [y0, y1] = niceDomain(ys[0] - 0.01, ys[1] + 0.01, 5);
        const y = linear([y0, y1], [height - m.bottom, m.top]);
        const yTicks = niceTicks(y0, y1, 5);
        const xTicks = niceTicks(x.domain[0], 100, 6).map((d) => ({ day: d, label: String(d), major: false }));

        const xy = dots.map((d) => [x(d.advisor!.score!), y(d.excess_return_pct!)] as const);
        const nearest = (px: number, py: number) => {
          let best = -1;
          let bestD = Infinity;
          xy.forEach(([dx, dy], i) => {
            const d = (dx - px) ** 2 + (dy - py) ** 2;
            if (d < bestD) {
              bestD = d;
              best = i;
            }
          });
          return bestD <= 40 * 40 ? best : -1;
        };
        const describe = (i: number) => `${dots[i]!.symbol} ${dots[i]!.decision_day}: score ${dots[i]!.advisor!.score}, ${points(dots[i]!.excess_return_pct)} against ${benchmarkName}, ${dots[i]!.result}`;
        const scrub = useScrub({ xs: xy.map((p) => p[0]), left: m.left, top: m.top, nearest, describe, label: "Move between the dots with the arrow keys" });
        const at = scrub.index !== null && scrub.index >= 0 ? dots[scrub.index] : undefined;

        // the bucket means: a step through the buckets that have outcomes, each held across its own score range
        const stepPts = shown.flatMap((b) => [[x(b.low), y(b.mean_excess_pct ?? 0)] as const, [x(Math.min(b.high, 100)), y(b.mean_excess_pct ?? 0)] as const]);

        const svg = (
          <>
            <YAxis scale={y} ticks={yTicks} x={m.left} right={m.left + innerW} format={(v) => pct(v, 0, true)} emphasize={0} />
            <XAxis scale={x} ticks={xTicks} y={height - m.bottom + 4} />
            <text className="axis-title" x={m.left + innerW / 2} y={height - 5} textAnchor="middle" aria-hidden="true">
              the Advisor&apos;s score
            </text>
            {shown.map((b) => {
              const cx = (x(b.low) + x(Math.min(b.high, 100))) / 2;
              return (
                <text key={b.low} className="tick-label" x={cx} y={height - m.bottom + 33} textAnchor="middle" aria-hidden="true">
                  n={b.n}
                  {b.small_sample ? "*" : ""}
                </text>
              );
            })}
            {shown.map((b) => (
              <line key={`s${b.low}`} className={`bucket-mean${b.small_sample ? " small" : ""}`} x1={x(b.low)} x2={x(Math.min(b.high, 100))} y1={y(b.mean_excess_pct ?? 0)} y2={y(b.mean_excess_pct ?? 0)} />
            ))}
            {stepPts.length > 2 && <path className="bucket-step" d={stepPath(stepPts)} />}
            {dots.map((d, i) => {
              const color = RESULT_COLOR[d.result] ?? "var(--neutral)";
              return d.hypothetical ? (
                <circle key={d.key} className="dot-hypo" cx={xy[i]![0]} cy={xy[i]![1]} r={4.5} fill="none" stroke={color} strokeWidth={1.8} />
              ) : (
                <circle key={d.key} className="dot-trade" cx={xy[i]![0]} cy={xy[i]![1]} r={4.5} fill={color} />
              );
            })}
            {at && scrub.index !== null && <Crosshair x={xy[scrub.index]![0]} top={m.top} bottom={height - m.bottom} dots={[{ y: xy[scrub.index]![1], color: RESULT_COLOR[at.result] ?? "var(--neutral)" }]} />}
          </>
        );
        const overlay = (
          <>
            <div className="chart-probe" style={{ left: m.left, top: m.top, width: innerW, height: height - m.top - m.bottom }} {...scrub.bind} />
            {at && scrub.index !== null && (
              <Tooltip x={xy[scrub.index]![0]} y={xy[scrub.index]![1] - 10} plotWidth={width} title={`${at.symbol} · ${at.decision_day}`}>
                <TipRow value={points(at.excess_return_pct)} name={`against ${benchmarkName}`} />
                <TipRow value={`score ${at.advisor!.score}`} name={at.verdict} />
                <TipRow value={at.result} name={at.hypothetical ? "not bought" : "traded"} />
              </Tooltip>
            )}
          </>
        );
        return { svg, overlay };
      }}
    />
  );
}
