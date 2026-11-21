// "Was the Advisor's score right?" (spec 13.2 (4)): one dot per outcome, x = the Advisor's score, y = the excess return over the benchmark, coloured by result (won green, lost red, open grey; a ring is a stock
// the desk did not buy), and the buckets of `by_score` as a stepped line with each bucket's n. A bucket under the minimum sample is drawn dashed and grey: no colour of success or failure (P9).
// FORWARD (design/advisor_exit_plan_spec.md E6): a run that kept the forward excess draws THAT, the stock against the benchmark 20 sessions after the decision day whatever the plan did, for the strategy-family picks (a company the
// Advisor scored neutral is in the family table); the win rate of a bucket is then "won by its plan". A run saved before it existed is drawn by the plan, as it always was.
import { useMemo } from "react";
import { pct, points } from "../format";
import type { RecOutcome, ScoreStats } from "../proving/types";
import { XAxis, YAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { Crosshair, useScrub } from "./Crosshair";
import { stepPath } from "./path";
import { extent, linear, niceDomain, niceTicks } from "./scale";
import { TipRow, Tooltip } from "./Tooltip";

const HEADLINE_SESSIONS = 20;
const RESULT_COLOR: Record<string, string> = { won: "var(--pos)", lost: "var(--neg)", open: "var(--neutral)" };

export interface CalibrationProps {
  outcomes: RecOutcome[];
  buckets: ScoreStats[];
  benchmarkName?: string;
  minSample: number;
  fallbackWidth?: number;
}

export function CalibrationChart({ outcomes, buckets, benchmarkName = "SPY", minSample, fallbackWidth = 460 }: CalibrationProps) {
  const forwardOf = (b: ScoreStats) => b.forward?.find((x) => x.sessions === HEADLINE_SESSIONS) ?? null;
  const byForward = buckets.some((b) => (forwardOf(b)?.n ?? 0) > 0); // a run that kept the forward excess draws it; an older one draws the plan's
  const excessOf = (o: RecOutcome) => (byForward ? (o.forward?.find((f) => f.sessions === HEADLINE_SESSIONS)?.excess_return_pct ?? null) : o.excess_return_pct);
  // the dots: an outcome with a score and an excess return (what by_score is made of), in score order so the keyboard walks left to right
  const dots = useMemo(
    () =>
      outcomes
        .filter((o) => o.advisor?.score !== null && o.advisor?.score !== undefined && excessOf(o) !== null && (byForward ? !!o.advisor?.family : o.result === "won" || o.result === "lost" || o.result === "open"))
        .sort((a, b) => a.advisor!.score! - b.advisor!.score! || a.decision_day.localeCompare(b.decision_day)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [outcomes, byForward],
  );
  const nOf = (b: ScoreStats) => (byForward ? (forwardOf(b)?.n ?? 0) : b.n);
  const meanOf = (b: ScoreStats) => (byForward ? (forwardOf(b)?.mean_excess_pct ?? null) : b.mean_excess_pct);
  const smallOf = (b: ScoreStats) => (byForward ? (forwardOf(b)?.small_sample ?? true) : b.small_sample);
  const shown = buckets.filter((b) => nOf(b) > 0);

  const table: TableSpec = byForward
    ? {
        caption: `The Advisor's score against the excess return ${HEADLINE_SESSIONS} sessions after the decision day, by bucket (strategy-family picks)`,
        columns: ["Score bucket", "Picks", "Forward excess", "Std. error", "Won by its plan", "Sample"],
        numeric: [1, 2, 3, 4],
        rows: buckets.map((b) => {
          const f = forwardOf(b);
          return [
            `${b.low}–${Math.min(b.high - 1, 100)}`,
            f?.n ?? 0,
            f && f.n ? points(f.mean_excess_pct) : "—",
            f && f.n ? points(f.se_pct) : "—",
            b.win_rate === null ? "—" : `${Math.round(b.win_rate * 100)}%`,
            !f || f.n === 0 ? "none" : f.small_sample ? `small sample (under ${minSample})` : f.inside_chance ? "within chance" : "beyond chance",
          ];
        }),
      }
    : {
        caption: "The Advisor's score against the excess return, by bucket",
        columns: ["Score bucket", "Outcomes", "Mean excess", "Win rate", "Sample"],
        numeric: [1, 2, 3],
        rows: buckets.map((b) => [`${b.low}–${Math.min(b.high - 1, 100)}`, b.n, b.n ? points(b.mean_excess_pct) : "—", b.win_rate === null ? "—" : `${Math.round(b.win_rate * 100)}%`, b.n === 0 ? "none" : b.small_sample ? `small sample (under ${minSample})` : "enough"]),
      };
  const label = `Calibration chart. ${dots.length} ${byForward ? `picks by Advisor score against excess return over ${benchmarkName} ${HEADLINE_SESSIONS} sessions after the decision day` : `outcomes by Advisor score against excess return over ${benchmarkName}`}. ${shown.map((b) => `Scores ${b.low} to ${Math.min(b.high - 1, 100)}: ${nOf(b)} ${byForward ? "picks" : "outcomes"}, mean ${points(meanOf(b))}${smallOf(b) ? ", small sample" : ""}`).join(". ")}.`;

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
        const ys = extent([0, ...dots.map((d) => excessOf(d)), ...shown.map((b) => meanOf(b))]) ?? [-0.05, 0.05];
        const [y0, y1] = niceDomain(ys[0] - 0.01, ys[1] + 0.01, 5);
        const y = linear([y0, y1], [height - m.bottom, m.top]);
        const yTicks = niceTicks(y0, y1, 5);
        const xTicks = niceTicks(x.domain[0], 100, 6).map((d) => ({ day: d, label: String(d), major: false }));

        const xy = dots.map((d) => [x(d.advisor!.score!), y(excessOf(d)!)] as const);
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
        const describe = (i: number) => `${dots[i]!.symbol} ${dots[i]!.decision_day}: score ${dots[i]!.advisor!.score}, ${points(excessOf(dots[i]!))} against ${benchmarkName}, ${dots[i]!.result}`;
        const scrub = useScrub({ xs: xy.map((p) => p[0]), left: m.left, top: m.top, nearest, describe, label: "Move between the dots with the arrow keys" });
        const at = scrub.index !== null && scrub.index >= 0 ? dots[scrub.index] : undefined;

        // the bucket means: a step through the buckets that have outcomes, each held across its own score range
        const stepPts = shown.flatMap((b) => [[x(b.low), y(meanOf(b) ?? 0)] as const, [x(Math.min(b.high, 100)), y(meanOf(b) ?? 0)] as const]);

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
                  n={nOf(b)}
                  {smallOf(b) ? "*" : ""}
                </text>
              );
            })}
            {shown.map((b) => (
              <line key={`s${b.low}`} className={`bucket-mean${smallOf(b) ? " small" : ""}`} x1={x(b.low)} x2={x(Math.min(b.high, 100))} y1={y(meanOf(b) ?? 0)} y2={y(meanOf(b) ?? 0)} />
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
                <TipRow value={points(excessOf(at))} name={byForward ? `against ${benchmarkName} after ${HEADLINE_SESSIONS} sessions` : `against ${benchmarkName}`} />
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
