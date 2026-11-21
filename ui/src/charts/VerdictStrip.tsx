// "Was the Critic right?" (spec 13.2 (4)): a strip plot of the excess returns of the stocks the Critic approved, the ones it denied and the neutral passes, a bar at each group's mean (the server's
// `mean_excess_pct`), the group's n beside its name, and "small sample" where n is under the minimum (P9). The sentence under it is the server's own (`critic_value.sentence`).
import { useMemo, useState } from "react";
import { points } from "../format";
import type { RecOutcome, Verdict, VerdictStats } from "../proving/types";
import { XAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { extent, linear, niceDomain, niceTicks } from "./scale";
import { TipRow, Tooltip } from "./Tooltip";

const RESULT_COLOR: Record<string, string> = { won: "var(--pos)", lost: "var(--neg)", open: "var(--neutral)" };
const ORDER: Verdict[] = ["approved", "denied", "neutral", "unreviewed"];
const WORD: Record<Verdict, string> = { approved: "Approved", denied: "Denied", neutral: "Neutral pass", unreviewed: "Not reviewed" };

export interface VerdictStripProps {
  outcomes: RecOutcome[];
  groups: VerdictStats[];
  benchmarkName?: string;
  fallbackWidth?: number;
}

export function VerdictStrip({ outcomes, groups, benchmarkName = "SPY", fallbackWidth = 460 }: VerdictStripProps) {
  const rows = useMemo(() => ORDER.filter((v) => (groups.find((g) => g.verdict === v)?.n ?? 0) > 0 || (v !== "unreviewed" && v !== "neutral")), [groups]);
  const dotsOf = (v: Verdict) => outcomes.filter((o) => o.verdict === v && o.excess_return_pct !== null && (o.result === "won" || o.result === "lost" || o.result === "open"));
  const [hot, setHot] = useState<{ o: RecOutcome; x: number; y: number } | null>(null);

  const table: TableSpec = {
    caption: `Excess return over ${benchmarkName} by the Critic's verdict`,
    columns: ["Verdict", "Outcomes", "Mean excess", "Win rate", "Sample"],
    numeric: [1, 2, 3],
    rows: rows.map((v) => {
      const g = groups.find((x) => x.verdict === v);
      return [WORD[v], g?.n ?? 0, g && g.n ? points(g.mean_excess_pct) : "—", g?.win_rate === null || g === undefined ? "—" : `${Math.round(g.win_rate * 100)}%`, !g || g.n === 0 ? "none" : g.small_sample ? "small sample" : "enough"];
    }),
  };
  const label = `Critic chart. ${rows.map((v) => { const g = groups.find((x) => x.verdict === v); return `${WORD[v]}: ${g?.n ?? 0} outcomes, mean ${g && g.n ? points(g.mean_excess_pct) : "none"} against ${benchmarkName}${g?.small_sample && g.n ? ", small sample" : ""}`; }).join(". ")}.`;

  return (
    <ChartFrame
      className="verdictstrip"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => 34 + rows.length * 66 + 48}
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
          <Key kind="swatch" color="var(--text)">
            Group mean
          </Key>
        </>
      }
      draw={(width, height) => {
        const m = { left: 118, right: 16, top: 10, bottom: 48 };
        const innerW = Math.max(60, width - m.left - m.right);
        const all = outcomes.filter((o) => o.excess_return_pct !== null && rows.includes(o.verdict));
        const ex = extent([0, ...all.map((o) => o.excess_return_pct), ...groups.map((g) => g.mean_excess_pct)]) ?? [-0.05, 0.05];
        const [d0, d1] = niceDomain(ex[0] - 0.01, ex[1] + 0.01, 5);
        const x = linear([d0, d1], [m.left, m.left + innerW]);
        const rowH = (height - m.top - m.bottom) / Math.max(1, rows.length);
        const ticks = niceTicks(d0, d1, 5);
        const svg = (
          <>
            {ticks.map((t) => (
              <line key={t} className={t === 0 ? "grid vert strong" : "grid vert"} x1={x(t)} x2={x(t)} y1={m.top} y2={height - m.bottom} />
            ))}
            <XAxis scale={x} ticks={ticks.map((t) => ({ day: t, label: points(t, 0).replace(" pts", ""), major: false }))} y={height - m.bottom + 4} />
            <text className="axis-title" x={m.left + innerW / 2} y={height - 6} textAnchor="middle" aria-hidden="true">
              excess return over {benchmarkName}, in points
            </text>
            {rows.map((v, r) => {
              const g = groups.find((q) => q.verdict === v);
              const cy = m.top + rowH * r + rowH / 2;
              const dots = dotsOf(v);
              return (
                <g key={v}>
                  {r > 0 && <line className="grid" x1={4} x2={m.left + innerW} y1={m.top + rowH * r} y2={m.top + rowH * r} />}
                  <text className="row-label" x={m.left - 12} y={cy - 6} textAnchor="end">
                    {WORD[v]}
                  </text>
                  <text className="row-sub" x={m.left - 12} y={cy + 9} textAnchor="end">
                    n={g?.n ?? 0}
                    {g && g.n > 0 && g.small_sample ? " · small sample" : ""}
                  </text>
                  {dots.map((o, i) => {
                    const jitter = ((((i * 7919) % 97) / 97) - 0.5) * (rowH * 0.62);
                    return <circle key={o.key} className={o.hypothetical ? "dot-hypo" : "dot-trade"} cx={x(o.excess_return_pct!)} cy={cy + jitter} r={4.2} fill={o.hypothetical ? "none" : (RESULT_COLOR[o.result] ?? "var(--neutral)")} stroke={o.hypothetical ? (RESULT_COLOR[o.result] ?? "var(--neutral)") : "none"} strokeWidth={1.8} onPointerEnter={() => setHot({ o, x: x(o.excess_return_pct!), y: cy + jitter })} onPointerLeave={() => setHot(null)} />;
                  })}
                  {g && g.n > 0 && g.mean_excess_pct !== null && (
                    <g className={`mean-bar${g.small_sample ? " small" : ""}`}>
                      <rect x={x(g.mean_excess_pct) - 1.5} y={cy - rowH * 0.36} width={3} height={rowH * 0.72} rx={1.5} />
                    </g>
                  )}
                </g>
              );
            })}
          </>
        );
        const overlay = hot && (
          <Tooltip x={hot.x} y={hot.y - 10} plotWidth={width} title={`${hot.o.symbol} · ${hot.o.decision_day}`}>
            <TipRow value={points(hot.o.excess_return_pct)} name={`against ${benchmarkName}`} />
            <TipRow value={hot.o.result} name={hot.o.hypothetical ? "not bought" : "traded"} />
          </Tooltip>
        );
        return { svg, overlay };
      }}
    />
  );
}
