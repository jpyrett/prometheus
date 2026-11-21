// "How did the trades end?" (spec 13.2 (4)): the exit mix as one stacked bar (target, stop, trailing stop when there is one, time stop, open), the average win against the average loss as two bars from a common zero, and the families
// as small horizontal bars of their mean FORWARD excess (the stock against the benchmark 20 sessions after the decision day, whatever the plan did: design/advisor_exit_plan_spec.md E6) with the n and the standard error;
// a run saved before the forward excess existed shows the mean excess BY THE PLAN as it always did. A family's win rate is "won by its plan". Every figure is the server's (P12); a family under the minimum sample is grey and says so,
// and so is a mean that chance alone gives (within two standard errors of zero).
import { useState } from "react";
import { pct, points } from "../format";
import type { FamilyStats, OutcomeStats } from "../proving/types";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { extent, linear, niceDomain } from "./scale";
import { Tooltip, TipRow } from "./Tooltip";

const ALL_PARTS = [
  { key: "target", label: "Target", color: "var(--pos)" },
  { key: "stop", label: "Stop", color: "var(--neg)" },
  { key: "trailing_stop", label: "Trailing stop", color: "var(--text-2)" },
  { key: "time_stop", label: "Time stop", color: "var(--neutral)" },
  { key: "open", label: "Still open", color: "var(--info)" },
] as const;

/** The horizon the family bars are drawn at; the table lists the others. */
const HEADLINE_SESSIONS = 20;

export interface ExitMixProps {
  outcomes: OutcomeStats;
  families: FamilyStats[];
  benchmarkName?: string;
  fallbackWidth?: number;
}

export function ExitMix({ outcomes, families, benchmarkName = "SPY", fallbackWidth = 460 }: ExitMixProps) {
  const [hot, setHot] = useState<{ x: number; y: number; title: string; rows: [string, string][] } | null>(null);
  const exits = outcomes.exits;
  const count = (key: (typeof ALL_PARTS)[number]["key"]) => exits[key] ?? 0; // a run saved before the trailing stop has no such count
  const PARTS = ALL_PARTS.filter((p) => p.key !== "trailing_stop" || count(p.key) > 0); // the trailing stop is a part of the bar only when a trade ended by one
  const total = PARTS.reduce((s, p) => s + count(p.key), 0);
  const forwardOf = (f: FamilyStats, sessions: number) => f.forward?.find((x) => x.sessions === sessions) ?? null;
  const byForward = families.some((f) => (forwardOf(f, HEADLINE_SESSIONS)?.n ?? 0) > 0); // a run that kept the forward excess draws it; an older one draws the plan's
  const shownFamilies = byForward ? families.filter((f) => (forwardOf(f, HEADLINE_SESSIONS)?.n ?? 0) > 0) : families.filter((f) => f.n > 0);
  const valueOf = (f: FamilyStats) => (byForward ? forwardOf(f, HEADLINE_SESSIONS)?.mean_excess_pct ?? null : f.mean_excess_pct);
  const forwardWords = (f: FamilyStats) =>
    (f.forward ?? [])
      .filter((x) => x.n > 0)
      .map((x) => `${x.sessions} sessions ${points(x.mean_excess_pct)} (se ${points(x.se_pct)}, n ${x.n}${x.inside_chance ? ", within chance" : ""})`)
      .join("; ");

  const table: TableSpec = {
    caption: "How the trades ended",
    columns: ["", "Number", "Detail"],
    numeric: [1],
    rows: [
      ...PARTS.map((p) => [p.label, count(p.key), total ? `${Math.round((count(p.key) / total) * 100)}% of ${total}` : "—"]),
      ["Average win", outcomes.won, pct(outcomes.avg_win_pct)],
      ["Average loss", outcomes.lost, pct(outcomes.avg_loss_pct)],
      ...shownFamilies.map((f) => [
        `Family: ${f.family.replace(/_/g, " ")}`,
        f.n,
        byForward
          ? `after the decision day, against ${benchmarkName}: ${forwardWords(f)}. By its plan: ${points(f.mean_excess_pct)}, won by its plan ${f.win_rate === null ? "—" : `${Math.round(f.win_rate * 100)}% of ${f.closed}`}${f.small_sample ? " (small sample)" : ""}`
          : `${points(f.mean_excess_pct)} against ${benchmarkName}${f.small_sample ? " (small sample)" : ""}`,
      ]),
    ],
  };
  const label = `Exit mix. ${total} trades followed: ${PARTS.map((p) => `${count(p.key)} ${p.label.toLowerCase()}`).join(", ")}. Average win ${pct(outcomes.avg_win_pct)} over ${outcomes.won} winners, average loss ${pct(outcomes.avg_loss_pct)} over ${outcomes.lost} losers.${outcomes.small_sample ? " Small sample." : ""}`;

  return (
    <ChartFrame
      className="exitmix"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => 140 + (shownFamilies.length ? 30 + shownFamilies.length * 22 : 0)}
      legend={PARTS.map((p) => (
        <Key key={p.key} kind="swatch" color={p.color}>
          {p.label} {count(p.key)}
        </Key>
      ))}
      draw={(width) => {
        const m = { left: 12, right: 14 };
        const innerW = Math.max(80, width - m.left - m.right);
        let cursor = m.left;
        const gap = 2;
        const segs = PARTS.filter((p) => count(p.key) > 0).map((p) => {
          const w = total ? (count(p.key) / total) * (innerW - gap * 3) : 0;
          const seg = { ...p, x: cursor, w };
          cursor += w + gap;
          return seg;
        });

        // average win against average loss: two bars from a common zero
        const ex = extent([0, outcomes.avg_win_pct, outcomes.avg_loss_pct]) ?? [-0.05, 0.05];
        const [d0, d1] = niceDomain(Math.min(ex[0], -0.001), Math.max(ex[1], 0.001), 4);
        const labelW = 88;
        const bx = linear([d0, d1], [m.left + labelW, m.left + innerW - 96]);
        const barRow = (y: number, value: number | null, name: string, color: string, n: string) => (
          <g key={name}>
            <text className="row-label" x={m.left} y={y + 9} >
              {name}
            </text>
            {value !== null && <rect className="hbar" x={Math.min(bx(0), bx(value))} y={y} width={Math.abs(bx(value) - bx(0))} height={13} rx={3} fill={color} />}
            <text className="mean-text" x={value !== null && value < 0 ? bx(0) + 6 : (value !== null ? Math.max(bx(0), bx(value)) : bx(0)) + 6} y={y + 10} textAnchor="start">
              {value === null ? "—" : `${pct(value)} ${n}`}
            </text>
          </g>
        );

        // the families: mean excess over the benchmark, one small bar each
        const fx = extent([0, ...shownFamilies.map((f) => valueOf(f))]) ?? [-0.05, 0.05];
        const [f0, f1] = niceDomain(Math.min(fx[0], -0.001), Math.max(fx[1], 0.001), 4);
        const famX = linear([f0, f1], [m.left + 112, m.left + innerW - 120]);
        const famTop = 124;

        const svg = (
          <>
            <text className="row-sub" x={m.left} y={14} aria-hidden="true">
              How {total} followed recommendations ended
            </text>
            <g aria-hidden="true">
              {segs.map((s) => (
                <g key={s.key} onPointerEnter={() => setHot({ x: s.x + s.w / 2, y: 22, title: s.label, rows: [[String(count(s.key)), `of ${total} trades`]] })} onPointerLeave={() => setHot(null)}>
                  <rect className="seg" x={s.x} y={22} width={Math.max(0, s.w)} height={28} rx={4} fill={s.color} />
                </g>
              ))}
              {total === 0 && <rect className="hbar" x={m.left} y={22} width={innerW} height={28} rx={4} fill="var(--surface-3)" />}
            </g>
            <line className="grid vert strong" x1={bx(0)} x2={bx(0)} y1={64} y2={106} />
            {barRow(66, outcomes.avg_win_pct, "Average win", "var(--pos)", `· ${outcomes.won}`)}
            {barRow(88, outcomes.avg_loss_pct, "Average loss", "var(--neg)", `· ${outcomes.lost}`)}
            {shownFamilies.length > 0 && (
              <g>
                <text className="row-sub" x={m.left} y={famTop - 4} aria-hidden="true">
                  {byForward ? `Mean excess over ${benchmarkName} ${HEADLINE_SESSIONS} sessions after the decision day, by strategy family` : `Mean excess over ${benchmarkName} by strategy family`}
                </text>
                <line className="grid vert strong" x1={famX(0)} x2={famX(0)} y1={famTop + 2} y2={famTop + 4 + shownFamilies.length * 22} />
                {shownFamilies.map((f, i) => {
                  const fwd = byForward ? forwardOf(f, HEADLINE_SESSIONS) : null;
                  const v = valueOf(f) ?? 0;
                  const n = fwd ? fwd.n : f.n;
                  const small = fwd ? fwd.small_sample : f.small_sample;
                  const y = famTop + 6 + i * 22;
                  const color = small || fwd?.inside_chance ? "var(--neutral)" : v >= 0 ? "var(--pos)" : "var(--neg)"; // no colour of success or failure for a small sample or a mean that chance alone gives
                  const rows: [string, string][] = fwd
                    ? [[points(fwd.mean_excess_pct), `against ${benchmarkName} after ${fwd.sessions} sessions`], [`se ${points(fwd.se_pct)}`, fwd.inside_chance ? "within chance" : "beyond chance"], [`n=${fwd.n}`, small ? "small sample" : "enough"]]
                    : [[points(f.mean_excess_pct), `against ${benchmarkName}`], [`n=${f.n}`, f.small_sample ? "small sample" : "enough"]];
                  return (
                    <g key={f.family} onPointerEnter={() => setHot({ x: famX(v), y, title: f.family.replace(/_/g, " "), rows })} onPointerLeave={() => setHot(null)}>
                      <text className="row-label" x={m.left} y={y + 9}>
                        {f.family.replace(/_/g, " ")}
                      </text>
                      <rect className="hbar" x={Math.min(famX(0), famX(v))} y={y} width={Math.max(2, Math.abs(famX(v) - famX(0)))} height={12} rx={3} fill={color} />
                      <text className="mean-text" x={m.left + innerW} y={y + 10} textAnchor="end">
                        {points(v)} · n={n}
                        {small ? "*" : ""}
                      </text>
                    </g>
                  );
                })}
              </g>
            )}
          </>
        );
        const overlay = hot && (
          <Tooltip x={hot.x} y={hot.y} plotWidth={width} title={hot.title}>
            {hot.rows.map(([v, name]) => (
              <TipRow key={name} value={v} name={name} />
            ))}
          </Tooltip>
        );
        return { svg, overlay };
      }}
    />
  );
}
