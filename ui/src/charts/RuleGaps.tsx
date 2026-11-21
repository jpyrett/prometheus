// "Does each rule of the Critic earn its place?" (Critic v2 rule C1, `design/critic_v2_evidence_spec.md` section 6): one row for each finding code of the Critic's reviews, a dot at the GAP (the mean excess return over SPY of the
// recommendations the rule fired on, minus the mean of all the others) with its two-standard-error bar, on ONE axis, zero marked. A gap under zero means the rule picks losers, which is what a denying rule is for; over zero, it denies winners.
// The colour is the server's reading, never ours (P12): GREEN only for a rule that picks losers beyond chance, RED only for one that denies winners beyond chance, GREY inside chance, with too few outcomes, and for a suitability rule
// (the user's own risk profile: listed and greyed, never judged by returns). The table twin has every figure.
import { useMemo, useState } from "react";
import { points } from "../format";
import type { CriticReading } from "../proving/types";
import { XAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { extent, linear, niceDomain, niceTicks } from "./scale";
import { TipRow, Tooltip } from "./Tooltip";

/** One row as the chart draws it: prepared by the card from a rule's row, or from its cell of one theme. */
export interface RuleGapRow {
  key: string;
  label: string;
  sub: string;
  kind: string;
  suitability: boolean;
  /** Recommendations with an excess return in the group the rule fired on. */
  n: number;
  fired: number;
  closed: number;
  won: number;
  hitMean: number | null;
  restMean: number | null;
  gap: number | null;
  se: number | null;
  inside: boolean | null;
  reading: CriticReading;
  sentence: string;
}

const COLOR: Record<CriticReading, string> = { picks_losers: "var(--pos)", denies_winners: "var(--neg)", no_clear_difference: "var(--neutral)", too_few: "var(--neutral)", suitability: "var(--neutral)" };
const READING: Record<CriticReading, string> = { picks_losers: "picks losers", denies_winners: "denies winners", no_clear_difference: "no clear difference", too_few: "too few to say", suitability: "suitability: not judged" };

export interface RuleGapsProps {
  rows: RuleGapRow[];
  band?: number;
  benchmarkName?: string;
  fallbackWidth?: number;
}

export function RuleGaps({ rows, band = 2, benchmarkName = "SPY", fallbackWidth = 640 }: RuleGapsProps) {
  const [hot, setHot] = useState<{ row: RuleGapRow; x: number; y: number } | null>(null);
  const spans = useMemo(() => rows.flatMap((r) => (r.gap === null || r.se === null ? [] : [r.gap - band * r.se, r.gap + band * r.se, r.gap])), [rows, band]);

  const table: TableSpec = {
    caption: `Gap in excess return over ${benchmarkName}: what each rule of the Critic fired on, minus the others`,
    columns: ["Rule", "Kind", "Fired", "Won", "Mean (fired)", "Mean (others)", "Gap", `± ${band} se`, "Reading"],
    numeric: [2, 3, 4, 5, 6, 7],
    rows: rows.map((r) => [
      r.label,
      r.suitability ? "suitability" : r.kind,
      r.fired,
      r.closed ? `${r.won}/${r.closed}` : "—",
      points(r.hitMean),
      points(r.restMean),
      r.suitability ? "—" : points(r.gap),
      r.suitability || r.se === null ? "—" : points(band * r.se),
      READING[r.reading],
    ]),
  };
  const judged = rows.filter((r) => !r.suitability && r.gap !== null);
  const label = `Rules of the Critic against ${benchmarkName}. ${rows.length} rows. ${["picks_losers", "denies_winners", "no_clear_difference", "too_few", "suitability"].map((k) => `${rows.filter((r) => r.reading === k).length} ${READING[k as CriticReading]}`).join(", ")}. ${judged
    .slice(0, 6)
    .map((r) => `${r.label}: gap ${points(r.gap)}`)
    .join(". ")}.`;

  return (
    <ChartFrame
      className="rulegaps"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => 30 + rows.length * 34 + 52}
      legend={
        <>
          <Key kind="dot" color="var(--pos)">
            Picks losers
          </Key>
          <Key kind="dot" color="var(--neg)">
            Denies winners
          </Key>
          <Key kind="dot" color="var(--neutral)">
            Inside chance, too few, or not judged
          </Key>
        </>
      }
      draw={(width, height) => {
        const m = { left: 206, right: 18, top: 8, bottom: 46 };
        const innerW = Math.max(80, width - m.left - m.right);
        const ex = extent([0, ...spans]) ?? [-0.05, 0.05];
        const [d0, d1] = niceDomain(ex[0] - 0.005, ex[1] + 0.005, 5);
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
              gap in excess return over {benchmarkName}, in points: fired minus the others (left of zero: the rule picks losers)
            </text>
            {rows.map((r, i) => {
              const cy = m.top + rowH * i + rowH / 2;
              const color = COLOR[r.reading];
              const draws = !r.suitability && r.gap !== null && r.se !== null;
              return (
                <g key={r.key} className={`rule-row${r.suitability ? " greyed" : ""}`} data-reading={r.reading} data-key={r.key}>
                  {i > 0 && <line className="grid" x1={4} x2={m.left + innerW} y1={m.top + rowH * i} y2={m.top + rowH * i} />}
                  <text className="row-label" x={m.left - 12} y={cy - 5} textAnchor="end">
                    {r.label}
                  </text>
                  <text className="row-sub" x={m.left - 12} y={cy + 9} textAnchor="end">
                    {r.sub}
                  </text>
                  {draws ? (
                    <g onPointerEnter={() => setHot({ row: r, x: x(r.gap!), y: cy })} onPointerLeave={() => setHot(null)}>
                      <line className="gap-bar" x1={x(r.gap! - band * r.se!)} x2={x(r.gap! + band * r.se!)} y1={cy} y2={cy} stroke={color} />
                      <line className="gap-cap" x1={x(r.gap! - band * r.se!)} x2={x(r.gap! - band * r.se!)} y1={cy - 4} y2={cy + 4} stroke={color} />
                      <line className="gap-cap" x1={x(r.gap! + band * r.se!)} x2={x(r.gap! + band * r.se!)} y1={cy - 4} y2={cy + 4} stroke={color} />
                      <circle className="gap-dot" cx={x(r.gap!)} cy={cy} r={5} fill={color} />
                    </g>
                  ) : (
                    <text className="row-sub" x={m.left + 8} y={cy + 4}>
                      {r.suitability ? "not judged by returns" : "no gap to draw"}
                    </text>
                  )}
                </g>
              );
            })}
          </>
        );
        const overlay = hot && (
          <Tooltip x={hot.x} y={hot.y - 10} plotWidth={width} title={hot.row.label}>
            <TipRow value={points(hot.row.gap)} name="gap (fired minus the others)" />
            <TipRow value={`± ${points(band * (hot.row.se ?? 0))}`} name={`${band} standard errors`} />
            <TipRow value={`${hot.row.n} / ${hot.row.fired}`} name="with an outcome / fired" />
            <TipRow value={READING[hot.row.reading]} name="reading" />
          </Tooltip>
        );
        return { svg, overlay };
      }}
    />
  );
}
