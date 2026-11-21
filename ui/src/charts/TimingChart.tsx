// "How long it took" (spec 7.6 and 13.2): where the time of a run went, drawn from the server's own measurements (rule P12: the page adds nothing up and works out no share, it draws what it was sent).
//   1. the run's wall time as one bar split into its phases: setup, research (the days at once), walk;
//   2. the work of each desk stage, all days added up (the days ran at once, so this is more than the wall time), in the agents' own hues;
//   3. the time spent in each model, a thin bar each, with its calls and tokens.
// The table twin lists every figure: the phases, the stages, and the per-model table (calls, failures, seconds, tokens).
import { useState } from "react";
import { count, tokens } from "../format";
import { phasesOf, secondsText as sec, stagesOf } from "../proving/timing";
import type { RunTiming } from "../proving/types";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { Tooltip, TipRow } from "./Tooltip";

const PHASE_COLOR = { setup: "var(--text-3)", research: "var(--text-2)", walk: "var(--text)" } as const;
const shortModel = (m: string) => m.split("/").pop() ?? m;

export interface TimingChartProps {
  timing: RunTiming;
  fallbackWidth?: number;
}

export function TimingChart({ timing, fallbackWidth = 760 }: TimingChartProps) {
  const [hot, setHot] = useState<{ x: number; y: number; title: string; rows: [string, string][] } | null>(null);
  const phases = phasesOf(timing);
  const stages = stagesOf(Object.fromEntries(Object.entries(timing.stages ?? {}).map(([k, v]) => [k, v.total])));
  const models = Object.entries(timing.models ?? {}).sort((a, b) => b[1].seconds - a[1].seconds);
  const maxOf = (key: string) => timing.stages?.[key]?.max;

  const table: TableSpec = {
    caption: "How long the run took, by phase, by desk stage and by model",
    columns: ["Part", "Name", "Seconds", "Calls", "Failed", "Tokens in", "Tokens out"],
    numeric: [2, 3, 4, 5, 6],
    rows: [
      ...phases.map((p) => ["Phase", p.label, p.seconds.toFixed(2), "—", "—", "—", "—"]),
      ...stages.map((s) => ["Stage, all days added", s.label, s.seconds.toFixed(2), "—", "—", "—", "—"]),
      ...models.map(([name, m]) => ["Model", name, m.seconds.toFixed(2), count(m.calls), count(m.failed), count(m.tokens_in), count(m.tokens_out)]),
    ],
  };
  const label = `How long the run took. ${phases.map((p) => `${p.label.toLowerCase()} ${sec(p.seconds)}`).join(", ")}.${stages.length ? ` The work of the desk's stages added over the days: ${stages.map((s) => `${s.label} ${sec(s.seconds)}`).join(", ")}.` : ""}${models.length ? ` Models: ${models.map(([n, m]) => `${shortModel(n)} ${sec(m.seconds)} in ${m.calls} calls`).join(", ")}.` : ""}`;

  const rows = (phases.length ? 1 : 0) + (stages.length ? 1 : 0);
  const height = 14 + rows * 66 + (models.length ? 22 + models.length * 24 : 0) + 6;

  return (
    <ChartFrame
      className="timing"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => height}
      legend={
        <>
          {phases.map((p) => (
            <Key key={p.key} kind="swatch" color={PHASE_COLOR[p.key]}>
              {p.label} {sec(p.seconds)}
            </Key>
          ))}
          {stages.map((s) => (
            <Key key={s.key} kind="swatch" color={s.color}>
              {s.label} {sec(s.seconds)}
            </Key>
          ))}
        </>
      }
      draw={(width) => {
        const m = { left: 12, right: 14 };
        const innerW = Math.max(120, width - m.left - m.right);
        const gap = 2;
        const bar = (parts: { key: string; w: number; color: string; title: string; rows: [string, string][] }[], y: number) => {
          let cursor = m.left;
          return parts.map((p) => {
            const x = cursor;
            cursor += p.w + gap;
            return (
              <rect key={p.key} className="tseg" x={x} y={y} width={Math.max(2, p.w)} height={28} rx={4} fill={p.color} onPointerEnter={() => setHot({ x: x + p.w / 2, y: y + 14, title: p.title, rows: p.rows })} onPointerLeave={() => setHot(null)} />
            );
          });
        };
        const split = (values: number[]) => {
          const total = values.reduce((a, v) => a + v, 0);
          const room = innerW - gap * Math.max(0, values.length - 1);
          return values.map((v) => (total > 0 ? (v / total) * room : 0));
        };
        // each block is 66 px: its caption, the 28 px bar and the air under it
        const yPhases = 14;
        const yStages = yPhases + (phases.length ? 66 : 0);
        const yModels = yStages + (stages.length ? 66 : 0);
        const svg = (
          <>
            {phases.length > 0 && (
              <g>
                <text className="row-sub" x={m.left} y={yPhases}>
                  The run, start to finish
                </text>
                {bar(
                  split(phases.map((p) => p.seconds)).map((w, i) => ({ key: phases[i]!.key, w, color: PHASE_COLOR[phases[i]!.key], title: phases[i]!.label, rows: [[sec(phases[i]!.seconds), phases[i]!.key === "research" ? `${timing.parallel_days ?? "the"} days at once` : phases[i]!.key === "walk" ? "the days walked in order" : "agents and data made ready"]] as [string, string][] })),
                  yPhases + 8,
                )}
              </g>
            )}
            {stages.length > 0 && (
              <g>
                <text className="row-sub" x={m.left} y={yStages}>
                  The work of each stage, every day added up
                </text>
                {bar(
                  split(stages.map((s) => s.seconds)).map((w, i) => ({ key: stages[i]!.key, w, color: stages[i]!.color, title: stages[i]!.label, rows: [[sec(stages[i]!.seconds), "of work in all"], ...(maxOf(stages[i]!.key) !== undefined ? ([[sec(maxOf(stages[i]!.key)!), "its longest day"]] as [string, string][]) : [])] as [string, string][] })),
                  yStages + 8,
                )}
              </g>
            )}
            {models.length > 0 && (
              <g>
                <text className="row-sub" x={m.left} y={yModels}>
                  Time in each model
                </text>
                {(() => {
                  const labelW = Math.min(190, width * 0.3);
                  const top = Math.max(...models.map(([, v]) => v.seconds), 0.001);
                  const room = Math.max(40, innerW - labelW - 200);
                  return models.map(([name, mv], i) => {
                    const yy = yModels + 12 + i * 24;
                    const w = Math.max(2, (mv.seconds / top) * room);
                    return (
                      <g key={name} onPointerEnter={() => setHot({ x: m.left + labelW + w, y: yy + 6, title: name, rows: [[sec(mv.seconds), "in the model"], [count(mv.calls), mv.failed ? `calls, ${mv.failed} failed` : "calls"], [`${tokens(mv.tokens_in)} in · ${tokens(mv.tokens_out)} out`, "tokens"]] })} onPointerLeave={() => setHot(null)}>
                        <text className="row-label" x={m.left} y={yy + 10}>
                          {shortModel(name)}
                        </text>
                        <rect className="hbar" x={m.left + labelW} y={yy} width={w} height={12} rx={3} fill="var(--text-2)" />
                        <text className="mean-text" x={m.left + labelW + w + 8} y={yy + 10}>
                          {sec(mv.seconds)} · {count(mv.calls)} calls
                        </text>
                      </g>
                    );
                  });
                })()}
              </g>
            )}
          </>
        );
        const overlay = hot && (
          <Tooltip x={Math.max(60, Math.min(width - 60, hot.x))} y={hot.y} plotWidth={width} title={hot.title}>
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

