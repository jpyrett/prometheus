// The plan ladder of the Live view (spec 13.2): one row per open position, a horizontal track from the stop (red end) to the target (green end), the cost basis as a tick and the price now as a marker that
// moves when a new snapshot arrives. The row's status says in words where the price stands against the plan. A position bought without a plan shows its price against its cost only.
// The view WATCHES: nothing here sells or places anything (rule P11).
import { money, pct, shares as shareText } from "../format";
import type { LivePosition, LiveStatus } from "../proving/types";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { diamond } from "./path";
import { extent, linear, shortDate } from "./scale";

/** The stop in force: a trailing plan's is the server's `stop_now` (the plan's stop or the highest close less its distance, whichever is higher), any other plan's is its stop. */
export function stopOf(p: LivePosition): number | null {
  return (p.plan?.trailing ? p.plan.stop_now ?? p.plan.stop : p.plan?.stop) ?? null;
}

export const STATUS_WORDS: Record<LiveStatus, string> = {
  in_plan: "IN PLAN",
  stop_hit: "STOP HIT",
  target_hit: "TARGET HIT",
  time_stop_passed: "TIME STOP PASSED",
  no_plan: "NO PLAN",
};

const ROW_H = 78;

export function PlanLadder({ positions, fallbackWidth = 760 }: { positions: LivePosition[]; fallbackWidth?: number }) {
  const table: TableSpec = {
    caption: "Open positions against their plans",
    columns: ["Symbol", "Shares", "Cost", "Price now", "Stop", "Target", "Value", "Gain", "Status"],
    numeric: [1, 2, 3, 4, 5, 6, 7],
    rows: positions.map((p) => [p.symbol, shareText(p.shares), money(p.cost_basis), money(p.price), money(stopOf(p)), money(p.plan?.target), money(p.value), p.gain_pct === null ? "—" : `${money(p.gain)} (${pct(p.gain_pct)})`, STATUS_WORDS[p.status]]),
  };
  const label = `Plan ladder. ${positions.length} open position${positions.length === 1 ? "" : "s"}. ${positions.map((p) => `${p.symbol}: ${p.price === null ? "no price" : money(p.price)}, ${stopOf(p) ? `${p.plan?.trailing ? "trailing stop" : "stop"} ${money(stopOf(p))}, ` : ""}${p.plan?.target ? `target ${money(p.plan.target)}, ` : ""}${STATUS_WORDS[p.status].toLowerCase()}`).join("; ")}.`;

  return (
    <ChartFrame
      className="ladder"
      label={label}
      table={table}
      fallbackWidth={fallbackWidth}
      height={() => Math.max(ROW_H, positions.length * ROW_H) + 8}
      legend={
        <>
          <Key color="var(--neg)">Stop</Key>
          <Key color="var(--pos)">Target</Key>
          <Key kind="swatch" color="var(--text-2)">
            Cost
          </Key>
          <Key kind="triangle" color="var(--text)">
            Price now
          </Key>
        </>
      }
      draw={(width) => {
        const left = 150;
        const right = Math.min(150, Math.max(96, width * 0.17));
        const trackL = left + 14;
        const trackR = Math.max(trackL + 60, width - right - 14);
        const svg = (
          <>
            {positions.map((p, i) => {
              const top = 4 + i * ROW_H;
              const mid = top + 40;
              const stop = stopOf(p);
              const trailing = p.plan?.trailing === true;
              const target = p.plan?.target ?? null;
              // with a plan the track is stop to target with room for the price beyond either end; without one it is centred on the cost, wide enough that a price a few cents off the cost sits beside it
              const span = extent([stop, target, p.cost_basis, p.price]) ?? [p.cost_basis, p.cost_basis];
              const planned = (stop !== null && target !== null) || stop !== null; // a stop and a target, or (a company the Advisor scored neutral: Advisor v2) a stop and a time stop and no target
              const reach = Math.max(Math.abs((p.price ?? p.cost_basis) - p.cost_basis) * 1.4, p.cost_basis * 0.05);
              const domain: [number, number] = planned ? [span[0] - ((span[1] - span[0]) * 0.1 || 1), span[1] + ((span[1] - span[0]) * 0.1 || 1)] : [p.cost_basis - reach, p.cost_basis + reach];
              const x = linear(domain, [trackL, trackR]);
              const hit = p.status === "stop_hit" ? "var(--neg)" : p.status === "target_hit" ? "var(--pos)" : "var(--text)";
              return (
                <g key={p.symbol} className={`lad-row ${p.status}`}>
                  {i > 0 && <line className="grid" x1={4} x2={width - 4} y1={top - 4} y2={top - 4} />}
                  <text className="lad-sym" x={8} y={mid - 4}>
                    {p.symbol}
                  </text>
                  <text className="row-sub" x={8} y={mid + 11}>
                    {shareText(p.shares)} sh{p.since ? ` · since ${shortDate(p.since)}` : ""}
                  </text>
                  {stop !== null && target !== null ? (
                    <>
                      <defs>
                        <linearGradient id={`lad${i}`} x1="0" x2="1">
                          <stop offset="0" stopColor="var(--neg)" stopOpacity="0.42" />
                          <stop offset="0.5" stopColor="var(--text-3)" stopOpacity="0.16" />
                          <stop offset="1" stopColor="var(--pos)" stopOpacity="0.42" />
                        </linearGradient>
                      </defs>
                      <rect x={x(stop)} y={mid - 4} width={Math.max(0, x(target) - x(stop))} height={8} rx={4} fill={`url(#lad${i})`} />
                      <line className="lad-cap stop" x1={x(stop)} x2={x(stop)} y1={mid - 11} y2={mid + 11} />
                      <line className="lad-cap target" x1={x(target)} x2={x(target)} y1={mid - 11} y2={mid + 11} />
                      <text className="tick-label" x={x(stop)} y={mid + 25} textAnchor="middle">
                        stop {money(stop)}
                      </text>
                      <text className="tick-label" x={x(target)} y={mid + 25} textAnchor="middle">
                        target {money(target)}
                      </text>
                    </>
                  ) : stop !== null ? (
                    <>
                      <rect x={trackL} y={mid - 2} width={Math.max(0, trackR - trackL)} height={4} rx={2} className="lad-bare" />
                      <line className="lad-cap stop" x1={x(stop)} x2={x(stop)} y1={mid - 11} y2={mid + 11} />
                      <text className="tick-label" x={x(stop)} y={mid + 25} textAnchor="middle">
                        {trailing ? "trailing stop" : "stop"} {money(stop)}
                      </text>
                      <text className="tick-label" x={trackR} y={mid + 25} textAnchor="end">
                        {p.plan?.time_stop_date ? `time stop ${shortDate(p.plan.time_stop_date)}, no target` : "no target"}
                      </text>
                    </>
                  ) : (
                    <>
                      <rect x={trackL} y={mid - 2} width={Math.max(0, trackR - trackL)} height={4} rx={2} className="lad-bare" />
                      <text className="tick-label" x={trackR} y={mid + 25} textAnchor="end">
                        no plan: price against cost only
                      </text>
                    </>
                  )}
                  <line className="lad-cost" x1={x(p.cost_basis)} x2={x(p.cost_basis)} y1={mid - 26} y2={mid + 12} />
                  <text className="tick-label" x={x(p.cost_basis)} y={mid - 31} textAnchor="middle">
                    cost {money(p.cost_basis)}
                  </text>
                  {p.price !== null && (
                    <g className="lad-price" style={{ transform: `translate(${x(p.price)}px, ${mid}px)` }}>
                      <path d={diamond(0, 0, 8)} fill={hit} />
                      <text className="lad-now" y={-16} textAnchor="middle">
                        now {money(p.price)}
                      </text>
                    </g>
                  )}
                  <text className="lad-value" x={width - 8} y={mid - 4} textAnchor="end">
                    {money(p.value)}
                  </text>
                  <text className={`lad-gain ${p.gain !== null && p.gain < 0 ? "neg" : "pos"}`} x={width - 8} y={mid + 11} textAnchor="end">
                    {p.gain === null ? "—" : `${p.gain >= 0 ? "+" : "−"}${money(Math.abs(p.gain))} ${pct(p.gain_pct)}`}
                  </text>
                  <text className={`lad-status ${p.status}`} x={width - 8} y={mid + 26} textAnchor="end">
                    {STATUS_WORDS[p.status]}
                  </text>
                </g>
              );
            })}
          </>
        );
        return { svg };
      }}
    />
  );
}
