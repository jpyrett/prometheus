// The equity chart (spec 13.2, the centrepiece): the desk's equity as a line with a soft wash above and below the starting cash, SPY as a blue line, the starting cash as a dashed baseline,
// a cash-versus-invested band, an underwater strip (drawdown) on the same x scale, a buy as an up-triangle and an exit as a dot coloured by its reason, the decision days as ticks on the x axis,
// and, above the plot, the research strip: one tile per decision day that fills as the day's research comes back. Every number is drawn as the server sent it (P12).
// Motion: the playhead (ember) sweeps while the points are being drawn; everything else is still, and the chart is complete without any motion.
import { useMemo, useState, type ReactNode } from "react";
import { moneyAxis, money as moneyOf, pct, shares as shareText } from "../format";
import { secondsText as sec, stagesOf } from "../proving/timing";
import type { CurveEvent, DayStatus, EquityPoint } from "../proving/types";
import { useSvgId, XAxis, YAxis } from "./Axis";
import { ChartFrame, Key, type TableSpec } from "./ChartFrame";
import { Crosshair, useScrub } from "./Crosshair";
import { areaPath, linePath, n, triangleUp } from "./path";
import { dayNumber, extent, linear, niceDomain, niceTicks, shortDate, timeTicks } from "./scale";
import { Tooltip, TipRow } from "./Tooltip";

export type TileState = "pending" | "busy" | "done" | "cached" | "failed";

/** One decision day on the x axis and on the research strip. */
export interface DayMark {
  day: string;
  state: TileState;
  /** The walk's own status of the day, once it has been walked. */
  status?: DayStatus | null;
  recommended: number;
  approved: number;
  denied: number;
  traded: number;
  reason?: string | null;
  /** The day's desk (the work-order view) is there to open. */
  hasDesk: boolean;
  /** The seconds the day's research took, and the split of them by desk stage (a cached day has none). */
  seconds?: number | null;
  stages?: Record<string, number>;
  /** Few articles were stored for the day's companies (the caveat news_coverage_thin), and how many were found. */
  thinNews?: boolean;
  articles?: number | null;
}

export interface EquityChartProps {
  /** Every point received so far. */
  points: EquityPoint[];
  /** How many of them the playhead has drawn. */
  shown: number;
  /** The playhead is moving (the points are being drawn). */
  sweeping: boolean;
  events: CurveEvent[];
  days: DayMark[];
  /** Draw the research strip above the plot. */
  research: boolean;
  startCash: number;
  /** The x axis: the plan's range (drawn before a single point exists). */
  domain: { first: string; last: string } | null;
  benchmarkName?: string;
  /** Dollars (a Time Machine run) or an index that starts at 100 (the Track Record). */
  unit?: "money" | "index";
  /** Draw the underwater strip (a curve with no drawdown figures, the Live view's history, has none). */
  underwater?: boolean;
  /** The words of the starting-cash baseline's key and label, when it is not "starting cash". */
  baselineWord?: string;
  /** What the main line is called ("The desk"; the Live view's is "The account"). */
  seriesName?: string;
  /** The worst drawdown the server computed, for the chart's accessible name. */
  maxDrawdown?: number | null;
  selectedDay?: string | null;
  onOpenDay?: (day: string) => void;
  status?: ReactNode;
  toolbar?: ReactNode;
  fallbackWidth?: number;
}

const REASON_COLOR: Record<string, string> = { target: "var(--pos)", stop: "var(--neg)", time_stop: "var(--neutral)" };
const REASON_WORD: Record<string, string> = { target: "target", stop: "stop", time_stop: "time stop" };

/** The words of a tile's time: the day's seconds and its split by stage ("73.3 s: Data Clerk 38.5 s, Advisor 12.4 s, Critic 22.2 s"); nothing for a cached or waiting day. */
export function tileTimeWords(mark: DayMark): string {
  if (mark.state === "pending" || mark.state === "busy" || mark.state === "cached" || !mark.seconds) return "";
  const parts = stagesOf(mark.stages).filter((s) => s.seconds >= 0.05);
  return `${sec(mark.seconds)}${parts.length ? `: ${parts.map((s) => `${s.label} ${sec(s.seconds)}`).join(", ")}` : ""}`;
}

export function tileLabel(mark: DayMark): string {
  const date = shortDate(mark.day, true);
  const thin = mark.thinNews ? `; few stored articles${mark.articles != null ? ` (${mark.articles})` : ""}` : "";
  switch (mark.state) {
    case "pending":
      return `${date}: waiting to be researched`;
    case "busy":
      return `${date}: being researched`;
    case "failed":
      return `${date}: the research failed${mark.reason ? ` (${mark.reason})` : ""}`;
    default: {
      const time = tileTimeWords(mark);
      return `${date}: ${mark.approved} of ${mark.recommended} recommended stocks approved${mark.state === "cached" ? ", from the cache" : ""}${mark.traded ? `, ${mark.traded} traded` : ""}${time ? `, researched in ${time}` : ""}${thin}`;
    }
  }
}

export function tickLabel(mark: DayMark): string {
  return `${shortDate(mark.day, true)}: decision day${mark.hasDesk ? ", open its desk" : ""}`;
}

export function EquityChart({ points, shown, sweeping, events, days, research, startCash, domain, benchmarkName = "SPY", unit = "money", underwater = true, baselineWord, seriesName, maxDrawdown, selectedDay, onOpenDay, status, toolbar, fallbackWidth }: EquityChartProps) {
  const id = useSvgId("eq");
  const index = unit === "index";
  const money = index ? (v: number | null | undefined) => (v === null || v === undefined ? "\u2014" : v.toFixed(2)) : moneyOf;
  const axisMoney = index ? (v: number) => String(Math.round(v * 10) / 10) : moneyAxis;
  const startWord = baselineWord ? `${baselineWord} ${money(startCash)}` : index ? "100 at the first entry" : `starting cash ${moneyOf(startCash)}`;
  const visible = useMemo(() => points.slice(0, Math.max(0, Math.min(points.length, shown))), [points, shown]);

  const table: TableSpec = useMemo(() => {
    const byDate = new Map<string, CurveEvent[]>();
    for (const e of events) byDate.set(e.date, [...(byDate.get(e.date) ?? []), e]);
    return {
      caption: `Equity by day: the desk against ${benchmarkName}`,
      columns: ["Date", "Desk equity", benchmarkName, "Cash", "Invested", "Drawdown", "Trades"],
      numeric: [1, 2, 3, 4, 5],
      rows: points.map((p) => [
        p.date,
        money(p.equity),
        p.benchmark === null ? "—" : money(p.benchmark),
        money(p.cash),
        money(p.invested),
        pct(p.drawdown_pct),
        (byDate.get(p.date) ?? []).map((e) => `${e.kind === "buy" ? "bought" : `sold (${REASON_WORD[e.reason ?? ""] ?? e.reason ?? "exit"})`} ${e.symbol} ${shareText(e.shares)} sh`).join("; "),
      ]),
    };
  }, [points, events, benchmarkName]);

  const first = points[0];
  const last = points[points.length - 1];
  const label = last
    ? `${index ? "Index chart. $1 put into every traded recommendation went from 100" : `Equity chart. The desk's account went from ${money(startCash)}`} to ${money(last.equity)} by ${last.date}${last.benchmark !== null ? `, ${benchmarkName} to ${money(last.benchmark)}` : ""}${maxDrawdown !== null && maxDrawdown !== undefined ? `. Deepest drawdown ${pct(maxDrawdown)}` : ""}. ${events.length} trades marked.`
    : `Equity chart. No trading day has been drawn yet${first ? "" : "; the decision days of the plan are on the axis"}.`;

  const legend = (
    <>
      <Key color="var(--text)">{seriesName ?? (index ? "Traded recommendations" : "The desk")}</Key>
      {points.some((p) => p.benchmark !== null) && <Key color="var(--info)">{benchmarkName}</Key>}
      <Key color="var(--text-3)" dashed>
        {startWord.charAt(0).toUpperCase() + startWord.slice(1)}
      </Key>
      {events.length > 0 && (
        <>
          <Key kind="triangle" color="var(--text)">
            Buy
          </Key>
          <Key kind="dot" color="var(--pos)">
            Target
          </Key>
          <Key kind="dot" color="var(--neg)">
            Stop
          </Key>
          <Key kind="dot" color="var(--neutral)">
            Time stop
          </Key>
        </>
      )}
    </>
  );

  return (
    <ChartFrame
      className="equity"
      label={label}
      table={table}
      legend={legend}
      status={status}
      toolbar={toolbar}
      fallbackWidth={fallbackWidth}
      height={(w) => (w < 640 ? 360 : 440)}
      draw={(width, height) => {
        const m = { left: 60, right: 18, top: 8 };
        const innerW = Math.max(60, width - m.left - m.right);
        const narrow = width < 640;
        const stripH = research ? 40 : 0;
        const stripGap = research ? 10 : 0;
        const bandH = index ? 0 : 12;
        const uwH = !underwater ? 0 : narrow ? 46 : 64;
        const axisH = 46;
        const mainTop = m.top + stripH + stripGap;
        const mainH = Math.max(90, height - mainTop - 16 - bandH - 10 - uwH - (underwater ? 6 : 0) - axisH);
        const mainBottom = mainTop + mainH;
        const bandTop = mainBottom + (index ? 0 : 16);
        const uwTop = bandTop + bandH + (index ? 14 : 10);
        const uwBottom = uwTop + uwH;
        const axisY = uwBottom + (underwater ? 6 : 0);

        // x: the plan's range, else what has been drawn; the plot is fixed from the first moment so the curve grows into it
        const dayNumbers = points.map((p) => dayNumber(p.date));
        const first0 = domain ? dayNumber(domain.first) : (dayNumbers[0] ?? 0);
        const last0 = domain ? dayNumber(domain.last) : (dayNumbers[dayNumbers.length - 1] ?? first0 + 1);
        const markDays = days.map((d) => dayNumber(d.day));
        const lo = Math.min(first0, ...markDays, ...(dayNumbers.length ? [dayNumbers[0]!] : []));
        const hi = Math.max(last0, ...markDays, ...(dayNumbers.length ? [dayNumbers[dayNumbers.length - 1]!] : []), lo + 1);
        const x = linear([lo, hi], [m.left, m.left + innerW]);

        // y: the equity of the desk and of the benchmark, the starting cash always inside
        const range = extent([startCash, ...points.map((p) => p.equity), ...points.map((p) => p.benchmark)]) ?? [startCash, startCash];
        const spread = range[1] - range[0] || startCash * 0.02;
        const [d0, d1] = niceDomain(range[0] - spread * 0.08, range[1] + spread * 0.08, 4);
        const y = linear([d0, d1], [mainBottom, mainTop]);
        const yTicks = niceTicks(d0, d1, 4);
        // a flat start (the first days of a walk) has ticks a few dollars apart: they are written in whole dollars, not as $10k three times
        const axisFormat = d1 - d0 < 2000 && !index ? (v: number) => `$${Math.round(v).toLocaleString("en-US")}` : axisMoney;
        const baseY = y(startCash);

        const worst = Math.min(0, ...points.map((p) => p.drawdown_pct));
        const dd = linear([Math.min(worst, -0.01) * 1.05, 0], [uwBottom, uwTop]);
        const ddTicks = worst < -0.005 ? [0, worst] : [0]; // a second tick only when it reads as something other than 0%

        const xy = (v: (p: EquityPoint) => number | null) => visible.map((p) => [x(dayNumber(p.date)), v(p)] as const).filter((q): q is readonly [number, number] => q[1] !== null).map(([px, v2]) => [px, y(v2)] as const);
        const deskXY = xy((p) => p.equity);
        const benchXY = xy((p) => p.benchmark);
        const lastShown = visible[visible.length - 1];
        const lastX = lastShown ? x(dayNumber(lastShown.date)) : null;

        const investedXY = visible.map((p) => [x(dayNumber(p.date)), bandTop + bandH - (p.equity > 0 ? Math.min(1, p.invested / p.equity) : 0) * bandH] as const);
        const ddXY = visible.map((p) => [x(dayNumber(p.date)), dd(p.drawdown_pct)] as const);

        const ticks = timeTicks(lo, hi, narrow ? 4 : 9);

        // the events: markers on the curve. Several buys of one day are ONE triangle under the curve with their count, and the exits of one day for one reason are one dot on it, so a dense curve stays readable;
        // the tooltip and the table list each trade.
        const lastDate = lastShown?.date ?? "";
        const byDate = new Map(visible.map((p) => [p.date, p]));
        const grouped = new Map<string, { date: string; kind: CurveEvent["kind"]; reason: string | null; count: number }>();
        for (const e of events) {
          if (e.date > lastDate || !byDate.has(e.date)) continue;
          const key = `${e.date}|${e.kind}|${e.kind === "exit" ? (e.reason ?? "") : ""}`;
          const g = grouped.get(key);
          if (g) g.count += 1;
          else grouped.set(key, { date: e.date, kind: e.kind, reason: e.kind === "exit" ? e.reason : null, count: 1 });
        }
        const exitsOnDay = new Map<string, number>();
        const markers = [...grouped.values()].map((g) => {
          const k = g.kind === "exit" ? (exitsOnDay.get(g.date) ?? 0) : 0;
          if (g.kind === "exit") exitsOnDay.set(g.date, k + 1);
          return { ...g, k, p: byDate.get(g.date)! };
        });

        // the crosshair
        const scrubXs = points.map((p) => x(dayNumber(p.date)));
        const eventsOn = (date: string) => events.filter((e) => e.date === date);
        const describe = (i: number) => {
          const p = points[i]!;
          return `${shortDate(p.date, true)}: desk ${money(p.equity)}${p.benchmark !== null ? `, ${benchmarkName} ${money(p.benchmark)}` : ""}${eventsOn(p.date).length ? `, ${eventsOn(p.date).length} trade${eventsOn(p.date).length === 1 ? "" : "s"}` : ""}`;
        };
        const scrub = useScrub({ xs: scrubXs, left: m.left, describe, label: "Move along the equity chart with the arrow keys" });
        const at = scrub.index === null || scrub.index >= visible.length ? null : points[scrub.index]!;

        // the research strip: a tile shows its approved-over-recommended and its seconds; on hover or focus it opens the split of those seconds by stage
        const [tipDay, setTipDay] = useState<string | null>(null);
        const spacing = days.length > 1 ? Math.min(...days.slice(1).map((d, i) => x(dayNumber(d.day)) - x(dayNumber(days[i]!.day))).filter((v) => v > 0), 1e9) : 60;
        const tileW = Math.max(5, Math.min(44, spacing - 3));
        const showText = tileW >= 26;

        const svg = (
          <>
            <defs>
              <clipPath id={`${id}a`}>
                <rect x={m.left} y={mainTop - 2} width={innerW} height={Math.max(0, baseY - mainTop + 2)} />
              </clipPath>
              <clipPath id={`${id}b`}>
                <rect x={m.left} y={baseY} width={innerW} height={Math.max(0, mainBottom - baseY + 2)} />
              </clipPath>
            </defs>

            {research && (
              <g className="research" role="group" aria-label="Research of each decision day">
                {days.map((d) => {
                  const cx = x(dayNumber(d.day));
                  const text = d.state === "failed" ? "!" : d.state === "done" || d.state === "cached" ? `${d.approved}/${d.recommended}` : "";
                  const sub = d.state === "done" && d.seconds ? `${Math.round(d.seconds)} s` : d.state === "cached" && tileW >= 36 ? "cache" : "";
                  const open = d.hasDesk && onOpenDay;
                  return (
                    <g
                      key={d.day}
                      className={`rs-tile ${d.state}${(d.state === "done" || d.state === "cached") && d.approved === 0 ? " zero" : ""}${selectedDay === d.day ? " sel" : ""}${open ? " open" : ""}${d.thinNews ? " thin" : ""}`}
                      transform={`translate(${n(cx - tileW / 2)} ${m.top})`}
                      role={open ? "button" : "img"}
                      tabIndex={open ? 0 : undefined}
                      aria-label={tileLabel(d)}
                      onClick={open ? () => onOpenDay(d.day) : undefined}
                      onKeyDown={open ? (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpenDay(d.day)) : undefined}
                      onPointerEnter={() => setTipDay(d.day)}
                      onPointerLeave={() => setTipDay((cur) => (cur === d.day ? null : cur))}
                      onFocus={() => setTipDay(d.day)}
                      onBlur={() => setTipDay((cur) => (cur === d.day ? null : cur))}
                    >
                      <rect className="rs-box" width={tileW} height={stripH} rx={showText ? 6 : 3} />
                      {d.state === "cached" && <circle className="rs-cache" cx={tileW - 5} cy={5} r={2.6} />}
                      {d.thinNews && <rect className="rs-thin" x={4} y={stripH - 4} width={Math.max(2, tileW - 8)} height={2} rx={1} />}
                      {showText && (
                        <>
                          <text className="rs-text" x={tileW / 2} y={sub ? 15 : stripH / 2} dy={sub ? undefined : "0.34em"} textAnchor="middle">
                            {text}
                          </text>
                          {sub && (
                            <text className="rs-sub" x={tileW / 2} y={28} textAnchor="middle">
                              {sub}
                            </text>
                          )}
                        </>
                      )}
                    </g>
                  );
                })}
              </g>
            )}

            <YAxis scale={y} ticks={yTicks} x={m.left} right={m.left + innerW} format={axisFormat} />
            <line className="baseline" x1={m.left} x2={m.left + innerW} y1={baseY} y2={baseY} />

            {deskXY.length > 1 && (
              <>
                <path className="wash pos" d={areaPath(deskXY, baseY)} clipPath={`url(#${id}a)`} />
                <path className="wash neg" d={areaPath(deskXY, baseY)} clipPath={`url(#${id}b)`} />
              </>
            )}
            {deskXY.length === 1 && <circle className="solo" cx={deskXY[0]![0]} cy={deskXY[0]![1]} r={4.5} />}
            {benchXY.length > 1 && <path className="line bench" d={linePath(benchXY)} />}
            {deskXY.length > 1 && <path className="line desk" d={linePath(deskXY)} />}

            {markers.map((g, i) => {
              const cx = x(dayNumber(g.date)) + (g.k % 2 === 0 ? 1 : -1) * Math.ceil(g.k / 2) * 11;
              const cy = y(g.p.equity);
              return g.kind === "buy" ? (
                <g key={i} className="mk-group" data-count={g.count}>
                  <path className="mk buy" d={triangleUp(cx, cy + 10, 11)} />
                  {g.count > 1 && (
                    <text className="mk-count" x={cx} y={cy + 28} textAnchor="middle">
                      ×{g.count}
                    </text>
                  )}
                </g>
              ) : (
                <g key={i} className="mk-group" data-count={g.count}>
                  <circle className="mk exit" cx={cx} cy={cy} r={5} fill={REASON_COLOR[g.reason ?? ""] ?? "var(--neutral)"} />
                  {g.count > 1 && (
                    <text className="mk-count" x={cx} y={cy - 9} textAnchor="middle">
                      ×{g.count}
                    </text>
                  )}
                </g>
              );
            })}

            {!index && (
            <g className="band" aria-hidden="true">
              <rect className="band-track" x={m.left} y={bandTop} width={innerW} height={bandH} rx={3} />
              {investedXY.length > 1 && <path className="band-fill" d={areaPath(investedXY, bandTop + bandH)} />}
              <text className="tick-label" x={m.left - 8} y={bandTop + bandH / 2} dy="0.32em" textAnchor="end">
                invested
              </text>
            </g>
            )}

            {underwater && (
            <g className="underwater" aria-hidden="true">
              <YAxis scale={dd} ticks={ddTicks} x={m.left} right={m.left + innerW} format={(v) => pct(v, 0, false)} emphasize={0} />
              {ddXY.length > 1 && <path className="wash neg dd" d={areaPath(ddXY, dd(0))} />}
              {ddXY.length > 1 && <path className="line dd" d={linePath(ddXY)} />}
            </g>
            )}

            <XAxis scale={x} ticks={ticks} y={axisY} gridTo={mainTop} />

            <g className="day-ticks" role="group" aria-label="Decision days">
              {days.map((d) => {
                const cx = x(dayNumber(d.day));
                const open = d.hasDesk && onOpenDay;
                return (
                  <g
                    key={d.day}
                    className={`dtick ${d.state}${selectedDay === d.day ? " sel" : ""}${open ? " open" : ""}`}
                    transform={`translate(${n(cx)} ${axisY + 24})`}
                    role={open ? "button" : "img"}
                    tabIndex={open ? 0 : undefined}
                    aria-label={tickLabel(d)}
                    onClick={open ? () => onOpenDay(d.day) : undefined}
                    onKeyDown={open ? (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpenDay(d.day)) : undefined}
                  >
                    <rect className="hit" x={-9} y={-10} width={18} height={20} />
                    <circle className="dot" r={3.6} />
                  </g>
                );
              })}
            </g>

            {sweeping && lastX !== null && lastShown && (
              <g className="pv-playhead" aria-hidden="true">
                <line x1={lastX} x2={lastX} y1={mainTop} y2={uwBottom} />
                <circle cx={lastX} cy={y(lastShown.equity)} r={5} />
              </g>
            )}

            {at && scrub.index !== null && (
              <Crosshair
                x={x(dayNumber(at.date))}
                top={mainTop}
                bottom={uwBottom}
                dots={[
                  { y: y(at.equity), color: "var(--text)" },
                  ...(at.benchmark !== null ? [{ y: y(at.benchmark), color: "var(--info)" }] : []),
                ]}
              />
            )}
          </>
        );

        const tipMark = research && tipDay ? days.find((d) => d.day === tipDay) : undefined;
        const overlay = (
          <>
            {tipMark && <TileTip mark={tipMark} x={Math.max(110, Math.min(width - 110, x(dayNumber(tipMark.day))))} y={m.top + stripH + 8} plotWidth={width} />}
            <div className="chart-probe" style={{ left: m.left, top: mainTop, width: innerW, height: uwBottom - mainTop }} {...scrub.bind} />
            {at && (
              <Tooltip x={x(dayNumber(at.date))} y={y(at.equity) - 10} plotWidth={width} title={shortDate(at.date, true)}>
                <TipRow color="var(--text)" value={money(at.equity)} name={index ? "traded recommendations" : "the desk"} />
                {at.benchmark !== null && <TipRow color="var(--info)" value={money(at.benchmark)} name={benchmarkName} />}
                {!index && <TipRow value={`${money(at.cash)} / ${money(at.invested)}`} name="cash / invested" />}
                {at.drawdown_pct < 0 && <TipRow color="var(--neg)" value={pct(at.drawdown_pct)} name="below its peak" />}
                {eventsOn(at.date).map((e, i) => (
                  <div className="tip-event" key={i}>
                    {e.kind === "buy" ? "Bought" : `Sold at the ${REASON_WORD[e.reason ?? ""] ?? "exit"}:`} <b>{e.symbol}</b> {shareText(e.shares)} sh at {money(e.price)}
                  </div>
                ))}
                {at.stale.length > 0 && <div className="tip-event muted">no bar today for {at.stale.join(", ")}: marked at the last close</div>}
              </Tooltip>
            )}
          </>
        );
        return { svg, overlay };
      }}
    />
  );
}

/** What a research tile opens on hover or focus: the day's seconds as one bar split by desk stage in the agents' own hues, the stages as rows, and what the day is (cached, few articles). Aria-hidden like every tooltip: the tile's label says the same. */
function TileTip({ mark, x, y, plotWidth }: { mark: DayMark; x: number; y: number; plotWidth: number }) {
  const stages = stagesOf(mark.stages).filter((s) => s.seconds >= 0.05);
  const total = stages.reduce((a, s) => a + s.seconds, 0);
  if (mark.state === "pending" || mark.state === "busy") return null;
  return (
    <Tooltip x={x} y={y} plotWidth={plotWidth} placement="below" title={`${shortDate(mark.day, true)} · ${mark.state === "cached" ? "from the cache" : mark.state === "failed" ? "the research failed" : "researched"}`}>
      {mark.state === "cached" && <div className="tip-event">Nothing was asked of a model: the research of this day was stored by an earlier run.</div>}
      {mark.state !== "cached" && stages.length > 0 && (
        <>
          <div className="stagebar" aria-hidden="true">
            {stages.map((s) => (
              <span key={s.key} style={{ flexGrow: s.seconds, background: s.color }} />
            ))}
          </div>
          {stages.map((s) => (
            <TipRow key={s.key} color={s.color} value={sec(s.seconds)} name={s.label} />
          ))}
        </>
      )}
      {mark.seconds ? <TipRow value={sec(mark.seconds)} name={total > 0 ? "the whole day" : "the day"} /> : null}
      {mark.thinNews && <div className="tip-event">Few articles were stored for this day{mark.articles != null ? ` (${mark.articles})` : ""}: the research saw less than it would have live.</div>}
    </Tooltip>
  );
}
