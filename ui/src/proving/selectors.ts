// What the Proving Ground says about its state: small pure functions of ProvingState. They pick, order, count what is already there and put it in words; none works out a statistic (P12).
import type { WorkOrderView } from "../api/types";
import type { DayMark } from "../charts/EquityChart";
import { money, pct, points, rate } from "../format";
import { initialRunState, type RunState } from "../state/runReducer";
import type { ProvingState } from "./provingReducer";
import type { NotTradedReason, RecOutcome, Result, Verdict } from "./types";

/** The decision days on the research strip and the x axis: the plan's days, each with what is known of it (a research result, the walk's row, or nothing yet). */
export function dayMarks(s: ProvingState): DayMark[] {
  const walked = new Map(s.decisionDays.map((d) => [d.day, d]));
  const thin = new Set(s.newsThin);
  // The research pool takes the days in order, `parallelDays` at a time: the first days still waiting are the ones being researched now.
  let busyLeft = s.phase === "running" && s.step === "research" ? (s.parallelDays ?? 8) : 0;
  return [...new Set(s.plannedDays)].map((day) => {
    const r = s.research[day];
    const w = walked.get(day);
    // a day's seconds and the split of them by stage: the engine's own result while it comes in; the run's measured timing once it is known (a saved run played again sends 0 seconds in the result)
    const measured = s.timing?.per_day?.[day];
    const seconds = r?.seconds ? r.seconds : (measured?.seconds ?? null);
    const stages = r?.stages && Object.keys(r.stages).length > 0 ? r.stages : (measured?.stages ?? undefined);
    const base = { day, hasDesk: day in s.desks || s.deskDays.includes(day), seconds, stages, thinNews: thin.has(day), articles: s.newsSources[day] ?? null };
    if (r && r.status !== "busy") {
      const failed = r.status === "failed";
      return { ...base, state: failed ? ("failed" as const) : r.cached || r.status === "cached" ? ("cached" as const) : ("done" as const), status: w?.status ?? null, recommended: r.recommended ?? 0, approved: r.approved ?? 0, denied: r.denied ?? 0, traded: w?.traded ?? 0, reason: r.reason ?? w?.reason ?? null };
    }
    if (w) {
      const failed = w.status === "failed";
      return { ...base, state: failed ? ("failed" as const) : w.status === "cached" ? ("cached" as const) : ("done" as const), status: w.status, recommended: w.recommended, approved: w.approved, denied: w.denied, traded: w.traded, reason: w.reason };
    }
    if (r?.status === "busy" || busyLeft > 0) {
      if (r?.status !== "busy") busyLeft -= 1;
      return { ...base, state: "busy" as const, status: null, recommended: 0, approved: 0, denied: 0, traded: 0, reason: null };
    }
    return { ...base, state: "pending" as const, status: null, recommended: 0, approved: 0, denied: 0, traded: 0, reason: null };
  });
}

/** "Researching 26 decision days, 8 at a time: 11 done": a count of the tiles that came back. */
export function researchLine(s: ProvingState): string | null {
  const total = [...new Set(s.plannedDays)].length;
  if (total === 0) return null;
  const done = Object.values(s.research).filter((r) => r.status !== "busy").length;
  const cached = Object.values(s.research).filter((r) => r.cached || r.status === "cached").length;
  if (s.step === "research") return `Researching ${total} decision day${total === 1 ? "" : "s"}${s.parallelDays ? `, ${s.parallelDays} at a time` : ""}: ${done} done${cached ? `, ${cached} from the cache` : ""}`;
  if (s.step === "walk") return `Walking the days in order: the desk trades what each research day found`;
  return null;
}

/** The Desk tab's own components draw a decision day: a finished RunState that holds that day's work-order view. */
export function deskState(view: WorkOrderView): RunState {
  // A decision day is over: an order still "active" in its work order was placed that day and fills at the next open, so it reads PLACED, not as the Desk's busy PLACING…
  const rows = view.rows.map((r) => (r.trade?.status === "active" ? { ...r, trade: { ...r.trade, status: "placed" } } : r));
  return { ...initialRunState, phase: "finished", workOrder: { ...view, rows }, stages: initialRunState.stages.map((s) => ({ ...s, state: "done" as const })) };
}

export const NOT_TRADED_WORDS: Record<NotTradedReason, string> = {
  denied: "denied by the Critic",
  neutral_off: "neutral, and neutral passes are off",
  beyond_count: "beyond the number you asked for",
  no_cash: "no cash was left",
  not_filled: "the order was not filled",
  failed: "the trade failed",
  cooldown: "the Advisor's cooldown held it back",
};

export function notTradedWords(o: RecOutcome): string {
  return `not traded: ${o.not_traded_reason ? NOT_TRADED_WORDS[o.not_traded_reason] : "not bought"}`;
}

export const VERDICT_WORDS: Record<Verdict, string> = { approved: "Approved", denied: "Denied", neutral: "Neutral pass", unreviewed: "Not reviewed" };
export const RESULT_WORDS: Record<Result, string> = { won: "WON", lost: "LOST", open: "OPEN", not_filled: "NOT FILLED", no_data: "NO DATA" };

/** The wall's order: decision day, then the Advisor's rank, then the symbol. */
export function wallOrder(outcomes: RecOutcome[]): RecOutcome[] {
  return [...outcomes].sort((a, b) => a.decision_day.localeCompare(b.decision_day) || (a.advisor?.rank ?? 99) - (b.advisor?.rank ?? 99) || a.symbol.localeCompare(b.symbol));
}

export interface Filters {
  verdict: Verdict | "all";
  result: Result | "all";
  traded: "all" | "traded" | "not_traded";
}
export const NO_FILTERS: Filters = { verdict: "all", result: "all", traded: "all" };

export function filterOutcomes(outcomes: RecOutcome[], f: Filters): RecOutcome[] {
  return outcomes.filter((o) => (f.verdict === "all" || o.verdict === f.verdict) && (f.result === "all" || o.result === f.result) && (f.traded === "all" || (f.traded === "traded" ? o.traded : o.hypothetical)));
}

export interface Kpi {
  key: string;
  label: string;
  /** The value as words, from the server's figure. */
  value: string;
  /** The figure itself, for the count-up. */
  raw: number | null;
  format: (n: number) => string;
  sub: string;
  tone: "pos" | "neg" | "neutral";
  big?: boolean;
  /** A rate or an average over fewer outcomes than the minimum sample: shown with a mark and no colour of success or failure (P9). */
  small?: boolean;
}

/** The five tiles of the hero, from the server's statistics. Null before the run has any. The Track Record has no curve of its own in dollars: its tiles show the index as it ended, SPY's, and the counts. */
export function kpis(s: ProvingState, benchmark = "SPY"): Kpi[] | null {
  const st = s.stats;
  if (!st) return null;
  const small = st.outcomes.small_sample;
  // the server's own counts when it sent the tiles (a saved run's view); else the rows the wall shows
  const rc = s.kpis?.recommendations ?? { total: s.outcomes.length, traded: s.outcomes.filter((o) => o.traded).length, denied: s.outcomes.filter((o) => o.verdict === "denied").length, open: s.outcomes.filter((o) => o.result === "open").length };
  const trades = rc.traded;
  const denied = rc.denied;
  const open = rc.open;
  const win: Kpi = { key: "win", label: "Win rate", value: rate(st.outcomes.win_rate), raw: st.outcomes.win_rate, format: (n) => rate(n), sub: `of ${st.outcomes.closed} closed`, tone: "neutral", small };
  const recs: Kpi = { key: "recs", label: "Recommendations", value: String(rc.total), raw: rc.total, format: (n) => String(Math.round(n)), sub: `${trades} traded \u00b7 ${denied} denied \u00b7 ${open} open`, tone: "neutral" };

  if (s.unit === "index") {
    const last = s.points[s.points.length - 1];
    return [
      { key: "index", label: "Traded recommendations", value: last ? last.equity.toFixed(1) : "\u2014", raw: last?.equity ?? null, format: (n) => n.toFixed(1), sub: "index, 100 at the first entry", tone: "neutral", big: true },
      { key: "bench", label: `${benchmark} over the same days`, value: last?.benchmark != null ? last.benchmark.toFixed(1) : "\u2014", raw: last?.benchmark ?? null, format: (n) => n.toFixed(1), sub: "the same dollars, same entries", tone: "neutral" },
      win,
      { key: "closed", label: "Closed / still open", value: `${st.outcomes.closed} / ${st.outcomes.open}`, raw: null, format: (n) => String(n), sub: `${st.outcomes.not_filled + st.outcomes.no_data} with no entry yet`, tone: "neutral" },
      recs,
    ];
  }

  const pf = st.portfolio;
  const sign = (v: number | null): "pos" | "neg" | "neutral" => (v === null || small || v === 0 ? "neutral" : v > 0 ? "pos" : "neg");
  return [
    { key: "return", label: "Return", value: pf ? pct(pf.return_pct) : "\u2014", raw: pf ? pf.return_pct : null, format: (n) => pct(n), sub: pf ? `${money(pf.start_equity)} to ${money(pf.end_equity)}` : "no curve yet", tone: sign(pf?.return_pct ?? null) },
    {
      key: "excess",
      label: `vs ${benchmark}`,
      value: pf?.excess_return_pct != null ? points(pf.excess_return_pct) : "\u2014",
      raw: pf?.excess_return_pct ?? null,
      format: (n) => points(n),
      sub: pf?.benchmark_return_pct != null ? `${benchmark} ${pct(pf.benchmark_return_pct)}` : "no benchmark curve",
      tone: sign(pf?.excess_return_pct ?? null),
      big: true,
    },
    win,
    { key: "drawdown", label: "Max drawdown", value: pf ? pct(pf.max_drawdown_pct) : "\u2014", raw: pf ? pf.max_drawdown_pct : null, format: (n) => pct(n), sub: pf?.drawdown_from && pf.drawdown_to ? `${pf.drawdown_from} to ${pf.drawdown_to}` : "no curve yet", tone: "neutral" },
    recs,
  ];
}
