// What the page says about the state: small pure functions of RunState (and `now`, which the caller supplies).
import type { WorkRow } from "../api/types";
import { cap, mmss, round6, secs, shares as shareText, signedShares, money } from "../format";
import type { RunState, StageView } from "./runReducer";

export type Tone = "pos" | "neg" | "warn" | "info" | "neutral" | "brand";

/** Seconds to show beside a stage: counting up while it is busy or waiting, final once it is done, nothing before it ran. */
export function stageSeconds(stage: StageView, nowSeconds: number): string | null {
  if ((stage.state === "busy" || stage.state === "waiting") && stage.since !== null) return secs(nowSeconds - stage.since);
  if (stage.state === "done" || stage.state === "failed") return secs(stage.seconds);
  return null;
}

export function busyStage(state: RunState): StageView | undefined {
  return state.stages.find((s) => s.state === "busy" || s.state === "waiting");
}

/** The elapsed time of the run: now - run_started.ts, frozen at run_finished.ts. */
export function elapsedSeconds(state: RunState, nowSeconds: number): number {
  if (state.startedTs === null) return 0;
  const end = state.phase === "finished" && state.finishedTs !== null ? state.finishedTs : nowSeconds;
  return Math.max(0, end - state.startedTs);
}

export interface RunBoxView {
  label: string;
  tone: Tone;
  timer: string;
  line: string;
  spinning: boolean;
}

const FINISHED_LABEL: Record<string, { label: string; tone: Tone }> = {
  traded: { label: "COMPLETED", tone: "pos" },
  waiting: { label: "WAITING", tone: "pos" },
  no_trade: { label: "NOTHING TRADED", tone: "warn" },
  looked: { label: "COMPLETED", tone: "pos" },
  stopped: { label: "STOPPED", tone: "neg" },
  failed: { label: "STOPPED", tone: "neg" },
};

/** One line of what is happening, from the busy stage and the work order ("Critic is reviewing 3 recommendations"). */
export function whatIsHappening(state: RunState): string {
  if (state.question) return "Waiting for your answer";
  const stage = busyStage(state);
  if (!stage) return state.phase === "running" ? "Starting the desk" : "";
  const rows = state.workOrder?.rows ?? [];
  switch (stage.key) {
    case "work_order":
      return "Account Manager is reading your request";
    case "gather":
      return "Data Clerk is gathering the data";
    case "recommend": {
      const found = rows.filter((r) => r.group === "found").length;
      return found ? `Advisor is ranking ${found} companies` : "Advisor is ranking the companies";
    }
    case "critique": {
      const n = rows.filter((r) => r.group === "recommended").length;
      return n ? `Critic is reviewing ${n} recommendation${n === 1 ? "" : "s"}` : "Critic is reviewing the recommendations";
    }
    case "prepare":
      return "Account Manager is deciding what may trade";
    case "trade": {
      const placed = rows.filter((r) => r.decision?.will_trade.startsWith("yes_"));
      const settled = placed.filter((r) => r.trade && (r.trade.status === "filled" || r.trade.status === "rejected")).length;
      return placed.length ? `Trader is placing trade ${Math.min(settled + 1, placed.length)} of ${placed.length}` : "Trader is placing the orders";
    }
  }
}

export function runBoxView(state: RunState, nowSeconds: number): RunBoxView {
  if (state.phase === "idle") return { label: "READY", tone: "neutral", timer: "00:00", line: "Type a request and press Go!", spinning: false };
  const timer = mmss(elapsedSeconds(state, nowSeconds));
  if (state.phase === "finished" && state.result) {
    const f = FINISHED_LABEL[state.result.status] ?? { label: "COMPLETED", tone: "pos" as Tone };
    return { label: f.label, tone: f.tone, timer, line: state.result.headline, spinning: false };
  }
  return { label: state.question ? "WAITING FOR YOU" : "RUNNING", tone: state.question ? "warn" : "brand", timer, line: whatIsHappening(state), spinning: true };
}

/** The Trade column of the work-order table (spec 5.3 (10)). */
export function tradeWords(row: WorkRow, state: RunState): { text: string; tone: Tone } | null {
  const trade = row.trade;
  if (trade?.status === "filled") return { text: "FILLED", tone: "pos" };
  if (trade?.status === "rejected") return { text: "REJECTED", tone: "neg" };
  // an order a past decision day left placed, to fill at the next open (the Proving Ground's day desk): not busy any more, not yet a fill
  if (trade?.status === "placed") return { text: "PLACED", tone: "info" };
  const decision = row.decision;
  if (decision?.will_trade === "pending_user") return { text: "PENDING YOU", tone: "warn" };
  if (decision?.will_trade.startsWith("yes_")) {
    const tradeStage = state.stages.find((s) => s.key === "trade");
    if (trade?.status === "active" || (tradeStage && (tradeStage.state === "busy" || tradeStage.state === "waiting"))) return { text: "PLACING…", tone: "brand" };
    return state.phase === "finished" ? { text: "NO TRADE", tone: "neutral" } : null;
  }
  if (decision) return { text: "NO TRADE", tone: "neutral" };
  if (row.group === "scored" || row.group === "found") return null;
  return state.phase === "running" ? null : { text: "NO TRADE", tone: "neutral" };
}

export function orderWords(row: WorkRow): { text: string; dollars: string } | null {
  const d = row.decision;
  if (!d || !d.shares || !d.will_trade.startsWith("yes_")) return null;
  const verb = d.action === "reduce" ? "Sell" : "Buy";
  // an order in dollars (entry 111) says its dollars first and the shares as an estimate; `dollars` is the second line of the cell
  if (d.dollars) return { text: `${verb} ${money(d.dollars)}`, dollars: `about ${shareText(d.shares)} sh` };
  return { text: `${verb} ${shareText(d.shares)}`, dollars: money(d.est_cost) };
}

export function statusBadge(state: RunState): { text: string; tone: Tone } | null {
  const status = state.workOrder?.status;
  if (!status) return null;
  const base = status.replace(/_/g, " ").toUpperCase();
  const trading = state.stages.find((s) => s.key === "trade");
  if (status === "decided" && trading && (trading.state === "busy" || trading.state === "waiting")) return { text: "DECIDED → TRADING", tone: "brand" };
  if (status === "traded") return { text: base, tone: "pos" };
  if (status === "data_failed") return { text: base, tone: "neg" };
  return { text: base, tone: "neutral" };
}

/** `+8 NEW`, `+8`, `−5`: the badge of a position the run changed. */
export function changeBadge(change: number, shares: number | null): { text: string; tone: Tone } {
  // quantities are compared at the 6 places a share quantity has (a float `===` would call 0.1 + 0.2 and 0.3 different)
  if (change > 0 && shares !== null && round6(shares) === round6(change)) return { text: `${signedShares(change)} NEW`, tone: "pos" };
  if (shares !== null && round6(shares) === 0) return { text: `${signedShares(change)} SOLD`, tone: "neg" };
  return { text: signedShares(change), tone: change > 0 ? "pos" : "neg" };
}

export function agentWords(agent: string): string {
  return agent === "account_manager" ? "Account Manager" : agent === "data_clerk" ? "Data Clerk" : cap(agent);
}
