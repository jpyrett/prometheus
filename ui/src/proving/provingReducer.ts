// The one reducer of the Proving Ground's run (spec 13.3): provingReducer(state, event) -> state. Pure: no clock, no network, no DOM.
// It folds the events of the Time Machine's stream (pv_research, pv_started, pv_day, pv_decision, pv_outcome, pv_finished, and the shared log, activity, run_started and run_finished)
// and it can be loaded with a whole saved run (a run opened from the list, the Track Record). It keeps what the server sent and works out nothing of its own (P12): a count of tiles that are done
// is a count, an equity point is drawn where it arrived.
import type { ActivityData, LogAgent, LogLevel, RunFinishedData, ServerEvent, WorkOrderView } from "../api/types";
import { caveatItems } from "./caveats";
import type { CaveatItem, CurveEvent, DecisionDay, EquityPoint, ProvingRun, PvResearchData, PvStartedData, RecOutcome, RunRange, RunTiming, ServerKpis, Speed, Stats } from "./types";

export type ProvingPhase = "idle" | "running" | "finished";
/** Where a running test is: researching the decision days (in parallel), walking the days in order, done. */
export type ProvingStep = "idle" | "research" | "walk" | "done";

export interface ProvingLogLine {
  id: number;
  ts: number;
  agent: LogAgent;
  level: LogLevel;
  text: string;
}

export interface ProvingState {
  phase: ProvingPhase;
  step: ProvingStep;
  /** The id of the run on the stream (a replay of a saved run has its own). */
  runId: string | null;
  /** The id the run is saved under: what "Watch it again", the address and the day-by-day desks use. */
  savedId: string | null;
  lastSeq: number;
  /** The state was built from a stream of events (a live run, or a saved run played again), not loaded whole: the page draws it at its own pace. */
  streamed: boolean;
  prompt: string | null;
  understood: string | null;
  range: RunRange | null;
  /** Every decision day of the plan, in order (known at pv_started). */
  plannedDays: string[];
  startingCash: number | null;
  parallelDays: number | null;
  research: Record<string, PvResearchData>;
  points: EquityPoint[];
  events: CurveEvent[];
  /** The walk's row for each decision day it has reached. */
  decisionDays: DecisionDay[];
  /** The decision days that have a desk the server can show. */
  deskDays: string[];
  /** The desk's own work-order view of a decision day when the page has it (the stream sent it, or it was read from the server): the same table and panel as the Desk tab. */
  desks: Record<string, WorkOrderView>;
  outcomes: RecOutcome[];
  /** The keys of the outcomes that arrived (or changed) since the last time the page looked: the card that flashes once. */
  freshKeys: string[];
  caveats: CaveatItem[];
  /** The five tiles as the server worked them out (a saved run's view); a live run's come from its statistics. */
  kpis: ServerKpis | null;
  stats: Stats | null;
  seconds: number | null;
  /** How the run was made: fast or faithful (null for a run saved before the speeds). */
  speed: Speed | null;
  /** What the server measured: the seconds of each phase, of each day and stage, and of each model. */
  timing: RunTiming | null;
  /** The decision days on which few articles were stored for the companies (the caveat news_coverage_thin), and the count of articles found on each day. Known when the saved run has been read. */
  newsThin: string[];
  newsSources: Record<string, number>;
  langsmith: { trace_id: string | null; url: string | null } | null;
  models: Record<string, unknown>;
  configVersions: Record<string, unknown>;
  status: "finished" | "stopped" | "failed" | null;
  /** run_finished: why a run did not finish, in the server's words. */
  failure: string | null;
  log: ProvingLogLine[];
  activity: ActivityData | null;
  startedTs: number | null;
  finishedTs: number | null;
  /** What kind of run is shown: a Time Machine run, or the Track Record loaded whole. */
  kind: ProvingRun["kind"] | null;
  /** Where each outcome came from (the Track Record only). */
  sources: Record<string, "run" | "fixture">;
  /** The unit of the curve: dollars for a Time Machine run, an index (100 at the first entry) for the Track Record. */
  unit: "money" | "index";
}

export const initialProvingState: ProvingState = {
  phase: "idle",
  step: "idle",
  runId: null,
  savedId: null,
  lastSeq: 0,
  streamed: false,
  prompt: null,
  understood: null,
  range: null,
  plannedDays: [],
  startingCash: null,
  parallelDays: null,
  research: {},
  points: [],
  events: [],
  decisionDays: [],
  deskDays: [],
  desks: {},
  outcomes: [],
  freshKeys: [],
  caveats: [],
  kpis: null,
  stats: null,
  seconds: null,
  speed: null,
  timing: null,
  newsThin: [],
  newsSources: {},
  langsmith: null,
  models: {},
  configVersions: {},
  status: null,
  failure: null,
  log: [],
  activity: null,
  startedTs: null,
  finishedTs: null,
  kind: null,
  sources: {},
  unit: "money",
};

export type ProvingAction =
  | ServerEvent
  | { type: "load"; run: ProvingRun; desks?: Record<string, WorkOrderView> }
  /** The saved run's own record read after a streamed run finished: its models, config versions, tiles and desks (the stream does not send them). Nothing already drawn changes. */
  | { type: "details"; run: ProvingRun }
  | { type: "reset" };

function promptOf(request: Record<string, unknown> | null | undefined): string | null {
  const p = request?.prompt;
  return typeof p === "string" ? p : null;
}

/** The run's measured timing. A run saved before the speeds sends `timing: {}` and keeps its phases in `coverage.timing`: that is read when the run's own is empty. */
function timingOf(run: ProvingRun): RunTiming | null {
  if (run.timing && Object.keys(run.timing).length > 0) return run.timing;
  const found = (run.coverage as { timing?: RunTiming }).timing;
  return found && Object.keys(found).length > 0 ? found : null;
}

function coverageThin(run: ProvingRun): string[] {
  const days = (run.coverage as { news_thin_days?: unknown }).news_thin_days;
  return Array.isArray(days) ? days.filter((d): d is string => typeof d === "string") : [];
}

function coverageSources(run: ProvingRun): Record<string, number> {
  const found = (run.coverage as { news_sources?: unknown }).news_sources;
  return found && typeof found === "object" ? (found as Record<string, number>) : {};
}

/** A whole saved run as a state (a run opened from the list, the Track Record): finished, with everything in place. */
export function stateOfRun(run: ProvingRun, desks: Record<string, WorkOrderView> = {}): ProvingState {
  // Where each outcome came from: the outcome says so itself (`source`: a run of this page, a recorded session, or a decision day of a Time Machine run); a run saved before that field says it in `coverage.sources`.
  const mapped = (run.coverage as { sources?: Record<string, "run" | "fixture"> }).sources ?? {};
  const sources: Record<string, "run" | "fixture"> = {};
  for (const o of run.outcomes) {
    const s = o.source === "run" || o.source === "fixture" ? o.source : mapped[o.key];
    if (s) sources[o.key] = s;
  }
  return {
    ...initialProvingState,
    phase: "finished",
    step: "done",
    runId: run.run_id,
    savedId: run.run_id,
    prompt: promptOf(run.request),
    understood: run.understood || null,
    range: run.range,
    plannedDays: [...new Set(run.decision_days.map((d) => d.day))],
    startingCash: run.kind === "time_machine" ? ((run.request?.cash as number | undefined) ?? run.stats.portfolio?.start_equity ?? null) : (run.stats.portfolio?.start_equity ?? run.equity[0]?.equity ?? null),
    points: run.equity,
    events: run.events,
    decisionDays: run.decision_days,
    deskDays: run.desks ?? Object.keys(desks),
    desks,
    outcomes: run.outcomes,
    caveats: caveatItems(run.caveats),
    kpis: run.kpis ?? null,
    stats: run.stats,
    seconds: run.seconds,
    speed: run.speed ?? run.timing?.speed ?? null,
    timing: timingOf(run),
    newsThin: coverageThin(run),
    newsSources: coverageSources(run),
    langsmith: run.langsmith,
    models: run.models,
    configVersions: run.config_versions,
    status: run.status,
    kind: run.kind,
    sources,
    unit: run.kind === "track" ? "index" : "money",
  };
}

function upsert(outcomes: RecOutcome[], outcome: RecOutcome): RecOutcome[] {
  const at = outcomes.findIndex((o) => o.key === outcome.key);
  if (at < 0) return [...outcomes, outcome];
  const next = outcomes.slice();
  next[at] = outcome;
  return next;
}

export function provingReducer(state: ProvingState, action: ProvingAction): ProvingState {
  if (action.type === "reset") return initialProvingState;
  if (action.type === "load") return stateOfRun(action.run, action.desks);
  if (action.type === "details") {
    const r = action.run;
    if (state.savedId !== r.run_id) return state;
    return { ...state, models: r.models, configVersions: r.config_versions, kpis: r.kpis ?? state.kpis, deskDays: r.desks ?? state.deskDays, langsmith: r.langsmith ?? state.langsmith, caveats: state.caveats.length ? state.caveats : caveatItems(r.caveats), speed: state.speed ?? r.speed ?? null, timing: state.timing ?? timingOf(r), newsThin: coverageThin(r), newsSources: coverageSources(r) };
  }
  const event = action;
  // A new run replaces everything; the same run's own start seen again after a reconnect changes nothing.
  if (event.type === "run_started") {
    // a Desk run's start is not ours (the Proving Ground's run says mode "proving")
    if (String(event.data.mode) !== "proving") return state;
    if (event.data.run_id === state.runId && event.seq <= state.lastSeq) return state;
    return { ...initialProvingState, streamed: true, phase: "running", step: "research", runId: event.data.run_id, lastSeq: event.seq, prompt: event.data.prompt ?? null, speed: event.data.options?.speed ?? null, startedTs: event.data.ts };
  }
  if (event.seq <= state.lastSeq) return state;
  const next: ProvingState = { ...state, lastSeq: event.seq, freshKeys: [] };

  switch (event.type) {
    case "pv_started": {
      const d: PvStartedData = event.data;
      // a stream that opens with pv_started (no run_started) starts the run here
      const base = state.phase === "idle" ? { ...initialProvingState, lastSeq: event.seq, runId: d.run_id ?? null, startedTs: d.ts ?? null } : next;
      return {
        ...base,
        streamed: true,
        phase: "running",
        step: "research",
        prompt: promptOf(d.request) ?? base.prompt,
        understood: d.understood ?? null,
        savedId: d.run_id ?? base.savedId,
        range: d.range,
        plannedDays: [...new Set(d.decision_days)],
        startingCash: d.starting_cash,
        parallelDays: d.parallel_days ?? null,
        speed: d.speed ?? d.timing?.speed ?? base.speed,
        caveats: caveatItems(d.caveats),
      };
    }
    case "pv_research":
      return { ...next, research: { ...state.research, [event.data.day]: event.data } };
    case "pv_day": {
      const d = event.data;
      const row = d.decision_day ?? d.decision ?? null;
      return {
        ...next,
        step: "walk",
        points: [...state.points, d.point],
        events: d.events && d.events.length ? [...state.events, ...d.events] : state.events,
        decisionDays: row ? [...state.decisionDays.filter((x) => x.day !== row.day), row] : state.decisionDays,
      };
    }
    case "pv_decision":
      return { ...next, deskDays: state.deskDays.includes(event.data.day) ? state.deskDays : [...state.deskDays, event.data.day], desks: event.data.view ? { ...state.desks, [event.data.day]: event.data.view } : state.desks };
    case "pv_outcome":
      return { ...next, outcomes: upsert(state.outcomes, event.data.outcome), freshKeys: [event.data.outcome.key] };
    case "pv_finished": {
      const d = event.data;
      return {
        ...next,
        phase: "finished",
        step: "done",
        stats: d.stats,
        caveats: caveatItems(d.caveats),
        seconds: d.seconds ?? state.seconds,
        timing: d.timing ?? null,
        speed: d.speed ?? d.timing?.speed ?? state.speed,
        langsmith: d.langsmith ?? state.langsmith,
        status: d.status ?? "finished",
        savedId: d.saved_id ?? d.run_id ?? state.savedId,
        kind: "time_machine",
      };
    }
    case "log":
      if (state.phase === "idle") return state;
      return { ...next, log: [...state.log, { id: event.seq, ts: event.data.ts, agent: event.data.agent, level: event.data.level, text: event.data.text }] };
    case "activity":
      return state.phase === "idle" ? state : { ...next, activity: event.data };
    case "run_finished": {
      if (state.phase === "idle") return state;
      const d: RunFinishedData = event.data;
      const failed = d.status === "stopped" || d.status === "failed";
      // A stopped or failed run ends here with what it has; a finished one already said so in pv_finished.
      return { ...next, phase: "finished", step: "done", finishedTs: d.ts, failure: failed ? d.headline : state.failure, status: failed ? (d.status as "stopped" | "failed") : (state.status ?? "finished") };
    }
    default:
      return next;
  }
}

/** Fold a list of events (tests, the fixture mode). */
export function replayProving(events: ServerEvent[], from: ProvingState = initialProvingState): ProvingState {
  return events.reduce<ProvingState>((s, e) => provingReducer(s, e), from);
}
