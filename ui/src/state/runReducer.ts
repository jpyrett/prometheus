// The one reducer of the page (spec 5.4): runReducer(state, event) -> state. Pure: no clock, no network, no DOM.
// The server's events are the only input. Seconds counters read `since` (the server's own `ts` of the busy event)
// so a page reloaded in the middle of a run counts from the right moment.
import type {
  AccountView,
  ActivityData,
  LogAgent,
  LogLevel,
  PortfolioData,
  QuestionData,
  RunFinishedData,
  RunStartedData,
  ServerEvent,
  StageInfo,
  StageKey,
  StageState,
  TradeMode,
  WorkOrderView,
} from "../api/types";

export interface StageView extends StageInfo {
  state: StageState;
  node: string | null;
  /** Final seconds of a finished stage (the server's `stage.seconds`). */
  seconds: number;
  /** Server epoch seconds when the stage became busy (null unless busy or waiting). */
  since: number | null;
}

export interface LogLine {
  id: number;
  ts: number;
  agent: LogAgent;
  level: LogLevel;
  text: string;
  block: boolean;
}

export type Phase = "idle" | "running" | "finished";

export interface RunState {
  phase: Phase;
  runId: string | null;
  mode: "live" | "replay" | null;
  /** The number of the last event applied: the reconnect passes it back. */
  lastSeq: number;
  prompt: string | null;
  tradeMode: TradeMode | null;
  startedTs: number | null;
  finishedTs: number | null;
  stages: StageView[];
  log: LogLine[];
  /** The one open question, or null. */
  question: QuestionData | null;
  workOrder: WorkOrderView | null;
  /** The account as the run last reported it, with what changed against the first one. */
  portfolio: PortfolioData | null;
  /** The first account view of the run: the 'before'. */
  portfolioBefore: AccountView | null;
  activity: ActivityData | null;
  result: RunFinishedData | null;
}

/** The six stages of the desk, so the idle page shows the pipeline before the server has said anything. Same keys, labels and agents as `views.STAGES` on the server (a run_started replaces them). */
export const DEFAULT_STAGES: StageInfo[] = [
  { key: "work_order", label: "Create Work Order", agent: "account_manager", agent_label: "Account Manager" },
  { key: "gather", label: "Gather Data", agent: "data_clerk", agent_label: "Data Clerk" },
  { key: "recommend", label: "Create Recommendations", agent: "advisor", agent_label: "Advisor" },
  { key: "critique", label: "Critique · Approve / Deny", agent: "critic", agent_label: "Critic" },
  { key: "prepare", label: "Prepare Trading Order", agent: "account_manager", agent_label: "Account Manager" },
  { key: "trade", label: "Trade · Place Order", agent: "trader", agent_label: "Trader" },
];

function pendingStages(info: StageInfo[]): StageView[] {
  return info.map((s) => ({ ...s, state: "pending", node: null, seconds: 0, since: null }));
}

export const initialRunState: RunState = {
  phase: "idle",
  runId: null,
  mode: null,
  lastSeq: 0,
  prompt: null,
  tradeMode: null,
  startedTs: null,
  finishedTs: null,
  stages: pendingStages(DEFAULT_STAGES),
  log: [],
  question: null,
  workOrder: null,
  portfolio: null,
  portfolioBefore: null,
  activity: null,
  result: null,
};

function started(data: RunStartedData, seq: number): RunState {
  return {
    ...initialRunState,
    phase: "running",
    runId: data.run_id,
    mode: data.mode,
    lastSeq: seq,
    prompt: data.prompt,
    tradeMode: data.trade_mode,
    startedTs: data.ts,
    stages: pendingStages(data.stages.length ? data.stages : DEFAULT_STAGES),
  };
}

function patchStage(stages: StageView[], key: StageKey, patch: (s: StageView) => StageView): StageView[] {
  return stages.map((s) => (s.key === key ? patch(s) : s));
}

export function runReducer(state: RunState, event: ServerEvent): RunState {
  // A new run replaces everything (the same run's own run_started, seen again after a reconnect, changes nothing). Any other event already applied is ignored.
  if (event.type === "run_started") return event.data.run_id === state.runId && event.seq <= state.lastSeq ? state : started(event.data, event.seq);
  if (event.seq <= state.lastSeq) return state;
  const next: RunState = { ...state, lastSeq: event.seq };

  switch (event.type) {
    case "stage": {
      const d = event.data;
      return {
        ...next,
        stages: patchStage(state.stages, d.key, (s) => {
          const active = d.state === "busy" || d.state === "waiting";
          // The counter keeps counting from the first busy event of the node: waiting and busy again is the same stage.
          const since = active ? (s.since !== null && s.node === d.node && (s.state === "busy" || s.state === "waiting") ? s.since : d.ts) : null;
          return { ...s, state: d.state, node: d.node, seconds: d.seconds, since };
        }),
      };
    }
    case "log":
      return { ...next, log: [...state.log, { id: event.seq, ts: event.data.ts, agent: event.data.agent, level: event.data.level, text: event.data.text, block: event.data.block }] };
    case "question": {
      // While a question is open the stage that is busy waits for the user (the server says so too: this keeps the page right if it does not).
      const stages = state.stages.map((s) => (s.state === "busy" ? { ...s, state: "waiting" as const } : s));
      return { ...next, question: event.data, stages };
    }
    case "question_closed": {
      if (state.question?.question_id !== event.data.question_id) return next;
      const stages = state.stages.map((s) => (s.state === "waiting" ? { ...s, state: "busy" as const } : s));
      return { ...next, question: null, stages };
    }
    case "work_order": {
      const { ts: _ts, ...view } = event.data;
      return { ...next, workOrder: view };
    }
    case "portfolio": {
      const before = state.portfolioBefore ?? event.data;
      return { ...next, portfolio: event.data, portfolioBefore: before };
    }
    case "activity":
      return { ...next, activity: event.data };
    case "run_finished": {
      // A stage still busy or waiting when the run ended (stopped, failed) is where it stopped.
      const open = event.data.status === "stopped" || event.data.status === "failed";
      const stages = state.stages.map((s) => (s.state === "busy" || s.state === "waiting" ? { ...s, state: open ? ("failed" as const) : ("done" as const), since: null } : s));
      return { ...next, phase: "finished", question: null, stages, result: event.data, finishedTs: event.data.ts };
    }
  }
}

/** Fold a whole list of events (tests, the fixture mode). */
export function replayEvents(events: ServerEvent[], from: RunState = initialRunState): RunState {
  return events.reduce(runReducer, from);
}
