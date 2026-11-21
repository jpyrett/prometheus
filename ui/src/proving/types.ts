// The shapes of the Proving Ground as the server sends them (design/proving_ground_spec.md sections 8 and 9; the models are
// prometheus/proving/record.py). Money and prices arrive as JSON numbers, dates as ISO texts, and every `_pct` is a FRACTION (0.0077 is 0.77%).
// The page draws these numbers and computes no statistic of its own (rule P12).
import type { WorkOrderView } from "../api/types";

export type Verdict = "approved" | "denied" | "neutral" | "unreviewed";
export type Result = "won" | "lost" | "open" | "not_filled" | "no_data";
export type ExitReason = "stop" | "trailing_stop" | "target" | "time_stop";
/** How a run was made (prometheus/proving/speed.py): FAST asks the Advisor's model with its reasoning switched off and the Critic's at a lower effort; FAITHFUL asks the models as the live desk does. Every program rule is the same in both. */
export type Speed = "fast" | "faithful";
/** Where an outcome's recommendation came from: a run of this page, a recorded session of the project (the Track Record), or a decision day of a Time Machine run. */
export type OutcomeSource = "run" | "fixture" | "time_machine";
export type NotTradedReason = "denied" | "neutral_off" | "beyond_count" | "no_cash" | "not_filled" | "failed" | "cooldown";
export type RunKind = "time_machine" | "track" | "live";
export type RunStatus = "finished" | "stopped" | "failed";
export type DayStatus = "traded" | "no_trade" | "failed" | "cached";
export type Tone = "pos" | "neg" | "neutral";
export type CaveatCode =
  | "fundamentals_not_point_in_time"
  | "symbol_directory_is_todays"
  | "price_returns_no_dividends"
  | "fills_at_next_open_no_slippage"
  | "daily_bars_stop_before_target"
  | "news_coverage_thin"
  | "small_sample"
  | "open_positions_marked"
  | "split_adjusted"
  | "model_knowledge_cutoff"
  | "fast_models"
  | "analysts_not_point_in_time";

/** The Advisor's plan: prices and dates, no size. */
export interface PlanView {
  entry_low: number | null;
  entry_high: number | null;
  entry_mid: number | null;
  limit_price: number | null;
  stop: number | null;
  target: number | null;
  time_stop_date: string | null;
  horizon_bars: number | null;
  order_style: string | null;
  /** The stop TRAILS the highest close (the trend family since Advisor config version 3: no target); absent in a run saved before. */
  trailing?: boolean;
  trail_distance?: number | null;
}

export interface AdvisorCell {
  score: number | null;
  stance: string | null;
  family: string | null;
  rank: number | null;
}

export interface CriticCell {
  approved: boolean;
  score: number | null;
  grade: string | null;
  denial_reasons: string[];
  /** What the review found, by code (Critic v2); null in a run saved before they were kept. */
  config_version?: string | null;
  blockers?: string[] | null;
  warnings?: string[] | null;
  human?: string[] | null;
}

export interface EntryView {
  date: string;
  price: number;
  shares: number;
  amount: number;
}

export interface ExitView {
  date: string;
  price: number;
  reason: ExitReason;
  sessions_held: number;
}

export interface Horizon {
  sessions: number;
  return_pct: number | null;
  benchmark_return_pct: number | null;
}

/** The stock's return from the close of the decision day to the close `sessions` sessions later, against the benchmark's over the same dates, whatever the plan did: it measures the pick, not where the plan put its stop and target. Absent in a run saved before it existed. */
export interface Forward {
  sessions: number;
  return_pct: number | null;
  benchmark_return_pct: number | null;
  excess_return_pct: number | null;
}

/** One day of a position's path: the close, the return since the entry price, and the benchmark's return over the same days. */
export interface PathPoint {
  date: string;
  close: number;
  return_pct: number;
  benchmark_return_pct: number | null;
}

export interface RecOutcome {
  key: string;
  decision_day: string;
  symbol: string;
  company: string | null;
  record_id: string | null;
  review_id: string | null;
  advisor: AdvisorCell | null;
  critic: CriticCell | null;
  trust_score: number | null;
  verdict: Verdict;
  traded: boolean;
  /** P8: not traded, one share-equivalent followed by its plan; never in the equity curve. */
  hypothetical: boolean;
  not_traded_reason: NotTradedReason | null;
  reject_code: string | null;
  /** Older saved runs do not say. */
  source?: OutcomeSource | null;
  plan: PlanView;
  entry: EntryView | null;
  exit: ExitView | null;
  result: Result;
  return_pct: number | null;
  benchmark_return_pct: number | null;
  excess_return_pct: number | null;
  sector_return_pct: number | null;
  gain: number | null;
  horizons: Horizon[];
  forward?: Forward[];
  max_favorable_pct: number | null;
  max_adverse_pct: number | null;
  path: PathPoint[];
  split_adjusted: boolean;
  notes: string[];
}

export interface EquityPoint {
  date: string;
  equity: number;
  cash: number;
  invested: number;
  benchmark: number | null;
  /** A fraction at or below zero. */
  drawdown_pct: number;
  stale: string[];
}

/** A buy or an exit: a marker on the curve. */
export interface CurveEvent {
  date: string;
  kind: "buy" | "exit";
  symbol: string;
  shares: number;
  price: number;
  amount: number;
  reason: string | null;
  key: string | null;
}

export interface DecisionDay {
  day: string;
  status: DayStatus;
  recommended: number;
  approved: number;
  denied: number;
  traded: number;
  spent: number;
  reason: string | null;
  trace_id: string | null;
}

/** One model call of a decision day, as the engine timed it (`start` is seconds into the day). */
export interface CallTiming {
  model: string;
  stage: string;
  start: number;
  seconds: number;
  tokens_in: number;
  tokens_out: number;
  ok: boolean;
}

/** The seconds of one researched decision day, split by the desk's stages (the stages run one after another: they add up to the day). A cached day has none. */
export interface DayTiming {
  seconds: number;
  stages: Record<string, number>;
  calls?: CallTiming[];
  attempts?: number;
}

export interface ModelTiming {
  calls: number;
  failed: number;
  seconds: number;
  tokens_in: number;
  tokens_out: number;
}

/** What the server measured of a run (`ProvingRun.timing`, and `pv_finished.timing`). A run saved before the speeds has only the first four fields. */
export interface RunTiming {
  speed?: Speed;
  research_seconds?: number;
  walk_seconds?: number;
  setup_seconds?: number;
  link_seconds?: number;
  cached_days?: number;
  parallel_days?: number;
  per_day?: Record<string, DayTiming>;
  /** Each desk stage's work added up over the days (they ran at once, so it is more than the wall time) and its longest single day. */
  stages?: Record<string, { total: number; max: number }>;
  models?: Record<string, ModelTiming>;
}

export interface RunRange {
  first_day: string;
  last_day: string;
  last_bar_day: string | null;
}

// ---------------------------------------------------------------- the statistics (8.3)

export interface PortfolioStats {
  start: string | null;
  end: string | null;
  start_equity: number;
  end_equity: number;
  return_pct: number;
  benchmark_return_pct: number | null;
  excess_return_pct: number | null;
  max_drawdown_pct: number;
  drawdown_from: string | null;
  drawdown_to: string | null;
  best_day_pct: number | null;
  worst_day_pct: number | null;
  days: number;
  invested_avg_pct: number | null;
}

export interface Exits {
  stop: number;
  /** Absent in a run saved before the trailing stop existed. */
  trailing_stop?: number;
  target: number;
  time_stop: number;
  open: number;
}

export interface OutcomeStats {
  n: number;
  closed: number;
  open: number;
  won: number;
  lost: number;
  not_filled: number;
  no_data: number;
  win_rate: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  profit_factor: number | null;
  avg_sessions_held: number | null;
  exits: Exits;
  small_sample: boolean;
}

export interface GroupStats {
  n: number;
  closed: number;
  mean_return_pct: number | null;
  mean_excess_pct: number | null;
  win_rate: number | null;
  small_sample: boolean;
}

/** The mean FORWARD excess of a group at one horizon, with its n, the standard error of the mean and whether chance alone gives a mean this size (within two standard errors of zero; null with fewer than two). */
export interface ForwardStats {
  sessions: number;
  n: number;
  mean_excess_pct: number | null;
  se_pct: number | null;
  inside_chance: boolean | null;
  small_sample: boolean;
}

export interface VerdictStats extends GroupStats {
  verdict: Verdict;
}
export interface ScoreStats extends GroupStats {
  low: number;
  high: number;
  /** The plan-independent excess of the strategy-family picks of the score bucket; absent in a run saved before it existed. The win rate above is "won by its plan". */
  forward?: ForwardStats[];
}
export interface FamilyStats extends GroupStats {
  /** A strategy family, or "neutral" for the companies the Advisor scored neutral. */
  family: string;
  forward?: ForwardStats[];
}
export interface SymbolStats {
  symbol: string;
  n: number;
  gain: number | null;
  return_pct: number | null;
}

export interface CriticValue {
  denied_n: number;
  denied_mean_excess_pct: number | null;
  approved_n: number;
  approved_mean_excess_pct: number | null;
  sentence: string;
}

export interface Headline {
  tone: Tone;
  sentence: string;
}

export interface Stats {
  portfolio: PortfolioStats | null;
  outcomes: OutcomeStats;
  by_verdict: VerdictStats[];
  by_score: ScoreStats[];
  by_family: FamilyStats[];
  by_symbol: SymbolStats[];
  critic_value: CriticValue;
  headline: Headline;
}

/** A caveat as the page shows it: its code and its one sentence. The saved-run view sends objects; a bare ProvingRun (and the Track Record) sends the code only. */
export type CaveatEntry = string | { code: string; sentence: string };

export interface CaveatItem {
  code: string;
  sentence: string;
}

/** The tiles of the headline strip as `proving_views.kpis` sends them: every figure a field of `Stats`. */
export interface ServerKpis {
  return_pct: number | null;
  excess_return_pct: number | null;
  benchmark_return_pct: number | null;
  win_rate: { value: number | null; n: number; small_sample: boolean };
  max_drawdown_pct: number | null;
  recommendations: { total: number; traded: number; denied: number; open: number };
}

/** One saved run as the page's view (8.2; `proving_views.run_view`): the run's JSON, plus its caveats as sentences, the five tiles and the days that have a desk. */
export interface ProvingRun {
  run_id: string;
  kind: RunKind;
  status: RunStatus;
  created_utc: string;
  seconds: number;
  request: Record<string, unknown> | null;
  understood: string;
  range: RunRange | null;
  decision_days: DecisionDay[];
  equity: EquityPoint[];
  events: CurveEvent[];
  outcomes: RecOutcome[];
  stats: Stats;
  caveats: CaveatEntry[];
  /** `proving_views.run_view` only. */
  caveat_codes?: string[];
  /** Older saved runs do not say (they were made before the speeds). */
  speed?: Speed | null;
  timing?: RunTiming | null;
  kpis?: ServerKpis;
  /** The days whose desk the server can show (`GET .../days/{day}`). */
  desks?: string[];
  coverage: Record<string, unknown>;
  models: Record<string, unknown>;
  config_versions: Record<string, unknown>;
  langsmith: { trace_id: string | null; url: string | null } | null;
}

/** `GET /api/proving/runs/{id}/days/{day}` (`proving_views.day_view`): the desk of one decision day, the same view the Desk tab draws. */
export interface DayView {
  day: string;
  title: string;
  view: WorkOrderView;
  decision_day: DecisionDay | null;
}

// ---------------------------------------------------------------- the events of a run (9)

export interface PvStartedData {
  run_id?: string;
  /** The Account Manager's own words for what it read. */
  understood?: string;
  request?: Record<string, unknown> | null;
  range: RunRange | null;
  /** Every decision day of the plan, in order. */
  decision_days: string[];
  starting_cash: number | null;
  caveats?: CaveatEntry[];
  parallel_days?: number | null;
  phase?: string;
  speed?: Speed | null;
  timing?: { speed?: Speed; parallel_days?: number | null; estimate_seconds?: number | null } | null;
  ts?: number;
}

export interface PvResearchData {
  day: string;
  status: DayStatus | "busy" | "done";
  recommended?: number;
  approved?: number;
  denied?: number;
  seconds?: number;
  cached?: boolean;
  reason?: string | null;
  /** How many decision days have come back, and of how many. */
  done?: number;
  total?: number;
  usage?: unknown;
  /** The day's seconds by desk stage (empty for a cached day). `seconds` is the day's own wall time when the engine sent it; a saved run played again sends 0 and the day's seconds are in `pv_finished.timing`. */
  stages?: Record<string, number>;
  ts?: number;
}

export interface PvDayData {
  point: EquityPoint;
  events?: CurveEvent[];
  /** The walk's row of the day, on a decision day (`decision_day`; `decision` is accepted too). */
  decision_day?: DecisionDay | null;
  decision?: DecisionDay | null;
  index?: number;
  total?: number;
  ts?: number;
}

export interface PvDecisionData {
  day: string;
  /** The day's work order as the engine kept it (raw): the page reads the day's desk from the server (`GET .../days/{day}`) when it opens it. `view` is the Desk's view when the stream sends one. */
  order?: unknown;
  view?: WorkOrderView;
  understood?: string;
  ts?: number;
}

export interface PvOutcomeData {
  outcome: RecOutcome;
  ts?: number;
}

export interface PvFinishedData {
  run_id?: string;
  /** The id the run was saved under (what "Watch it again" and the address use). */
  saved_id?: string;
  status?: RunStatus;
  stats: Stats;
  headline?: Headline;
  caveats: CaveatEntry[];
  equity?: EquityPoint[];
  seconds?: number;
  timing?: RunTiming | null;
  speed?: Speed | null;
  langsmith?: { trace_id: string | null; url: string | null } | null;
  range_first_day?: string | null;
  ts?: number;
}

// ---------------------------------------------------------------- the routes (9)

/** What is stored in one month (`GET /api/proving/coverage`: `months` is an object keyed by "2026-10"). */
export interface CoverageMonth {
  /** A kind with nothing stored that month is left out of the answer, not sent as 0. */
  bars?: number;
  news?: number;
  press_releases?: number;
  filings?: number;
  market_news?: number;
  news_chunks?: number;
}

export interface Coverage {
  first_bar_day: string | null;
  last_bar_day: string | null;
  usable_last_day?: string | null;
  /** The first and last day a test may start and end on. */
  usable: { first_day: string | null; last_day: string | null } | null;
  months: Record<string, CoverageMonth>;
  symbols_with_splits?: string[];
  notes?: string[];
  /** The days a handle may land on, if the server ever sends them (it does not today: the page snaps to weekdays and the plan refuses a holiday). */
  trading_days?: string[];
}

export type Cadence = "daily" | "twice_weekly" | "weekly" | "every_two_weeks";

export interface TimeMachineRequest {
  prompt: string;
  cash: number;
  start: string;
  end: string;
  cadence: Cadence;
  risk: "conservative" | "moderate" | "aggressive";
  neutral: boolean;
  strict: boolean;
  /** Left out (or null) is the server's default: fast. */
  speed?: Speed | null;
}

/** `POST /api/proving/plan` (`engine.plan`): free; nothing is read with a model. */
export interface PlanAnswer {
  /** The reasons the request cannot run, each with the usable range, in the server's words; empty when it can. */
  problems: string[];
  cadence: string;
  decision_days: string[];
  cached_days: string[];
  /** The cache key holds the way the request was read: the count of cached days is known only after a first run of the same words. */
  cache_known: boolean;
  paid_days: number;
  caveats: CaveatEntry[];
  cost_estimate_usd: number | null;
  range: { first_day: string | null; last_day: string | null };
  last_bar_day: string | null;
  /** The speed the plan is for, and what the server expects of it (0 seconds when every day is cached; null when none was measured). */
  speed?: Speed;
  timing?: PlanTiming;
}

export interface PlanTiming {
  speed: Speed;
  parallel_days: number;
  cached_days: number;
  paid_days: number;
  estimate_seconds: number | null;
}

export interface RunIndexRow {
  run_id: string;
  kind: RunKind;
  status?: RunStatus;
  created_utc: string;
  prompt: string;
  range: RunRange | null;
  /** The index row carries the headline's sentence only (a string); a full {tone, sentence} is accepted too. */
  headline: string | Headline | null;
  return_pct: number | null;
  excess_return_pct: number | null;
  /** Fast or faithful; a row saved before the speeds has none. */
  speed?: Speed | null;
  /** The equity curve in a few points, for the card's own sparkline, when the index carries it. */
  spark?: number[] | null;
}

/** The Track Record is a ProvingRun of kind `track` (recorded: tests/fixtures/proving_real/track.json). Its curve is an INDEX (100 at the first entry), not dollars; `coverage` says where each outcome came from. */
export interface TrackCoverage {
  /** An outcome's key to where it came from: a run of this page, or a recorded session. */
  sources: Record<string, "run" | "fixture">;
  captures: { source: "run" | "fixture"; name: string; decision_day: string; request: string; outcomes: number }[];
  last_bar_day: string | null;
  /** The server's own words for what the curve is. */
  index: string;
}

export interface LivePlan {
  stop: number | null;
  target: number | null;
  time_stop_date: string | null;
  /** A trailing plan (no target): `stop` is its starting stop, `stop_now` the stop in force now and `highest_close` what it rises from; absent in a snapshot made before. */
  trailing?: boolean;
  trail_distance?: number | null;
  highest_close?: number | null;
  stop_now?: number | null;
}

export type LiveStatus = "in_plan" | "stop_hit" | "target_hit" | "time_stop_passed" | "no_plan";

export interface LivePosition {
  symbol: string;
  name: string | null;
  shares: number;
  cost_basis: number;
  price: number | null;
  price_time: string | null;
  simulated: boolean;
  value: number | null;
  gain: number | null;
  gain_pct: number | null;
  day_change_pct: number | null;
  plan: LivePlan | null;
  status: LiveStatus;
  /** What the status rests on: "the quote now", "the low of 2026-11-17". */
  evidence: string | null;
  since: string | null;
  record_id: string | null;
  notes: string[];
}

export interface LiveAlert {
  symbol: string;
  kind: string;
  sentence: string;
}

export interface LiveSession {
  state: string;
  open: boolean;
  reason: string;
  next_open_utc: string | null;
}

export interface HistoryPoint {
  date: string;
  equity: number;
  cash: number;
  invested: number;
  stale: string[];
}

/** One live snapshot (recorded: live_snapshot_oil.json, live_snapshot_plan.json). It reads and tells; nothing in it can sell (P11). */
export interface LiveSnapshot {
  account_id: number;
  name: string | null;
  positions: LivePosition[];
  cash: number;
  invested_at_cost: number;
  market_value: number;
  equity: number;
  gain: number;
  gain_pct: number | null;
  as_of_utc: string;
  alerts: LiveAlert[];
  session: LiveSession;
  /** The account's equity from its first day (the book and the stored closes); empty for an account opened today. */
  history: HistoryPoint[];
  last_bar_day: string | null;
  clock_note: string | null;
  notes: string[];
}


// The report card of the Critic's rules (GET /api/proving/critic-report, `prometheus/proving/report_card.py`; Critic v2 rule C1). The server computes every number, the sentences and the readings; the page only draws them (P12).
export type CriticReading = "denies_winners" | "picks_losers" | "no_clear_difference" | "too_few" | "suitability";

/** One group of recommendations: how many have an excess return over SPY (`n`), how many closed and how many of those won, and their mean excess (a fraction). */
export interface CriticGroup {
  n: number;
  closed: number;
  won: number;
  mean_excess: number | null;
}

/** One theme's cell of a row: the recommendations the rule fired on in that theme against the others of the theme. */
export interface CriticThemeCell {
  n: number;
  closed: number;
  won: number;
  mean_excess: number | null;
  rest_n: number;
  rest_closed: number;
  rest_won: number;
  rest_mean_excess: number | null;
  gap: number | null;
  se: number | null;
  inside_chance: boolean | null;
  sentence_code: CriticReading;
}

/** One rule (a finding code), or the verdict itself (`kind: "verdict"`, `code: "denied"`: the denied against the approved). `gap` is `hit` minus `rest`: negative means the rule picks losers. */
export interface CriticRuleRow {
  code: string;
  kind: "blocker" | "human" | "warning" | "verdict";
  suitability: boolean;
  fired: number;
  hit: CriticGroup;
  rest: CriticGroup;
  gap: number | null;
  se: number | null;
  inside_chance: boolean | null;
  sentence_code: CriticReading;
  sentence: string;
  by_theme: Record<string, CriticThemeCell>;
}

export interface CriticFamilyRow {
  family: string;
  n: number;
  closed: number;
  won: number;
  mean_excess: number | null;
  denied: CriticGroup;
  approved: CriticGroup;
  gap: number | null;
  se: number | null;
  inside_chance: boolean | null;
  sentence_code: CriticReading;
  sentence: string;
  /** The plan-independent forward excess of the family's recommendations, of the ones the Critic denied and of the ones it approved (the same figures as `ForwardStats`); absent in a card made before it existed. */
  forward?: { all: ForwardStats[]; denied: ForwardStats[]; approved: ForwardStats[] };
}

export interface CriticVersionCard {
  config_version: string;
  reviewed: number;
  population: number;
  suitability_excluded: number;
  verdict: CriticRuleRow;
  rows: CriticRuleRow[];
  by_family: CriticFamilyRow[];
}

export interface CriticReport {
  min_sample: number;
  chance_band: number;
  limits: string[];
  sentences: Record<string, string>;
  runs: { run_id: string; kind: string; theme: string; prompt: string | null; speed: string; created_utc: string; reviewed: number }[];
  themes: string[];
  config_versions: string[];
  unmatched: number;
  by_config_version: Record<string, CriticVersionCard>;
}
