// The shapes of design/ui_v1_spec.md sections 4.1 to 4.5, as the Python server (prometheus/application/web/) really sends them.
import type { PvDayData, PvDecisionData, PvFinishedData, PvOutcomeData, PvResearchData, PvStartedData } from "../proving/types";

export type AgentKey = "account_manager" | "data_clerk" | "advisor" | "critic" | "trader";
export type LogAgent = AgentKey | "system" | "you";
export type LogLevel = "info" | "good" | "warn" | "error";
export type StageKey = "work_order" | "gather" | "recommend" | "critique" | "prepare" | "trade";
export type StageState = "pending" | "busy" | "waiting" | "done" | "skipped" | "failed";
export type TradeMode = "ask" | "auto";
export type RiskLevel = "conservative" | "moderate" | "aggressive";
export type RunStatus = "traded" | "waiting" | "no_trade" | "looked" | "news" | "quote" | "stopped" | "failed";

export interface StageInfo {
  key: StageKey;
  label: string;
  agent: AgentKey;
  agent_label: string;
}

// ---------------------------------------------------------------- the account (4.5)

export interface Position {
  symbol: string;
  name: string | null;
  shares: number;
  cost_basis: number;
  value: number;
  /** The shares a waiting sale has set aside (design choices entry 140); absent in a recorded session. */
  reserved?: number;
}

export interface AccountInfo {
  id: number;
  name: string;
  age: number | null;
  risk: RiskLevel;
  mode: TradeMode;
  sample: boolean;
  read_only: boolean;
}

export interface AccountView {
  account: AccountInfo;
  positions: Position[];
  /** ALL the cash of the account. */
  cash: number;
  /** The part of it that may be spent: the cash less what waiting orders have set aside (entry 140). Absent in a recorded session: then it is all of the cash. */
  available?: number;
  /** The cash the waiting orders hold. */
  reserved_cash?: number;
  invested: number;
  total: number;
  as_of_utc: string | null;
}

/** A waiting limit order of the account (design choices entry 140): a DAY order, with its cash (a purchase) or its shares (a sale) set aside. `limit_price` is the ceiling of a purchase and the floor of a sale. */
export interface OpenOrder {
  order_id: number;
  account_id: number;
  symbol: string;
  company: string | null;
  side: "buy" | "sell";
  shares: number | null;
  dollars: number | null;
  limit_price: number;
  reserved_cash: number;
  reserved_shares: number;
  request: string;
  placed_utc: string;
  expires_utc: string;
  status: "open" | "filled" | "expired" | "cancelled" | "rejected";
  reason: string | null;
  /** Design choices entry 143: how the user permitted selling holdings to pay for this purchase WHEN ITS PRICE IS REACHED ("words": their own words, "setting": ALWAYS_RAISE_FUNDS); absent or null: never. */
  raise_funds_by?: "words" | "setting" | null;
}

export interface OrdersView {
  orders: OpenOrder[];
  reserved_cash: number;
  available: number;
  /** A watcher (a server) holds the advisory lock now: the orders are being watched. */
  watching: boolean;
}

/** One line the watcher or the Cancel route sent (`/api/orders/events`). */
export interface OrderNotice {
  type: "placed" | "filled" | "expired" | "cancelled" | "rejected";
  order_id: number;
  account_id: number;
  symbol: string;
  side: string;
  text: string;
  at: string;
  ts: number;
}

/** The `portfolio` event: the account view plus what changed against the first one of the run. */
export interface PortfolioData extends AccountView {
  changes: Record<string, number>;
  cash_change: number;
  cash_before: number;
  ts: number;
}

// ---------------------------------------------------------------- the work order (4.4)

export type RowGroup = "recommended" | "neutral" | "direct" | "scored" | "found" | "trade";

export interface Cited {
  text: string;
  ref: string | null;
}

export interface EvidenceItem {
  source: string | null;
  direction: string | null;
  quote: string | null;
  verified: boolean | null;
  counted: boolean | null;
  reason: string | null;
}

export interface ScoreBar {
  name: string;
  label: string;
  weight: number;
  value: number;
  available: boolean;
  contribution: number;
}

export interface KeyNumber {
  label: string;
  value: string;
}

export interface Plan {
  entry_zone: string | null;
  stop: string | null;
  target: string | null;
  order: string | null;
  time_stop: string | null;
  horizon: string | null;
  invalidation: string[];
  notes: string[];
}

export interface ReviewDimension {
  name: string;
  label: string;
  points: number;
  weight: number;
  applicable: boolean;
}

export interface Review {
  approved: boolean;
  score: number;
  grade: string;
  summary: string | null;
  dimensions: ReviewDimension[];
  blockers: { code: string; message: string }[];
  warnings: { code: string; message: string }[];
  risk: { level: string | null; score: number | null; tolerance: string | null; allowed: string | null } | null;
  /** Critic version 3: the three groups of judgment (news, performance, price), bad when a sign of the group fired. A denial needs two bad groups (or a fault, or a profile finding). Absent or empty on a review made before version 3. */
  groups?: ReviewGroup[];
  /** Critic version 3: the analysts' recommendation counts in one line, as the Critic words it. Null or absent when the review has none. */
  analysts?: string | null;
  /** Critic version 3: the net chart points as a bonus on the score (a bearish chart lowers it). */
  pattern_bonus?: number;
  valid_until_utc: string | null;
  requires_human: boolean;
}

export interface ReviewGroup {
  name: string;
  label: string;
  bad: boolean;
  codes: string[];
}

export interface DecisionCell {
  will_trade: string;
  words: string;
  action: string | null;
  shares: number | null;
  est_cost: number | null;
  /** An order in dollars (design choices entry 111): the most the purchase may cost; `shares` and `price` are then the estimate at the price it was sized at. null: an order in shares. */
  dollars?: number | null;
  price?: number | null;
  origin: string | null;
}

export interface TradeCell {
  status: string;
  side: string | null;
  shares: number | null;
  price: number | null;
  amount: number | null;
}

export interface OrderPart {
  decision?: DecisionCell & {
    reason: string | null;
    mode: string | null;
    price: number | null;
    balance: string | null;
    binding: string | null;
    allocation: string | null;
  };
  trade?: TradeCell & {
    trade_id: string | null;
    reason: string | null;
    reject_code: string | null;
    order_type: string | null;
    mode: string | null;
    slippage_bps: number | null;
    price_source: string | null;
    booked: unknown;
  };
}

export interface Strategy {
  name: string;
  passed: boolean;
  score: number | null;
  note: string | null;
}

export interface RowDetail {
  header?: { score: number | null; stance: string | null; rank: number | null; of: number | null; strategy: string | null };
  thesis?: string | null;
  bull?: Cited[];
  bear?: Cited[];
  evidence?: EvidenceItem[];
  score_build?: ScoreBar[];
  key_numbers?: KeyNumber[];
  plan?: Plan | null;
  review?: Review | null;
  order?: OrderPart | null;
  strategies?: Strategy[];
  message?: string;
  your_order?: string;
}

export interface WorkRow {
  key: string;
  symbol: string;
  company: string | null;
  rank: number | null;
  group: RowGroup;
  advisor: { score: number | null; stance: string | null; reason: string | null } | null;
  critic: { approved: boolean; score: number; grade: string } | null;
  decision: DecisionCell | null;
  trade: TradeCell | null;
  reason?: string;
  detail: RowDetail;
}

export interface WorkOrderView {
  request_id: string | null;
  status: string | null;
  kind: "research" | "direct";
  understood: string | null;
  rows: WorkRow[];
  /** The scored companies below the number the request asked for (entry 144): named, not listed as rows. */
  beyond?: { top: number; count: number; symbols: string[] } | null;
  ts?: number;
}

// ---------------------------------------------------------------- the news (design choices entry 149)

/** One ARTICLE of a news run (the chunks of an article are joined): `number` is its first chunk's source number and `numbers` every chunk's, so each `[n]` of the summary finds its article. Text only: nothing here is HTML. */
export interface NewsItem {
  number: number;
  numbers: number[];
  headline: string | null;
  publisher: string | null;
  date: string | null;
  published_utc: string | null;
  url: string | null;
  symbols: string[];
  cited: boolean;
}

/** The `news` event: the news of a request that only asked for the news (`views.news_view`). */
export interface NewsView {
  request_id: string | null;
  status: string | null;
  stocks: { symbol: string; company: string | null }[];
  not_found: { typed: string; reason: string }[];
  /** A theme searched as a term (no stock is named). */
  topic: string | null;
  /** The rolling window: how many days back from now; null for a dated window (`window`). */
  days: number | null;
  /** A DATED window (entry 151): the first and the last day asked for (ISO dates, US Eastern days, `to` included; the same day twice for one day). Absent or null for a rolling window (`days`). */
  window?: { from: string; to: string } | null;
  /** True when the latest news of every stock was fetched just before the search. */
  fetched: boolean;
  /** Why the latest news could not be fetched ("could not fetch the latest news (reason): showing the stored news"); null when it was, or for a theme. */
  fetch_note: string | null;
  summary: string;
  items: NewsItem[];
  problems: string[];
  notes: string[];
  no_supporting_text: boolean;
}

// ---------------------------------------------------------------- the current price (design choices entry 150)

/** One stock of a quote run: the Data Clerk's quote, copied (the change is Finnhub's, nothing is computed here), and `when` says in words when the price is from ("live quote, last trade 2026-11-20 15:59 ET", "market close of 2026-11-20 (simulated clock)"). */
export interface QuoteStock {
  symbol: string;
  company: string | null;
  price: number | null;
  change: number | null;
  change_percent: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  previous_close: number | null;
  as_of_utc: string | null;
  source: string | null;
  simulated: boolean;
  when: string;
}

/** The `quote` event: the current price of a request that only asked for the price (`views.quote_view`). The price is said in the run's output (one line for each stock, written by the server): no pane and no switch of the right column. */
export interface QuoteView {
  request_id: string | null;
  status: string | null;
  stocks: QuoteStock[];
  not_found: { typed: string; reason: string }[];
  problems: string[];
  notes: string[];
}

// ---------------------------------------------------------------- the events (4.2)

export interface RunStartedData {
  run_id: string;
  mode: "live" | "replay";
  account_id: number;
  prompt: string;
  options: { strict: boolean; neutral: boolean; fetch: boolean; raise_funds?: boolean; ignore_critic?: boolean; speed?: "fast" | "faithful" | null };
  trade_mode: TradeMode;
  started_utc: string;
  stages: StageInfo[];
  source?: string | null;
  ts: number;
}

export interface StageData {
  key: StageKey;
  state: Exclude<StageState, "pending">;
  node: string;
  seconds: number;
  ts: number;
}

export interface LogData {
  agent: LogAgent;
  level: LogLevel;
  text: string;
  block: boolean;
  ts: number;
}

export type QuestionKind = "confirm" | "clarify" | "approve" | "missing_company" | "raise_funds" | "funding";

/** A purchase a funding question is about (entry 142): what it costs at its price. */
export interface FundingBuy {
  symbol: string;
  shares: number;
  price: number | null;
  cost: number;
}

/** One sale of RaiseFunds' plan, as the funding card shows it (entry 142, R4): the price, about what it brings and the gain (negative: a loss). */
export interface FundingLeg {
  symbol: string;
  shares: number;
  price: number | null;
  amount: number;
  gain: number;
}

export interface ProposalItem {
  symbol: string;
  action: string;
  shares: number;
  est_cost: number;
  dollars?: number | null;
  price?: number | null;
}

export interface QuestionData {
  question_id: string;
  kind: QuestionKind;
  title: string;
  text: string;
  question?: string | null;
  facts?: string[];
  n?: number | null;
  of?: number | null;
  symbol?: string | null;
  shares?: number | null;
  est_cost?: number | null;
  dollars?: number | null;
  price?: number | null;
  action?: string | null;
  cash?: number | null;
  proposal?: ProposalItem[] | null;
  scorecard?: string | null;
  /** A stock the Data Clerk did not find (`missing_company`): what the user typed, the Data Clerk's reason, and the stocks that stay on the order. `question` is the Account Manager's sentence; `wording` says who wrote it (the model, checked by code, or the code itself). */
  typed?: string | null;
  reason?: string | null;
  kept?: KeptStock[];
  wording?: "model" | "code" | null;
  /** The cash does not cover the purchases and nothing permits selling holdings (`raise_funds`, entry 142, R3): the cash, what the purchases cost, how short it is, and the purchases. */
  shortfall?: number | null;
  cost?: number | null;
  buys?: FundingBuy[];
  /** The plan RaiseFunds made (`funding`, R4): the sales, the cash before (`cash`) and the cash after the sales and the purchases. */
  legs?: FundingLeg[];
  cash_after?: number | null;
  ts: number;
}

export interface KeptStock {
  typed: string;
  symbol: string;
  company: string;
  how?: string | null;
}

export interface QuestionClosedData {
  question_id: string;
  answer: boolean | string | null;
  by: "you" | "timeout" | "stopped";
  ts: number;
}

export interface ActivityData {
  calls: number;
  tokens_in: number;
  tokens_out: number;
  model: string | null;
  node: string | null;
  ts: number;
}

export interface RunFinishedData {
  status: RunStatus;
  headline: string;
  reasons: string[];
  path: string[];
  seconds: Record<string, number>;
  total_seconds: number;
  ts: number;
}

export type ServerEvent =
  | { seq: number; type: "run_started"; data: RunStartedData }
  | { seq: number; type: "stage"; data: StageData }
  | { seq: number; type: "log"; data: LogData }
  | { seq: number; type: "question"; data: QuestionData }
  | { seq: number; type: "question_closed"; data: QuestionClosedData }
  | { seq: number; type: "work_order"; data: WorkOrderView & { ts: number } }
  | { seq: number; type: "portfolio"; data: PortfolioData }
  | { seq: number; type: "news"; data: NewsView & { ts: number } }
  | { seq: number; type: "quote"; data: QuoteView & { ts: number } }
  | { seq: number; type: "activity"; data: ActivityData }
  | { seq: number; type: "run_finished"; data: RunFinishedData }
  // the Proving Ground's run (design/proving_ground_spec.md section 9): the Desk's reducer ignores them, the Proving Ground's folds them
  | { seq: number; type: "pv_research"; data: PvResearchData }
  | { seq: number; type: "pv_started"; data: PvStartedData }
  | { seq: number; type: "pv_day"; data: PvDayData }
  | { seq: number; type: "pv_decision"; data: PvDecisionData }
  | { seq: number; type: "pv_outcome"; data: PvOutcomeData }
  | { seq: number; type: "pv_finished"; data: PvFinishedData };

export type ServerEventType = ServerEvent["type"];
export const EVENT_TYPES: ServerEventType[] = [
  "run_started",
  "stage",
  "log",
  "question",
  "question_closed",
  "work_order",
  "portfolio",
  "news",
  "quote",
  "activity",
  "run_finished",
  "pv_research",
  "pv_started",
  "pv_day",
  "pv_decision",
  "pv_outcome",
  "pv_finished",
];

// ---------------------------------------------------------------- health and requests (4.1)

export interface Health {
  ok: boolean;
  mode: "live" | "replay";
  book: string;
  clock_note: string;
  clock_pill?: string;
  defaults: { prompt: string; strict: boolean; neutral: boolean; fetch: boolean; raise_funds?: boolean; ignore_critic?: boolean };
  limits: { prompt_chars: number };
  active_run: string | null;
  /** What the active run is, when the server says (a desk run or a Proving Ground run): the page attaches it to the right tab. */
  active_kind?: "desk" | "proving" | null;
  tracing?: "langfuse" | "off";
  replay_file?: string | null;
}

export interface RunRequest {
  account_id: number;
  prompt: string;
  strict: boolean;
  neutral: boolean;
  fetch: boolean;
  /** The box "Always raise funds" (entry 142, R12): the setting ALWAYS_RAISE_FUNDS for this one run; off unless ticked. */
  raise_funds: boolean;
  /** The box "Ignore Critic" (entry 148): skip the Critic's step for this one run; off unless ticked. Strict cannot go with it. */
  ignore_critic: boolean;
  /** The Mode switch: what the run uses. The server saves it through the Account Manager when it differs from the account's. */
  mode: TradeMode;
}

export interface RawRecords {
  [name: string]: unknown;
}
