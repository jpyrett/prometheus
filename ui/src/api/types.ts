// The shapes of design/ui_v1_spec.md sections 4.1 to 4.5, as the Python server (prometheus/application/web/) really sends them.

export type AgentKey = "account_manager" | "data_clerk" | "advisor" | "critic" | "trader";
export type LogAgent = AgentKey | "system" | "you";
export type LogLevel = "info" | "good" | "warn" | "error";
export type StageKey = "work_order" | "gather" | "recommend" | "critique" | "prepare" | "trade";
export type StageState = "pending" | "busy" | "waiting" | "done" | "skipped" | "failed";
export type TradeMode = "ask" | "auto";
export type RiskLevel = "conservative" | "moderate" | "aggressive";
export type RunStatus = "traded" | "no_trade" | "looked" | "stopped" | "failed";

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
  cash: number;
  invested: number;
  total: number;
  as_of_utc: string | null;
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
  valid_until_utc: string | null;
  requires_human: boolean;
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
  ts?: number;
}

// ---------------------------------------------------------------- the events (4.2)

export interface RunStartedData {
  run_id: string;
  mode: "live" | "replay";
  account_id: number;
  prompt: string;
  options: { strict: boolean; neutral: boolean; fetch: boolean };
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

export type QuestionKind = "confirm" | "clarify" | "approve" | "missing_company";

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
  | { seq: number; type: "activity"; data: ActivityData }
  | { seq: number; type: "run_finished"; data: RunFinishedData };

export type ServerEventType = ServerEvent["type"];
export const EVENT_TYPES: ServerEventType[] = ["run_started", "stage", "log", "question", "question_closed", "work_order", "portfolio", "activity", "run_finished"];

// ---------------------------------------------------------------- health and requests (4.1)

export interface Health {
  ok: boolean;
  mode: "live" | "replay";
  book: string;
  clock_note: string;
  clock_pill?: string;
  defaults: { prompt: string; strict: boolean; neutral: boolean; fetch: boolean };
  limits: { prompt_chars: number };
  active_run: string | null;
  tracing?: "langfuse" | "off";
  replay_file?: string | null;
}

export interface RunRequest {
  account_id: number;
  prompt: string;
  strict: boolean;
  neutral: boolean;
  fetch: boolean;
  /** The Mode switch: what the run uses. The server saves it through the Account Manager when it differs from the account's. */
  mode: TradeMode;
}

export interface RawRecords {
  [name: string]: unknown;
}
