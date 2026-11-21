// fetch wrappers for the routes of spec 4.1. A refusal is {"error": "<message for the user>"} with a 4xx: it becomes an ApiError with that message.
import type { Coverage, CriticReport, DayView, LiveSnapshot, PlanAnswer, ProvingRun, RunIndexRow, TimeMachineRequest } from "../proving/types";
import type { AccountView, Health, RawRecords, RiskLevel, RunRequest, TradeMode } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "the desk server could not be reached");
  }
  if (response.status === 204) return undefined as T;
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const message = payload && typeof payload === "object" && "error" in payload && typeof (payload as { error: unknown }).error === "string" ? (payload as { error: string }).error : `the server refused the request (${response.status})`;
    throw new ApiError(response.status, message);
  }
  return payload as T;
}

export interface NewAccount {
  name: string;
  age: number;
  risk: RiskLevel;
  mode: TradeMode;
  /** A string: the server refuses a float (parse_money). */
  cash: string;
}

export const api = {
  health: () => call<Health>("GET", "/api/health"),
  account: (id: number) => call<AccountView>("GET", `/api/accounts/${id}`),
  createAccount: (body: NewAccount) => call<AccountView>("POST", "/api/accounts", body),
  patchAccount: (id: number, body: Partial<{ name: string; age: number; risk: RiskLevel; mode: TradeMode }>) => call<AccountView>("PATCH", `/api/accounts/${id}`, body),
  funds: (id: number, action: "add" | "remove", amount: string) => call<AccountView>("POST", `/api/accounts/${id}/funds`, { action, amount }),
  startRun: (body: RunRequest) => call<{ run_id: string }>("POST", "/api/runs", body),
  answer: (runId: string, questionId: string, answer: boolean | string) => call<void>("POST", `/api/runs/${runId}/answers`, { question_id: questionId, answer }),
  stop: (runId: string) => call<{ stopping: boolean }>("POST", `/api/runs/${runId}/stop`),
  records: (runId: string, key: string) => call<RawRecords>("GET", `/api/runs/${runId}/records/${encodeURIComponent(key)}`),
  // the Proving Ground (design/proving_ground_spec.md section 9)
  coverage: () => call<Coverage>("GET", "/api/proving/coverage"),
  /** Free: validates and counts; reads nothing with a model. */
  provingPlan: (body: TimeMachineRequest) => call<PlanAnswer>("POST", "/api/proving/plan", body),
  /** PAID in a live server: one desk run per decision day. */
  startProving: (body: TimeMachineRequest) => call<{ run_id: string }>("POST", "/api/proving/runs", body),
  provingRuns: () => call<RunIndexRow[] | { runs: RunIndexRow[] }>("GET", "/api/proving/runs"),
  provingRun: (id: string) => call<ProvingRun>("GET", `/api/proving/runs/${encodeURIComponent(id)}`),
  /** Free: a saved run played again through the same events. */
  /** The desk of one decision day of a saved run: the Desk tab's own view (the route is `proving_views.day_view`'s). */
  provingDay: (id: string, day: string) => call<DayView>("GET", `/api/proving/runs/${encodeURIComponent(id)}/days/${encodeURIComponent(day)}`),
  provingRecords: (id: string, day: string, key: string) => call<RawRecords>("GET", `/api/proving/runs/${encodeURIComponent(id)}/days/${encodeURIComponent(day)}/records/${encodeURIComponent(key)}`),
  replayProving: (id: string) => call<{ run_id: string }>("POST", `/api/proving/runs/${encodeURIComponent(id)}/replay`),
  deleteProving: (id: string) => call<void>("DELETE", `/api/proving/runs/${encodeURIComponent(id)}`),
  track: () => call<ProvingRun>("GET", "/api/proving/track"),
  /** Free: every finding code of the Critic scored against the saved outcomes (the report card of its rules). */
  criticReport: () => call<CriticReport>("GET", "/api/proving/critic-report"),
  live: (accountId: number) => call<LiveSnapshot>("GET", `/api/proving/live/${accountId}`),
};
