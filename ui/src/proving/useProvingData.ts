// The Proving Ground's reads: the coverage, the saved runs, the plan line, the Track Record and the Live snapshot. Each is a small hook over `api`; the free routes are called as the page needs them,
// and nothing here starts a paid run (that is `useProving().launch`, behind the Launch button).
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Coverage, CriticReport, LiveSnapshot, PlanAnswer, ProvingRun, RunIndexRow, TimeMachineRequest } from "./types";

export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

function words(e: unknown, fallback: string): string {
  return e instanceof ApiError ? e.message : fallback;
}

function useLoaded<T>(load: () => Promise<T>, deps: unknown[], enabled = true): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    setLoading(true);
    load()
      .then((d) => live && (setData(d), setError(null)))
      .catch((e) => live && setError(words(e, "the server could not be reached")))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, enabled, ...deps]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

export const useCoverage = (enabled = true) => useLoaded<Coverage>(() => api.coverage(), [], enabled);
export const useTrack = (enabled = true) => useLoaded<ProvingRun>(() => api.track(), [], enabled);
export const useCriticReport = (enabled = true) => useLoaded<CriticReport>(() => api.criticReport(), [], enabled);

/** The saved runs, newest first as the server sends them (the answer may be a list or {runs: [...]}). */
export function useRunIndex(enabled = true, refreshKey: unknown = 0): Loaded<RunIndexRow[]> {
  return useLoaded<RunIndexRow[]>(async () => {
    const answer = await api.provingRuns();
    return Array.isArray(answer) ? answer : answer.runs;
  }, [refreshKey], enabled);
}

/** The plan line under the launch pad: asked again (free) 350 ms after the inputs stop changing. A refusal is the server's own sentence. */
export function usePlan(request: TimeMachineRequest | null, enabled = true): { plan: PlanAnswer | null; problem: string | null; loading: boolean } {
  const [plan, setPlan] = useState<PlanAnswer | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const key = request ? JSON.stringify(request) : "";
  const latest = useRef(0);
  useEffect(() => {
    if (!enabled || !request) return;
    const mine = ++latest.current;
    setLoading(true);
    const timer = setTimeout(() => {
      api
        .provingPlan(request)
        .then((p) => {
          if (mine !== latest.current) return;
          setPlan(p);
          setProblem(p.problems.length ? p.problems.join("; ") : null);
        })
        .catch((e) => {
          if (mine !== latest.current) return;
          setPlan(null);
          setProblem(words(e, "the plan could not be read"));
        })
        .finally(() => mine === latest.current && setLoading(false));
    }, 350);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  return { plan, problem, loading };
}

const LIVE_RETRY_MS = 5000;

/** The Live snapshot of an account: one now, then a new one on every message of the server's stream (it polls while the market is open). The view only reads. */
export function useLive(accountId: number | null, enabled: boolean): Loaded<LiveSnapshot> & { streaming: boolean } {
  const first = useLoaded<LiveSnapshot>(() => api.live(accountId as number), [accountId], enabled && accountId !== null);
  const [latest, setLatest] = useState<LiveSnapshot | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    setLatest(null);
  }, [accountId, enabled]);
  useEffect(() => {
    if (!enabled || accountId === null || typeof EventSource === "undefined") return;
    const source = new EventSource(`/api/proving/live/${accountId}/events`);
    let again: ReturnType<typeof setTimeout> | undefined;
    const take = (m: MessageEvent) => {
      try {
        setLatest(JSON.parse(m.data) as LiveSnapshot);
        setStreaming(true);
      } catch {
        /* a bad message is skipped */
      }
    };
    source.addEventListener("message", take);
    source.addEventListener("snapshot", take as EventListener);
    source.onerror = () => {
      setStreaming(false);
      // A stream the server refused (it was restarting, or answered with an error) is CLOSED for good by the browser: open a new one, so the view recovers by itself.
      if (source.readyState === EventSource.CLOSED && again === undefined) again = setTimeout(() => setAttempt((n) => n + 1), LIVE_RETRY_MS);
    };
    return () => {
      if (again !== undefined) clearTimeout(again);
      source.close();
    };
  }, [accountId, enabled, attempt]);
  // The first read failed and no snapshot has come: ask again until one does (a server that was down or restarting must not leave its error on the screen).
  const stuck = enabled && accountId !== null && first.error !== null && latest === null && !first.loading;
  const { reload } = first;
  useEffect(() => {
    if (!stuck) return;
    const timer = setTimeout(reload, LIVE_RETRY_MS);
    return () => clearTimeout(timer);
  }, [stuck, reload, first.error]);
  return { ...first, error: latest ? null : first.error, data: latest ?? first.data, streaming };
}
