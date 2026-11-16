// The account (spec 5.4, D7): loaded from GET /api/accounts/{id}, remembered by its number in localStorage. A `portfolio` event of a run replaces its portfolio.
import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type NewAccount } from "../api/client";
import type { AccountView, Health, PortfolioData, RiskLevel, TradeMode } from "../api/types";

const KEY = "prometheus.account";

function remembered(): number | null {
  try {
    const raw = localStorage.getItem(KEY);
    const n = raw === null ? NaN : Number(raw);
    return Number.isInteger(n) && n >= 1 ? n : null;
  } catch {
    return null;
  }
}

function remember(id: number | null) {
  try {
    if (id === null) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, String(id));
  } catch {
    /* private window or blocked storage: the account just is not remembered */
  }
}

export type AccountStatus = "loading" | "none" | "ready" | "error";

export function useAccount(health: Health | null) {
  const [view, setView] = useState<AccountView | null>(null);
  const [status, setStatus] = useState<AccountStatus>("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const replay = health?.mode === "replay";

  const load = useCallback(async (id: number) => {
    setStatus("loading");
    setLoadError(null);
    try {
      const v = await api.account(id);
      setView(v);
      setStatus("ready");
      if (v.account.id > 0) remember(v.account.id);
      return v;
    } catch (e) {
      setView(null);
      setStatus("error");
      setLoadError(e instanceof ApiError ? e.message : "the account could not be read");
      throw e;
    }
  }, []);

  useEffect(() => {
    if (!health) return;
    if (health.mode === "replay") {
      load(0).catch(() => undefined);
      return;
    }
    const id = remembered();
    if (id === null) setStatus("none");
    else load(id).catch(() => undefined);
  }, [health, load]);

  const create = useCallback(async (form: NewAccount) => {
    const v = await api.createAccount(form);
    setView(v);
    setStatus("ready");
    remember(v.account.id);
    return v;
  }, []);

  const patch = useCallback(
    async (body: Partial<{ name: string; age: number; risk: RiskLevel; mode: TradeMode }>) => {
      if (!view) throw new ApiError(0, "there is no account");
      const v = await api.patchAccount(view.account.id, body);
      setView(v);
      return v;
    },
    [view],
  );

  const funds = useCallback(
    async (action: "add" | "remove", amount: string) => {
      if (!view) throw new ApiError(0, "there is no account");
      const v = await api.funds(view.account.id, action, amount);
      setView(v);
      return v;
    },
    [view],
  );

  /** A run's `portfolio` event replaces the holdings and the cash (the person and the profile stay as the account says). */
  const applyPortfolio = useCallback((p: PortfolioData) => {
    setView((current) => (current === null || current.account.id === p.account.id || replay ? { account: p.account, positions: p.positions, cash: p.cash, invested: p.invested, total: p.total, as_of_utc: p.as_of_utc } : current));
  }, [replay]);

  const forget = useCallback(() => {
    remember(null);
    setView(null);
    setStatus("none");
    setLoadError(null);
  }, []);

  return { view, status, loadError, replay, load, create, patch, funds, applyPortfolio, forget };
}
