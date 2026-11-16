import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type { Health, TradeMode, WorkRow } from "./api/types";
import { ActivityFeed } from "./components/ActivityFeed";
import { Header } from "./components/Header";
import { Pipeline } from "./components/Pipeline";
import { PortfolioCard } from "./components/PortfolioCard";
import { ProfileCard } from "./components/ProfileCard";
import { RecommendationPanel } from "./components/RecommendationPanel";
import { RequestCard } from "./components/RequestCard";
import { WorkOrderTable } from "./components/WorkOrderTable";
import { Card } from "./components/ui/Card";
import { loadFixture, parseFixtureParams } from "./devFixture";
import { ClockContext } from "./state/useNow";
import { useAccount } from "./state/useAccount";
import { useRun } from "./state/useRun";
import { useTheme } from "./state/useTheme";

const fixtureParams = typeof window === "undefined" ? null : parseFixtureParams(window.location.search);

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  const run = useRun();
  const account = useAccount(health);
  const theme = useTheme();
  const [prompt, setPrompt] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [picked, setPicked] = useState(false);
  const [openNew, setOpenNew] = useState(false);
  // The trade mode on show (Jacob: the Mode switch on the request card): the account's own once there is an account, else the page's value, which Create account sends.
  const [newMode, setNewMode] = useState<TradeMode>("ask");
  const [frozenNow, setFrozenNow] = useState<number | null>(null);
  const state = run.state;
  const running = state.phase === "running";

  // The server's health: the mode, the defaults, the book, the active run.
  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setPrompt((p) => p ?? h.defaults.prompt);
      })
      .catch((e) => setOffline(e instanceof ApiError ? e.message : "the desk server could not be reached"));
  }, []);

  // A run already going on (the page was reloaded): attach and everything redraws from the replayed events.
  const { attach, feed } = run;
  useEffect(() => {
    if (health?.active_run && !fixtureParams) attach(health.active_run);
  }, [health?.active_run, attach]);

  // The recorded stream of a screenshot (?fixture=...): fed once the account has loaded, so its portfolio events land on top of it.
  const fixtureFed = useRef(false);
  useEffect(() => {
    if (!fixtureParams || fixtureFed.current || !health || account.status === "loading") return;
    fixtureFed.current = true;
    if (fixtureParams.theme) theme.set(fixtureParams.theme, false);
    void loadFixture(fixtureParams).then((f) => {
      if (!f) return;
      setFrozenNow(f.now);
      feed(f.events);
    });
  }, [health, account.status, feed, theme]);

  // A run's portfolio replaces the account's holdings and cash.
  const { applyPortfolio } = account;
  useEffect(() => {
    if (state.portfolio) applyPortfolio(state.portfolio);
  }, [state.portfolio, applyPortfolio]);

  // A new run starts with nothing chosen.
  useEffect(() => {
    setPicked(false);
    setSelected(null);
  }, [state.runId]);

  // The first row is selected when recommendations first arrive, unless the user already chose one.
  const rows = useMemo<WorkRow[]>(() => state.workOrder?.rows ?? [], [state.workOrder]);
  useEffect(() => {
    if (rows.length === 0) return;
    const there = selected !== null && rows.some((r) => r.key === selected);
    if (picked && there) return;
    const target = rows.find((r) => r.group !== "found")?.key ?? null;
    if (picked && !there) setPicked(false);
    if (target !== selected) setSelected(target);
  }, [rows, picked, selected]);

  const choose = useCallback((key: string) => {
    setPicked(true);
    setSelected(key);
  }, []);

  const rowOf = useCallback((symbol: string) => rows.find((r) => r.symbol === symbol), [rows]);
  const selectedRow = rows.find((r) => r.key === selected) ?? null;
  const view = account.view;
  const replay = health?.mode === "replay";
  const readOnlyReason = replay ? "a recorded session: its account is read-only" : view?.account.sample ? "sample accounts are read-only here" : view?.account.read_only ? "this account is read-only" : null;
  const changed = state.portfolio ? { changes: state.portfolio.changes, cash_change: state.portfolio.cash_change, cash_before: state.portfolio.cash_before } : null;

  const mode: TradeMode = view ? view.account.mode : newMode;
  const { patch } = account;
  const changeMode = useCallback((next: TradeMode) => (view ? patch({ mode: next }) : (setNewMode(next), Promise.resolve())), [view, patch]);

  const accountActions = useMemo(
    () => ({
      create: async (form: Parameters<typeof account.create>[0]) => {
        await account.create(form);
        setOpenNew(false);
      },
      patch: account.patch,
      funds: account.funds,
    }),
    [account.create, account.patch, account.funds], // eslint-disable-line react-hooks/exhaustive-deps
  );

  return (
    <ClockContext.Provider value={frozenNow}>
      <div className="app">
        <Header health={health} account={view} theme={theme.theme} onToggleTheme={theme.toggle} onSwitch={account.load} onNewAccount={() => setOpenNew(true)} busy={running} />
        {offline && (
          <div className="bookbar" role="alert">
            <b>{offline}.</b> Start it with <code>./run_ui.sh</code> or <code>python -m prometheus.application.web</code>.
          </div>
        )}
        <Pipeline state={state} />
        <main className="grid">
          <div className="col">
            <PortfolioCard view={view} changed={changed} before={state.portfolioBefore} />
            <ProfileCard view={view} running={running} readOnlyReason={readOnlyReason} loadError={account.loadError} actions={accountActions} forceOpen={openNew} onCancel={() => setOpenNew(false)} newMode={mode} />
          </div>
          <div className="col">
            {prompt !== null && health ? (
              <RequestCard
                prompt={prompt}
                onPrompt={setPrompt}
                maxChars={health.limits.prompt_chars}
                accountId={view ? view.account.id : null}
                running={running}
                starting={run.starting}
                defaults={health.defaults}
                onGo={(request) => void run.start(request)}
                onStop={() => void run.stop()}
                error={run.error}
                blockedReason={!replay && view?.account.sample ? "sample accounts are read-only here" : null}
                mode={mode}
                onMode={changeMode}
                modeLockedReason={readOnlyReason}
                replay={replay}
              />
            ) : (
              <Card title="Your request" right="in your own words">
                <div className="card-b">
                  <div className="empty">
                    <b>{offline ? "The desk is not reachable" : "Connecting to the desk…"}</b>
                  </div>
                </div>
              </Card>
            )}
            <ActivityFeed state={state} answer={run.answer} rowOf={rowOf} reconnecting={run.reconnecting} />
          </div>
          <div className="col right">
            <WorkOrderTable state={state} selectedKey={selected} onSelect={choose} />
            <RecommendationPanel row={selectedRow} runId={state.runId} waitingForAdvisor={rows.length > 0 || running} />
          </div>
        </main>
      </div>
    </ClockContext.Provider>
  );
}
