import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, ApiError } from "./api/client";
import type { Health, TradeMode, WorkRow } from "./api/types";
import { ActivityFeed } from "./components/ActivityFeed";
import { Header } from "./components/Header";
import { OpenOrdersCard } from "./components/OpenOrdersCard";
import { Pipeline } from "./components/Pipeline";
import { PortfolioCard } from "./components/PortfolioCard";
import { ProfileCard } from "./components/ProfileCard";
import { RecommendationPanel } from "./components/RecommendationPanel";
import { RequestCard } from "./components/RequestCard";
import { WorkOrderTable } from "./components/WorkOrderTable";
import { Card } from "./components/ui/Card";
import { loadFixture, parseFixtureParams } from "./devFixture";
import { formatHash, parseHash, type Route } from "./proving/hash";
import { ProvingGround } from "./proving/ProvingGround";
import { useProving } from "./proving/useProving";
import { ClockContext } from "./state/useNow";
import { useAccount } from "./state/useAccount";
import { useOrders } from "./state/useOrders";
import { useRun } from "./state/useRun";
import { useTheme } from "./state/useTheme";

/** A tab's content: hidden when another tab is shown, but still mounted, so a run and what was typed carry on. `display: contents` keeps its children the page's own flex items. */
function Pane({ hidden, children }: { hidden: boolean; children: ReactNode }) {
  return (
    <div className="tabpane" hidden={hidden}>
      {children}
    </div>
  );
}

const fixtureParams = typeof window === "undefined" ? null : parseFixtureParams(window.location.search);

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  const run = useRun();
  const proving = useProving();
  const account = useAccount(health);
  // The address says which tab: "" is the Desk, "#proving..." the Proving Ground (no router: the hash).
  const [route, setRoute] = useState<Route>(() => (typeof window === "undefined" ? { tab: "desk" } : parseHash(window.location.hash)));
  const [visitedProving, setVisitedProving] = useState(route.tab === "proving");
  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    if (route.tab === "proving") setVisitedProving(true);
  }, [route]);
  const navigate = useCallback((next: Route) => {
    setRoute(next);
    const want = formatHash(next);
    if (formatHash(parseHash(window.location.hash)) === want) return;
    if (want) window.location.hash = want;
    else window.history.pushState(null, "", window.location.pathname + window.location.search);
  }, []);
  const tab = route.tab;
  const provingRunning = proving.state.phase === "running";
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
  const attachProving = proving.attach;
  useEffect(() => {
    if (!health?.active_run || fixtureParams) return;
    // the health answer says which kind of run is active ("desk" or "proving"): the run goes to its own tab's state
    if (health.active_kind === "proving") attachProving(health.active_run);
    else attach(health.active_run);
  }, [health?.active_run, health?.active_kind, attach, attachProving]);

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
  // the waiting limit orders (entry 140): read again when a run ends, and whenever the watcher says something
  const orders = useOrders(view ? view.account.id : null, !replay && !fixtureParams, `${state.runId}:${state.phase}`);
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

  const sampleBlock = !replay && view?.account.sample ? "sample accounts are read-only here" : null;
  const requestBlocked = provingRunning ? "a Proving Ground test is running: wait for it to finish, or stop it" : sampleBlock;
  // the Proving Ground stays mounted (hidden) after its first visit, with the view it was on
  const lastProving = useRef<(Route & { tab: "proving" }) | null>(null);
  if (route.tab === "proving") lastProving.current = route;
  const proveRoute = lastProving.current;

  return (
    <ClockContext.Provider value={frozenNow}>
      <div className={`app${tab === "proving" ? " proving-on" : ""}`}>
        <Header health={health} account={view} theme={theme.theme} onToggleTheme={theme.toggle} onSwitch={account.load} onNewAccount={() => setOpenNew(true)} busy={running || provingRunning} tab={tab} onTab={(r) => navigate(r.tab === "proving" && lastProving.current ? lastProving.current : r)} deskRunning={running} provingRunning={provingRunning} />
        {offline && (
          <div className="bookbar" role="alert">
            <b>{offline}.</b> Start it with <code>./run_ui.sh</code> or <code>python -m prometheus.application.web</code>.
          </div>
        )}
        <Pane hidden={tab !== "desk"}>
        <Pipeline state={state} />
        <main className="grid">
          <div className="col">
            <PortfolioCard view={view} changed={changed} before={state.portfolioBefore} />
            <OpenOrdersCard view={orders.view} notices={orders.notices} error={orders.error} cancelling={orders.cancelling} onCancel={(id) => void orders.cancel(id)} readOnlyReason={readOnlyReason} />
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
                blockedReason={requestBlocked}
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
        </Pane>
        {visitedProving && proveRoute && (
          <Pane hidden={tab !== "proving"}>
            <ProvingGround route={proveRoute} onRoute={navigate} proving={proving} health={health} accountId={view ? view.account.id : null} defaultRisk={view?.account.risk ?? "moderate"} deskRunning={running} active={tab === "proving"} />
          </Pane>
        )}
      </div>
    </ClockContext.Provider>
  );
}
