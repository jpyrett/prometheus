// The Proving Ground tab (spec 13.2): three views of one question, "did the desk's recommendations work?". Time Machine (the backtest: launch pad, then the run), Track Record and Live.
// It is a report, so unlike the Desk tab it scrolls. The run's own state lives in the App (so a test goes on while the Desk tab is shown); everything else is read here, free.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Health, RiskLevel } from "../api/types";
import { Button } from "../components/ui/Button";
import type { Route } from "./hash";
import { formatHash } from "./hash";
import { LaunchPad } from "./LaunchPad";
import { Live } from "./Live";
import { RunView } from "./RunView";
import { SavedRuns } from "./SavedRuns";
import { TrackRecord } from "./TrackRecord";
import type { useProving } from "./useProving";
import { useCoverage, useRunIndex } from "./useProvingData";

export interface ProvingGroundProps {
  route: Route & { tab: "proving" };
  onRoute: (route: Route) => void;
  proving: ReturnType<typeof useProving>;
  health: Health | null;
  accountId: number | null;
  defaultRisk: RiskLevel;
  deskRunning: boolean;
  /** The tab is on screen (what is read for it is read only then). */
  active: boolean;
}

const VIEWS = [
  { view: "time_machine", label: "Time Machine", what: "replay the desk over past months" },
  { view: "track", label: "Track Record", what: "score what the desk already recommended" },
  { view: "live", label: "Live", what: "watch the paper account against its plans" },
] as const;

export function ProvingGround({ route, onRoute, proving, health, accountId, defaultRisk, deskRunning, active }: ProvingGroundProps) {
  const { state } = proving;
  const view = route.view === "run" ? "time_machine" : route.view;
  const replay = health?.mode === "replay";
  const running = state.phase === "running";
  const coverage = useCoverage(active && view === "time_machine");
  const [listKey, setListKey] = useState(0);
  const runs = useRunIndex(active && view === "time_machine", listKey);
  const [problem, setProblem] = useState<string | null>(null);

  // a saved run named in the address is opened (unless it is already what is on show)
  // Once per address: "Watch it again" and "New test" reset the run's state, and the address must not open the saved run again over the stream that is about to be played.
  const { open } = proving;
  const wanted = route.view === "run" ? route.runId : null;
  const openedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!wanted) {
      openedFor.current = null;
      return;
    }
    if (!active || running || openedFor.current === wanted) return;
    openedFor.current = wanted;
    if (state.savedId !== wanted) void open(wanted);
  }, [active, wanted, state.savedId, running, open]);

  // a run that has just finished is in the list
  useEffect(() => {
    if (state.phase === "finished" && state.streamed) setListKey((k) => k + 1);
  }, [state.phase, state.streamed]);

  const newTest = useCallback(() => {
    proving.reset();
    onRoute({ tab: "proving", view: "time_machine" });
  }, [proving, onRoute]);

  const blocked = replay ? "A recorded session: no model is called, so no new test can be paid for. Open or watch a saved run instead." : deskRunning ? "A desk run is active: wait for it to finish, or stop it." : null;
  const showRun = state.phase !== "idle" && state.kind !== "track";

  return (
    <div className="proving">
      <nav className="subtabs" aria-label="Proving Ground views">
        {VIEWS.map((v) => (
          <a key={v.view} href={formatHash({ tab: "proving", view: v.view } as Route) || "#proving"} className={`subtab${view === v.view ? " on" : ""}`} aria-current={view === v.view ? "page" : undefined} onClick={(e) => (e.preventDefault(), onRoute({ tab: "proving", view: v.view } as Route))}>
            <b>{v.label}</b>
            <small>{v.what}</small>
            {v.view === "time_machine" && running && <i className="dot pulse busy-dot" style={{ ["--agent" as string]: "var(--brand)" }} title="a test is running" />}
          </a>
        ))}
      </nav>

      {view === "time_machine" && (
        <>
          {showRun ? (
            <RunView
              state={state}
              actions={
                <>
                  {running ? (
                    <Button variant="ghost" className="stop" onClick={() => void proving.stop()}>
                      Stop the test
                    </Button>
                  ) : (
                    <>
                      <Button onClick={newTest}>New test</Button>
                      {state.savedId && (
                        <Button disabled={deskRunning} title={deskRunning ? "a desk run is active" : "Plays the saved run again through the same events (free: no model is called)"} onClick={() => void proving.watch(state.savedId!)}>
                          Watch it again
                        </Button>
                      )}
                    </>
                  )}
                  {state.langsmith?.url && (
                    <a className="btn ghost" href={state.langsmith.url} target="_blank" rel="noreferrer noopener">
                      Trace
                    </a>
                  )}
                </>
              }
            />
          ) : (
            <>
              <LaunchPad
                coverage={coverage.data}
                coverageError={coverage.error}
                blockedReason={blocked}
                starting={proving.starting}
                running={running}
                defaultRisk={defaultRisk}
                error={proving.error}
                onLaunch={(request) => void proving.launch(request)}
                onStop={() => void proving.stop()}
              />
              <SavedRuns
                rows={runs.data}
                error={problem ?? runs.error}
                disabled={deskRunning || running}
                onOpen={(id) => onRoute({ tab: "proving", view: "run", runId: id })}
                onWatch={(id) => void proving.watch(id)}
                onDelete={async (id) => {
                  try {
                    await api.deleteProving(id);
                    setProblem(null);
                    runs.reload();
                  } catch (e) {
                    setProblem(e instanceof ApiError ? e.message : "the run could not be deleted");
                  }
                }}
              />
            </>
          )}
          {proving.error && showRun && (
            <p className="err" role="alert">
              {proving.error}
            </p>
          )}
        </>
      )}
      {view === "track" && <TrackRecord enabled={active} onTimeMachine={() => onRoute({ tab: "proving", view: "time_machine" })} />}
      {view === "live" && <Live accountId={accountId} enabled={active} />}
    </div>
  );
}
