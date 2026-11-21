// A run on the page (spec 13.2, "during and after a run"): the hero, the equity chart with its research strip and playhead, the outcome wall, the quality row, the day's desk and the caveats.
// The Time Machine uses it for a live run, a saved run and one played again; the Track Record uses it for its own run. It only draws what the state holds.
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { WorkOrderView } from "../api/types";
import { EquityChart } from "../charts/EquityChart";
import { Card } from "../components/ui/Card";
import { clock } from "../format";
import { Caveats } from "./Caveats";
import { DayDesk } from "./DayDesk";
import { Hero } from "./Hero";
import { OutcomeWall } from "./OutcomeWall";
import type { ProvingState } from "./provingReducer";
import { QualityRow } from "./QualityRow";
import { TimingCard } from "./TimingCard";
import { dayMarks, researchLine } from "./selectors";
import type { RecOutcome } from "./types";
import { useSweep } from "./useSweep";

export interface RunViewProps {
  state: ProvingState;
  benchmark?: string;
  minSample?: number;
  /** Buttons for the hero (Stop, Watch it again, a trace link). */
  actions?: ReactNode;
  /** The wall may show fewer outcomes than the run holds (the Track Record's source filter); the figures above still count them all. */
  wallOutcomes?: RecOutcome[];
  /** A line under the equity chart's title (the Track Record says what its curve is). */
  chartNote?: string | null;
  kicker?: string;
}

export function RunView({ state, benchmark = "SPY", minSample = 20, actions, wallOutcomes, chartNote, kicker }: RunViewProps) {
  const sweep = useSweep(state.points.length, state.streamed);
  const [day, setDay] = useState<string | null>(null);
  const desk = useRef<HTMLDivElement>(null);
  const running = state.phase === "running";
  const marks = state.kind === "track" ? [] : dayMarks(state);
  const showStrip = state.kind !== "track" && state.plannedDays.length > 0;
  // A day's desk is on the stream only when the server sends it; otherwise it is read from the server when the day is opened (the same view the Desk tab draws), once.
  const [fetched, setFetched] = useState<Record<string, WorkOrderView>>({});
  const [deskError, setDeskError] = useState<string | null>(null);
  const asked = useRef<Set<string>>(new Set());
  const savedId = state.savedId;
  useEffect(() => {
    setFetched({});
    setDeskError(null);
    asked.current = new Set();
  }, [savedId]);
  const desks = { ...fetched, ...state.desks };
  const needDesk = useCallback(
    (d: string) => {
      if (!savedId || asked.current.has(d)) return;
      asked.current.add(d);
      api
        .provingDay(savedId, d)
        .then((v) => setFetched((f) => ({ ...f, [d]: v.view })))
        .catch((e) => (asked.current.delete(d), setDeskError(e instanceof Error ? e.message : "the day's desk could not be read")));
    },
    [savedId],
  );
  const openDay = (d: string) => {
    setDeskError(null);
    setDay((cur) => (cur === d ? null : d));
    if (day !== d) needDesk(d);
  };
  const view = day ? desks[day] : undefined;
  const loadRecords = useCallback((key: string) => api.provingRecords(savedId ?? "", day ?? "", key), [savedId, day]);

  // opening a day scrolls its desk into view
  useEffect(() => {
    if (day && view) desk.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }, [day, view]);
  // a new run starts with no day open
  useEffect(() => setDay(null), [state.runId]);

  const domain = state.range ? { first: state.range.first_day, last: state.range.last_bar_day ?? state.range.last_day } : null;
  const walking = state.step === "walk" && running;

  return (
    <div className="runview">
      <Hero state={state} shown={sweep.shown} benchmark={benchmark} actions={actions} kicker={kicker} />
      <Card title={state.kind === "track" ? "$1 in every traded recommendation, against SPY" : "The desk's account, day by day"} className="equity-card" right={state.range ? `through ${state.range.last_bar_day ?? state.range.last_day}` : undefined}>
        <div className="card-b">
          {chartNote && <p className="chart-note muted">{chartNote}</p>}
          <EquityChart
            points={state.points}
            shown={sweep.shown}
            sweeping={sweep.sweeping || walking}
            events={state.events}
            days={marks}
            research={showStrip}
            startCash={state.startingCash ?? state.points[0]?.equity ?? 0}
            domain={domain}
            benchmarkName={benchmark}
            unit={state.unit}
            maxDrawdown={state.stats?.portfolio?.max_drawdown_pct ?? null}
            selectedDay={day}
            onOpenDay={openDay}
            status={running ? researchLine(state) : null}
            toolbar={
              sweep.sweeping ? (
                <button type="button" className="linkbtn skip" onClick={sweep.skip}>
                  Skip to the end
                </button>
              ) : undefined
            }
          />
          {state.points.length === 0 && !running && state.kind !== "track" && <p className="muted">This run has no equity curve: nothing was bought.</p>}
          {state.points.length === 0 && state.kind === "track" && <p className="muted">No traded recommendation has a bar to follow yet, so there is no curve.</p>}
        </div>
      </Card>
      {day && (
        <div ref={desk}>
          {view ? (
            <DayDesk day={day} view={view} onClose={() => setDay(null)} loadRecords={savedId ? loadRecords : undefined} />
          ) : (
            <Card title={`The desk on ${day}`} right={<button type="button" className="linkbtn" onClick={() => setDay(null)}>Close</button>}>
              <div className="empty">
                <b>{deskError ?? "Reading the desk of that day…"}</b>
              </div>
            </Card>
          )}
        </div>
      )}
      <OutcomeWall outcomes={wallOutcomes ?? state.outcomes} desks={desks} onNeedDesk={needDesk} freshKeys={state.freshKeys} benchmark={benchmark} sources={state.kind === "track" ? state.sources : undefined} />
      <QualityRow outcomes={state.outcomes} stats={state.stats} minSample={minSample} benchmark={benchmark} />
      {state.kind !== "track" && <TimingCard state={state} />}
      <Caveats state={state} />
      {state.log.length > 0 && (
        <details className="pv-log">
          <summary>Log of the run ({state.log.length} lines)</summary>
          <ol className="num" role="log">
            {state.log.map((l) => (
              <li key={l.id} className={l.level}>
                <span className="muted">{clock(l.ts)}</span> {l.text}
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
