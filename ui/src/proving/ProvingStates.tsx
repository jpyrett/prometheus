// The Proving Ground's part of the states sheet (?states): its charts and views drawn from the REAL recordings (tests/fixtures/proving_real/), so the states cannot drift from the page.
// Where a state needs data no recording covers (a research tile that is busy, a failed day) the state is laid over real dates and marked illustrative.
import { EquityChart, type DayMark, type TileState } from "../charts/EquityChart";
import { PlanLadder } from "../charts/PlanLadder";
import { PlanChart } from "../charts/PlanChart";
import { Card } from "../components/ui/Card";
import { fastDayView, fastRun, liveOilSnapshot, livePlanSnapshot, planFaithfulFixture, planFixture, savedRunsIndex, trackRun } from "../test/proving";
import type { WorkOrderView } from "../api/types";
import { Hero } from "./Hero";
import { Live } from "./Live";
import { OutcomeWall } from "./OutcomeWall";
import { QualityRow } from "./QualityRow";
import { stateOfRun } from "./provingReducer";
import { RunView } from "./RunView";
import { SavedRuns } from "./SavedRuns";
import { SpeedSwitch } from "./SpeedSwitch";
import type { RecOutcome } from "./types";

function pick(...tests: ((o: RecOutcome) => boolean)[]): RecOutcome[] {
  return tests.map((t) => trackRun.outcomes.find(t)).filter((o): o is RecOutcome => !!o);
}

export default function ProvingStates() {
  const track = stateOfRun(trackRun);
  const cards = pick((o) => o.result === "won", (o) => o.result === "lost" && o.traded, (o) => o.result === "open" && o.traded, (o) => o.result === "open" && o.hypothetical, (o) => o.result === "no_data");
  const dates = track.points.map((p) => p.date);
  const states: TileState[] = ["done", "cached", "busy", "failed", "pending"];
  const marks: DayMark[] = dates.slice(0, 5).map((day, i) => ({ day, state: states[i]!, recommended: i < 2 ? 5 : 0, approved: i === 0 ? 3 : 0, denied: i === 0 ? 2 : 0, traded: 0, hasDesk: i < 2 }));

  // the real fast six-month run, with the desk of its first decision day (both recorded from the replay server by scripts/record_proving.mjs)
  const desks: Record<string, WorkOrderView> = { [fastDayView.day]: fastDayView.view };

  return (
    <div className="gallery pv-states">
      <h1>Proving Ground</h1>
      <p>The charts and views of the new tab, from the real recordings of tests/fixtures/proving_real/. States laid over real days are marked illustrative.</p>
      <div className="pv-states-stack">
        <div className="pv-run">
          <RunView state={stateOfRun(fastRun, desks)} />
        </div>
        <Card title="Speed switch: Fast and Faithful, each with its own time (the two recorded plans)">
          <div className="card-b pv-speeds">
            <SpeedSwitch value="fast" onChange={() => undefined} plans={{ fast: planFixture, faithful: planFaithfulFixture }} />
            <SpeedSwitch value="faithful" onChange={() => undefined} plans={{ fast: planFixture, faithful: planFaithfulFixture }} />
          </div>
        </Card>
        <SavedRuns rows={savedRunsIndex} error={null} disabled={false} onOpen={() => undefined} onWatch={() => undefined} onDelete={() => undefined} />
        <Hero state={track} shown={track.points.length} kicker="The Track Record, as recorded" />
        <Card title="Equity chart: the Track Record's index against SPY (real recording)">
          <div className="card-b">
            <EquityChart points={track.points} shown={track.points.length} sweeping={false} events={[]} days={[]} research={false} startCash={100} domain={null} unit="index" maxDrawdown={null} />
          </div>
        </Card>
        <Card title="Equity chart: the research strip and the playhead (illustrative states on real days)">
          <div className="card-b">
            <EquityChart points={track.points.slice(0, 3)} shown={3} sweeping events={[]} days={marks} research startCash={100} unit="index" domain={{ first: dates[0]!, last: dates[dates.length - 1]! }} maxDrawdown={null} status="Researching 5 decision days, 8 at a time: 2 done, 1 from the cache" onOpenDay={() => undefined} />
          </div>
        </Card>
        <Card title="Plan charts: won, lost, open, not bought, no data (real outcomes)">
          <div className="card-b pv-cards">
            {cards.map((o) => (
              <div key={o.key} className={`pv-card${o.hypothetical ? " hypo" : ""}`}>
                <b className="tick">{o.symbol}</b> <span className="muted">{o.result}{o.hypothetical ? " · not traded" : ""}</span>
                <PlanChart outcome={o} />
              </div>
            ))}
          </div>
        </Card>
        <OutcomeWall outcomes={track.outcomes} desks={{}} benchmark="SPY" sources={track.sources} />
        <QualityRow outcomes={track.outcomes} stats={track.stats} minSample={20} />
        <Card title="Plan ladder: every stop hit (real snapshot)">
          <div className="card-b">
            <PlanLadder positions={livePlanSnapshot.positions} />
          </div>
        </Card>
        <Card title="Plan ladder: positions with no plan (real snapshot)">
          <div className="card-b">
            <PlanLadder positions={liveOilSnapshot.positions} />
          </div>
        </Card>
        <Card title="Live: loading and no account">
          <div className="card-b">
            <Live accountId={null} enabled={false} />
          </div>
        </Card>
      </div>
    </div>
  );
}
