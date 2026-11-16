// "Component states" (?states): the sheet under the mockup, drawn with the page's own components so the states cannot drift from the page.
// It is a development view: nothing here talks to the server, and the text of the run-box examples that no capture covers is the mockup's own, marked as illustrative.
import { useEffect } from "react";
import type { QuestionData } from "./api/types";
import { StepList } from "./components/Pipeline";
import { QuestionCard } from "./components/QuestionCard";
import { RecommendationPanel } from "./components/RecommendationPanel";
import { RunBox } from "./components/RunBox";
import { WorkOrderTable } from "./components/WorkOrderTable";
import { Badge } from "./components/ui/Badge";
import { Button } from "./components/ui/Button";
import { Card } from "./components/ui/Card";
import { Check } from "./components/ui/Check";
import { Switch } from "./components/ui/Switch";
import { initialRunState, replayEvents, type RunState, type StageView } from "./state/runReducer";
import { useTheme } from "./state/useTheme";
import { approveQuestionEvent, bobEvents, clarifyQuestionEvent, confirmQuestionEvent, missingQuestionEvent } from "./statesData";

const noop = () => undefined;
const answer = () => Promise.resolve();

function stage(base: StageView, state: StageView["state"], label: string, since: number | null = null): StageView {
  return { ...base, state, label, since, seconds: 2.1, node: base.node ?? base.key };
}

export default function StatesPage() {
  const theme = useTheme();
  useEffect(() => {
    const want = new URLSearchParams(window.location.search).get("theme");
    if (want === "light" || want === "dark") theme.set(want, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const finished = replayEvents(bobEvents);
  const [a, b, c, d, e, f] = initialRunState.stages;
  const states: StageView[] = [stage(a!, "done", "done"), stage(b!, "busy", "busy", 1000), stage(c!, "waiting", "waiting for you", 1000), stage(d!, "skipped", "skipped"), stage(e!, "failed", "stopped here"), stage(f!, "pending", "not reached")];
  const asQuestion = (event: { data: unknown }) => event.data as QuestionData;
  const row = finished.workOrder!.rows[0];
  const runState = (patch: Partial<RunState>): RunState => ({ ...finished, ...patch });
  const nothing = runState({ result: { ...finished.result!, status: "no_trade", headline: "Nothing was traded: the Critic denied 1 of 1: APA (risk_above_profile)." } });
  const stopped = runState({ result: { ...finished.result!, status: "failed", headline: "The Data Clerk could not prepare the data: no listed company was found to score." } });

  return (
    <div className="gallery">
      <h1>Component states</h1>
      <p>What the dashboard cannot show at once. The page&apos;s own components, the same tokens. The run-box examples marked illustrative use the mockup&apos;s wording: no capture of them exists.</p>
      <div className="gal">
        <Card title="Pipeline · every stage state" className="wide">
          <div className="card-b">
            <StepList stages={states} now={1003.4} ended />
          </div>
        </Card>

        <Card title="Question · confirm what was understood">
          <QuestionCard question={asQuestion(confirmQuestionEvent)} answer={answer} />
          <p className="cap">Ask mode. Pinned to the bottom of Activity; the stage turns &quot;waiting for you&quot;.</p>
        </Card>
        <Card title="Question · how much (Ask mode)">
          <QuestionCard question={asQuestion(clarifyQuestionEvent)} answer={answer} />
          <p className="cap">A free-text answer. The facts block is the Account Manager&apos;s own lines.</p>
        </Card>
        <Card title="Question · approve a trade (Ask mode)">
          <QuestionCard question={asQuestion(approveQuestionEvent)} answer={answer} row={row} />
        </Card>
        <Card title="Question · a stock that was not found (Ask mode)">
          <QuestionCard question={asQuestion(missingQuestionEvent)} answer={answer} />
          <p className="cap">The Data Clerk could not list it. The sentence is the Account Manager&apos;s model&apos;s, checked by code; the block is what stays on the order.</p>
        </Card>

        <Card title="Run box · idle, done, nothing traded, stopped">
          <div className="card-b runs">
            <div className="rb-sample">
              <RunBox state={initialRunState} />
            </div>
            <div className="rb-sample">
              <RunBox state={finished} />
            </div>
            <div className="rb-sample">
              <RunBox state={nothing} />
              <small className="muted">illustrative</small>
            </div>
            <div className="rb-sample">
              <RunBox state={stopped} />
              <small className="muted">illustrative</small>
            </div>
          </div>
        </Card>

        <Card title="Empty states">
          <WorkOrderTable state={initialRunState} selectedKey={null} onSelect={noop} />
          <div style={{ borderTop: "1px solid var(--border)" }}>
            <RecommendationPanel row={null} runId={null} waitingForAdvisor={false} />
          </div>
        </Card>

        <Card title="Buttons, chips, checks">
          <div className="card-b" style={{ display: "grid", gap: 12 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <Button variant="go">Go!</Button>
              <Button>Add</Button>
              <Button variant="ghost">Clear</Button>
              <Button variant="yes">Approve</Button>
              <Button variant="no">Deny</Button>
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <Badge tone="pos">BULLISH</Badge>
              <Badge tone="neutral">NEUTRAL</Badge>
              <Badge tone="neg">BEARISH</Badge>
              <Badge tone="pos">APPROVE 86 · B</Badge>
              <Badge tone="neg">DENY 49 · F</Badge>
              <Badge tone="warn">PENDING YOU</Badge>
              <Badge tone="info">NEUTRAL PASS</Badge>
              <Badge tone="brand">PLACING…</Badge>
              <Badge tone="pos">FILLED</Badge>
              <Badge tone="neg">REJECTED</Badge>
            </div>
            <div style={{ display: "flex", gap: 22, flexWrap: "wrap" }}>
              {(["ask", "auto"] as const).map((m) => (
                <div className="mode" key={m}>
                  <span className="k">Mode</span>
                  <span className={`opt${m === "ask" ? " on" : ""}`}>Ask</span>
                  <Switch checked={m === "auto"} onChange={() => undefined} label={`Trade mode: ${m === "auto" ? "Auto" : "Ask"}`} />
                  <span className={`opt${m === "auto" ? " on" : ""}`}>Auto</span>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <Check label="Strict recommendations" defaultChecked={false} readOnly />
              <Check label="Neutral recommendations as passes" defaultChecked readOnly />
              <Check
                label={
                  <>
                    Refresh market data first <span className="paid">(paid)</span>
                  </>
                }
                defaultChecked
                readOnly
              />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
