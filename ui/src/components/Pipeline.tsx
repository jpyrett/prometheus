import type { StageView } from "../state/runReducer";
import type { RunState } from "../state/runReducer";
import { stageSeconds } from "../state/selectors";
import { useNow } from "../state/useNow";
import { CheckIcon, CrossIcon } from "./icons";
import { RunBox } from "./RunBox";

const AGENT_VAR: Record<string, string> = { account_manager: "--am", data_clerk: "--clerk", advisor: "--advisor", critic: "--critic", trader: "--trader" };

function nodeContent(stage: StageView, index: number) {
  switch (stage.state) {
    case "done":
      return <CheckIcon />;
    case "waiting":
      return "?";
    case "skipped":
      return "–";
    case "failed":
      return <CrossIcon />;
    default:
      return String(index + 1);
  }
}

const STATE_WORDS: Record<string, string> = { pending: "not reached", busy: "working", waiting: "waiting for you", done: "done", skipped: "skipped", failed: "stopped here" };

/** The six stages on their track. Exported so the states sheet draws the very same list. */
export function StepList({ stages, now, ended }: { stages: StageView[]; now: number; ended: boolean }) {
  return (
    <ol className="steps">
      {stages.map((stage, i) => {
        const secs = stageSeconds(stage, now);
        return (
          <li key={stage.key} className={`step ${stage.state}`} style={{ ["--agent" as string]: `var(${AGENT_VAR[stage.agent] ?? "--text-3"})` }} aria-current={stage.state === "busy" || stage.state === "waiting" ? "step" : undefined}>
            <div className="node" aria-hidden="true">
              {nodeContent(stage, i)}
            </div>
            <div className="label" title={stage.label}>
              {stage.label}
            </div>
            <div className="meta">
              <i className="dot" />
              <span>
                {stage.agent_label}
                {secs !== null ? (
                  <>
                    {" · "}
                    <span className="num">{secs}</span>
                  </>
                ) : ended && stage.state === "pending" ? (
                  " · not reached"
                ) : null}
              </span>
            </div>
            <span className="sr-only">{STATE_WORDS[stage.state]}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** A request that only asked for the news (entry 149, N9) is the Account Manager and the Data Clerk: once it has finished the pipeline shows those two stages and not the four it never reached. */
export function Pipeline({ state }: { state: RunState }) {
  const now = useNow(state.phase === "running");
  return (
    <section className="card pipeline" aria-label="Desk pipeline">
      <div className="lane">
        <h2>Desk pipeline</h2>
        <StepList stages={state.result?.status === "news" ? state.stages.slice(0, 2) : state.stages} now={now} ended={state.phase === "finished"} />
      </div>
      <RunBox state={state} />
    </section>
  );
}
