import { count, tokens } from "../format";
import type { RunState } from "../state/runReducer";
import { runBoxView } from "../state/selectors";
import { useNow } from "../state/useNow";
import { Gears } from "./Gears";

export function RunBox({ state }: { state: RunState }) {
  const now = useNow(state.phase === "running");
  const view = runBoxView(state, now);
  const a = state.activity;
  return (
    <div className={`runbox ${view.tone}`}>
      <span className="sr-only" role="status">
        {view.label}. {view.line}
      </span>
      <Gears spinning={view.spinning} />
      <div className="rb-text">
        <small className="rb-label">{view.label}</small>
        <span className="t" aria-hidden="true">
          {view.timer}
        </span>
        <small className="rb-line" title={view.line}>
          {view.line}
        </small>
        {a && a.calls > 0 && (
          <small className="rb-act" title={a.model ?? undefined}>
            {count(a.calls)} model call{a.calls === 1 ? "" : "s"} · {tokens(a.tokens_in + a.tokens_out)} tokens
          </small>
        )}
      </div>
    </div>
  );
}
