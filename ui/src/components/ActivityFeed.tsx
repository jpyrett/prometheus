import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { RunState, LogLine } from "../state/runReducer";
import { agentWords } from "../state/selectors";
import { clock } from "../format";
import { Card } from "./ui/Card";
import { QuestionCard } from "./QuestionCard";
import type { WorkRow } from "../api/types";

const AGENT_VAR: Record<string, string> = { account_manager: "--am", data_clerk: "--clerk", advisor: "--advisor", critic: "--critic", trader: "--trader", system: "--text-3", you: "--text-2" };

function agentName(agent: string): string {
  return agent === "system" ? "System" : agent === "you" ? "You" : agentWords(agent);
}

function Line({ line, cursor }: { line: LogLine; cursor: boolean }) {
  const tone = line.level === "good" ? " ok" : line.level === "warn" ? " warnl" : line.level === "error" ? " bad" : "";
  return (
    <div className={`line${line.block ? " has-block" : ""}`} data-agent={line.agent} data-level={line.level}>
      <time>{clock(line.ts)}</time>
      <span className="agent" style={{ ["--agent" as string]: `var(${AGENT_VAR[line.agent] ?? "--text-3"})` }}>
        <i className="dot" />
        {agentName(line.agent)}
      </span>
      {line.block ? (
        <pre className={`block${tone}`}>{line.text}</pre>
      ) : (
        <span className={`msg${tone}`}>
          {line.text}
          {cursor && <span className="cursor" aria-hidden="true" />}
        </span>
      )}
    </div>
  );
}

export interface ActivityFeedProps {
  state: RunState;
  answer: (questionId: string, value: boolean | string) => Promise<void>;
  /** The Critic's verdict of the stock a question is about, for the approval card. */
  rowOf: (symbol: string) => WorkRow | undefined;
  /** The connection dropped mid-run and is being opened again. */
  reconnecting?: boolean;
}

export function ActivityFeed({ state, answer, rowOf, reconnecting = false }: ActivityFeedProps) {
  const scroller = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);
  const waiting = state.question !== null;

  // Follow the newest line unless the user has scrolled up.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && pinned) el.scrollTop = el.scrollHeight;
  }, [state.log.length, pinned, waiting]);

  useEffect(() => {
    if (state.log.length === 0) setPinned(true);
  }, [state.log.length]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
  };
  const jump = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
    setPinned(true);
  };

  const last = state.log[state.log.length - 1];
  const showCursor = state.phase === "running" && !waiting;
  const lastPlainId = showCursor && last && !last.block ? last.id : null;

  return (
    <Card
      title="Activity"
      grow
      className="activity"
      right={
        reconnecting ? (
          <>
            <i className="dot pulse" style={{ ["--agent" as string]: "var(--warn)" }} />
            connection lost: reconnecting…
          </>
        ) : waiting ? (
          <>
            <i className="dot pulse" style={{ ["--agent" as string]: "var(--warn)" }} />
            <b className="waiting-words">Waiting for you</b>
          </>
        ) : (
          <>
            <i className="dot" style={{ ["--agent" as string]: state.phase === "running" ? "var(--pos)" : "var(--text-3)" }} />
            {state.phase === "running" ? "live · from the Account Manager" : "from the Account Manager"}
          </>
        )
      }
    >
      <div className="feed-wrap">
        <div className="feed" role="log" aria-live="polite" aria-label="What the desk reports" ref={scroller} onScroll={onScroll} tabIndex={0}>
          {state.log.length === 0 ? (
            <div className="empty">
              <b>Nothing yet.</b>The Account Manager will report here as the desk works.
            </div>
          ) : (
            state.log.map((line) => <Line key={line.id} line={line} cursor={line.id === lastPlainId} />)
          )}
        </div>
        {!pinned && state.log.length > 0 && (
          <button type="button" className="jump" onClick={jump}>
            Jump to latest ↓
          </button>
        )}
      </div>
      {state.question && <QuestionCard key={state.question.question_id} question={state.question} answer={answer} row={state.question.symbol ? rowOf(state.question.symbol) : undefined} />}
    </Card>
  );
}
