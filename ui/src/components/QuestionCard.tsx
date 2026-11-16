import { useEffect, useRef, useState } from "react";
import type { QuestionData, WorkRow } from "../api/types";
import { money, shares as shareText } from "../format";
import { Badge } from "./ui/Badge";
import { Button } from "./ui/Button";

const SENT = "sent";

/** What the Account Manager asks (spec 5.3 (5)): confirm what was understood, how much, approve a trade. Pinned to the bottom of Activity; focus goes to its first control. */
export function QuestionCard({ question, answer, row }: { question: QuestionData; answer: (id: string, value: boolean | string) => Promise<void>; row?: WorkRow }) {
  const [sent, setSent] = useState<string | null>(null);
  const first = useRef<HTMLElement | null>(null);

  useEffect(() => {
    first.current?.focus();
  }, [question.question_id]);

  const send = async (value: boolean | string) => {
    setSent(SENT);
    await answer(question.question_id, value);
  };
  const locked = sent !== null;

  return (
    <div className="qcard" role="group" aria-label={question.title}>
      {question.kind === "confirm" && <Confirm question={question} locked={locked} send={send} first={first} />}
      {question.kind === "clarify" && <Clarify question={question} locked={locked} send={send} first={first} />}
      {question.kind === "approve" && <Approve question={question} locked={locked} send={send} first={first} row={row} />}
      {question.kind === "missing_company" && <MissingCompany question={question} locked={locked} send={send} first={first} />}
    </div>
  );
}

interface Inner {
  question: QuestionData;
  locked: boolean;
  send: (value: boolean | string) => Promise<void>;
  first: React.MutableRefObject<HTMLElement | null>;
}

function Confirm({ question, locked, send, first }: Inner) {
  return (
    <>
      <h3>
        <Badge tone="warn">ACCOUNT MANAGER ASKS</Badge>
      </h3>
      <p>{question.text}</p>
      <div className="acts">
        <Button variant="yes" disabled={locked} onClick={() => void send(true)} ref={(el) => void (first.current = el)}>
          Yes, go ahead
        </Button>
        <Button variant="no" disabled={locked} onClick={() => void send(false)}>
          No, stop
        </Button>
      </div>
    </>
  );
}

function Clarify({ question, locked, send, first }: Inner) {
  const [text, setText] = useState("");
  const ask = question.question ?? question.text;
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !locked) void send(text.trim());
  };
  return (
    <form onSubmit={submit}>
      <h3>
        <Badge tone="warn">{question.n && question.of ? `ACCOUNT MANAGER ASKS · ${question.n} OF ${question.of}` : "ACCOUNT MANAGER ASKS"}</Badge>
      </h3>
      <p>{ask}</p>
      {question.facts && question.facts.length > 0 && <pre>{question.facts.join("\n")}</pre>}
      <div className="acts">
        <input ref={(el) => void (first.current = el)} className="in" aria-label="Your answer" disabled={locked} value={text} onChange={(e) => setText(e.target.value)} placeholder={'e.g. "20%", "$5,000", "10 shares each", "as much as I can"'} />
        <Button type="submit" variant="yes" disabled={locked || !text.trim()}>
          Send
        </Button>
      </div>
    </form>
  );
}

/** A stock the Data Clerk did not find, in Ask mode: the Account Manager says what is missing and why, shows what stays on the order, and asks whether to go on without it. The sentence is the Account Manager's own (its model's, checked by code, or the code's); the answer is a yes or a no. */
function MissingCompany({ question, locked, send, first }: Inner) {
  const typed = question.typed ?? "";
  const sentence = question.question ?? question.text;
  const at = typed ? sentence.toLowerCase().indexOf(typed.toLowerCase()) : -1;
  return (
    <>
      <h3>
        <Badge tone="warn">{question.n && question.of && question.of > 1 ? `ACCOUNT MANAGER ASKS · ${question.n} OF ${question.of}` : "ACCOUNT MANAGER ASKS"}</Badge>
      </h3>
      <p>
        {at >= 0 ? (
          <>
            {sentence.slice(0, at)}
            <b style={{ color: "var(--text)" }}>{sentence.slice(at, at + typed.length)}</b>
            {sentence.slice(at + typed.length)}
          </>
        ) : (
          sentence
        )}
      </p>
      <div className="acts">
        <Button variant="yes" disabled={locked} onClick={() => void send(true)} ref={(el) => void (first.current = el)}>
          OK, continue without it
        </Button>
        <Button variant="no" disabled={locked} onClick={() => void send(false)}>
          Stop the order
        </Button>
      </div>
      {question.facts && question.facts.length > 0 && (
        <pre aria-label="What stays on the order" style={{ margin: "12px 0 0" }}>
          {question.facts.join("\n")}
        </pre>
      )}
      <p className="cap">Nothing is left out without your yes. No answer means the order is not placed.</p>
    </>
  );
}

function verb(action: string | null | undefined): string {
  return action === "reduce" ? "Sell" : "Buy";
}

function Approve({ question, locked, send, first, row }: Inner & { row?: WorkRow }) {
  const [card, setCard] = useState(false);
  const q = question;
  const many = (q.of ?? 1) > 1;
  const showProposal = many && q.n === 1 && q.proposal && q.proposal.length > 0;
  const critic = row?.critic;
  return (
    <>
      <h3>
        <Badge tone="warn">{q.title.toUpperCase()}</Badge>
      </h3>
      {showProposal && (
        <ul className="proposal" aria-label="The whole proposal">
          {q.proposal!.map((p) => (
            <li key={p.symbol}>
              <span className="tick">{p.symbol}</span>
              <span>
                {p.dollars ? `${verb(p.action)} ${money(p.dollars)}` : `${verb(p.action)} ${shareText(p.shares)}`}
              </span>
              <span className="num">{p.dollars ? `about ${shareText(p.shares)} sh` : money(p.est_cost)}</span>
            </li>
          ))}
        </ul>
      )}
      <p>
        <b style={{ color: "var(--text)" }}>
          {q.dollars ? `${verb(q.action)} ${money(q.dollars)} of ${q.symbol}` : `${verb(q.action)} ${shareText(q.shares)} shares of ${q.symbol}`}
        </b>
        {q.dollars ? (
          <>
            , about {shareText(q.shares)} shares{q.price ? ` at ${money(q.price)}` : ""}, of <span className="num">{money(q.cash)}</span> cash.
          </>
        ) : (
          <>
            , about <span className="num">{money(q.est_cost)}</span> of <span className="num">{money(q.cash)}</span> cash.
          </>
        )}
        {critic ? ` Critic: ${critic.approved ? "APPROVE" : "DENY"} ${critic.score} (${critic.grade}).` : ""}
        {q.question ? ` ${q.question}` : ""}
      </p>
      {card && <pre aria-label="The Critic's scorecard">{q.scorecard ?? q.text}</pre>}
      <div className="acts">
        <Button variant="yes" disabled={locked} onClick={() => void send(true)} ref={(el) => void (first.current = el)}>
          Approve
        </Button>
        <Button variant="no" disabled={locked} onClick={() => void send(false)}>
          Deny
        </Button>
        <Button variant="ghost" style={{ marginLeft: "auto" }} aria-expanded={card} onClick={() => setCard((c) => !c)}>
          {card ? "Hide the scorecard" : "See the scorecard"}
        </Button>
      </div>
      <p className="cap">Nothing is traded without a yes. No answer means no.</p>
    </>
  );
}
