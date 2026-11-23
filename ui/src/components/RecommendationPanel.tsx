import { useEffect, useState, type ReactNode } from "react";
import { api, ApiError } from "../api/client";
import type { OrderPart, RawRecords, Review, WorkRow } from "../api/types";
import { cap, money, price4, shares as shareText, signedDecimal, words } from "../format";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";
import { Meter } from "./ui/Meter";
import { ScoreDial } from "./ui/ScoreDial";
import { stanceTone } from "./WorkOrderTable";

export interface RecommendationPanelProps {
  row: WorkRow | null;
  runId: string | null;
  /** Rows exist but the Advisor has not answered (the empty state says so). */
  waitingForAdvisor: boolean;
  /** Where the Raw record is read from when it is not a Desk run (a day of a Proving Ground run): given the row's key, the records. */
  loadRecords?: (key: string) => Promise<RawRecords>;
}

function Sec({ title, right, children, className = "" }: { title: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`sec ${className}`.trim()}>
      <h4>
        {title}
        {right !== undefined && <span className="right muted">{right}</span>}
      </h4>
      {children}
    </div>
  );
}

const has = <T,>(list: T[] | undefined | null): list is T[] => Array.isArray(list) && list.length > 0;

/** The selected stock: the sections of spec 4.4.1, each only when it has data (spec 5.3 (11)). */
export function RecommendationPanel({ row, runId, waitingForAdvisor, loadRecords }: RecommendationPanelProps) {
  if (!row) {
    return (
      <Card title="Recommendation" grow className="recommendation" right="by the Advisor · reviewed by the Critic">
        <div className="empty">
          <b>Pick a stock</b>
          {waitingForAdvisor ? "The Advisor has not answered yet." : "Select a row in the work order to read its recommendation."}
        </div>
      </Card>
    );
  }
  const d = row.detail;
  const header = d.header;
  const score = header?.score ?? row.advisor?.score ?? null;
  const stance = header?.stance ?? row.advisor?.stance ?? null;
  const meta = [
    header?.rank != null ? `Rank ${header.rank}${header.of ? ` of ${header.of} scored` : ""}` : null,
    header?.strategy ? `${header.strategy} strategy` : null,
    score !== null ? `Advisor score ${Math.round(score)} of 100` : null,
  ].filter(Boolean);

  return (
    <Card title="Recommendation" grow className="recommendation" right={row.decision?.origin === "advisor" ? "by the Advisor · Critic skipped" : "by the Advisor · reviewed by the Critic"}>
      <div className="scroll rec-scroll" role="group" tabIndex={0} aria-label={`Recommendation for ${row.symbol}`}>
        <div className="rec-h">
          {score !== null && <ScoreDial score={score} label={`${row.symbol}: Advisor score ${Math.round(score)} of 100`} />}
          <div>
            <h3>
              <span className="tick">{row.symbol}</span> <span className="company" title={row.company ?? undefined}>{row.company ?? ""}</span>
              {stance ? <Badge tone={stanceTone(stance)}>{stance.toUpperCase()}</Badge> : row.group === "direct" ? <Badge tone="info">YOUR ORDER</Badge> : null}
            </h3>
            {meta.length > 0 && <small>{meta.join(" · ")}</small>}
          </div>
        </div>

        {(d.message || d.your_order) && (
          <Sec title={row.group === "direct" ? "Your own order" : row.group === "scored" ? "Not recommended" : "About this row"}>
            {d.your_order && <p className="your-order">{d.your_order}</p>}
            {d.message && <p>{d.message}</p>}
          </Sec>
        )}
        {has(d.strategies) && (
          <Sec title="Strategies tried">
            <ul className="strategies">
              {d.strategies.map((s) => (
                <li key={s.name}>
                  <Badge tone={s.passed ? "pos" : "neutral"}>{s.passed ? "PASSED" : "DID NOT PASS"}</Badge>
                  <b>{words(s.name)}</b>
                  {s.score !== null && <span className="num muted">{s.score}</span>}
                  {s.note && <span className="muted note-text">{s.note}</span>}
                </li>
              ))}
            </ul>
          </Sec>
        )}
        {d.thesis && (
          <Sec title="Thesis">
            <p>{d.thesis}</p>
          </Sec>
        )}
        {(has(d.bull) || has(d.bear)) && (
          <div className="sec cases">
            <div className="bull">
              <h4>Bull case</h4>
              {has(d.bull) ? (
                <ul>
                  {d.bull.map((c, i) => (
                    <li key={i}>
                      {c.text}
                      {c.ref && <span className="ref">[{c.ref.replace(/^\[|\]$/g, "")}]</span>}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">none recorded</p>
              )}
            </div>
            <div className="bear">
              <h4>Bear case</h4>
              {has(d.bear) ? (
                <ul>
                  {d.bear.map((c, i) => (
                    <li key={i}>
                      {c.text}
                      {c.ref && <span className="ref">[{c.ref.replace(/^\[|\]$/g, "")}]</span>}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">none recorded</p>
              )}
            </div>
          </div>
        )}
        {has(d.evidence) && (
          <Sec title="Evidence" right={`${d.evidence.length} source${d.evidence.length === 1 ? "" : "s"}`}>
            <ul className="evidence">
              {d.evidence.map((e, i) => (
                <li key={i}>
                  <span className="ev-top">
                    {e.direction && <Badge tone={e.direction === "bullish" ? "pos" : e.direction === "bearish" ? "neg" : "neutral"}>{e.direction.toUpperCase()}</Badge>}
                    {e.counted === false && <Badge tone="neutral" title={e.reason ?? undefined}>NOT COUNTED</Badge>}
                    {e.verified === false && <Badge tone="warn" title={e.reason ?? undefined}>UNVERIFIED</Badge>}
                    {e.source && <span className="muted ev-source">{e.source}</span>}
                  </span>
                  {e.quote && <span className="ev-quote">{e.quote}</span>}
                </li>
              ))}
            </ul>
          </Sec>
        )}
        {has(d.score_build) && (
          <Sec title="How the score was built" right={d.score_build.map((b) => b.weight).join(" / ")}>
            {d.score_build.map((b) => (
              <Meter key={b.name} label={b.label} small={`${b.weight}%`} fill={((b.value + 1) / 2) * 100} value={b.available ? signedDecimal(b.value) : "n/a"} dim={!b.available} title={b.available ? `contribution ${signedDecimal(b.contribution, 3)}` : "not available for this stock"} />
            ))}
          </Sec>
        )}
        {has(d.key_numbers) && (
          <Sec title="Key numbers">
            <dl className="kv keynums">
              {d.key_numbers.map((k) => (
                <div key={k.label}>
                  <dt>{k.label}</dt>
                  <dd className="num">{k.value}</dd>
                </div>
              ))}
            </dl>
          </Sec>
        )}
        {d.plan && <PlanSection plan={d.plan} />}
        {d.review && <ReviewSection review={d.review} />}
        {d.order && <OrderSection order={d.order} />}
        <RawRecord runId={runId} rowKey={row.key} loadRecords={loadRecords} />
      </div>
    </Card>
  );
}

function PlanSection({ plan }: { plan: NonNullable<import("../api/types").RowDetail["plan"]> }) {
  const tiles: [string, string | null][] = [
    ["Entry zone", plan.entry_zone],
    ["Stop", plan.stop],
    ["Target", plan.target],
    ["Order", plan.order],
    ["Time stop", plan.time_stop],
    ["Horizon", plan.horizon],
  ];
  return (
    <Sec title="Trade plan" right="prices and dates only: the Advisor sets no size">
      <div className="plan">
        {tiles.map(([label, value]) => (
          <div key={label}>
            <small>{label}</small>
            <b>{value ?? "—"}</b>
          </div>
        ))}
      </div>
      {has(plan.invalidation) && (
        <div className="note">
          <Badge tone="neutral">INVALIDATED IF</Badge>
          <span>{plan.invalidation.join("; ")}</span>
        </div>
      )}
      {has(plan.notes) &&
        plan.notes.map((n, i) => (
          <div className="note" key={i}>
            <span>{n}</span>
          </div>
        ))}
    </Sec>
  );
}

function ReviewSection({ review }: { review: Review }) {
  const r = review.risk;
  return (
    <Sec title="Critic review">
      <div className="verdict">
        <Badge tone={review.approved ? "pos" : "neg"}>{review.approved ? "APPROVE" : "DENY"}</Badge>
        <span className="big num">{review.score}</span>
        <span className="muted">
          grade {review.grade}
          {r && r.level ? ` · risk ${r.level}${r.score !== null ? ` (${r.score})` : ""}${r.tolerance ? ` for a ${r.tolerance} profile` : ""}` : ""}
        </span>
      </div>
      {review.summary && <p className="summary">{review.summary}</p>}
      {review.dimensions.map((x) => (
        <Meter key={x.name} label={x.label} fill={x.applicable && x.weight > 0 ? (x.points / x.weight) * 100 : 0} tone="critic" value={x.applicable ? `${Math.round(x.points)}/${x.weight}` : "—"} dim={!x.applicable} title={x.applicable ? undefined : "not applicable to this recommendation"} />
      ))}
      {review.groups && review.groups.length > 0 && (
        <div className="groups" role="group" aria-label="The three groups of judgment: a denial needs bad signs in two of them">
          {review.groups.map((g) => (
            <span className="group" key={g.name} title={g.bad ? g.codes.join(", ") : "no sign in this group"}>
              <Badge tone={g.bad ? "neg" : "pos"}>
                {g.label.toUpperCase()} {g.bad ? "BAD" : "CLEAN"}
              </Badge>
              {g.bad && <small className="muted"> {g.codes.join(", ")}</small>}
            </span>
          ))}
        </div>
      )}
      {review.analysts && (
        <div className="note analysts">
          <Badge tone="info">ANALYSTS</Badge>
          <span>{review.analysts}</span>
        </div>
      )}
      {review.blockers.map((b) => (
        <div className="note blocker" key={`b-${b.code}`}>
          <Badge tone="neg">BLOCKER</Badge>
          <span>
            <b>{b.code}</b>: {b.message}
          </span>
        </div>
      ))}
      {review.warnings.length > 0 && (
        <div className="note">
          <Badge tone="warn">
            {review.warnings.length} WARNING{review.warnings.length === 1 ? "" : "S"}
          </Badge>
          <span>
            {review.warnings.map((w, i) => (
              <span key={`${w.code}-${i}`} className="warn-line">
                {w.message}
              </span>
            ))}
          </span>
        </div>
      )}
      {review.requires_human && !review.approved && (
        <div className="note blocker">
          <Badge tone="neg">DENIED</Badge>
          <span>
            <b>needs_human_review</b>: a finding asks for a person to look, so the Critic denied it and nothing is traded for it.
          </span>
        </div>
      )}
    </Sec>
  );
}

function OrderSection({ order }: { order: OrderPart }) {
  const d = order.decision;
  const t = order.trade;
  const fillTone = t?.status === "filled" ? "pos" : t?.status === "rejected" ? "neg" : "neutral";
  return (
    <Sec title="Order and fill">
      <dl className="kv">
        {d && (
          <>
            <dt>Decision</dt>
            <dd>
              <Badge tone={d.will_trade.startsWith("yes_") ? "pos" : d.will_trade === "pending_user" ? "warn" : "neutral"}>{d.words.toUpperCase()}</Badge>
              {d.shares ? (
                <>
                  {" "}
                  {d.dollars ? (
                    <>
                      {d.action === "reduce" ? "sell" : "buy"} <span className="num">{money(d.dollars)}</span>, about {shareText(d.shares)} shares{d.price ? ` at ${money(d.price)}` : ""}
                    </>
                  ) : (
                    <>
                      {d.action === "reduce" ? "sell" : "buy"} {shareText(d.shares)} shares, about <span className="num">{money(d.est_cost)}</span>
                    </>
                  )}
                </>
              ) : null}
              {d.origin && <span className="muted"> · {d.origin}</span>}
            </dd>
            {d.reason && (
              <>
                <dt>Why</dt>
                <dd>{d.reason}</dd>
              </>
            )}
            {d.mode && (
              <>
                <dt>Mode</dt>
                <dd>{d.mode === "auto" ? "Auto-trade" : "Ask me first"}{d.allocation ? ` · ${words(d.allocation)}` : ""}</dd>
              </>
            )}
          </>
        )}
        {t && (
          <>
            <dt>{t.status === "filled" ? "Fill" : "Trade"}</dt>
            <dd>
              <Badge tone={fillTone}>{t.status.toUpperCase()}</Badge>
              {t.status === "filled" && t.shares && t.price !== null ? (
                <>
                  {" "}
                  <span className="num">
                    {shareText(t.shares)} @ {price4(t.price)} = {money(t.amount)}
                  </span>
                </>
              ) : null}
              {[t.mode, t.order_type ? words(t.order_type) : null, t.price_source ? words(t.price_source) : null, t.slippage_bps != null ? `${t.slippage_bps} bps` : null].filter(Boolean).length > 0 && (
                <span className="muted"> · {[t.mode, t.order_type ? words(t.order_type) : null, t.price_source ? words(t.price_source) : null, t.slippage_bps != null ? `${t.slippage_bps} bps` : null].filter(Boolean).join(", ")}</span>
              )}
            </dd>
            {t.reason && (
              <>
                <dt>{t.status === "rejected" ? "Reason" : "Report"}</dt>
                <dd>{cap(t.reason)}{t.reject_code ? <span className="muted"> ({t.reject_code})</span> : null}</dd>
              </>
            )}
          </>
        )}
      </dl>
    </Sec>
  );
}

function RawRecord({ runId, rowKey, loadRecords }: { runId: string | null; rowKey: string; loadRecords?: (key: string) => Promise<RawRecords> }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    // A different row or run: the fold starts closed and empty.
    setOpen(false);
    setText(null);
    setProblem(null);
  }, [rowKey, runId]);

  useEffect(() => {
    if (!open || text !== null || (!runId && !loadRecords)) return;
    let live = true;
    (loadRecords ? loadRecords(rowKey) : api.records(runId!, rowKey))
      .then((records) => live && setText(JSON.stringify(records, null, 2)))
      .catch((e) => live && setProblem(e instanceof ApiError ? e.message : "the record could not be read"));
    return () => {
      live = false;
    };
  }, [open, text, runId, rowKey, loadRecords]);

  return (
    <div className="sec raw">
      <details open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary>Raw record</summary>
        {open && (problem ? <p className="err">{problem}</p> : text === null ? <p className="muted">{runId || loadRecords ? "reading…" : "no run to read it from"}</p> : <pre className="rawjson">{text}</pre>)}
      </details>
    </div>
  );
}
