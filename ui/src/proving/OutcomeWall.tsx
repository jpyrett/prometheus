// The outcome wall (spec 13.2 (3)): one small card per recommendation in decision-day order, filterable by verdict and result. A card shows the ticker, the Advisor's score dial, the Critic's verdict badge,
// the plan chart, the return and the excess, and WON / LOST / OPEN / NOT FILLED. A stock the desk did not buy (P8) is grey with a dashed outline and says why. A card that closes while the run is
// watched flashes once. Click: the full recommendation in a side sheet, drawn by the Desk's own RecommendationPanel.
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlanChart } from "../charts/PlanChart";
import { shortDate } from "../charts/scale";
import { RecommendationPanel } from "../components/RecommendationPanel";
import { Badge, type BadgeTone } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { ScoreDial } from "../components/ui/ScoreDial";
import { money, pct, points, shares as shareText } from "../format";
import type { WorkOrderView, WorkRow } from "../api/types";
import { ChipGroup } from "./Chips";

// The Critic v3 denies for two GROUPS (design choices entry 137): `group:news` is said in words, as the Critic's own DENY sentence says it; every other code is shown as recorded.
const GROUP_WORDS: Record<string, string> = { "group:news": "poor news", "group:performance": "poor performance", "group:price": "a price too high" };
const reasonWords = (code: string) => GROUP_WORDS[code] ?? code;
import type { RecOutcome, Result, Verdict } from "./types";
import { filterOutcomes, NO_FILTERS, notTradedWords, RESULT_WORDS, VERDICT_WORDS, wallOrder, type Filters } from "./selectors";

const RESULT_TONE: Record<Result, BadgeTone> = { won: "pos", lost: "neg", open: "info", not_filled: "neutral", no_data: "neutral" };
const PAGE = 24;

export function criticBadge(o: RecOutcome): { text: string; tone: BadgeTone } {
  if (o.critic) return { text: `${o.critic.approved ? "APPROVED" : "DENIED"} ${o.critic.score ?? ""}${o.critic.grade ? ` · ${o.critic.grade}` : ""}`.replace(/\s+/g, " ").trim(), tone: o.critic.approved ? "pos" : "neg" };
  return o.verdict === "neutral" ? { text: "NEUTRAL PASS", tone: "info" } : { text: "NOT REVIEWED", tone: "neutral" };
}

export interface OutcomeWallProps {
  outcomes: RecOutcome[];
  /** The desk's own view of each decision day (when the run has them): the side sheet reads the recommendation from it. */
  desks: Record<string, WorkOrderView>;
  /** Ask for the desk of a decision day (the sheet opens on a card whose day's desk is not here yet). */
  onNeedDesk?: (day: string) => void;
  /** The keys of the outcomes that just arrived: their card flashes once. */
  freshKeys?: string[];
  benchmark?: string;
  /** The Track Record: where each outcome came from. */
  sources?: Record<string, "run" | "fixture">;
}

export function OutcomeWall({ outcomes, desks, onNeedDesk, freshKeys = [], benchmark = "SPY", sources }: OutcomeWallProps) {
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [limit, setLimit] = useState(PAGE);
  const [open, setOpen] = useState<RecOutcome | null>(null);
  const close = useCallback(() => setOpen(null), []);
  const ordered = useMemo(() => wallOrder(outcomes), [outcomes]);
  const shown = useMemo(() => filterOutcomes(ordered, filters), [ordered, filters]);
  const count = (f: Partial<Filters>) => filterOutcomes(ordered, { ...filters, ...f }).length;

  // a card that arrives after the wall is on screen flashes once (the existing 600 ms wash)
  const seen = useRef<Set<string>>(new Set());
  const [flash, setFlash] = useState<Set<string>>(new Set());
  useEffect(() => {
    const fresh = freshKeys.filter((k) => !seen.current.has(k));
    for (const o of outcomes) seen.current.add(o.key);
    if (fresh.length === 0) return;
    setFlash(new Set(fresh));
    const t = setTimeout(() => setFlash(new Set()), 700);
    return () => clearTimeout(t);
  }, [freshKeys, outcomes]);

  return (
    <Card title="Every recommendation, followed by its plan" className="wall" right={`${outcomes.length} recommendation${outcomes.length === 1 ? "" : "s"}`}>
      <div className="card-b">
        <div className="wall-filters">
          <ChipGroup<Verdict | "all">
            label="Filter by the Critic's verdict"
            value={filters.verdict}
            onChange={(verdict) => setFilters({ ...filters, verdict })}
            options={[
              { value: "all", label: "All verdicts", count: count({ verdict: "all" }) },
              { value: "approved", label: "Approved", count: count({ verdict: "approved" }) },
              { value: "denied", label: "Denied", count: count({ verdict: "denied" }) },
              { value: "neutral", label: "Neutral", count: count({ verdict: "neutral" }) },
            ]}
          />
          <ChipGroup<Result | "all">
            label="Filter by result"
            value={filters.result}
            onChange={(result) => setFilters({ ...filters, result })}
            options={[
              { value: "all", label: "Any result", count: count({ result: "all" }) },
              { value: "won", label: "Won", count: count({ result: "won" }) },
              { value: "lost", label: "Lost", count: count({ result: "lost" }) },
              { value: "open", label: "Open", count: count({ result: "open" }) },
              { value: "not_filled", label: "Not filled", count: count({ result: "not_filled" }) },
            ]}
          />
          <ChipGroup<Filters["traded"]>
            label="Filter by whether the desk bought it"
            value={filters.traded}
            onChange={(traded) => setFilters({ ...filters, traded })}
            options={[
              { value: "all", label: "Bought or not" },
              { value: "traded", label: "Bought", count: count({ traded: "traded" }) },
              { value: "not_traded", label: "Not bought", count: count({ traded: "not_traded" }) },
            ]}
          />
        </div>
        {shown.length === 0 ? (
          <div className="empty">
            <b>{outcomes.length === 0 ? "No recommendation has an outcome yet" : "Nothing matches these filters"}</b>
            {outcomes.length === 0 ? "Each one appears here as the test follows it." : "Change a filter to see the others."}
          </div>
        ) : (
          <div className="wall-grid">
            {shown.slice(0, limit).map((o) => (
              <OutcomeCard key={o.key} outcome={o} flash={flash.has(o.key)} onOpen={() => setOpen(o)} benchmark={benchmark} source={sources?.[o.key]} />
            ))}
          </div>
        )}
        {shown.length > limit && (
          <div className="wall-more">
            <button type="button" className="btn" onClick={() => setLimit((l) => l + PAGE)}>
              Show {Math.min(PAGE, shown.length - limit)} more of {shown.length - limit}
            </button>
          </div>
        )}
      </div>
      {open && <OutcomeSheet outcome={open} desk={desks[open.decision_day] ?? null} onClose={close} onNeedDesk={onNeedDesk} />}
    </Card>
  );
}

function OutcomeCard({ outcome: o, flash, onOpen, benchmark, source }: { outcome: RecOutcome; flash: boolean; onOpen: () => void; benchmark: string; source?: "run" | "fixture" }) {
  const critic = criticBadge(o);
  const closed = o.result === "won" || o.result === "lost";
  return (
    <article className={`oc${o.hypothetical ? " hypo" : ""}${flash ? " flash" : ""} ${o.result}`} data-result={o.result} data-key={o.key} aria-label={`${o.symbol}, ${shortDate(o.decision_day, true)}: ${RESULT_WORDS[o.result].toLowerCase()}${o.hypothetical ? ", not traded" : ""}`}>
      <button type="button" className="oc-head" onClick={onOpen} aria-label={`Open the recommendation for ${o.symbol} of ${o.decision_day}`}>
        {o.advisor?.score != null ? <ScoreDial score={o.advisor.score} label={`${o.symbol}: Advisor score ${o.advisor.score} of 100`} /> : <span className="dial-none" aria-hidden="true" />}
        <span className="oc-id">
          <span className="tick">{o.symbol}</span>
          <span className="sub" title={o.company ?? undefined}>
            {o.company ?? " "}
          </span>
        </span>
        <span className="oc-day num">{shortDate(o.decision_day)}</span>
      </button>
      <div className="oc-badges">
        <Badge tone={critic.tone}>{critic.text}</Badge>
        <Badge tone={RESULT_TONE[o.result]}>{RESULT_WORDS[o.result]}</Badge>
        {source === "fixture" && (
          <Badge tone="neutral" title="From a recorded session of the project, not a run of this page">
            RECORDED
          </Badge>
        )}
      </div>
      <PlanChart outcome={o} />
      <div className="oc-foot">
        {o.return_pct !== null ? (
          <>
            <span className={`oc-ret num ${closed ? (o.return_pct >= 0 ? "pos" : "neg") : ""}`}>{pct(o.return_pct)}</span>
            <span className="oc-vs muted num">
              {o.excess_return_pct !== null ? `${points(o.excess_return_pct)} vs ${benchmark}` : `no ${benchmark} figure`}
            </span>
          </>
        ) : (
          <span className="muted">no return to show</span>
        )}
      </div>
      {o.hypothetical ? (
        <div className="oc-why">{notTradedWords(o)}</div>
      ) : (
        o.entry && (
          <div className="oc-why muted num">
            {shareText(o.entry.shares)} sh at {money(o.entry.price)}
            {o.gain !== null ? ` · ${o.gain >= 0 ? "+" : "−"}${money(Math.abs(o.gain))}` : ""}
          </div>
        )
      )}
    </article>
  );
}

/** The recommendation behind a card, in a sheet from the right: the Desk's own panel when the day's desk is known, else what the outcome itself carries. */
export function OutcomeSheet({ outcome: o, desk, onClose, onNeedDesk }: { outcome: RecOutcome; desk: WorkOrderView | null; onClose: () => void; onNeedDesk?: (day: string) => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!desk) onNeedDesk?.(o.decision_day);
  }, [desk, o.decision_day, onNeedDesk]);
  const before = useRef<Element | null>(null);
  useEffect(() => {
    before.current = document.activeElement;
    closeRef.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("keydown", esc);
      if (before.current instanceof HTMLElement) before.current.focus();
    };
  }, [onClose]);
  const row: WorkRow | null = desk?.rows.find((r) => r.symbol === o.symbol && r.group !== "found") ?? null;
  return (
    <>
      <div className="sheet-scrim" onClick={onClose} aria-hidden="true" />
      <aside className="sheet" role="dialog" aria-modal="true" aria-label={`The recommendation for ${o.symbol} of ${o.decision_day}`}>
        <div className="sheet-h">
          <div>
            <b className="tick">{o.symbol}</b> <span className="muted">{o.company ?? ""}</span>
            <div className="muted">
              {VERDICT_WORDS[o.verdict]} on {shortDate(o.decision_day, true)} · {RESULT_WORDS[o.result].toLowerCase()}
              {o.hypothetical ? ` · ${notTradedWords(o)}` : ""}
            </div>
          </div>
          <button ref={closeRef} type="button" className="btn ghost" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="sheet-b">
          {row ? (
            <RecommendationPanel row={row} runId={null} waitingForAdvisor={false} />
          ) : (
            <Card title="What the outcome recorded">
              <div className="card-b">
                <p className="muted">The desk&apos;s full view of this day is not kept in this record, so the recommendation&apos;s text is not here. What was recorded:</p>
                <dl className="kv">
                  <dt>Advisor</dt>
                  <dd>{o.advisor ? `score ${o.advisor.score ?? "—"} · ${o.advisor.stance ?? "—"} · ${o.advisor.family ?? "—"}` : "—"}</dd>
                  <dt>Critic</dt>
                  <dd>{o.critic ? `${o.critic.approved ? "approved" : "denied"} ${o.critic.score ?? ""} ${o.critic.grade ?? ""}${o.critic.denial_reasons.length ? ` (${o.critic.denial_reasons.map(reasonWords).join(", ")})` : ""}` : VERDICT_WORDS[o.verdict]}</dd>
                  <dt>Plan</dt>
                  <dd className="num">
                    entry {o.plan.entry_low !== null ? `${money(o.plan.entry_low)} to ${money(o.plan.entry_high)}` : "—"} · {o.plan.trailing ? "starting stop" : "stop"} {money(o.plan.stop)} · {o.plan.trailing ? `no target, the stop trails ${money(o.plan.trail_distance)} under the highest close` : `target ${money(o.plan.target)}`}
                    {o.plan.time_stop_date ? ` · time stop ${o.plan.time_stop_date}` : ""}
                  </dd>
                  {o.notes.length > 0 && (
                    <>
                      <dt>Notes</dt>
                      <dd>{o.notes.join("; ")}</dd>
                    </>
                  )}
                </dl>
              </div>
            </Card>
          )}
          {(o.forward ?? []).some((f) => f.return_pct !== null) && (
            <Card title="After the decision day">
              <div className="card-b">
                <p className="muted">The stock&apos;s return from the close of the decision day, whatever the plan did (it measures the pick, not where the plan put its stop and target):</p>
                <dl className="kv">
                  {(o.forward ?? [])
                    .filter((f) => f.return_pct !== null)
                    .map((f) => (
                      <Fragment key={f.sessions}>
                        <dt>{f.sessions} sessions</dt>
                        <dd className="num">
                          {pct(f.return_pct)}
                          {f.excess_return_pct !== null ? ` · ${points(f.excess_return_pct)} against the benchmark` : ""}
                        </dd>
                      </Fragment>
                    ))}
                </dl>
              </div>
            </Card>
          )}
        </div>
      </aside>
    </>
  );
}
