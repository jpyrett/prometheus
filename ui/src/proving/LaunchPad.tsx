// The launch pad of the Time Machine (spec 13.2): the request in the user's own words, the starting cash, the risk, the Neutral and Strict boxes, the speed (Fast or Faithful, each with its own time from the plan),
// the range picker (a chart of what is stored) with a cadence control and quick ranges, a plan line that is asked again, free, whenever an input changes, three quick prompts, and Launch (ember). Launch is a PAID run in a live server, so the plan line
// says what it will cost before anyone presses it; it is disabled, with its reason, while a desk run is active or in a recorded session.
import { useEffect, useMemo, useState } from "react";
import { pickableDays, quickRange, RangePicker } from "../charts/RangePicker";
import { dayNumber } from "../charts/scale";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Check } from "../components/ui/Check";
import { SmallGear } from "../components/Gears";
import type { RiskLevel } from "../api/types";
import { money } from "../format";
import { ChipGroup } from "./Chips";
import { caveatItems } from "./caveats";
import { SpeedSwitch } from "./SpeedSwitch";
import { planTimeWords } from "./timing";
import type { Cadence, Coverage, PlanAnswer, Speed, TimeMachineRequest } from "./types";
import { usePlan } from "./useProvingData";

/** A quick test: the prompt, and, for the first, the whole launch pad. */
export interface QuickTest {
  label: string;
  prompt: string;
  /** Present only on a test whose research is kept warm in the cache: a click fills the whole pad with it. */
  whole?: Omit<TimeMachineRequest, "prompt">;
  title: string;
}

/** The first chip is exactly the request whose research of 23 weekly decision days is in the cache (the real fast run of tests/fixtures/proving_real/oil_weekly_6m_fast.json, and a test says the two are one): Launch replays it from the cache in seconds. The other two are requests of the project's own recorded sessions. */
export const QUICK_TESTS: QuickTest[] = [
  {
    label: "Oil, $3,000 split evenly",
    prompt: "Buy the 3 most recommended oil stocks with $3,000, split evenly",
    whole: { cash: 10000, start: "2026-06-08", end: "2026-11-09", cadence: "weekly", risk: "moderate", neutral: true, strict: false, speed: "fast" },
    title: "Fills the whole pad: June 8 to November 9, weekly, $10,000, Neutral on, moderate, fast. Its research is already in the cache, so Launch replays it in seconds.",
  },
  { label: "AI stocks, top 3", prompt: 'I want to buy the best "AI stocks" as a theme. Buy $20000 worth of the top 3', title: "Fills the request only" },
  { label: "Oil, all the money", prompt: "I want to buy oil stocks. Use as much of the money in my account as you can to buy shares of the 3 most recommended oil companies.", title: "Fills the request only" },
];
const WARM = QUICK_TESTS[0]!;

const CADENCES: { value: Cadence; label: string }[] = [
  { value: "daily", label: "Daily" },
  { value: "twice_weekly", label: "Twice a week" },
  { value: "weekly", label: "Weekly" },
  { value: "every_two_weeks", label: "Every two weeks" },
];
const RISKS: { value: RiskLevel; label: string }[] = [
  { value: "conservative", label: "Conservative" },
  { value: "moderate", label: "Moderate" },
  { value: "aggressive", label: "Aggressive" },
];
const QUICK: { label: string; months: number | null }[] = [
  { label: "1M", months: 1 },
  { label: "3M", months: 3 },
  { label: "6M", months: 6 },
  { label: "Max", months: null },
];

export interface LaunchPadProps {
  coverage: Coverage | null;
  coverageError: string | null;
  /** The reason Launch cannot be pressed (a desk run is active; a recorded session), or null. */
  blockedReason: string | null;
  starting: boolean;
  running: boolean;
  defaultRisk: RiskLevel;
  error: string | null;
  onLaunch: (request: TimeMachineRequest) => void;
  onStop: () => void;
  /** A fixed answer for the plan line (tests); the page asks the server. */
  planOverride?: { plan: PlanAnswer | null; problem: string | null; loading: boolean };
}

export function LaunchPad({ coverage, coverageError, blockedReason, starting, running, defaultRisk, error, onLaunch, onStop, planOverride }: LaunchPadProps) {
  // The pad opens on the warm request (the first chip), with the account's own risk.
  const [prompt, setPrompt] = useState(WARM.prompt);
  const [cash, setCash] = useState(String(WARM.whole!.cash));
  const [risk, setRisk] = useState<RiskLevel>(defaultRisk);
  const [neutral, setNeutral] = useState(WARM.whole!.neutral);
  const [strict, setStrict] = useState(WARM.whole!.strict);
  const [cadence, setCadence] = useState<Cadence>(WARM.whole!.cadence);
  const [speed, setSpeed] = useState<Speed>("fast");
  const [range, setRange] = useState<{ start: string; end: string } | null>(null);
  const days = useMemo(() => (coverage ? pickableDays(coverage) : []), [coverage]);

  // the range starts as the warm request's own when both its days can be picked, else as the last six months of what can be tested
  useEffect(() => {
    if (range !== null || days.length < 2) return;
    const w = WARM.whole!;
    setRange(days.includes(dayNumber(w.start)) && days.includes(dayNumber(w.end)) ? { start: w.start, end: w.end } : quickRange(days, 6));
  }, [days, range]);

  /** A quick test: its prompt, and, when it carries the whole pad, every other input too. */
  const apply = (q: QuickTest) => {
    setPrompt(q.prompt);
    const w = q.whole;
    if (!w) return;
    setCash(String(w.cash));
    setRange({ start: w.start, end: w.end });
    setCadence(w.cadence);
    setRisk(w.risk);
    setNeutral(w.neutral);
    setStrict(w.strict);
    setSpeed(w.speed ?? "fast");
  };

  const cashNumber = Number(cash.replace(/[,$\s]/g, ""));
  const cashOk = Number.isFinite(cashNumber) && cashNumber > 0;
  const request: TimeMachineRequest | null = range && cashOk && prompt.trim() ? { prompt: prompt.trim(), cash: cashNumber, start: range.start, end: range.end, cadence, risk, neutral: strict ? false : neutral, strict, speed } : null;
  // Both speeds are planned (free), each by its own hook: the one chosen decides Launch and the plan line, the other is what its side of the switch says, and moving the switch needs no new answer.
  const planning = !planOverride && !blockedReason?.startsWith("a recorded");
  const fastPlan = usePlan(request ? { ...request, speed: "fast" } : null, planning);
  const faithfulPlan = usePlan(request ? { ...request, speed: "faithful" } : null, planning);
  const chosen = speed === "fast" ? fastPlan : faithfulPlan;
  const { plan, problem, loading } = planOverride ?? chosen;
  const plans: Record<Speed, PlanAnswer | null> = { fast: fastPlan.plan, faithful: faithfulPlan.plan };
  const busy = starting || running;
  // Launch waits for the plan: the cost and the cache hits are on the screen before a paid run can be started
  const canLaunch = request !== null && !busy && !blockedReason && !problem && plan !== null && !loading;
  const tick = plan?.decision_days ?? [];

  return (
    <Card title="Launch a test" className="launchpad" right="replays the real desk at past moments, on the stored data">
      <div className="card-b">
        <div className="lp-top">
          <div className="lp-request">
            <label className="f" htmlFor="pv-prompt">
              Your request, in your own words
            </label>
            <textarea id="pv-prompt" className="in" value={prompt} maxLength={600} onChange={(e) => setPrompt(e.target.value)} disabled={busy} />
            <div className="lp-quick" role="group" aria-label="Quick prompts">
              <span className="muted">Try:</span>
              {QUICK_TESTS.map((q) => (
                <button key={q.label} type="button" className="chip" disabled={busy} onClick={() => apply(q)} title={`${q.prompt}\n${q.title}`}>
                  {q.label}
                </button>
              ))}
            </div>
            <div className="lp-checks">
              <Check label="Neutral recommendations as passes" checked={strict ? false : neutral} disabled={busy || strict} onChange={(e) => setNeutral(e.target.checked)} />
              <Check label="Strict recommendations" checked={strict} disabled={busy} onChange={(e) => setStrict(e.target.checked)} title="Only what the Critic approves may trade" />
            </div>
          </div>
          <div className="lp-side">
            <div className="field">
              <label className="f" htmlFor="pv-cash">
                Starting cash
              </label>
              <div className="money">
                <input id="pv-cash" className="in num" inputMode="decimal" value={cash} onChange={(e) => setCash(e.target.value)} disabled={busy} aria-invalid={!cashOk} />
              </div>
            </div>
            <div className="field">
              <span className="f" id="pv-risk-l">
                Risk profile
              </span>
              <ChipGroup<RiskLevel> label="Risk profile" options={RISKS} value={risk} onChange={setRisk} disabled={busy} />
            </div>
            <div className="field">
              <span className="f" id="pv-speed-l">
                Speed
              </span>
              <SpeedSwitch value={speed} onChange={setSpeed} plans={plans} disabled={busy} />
            </div>
          </div>
        </div>

        <div className="lp-range">
          <div className="lp-range-bar">
            <span className="f">What to test</span>
            <ChipGroup<Cadence> label="How often the desk decides" options={CADENCES} value={cadence} onChange={setCadence} disabled={busy} />
            <div className="spacer" />
            <div className="chips" role="group" aria-label="Quick ranges">
              {QUICK.map((q) => (
                <button key={q.label} type="button" className="chip" disabled={busy || days.length < 2} onClick={() => setRange(quickRange(days, q.months))}>
                  {q.label}
                </button>
              ))}
            </div>
          </div>
          {coverage ? (
            range ? (
              <RangePicker coverage={coverage} range={range} onChange={setRange} decisionDays={tick} disabled={busy} />
            ) : (
              <div className="empty">
                <b>Nothing stored to test against yet</b>
                {coverage.usable?.first_day ? "" : "The Proving Ground needs stored day bars and news (scripts/backfill_history.py)."}
              </div>
            )
          ) : (
            <div className="empty">
              <b>{coverageError ?? "Reading what is stored…"}</b>
            </div>
          )}
        </div>

        <div className="lp-plan" role="status" aria-live="polite">
          <PlanLine plan={plan} problem={problem} loading={loading} range={range} />
        </div>
        {plan && plan.caveats.length > 0 && (
          <ul className="lp-caveats muted">
            {caveatItems(plan.caveats).map((c) => (
              <li key={c.code}>{c.sentence}</li>
            ))}
          </ul>
        )}

        <div className="lp-go">
          {error && (
            <p className="err" role="alert">
              {error}
            </p>
          )}
          {running ? (
            <Button variant="ghost" className="stop" onClick={onStop}>
              Stop the test
            </Button>
          ) : null}
          <Button variant="go" disabled={!canLaunch} onClick={() => request && onLaunch(request)} title={blockedReason ?? (problem ? problem : !request ? "Fill in the request, the cash and the range" : "Runs the real desk once for every decision day: paid model calls")}>
            {busy ? (
              <>
                <SmallGear />
                Running…
              </>
            ) : (
              "Launch"
            )}
          </Button>
        </div>
        {blockedReason && (
          <p className="hint lp-blocked" role="note">
            {blockedReason}
          </p>
        )}
      </div>
    </Card>
  );
}

function PlanLine({ plan, problem, loading, range }: { plan: PlanAnswer | null; problem: string | null; loading: boolean; range: { start: string; end: string } | null }) {
  if (problem) return <span className="err">{problem}</span>;
  if (!plan) return <span className="muted">{loading ? "Counting the decision days…" : range ? "The plan appears here." : "Choose a range."}</span>;
  const n = plan.decision_days.length;
  const time = planTimeWords(plan);
  return (
    <span className={loading ? "muted" : undefined}>
      <b className="num">{n}</b> decision day{n === 1 ? "" : "s"}
      {plan.cache_known ? (
        <>
          , <b className="num">{plan.cached_days.length}</b> from the cache
        </>
      ) : null}
      , <b className="num">{plan.paid_days}</b> paid desk run{plan.paid_days === 1 ? "" : "s"}
      {plan.cache_known ? null : <span className="muted"> (the cache is checked once these words have been read)</span>}
      {plan.cost_estimate_usd !== null ? (
        <>
          {" "}
          · about <b className="num">{money(plan.cost_estimate_usd)}</b>
        </>
      ) : null}
      {time ? (
        <>
          {" "}
          · <b>{time}</b>
        </>
      ) : null}
    </span>
  );
}
