"""`Critic.review_recommendations`: the whole step, in order (Critic v1 spec section 7, design choices entry 59; the walk-through is design/how_to_code_critic.md).

    order = Critic().review_recommendations(order, risk_view, now)

ONE work order in (status `recommended`), the SAME work order out (status `reviewed`), with one review per recommendation on `order.reviews`. The nine steps:

    1  preconditions    the order has recommendations; `now` has a timezone; an order already reviewed comes back unchanged
    2  integrity        each record against itself and against the order (integrity.py): the Advisor's arithmetic worked again, every quote looked up
    3  four requests    to the Data Clerk, for ALL the records at once: the key metrics again, a current quote, the sources read again, the analysts' recommendation counts (recheck.gather)
    4  accuracy         what came back against what the record says (recheck.compare, check_sources, cross_checks)
    5  evidence         ONE model call per record: does each quote really support the claim? plus a control claim; then the return worked again without the quotes that did not hold (evidence.py)
    6  timing           the market's session, the record's date, the price against the entry zone (timing.py)
    7  risk fit         the risk against the user's profile: the company's risk, never how much to buy (risk.py)
    8  value            the price against its six-month average, the valuation (value.py); then the chart patterns: crosses, structure, dead cat bounce, the swing patterns (chart.py, patterns.py), the returns (performance); then the analysts (analysts.py)
    9  verdict          the seven scores, the bonuses, the findings, the score, the verdict (APPROVED or DENIED, nothing between: a fault or a profile finding denies alone, bad signs in two of the groups news, performance and price deny, and a finding that
                        asks for a person to look denies it too, `needs_human_review`, entry 103; reasons.py, scoring.py), the summary; write out

The Critic only deals with recommendations TO BUY (entry 105: a record that is not one, or that says something is held, is a `record_defect` and is not reviewed): it does not know the user's money (it is given the profile's `RiskView`, which holds no size limit, and no portfolio), and it sizes nothing. How much to buy is the Account Manager's (agents/account_manager/sizing.py).

(In the code, timing runs before the evidence: the evidence step does not need it, but the risk and the value steps need the deal price it finds.) Every check runs for every record (spec P6):
the user reads the whole scorecard. Nothing raises for a data, API or model failure (P7): a failed step is a finding, and what could not be checked is not approved (P3). The Advisor's records are
never edited (P5): the reviews are written beside them and joined by `record_id`; the only thing added to a record is its `critic` block (feedback.py, Amendment 2).
"""
import concurrent.futures
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from pydantic import ValidationError

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic import evidence as ev
from prometheus.agents.critic import analysts, chart, feedback, integrity, recheck, risk, scoring, timing, value
from prometheus.agents.critic.claims import Claim, claims_of
from prometheus.agents.critic.common import blocker, et_text, metric_value, utc_text, warning
from prometheus.agents.critic.record import AccuracyDetail, AnalystsDetail, Audit, ChartDetail, Dimension, Finding, Integrity, Review, RiskDetail, TimingDetail, ValueDetail
from prometheus.common.market_calendar import eastern_date, session_at
from prometheus.common.user_profile import RiskView
from prometheus.common.trace import traced
from prometheus.common.trading_clock import trading_now
from prometheus.common.work_order import WorkOrder


@dataclass
class Work:
    """One record on its way through the steps: what each step found, until the verdict is made."""
    index: int
    raw: dict
    rec: Recommendation | None = None
    day: date | None = None
    claims: list[Claim] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    dimensions: dict[str, Dimension] = field(default_factory=dict)
    integrity_result: integrity.IntegrityResult | None = None
    rechecks: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    cross: list = field(default_factory=list)
    identity: object = None
    match_rate: float | None = None
    evidence_result: ev.EvidenceResult | None = None
    timing_result: timing.TimingResult | None = None
    risk_result: risk.RiskResult | None = None
    value_result: value.ValueResult | None = None
    chart_result: chart.ChartResult | None = None
    analysts_result: analysts.AnalystsResult | None = None
    metric_order: str | None = None


def _schema_problem(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:3])


def _classify(w: Work, limit: int) -> bool:
    """Whether the record can be reviewed at all. One that cannot gets a blocker that says why and no other check: it is beyond the review's limit, an Advisor version 0 record (no schema 2
    sections), it is not a recommendation to BUY (the Critic judges nothing else: a holding, a sale, a hold; design choices entry 105), or it does not match the schema (a record that says something is held
    does not: the models refuse it)."""
    if w.index >= limit:
        w.findings.append(blocker("accuracy", "record_not_reviewable", f"The review covers the first {limit} recommendations; this one is beyond them."))
    elif not isinstance(w.raw.get("identity"), dict):
        w.findings.append(blocker("accuracy", "record_not_reviewable", "The record has no schema 2 sections (an Advisor version 0 record): there is nothing the Critic can check it against."))
    else:
        action = w.raw["action"].get("action") if isinstance(w.raw.get("action"), dict) else None
        if isinstance(action, str) and action != "open":
            w.findings.append(blocker("accuracy", "record_defect", f"The record is not a recommendation to buy (its action is {action!r}): the Critic does not judge a holding or a sale."))
        else:
            try:
                w.rec = Recommendation.model_validate(w.raw)
            except ValidationError as error:
                w.findings.append(blocker("accuracy", "record_defect", f"The record does not match the Advisor's record schema: {_schema_problem(error)}."))
    return w.rec is not None


@traced("critic.review_order")
def review_order(critic, order: WorkOrder, view: RiskView, now: datetime | None = None) -> WorkOrder:
    """The nine steps (see the module text). Returns the SAME order."""
    started = time.monotonic()
    now = now or trading_now()   # the real clock, or the simulated one when USE_LIVE_TRADING=false (entry 76)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    config, advisor = critic.config, critic.advisor_config
    # ---- 1 preconditions
    if order.status in ("reviewed", "decided", "traded"):   # already reviewed; a decided or traded order (Trader v1) was reviewed before it got there
        return order
    if order.status != "recommended":
        order.errors = [*order.errors, f"the Critic needs an order with recommendations (status {order.status})"]
        return order
    model = critic.model
    run = {"stage": "review", "model": critic.model_name, "tokens_in": 0, "tokens_out": 0, "calls": 0, "dropped": [], "error": None, "errors": list(critic.errors), "records": len(order.recommendations),
           "records_called": 0, "controls_failed": 0, "labels_from_store": 0, "metric_order": None, "quote_order": None, "verify_order": None, "analysts_order": None, "independent_model": None, "advisor_model": None, "seconds": 0.0,
           "tolerance": view.risk_tolerance, "now_et": et_text(now), "session": session_at(now).state, "ignore_market_hours": timing.market_hours_ignored()}
    works = [Work(i, raw) for i, raw in enumerate(order.recommendations)]

    # ---- 2 integrity (no network)
    valid = [w for w in works if _classify(w, config.limits.max_records)]
    pairs: dict[str, int] = {}
    for w in valid:
        pairs[w.rec.symbol] = pairs.get(w.rec.symbol, 0) + 1
    reviewable = []
    for w in valid:
        try:
            w.day = timing.record_day(w.rec, now)
            w.integrity_result = integrity.check(w.rec, order, advisor, duplicate=pairs[w.rec.symbol] > 1, max_feature_age_days=config.accuracy.max_feature_age_days)
            w.claims = claims_of(w.rec, order)
        except Exception as error:  # a record so damaged that its checks could not run: that is a defect of the record, said plainly
            w.findings.append(blocker("accuracy", "record_defect", f"The record could not be checked ({type(error).__name__}): it is damaged."))
            run["errors"].append(f"{w.raw.get('symbol', '?')}: integrity raised {type(error).__name__}")
            continue
        w.findings += [blocker("accuracy", "record_defect", f"{name}: {why}.", value=name) for name, why in w.integrity_result.failures]
        if w.integrity_result.no_thesis:
            w.findings.append(warning("accuracy", "no_thesis", "The record has neither a thesis nor the reason it has none."))
        reviewable.append(w)

    # ---- 3 the four requests to the Data Clerk
    items = [(w.rec, str(w.day or eastern_date(now)), w.claims) for w in reviewable]
    gathered = recheck.gather(critic.clerk, order, items, now, config) if items else recheck.Gathered()
    run["errors"].extend(gathered.errors)
    run["metric_order"] = ",".join(gathered.metric_orders.values()) or None
    run["quote_order"], run["verify_order"], run["analysts_order"] = gathered.quote_order, ",".join(gathered.verify_orders) or None, gathered.analysts_order

    # ---- 4 to 8, per record (one after the other, as always; `critic.parallel` above 1 reviews the records at once, see `_check_all`)
    _check_all(reviewable, critic, model, order, view, gathered, now, run)

    # ---- 9 the verdicts
    reviews = [_build(w, critic, order, now, gathered, run) for w in works]
    run["seconds"] = round(time.monotonic() - started, 1)
    run["error"] = run["errors"][0] if run["errors"] else None
    order.reviews = [r.model_dump(mode="json") for r in reviews]
    order.critic_runs = [*order.critic_runs, run]
    feedback.attach(order, config)         # the feedback goes on each recommendation too, for the user and a screen: the Advisor's own fields are not touched
    order.schema_version = max(order.schema_version, 3)
    order.status = "reviewed"
    for review in order.reviews:
        critic.ledger.append(review)      # append only: the reviews are the Critic's record, and the ledger is never read back by this step
    return order


NOT_SET = object()      # a part of a parallel review that never reached the line that sets a field of the run
COUNTERS = ("tokens_in", "tokens_out", "calls", "records_called", "controls_failed", "labels_from_store")
SET_BY_CHECK = ("independent_model", "advisor_model")


def _guarded(w: Work, critic, model, order: WorkOrder, view: RiskView, gathered: recheck.Gathered, now: datetime, run: dict) -> None:
    """`_check` for one record, never raising: a bug in a check must never become an approval (P3), nor a crash (P7)."""
    try:
        _check(w, critic, model, order, view, gathered, now, run)
    except Exception as error:
        w.findings.append(blocker("accuracy", "record_defect", f"The review of this record failed ({type(error).__name__}): it cannot be approved."))
        run["errors"].append(f"{w.rec.symbol}: the review raised {type(error).__name__}")
        w.dimensions = {}


def _check_all(reviewable: list[Work], critic, model, order: WorkOrder, view: RiskView, gathered: recheck.Gathered, now: datetime, run: dict) -> None:
    """Steps 4 to 8 for every record. With `critic.parallel` of 1 (the default, and every live run) the records are checked one after the other, exactly as before. With more, the records are checked AT ONCE (the model call of each
    is the slow part, and the records do not depend on one another), each in a copy of the caller's context (the as-of day and the trace follow) and with a run record of its own, and the parts are put back IN THE ORDER OF THE RECORDS:
    the counters are added, the drops and the errors are appended record by record, and the fields a check sets (`independent_model`, `advisor_model`) take the last record's value, so the review that comes out is the one the sequential
    loop makes. It stays sequential when the review's model-call limit could bind (more records than `limits.max_model_calls`): then WHICH record gets its call depends on the order, and the order stays the loop's."""
    workers = min(critic.parallel, len(reviewable))
    if workers <= 1 or len(reviewable) > critic.config.limits.max_model_calls:
        for w in reviewable:
            _guarded(w, critic, model, order, view, gathered, now, run)
        return
    parts = []
    for _ in reviewable:
        part = {k: (list(v) if isinstance(v, list) else v) for k, v in run.items()}
        part.update({k: 0 for k in COUNTERS}, dropped=[], errors=[], **{k: NOT_SET for k in SET_BY_CHECK})
        parts.append(part)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, _guarded, w, critic, model, order, view, gathered, now, part) for w, part in zip(reviewable, parts)]
        for future in futures:
            future.result()
    for part in parts:
        for key in COUNTERS:
            run[key] += part[key]
        run["dropped"].extend(part["dropped"])
        run["errors"].extend(part["errors"])
        for key in SET_BY_CHECK:
            if part[key] is not NOT_SET:
                run[key] = part[key]


def _check(w: Work, critic, model, order: WorkOrder, view: RiskView, gathered: recheck.Gathered, now: datetime, run: dict) -> None:
    """Steps 4 to 8 for one record: fills the work's parts and findings."""
    rec, config, advisor = w.rec, critic.config, critic.advisor_config
    day = str(w.day or eastern_date(now))
    fetched, quote = gathered.metrics_for(rec, day), gathered.quotes.get(rec.symbol)
    w.metric_order = gathered.metric_orders.get(day)
    names, decisive = recheck.key_metrics(rec, config.limits.max_metrics)
    # 4 accuracy
    w.rechecks, found, w.match_rate = recheck.compare(rec, fetched, names, decisive, config.accuracy)
    w.findings += found
    w.sources, found, source_rate = recheck.check_sources(rec, w.claims, order, gathered.verification, gathered.verified)
    w.findings += found
    w.cross, found, w.identity, cross_rate = recheck.cross_checks(rec, fetched, quote, gathered.verification, config.accuracy)
    w.findings += found
    accuracy, accuracy_measures = recheck.accuracy_score(w.match_rate, source_rate, cross_rate, config.accuracy)
    # 6 timing (before the risk and the value, which need the deal price it finds)
    w.timing_result = timing.assess(rec, now, quote, (fetched or {}).get("last_close"), config.timing, ignore_market_hours=run["ignore_market_hours"])
    w.findings += w.timing_result.findings
    # 5 evidence
    independent = ev.independence(critic.model_name, rec.advisor.get("model"))
    run["independent_model"], run["advisor_model"] = independent, rec.advisor.get("model")
    w.evidence_result = ev.assess(model, rec, w.claims, w.sources, gathered.verification, order, advisor, config, now, critic.model_name, independent, run, store=critic.label_store)
    w.findings += w.evidence_result.findings
    # 7 risk fit
    w.risk_result = risk.assess(rec, view, fetched, w.timing_result.deal_price, config.risk)
    w.findings += w.risk_result.findings
    # 8 value
    w.value_result = value.assess(rec, fetched, w.timing_result.deal_price, config.value, config.verdict.unknown_credit)
    w.findings += w.value_result.findings
    w.findings += value.history_warnings(rec, w.timing_result.deal_price, critic.history, eastern_date(now), config.value)
    w.chart_result = chart.assess(rec, fetched, config.chart)
    w.findings += w.chart_result.findings
    w.findings += chart.poor_performance(rec, metric_value(fetched, "ret_63d"), metric_value(fetched, "ret_21d"), config.performance)
    w.analysts_result = analysts.assess(rec.symbol, gathered.analysts.get(rec.symbol), config.analysts)
    w.findings += w.analysts_result.findings
    dimension = lambda name, score, applicable=True, measures=None: scoring.make_dimension(name, config, score, applicable, measures)
    w.dimensions = {"accuracy": dimension("accuracy", accuracy, True, accuracy_measures), "evidence": dimension("evidence", w.evidence_result.score, True, w.evidence_result.measures),
                    "risk_fit": dimension("risk_fit", w.risk_result.score, w.risk_result.applicable, w.risk_result.measures),
                    "timing": dimension("timing", w.timing_result.score, True, w.timing_result.measures),
                    "value": dimension("value", w.value_result.score, w.value_result.applicable, w.value_result.measures),
                    "chart": dimension("chart", w.chart_result.score, w.chart_result.applicable, w.chart_result.measures),
                    "analysts": dimension("analysts", w.analysts_result.score, w.analysts_result.applicable, w.analysts_result.measures)}


def _build(w: Work, critic, order: WorkOrder, now: datetime, gathered: recheck.Gathered, run: dict) -> Review:
    """The Review of one record, from its work: the bonus, the verdict, the summary, how long an approval lasts."""
    config, rec, raw = critic.config, w.rec, w.raw
    identity = raw.get("identity") if isinstance(raw.get("identity"), dict) else {}
    action = (raw.get("action") or {}).get("action", "?") if isinstance(raw.get("action"), dict) else "?"
    if not w.dimensions:   # a record that could not be reviewed (or whose review failed): every dimension at 0, and the blocker says why
        w.dimensions = {name: scoring.make_dimension(name, config, 0.0) for name in config.weights}
    bonus = 0.0
    if rec is not None and w.value_result is not None and w.value_result.applicable:
        red_flag_passed = next((g.passed for g in rec.gates if g.gate == "red_flag"), False)
        if scoring.great_deal_bonus_allowed(w.findings, rec.quality.multiplier, red_flag_passed, config.value.bonus.min_quality):
            bonus = value.bonus_points(w.value_result.discount, config.value.bonus)
    pattern_bonus = chart.pattern_bonus(w.chart_result.detail.points, config.chart.bonus) if w.chart_result is not None and w.chart_result.applicable else 0.0
    verdict = scoring.decide(w.findings, w.dimensions, bonus, config, pattern_bonus)
    blockers, warnings, notes = scoring.partition(w.findings, verdict)
    until = None
    if verdict.approved and w.timing_result is not None and w.timing_result.session is not None:
        until = utc_text(timing.valid_until(w.timing_result.session, now, config.timing.review_ttl_minutes))
    advisor_model = rec.advisor.get("model") if rec else None
    audit = Audit(critic_model=critic.model_name, advisor_model=advisor_model, independent_model=ev.independence(critic.model_name, advisor_model) if rec else None, config_version=config.config_version,
                  metric_order=w.metric_order, quote_order=gathered.quote_order, verify_order=",".join(gathered.verify_orders) or None, analysts_order=gathered.analysts_order,
                  run={k: run[k] for k in ("model", "calls", "tokens_in", "tokens_out", "records_called", "controls_failed", "labels_from_store")})
    return Review(
        review_id=uuid.uuid4().hex[:12], record_id=identity.get("record_id") or f"record-{w.index}", request_id=identity.get("request_id") or order.request_id, symbol=raw.get("symbol", "?"), action=action,
        reviewed_utc=utc_text(now), valid_until_utc=until, approved=verdict.approved, score=verdict.score, score_before_blockers=verdict.score_before_blockers, grade=verdict.grade, bonus=bonus,
        pattern_bonus=pattern_bonus, groups=verdict.groups,
        requires_human=verdict.requires_human, requires_human_reasons=verdict.requires_human_reasons, denial_reasons=verdict.denial_reasons,
        summary=scoring.summarize(raw.get("symbol", "?"), verdict, w.findings, w.dimensions),
        blockers=blockers, warnings=warnings, notes=notes, dimensions=w.dimensions,
        accuracy=_accuracy_detail(w), evidence=w.evidence_result.detail if w.evidence_result else ev.EvidenceResult().detail,
        timing=w.timing_result.detail if w.timing_result and w.timing_result.detail else _closed_timing(now), risk=w.risk_result.detail if w.risk_result else RiskDetail(),
        value=w.value_result.detail if w.value_result else ValueDetail(), chart=w.chart_result.detail if w.chart_result else ChartDetail(),
        analysts=w.analysts_result.detail if w.analysts_result else AnalystsDetail(), audit=audit)


def _accuracy_detail(w: Work) -> AccuracyDetail:
    checked, failed = (w.integrity_result.checked, w.integrity_result.failed) if w.integrity_result else ([], [])
    detail = AccuracyDetail(integrity=Integrity(checked=checked, failed=failed), rechecked=w.rechecks, sources=w.sources, cross=w.cross, match_rate=w.match_rate)
    if w.identity is not None:
        detail.identity = w.identity
    return detail


def _closed_timing(now: datetime) -> TimingDetail:
    """The timing section of a record that could not be reviewed: just where the market is."""
    session = session_at(now)
    return TimingDetail(session=session.state, trading_date=str(session.trading_date) if session.trading_date else None, session_reason=session.reason,
                        window_closes_utc=utc_text(session.window_closes_utc), next_open_utc=utc_text(session.next_open_utc))
