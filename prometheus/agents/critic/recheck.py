"""Accuracy: ask the Data Clerk again, and compare what comes back with what the record says (Critic v1 spec section 8.1, steps 3 and 4).

The Critic does not recalculate the Advisor's work. It asks the Data Clerk, directly, for the few numbers and sources the recommendation rests on, and checks that the old and the new values
match. A Data Clerk or an Advisor that invented a number, a quote or a company cannot make a second, independent request agree with it.

    gathered = gather(clerk, order, items, now, config)                       # THREE kinds of request to the Data Clerk, all code, no model
    rechecks, findings = compare(rec, gathered.metrics_for(rec), names, decisive, config.accuracy)
    sources, findings = check_sources(rec, claims, order, gathered.verification, config.accuracy)

The three requests (one of each for ALL the records of a review, as the Advisor asks once for all its candidates):

    calculate_financials   `DataClerk.request_metrics`      the key metrics, recomputed from the stored bars and the Finnhub payload of today
    fetch_quote            `DataClerk.request_quotes`       the current price of every recommended symbol
    verify_data            `DataClerk.request_verification` the cited chunks and evidence items read again, the symbols looked up, the newest stored text

A request that fails is a FINDING, not a crash and not a pass (spec P3, P7): its results are simply empty, and the checks that needed them say so.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.chart import CHART_METRICS, TREND_METRICS
from prometheus.agents.critic.claims import Claim
from prometheus.agents.critic.common import blocker, metric_value, quote_in_text, warning
from prometheus.agents.critic.config import Accuracy, CriticConfig
from prometheus.agents.critic.names import same_company
from prometheus.agents.critic.record import CrossCheck, Finding, Identity, Measure, Recheck, SourceCheck
from prometheus.agents.data_clerk.metrics_catalog import METRIC_NAMES
from prometheus.common.market_calendar import eastern_date, is_trading_day, load_calendar, previous_trading_day
from prometheus.common.work_order import MAX_QUOTE_SYMBOLS, MAX_VERIFY_CHUNKS, WorkOrder, new_work_order, parse_utc

REQUESTER = "critic"
DECISIVE = ("last_close", "atr_14")
ALWAYS = ("sma_50", "vol_63d_ann", "beta_63d", "max_drawdown_63d", "atr_pct", "ret_21d", "earnings_yield", "negative_equity")
OWN = ("sma_126", "high_126d", "low_126d", "eps", "ps_ratio", "high_52w", "pct_of_52w_high", *CHART_METRICS)   # not on the record (nothing to compare): the Critic's own judgements of risk, value and chart read them
ASKED_ONLY = TREND_METRICS      # asked for with the review but never compared with the record's: the Critic's own reading (rule C5), which must change no accuracy number of any review (the share of values that matched)
BLEND_GROUPS = ("performance", "historical", "filings")


# ---------------------------------------------------------------- what to ask for

def key_metrics(rec: Recommendation, cap: int) -> tuple[list[str], set[str]]:
    """(the metric names to fetch again, the DECISIVE ones among them). Decisive: the close, the ATR and every number the strategy that opened the position was scored from; a mismatch in one of
    them is a blocker by itself. The rest are the numbers the record's risk and blend lean on (the stock's own prices, its history, its filings) and the Critic's own (`OWN`)."""
    used = [f for f in rec.families if f.family == rec.trade.family]
    decisive = [*DECISIVE, *(n for f in used for n in f.features_used if n in METRIC_NAMES and n in rec.features)]
    blend = [p["metric"] for g in BLEND_GROUPS for p in ((rec.blend or {}).get("components", {}).get(g, {}).get("parts", [])) if "metric" in p]
    decisive = list(dict.fromkeys(n for n in decisive if n in METRIC_NAMES))
    rest = [n for n in dict.fromkeys([*ALWAYS, *blend, *OWN, *ASKED_ONLY]) if n in METRIC_NAMES and n not in decisive]
    return [*decisive, *rest][:max(cap, len(decisive))], set(decisive)


@dataclass
class Gathered:
    """What the Data Clerk answered, for all the records of one review."""
    metrics: dict[tuple[str, str], dict] = field(default_factory=dict)     # (record date, symbol) -> {metric: result}
    quotes: dict[str, dict] = field(default_factory=dict)                  # symbol -> quote
    verification: dict = field(default_factory=dict)                       # chunks / evidence / symbols / fresh, merged over every verify order
    metric_orders: dict[str, str] = field(default_factory=dict)            # record date -> the request id
    quote_order: str | None = None
    verify_orders: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verified: bool = False                                                 # at least one verify order was answered

    def metrics_for(self, rec: Recommendation, record_date: str) -> dict | None:
        return self.metrics.get((record_date, rec.symbol))


def _attempt(label: str, errors: list[str], call):
    """Run one Data Clerk request; a failure is named in `errors` (the kind, never request details) and gives None."""
    try:
        return call()
    except Exception as error:  # a bug or an outage in the Data Clerk must not stop the review (spec P7)
        errors.append(f"{label}: {type(error).__name__}: {' '.join(str(error).split())[:160]}")
        return None


def gather(clerk, order: WorkOrder, items: list[tuple[Recommendation, str, list[Claim]]], now: datetime, config: CriticConfig) -> Gathered:
    """The three requests. `items` is [(record, record date "YYYY-MM-DD", its claims)] for the records that are being reviewed."""
    got = Gathered()
    by_date: dict[str, list[tuple[Recommendation, list[str]]]] = {}
    for rec, day, _ in items:
        by_date.setdefault(day, []).append((rec, key_metrics(rec, config.limits.max_metrics)[0]))
    for day, group in by_date.items():
        symbols = list(dict.fromkeys(rec.symbol for rec, _ in group))
        names = list(dict.fromkeys(n for _, ns in group for n in ns))
        sector = {rec.symbol: rec.identity.sector_etf for rec, _ in group if rec.identity.sector_etf}
        metric_order = new_work_order(f"metrics to check the recommendations of {day}", requester=REQUESTER, action_type="calculate_financials", subject=symbols,
                                      metrics=[{"name": n} for n in names], sector=sector or None)
        _attempt("metrics", got.errors, lambda: clerk.request_metrics(metric_order, date.fromisoformat(day)))
        got.metric_orders[day] = metric_order.request_id
        if metric_order.status == "data_failed":
            got.errors.extend(f"metrics: {e}" for e in metric_order.errors)
        for symbol, results in (metric_order.data.metrics if metric_order.data else {}).items():
            got.metrics[(day, symbol)] = results

    recommended = list(dict.fromkeys(rec.symbol for rec, _, _ in items))
    quote_order = new_work_order("current prices to check the recommendations", requester=REQUESTER, action_type="fetch_quote", subject=recommended[:MAX_QUOTE_SYMBOLS], live=True)
    _attempt("quotes", got.errors, lambda: clerk.request_quotes(quote_order))
    got.quote_order = quote_order.request_id
    got.quotes = dict(quote_order.data.quotes) if quote_order.data else {}
    if quote_order.status == "data_failed":
        got.errors.extend(f"quotes: {e}" for e in quote_order.errors)

    chunks = list(dict.fromkeys(c.chunk_id for _, _, claims in items for c in claims if c.chunk_id and c.kind != "evidence"))
    evidence: dict[str, list[str]] = {}
    for rec, _, claims in items:
        ids = [c.evidence_id for c in claims if c.kind == "evidence" and c.evidence_id]
        if ids:
            evidence[rec.symbol] = list(dict.fromkeys([*evidence.get(rec.symbol, []), *ids]))
    batches = [chunks[i:i + MAX_VERIFY_CHUNKS] for i in range(0, len(chunks), MAX_VERIFY_CHUNKS)] or [[]]
    merged = {"chunks": {}, "evidence": {}, "symbols": {}, "fresh": {}}
    for number, batch in enumerate(batches):
        spec: dict = {"chunks": batch}
        if number == 0:   # the symbols, the evidence and the newest text are asked once, with the first batch of chunks
            spec.update(evidence=evidence, symbols=recommended, company={rec.symbol: rec.company for rec, _, _ in items if rec.company},
                        fresh={"since_days": config.evidence.fresh_since_days, "k": config.evidence.fresh_k})
        verify_order = new_work_order("sources to check the recommendations", requester=REQUESTER, action_type="verify_data", subject=recommended, verify=spec)
        _attempt("verification", got.errors, lambda: clerk.request_verification(verify_order, now))
        got.verify_orders.append(verify_order.request_id)
        if verify_order.status == "data_failed":
            got.errors.extend(f"verification: {e}" for e in verify_order.errors)
        if verify_order.data and verify_order.data.verification:
            got.verified = True
            for key, part in verify_order.data.verification.items():
                if key == "evidence":   # {symbol: {id: item}}: a symbol may come back from two orders
                    for symbol, found in part.items():
                        merged["evidence"].setdefault(symbol, {}).update(found)
                else:
                    merged[key].update(part)
    got.verification = merged
    return got


# ---------------------------------------------------------------- old against new

def kind_of(source: str, name: str, config: Accuracy) -> str:
    """How a recorded number is checked, from where it came: the stored bars, the provider (Finnhub), a filing, or derived from more than one."""
    if source.startswith("derived:"):
        return "derived"
    if source.startswith("day_aggs:"):
        return "bars"
    if source.startswith("finnhub_metric:"):
        return "provider_price_linked" if name in config.price_linked else "provider"
    if source.startswith("filing:"):
        return "filing"
    return "derived"


def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _same(old, new, kind: str, config: Accuracy) -> tuple[bool, float | None]:
    """(match, the tolerance used). A bool or a text must be equal; a number within `max(rel x the larger, abs floor)`."""
    if isinstance(old, (bool, str)) or isinstance(new, (bool, str)):
        return type(old) is type(new) and old == new, None      # `1 == True` in Python: a number must not match a bool
    base = kind.removesuffix("_changed")
    rel, floor = config.tolerance[kind], config.abs_floor[base]
    return abs(new - old) <= max(rel * max(abs(old), abs(new)), floor), rel


def compare(rec: Recommendation, fetched: dict | None, names: list[str], decisive: set[str], config: Accuracy) -> tuple[list[Recheck], list[Finding], float | None]:
    """Compare every named value that the record holds with the one the Data Clerk just calculated. Returns (the comparisons, the findings, the share that matched or None when nothing could be
    compared). See the spec's table for the findings: `data_unverified` and `data_mismatch` are blockers, `bars_changed` and `statements_changed` are warnings."""
    findings: list[Finding] = []
    if not fetched:
        findings.append(blocker("accuracy", "data_unverified", f"The Data Clerk returned no numbers for {rec.symbol}, so none of the record's values could be checked."))
        return [], findings, None
    rechecks, changed = [], set()
    for name in names:
        if name in ASKED_ONLY and name not in decisive:
            continue
        old = rec.features.get(name)
        if old is None or isinstance(old.get("value"), list):
            continue
        new = fetched.get(name) or {}
        kind = kind_of(old.get("source") or "", name, config)
        if new.get("value") is not None and new.get("source") != old.get("source") and kind in ("bars", "filing"):
            changed.add(kind)        # the same number from other bars or another filing: a looser tolerance, and a warning for the reader
            kind = f"{kind}_changed"
        if new.get("value") is None:
            rechecks.append(Recheck(name=name, kind=kind, recorded=old["value"], fetched=None, match=False, decisive=name in decisive, note=new.get("note") or "the Data Clerk gave no value"))
            continue
        match, tolerance = _same(old["value"], new["value"], kind, config)
        rechecks.append(Recheck(name=name, kind=kind, recorded=old["value"], fetched=new["value"], tolerance=tolerance, match=match, decisive=name in decisive))
    if not rechecks:
        findings.append(blocker("accuracy", "data_unverified", f"None of {rec.symbol}'s recorded values could be compared with a new calculation."))
        return rechecks, findings, None
    rate = sum(r.match for r in rechecks) / len(rechecks)
    bad = [r for r in rechecks if r.decisive and not r.match]
    if bad:
        findings.append(blocker("accuracy", "data_mismatch", "A number the recommendation was scored from does not match a new calculation: " +
                                "; ".join(f"{r.name} was {_fmt(r.recorded)}, now {_fmt(r.fetched)}" for r in bad[:4]), value=[r.name for r in bad], threshold="the tolerance of its kind"))
    elif rate < config.min_match_rate:
        findings.append(blocker("accuracy", "data_mismatch", f"Only {rate:.0%} of the recorded values match a new calculation (the floor is {config.min_match_rate:.0%}): " +
                                ", ".join(r.name for r in rechecks if not r.match)[:200], value=rate, threshold=config.min_match_rate))
    if "bars" in changed:
        findings.append(warning("accuracy", "bars_changed", "The stored price bars are not the ones the record was made from (another range of dates): a looser tolerance was used."))
    if "filing" in changed:
        findings.append(warning("accuracy", "statements_changed", "The filing the record's statements came from is not the newest one stored now: a looser tolerance was used.", requires_human=True))
    return rechecks, findings, rate


# ---------------------------------------------------------------- the sources, read again

def check_sources(rec: Recommendation, claims: list[Claim], order: WorkOrder, verification: dict, verified: bool) -> tuple[list[SourceCheck], list[Finding], float]:
    """Every counted claim's source, read again. Returns (one check per claim, the findings, the share of claims whose source was found with its quote in it and unaltered: 1.0 when the record cites nothing)."""
    findings: list[Finding] = []
    checks: list[SourceCheck] = []
    held: list[bool] = []
    on_order = {s.get("number"): s for s in (order.data.sources if order.data else [])}
    evidence_on_order = {i.get("id"): i for i in ((order.data.filing_evidence.get(rec.symbol) or {}).get("evidence", []) if order.data else [])}
    for claim in claims:
        if claim.kind == "evidence":
            now = ((verification.get("evidence") or {}).get(rec.symbol) or {}).get(claim.evidence_id) or {}
            found = bool(now.get("found"))
            in_text = found and (quote_in_text(claim.quote, now.get("quote")) or quote_in_text(claim.quote, now.get("what")))
            before = evidence_on_order.get(claim.evidence_id) or {}
            same, tagged, altered = found and now.get("quote") == before.get("quote"), True, False
            note = now.get("note")
        else:
            now = (verification.get("chunks") or {}).get(claim.chunk_id) or {}
            found = bool(now.get("found"))
            in_text = found and quote_in_text(claim.quote, now.get("text"))
            before = on_order.get(claim.source) or {}
            same = found and (now.get("text") or "") == (before.get("text") or "")
            tagged = found and rec.symbol in (now.get("tickers") or [])
            altered = found and bool(before) and (now.get("headline") != before.get("headline") or now.get("published_utc") != before.get("published_utc"))
            note = now.get("note")
        held.append(found and in_text and not altered)
        checks.append(SourceCheck(ref=claim.ref, chunk_id=claim.chunk_id, found=found, quote_in_text=bool(in_text), same_text=bool(same) if found else None, tagged=tagged if found else None,
                                  note=note or ("altered" if altered else None)))
        if not found:
            why = note if note else ("the Data Clerk's verification request failed" if not verified else "it was not returned")
            findings.append(blocker("accuracy", "source_missing", f"{claim.ref} ({claim.chunk_id or claim.evidence_id}) could not be read again: {why}.", value=claim.ref))
        elif not in_text:
            findings.append(blocker("accuracy", "source_missing", f"The quote of {claim.ref} is no longer in its source (read again from the store today).", value=claim.ref))
        elif altered:
            findings.append(blocker("accuracy", "source_altered", f"{claim.ref}: the stored headline or date is not the one the record was made from.", value=claim.ref))
        elif not same:
            findings.append(warning("accuracy", "source_text_changed", f"{claim.ref}: the quote is still there but the rest of the source's text changed.", value=claim.ref))
        if found and in_text and not tagged:
            findings.append(warning("accuracy", "source_not_tagged", f"{claim.ref}: the source read again does not list {rec.symbol} among its tickers, so it is not counted as evidence for it.", value=claim.ref))
    rate = sum(held) / len(held) if held else 1.0
    return checks, findings, rate


# ---------------------------------------------------------------- cross-checks against a second provider and the directory

def _regular_session_print(moment: datetime) -> bool:
    """True when `moment` is a trade in the regular session, including the closing print at the close itself. A pre-market or after-hours print is not the daily bar's close (the bar is the regular-session close)."""
    config = load_calendar()
    local = moment.astimezone(ZoneInfo(config.timezone))
    if not is_trading_day(local.date(), config):
        return False
    close = config.early_close if local.date() in config.early_close_dates else config.regular_close
    return config.regular_open <= local.time() <= close


def _close_from_quote(quote: dict, bar_day: date) -> float | None:
    """The other provider's close for the stored bar's session, or None when the quote does not carry that close. The quote's `price` is the last trade. It is that session's close only when the trade
    happened in the regular session (the closing print is at the close). An after-hours or pre-market print on the bar's own date is not compared: it can differ from the official close by more than the
    tolerance, and that difference is not two providers disagreeing. The bar from the trading day before the quote's date is compared with `previous_close`."""
    quoted_on = parse_utc(quote.get("as_of_utc"))
    if quoted_on is None:
        return None
    quote_day = eastern_date(quoted_on)
    if bar_day == previous_trading_day(quote_day):
        previous = quote.get("previous_close")
        return float(previous) if previous else None
    if bar_day == quote_day and _regular_session_print(quoted_on):
        return float(quote["price"]) if quote.get("price") else None
    return None


def cross_checks(rec: Recommendation, fetched: dict | None, quote: dict | None, verification: dict, config: Accuracy) -> tuple[list[CrossCheck], list[Finding], Identity, float]:
    """Checks that need a SECOND source: the close against the other price provider's, the symbol against the directory, the 52-week high against Finnhub's. Returns (the checks performed, the
    findings, the identity, the share that passed: 0.0 when none could be performed)."""
    findings: list[Finding] = []
    checks: list[CrossCheck] = []
    last, high = (fetched or {}).get("last_close") or {}, (fetched or {}).get("high_52w") or {}
    if last.get("value") is not None and quote and quote.get("price") and not quote.get("simulated"):   # a SIMULATED quote (USE_LIVE_TRADING=false, entry 76) is made from the stored bar: not a second provider
        bar_day = date.fromisoformat(last["as_of"]) if last.get("as_of") else None
        against = _close_from_quote(quote, bar_day) if bar_day is not None else None
        if against:
            passed = abs(last["value"] - against) <= config.cross_price_rel * max(last["value"], against)
            checks.append(CrossCheck(name="price", a=last["value"], b=against, tolerance=config.cross_price_rel, passed=passed))
            if not passed:
                findings.append(blocker("accuracy", "price_mismatch", f"The stored close {last['value']:g} and the other provider's {against:g} differ by more than {config.cross_price_rel:.1%}.",
                                        value=[last["value"], against], threshold=config.cross_price_rel))
    # the symbol against the directory
    listing = ((verification.get("symbols") or {}).get(rec.symbol)) or {}
    identity = Identity(listed=listing.get("listed"), listed_company=listing.get("company"), company=rec.company)
    if listing.get("listed") is False:
        checks.append(CrossCheck(name="identity", a=rec.symbol, b=None, passed=False, note=listing.get("note")))
        findings.append(blocker("accuracy", "symbol_unknown", f"{rec.symbol} is not in the symbol directory."))
    elif listing.get("listed") is True:
        mismatch = bool(rec.company and listing.get("company") and not same_company(rec.symbol, rec.company, listing["company"]))      # Critic v2, rule C4: the names are compared in the Critic's own `names.py`, not with the Advisor's rule for a quote
        checks.append(CrossCheck(name="identity", a=rec.company, b=listing.get("company"), passed=not mismatch, note="the names are not the same company, with the case, spacing, punctuation and corporate words taken out" if mismatch else None))
        if mismatch:
            findings.append(warning("accuracy", "company_mismatch", f"The record calls {rec.symbol} {rec.company!r} but the directory lists it as {listing['company']!r}.", requires_human=True))
    else:
        findings.append(warning("accuracy", "identity_unverified", f"{rec.symbol} could not be looked up in the symbol directory ({listing.get('note') or 'no answer'}).", requires_human=True))
    # the 52-week high: what the bars say against what Finnhub says (the bars are unadjusted: a split shows here)
    pct = metric_value(fetched, "pct_of_52w_high")
    if last.get("value") is not None and pct and high.get("value"):
        from_bars = last["value"] / pct
        passed = abs(from_bars - high["value"]) <= config.cross_high_52w_rel * max(from_bars, high["value"])
        checks.append(CrossCheck(name="high_52w", a=round(from_bars, 4), b=high["value"], tolerance=config.cross_high_52w_rel, passed=passed))
        if not passed:
            findings.append(warning("accuracy", "high_52w_differs", f"The 52-week high from the bars ({from_bars:.2f}) and Finnhub's ({high['value']:.2f}) differ by more than {config.cross_high_52w_rel:.0%} (a split?)."))
    rate = sum(c.passed for c in checks) / len(checks) if checks else 0.0
    return checks, findings, identity, rate


def accuracy_score(match_rate: float | None, source_rate: float, cross_rate: float, config: Accuracy) -> tuple[float, list[Measure]]:
    """`100 x (0.5 x the share of values that matched + 0.3 x the share of sources that held + 0.2 x the share of cross-checks that passed)`: a share that could not be measured counts 0."""
    parts = config.parts
    shares = (("metrics_match", match_rate or 0.0, parts.metrics, match_rate is not None), ("sources_held", source_rate, parts.sources, True), ("cross_checks_passed", cross_rate, parts.cross, True))
    measures = [Measure(name=name, x=round(value, 4), value=round(value, 4), weight=weight * 100, available=available) for name, value, weight, available in shares]
    return round(100 * sum(value * weight for _, value, weight, _ in shares), 2), measures
