"""The Track Record: every recommendation the desk already made in a real run, scored against the stored bars (design/proving_ground_spec.md section 10; P4, P5, P8, P12; design choices entry 126).

    run = build()                       # a `ProvingRun` of kind "track": outcomes, statistics, the "$1 in every traded recommendation" index against SPY, the caveats
    print(report(run))

FREE: stored bars only, no model, no network. The runs are the saved sessions (`data/proving/sessions/`, written at the end of every live run of the page and of `session.py`: `sessions.save_session`, source "run") and the real captures kept
with the tests (`tests/fixtures/session_real/`, source "fixture"), so the page is not empty on day one. Every capture is read by `recs.recs_of`: each recommendation of the Advisor, and each company it scored neutral that the Account
Manager decided on, with what became of it. The outcome of a recommendation the desk REALLY bought is scored from the REAL fill (its price, its shares and its date: never recomputed); every other one is HYPOTHETICAL (P8): one
share-equivalent bought at the next stored open after the decision day and followed by its plan. A neutral company has no plan: its outcome is followed to the last stored close and is `open`.
THE LAST BAR. Nothing dated today is read as a close (`bars.usable_last_day`): a recommendation made today has no outcome yet (`no_data`: the future is not stored), and the run says `last_bar_day`.
THE INDEX. "$1 in every traded recommendation": each REAL purchase puts one dollar in its stock on its entry day and the same dollar in SPY at the open of that day; the curve is 100 times the average value of those dollars on each stored day
(a closed position keeps its exit value, an open one is marked at the day's close), SPY's the same. It is written as the run's `equity` (equity = the index of the stocks, benchmark = SPY's, cash 0) with NO statistics of its own: the run's
`stats.portfolio` is None, as the spec says. The outcome's source ("run" or "fixture") is in `coverage["sources"]` (by outcome key); the captures read are in `coverage["captures"]`.
Money is `Decimal`, a return a measurement. Cached in memory by the captures' files and the last usable bar day.
"""
import time
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from prometheus.proving import bars as bars_door
from prometheus.proving import outcomes, recs, record, splits as splits_mod, stats as stats_mod
from prometheus.proving.config import ProvingConfig, load_config
from prometheus.proving.record import DecisionDay, EquityPoint, ProvingRun, RecOutcome, RunRange
from prometheus.proving.store import ProvingStore
from prometheus.agents.trader.config import load_config as load_trader_config

LOOKBACK_DAYS = 45          # calendar days of bars read before the first decision day: the plan's reference price and the benchmark's first open
_CACHE: dict = {"stamp": None, "run": None}


def caveats_of(outcome_list: list[RecOutcome], config: ProvingConfig, *, simulated_fills: bool = False, thin_news: bool = False, models_may_remember: bool = False, not_point_in_time: bool = False, fast_models: bool = False, analysts_used: bool = False) -> list[str]:
    """The caveat codes that apply (the closed list of record.CAVEATS, code decides): returns are price returns (always), a stop and a target are judged on daily bars, a hypothetical entry fills at the next open, an open position is
    marked, a split was adjusted, a small sample, and the Time Machine's own (fundamentals and the directory are today's, the models may remember, the news was thin, and, when a review used them, the analysts' counts are each
    month's last revision: Critic v3)."""
    codes = []
    if not_point_in_time:
        codes += ["fundamentals_not_point_in_time", "symbol_directory_is_todays"]
        if analysts_used:
            codes.append("analysts_not_point_in_time")
    codes.append("price_returns_no_dividends")
    if simulated_fills or any(o.hypothetical for o in outcome_list):
        codes.append("fills_at_next_open_no_slippage")
    codes.append("daily_bars_stop_before_target")
    if thin_news:
        codes.append("news_coverage_thin")
    if sum(1 for o in outcome_list if o.result in ("won", "lost")) < config.min_sample:
        codes.append("small_sample")
    if any(o.result == "open" for o in outcome_list):
        codes.append("open_positions_marked")
    if any(o.split_adjusted for o in outcome_list):
        codes.append("split_adjusted")
    if models_may_remember:
        codes.append("model_knowledge_cutoff")
    if fast_models:
        codes.append("fast_models")
    return codes


def _sector_of(order, rec: recs.Rec) -> str | None:
    for recommendation in [rec.record, *order.recommendations]:
        sector = ((recommendation or {}).get("identity") or {}).get("sector_etf")
        if isinstance(sector, str) and sector:
            return sector
    return None


def score_rec(order, rec: recs.Rec, day: date, bars: dict[str, list[dict]], config: ProvingConfig, trader_config, note: str | None = None) -> RecOutcome:
    """The outcome of one recommendation of a work order on a decision day: its identity from the order, its plan, the REAL fill when the desk bought it (else hypothetical, P8), the stored bars of the symbol, the benchmark and the sector.
    `note` is added to the outcome's notes (the engine says why a recommendation was held back)."""
    identity = record.identity_of(order, rec.record, day)
    if rec.kind == "neutral":
        identity["verdict"] = "neutral"
    identity["not_traded_reason"] = rec.not_traded_reason
    table, why = splits_mod.read(rec.symbol)
    sector = _sector_of(order, rec)
    done = outcomes.score(record.plan_of(rec.record), bars.get(rec.symbol, []), bars.get(config.benchmark), bars.get(sector) if sector else None, day, config.horizons, symbol=rec.symbol, identity=identity,
                          fill=rec.fill, config=config, trader_config=trader_config, splits=table, table_missing=why)
    return done.model_copy(update={"notes": [*done.notes, note]}) if note else done


def decision_row(day: date, order, found: list[recs.Rec], scored: list[RecOutcome], reason: str | None) -> DecisionDay:
    """One decision day's row: what was recommended, approved, denied and traded, and the money the real fills spent."""
    spent = sum((Decimal(str(r.fill.price)) * Decimal(str(r.fill.shares)) for r in found if r.fill is not None), Decimal(0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return DecisionDay(day=day, status="traded" if any(o.traded for o in scored) else "no_trade", recommended=sum(1 for r in found if r.kind == "recommendation"), approved=sum(1 for o in scored if o.verdict == "approved"),
                       denied=sum(1 for o in scored if o.verdict == "denied"), traded=sum(1 for o in scored if o.traded), spent=spent, reason=reason)


def score_capture(capture: recs.Capture, bars: dict[str, list[dict]], config: ProvingConfig, trader_config, today_bound: date) -> tuple[list[RecOutcome], DecisionDay | None]:
    """The outcomes of one capture (see the module text) and its row for the run's decision days. `bars` holds every symbol the capture needs (the call has read them for the whole run)."""
    order = recs.order_of(capture)
    day = recs.decision_day_of(order)
    found = recs.recs_of(order, capture.payload.get("intent"))
    if day is None or not found:
        return [], None
    scored = [score_rec(order, rec, day, bars, config, trader_config) for rec in found]
    row = decision_row(day, order, found, scored, f"{capture.source}: {capture.name}")
    row.trace_id = capture.payload.get("langsmith_trace_id")
    return scored, row


def _index_curve(traded: list[RecOutcome], days: list[str]) -> list[EquityPoint]:
    """The "$1 in every traded recommendation" index and SPY's, on each stored day from the first entry (see the module text)."""
    if not traded:
        return []
    paths = []
    for o in traded:
        by_day = {p.date.isoformat(): p for p in o.path}
        paths.append((o.entry.date.isoformat(), o.path[-1].date.isoformat() if o.exit is None else o.exit.date.isoformat(), by_day, o.path))
    points, peak = [], Decimal(0)
    for day in days:
        values, benches = [], []
        for entered, last, by_day, path in paths:
            if day < entered:
                continue
            known = [p for p in path if p.date.isoformat() <= day]
            if not known or known[-1].benchmark_return_pct is None:
                continue
            values.append(Decimal(1) + Decimal(str(known[-1].return_pct)))
            benches.append(Decimal(1) + Decimal(str(known[-1].benchmark_return_pct)))
        if not values:
            continue
        index = (Decimal(100) * sum(values) / len(values)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        bench = (Decimal(100) * sum(benches) / len(benches)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        peak = max(peak, index)
        points.append(EquityPoint(date=date.fromisoformat(day), equity=index, cash=Decimal("0.00"), invested=index, benchmark=bench, drawdown_pct=float((index / peak - 1).quantize(Decimal("0.000001"))) if peak else 0.0))
    return points


def build(store: ProvingStore | None = None, *, config: ProvingConfig | None = None, clerk=None, now: datetime | None = None) -> ProvingRun:
    """The Track Record over the saved sessions and the real captures (see the module text). Cached in memory while no capture file and no usable bar day has changed."""
    began = time.perf_counter()
    config = config or load_config()
    store = store or recs.default_store_for_reading()
    found = recs.captures(store)
    seen_requests, unique = set(), []
    for capture in found:
        request_id = capture.payload["order"]["request_id"]
        if request_id not in seen_requests:      # the same run saved twice (a fixture and a saved session): the saved session counts
            seen_requests.add(request_id)
            unique.append(capture)
    last = bars_door.usable_last_day(now)
    stamp = (str(store.root), tuple((c.source, c.name, c.mtime) for c in unique), last)
    if _CACHE["stamp"] == stamp:
        return _CACHE["run"]
    parsed = []
    for capture in unique:
        order = recs.order_of(capture)
        day = recs.decision_day_of(order)
        found_recs = recs.recs_of(order, capture.payload.get("intent"))
        if day is not None and found_recs:
            parsed.append((capture, day, found_recs, order))
    symbols = sorted({r.symbol for _, _, rs, _ in parsed for r in rs} | {config.benchmark} | {s for _, _, rs, order in parsed for r in rs for s in [_sector_of(order, r)] if s})
    first = min((day for _, day, _, _ in parsed), default=last)
    book = bars_door.read_bars(symbols, first - timedelta(days=LOOKBACK_DAYS), last, clerk=clerk)
    trader_config = load_trader_config()
    all_outcomes: list[RecOutcome] = []
    rows: list[DecisionDay] = []
    sources: dict[str, str] = {}
    captured: list[dict] = []
    keys: set[str] = set()
    for capture, day, _, _ in parsed:
        scored, row = score_capture(capture, book, config, trader_config, last)
        fresh = [o.model_copy(update={"source": capture.source}) for o in scored if o.key not in keys]      # a record id held by two captures is scored once; each outcome says where its recommendation came from (`coverage["sources"]` keeps saying it too)
        keys.update(o.key for o in fresh)
        all_outcomes.extend(fresh)
        sources.update({o.key: capture.source for o in fresh})
        if row is not None:
            rows.append(row)
            captured.append({"source": capture.source, "name": capture.name, "decision_day": day.isoformat(), "request": capture.payload.get("request"), "outcomes": len(fresh)})
    all_outcomes.sort(key=lambda o: (o.decision_day, o.key))
    rows.sort(key=lambda r: (r.day, r.reason or ""))
    days = sorted({b["date"] for b in book.get(config.benchmark, [])})
    traded = [o for o in all_outcomes if o.traded and o.entry is not None and o.path]
    curve = _index_curve(traded, days)
    summary = stats_mod.compute(all_outcomes, [], config)
    created = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    run = ProvingRun(run_id=f"track-{created.replace('-', '').replace(':', '').replace('T', '-')[:13]}", kind="track", status="finished", created_utc=created, seconds=round(time.perf_counter() - began, 3), understood=f"{len(unique)} saved runs, {len(all_outcomes)} recommendations scored",
                     range=None if not rows else RunRange(first_day=rows[0].day, last_day=rows[-1].day, last_bar_day=last), decision_days=rows, equity=curve, events=[], outcomes=all_outcomes, stats=summary, caveats=caveats_of(all_outcomes, config),
                     coverage={"sources": sources, "captures": captured, "last_bar_day": last.isoformat(), "index": "$1 in every traded recommendation, 100 at the first entry; benchmark: the same dollars in SPY at the open of each entry day"},
                     config_versions={"proving": config.config_version, "trader": getattr(trader_config, "config_version", None)})
    _CACHE.update(stamp=stamp, run=run)
    return run


def report(run: ProvingRun) -> str:
    """A run as text: the headline, the counts, each outcome in a line, the caveats."""
    s = run.stats
    lines = [f"{run.kind.upper()}  {run.run_id}  ({run.understood})", f"  {s.headline.sentence}", f"  {s.critic_value.sentence}"]
    if run.range is not None:
        lines.append(f"  decision days {run.range.first_day} to {run.range.last_day}; last usable bar {run.range.last_bar_day}")
    o = s.outcomes
    lines.append(f"  outcomes n={o.n}: won {o.won}, lost {o.lost}, open {o.open}, not filled {o.not_filled}, no data {o.no_data}" + ("" if o.win_rate is None else f"; win rate {o.win_rate:.0%} of {o.closed} closed") + ("  [small sample]" if o.small_sample else ""))
    sources = run.coverage.get("sources", {})
    for x in run.outcomes:
        how = "REAL fill" if x.traded else f"hypothetical ({x.not_traded_reason})"
        ret = "" if x.return_pct is None else f"  return {x.return_pct:+.2%}" + ("" if x.excess_return_pct is None else f" vs SPY {x.excess_return_pct:+.2%}")
        entry = "" if x.entry is None else f"  in {x.entry.date} @ {x.entry.price:,.4f}"
        lines.append(f"  {x.decision_day} {x.symbol:<5} {x.verdict:<10} {x.result:<10} {how}{entry}{ret}  [{sources.get(x.key, '?')}]" + ("" if not x.notes else "  note: " + "; ".join(x.notes)))
    if run.equity:
        first, final = run.equity[0], run.equity[-1]
        lines.append((f"  index of $1 in every traded recommendation: {first.date} {first.equity} -> {final.date} {final.equity}; SPY {first.benchmark} -> {final.benchmark}" if run.kind == "track" else
                      f"  equity: {first.date} ${first.equity:,.2f} -> {final.date} ${final.equity:,.2f} (cash ${final.cash:,.2f}); SPY ${first.benchmark:,.2f} -> ${final.benchmark:,.2f}"))
    lines.extend(f"  caveat: {sentence}" for sentence in record.caveat_sentences(run.caveats))
    return "\n".join(lines)
