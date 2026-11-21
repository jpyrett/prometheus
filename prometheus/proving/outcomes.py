"""One recommendation against the stored bars: PURE (design/proving_ground_spec.md section 5, P4, P5, P8 and P10).

    plan = record.plan_of(recommendation)                                   # the Advisor's plan as a `PlanView`
    outcome = score(plan, bars, benchmark_bars, sector_bars, entry_after=date(2026, 11, 13), horizons=[5, 10, 20], symbol="XOM", identity=record.identity_of(order, recommendation, date(2026, 11, 13)))
    outcome.result                     # won | lost | open | not_filled | no_data
    outcome.entry, outcome.exit        # entry: the next stored open after the decision day; exit: the plan's own stop, target or time stop
    outcome.path                       # the day-by-day close, the return since the entry and the benchmark's, from the entry day to the exit or the last bar

THE ENTRY (P4). The decision day is `entry_after`; the entry is the first stored open AFTER it, found the way the Trader's backtest finds it (`simulator.entry_date`: the first trading day after the decision, then the earliest stored bar
on or after it among the symbol's and the benchmark's), and the symbol must have its bar ON that day: no bar, no fill, and a price is never carried from another day or interpolated (`no_data`, with the reason). The order is what the
Trader's ticket would be for the plan alone (`tickets.limit_range` from the plan's entry mid and zone and the Trader's own ratios, `tickets.default_order`: a limit at the plan's limit price, or a market order when it has none), and it
fills or not by `simulator.fill_backtest`, slippage as the Trader's settings say (0): a limit the day never reached is `not_filled` with the Trader's own code (`limit_not_reached`, `price_outside_limits`). Nothing here rewrites a fill rule.
A REAL FILL (`fill=`: the Track Record's recommendations the desk really bought, and the Time Machine's) is the entry as it was: its date, price and shares, and the position is followed from the bar of that date, where, as `follow` says, only
the stop can fire (the order of prices inside a day is not known, so the bad case is assumed). Without a fill the outcome is HYPOTHETICAL (P8): ONE share-equivalent, a `gain` of None, never in an equity curve.

THE EXIT (P5). `simulator.follow` is called as it is: the stop before the target on a day that touches both, only the stop on the entry bar, the time stop at the first bar on or after its date, marked at the last stored close when no rule
fired (`open`; never a win). The day-by-day path is worked out separately from the same bars and CHECKED against `follow` (the session and the date of the exit, or of the last bar): a disagreement raises AssertionError, so there are not two
exit rules. `result` is `won` only for a close ABOVE the entry price (a close at the entry price is `lost`, with a note: a gain of zero is not a win), `lost` below it. A return is `follow`'s (the exit price over the entry price); the
benchmark's and the sector's run from the open of the entry bar to the close of the exit day (or the last bar), the way `follow` does it (`simulator._benchmark`: these three private helpers of the simulator, `_fraction`, `_benchmark` and
`_return_difference`, are used so that a return is worked out in ONE place).
HORIZONS: the position's return at each of 5, 10 and 20 sessions after the entry bar BY THE PLAN: a position that closed before then keeps its realized return (the benchmark read to the day it closed), one still open is marked at that session's
close, and a horizon not yet reached while the position is open is None. `max_favorable_pct` and `max_adverse_pct` are the best and the worst CLOSE since the entry, against the entry price, and the price a closed position was sold at (so a target never reads as a gain larger than the best the position ever showed).

SPLITS (P10). The stored bars are unadjusted, and the plan was made on the decision day, so EVERYTHING is read in the share basis of the DECISION DAY: with a split table (`splits=`) every bar after it is put into that basis (`splits.to_basis`: its
prices multiplied by the factor of the splits since the decision day) before the entry and `follow` read it, so the plan's prices, the entry, the stop and the path all mean what they meant when the plan was made (a position entered ON the split
day, from a decision the day before, included), and the outcome says so (`split_adjusted`, a note; the entry, the exit and the path are in that basis, and a real fill is brought into it). With NO table (`splits=None`: no file, or one that cannot be
read) a close-to-open move of more than `split_suspect_pct` (40%) from the decision day to the end of the position makes it `no_data`, with the move and the day in the note: a split would otherwise read as a loss and hit the stop. The benchmark's
and the sector's bars get the same suspect rule (their returns are then None, with a note).

PRICES ARE `Decimal` (entry 112): a stored bar's floats cross the price door inside the simulator once; the money (`gain`, the entry's `amount`) is `trade_amount`. A return is a measurement and stays a float, to six places, from Decimal prices.
No model, no network, no clock: the same bars give the same outcome.
"""
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_DOWN, Decimal
from zoneinfo import ZoneInfo

from prometheus.agents.trader import simulator, tickets
from prometheus.agents.trader.config import TraderConfig, load_config as load_trader_config
from prometheus.common import shares as sh
from prometheus.common.market_calendar import load_calendar
from prometheus.common.money import price_scale
from prometheus.common.portfolio import trade_amount
from prometheus.proving import splits as splits_mod
from prometheus.proving.config import ProvingConfig, load_config
from prometheus.proving.record import EntryView, ExitView, Horizon, Path, PlanView, RecOutcome


@dataclass(frozen=True)
class EntryFill:
    """A real fill of the desk (the Trader's): the day it filled, the price and the shares. Never recomputed."""
    date: date
    price: Decimal
    shares: Decimal


def entry_fill_of(trade) -> EntryFill | None:
    """The `EntryFill` of a filled purchase (a `Trade` or its dict, as the Trader wrote it on the work order), or None for anything else (a sale, a trade that did not fill)."""
    from prometheus.agents.trader.record import Trade
    found = trade if isinstance(trade, Trade) else Trade.model_validate(trade)
    if found.status != "filled" or found.side != "buy" or found.fill is None:
        return None
    return EntryFill(date.fromisoformat(found.fill.date), found.fill.price, found.fill.shares)


def _close_on_or_before(bars: list[dict], day: date) -> Decimal | None:
    """The close of the last stored bar on or before `day`, through the price door; None when there is none."""
    found = [b for b in bars if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] <= day.isoformat() and simulator.bar_price(b, "close") is not None]
    return simulator.bar_price(max(found, key=lambda b: b["date"]), "close") if found else None


def _ratio_text(split) -> str:
    return f"{split.split_to.normalize():f}-for-{split.split_from.normalize():f}"


def _suspect_note(symbol: str, moved: tuple[date, Decimal], table_missing: str) -> str:
    return f"{symbol} moved {moved[1] * 100:+.1f}% from one close to the next open on {moved[0].isoformat()}, which looks like a split, and {table_missing}"


def score(plan: PlanView, bars: list[dict], benchmark_bars: list[dict] | None, sector_bars: list[dict] | None, entry_after: date, horizons: list[int], *, symbol: str, identity: dict | None = None,
          fill: EntryFill | None = None, config: ProvingConfig | None = None, trader_config: TraderConfig | None = None, splits: splits_mod.SplitTable | None = None, table_missing: str = "") -> RecOutcome:
    """One recommendation scored against `bars` (the symbol's stored daily bars, oldest first, as `DataClerk.request_bars` gives them) and the benchmark's and the sector's over the same dates (None when they are not stored). `entry_after` is the
    decision day (the entry is the first stored open after it); `horizons` the sessions at which the return is read; `symbol` the ticker. `identity` is the dict of `record.identity_of` (key, company, record_id, advisor, critic, verdict ...: what
    the recommendation was; the caller may add `not_traded_reason`); without it the outcome is keyed by the day and the symbol. `fill` is the desk's real fill (else the outcome is hypothetical). `config` and `trader_config` default to the files.
    `splits` is the symbol's split table, and `table_missing` the reason it has none (for the note). Prices, shares and the path are in the share basis of the decision day (see SPLITS). Returns a `RecOutcome`: `no_data` when nothing can be priced (no bar, the future is not stored yet, a suspected split with no table), `not_filled`
    with the Trader's reason code when the entry order cannot fill, else `won`, `lost` or `open`. Raises AssertionError when the day-by-day path disagrees with `simulator.follow`."""
    config, tcfg = config or load_config(), trader_config or load_trader_config()
    who = {k: v for k, v in (identity or {}).items() if k not in ("traded", "hypothetical", "plan", "result")}
    who.setdefault("symbol", symbol)
    who.setdefault("decision_day", entry_after)
    who.setdefault("key", f"{entry_after.isoformat()}:{symbol}")
    traded = fill is not None
    if traded:
        who["not_traded_reason"] = None

    def finish(result: str, **more) -> RecOutcome:
        return RecOutcome(**who, traded=traded, hypothetical=not traded, plan=plan, result=result, **more)

    if not bars:
        return finish("no_data", notes=[f"no daily bar of {symbol} is stored"])

    # ---- the entry, in the share basis of the decision day (the plan's own)
    based = splits_mod.to_basis(bars, splits, entry_after)
    if fill is None:
        now = datetime.combine(entry_after, time(16, 0), tzinfo=ZoneInfo(load_calendar().timezone))
        entry_day = simulator.entry_date(now, {"symbol": based, "benchmark": benchmark_bars or []})
        if entry_day is None:
            return finish("no_data", notes=[f"no daily bar is stored after {entry_after.isoformat()} yet: the future is not stored"])
        entry_bar = simulator.bar_on(based, entry_day)
        if entry_bar is None:
            return finish("no_data", notes=[f"{symbol} has no stored bar on {entry_day.isoformat()}, the next session: no price is carried from another day"])
        ref = plan.entry_mid or _close_on_or_before(based, entry_after)
        if ref is None:
            return finish("no_data", notes=[f"{symbol} has no price at or before {entry_after.isoformat()} to measure the entry by"])
        zone = [plan.entry_low, plan.entry_high] if plan.entry_low is not None and plan.entry_high is not None else None
        limits = tickets.limit_range("buy", ref, zone, tcfg)
        if limits is None:
            return finish("not_filled", reject_code="price_outside_limits", notes=[f"the plan's reference price {ref:,.4f} is too far from its entry zone for any price to be paid"])
        kind, limit_price = tickets.default_order("buy", plan.limit_price, ref, *limits, tcfg)
        execution = simulator.fill_backtest("buy", kind, limit_price, entry_bar, limits[0], limits[1], tcfg.costs.slippage_bps)
        if execution.status != "filled":
            return finish("not_filled", reject_code=execution.reject_code, notes=[execution.reason])
        entry_day, fill_price, shares = entry_day, execution.price, Decimal(1)
    else:
        factor = splits.factor_between(entry_after, fill.date) if splits is not None else Decimal(1)      # a real fill after a split is brought into the basis of the decision day: the same money
        entry_day, fill_price, shares = fill.date, price_scale(fill.price * factor), sh.quantity((fill.shares / factor).quantize(sh.STEP, rounding=ROUND_DOWN))
    entry = entry_day.isoformat()

    # ---- the exit, by the plan: `follow` as it is, on the bars in the share basis of the decision day
    series = [b for b in based if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] >= entry and all(simulator.bar_price(b, k) is not None for k in ("open", "high", "low", "close"))]
    if not series or series[0]["date"] != entry:
        return finish("no_data", notes=[f"{symbol} has no stored bar on {entry}, the day of the entry: it cannot be followed"])
    outcome = simulator.follow(fill_price, shares, entry_day, based, plan.stop, plan.target, plan.time_stop_date.isoformat() if plan.time_stop_date else None, tcfg.costs.slippage_bps)
    closed = outcome.status == "closed"
    end = outcome.exit_date if closed else outcome.last_date
    end_day = date.fromisoformat(end)
    notes: list[str] = []

    if splits is None:
        moved = splits_mod.suspect_move(bars, entry_after, end_day, config.split_suspect_pct)
        if moved is not None:
            return finish("no_data", notes=[_suspect_note(symbol, moved, table_missing or f"no split table is stored for {symbol}") + ": it is not scored across it"])
    in_window = [s for s in splits.splits if entry_after < s.execution_date <= end_day] if splits is not None else []
    notes += [f"adjusted for the {_ratio_text(s)} split of {s.execution_date.isoformat()}" for s in in_window]

    bench = benchmark_bars
    if bench is not None and (moved := splits_mod.suspect_move(bench, entry_after, end_day, config.split_suspect_pct)) is not None:
        bench = None
        notes.append(_suspect_note("the benchmark", moved, "its split cannot be told from a market move") + ": its return is left out")
    sector = sector_bars
    if sector is not None and (moved := splits_mod.suspect_move(sector, entry_after, end_day, config.split_suspect_pct)) is not None:
        sector = None
        notes.append(_suspect_note("the sector fund", moved, "its split cannot be told from a market move") + ": its return is left out")

    # ---- the path, checked against `follow`
    held = outcome.sessions_held
    taken = series[:held + 1]
    if len(taken) != held + 1 or taken[-1]["date"] != end:
        raise AssertionError(f"{symbol}: the day-by-day path ends on {taken[-1]['date'] if taken else None} after {len(taken) - 1} sessions but `follow` says {end} after {held}: there must be one exit rule")
    bench_by_day = {b["date"]: b for b in bench or [] if isinstance(b, dict) and isinstance(b.get("date"), str)}
    bench_open = simulator.bar_price(bench_by_day.get(entry), "open")
    path = []
    for bar in taken:
        close, other = simulator.bar_price(bar, "close"), bench_by_day.get(bar["date"])
        against = simulator.bar_price(other, "close") if bench_open is not None and other is not None else None
        path.append(Path(date=date.fromisoformat(bar["date"]), close=close, return_pct=simulator._fraction(close, fill_price), benchmark_return_pct=None if against is None else simulator._fraction(against, bench_open)))

    # ---- the horizons, by the plan
    marks = []
    for sessions in horizons:
        if closed and held <= sessions:
            marks.append(Horizon(sessions=sessions, return_pct=outcome.return_pct, benchmark_return_pct=simulator._benchmark(bench, entry, end)))
        elif sessions < len(series):
            marks.append(Horizon(sessions=sessions, return_pct=simulator._fraction(simulator.bar_price(series[sessions], "close"), fill_price), benchmark_return_pct=simulator._benchmark(bench, entry, series[sessions]["date"])))
        else:
            marks.append(Horizon(sessions=sessions))

    bench_return = simulator._benchmark(bench, entry, end)
    excess = None if bench_return is None else simulator._return_difference(outcome.return_pct, bench_return)
    if closed:
        won = outcome.exit_price > fill_price
        if outcome.exit_price == fill_price:
            notes.append("it closed at its entry price: a gain of zero is not a win")
    result = ("won" if won else "lost") if closed else "open"
    extremes = [p.return_pct for p in path] + ([outcome.return_pct] if closed else [])      # the closes on the way, and the price the position was sold at
    gain = None if not traded else (outcome.realized_gain if closed else outcome.unrealized_gain)
    return finish(result, entry=EntryView(date=entry_day, price=fill_price, shares=shares, amount=trade_amount(shares, fill_price)),
                  exit=ExitView(date=end_day, price=outcome.exit_price, reason=outcome.exit_reason, sessions_held=held) if closed else None,
                  return_pct=outcome.return_pct, benchmark_return_pct=bench_return, excess_return_pct=excess, sector_return_pct=simulator._benchmark(sector, entry, end), gain=gain, horizons=marks,
                  max_favorable_pct=max(extremes), max_adverse_pct=min(extremes), path=path, split_adjusted=bool(in_window), notes=notes)
