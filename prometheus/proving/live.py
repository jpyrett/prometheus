"""The live monitor: the paper account against live quotes and against the plan of each recommendation that bought it (design/proving_ground_spec.md section 11, P11; design choices entry 126).

    snapshot = snapshot(manager, account_id)         # `manager` is a connected `AccountManager`; one `LiveSnapshot`
    LiveMonitor(manager_for).current(account_id)    # the same, polled: outside the session it serves the last snapshot and says the market is closed

P11: THE LIVE MONITOR WATCHES. It reads the book (`AccountManager.list_portfolio` and `history`), a quote for each position through the Account Manager's own door (`request_prices`: Finnhub, or the stored close when the clock
is simulated, entry 76) and the stored bars (`request_bars`); it compares and it tells. It never sells, never places, never writes a trade or a decision, and nothing here can: this module calls no door that writes (a test reads its
source for them). A stop or a target that was hit is an ALERT for the user; entry 105 is unchanged (only the user's own order or RaiseFunds sells).

A POSITION'S PLAN is the plan of the recommendation that bought it: the book's `trading.trades.record_id` of the position's purchase is looked up in the saved sessions and the real captures (`recs.record_index`) and the
recommendation's own stop, target and time stop are read (`record.plan_of`). A position bought by a direct order has no plan and shows none: it is `no_plan`, its price against its cost only. A position bought as a NEUTRAL company (`neutral-<request>-<n>`: the Advisor scored it neutral) has the
Advisor's EXIT plan since Advisor config version 2 (a stop and a time stop, no target: `recs.neutral_record_of` finds it on the saved session's candidates) and is judged by it like any other, with a note that says so; one bought before that has no plan and
is `no_plan`, its note says the Advisor made none. Nothing invents a stop.
THE STATUS of a position with a plan, judged in time order: for each stored bar AFTER the purchase day its low against the stop and then its high against the target (a day that touches both is a stop, as `simulator.follow` says; the
bar of the purchase day itself is not read: part of it happened before the purchase), the time stop's date (a bar on or after it) and last the QUOTE NOW (the stop before the target). `stop_hit`, `target_hit`, `time_stop_passed` or
`in_plan`, and `evidence` says which: "the quote now" or "the low of 2026-11-17". A symbol whose stored bars show a close-to-open move past `split_suspect_pct` since the purchase is judged on the quote alone (a split in the
bars would read as a stop), and says so.
THE EQUITY HISTORY is rebuilt from the book's own dated trades and cash ledger and the stored closes: for each stored trading day from the account's first day, the cash after the day's last ledger entry plus the shares held
at that day's close at the close on or before it (a day with no stored close for a held symbol keeps the previous one and says `stale`); the live point is the snapshot's own equity at its moment. TODAY is never in the history: a bar stored today may be of a session
that is still running, so it is not a close; the live point is today. An account that was opened today has an empty history and the live point alone.
Money is `Decimal` (entry 112): the value, the gain and the equity are `trade_amount` of the exact figures; a percentage and a day's change are measurements (floats). No model.
"""
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from prometheus.agents.trader import simulator
from prometheus.common import trading_clock
from prometheus.common.market_calendar import load_calendar, session_at
from prometheus.common.money import EXACT, Money, Price, price as price_door
from prometheus.common.portfolio import trade_amount
from prometheus.common.shares import PositiveShares
from prometheus.proving import recs, splits
from prometheus.proving.config import load_config
from prometheus.proving.record import Measure, plan_of
from prometheus.proving.store import ProvingStore

STATUSES = ("in_plan", "stop_hit", "target_hit", "time_stop_passed", "no_plan")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LivePlan(_Model):
    stop: Price | None = None
    target: Price | None = None
    time_stop_date: date | None = None


class LivePosition(_Model):
    symbol: str
    name: str | None = None
    shares: PositiveShares
    cost_basis: Price
    price: Price | None = None                  # the quote now (None: no quote could be had)
    price_time: str | None = None               # when the quote was made (UTC), as the provider says
    simulated: bool = False                     # the quote is a stored close (entry 76), not a live price
    value: Money | None = None
    gain: Money | None = None
    gain_pct: Measure | None = None             # a fraction (0.0123 is 1.23%)
    day_change_pct: Measure | None = None       # today's change, a fraction
    plan: LivePlan | None = None
    status: str = "no_plan"                     # in_plan | stop_hit | target_hit | time_stop_passed | no_plan
    evidence: str | None = None                 # what the status rests on: "the quote now", "the low of 2026-11-17", ...
    since: date | None = None                   # the day the position was bought (the latest purchase)
    record_id: str | None = None
    notes: list[str] = Field(default_factory=list)


class Alert(_Model):
    symbol: str
    kind: str                                   # stop_hit | target_hit | time_stop_passed
    sentence: str


class SessionView(_Model):
    state: str                                  # pre_market | regular | after_hours | closed
    open: bool
    reason: str = ""
    next_open_utc: str | None = None


class HistoryPoint(_Model):
    date: date
    equity: Money
    cash: Money
    invested: Money
    stale: list[str] = Field(default_factory=list)


class LiveSnapshot(_Model):
    account_id: int
    name: str | None = None
    positions: list[LivePosition]
    cash: Money
    invested_at_cost: Money
    market_value: Money                         # the positions at the quote now (a position with no quote is at its cost: it is listed in `notes`)
    equity: Money                               # cash + market_value
    gain: Money                                 # market value less cost, over the positions
    gain_pct: Measure | None = None
    as_of_utc: str
    alerts: list[Alert] = Field(default_factory=list)
    session: SessionView
    history: list[HistoryPoint] = Field(default_factory=list)       # the account's equity from its first day (book + stored closes); the live point is the snapshot itself
    last_bar_day: date | None = None
    clock_note: str | None = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- small readers

def _ratio(numerator: Decimal, denominator: Decimal) -> Measure | None:
    """A return as a fraction, to six places: a measurement from Decimal figures (P7)."""
    return None if denominator == 0 else float((EXACT.divide(numerator, denominator) - 1).quantize(Decimal("0.000001")))


def _quote_price(quote) -> Decimal | None:
    try:
        return price_door(quote.get("price")) if isinstance(quote, dict) else None
    except ValueError:
        return None


def _percent_to_fraction(value) -> Measure | None:
    """Finnhub's `change_percent` is in percent (1.23 is 1.23%): a fraction here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float((Decimal(str(value)) / 100).quantize(Decimal("0.000001")))


def _day(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def today_of(now: datetime) -> date:
    """The market day of a moment: the bound day or the simulated clock's day when the clock is not the real one, else its Eastern date."""
    return trading_clock.market_day(now)


# ---------------------------------------------------------------- the plan of each position

def _purchases(history: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for trade in history.get("trades", []):
        if trade.get("side") == "buy":
            out.setdefault(trade["symbol"], []).append(trade)
    return out


def _plan_for(symbol: str, buys: list[dict], index: dict) -> tuple[LivePlan | None, str | None, date | None, list[str]]:
    """(the plan, the record id, the day bought, notes) for a position: the plan of the recommendation of the LATEST purchase that has one."""
    notes: list[str] = []
    since = max((b["traded_on"] for b in buys), default=None)
    for buy in sorted(buys, key=lambda b: (b["traded_on"], b.get("trade_id", 0)), reverse=True):
        record_id = buy.get("record_id")
        if isinstance(record_id, str) and record_id in index:
            found = plan_of(index[record_id][0])
            if found.stop is None and found.target is None and found.time_stop_date is None:
                notes.append("the recommendation that bought it has no stop, target or time stop")
                return None, record_id, buy["traded_on"], notes
            if record_id.startswith("neutral-"):
                notes.append("bought as a company the Advisor scored neutral: it has a stop and a time stop, no target")
            return LivePlan(stop=found.stop, target=found.target, time_stop_date=found.time_stop_date), record_id, buy["traded_on"], notes
    last = max(buys, key=lambda b: (b["traded_on"], b.get("trade_id", 0)), default=None)
    record_id = last.get("record_id") if last else None
    if isinstance(record_id, str) and record_id.startswith("neutral-"):
        notes.append("bought as a company the Advisor scored neutral: it made no plan (no stop or target) for it")
    elif isinstance(record_id, str) and record_id.startswith("direct-"):
        notes.append("bought by your own order: there is no recommendation and no plan")
    elif isinstance(record_id, str):
        notes.append("the recommendation that bought it is not among the saved runs, so its plan is not known")
    else:
        notes.append("no recommendation is recorded for this purchase: there is no plan")
    return None, record_id if isinstance(record_id, str) else None, since, notes


def _judge(symbol: str, plan: LivePlan, price: Decimal | None, bars: list[dict], since: date | None, today: date, suspect_pct: Decimal) -> tuple[str, str | None, list[str]]:
    """(status, evidence, notes) of a position with a plan: time order, stop before target (see the module text)."""
    notes: list[str] = []
    after = sorted((b for b in bars if isinstance(b, dict) and isinstance(b.get("date"), str) and since is not None and b["date"] > since.isoformat()), key=lambda b: b["date"])
    use_bars = bool(after)
    if after and since is not None and splits.suspect_move(bars, since, _day(after[-1]["date"]) or since, suspect_pct) is not None:
        notes.append("the stored bars show a one-night move past the split limit since the purchase: the stop and the target are judged on the quote alone")
        use_bars = False
    if use_bars:
        for bar in after:
            when = bar["date"]
            if plan.time_stop_date is not None and _day(when) is not None and _day(when) >= plan.time_stop_date:
                return "time_stop_passed", f"the bar of {when}", notes
            low, high = simulator.bar_price(bar, "low"), simulator.bar_price(bar, "high")
            if plan.stop is not None and low is not None and low <= plan.stop:
                return "stop_hit", f"the low of {when}", notes
            if plan.target is not None and high is not None and high >= plan.target:
                return "target_hit", f"the high of {when}", notes
    if plan.time_stop_date is not None and today >= plan.time_stop_date:
        return "time_stop_passed", f"today is {today.isoformat()}", notes
    if price is not None:
        if plan.stop is not None and price <= plan.stop:
            return "stop_hit", "the quote now", notes
        if plan.target is not None and price >= plan.target:
            return "target_hit", "the quote now", notes
    return "in_plan", None, notes


def _sentence(position: LivePosition) -> str:
    plan, price = position.plan, position.price
    where = "" if position.evidence is None else f" ({position.evidence})"
    if position.status == "stop_hit":
        return f"{position.symbol} hit its stop at {plan.stop:,.4f}{where}: the price now is {price:,.4f}. Nothing was sold; selling is your decision."
    if position.status == "target_hit":
        return f"{position.symbol} reached its target of {plan.target:,.4f}{where}: the price now is {price:,.4f}. Nothing was sold; selling is your decision."
    return f"{position.symbol}'s time stop of {plan.time_stop_date.isoformat()} has passed{where}. Nothing was sold; selling is your decision."


# ---------------------------------------------------------------- the equity history

def _closes(bars: dict[str, list[dict]]) -> dict[str, list[tuple[str, Decimal]]]:
    out = {}
    for symbol, series in bars.items():
        rows = [(b["date"], simulator.bar_price(b, "close")) for b in series if isinstance(b, dict) and isinstance(b.get("date"), str)]
        out[symbol] = sorted((d, c) for d, c in rows if c is not None)
    return out


def equity_history(history: dict, bars: dict[str, list[dict]], days: list[str]) -> list[HistoryPoint]:
    """The account's equity at the close of each of `days` (stored trading days, ascending) from its first day: the cash after the day's last ledger entry plus the shares held that day at the close on or before it (the
    previous close, marked `stale`, for a held symbol with no bar that day). A day before the account's first ledger entry is not in it."""
    ledger = sorted(history.get("cash", []), key=lambda r: (r["occurred_on"], r.get("entry_id", 0)))
    trades = sorted(history.get("trades", []), key=lambda t: (t["traded_on"], t.get("trade_id", 0)))
    if not ledger:
        return []
    first = ledger[0]["occurred_on"].isoformat()
    closes = _closes(bars)
    points: list[HistoryPoint] = []
    for day in days:
        if day < first:
            continue
        cash = Decimal(0)
        for row in ledger:
            if row["occurred_on"].isoformat() <= day:
                cash = row["balance_after"]
        held: dict[str, Decimal] = {}
        for trade in trades:
            if trade["traded_on"].isoformat() <= day:
                sign = 1 if trade["side"] == "buy" else -1
                held[trade["symbol"]] = held.get(trade["symbol"], Decimal(0)) + sign * Decimal(trade["shares"])
        value, stale, priced = Decimal(0), [], True
        for symbol, shares in held.items():
            if shares <= 0:
                continue
            rows = [(d, c) for d, c in closes.get(symbol, []) if d <= day]
            if not rows:
                priced = False
                break
            if rows[-1][0] != day:
                stale.append(symbol)
            value += trade_amount(shares, rows[-1][1])
        if not priced:
            continue
        points.append(HistoryPoint(date=date.fromisoformat(day), equity=cash + value, cash=cash, invested=value, stale=stale))
    return points


# ---------------------------------------------------------------- the snapshot

def snapshot(manager, account_id: int, *, now: datetime | None = None, store: ProvingStore | None = None) -> LiveSnapshot:
    """One live snapshot of the account `manager` is connected to (see the module text). Reads the book, asks the Data Clerk for the quotes and the stored bars, writes nothing. Raises what the book raises (a missing account)."""
    now = now or datetime.now(timezone.utc)
    config = load_config()
    portfolio = manager.list_portfolio()
    history = manager.history()
    index = recs.record_index(store or recs.default_store_for_reading())
    positions = list(portfolio.positions)
    symbols = [p.symbol for p in positions]
    today = today_of(now)
    notes: list[str] = []
    quotes: dict = {}
    if symbols:
        order = manager.request_prices(symbols)
        quotes = dict(order.data.quotes) if order.data is not None else {}
        notes.extend(f"no quote for {problem.split(':')[0]}" for problem in (order.data.problems if order.data is not None else order.errors))
    buys = _purchases(history)
    first_day = min([b["traded_on"] for b in history.get("trades", [])] + [r["occurred_on"] for r in history.get("cash", [])] or [today])
    wanted = sorted({*symbols, *(t["symbol"] for t in history.get("trades", [])), config.benchmark})
    bars: dict[str, list[dict]] = {}
    if wanted:
        got = manager.request_bars(wanted, first_day, today)
        bars = dict(got.data.bars) if got.data is not None else {}
    last_bar = max((b["date"] for series in bars.values() for b in series if isinstance(b, dict) and isinstance(b.get("date"), str)), default=None)
    live_positions, alerts = [], []
    cost_total = value_total = Decimal(0)
    for position in positions:
        quote = quotes.get(position.symbol)
        price = _quote_price(quote)
        plan, record_id, since, why = _plan_for(position.symbol, buys.get(position.symbol, []), index)
        cost = trade_amount(position.shares, position.cost_basis)
        value = trade_amount(position.shares, price) if price is not None else None
        live = LivePosition(symbol=position.symbol, name=position.name, shares=position.shares, cost_basis=position.cost_basis, price=price, price_time=(quote or {}).get("as_of_utc") if isinstance(quote, dict) else None,
                            simulated=bool(isinstance(quote, dict) and quote.get("simulated")), value=value, gain=None if value is None else value - cost, gain_pct=None if value is None else _ratio(value, cost),
                            day_change_pct=_percent_to_fraction((quote or {}).get("change_percent")) if isinstance(quote, dict) else None, plan=plan, since=since, record_id=record_id, notes=list(why))
        if plan is not None:
            live.status, live.evidence, judged = _judge(position.symbol, plan, price, bars.get(position.symbol, []), since, today, config.split_suspect_pct)
            live.notes.extend(judged)
            if live.status != "in_plan":
                alerts.append(Alert(symbol=position.symbol, kind=live.status, sentence=_sentence(live)))
        if price is None:
            live.notes.append("no quote could be had: the position is counted at its cost")
        cost_total += cost
        value_total += value if value is not None else cost
        live_positions.append(live)
    session = session_at(now, load_calendar())
    equity = portfolio.cash_avail + value_total
    days = sorted({b["date"] for b in bars.get(config.benchmark, []) if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] < today.isoformat()})      # TODAY is the live point: a bar stored today may be of a session still running, and is never read as a close
    return LiveSnapshot(account_id=account_id, name=portfolio.user, positions=live_positions, cash=portfolio.cash_avail, invested_at_cost=cost_total, market_value=value_total, equity=equity, gain=value_total - cost_total,
                        gain_pct=_ratio(value_total, cost_total) if cost_total else None, as_of_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), alerts=alerts,
                        session=SessionView(state=session.state, open=session.is_open, reason=session.reason, next_open_utc=session.next_open_utc.strftime("%Y-%m-%dT%H:%M:%SZ")), history=equity_history(history, bars, days),
                        last_bar_day=_day(last_bar), clock_note=trading_clock.clock_note(now), notes=notes)


class LiveMonitor:
    """Polls `snapshot` for the page: while the market is open (or `only_in_session` is off) it takes a fresh snapshot, otherwise it serves the last one it made and says the market is closed. `manager_for(account_id)` gives a
    connected `AccountManager`. It keeps nothing but the last snapshot of each account in memory."""

    def __init__(self, manager_for: Callable[[int], object], store: ProvingStore | None = None):
        self.manager_for, self.store, self.last = manager_for, store, {}

    def current(self, account_id: int, now: datetime | None = None) -> LiveSnapshot:
        now = now or datetime.now(timezone.utc)
        config = load_config()
        session = session_at(now, load_calendar())
        previous = self.last.get(account_id)
        if config.live.only_in_session and not session.is_open and previous is not None:
            return previous.model_copy(update={"session": SessionView(state=session.state, open=False, reason=session.reason, next_open_utc=session.next_open_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))})
        fresh = snapshot(self.manager_for(account_id), account_id, now=now, store=self.store)
        self.last[account_id] = fresh
        return fresh


# ---------------------------------------------------------------- the text of `live`

def report(shot: LiveSnapshot) -> str:
    """The snapshot as the command prints it: each position against its plan, the totals, the alerts."""
    lines = [f"LIVE  account {shot.account_id}{'' if shot.name is None else f' ({shot.name})'}   as of {shot.as_of_utc}   market: {shot.session.state}{'' if shot.session.open else f' ({shot.session.reason}; next open {shot.session.next_open_utc})'}"]
    if shot.clock_note:
        lines.append(shot.clock_note)
    for p in shot.positions:
        now = "no quote" if p.price is None else f"{p.price:,.4f} at {p.price_time}{' (stored close, simulated)' if p.simulated else ''}"
        plan = "no plan" if p.plan is None else f"stop {p.plan.stop if p.plan.stop is not None else '-'}  target {p.plan.target if p.plan.target is not None else '-'}  time stop {p.plan.time_stop_date or '-'}"
        gain = "" if p.gain is None else f"  value ${p.value:,.2f}  gain ${p.gain:+,.2f} ({p.gain_pct:+.2%})"
        change = "" if p.day_change_pct is None else f"  today {p.day_change_pct:+.2%}"
        lines.append(f"  {p.symbol:<6} {p.shares} sh  cost {p.cost_basis:,.4f}  price {now}{gain}{change}")
        lines.append(f"         bought {p.since}  [{p.status}{'' if p.evidence is None else ' by ' + p.evidence}]  {plan}")
        lines.extend(f"         note: {note}" for note in p.notes)
    lines.append(f"  cash ${shot.cash:,.2f}  invested at cost ${shot.invested_at_cost:,.2f}  market value ${shot.market_value:,.2f}  equity ${shot.equity:,.2f}  gain ${shot.gain:+,.2f}" + ("" if shot.gain_pct is None else f" ({shot.gain_pct:+.2%})"))
    lines.extend(f"  ALERT {a.sentence}" for a in shot.alerts)
    lines.extend(f"  note: {n}" for n in shot.notes)
    lines.append(f"  equity history: {len(shot.history)} stored day(s)" + ("" if not shot.history else f", {shot.history[0].date} ${shot.history[0].equity:,.2f} to {shot.history[-1].date} ${shot.history[-1].equity:,.2f}") + f"; last stored bar {shot.last_bar_day}")
    return "\n".join(lines)
