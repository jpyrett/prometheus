"""The Trader's fill rules and the exit rules, as PURE functions (Trader v1 spec sections 6 and 10).

    execution = fill_paper("buy", "limit", 167.1, quote=167.0, limit_low=165.46, limit_high=168.74, slippage_bps=5)        # filled at 167.0835
    execution = fill_backtest("buy", "market", None, bar, limit_low=165.46, limit_high=168.74, slippage_bps=5)             # the entry bar's open, plus slippage
    day = entry_date(now, bars)                                                                                            # the one entry date of a backtest run
    outcome = follow(fill_price=167.18, shares=44, entry_day="2026-11-06", bars=bars, stop_price=157.23, target_price=168.38, time_stop_date="2026-12-31", slippage_bps=5)

No I/O, no clock, no model: every function reads only what it is given (the bar dicts, the numbers of the order, the moment `now`), so a test can walk each rule to its boundary and a later backtest harness can call
them for a date in the past. The money rules of the broker (the cash, the shares, the account) are NOT here: `broker.py` applies these rules to one trade and keeps the portfolio.

A bar is {"date": "YYYY-MM-DD", "open", "high", "low", "close", "volume"} (the Data Clerk's `request_bars`). `limit_low` and `limit_high` are the range the Trader's code computed for a trade (`tickets.limit_range`);
NO ORDER OF ANY TYPE FILLS OUTSIDE IT. What "market" means here: a market order fills at the market price only when that price is inside the limits; above `limit_high` (a buy) or below `limit_low` (a sale) it is rejected
`price_outside_limits`, because the price has left what the Critic judged and a new review is needed.

FILL RULES. `slip = slippage_bps / 10000`; a fill price is rounded to 4 places. For a BUY (a sale is the mirror image: slippage is taken OFF the price, "above" becomes "below" and `limit_high` becomes `limit_low`):

    paper, quote q       market  p = q (1 + slip); p above limit_high: rejected price_outside_limits, else filled at p
                         limit L q <= L: filled at min(L, q (1 + slip)); else ACTIVE (a resting order: the price is above the limit)
    backtest, entry bar  market  p = open (1 + slip); p above limit_high: rejected, else filled at p on the bar's date
                         limit L open <= L: filled at min(L, open (1 + slip)); else low <= L: filled at L; else rejected limit_not_reached (a day order, and the day is over)

THE ENTRY DATE of a backtest run (`entry_date`) is found in two steps, so that the future never leaks into a decision: the first trading day whose regular open (09:30 Eastern) is strictly after `now` (the calendar alone),
then the earliest stored bar on or after that day among the bars of every symbol asked for (so a day whose file is missing moves the whole run to the next stored day).

EXITS (`follow`): a filled purchase is followed through the bars after its entry by the Advisor's own plan (stop, target, time stop). On the entry bar only the stop can fire: the order of prices inside one day is not
known, so the bad case is assumed. On each later bar the first rule that holds wins, and the stop is checked before the target, so a day that touches both is a stop. It is a SIMULATION: it is never booked.
"""
import math
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from prometheus.agents.trader.record import Outcome
from prometheus.common.market_calendar import eastern_date, is_trading_day, load_calendar, next_trading_day
from prometheus.common.portfolio import trade_amount
from prometheus.common.work_order import iso_date

PRICE_PLACES = 4        # fill prices, and the prices a limit range is made of
RETURN_PLACES = 6       # a return is a fraction (0.0077 is 0.77%), as every `_pct` of the desk
SIDES = ("buy", "sell")
ORDER_TYPES = ("market", "limit")


@dataclass(frozen=True)
class Execution:
    """What the fill rules say about one order against one price: `filled` (at `price`, from `price_source`, with the slippage that was applied), `active` (a resting order) or `rejected` (with a code of the
    Trader's closed list). `reason` is code's own sentence."""
    status: str                          # filled | active | rejected
    reason: str = ""
    price: float | None = None
    price_source: str | None = None      # quote | bar_open | bar_limit
    slippage_bps: float = 0.0
    reject_code: str | None = None


def usable(value) -> bool:
    """A price or a quantity that can be traded on: a real number, finite, above zero (not a bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def r4(value: float) -> float:
    return round(value + 0.0, PRICE_PLACES)


def _rejected(code: str, reason: str) -> Execution:
    return Execution("rejected", reason, reject_code=code)


def _bps(price: float, base: float, side: str) -> float:
    """The slippage a fill really had against its base price, in basis points: what was added to a purchase or taken off a sale (never negative)."""
    moved = price / base - 1 if side == "buy" else 1 - price / base
    return round(max(0.0, moved) * 10000, 2)


def _inside(side: str, price: float, limit_low: float, limit_high: float) -> bool:
    return price <= limit_high if side == "buy" else price >= limit_low


def _outside_reason(side: str, price: float, limit_low: float, limit_high: float, what: str) -> str:
    if side == "buy":
        return f"{what} {price:,.4f} is above the highest price allowed, {limit_high:,.4f}: the price has moved past what the Critic judged, so a new review is needed"
    return f"{what} {price:,.4f} is below the lowest price allowed, {limit_low:,.4f}: the price has moved past what the Critic judged, so a new review is needed"


def _check(side: str, order_type: str, limit_price, limit_low, limit_high) -> Execution | None:
    """The arguments every fill rule shares. A rejection when they cannot be traded on; None when they are fine. A limit price outside the range is rejected here too: the tool refuses it first, and this is the
    second check, so that no order of any type can fill outside the limits."""
    if side not in SIDES:
        return _rejected("internal_error", f"side must be buy or sell, not {side!r}")
    if order_type not in ORDER_TYPES:
        return _rejected("internal_error", f"order type must be market or limit, not {order_type!r}")
    if not usable(limit_low) or not usable(limit_high) or limit_low > limit_high:
        return _rejected("price_outside_limits", "the trade has no price range to trade inside")
    if order_type == "limit":
        if not usable(limit_price):
            return _rejected("internal_error", "a limit order needs a limit price above zero")
        if not limit_low <= limit_price <= limit_high:
            return _rejected("price_outside_limits", f"the limit price {limit_price:,.4f} is outside the range {limit_low:,.4f} to {limit_high:,.4f}")
    return None


def fill_paper(side: str, order_type: str, limit_price, quote, limit_low: float, limit_high: float, slippage_bps: float) -> Execution:
    """One order against the current quote (paper mode). Returns an `Execution`: filled, `active` (a limit order the quote has not reached) or rejected. See the module text for the rules."""
    refused = _check(side, order_type, limit_price, limit_low, limit_high)
    if refused is not None:
        return refused
    if not usable(quote):
        return _rejected("price_unavailable", "there is no usable quote for the symbol")
    slip = slippage_bps / 10000
    moved = r4(quote * (1 + slip)) if side == "buy" else r4(quote * (1 - slip))
    if order_type == "market":
        if not _inside(side, moved, limit_low, limit_high):
            return _rejected("price_outside_limits", _outside_reason(side, moved, limit_low, limit_high, "the market price"))
        return Execution("filled", f"market order filled at the quote {quote:,.4f} with {slippage_bps:g} bps of slippage", moved, "quote", _bps(moved, quote, side))
    reached = quote <= limit_price if side == "buy" else quote >= limit_price
    if not reached:
        way = "above" if side == "buy" else "below"
        return Execution("active", f"resting: the quote {quote:,.4f} is {way} the limit {limit_price:,.4f}")
    price = r4(min(limit_price, quote * (1 + slip))) if side == "buy" else r4(max(limit_price, quote * (1 - slip)))
    return Execution("filled", f"limit order filled at {price:,.4f} (the quote is {quote:,.4f}, the limit {limit_price:,.4f})", price, "quote", _bps(price, quote, side))


def fill_backtest(side: str, order_type: str, limit_price, bar: dict, limit_low: float, limit_high: float, slippage_bps: float) -> Execution:
    """One day order against the entry bar (backtest mode): see the module text. `bar` is the symbol's bar on the run's entry date."""
    refused = _check(side, order_type, limit_price, limit_low, limit_high)
    if refused is not None:
        return refused
    if not isinstance(bar, dict) or not usable(bar.get("open")) or not usable(bar.get("low")) or not usable(bar.get("high")):
        return _rejected("price_unavailable", "the entry bar has no usable open, high and low")
    slip, opened, day = slippage_bps / 10000, float(bar["open"]), bar.get("date", "the entry date")
    moved = r4(opened * (1 + slip)) if side == "buy" else r4(opened * (1 - slip))
    if order_type == "market":
        if not _inside(side, moved, limit_low, limit_high):
            return _rejected("price_outside_limits", _outside_reason(side, moved, limit_low, limit_high, "the opening price"))
        return Execution("filled", f"market order filled at the open {opened:,.4f} of {day} with {slippage_bps:g} bps of slippage", moved, "bar_open", _bps(moved, opened, side))
    reached_at_open = opened <= limit_price if side == "buy" else opened >= limit_price
    if reached_at_open:
        price = r4(min(limit_price, opened * (1 + slip))) if side == "buy" else r4(max(limit_price, opened * (1 - slip)))
        capped = price == r4(limit_price) and price != moved
        return Execution("filled", f"limit order filled at {price:,.4f} (the open of {day} was {opened:,.4f}, the limit {limit_price:,.4f})", price, "bar_limit" if capped else "bar_open", _bps(price, opened, side))
    extreme = float(bar["low"]) if side == "buy" else float(bar["high"])      # a buy needs the day to trade down to the limit, a sale needs it to trade up
    if extreme <= limit_price if side == "buy" else extreme >= limit_price:
        return Execution("filled", f"limit order filled at its limit {limit_price:,.4f}: {day} traded {'down' if side == 'buy' else 'up'} to {extreme:,.4f}", r4(limit_price), "bar_limit", 0.0)
    return _rejected("limit_not_reached", f"the limit {limit_price:,.4f} was not reached on {day} (a day order): the day's {'low' if side == 'buy' else 'high'} was {extreme:,.4f}")


# ---------------------------------------------------------------- the entry date of a backtest

def calendar_entry_day(now: datetime) -> date:
    """Step one of the entry date: the first trading day whose regular open (09:30 Eastern) is STRICTLY after `now`. A decision at 10:30 on Thursday gives Friday, one at 08:00 on Thursday gives Thursday, one at exactly
    09:30 gives the next day (the open is not after it), one on a Saturday gives Monday. From the calendar alone: this is the rule that keeps the future out of the decision. Raises ValueError for a naive `now`."""
    config = load_calendar()
    day = eastern_date(now, config)              # raises ValueError for a naive now
    if not is_trading_day(day, config):
        return next_trading_day(day, config)
    opens = datetime.combine(day, config.regular_open, tzinfo=ZoneInfo(config.timezone))
    return day if opens > now else next_trading_day(day, config)


def entry_date(now: datetime, bars: dict[str, list[dict]]) -> date | None:
    """The one entry date of a backtest run: the earliest bar date on or after `calendar_entry_day(now)`, among the bars of EVERY symbol (`bars` = {symbol: [bar, ...]}). None when no bar is stored from that day on:
    the future is not stored yet, and the orders wait (`active`)."""
    start = calendar_entry_day(now)
    found = [d for series in bars.values() for bar in series if isinstance(bar, dict) and (d := iso_date(bar.get("date"))) is not None and d >= start]
    return min(found) if found else None


def bar_on(series: list[dict] | None, day: date | str | None) -> dict | None:
    """The bar of `day` (a date or "YYYY-MM-DD") in a series, or None."""
    wanted = day.isoformat() if isinstance(day, date) else day
    return next((b for b in series or [] if isinstance(b, dict) and b.get("date") == wanted), None)


# ---------------------------------------------------------------- following a purchase

def _gain(shares: int, price: float, base: float) -> float:
    return round(trade_amount(shares, price) - trade_amount(shares, base), 2)


def _benchmark(bars: list[dict] | None, entry: str, end: str) -> float | None:
    """The benchmark's return from the open of the entry bar to the close of the bar on `end` (or the last bar before it): None when its bars are not stored for those dates."""
    first = bar_on(bars, entry)
    closes = [b for b in bars or [] if isinstance(b, dict) and isinstance(b.get("date"), str) and entry <= b["date"] <= end and usable(b.get("close"))]
    if first is None or not usable(first.get("open")) or not closes:
        return None
    return round(float(closes[-1]["close"]) / float(first["open"]) - 1, RETURN_PLACES)


def follow(fill_price: float, shares: int, entry_day: date | str, bars: list[dict], stop_price: float | None, target_price: float | None, time_stop_date: str | None, slippage_bps: float,
           benchmark_bars: list[dict] | None = None) -> Outcome:
    """How a filled purchase went: the Advisor's stop, target and time stop, followed through `bars` (the symbol's bars, oldest first, the entry bar first) from the entry bar to the last stored bar. The first rule that
    holds ends the trade (see the module text); with none by the last bar the trade is `open` and marked at the last close. `None` for `target_price` (a capital-return plan has none), `stop_price` or `time_stop_date`
    skips that rule. A series that does not start with the entry bar cannot be followed: the outcome is `open` with nothing in it. `benchmark_bars` (the benchmark symbol's bars over the same dates) gives the benchmark's
    return from the entry bar's open to the exit date's close (or the last close), and the excess return; None when they are not stored. Returns are fractions (0.0077 is 0.77%). Pure: nothing is read or written."""
    entry = entry_day.isoformat() if isinstance(entry_day, date) else str(entry_day)
    series = [b for b in bars or [] if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] >= entry and all(usable(b.get(k)) for k in ("open", "high", "low", "close"))]
    if not series or series[0]["date"] != entry:
        return Outcome(status="open")
    slip = slippage_bps / 10000
    for index, bar in enumerate(series):
        opened, high, low, close = (float(bar[k]) for k in ("open", "high", "low", "close"))
        exit_price = reason = None
        if usable(stop_price):
            if opened <= stop_price:
                exit_price, reason = r4(opened * (1 - slip)), "stop"          # it gapped through the stop: sold at the open
            elif low <= stop_price:
                exit_price, reason = r4(stop_price * (1 - slip)), "stop"
        if reason is None and index > 0:                                    # on the entry bar only the stop can fire: the order of prices inside one day is not known, so the bad case is assumed
            if usable(target_price) and opened >= target_price:
                exit_price, reason = r4(opened), "target"
            elif usable(target_price) and high >= target_price:
                exit_price, reason = r4(target_price), "target"
            elif isinstance(time_stop_date, str) and bar["date"] >= time_stop_date:
                exit_price, reason = r4(close * (1 - slip)), "time_stop"
        if reason is not None:
            gain = _gain(shares, exit_price, fill_price)
            bench = _benchmark(benchmark_bars, entry, bar["date"])
            returned = round(exit_price / fill_price - 1, RETURN_PLACES)
            return Outcome(status="closed", exit_reason=reason, exit_date=bar["date"], exit_price=exit_price, sessions_held=index, realized_gain=gain, return_pct=returned,
                           benchmark_return_pct=bench, excess_return_pct=None if bench is None else round(returned - bench, RETURN_PLACES))
    last = series[-1]
    last_close = r4(float(last["close"]))
    returned = round(last_close / fill_price - 1, RETURN_PLACES)
    bench = _benchmark(benchmark_bars, entry, last["date"])
    return Outcome(status="open", sessions_held=len(series) - 1, return_pct=returned, last_date=last["date"], last_close=last_close, unrealized_gain=_gain(shares, last_close, fill_price),
                   benchmark_return_pct=bench, excess_return_pct=None if bench is None else round(returned - bench, RETURN_PLACES))
