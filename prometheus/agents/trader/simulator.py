"""The Trader's fill rules and the exit rules, as PURE functions (Trader v1 spec sections 6 and 10).

    execution = fill_paper("buy", "limit", Decimal("167.1"), quote=Decimal("167.0"), limit_low=Decimal("165.46"), limit_high=Decimal("168.74"), slippage_bps=Decimal(5))        # filled at 167.0835
    execution = fill_backtest("buy", "market", None, bar, limit_low=Decimal("165.46"), limit_high=Decimal("168.74"), slippage_bps=Decimal(5))                                   # the entry bar's open, plus slippage
    day = entry_date(now, bars)                                                                                                                                                # the one entry date of a backtest run
    outcome = follow(fill_price=Decimal("167.18"), shares=Decimal(44), entry_day="2026-11-06", bars=bars, stop_price=Decimal("157.23"), target_price=Decimal("168.38"), time_stop_date="2026-12-31", slippage_bps=Decimal(5))

PRICES, AMOUNTS AND SHARES ARE `Decimal` (design choices entry 112; design/decimal_money_spec.md). A price here is a Decimal at four places (`price_scale`, half up: the one rounding of a price); `Decimal * float` is a TypeError, so a float that was missed stops a test.
A STORED BAR is market data (floats, as the Data Clerk's `request_bars` gives them) and each price of it crosses into the account's code ONCE, through the price door (`bar_price`). A RETURN (`Outcome.return_pct` and the benchmark's) is a fraction, a measurement:
it is worked out from the Decimal prices and is a float at the very end (`_fraction`).

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
A TRAILING STOP (`trail_distance`, design/advisor_exit_plan_spec.md rule E4: the TREND family's plan, which has no target): the stop of bar i is the larger of the plan's stop and the HIGHEST CLOSE OF THE BARS BEFORE i (the entry bar's
close counts for the bar after it) less `trail_distance`, so it only ever rises, and no bar's own prices move its own stop (no look-ahead inside a day). A stop that fired above the plan's own stop is the exit reason `trailing_stop`; at the plan's
own stop it is `stop`. Without `trail_distance` nothing changes: a plan with a fixed stop is followed exactly as before.
"""
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from prometheus.agents.trader.record import Outcome
from prometheus.common.market_calendar import eastern_date, is_trading_day, load_calendar, next_trading_day
from prometheus.common.money import exact, price as price_door, price_scale
from prometheus.common.portfolio import trade_amount
from prometheus.common.work_order import iso_date

PRICE_PLACES = 4        # fill prices, and the prices a limit range is made of
RETURN_PLACES = 6       # a return is a fraction (0.0077 is 0.77%), as every `_pct` of the desk
BPS_STEP = Decimal("0.01")      # the slippage a fill really had is worked to two places of a basis point
RETURN_STEP = Decimal("0.000001")
SIDES = ("buy", "sell")
ORDER_TYPES = ("market", "limit")


@dataclass(frozen=True)
class Execution:
    """What the fill rules say about one order against one price: `filled` (at `price`, from `price_source`, with the slippage that was applied), `active` (a resting order) or `rejected` (with a code of the
    Trader's closed list). `reason` is code's own sentence."""
    status: str                          # filled | active | rejected
    reason: str = ""
    price: Decimal | None = None
    price_source: str | None = None      # quote | bar_open | bar_limit
    slippage_bps: Decimal = Decimal(0)
    reject_code: str | None = None


def usable(value) -> bool:
    """A price or a quantity that can be traded on: a number above zero (a Decimal or an int; a bool, a string and None are not). A float is a TypeError (`money.exact`): a price comes in through a door."""
    if value is None:
        return False
    try:
        return exact(value) > 0
    except ValueError:
        return False


def _rejected(code: str, reason: str) -> Execution:
    return Execution("rejected", reason, reject_code=code)


def _bps(price: Decimal, base: Decimal, side: str) -> Decimal:
    """The slippage a fill really had against its base price, in basis points: what was added to a purchase or taken off a sale (never negative), worked to two places."""
    moved = price / base - 1 if side == "buy" else 1 - price / base
    return (max(Decimal(0), moved) * 10000).quantize(BPS_STEP, rounding=ROUND_HALF_UP)


def _inside(side: str, price: Decimal, limit_low: Decimal, limit_high: Decimal) -> bool:
    return price <= limit_high if side == "buy" else price >= limit_low


def _outside_reason(side: str, price: Decimal, limit_low: Decimal, limit_high: Decimal, what: str) -> str:
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


def fill_paper(side: str, order_type: str, limit_price, quote, limit_low: Decimal, limit_high: Decimal, slippage_bps: Decimal) -> Execution:
    """One order against the current quote (paper mode). Returns an `Execution`: filled, `active` (a limit order the quote has not reached) or rejected. See the module text for the rules. The prices are Decimals (the quote came
    through a door, the Account Manager's `request_prices`); `slippage_bps` is a Decimal of the digits the setting is written with."""
    refused = _check(side, order_type, limit_price, limit_low, limit_high)
    if refused is not None:
        return refused
    if not usable(quote):
        return _rejected("price_unavailable", "there is no usable quote for the symbol")
    slip = exact(slippage_bps, "the slippage") / 10000
    moved = price_scale(quote * (1 + slip)) if side == "buy" else price_scale(quote * (1 - slip))
    if order_type == "market":
        if not _inside(side, moved, limit_low, limit_high):
            return _rejected("price_outside_limits", _outside_reason(side, moved, limit_low, limit_high, "the market price"))
        return Execution("filled", f"market order filled at the quote {quote:,.4f} with {slippage_bps:g} bps of slippage", moved, "quote", _bps(moved, quote, side))
    reached = quote <= limit_price if side == "buy" else quote >= limit_price
    if not reached:
        way = "above" if side == "buy" else "below"
        return Execution("active", f"resting: the quote {quote:,.4f} is {way} the limit {limit_price:,.4f}")
    price = price_scale(min(limit_price, quote * (1 + slip))) if side == "buy" else price_scale(max(limit_price, quote * (1 - slip)))
    return Execution("filled", f"limit order filled at {price:,.4f} (the quote is {quote:,.4f}, the limit {limit_price:,.4f})", price, "quote", _bps(price, quote, side))


def bar_price(bar, key: str) -> Decimal | None:
    """One price of a stored daily bar, THROUGH THE DOOR: the bars are market data (the Data Clerk's, floats as `json` read them) and a price crosses into the account's code here, once (`money.price`: four places, half up); None when the bar has none."""
    try:
        return price_door(bar.get(key)) if isinstance(bar, dict) else None
    except ValueError:
        return None


def fill_backtest(side: str, order_type: str, limit_price, bar: dict, limit_low: Decimal, limit_high: Decimal, slippage_bps: Decimal) -> Execution:
    """One day order against the entry bar (backtest mode): see the module text. `bar` is the symbol's bar on the run's entry date (the Data Clerk's stored bar, floats: its prices go through the price door here)."""
    refused = _check(side, order_type, limit_price, limit_low, limit_high)
    if refused is not None:
        return refused
    opened, high, low = bar_price(bar, "open"), bar_price(bar, "high"), bar_price(bar, "low")
    if opened is None or low is None or high is None:
        return _rejected("price_unavailable", "the entry bar has no usable open, high and low")
    slip, day = exact(slippage_bps, "the slippage") / 10000, bar.get("date", "the entry date")
    moved = price_scale(opened * (1 + slip)) if side == "buy" else price_scale(opened * (1 - slip))
    if order_type == "market":
        if not _inside(side, moved, limit_low, limit_high):
            return _rejected("price_outside_limits", _outside_reason(side, moved, limit_low, limit_high, "the opening price"))
        return Execution("filled", f"market order filled at the open {opened:,.4f} of {day} with {slippage_bps:g} bps of slippage", moved, "bar_open", _bps(moved, opened, side))
    reached_at_open = opened <= limit_price if side == "buy" else opened >= limit_price
    if reached_at_open:
        price = price_scale(min(limit_price, opened * (1 + slip))) if side == "buy" else price_scale(max(limit_price, opened * (1 - slip)))
        capped = price == price_scale(limit_price) and price != moved
        return Execution("filled", f"limit order filled at {price:,.4f} (the open of {day} was {opened:,.4f}, the limit {limit_price:,.4f})", price, "bar_limit" if capped else "bar_open", _bps(price, opened, side))
    extreme = low if side == "buy" else high      # a buy needs the day to trade down to the limit, a sale needs it to trade up
    if extreme <= limit_price if side == "buy" else extreme >= limit_price:
        return Execution("filled", f"limit order filled at its limit {limit_price:,.4f}: {day} traded {'down' if side == 'buy' else 'up'} to {extreme:,.4f}", price_scale(limit_price), "bar_limit", Decimal(0))
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

def _gain(shares: Decimal, price: Decimal, base: Decimal) -> Decimal:
    """What a sale at `price` makes over a purchase at `base`, in dollars to the cent: each side is `trade_amount`, the way the account book works a sale's gain."""
    return trade_amount(shares, price) - trade_amount(shares, base)


def _fraction(numerator: Decimal, denominator: Decimal) -> float:
    """A RETURN: `numerator` over `denominator` less one, as a fraction (0.0077 is 0.77%) to RETURN_PLACES places, half up, as a float. A return is a measurement, never added to money (design/decimal_money_spec.md section 3, last paragraph), so it is
    worked out from the Decimal prices and becomes a float here, at the very end."""
    return float((numerator / denominator - 1).quantize(RETURN_STEP, rounding=ROUND_HALF_UP))


def _benchmark(bars: list[dict] | None, entry: str, end: str) -> float | None:
    """The benchmark's return from the open of the entry bar to the close of the bar on `end` (or the last bar before it): None when its bars are not stored for those dates."""
    first = bar_on(bars, entry)
    closes = [b for b in bars or [] if isinstance(b, dict) and isinstance(b.get("date"), str) and entry <= b["date"] <= end and bar_price(b, "close") is not None]
    if first is None or bar_price(first, "open") is None or not closes:
        return None
    return _fraction(bar_price(closes[-1], "close"), bar_price(first, "open"))


def _return_difference(returned: float, bench: float | None) -> float | None:
    """The excess return, the return less the benchmark's, to RETURN_PLACES places: a difference of two returns (floats, measurements), worked in Decimal from the digits they are written with so that it carries no binary noise."""
    if bench is None:
        return None
    return float((Decimal(str(returned)) - Decimal(str(bench))).quantize(RETURN_STEP, rounding=ROUND_HALF_UP))


def trailing_stop(stop_price: Decimal | None, highest_close: Decimal | None, trail_distance: Decimal | None) -> Decimal | None:
    """The stop in force when the highest close so far is `highest_close`: the plan's `stop_price` or `highest_close - trail_distance` (the price door: four places, half up), whichever is higher. With no `trail_distance` it is the plan's stop as it
    is (a fixed stop), and with no close yet it is the plan's stop. A Decimal, or None when there is neither a stop nor a trail to make one from. The ONE place the trail is worked out: `follow`, the Proving Ground's account and the Live view all call it."""
    fixed = stop_price if usable(stop_price) else None
    if not usable(trail_distance) or highest_close is None:
        return fixed
    level = price_scale(exact(highest_close) - exact(trail_distance))
    return level if fixed is None else max(fixed, level)


def follow(fill_price: Decimal, shares: Decimal, entry_day: date | str, bars: list[dict], stop_price: Decimal | None, target_price: Decimal | None, time_stop_date: str | None, slippage_bps: Decimal,
           benchmark_bars: list[dict] | None = None, trail_distance: Decimal | None = None) -> Outcome:
    """How a filled purchase went: the Advisor's stop, target and time stop, followed through `bars` (the symbol's bars, oldest first, the entry bar first) from the entry bar to the last stored bar. The first rule that
    holds ends the trade (see the module text); with none by the last bar the trade is `open` and marked at the last close. `None` for `target_price` (a capital-return plan has none), `stop_price` or `time_stop_date`
    skips that rule. A series that does not start with the entry bar cannot be followed: the outcome is `open` with nothing in it. `benchmark_bars` (the benchmark symbol's bars over the same dates) gives the benchmark's
    return from the entry bar's open to the exit date's close (or the last close), and the excess return; None when they are not stored. Returns are fractions (0.0077 is 0.77%), floats worked out from Decimal prices at the very end.
    `trail_distance` (a Decimal price distance, None for a fixed stop) makes the stop a TRAILING one (see the module text). Prices are Decimals (the plan's stop and target are the Advisor's floats and cross through the price door in the caller; a stored bar's prices cross here, `bar_price`). Pure: nothing is read or written."""
    entry = entry_day.isoformat() if isinstance(entry_day, date) else str(entry_day)
    series = [b for b in bars or [] if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] >= entry and all(bar_price(b, k) is not None for k in ("open", "high", "low", "close"))]
    if not series or series[0]["date"] != entry:
        return Outcome(status="open")
    slip = exact(slippage_bps, "the slippage") / 10000
    highest = None                                                       # the highest close of the bars BEFORE the one being read (the trailing stop's base)
    for index, bar in enumerate(series):
        opened, high, low, close = (bar_price(bar, k) for k in ("open", "high", "low", "close"))
        exit_price = reason = None
        stop = trailing_stop(stop_price, highest, trail_distance)
        if usable(stop):
            kind = "trailing_stop" if usable(trail_distance) and (not usable(stop_price) or stop > stop_price) else "stop"
            if opened <= stop:
                exit_price, reason = price_scale(opened * (1 - slip)), kind          # it gapped through the stop: sold at the open
            elif low <= stop:
                exit_price, reason = price_scale(stop * (1 - slip)), kind
        if reason is None and index > 0:                                    # on the entry bar only the stop can fire: the order of prices inside one day is not known, so the bad case is assumed
            if usable(target_price) and opened >= target_price:
                exit_price, reason = price_scale(opened), "target"
            elif usable(target_price) and high >= target_price:
                exit_price, reason = price_scale(target_price), "target"
            elif isinstance(time_stop_date, str) and bar["date"] >= time_stop_date:
                exit_price, reason = price_scale(close * (1 - slip)), "time_stop"
        if reason is not None:
            gain = _gain(shares, exit_price, fill_price)
            bench = _benchmark(benchmark_bars, entry, bar["date"])
            returned = _fraction(exit_price, fill_price)
            return Outcome(status="closed", exit_reason=reason, exit_date=bar["date"], exit_price=exit_price, sessions_held=index, realized_gain=gain, return_pct=returned,
                           benchmark_return_pct=bench, excess_return_pct=_return_difference(returned, bench))
        highest = close if highest is None else max(highest, close)
    last = series[-1]
    last_close = price_scale(bar_price(last, "close"))
    returned = _fraction(last_close, fill_price)
    bench = _benchmark(benchmark_bars, entry, last["date"])
    return Outcome(status="open", sessions_held=len(series) - 1, return_pct=returned, last_date=last["date"], last_close=last_close, unrealized_gain=_gain(shares, last_close, fill_price),
                   benchmark_return_pct=bench, excess_return_pct=_return_difference(returned, bench))
