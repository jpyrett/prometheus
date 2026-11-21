"""The Proving Ground's door to the stored bars (design/proving_ground_spec.md P2, P4 and section 4.2 "Bars").

    bars = read_bars(["XOM", "SPY"], start, end)            # {symbol: [bar, ...]} oldest first, each {"date", "open", "high", "low", "close", "volume"}: the stored daily bars, never a fetch
    last = usable_last_day()                               # the last day whose stored bar may be read as a CLOSE

It asks the Data Clerk (`DataClerk.request_bars`, a `fetch_bars` work order: no model, no fetch, nothing written), the door the Trader's backtest and the follower use to see past a decision day. `read_bars` never asks for a day that is not
usable (see `usable_last_day`), so a result and a statistic never rest on a session that is still running.

A BAR STORED TODAY IS NOT A CLOSE. The stored bars are refreshed during the day (the bar of a session that is still running holds the day so far: the volume is a fraction of a day's), so nothing dated on or after TODAY'S market day is
read here; `usable_last_day` is the trading day before it. (After the close a day's bar is whole, but nothing tells a stored bar made at 15:00 from one made at 17:00, so the rule is the day before.)
"""
from datetime import date, datetime, timezone

from prometheus.common.market_calendar import eastern_date, load_calendar, previous_trading_day
from prometheus.common.work_order import new_work_order

REQUESTER = "proving_ground"


def usable_last_day(now: datetime | None = None) -> date:
    """The last day whose stored bar may be read as a close: the last trading day that has CLOSED at the REAL moment (`trading_clock.last_closed_day`: today after its close, else the trading day before). It does not depend on a day bound by `trading_clock.as_of`: the walk of the Time Machine reads the days after
    a decision day on purpose (the fill at the next open, the follower), and only the real today is the limit of what is stored as a close."""
    from prometheus.common.trading_clock import last_closed_day   # entry 129: today's bar is a close once today's regular session has closed
    return last_closed_day(now or datetime.now(timezone.utc))


def read_bars(symbols: list[str], start: date, end: date, clerk=None) -> dict[str, list[dict]]:
    """The stored daily bars of `symbols` from `start` to `end` (cut to `usable_last_day()`), as the Data Clerk gives them. A symbol with none is an empty list. Raises RuntimeError when the Data Clerk refused or could not read the store."""
    if clerk is None:
        from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
        clerk = DataClerk(allow_fetch=False)
    symbols = list(dict.fromkeys(symbols))
    last = usable_last_day()
    end = min(end, last)
    if not symbols or start > end:
        return {s: [] for s in symbols}
    order = new_work_order(f"Daily bars of {', '.join(symbols)} from {start.isoformat()} to {end.isoformat()}", requester=REQUESTER, action_type="fetch_bars", subject=symbols, bars={"start": start.isoformat(), "end": end.isoformat()})
    answered = clerk.request_bars(order)
    if answered.status != "data_ready" or answered.data is None:
        raise RuntimeError("the stored bars could not be read: " + "; ".join(answered.errors)[:300])
    return {s: [b for b in answered.data.bars.get(s, []) if b["date"] <= end.isoformat()] for s in symbols}
