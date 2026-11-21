"""The desk's clock: the real moment, or a SIMULATED one when `USE_LIVE_TRADING=false` (design choices entry 76).

    trading_now()                 # the moment every agent of the desk works at: now, or (USE_LIVE_TRADING=false) the last trading day that has closed, half an hour before its close
    market_day()                  # the trading day that moment belongs to (the date a recommendation is made for)
    use_live_trading()            # the setting: environment, then .env; true, yes, on or 1 (any case); unset means true
    clock_note()                  # one line for a report when the clock is simulated, else None
    as_of(day)                    # a context manager (design choices entry 127, the Proving Ground): while it is open the desk works AS OF a past trading day, whatever the setting
    bound_day()                   # the day `as_of` binds, or None
    horizon()                     # the last instant the desk may know while a day is bound: the end of that day, Eastern time, as an aware UTC datetime; None when no day is bound

WHY. The desk judges a recommendation at a moment: the market must be open, the recommendation must be for that trading day, a quote must be fresh, and an approval lasts until the window closes (Critic v1, entries
57 and 69). At night, at the weekend and on a holiday none of that can be true, so the desk could not be run for testing and development. With `USE_LIVE_TRADING=false` the desk works at a simulated moment:
the last trading day that has CLOSED (`last_closed_day`, entry 129: today itself once its regular session has closed, so Friday 16:12 is Friday; before the close, after midnight, at the weekend and on a holiday the trading day before, so Saturday and Sunday are Friday), `simulated_minutes_before_close` (configs/market_calendar.json, 30) before its regular close: inside the
regular session, the market is "open", and the data are the market close of that day. The moment is the same for every agent and does not advance, so an approval never runs out in the middle of a run.

WHAT IT IS NOT. It is not a switch for real trading: there is none, and `PAPER_TRADING` (the Trader's rule T4) is a separate setting that stays on. `USE_LIVE_TRADING=true` (the default when the setting is
absent: nothing changes for anyone who does not set it) only means "use the real clock and Finnhub's live quote". A typing mistake in the value is read as false, the simulated and safe side.

THE QUOTE. With the setting off, the Data Clerk answers a price request from the stored daily bar of that day (`DataClerk.request_quotes`: no Finnhub call), and marks the quote `simulated`. The Critic's
check of the stored close against a second provider skips such a quote: it would be the same bar compared with itself.

An explicit moment (the runner's `--now`, a replay) always wins over this clock: it is how a past day is replayed on purpose. Nothing here reads the real clock except `trading_now`, `market_day` and the
"day in the future" check of `as_of`; pure functions of a given moment (`previous_business_day`) stay pure. Needs nothing but the calendar: no network, no model.

AS OF A PAST DAY (the Proving Ground, design choices entry 127, design/proving_ground_spec.md section 4). The Time Machine replays the real desk at a past trading day D, and at D every input must be what was known on D. `as_of(D)` binds D in a
`contextvars.ContextVar`, so the binding follows `contextvars.copy_context()` into the threads that already copy the context (the Advisor's `calls.invoke_with_limit`, RaiseFunds) and LangGraph's node threads, and it never leaks to another
thread or another run. While a day is bound: `use_live_trading()` is False whatever the setting (so everything entry 76 built applies: the quote is the stored bar of D, marked `simulated`, and no second provider is asked), `trading_now()`
is `simulated_moment(D)`, `market_day()` is D, and `clock_note()` says so. Every data door reads `horizon()` itself (the retriever, the Data Clerk's tools and toolbox, the filing evidence, the ledger) and returns nothing dated after it: no
caller has to remember to pass a date. `as_of` refuses a day that is not a trading day, a day in the future, and a second day inside it (ValueError): one run is at one moment.
"""
import contextlib
import contextvars
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from prometheus.common.market_calendar import CalendarConfig, closed_reason, eastern_date, is_trading_day, load_calendar, previous_trading_day, simulated_moment
from prometheus.rag.config import setting

TRUE_WORDS = ("true", "yes", "on", "1")   # the values of USE_LIVE_TRADING that mean the real clock and the live quote; anything else, a typo included, is the simulation
NEVER_FETCHES = "the Proving Ground never fetches"   # the words of every refusal to fetch while a day is bound (a test of the Time Machine must not call a paid API or write the stored text)

_BOUND: contextvars.ContextVar[date | None] = contextvars.ContextVar("prometheus_as_of_day", default=None)   # the day `as_of` binds; None when nothing is bound


def bound_day() -> date | None:
    """The trading day `as_of` has bound in this context, or None."""
    return _BOUND.get()


@contextlib.contextmanager
def as_of(day: date):
    """Work AS OF a past trading day, for everything run inside the `with` (including threads that run in a copy of this context). Raises ValueError, before anything is bound, for a `day` that is not a `date` (a
    `datetime` is not one), a day the market was closed (`market_calendar.closed_reason` says why), a day after today's Eastern date, and a second, different day while one is already bound. The same day again is allowed
    (the binding is simply kept). Yields the day.

        with trading_clock.as_of(date(2026, 11, 9)):
            trading_now()      # 2026-11-09 15:30 ET, as UTC
            market_day()       # 2026-11-09
            horizon()          # 2026-11-10 04:59:59.999999 UTC (the end of the day in New York)
    """
    if type(day) is not date:
        raise ValueError(f"as_of needs a date (got {type(day).__name__})")
    config = load_calendar()
    if not is_trading_day(day, config):
        raise ValueError(f"as_of({day}): not a trading day ({closed_reason(day, config)})")
    today = eastern_date(datetime.now(timezone.utc), config)
    if day > today:
        raise ValueError(f"as_of({day}): that day is in the future (today is {today}); the desk cannot know it yet")
    current = _BOUND.get()
    if current is not None and current != day:
        raise ValueError(f"as_of({day}): the desk is already working as of {current}; one run is at one moment")
    token = _BOUND.set(day)
    try:
        yield day
    finally:
        _BOUND.reset(token)


def end_of_day(day: date, config: CalendarConfig | None = None) -> datetime:
    """The last instant of a calendar day in the market's own time zone (US Eastern), as an aware UTC datetime: 23:59:59.999999 on that day in New York."""
    config = config or load_calendar()
    return datetime.combine(day, time.max, tzinfo=ZoneInfo(config.timezone)).astimezone(timezone.utc)


def horizon() -> datetime | None:
    """The last instant the desk may know: the end of `bound_day()` in Eastern time, as an aware UTC datetime. None when no day is bound. It is the end of the day and not the simulated moment (15:30) on purpose:
    the stored daily bar of the day is its close (entry 76 works "with the market close of that day"), and an article of that evening is the same information set as the close. The decision is filled at the NEXT open, so nothing after
    the decision is used."""
    day = _BOUND.get()
    return None if day is None else end_of_day(day)


def use_live_trading() -> bool:
    """`USE_LIVE_TRADING` from the environment, then `.env`: true, yes, on or 1 mean the real clock and the live quote; unset means true (nothing changes); anything else means the simulated clock. While a day is
    bound (`as_of`) the answer is False whatever the setting: a past day has no live quote."""
    if _BOUND.get() is not None:
        return False
    return setting("USE_LIVE_TRADING", "true").strip().lower() in TRUE_WORDS


def refuse_when_bound(what: str) -> None:
    """Raise RuntimeError (`NEVER_FETCHES`) when a day is bound: a fetch calls a paid API and writes the stored text, and a test of a past day does neither. Called at the top of every fetcher."""
    day = _BOUND.get()
    if day is not None:
        raise RuntimeError(f"{NEVER_FETCHES}: {what} was refused while the desk works as of {day}")


def previous_business_day(moment: datetime, config: CalendarConfig | None = None) -> date:
    """The trading day before the Eastern date of `moment`: Friday for a Saturday, a Sunday and a Monday, the day before a holiday for the day after it. Pure: a function of the moment given.
    Raises ValueError for a moment without a timezone."""
    config = config or load_calendar()
    return previous_trading_day(eastern_date(moment, config), config)


def last_closed_day(moment: datetime, config: CalendarConfig | None = None) -> date:
    """The last trading day whose regular session has CLOSED at `moment` (design choices entry 129; Jacob, 2026-11-20): the Eastern date of `moment` itself when that is a trading day and the clock there is at or
    past its close (16:00, or 13:00 on an early-close day), else the trading day before it. So Friday 16:12 is Friday; Friday 11:00 is Thursday (Friday has no close yet); 00:30 on Saturday, all of Saturday and
    Sunday, and Monday before its close are Friday; the evening of a holiday is the trading day before it. Pure: a function of the moment given. Raises ValueError for a moment without a timezone."""
    config = config or load_calendar()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    local = moment.astimezone(ZoneInfo(config.timezone))
    day = local.date()
    if is_trading_day(day, config):
        close = config.early_close if day in config.early_close_dates else config.regular_close
        if local.time() >= close:
            return day
    return previous_trading_day(day, config)


def trading_now(real_now: datetime | None = None, config: CalendarConfig | None = None) -> datetime:
    """The moment the desk works at. `USE_LIVE_TRADING` true: `real_now` (the real clock when left out). False: the simulated moment of the LAST TRADING DAY THAT HAS CLOSED (`last_closed_day`: today after the close, else the trading day before; `market_calendar.simulated_moment`), whatever
    the real moment is, a trading day or not. While a day is bound (`as_of`): the simulated moment of THAT day, whatever the setting and the real moment. An aware datetime in every case. Raises ValueError for a
    `real_now` without a timezone."""
    real = real_now or datetime.now(timezone.utc)
    config = config or load_calendar()
    if real.tzinfo is None or real.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    if _BOUND.get() is not None:
        return simulated_moment(_BOUND.get(), config)
    if use_live_trading():
        return real
    return simulated_moment(last_closed_day(real, config), config)


def market_day(real_now: datetime | None = None, config: CalendarConfig | None = None) -> date:
    """The trading day the desk's moment belongs to: today's Eastern date when the clock is real, the last trading day that has closed when it is simulated (`last_closed_day`), the bound day inside `as_of`."""
    return eastern_date(trading_now(real_now, config), config)


def clock_note(real_now: datetime | None = None) -> str | None:
    """A line for the top of a report when the clock is simulated or a day is bound; None when it is real."""
    day = _BOUND.get()
    if day is not None:
        local = trading_now(real_now).astimezone(ZoneInfo(load_calendar().timezone))
        return (f"CLOCK    AS OF {day} (the Proving Ground): the desk works at {local:%Y-%m-%d %H:%M} ET, in the regular session of {local:%A}, {local:%Y-%m-%d}, with the market close of that day; nothing dated after that day "
                f"is read, no live quote is used and nothing is fetched. Nothing is sent to a broker either way.")
    if use_live_trading():
        return None
    local = trading_now(real_now).astimezone(ZoneInfo(load_calendar().timezone))
    return (f"CLOCK    SIMULATED (USE_LIVE_TRADING=false): the desk works at {local:%Y-%m-%d %H:%M} ET, in the regular session of {local:%A}, {local:%Y-%m-%d}, with the market close of that day; "
            f"no live quote is used. Nothing is sent to a broker either way.")
