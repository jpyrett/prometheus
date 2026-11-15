"""The desk's clock: the real moment, or a SIMULATED one when `USE_LIVE_TRADING=false` (design choices entry 76).

    trading_now()                 # the moment every agent of the desk works at: now, or (USE_LIVE_TRADING=false) the previous business day, half an hour before its close
    market_day()                  # the trading day that moment belongs to (the date a recommendation is made for)
    use_live_trading()            # the setting: environment, then .env; true, yes, on or 1 (any case); unset means true
    clock_note()                  # one line for a report when the clock is simulated, else None

WHY. The desk judges a recommendation at a moment: the market must be open, the recommendation must be for that trading day, a quote must be fresh, and an approval lasts until the window closes (Critic v1, entries
57 and 69). At night, at the weekend and on a holiday none of that can be true, so the desk could not be run for testing and development. With `USE_LIVE_TRADING=false` the desk works at a simulated moment:
the previous business day (the trading day before today's Eastern date, holidays and weekends skipped), `simulated_minutes_before_close` (configs/market_calendar.json, 30) before its regular close: inside the
regular session, the market is "open", and the data are the market close of that day. The moment is the same for every agent and does not advance, so an approval never runs out in the middle of a run.

WHAT IT IS NOT. It is not a switch for real trading: there is none, and `PAPER_TRADING` (the Trader's rule T4) is a separate setting that stays on. `USE_LIVE_TRADING=true` (the default when the setting is
absent: nothing changes for anyone who does not set it) only means "use the real clock and Finnhub's live quote". A typing mistake in the value is read as false, the simulated and safe side.

THE QUOTE. With the setting off, the Data Clerk answers a price request from the stored daily bar of that day (`DataClerk.request_quotes`: no Finnhub call), and marks the quote `simulated`. The Critic's
check of the stored close against a second provider skips such a quote: it would be the same bar compared with itself.

An explicit moment (the runner's `--now`, a replay) always wins over this clock: it is how a past day is replayed on purpose. Nothing here reads the real clock except `trading_now` and `market_day`; pure functions
of a given moment (`previous_business_day`) stay pure. Needs nothing but the calendar: no network, no model.
"""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from prometheus.common.market_calendar import CalendarConfig, eastern_date, load_calendar, previous_trading_day, simulated_moment
from prometheus.rag.config import setting

TRUE_WORDS = ("true", "yes", "on", "1")   # the values of USE_LIVE_TRADING that mean the real clock and the live quote; anything else, a typo included, is the simulation


def use_live_trading() -> bool:
    """`USE_LIVE_TRADING` from the environment, then `.env`: true, yes, on or 1 mean the real clock and the live quote; unset means true (nothing changes); anything else means the simulated clock."""
    return setting("USE_LIVE_TRADING", "true").strip().lower() in TRUE_WORDS


def previous_business_day(moment: datetime, config: CalendarConfig | None = None) -> date:
    """The trading day before the Eastern date of `moment`: Friday for a Saturday, a Sunday and a Monday, the day before a holiday for the day after it. Pure: a function of the moment given.
    Raises ValueError for a moment without a timezone."""
    config = config or load_calendar()
    return previous_trading_day(eastern_date(moment, config), config)


def trading_now(real_now: datetime | None = None, config: CalendarConfig | None = None) -> datetime:
    """The moment the desk works at. `USE_LIVE_TRADING` true: `real_now` (the real clock when left out). False: the simulated moment of the previous business day (`market_calendar.simulated_moment`), whatever
    the real moment is, a trading day or not. An aware datetime either way. Raises ValueError for a `real_now` without a timezone."""
    real = real_now or datetime.now(timezone.utc)
    config = config or load_calendar()
    if real.tzinfo is None or real.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    if use_live_trading():
        return real
    return simulated_moment(previous_business_day(real, config), config)


def market_day(real_now: datetime | None = None, config: CalendarConfig | None = None) -> date:
    """The trading day the desk's moment belongs to: today's Eastern date when the clock is real, the previous business day when it is simulated."""
    return eastern_date(trading_now(real_now, config), config)


def clock_note(real_now: datetime | None = None) -> str | None:
    """A line for the top of a report when the clock is simulated; None when it is real."""
    if use_live_trading():
        return None
    local = trading_now(real_now).astimezone(ZoneInfo(load_calendar().timezone))
    return (f"CLOCK    SIMULATED (USE_LIVE_TRADING=false): the desk works at {local:%Y-%m-%d %H:%M} ET, in the regular session of {local:%A}, {local:%Y-%m-%d}, with the market close of that day; "
            f"no live quote is used. Nothing is sent to a broker either way.")
