"""The US stock market's calendar: which days trade, and which window of the day a moment falls in (Critic v1 spec section 4.2, design choices entry 57).

    session = session_at(datetime(2026, 11, 13, 16, 0, tzinfo=ZoneInfo("America/New_York")))
    session.state            # "closed": Jacob's rule, no trading after 4 PM on a Friday
    session.reason           # "after the regular close on a day before a closed day"
    session.next_open_utc    # Monday 2026-11-16 04:00 Eastern, as a UTC datetime

Pure functions of the moment they are given: no network, no model, and NEVER the real clock (a caller passes `now`; a test passes any moment it likes). A `now` without a timezone is a
programming mistake and raises ValueError. Times are US Eastern (`zoneinfo` handles daylight saving; the standard library is enough, no dependency was added).

THE HOLIDAYS are computed from the NYSE's rules, so the calendar never goes stale: the file `configs/market_calendar.json` holds only what a rule cannot know (the early-close days,
and a closure nobody can compute, `extra_closed_dates`). The rules were checked against the NYSE's own published list for 2026, 2027 and 2028 (tests/test_market_calendar.py).

THE TRADING WINDOW of a trading day (a moment belongs to an interval that includes its start and excludes its end):

    pre-market    [04:00, 09:30)
    regular       [09:30, close)      close is 16:00, or 13:00 on an early-close day
    after-hours   [close, 20:00)      ONLY when the next calendar day is a trading day and today is not an early-close day

So the window ends at the regular close on a Friday, on the day before a holiday and on an early-close day: that is Jacob's rule "exclude after 4 PM on Friday", applied to every day whose
next day is closed (`after_hours_before_closed_day` in the config turns that around). Everything else, Saturdays, Sundays, holidays, before 04:00 and after the window's end, is `closed`.
"""
import functools
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, field_validator

from prometheus.rag.config import REPO_ROOT

CALENDAR_PATH = REPO_ROOT / "configs" / "market_calendar.json"
MONDAY, THURSDAY = 0, 3
STATES = ("pre_market", "regular", "after_hours", "closed")


class CalendarConfig(BaseModel):
    """`configs/market_calendar.json`, read and checked once: a missing or unknown key is an error at load."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    timezone: str
    pre_market_open: time
    regular_open: time
    regular_close: time
    after_hours_close: time
    early_close: time
    after_hours_before_closed_day: bool
    simulated_minutes_before_close: int       # USE_LIVE_TRADING=false: the moment the desk works at is this many minutes before the regular close of the day it simulates (`simulated_moment`)
    early_close_dates: frozenset[date]
    extra_closed_dates: dict[date, str]

    @field_validator("timezone")
    @classmethod
    def _zone_exists(cls, name: str) -> str:
        ZoneInfo(name)  # raises for an unknown zone
        return name

    def model_post_init(self, _) -> None:
        if not self.pre_market_open < self.regular_open < self.early_close < self.regular_close < self.after_hours_close:
            raise ValueError("the times must run pre_market_open < regular_open < early_close < regular_close < after_hours_close")
        shortest = (self.early_close.hour - self.regular_open.hour) * 60 + (self.early_close.minute - self.regular_open.minute)   # the regular session of an early-close day
        if not 0 < self.simulated_minutes_before_close < shortest:
            raise ValueError(f"simulated_minutes_before_close must be from 1 to {shortest - 1}, so that the simulated moment is inside the regular session of every day, early-close days included")


@functools.lru_cache(maxsize=None)
def _load(path: Path) -> CalendarConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: the market calendar file does not exist") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error.msg})") from None
    try:
        return CalendarConfig.model_validate({k: v for k, v in raw.items() if not k.startswith("_")})
    except Exception as error:  # pydantic's ValidationError names every bad key
        raise ValueError(f"{path}: {error}") from None


def load_calendar(path: str | Path | None = None) -> CalendarConfig:
    """The calendar's settings (the default file when `path` is left out). Raises ValueError, naming the file and the problem."""
    return _load(Path(path) if path else CALENDAR_PATH)


# ---------------------------------------------------------------- holidays, by rule

def easter(year: int) -> date:
    """Easter Sunday of a year in the Gregorian calendar (the anonymous Gregorian algorithm, Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1 to 4) given weekday of a month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """A holiday on a Saturday is kept on the Friday before it, one on a Sunday on the Monday after it."""
    return day - timedelta(days=1) if day.weekday() == 5 else day + timedelta(days=1) if day.weekday() == 6 else day


@functools.lru_cache(maxsize=None)
def holidays(year: int) -> dict[date, str]:
    """The days of `year` the NYSE is closed for a holiday, {date: name}. New Year's Day on a Saturday is NOT observed (the NYSE does not close on the Friday before it, so December 31 never
    appears here); the other fixed-date holidays are observed on the Friday or the Monday."""
    found = {
        _nth_weekday(year, 1, MONDAY, 3): "Martin Luther King, Jr. Day",
        _nth_weekday(year, 2, MONDAY, 3): "Washington's Birthday",
        easter(year) - timedelta(days=2): "Good Friday",
        _last_weekday(year, 5, MONDAY): "Memorial Day",
        _observed(date(year, 6, 19)): "Juneteenth National Independence Day",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, MONDAY, 1): "Labor Day",
        _nth_weekday(year, 11, THURSDAY, 4): "Thanksgiving Day",
        _observed(date(year, 12, 25)): "Christmas Day",
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        found[_observed(new_year)] = "New Year's Day"
    return found


def closed_reason(day: date, config: CalendarConfig | None = None) -> str | None:
    """Why the market is closed all day on `day`: "weekend", "holiday: <name>", or the reason of an extra closed date; None for a trading day."""
    config = config or load_calendar()
    if day.weekday() >= 5:
        return "weekend"
    if day in holidays(day.year):
        return f"holiday: {holidays(day.year)[day]}"
    if day in config.extra_closed_dates:
        return config.extra_closed_dates[day]
    return None


def is_trading_day(day: date, config: CalendarConfig | None = None) -> bool:
    return closed_reason(day, config) is None


def previous_trading_day(day: date, config: CalendarConfig | None = None) -> date:
    """The newest trading day strictly before `day`."""
    day -= timedelta(days=1)
    while not is_trading_day(day, config):
        day -= timedelta(days=1)
    return day


def next_trading_day(day: date, config: CalendarConfig | None = None) -> date:
    """The first trading day strictly after `day`."""
    day += timedelta(days=1)
    while not is_trading_day(day, config):
        day += timedelta(days=1)
    return day


# ---------------------------------------------------------------- the moment

@dataclass(frozen=True)
class Session:
    """Where a moment falls in the trading calendar."""
    state: str                              # pre_market | regular | after_hours | closed
    trading_date: date | None               # the trading day this moment belongs to; None when closed
    reason: str                             # why it is closed, in words; empty when open
    window_closes_utc: datetime | None      # when the whole trading window of the day ends (not the end of the sub-session); None when closed
    next_open_utc: datetime                 # the next pre-market open strictly after this moment
    last_trading_date: date                 # the newest trading day on or before this moment's Eastern date

    @property
    def is_open(self) -> bool:
        return self.state != "closed"


def _zone(config: CalendarConfig) -> ZoneInfo:
    return ZoneInfo(config.timezone)


def _utc(day: date, at: time, config: CalendarConfig) -> datetime:
    return datetime.combine(day, at, tzinfo=_zone(config)).astimezone(timezone.utc)


def _aware(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone (a naive time cannot be placed in the market's calendar)")
    return now


def eastern_date(now: datetime, config: CalendarConfig | None = None) -> date:
    """The calendar date of a moment in the market's own time zone (US Eastern): the date a recommendation is made on. NEVER the UTC date: between 19:00 and 20:00 Eastern the UTC date is
    already tomorrow."""
    return _aware(now).astimezone(_zone(config or load_calendar())).date()


def simulated_moment(day: date, config: CalendarConfig | None = None) -> datetime:
    """The moment the desk works at when it SIMULATES the market being open on `day` (`USE_LIVE_TRADING=false`, `common/trading_clock.py`): `simulated_minutes_before_close` before that day's regular close (16:00,
    or 13:00 on an early-close day), as an aware UTC datetime. It is always inside the regular session, so a recommendation made then is for `day`, the market is "open" and an approval has time left (until the
    window closes, or an hour). Raises ValueError for a day that is not a trading day: nothing happened on it."""
    config = config or load_calendar()
    if not is_trading_day(day, config):
        raise ValueError(f"{day} is not a trading day ({closed_reason(day, config)}): there is no market to simulate")
    close = config.early_close if day in config.early_close_dates else config.regular_close
    return (datetime.combine(day, close, tzinfo=_zone(config)) - timedelta(minutes=config.simulated_minutes_before_close)).astimezone(timezone.utc)


def regular_close_utc(day: date, config: CalendarConfig | None = None) -> datetime:
    """The moment the regular session of `day` closes (16:00 Eastern, 13:00 on an early-close day), as an aware UTC datetime. For a day that is not a trading day it is the same clock time: the caller asks about a
    day it has a bar for."""
    config = config or load_calendar()
    close = config.early_close if day in config.early_close_dates else config.regular_close
    return datetime.combine(day, close, tzinfo=_zone(config)).astimezone(timezone.utc)


def window_end(day: date, config: CalendarConfig | None = None) -> time:
    """When the trading window of a trading day ends: the after-hours close, or the regular close when there is no after-hours session that day."""
    config = config or load_calendar()
    early = day in config.early_close_dates
    close = config.early_close if early else config.regular_close
    after_hours = not early and (config.after_hours_before_closed_day or is_trading_day(day + timedelta(days=1), config))
    return config.after_hours_close if after_hours else close


def session_at(now: datetime, config: CalendarConfig | None = None) -> Session:
    """The session of a moment (see the module text). Raises ValueError for a `now` without a timezone."""
    config = config or load_calendar()
    local = _aware(now).astimezone(_zone(config))
    day, clock = local.date(), local.time()
    last = day if is_trading_day(day, config) else previous_trading_day(day, config)
    after = next_trading_day(day, config)
    if not is_trading_day(day, config):
        return Session("closed", None, closed_reason(day, config) or "closed", None, _utc(after, config.pre_market_open, config), last)
    close = config.early_close if day in config.early_close_dates else config.regular_close
    end = window_end(day, config)
    if clock < config.pre_market_open:
        return Session("closed", None, "before the pre-market open", None, _utc(day, config.pre_market_open, config), last)
    if clock >= end:
        reason = ("after the after-hours close" if end == config.after_hours_close
                  else "after the early close" if day in config.early_close_dates else "after the regular close on a day before a closed day")
        return Session("closed", None, reason, None, _utc(after, config.pre_market_open, config), last)
    state = "pre_market" if clock < config.regular_open else "regular" if clock < close else "after_hours"
    return Session(state, day, "", _utc(day, end, config), _utc(after, config.pre_market_open, config), last)
