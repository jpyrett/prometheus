"""The market calendar (Critic v1 spec section 4.2 and Appendix B.1). The dates are pinned on purpose: these are functions of the moment they are given, never of the real clock."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from prometheus.common import market_calendar as mc

ET = ZoneInfo("America/New_York")


def at(text: str) -> datetime:
    """An Eastern moment from "YYYY-MM-DD HH:MM"."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# the NYSE's published lists (nyse.com/markets/hours-calendars, read 2026-11-11)


def test_an_extra_closed_date_closes_the_day(tmp_path):
    path = tmp_path / "calendar.json"
    path.write_text(mc.CALENDAR_PATH.read_text(encoding="utf-8").replace('"extra_closed_dates": {}', '"extra_closed_dates": {"2026-11-12": "a day of mourning"}'), encoding="utf-8")
    config = mc.load_calendar(path)
    s = mc.session_at(at("2026-11-12 10:00"), config)
    assert s.state == "closed" and s.reason == "a day of mourning" and s.next_open_utc == utc("2026-11-13T09:00:00Z")
    assert mc.previous_trading_day(date(2026, 11, 13), config) == date(2026, 11, 11)


def test_after_hours_before_a_closed_day_can_be_turned_on(tmp_path):
    path = tmp_path / "calendar.json"
    path.write_text(mc.CALENDAR_PATH.read_text(encoding="utf-8").replace('"after_hours_before_closed_day": false', '"after_hours_before_closed_day": true'), encoding="utf-8")
    assert mc.session_at(at("2026-11-13 16:00"), mc.load_calendar(path)).state == "after_hours"
    assert mc.session_at(at("2026-11-27 13:00"), mc.load_calendar(path)).state == "closed"       # an early-close day never has after-hours


def test_the_calendar_file_is_checked_when_it_is_read(tmp_path):
    good = mc.CALENDAR_PATH.read_text(encoding="utf-8")
    for name, text, message in [("missing.json", None, "does not exist"), ("bad.json", "{", "not valid JSON"), ("unknown.json", good.replace('"timezone"', '"timezne"'), "timezne"),
                                ("zone.json", good.replace("America/New_York", "Mars/Olympus"), "Mars/Olympus"),
                                ("order.json", good.replace('"regular_open": "09:30"', '"regular_open": "03:00"'), "pre_market_open < regular_open")]:
        path = tmp_path / name
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            mc.load_calendar(path)


def test_every_day_of_2026_is_either_closed_all_day_or_has_a_window_ending_at_one_of_three_times():
    day, ends = date(2026, 1, 1), set()
    while day.year == 2026:
        noon = mc.session_at(datetime(day.year, day.month, day.day, 10, 0, tzinfo=ET))
        if mc.is_trading_day(day):
            assert noon.state == "regular" and noon.trading_date == day
            ends.add(noon.window_closes_utc.astimezone(ET).strftime("%H:%M"))
        else:
            assert noon.state == "closed" and noon.trading_date is None and noon.reason
        day += timedelta(days=1)
    assert ends == {"13:00", "16:00", "20:00"}
