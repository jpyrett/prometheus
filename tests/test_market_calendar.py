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
PUBLISHED = {
    2026: ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"],
    2027: ["2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"],
    2028: ["2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25"],
}


@pytest.mark.parametrize("year", PUBLISHED)
def test_the_holiday_rules_give_the_nyses_published_list(year):
    assert sorted(d.isoformat() for d in mc.holidays(year)) == PUBLISHED[year]


def test_new_years_day_on_a_saturday_is_not_observed_and_december_31_stays_open():
    assert date(2028, 1, 1) not in mc.holidays(2028) and date(2027, 12, 31) not in mc.holidays(2027)
    assert mc.is_trading_day(date(2027, 12, 31))


def test_holiday_names_and_easter():
    assert mc.holidays(2026)[date(2026, 4, 3)] == "Good Friday" and mc.holidays(2026)[date(2026, 11, 26)] == "Thanksgiving Day"
    assert [mc.easter(y) for y in (2025, 2026, 2027, 2028)] == [date(2025, 4, 20), date(2026, 4, 5), date(2027, 3, 28), date(2028, 4, 16)]


def test_trading_days_skip_weekends_and_holidays():
    assert not mc.is_trading_day(date(2026, 11, 14)) and not mc.is_trading_day(date(2026, 11, 26)) and mc.is_trading_day(date(2026, 11, 25))
    assert mc.previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)      # Thanksgiving in between
    assert mc.previous_trading_day(date(2026, 11, 16)) == date(2026, 11, 13)      # the weekend in between
    assert mc.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27) and mc.next_trading_day(date(2026, 11, 13)) == date(2026, 11, 16)


@pytest.mark.parametrize("moment, state, trading_date, window_closes, next_open", [
    ("2026-11-11 10:00", "regular", date(2026, 11, 11), "2026-11-12T01:00:00Z", "2026-11-12T09:00:00Z"),       # the window closes at 20:00 Eastern
    ("2026-11-11 22:13", "closed", None, None, "2026-11-12T09:00:00Z"),                                        # after the after-hours close
    ("2026-11-16 03:59", "closed", None, None, "2026-11-16T09:00:00Z"),
    ("2026-11-16 04:00", "pre_market", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),
    ("2026-11-16 09:29", "pre_market", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),
    ("2026-11-16 09:30", "regular", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),
    ("2026-11-16 15:59", "regular", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),
    ("2026-11-16 16:00", "after_hours", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),   # Tuesday is a trading day: after-hours runs
    ("2026-11-16 19:59", "after_hours", date(2026, 11, 16), "2026-11-17T01:00:00Z", "2026-11-17T09:00:00Z"),
    ("2026-11-16 20:00", "closed", None, None, "2026-11-17T09:00:00Z"),
    ("2026-11-13 15:59", "regular", date(2026, 11, 13), "2026-11-13T21:00:00Z", "2026-11-16T09:00:00Z"),       # a Friday: the window ends at 16:00 (Jacob's rule)
    ("2026-11-13 16:00", "closed", None, None, "2026-11-16T09:00:00Z"),
    ("2026-11-14 12:00", "closed", None, None, "2026-11-16T09:00:00Z"),                                        # a Saturday
    ("2026-11-15 12:00", "closed", None, None, "2026-11-16T09:00:00Z"),                                        # a Sunday
    ("2026-11-25 15:59", "regular", date(2026, 11, 25), "2026-11-25T21:00:00Z", "2026-11-27T09:00:00Z"),       # the day before Thanksgiving: no after-hours
    ("2026-11-25 16:00", "closed", None, None, "2026-11-27T09:00:00Z"),
    ("2026-11-26 11:00", "closed", None, None, "2026-11-27T09:00:00Z"),                                        # Thanksgiving
    ("2026-11-27 12:59", "regular", date(2026, 11, 27), "2026-11-27T18:00:00Z", "2026-11-30T09:00:00Z"),       # an early close
    ("2026-11-27 13:00", "closed", None, None, "2026-11-30T09:00:00Z"),
    ("2026-07-03 10:00", "closed", None, None, "2026-07-06T08:00:00Z"),                                        # Independence Day, observed
    ("2026-07-01 10:00", "regular", date(2026, 7, 1), "2026-07-02T00:00:00Z", "2026-07-02T08:00:00Z"),         # daylight time: 20:00 Eastern is 00:00 UTC
    ("2026-07-02 17:00", "closed", None, None, "2026-07-06T08:00:00Z"),                                        # the day before the observed holiday: no after-hours
])
def test_the_session_of_a_moment(moment, state, trading_date, window_closes, next_open):
    s = mc.session_at(at(moment))
    assert (s.state, s.trading_date) == (state, trading_date)
    assert s.window_closes_utc == (utc(window_closes) if window_closes else None)
    assert s.next_open_utc == utc(next_open)
    assert s.is_open == (state != "closed")


@pytest.mark.parametrize("moment, reason", [
    ("2026-11-11 22:13", "after the after-hours close"), ("2026-11-16 03:59", "before the pre-market open"), ("2026-11-14 12:00", "weekend"),
    ("2026-11-26 11:00", "holiday: Thanksgiving Day"), ("2026-11-13 16:00", "after the regular close on a day before a closed day"),
    ("2026-11-27 13:00", "after the early close"), ("2026-07-03 10:00", "holiday: Independence Day")])
def test_a_closed_moment_says_why(moment, reason):
    assert mc.session_at(at(moment)).reason == reason


def test_an_open_moment_has_no_reason():
    assert mc.session_at(at("2026-11-11 10:00")).reason == ""


@pytest.mark.parametrize("moment, last", [("2026-11-11 22:13", date(2026, 11, 11)), ("2026-11-14 12:00", date(2026, 11, 13)), ("2026-11-26 11:00", date(2026, 11, 25)),
                                          ("2026-11-16 03:00", date(2026, 11, 16))])
def test_the_last_trading_date_is_the_newest_trading_day_on_or_before_the_easterns_date(moment, last):
    assert mc.session_at(at(moment)).last_trading_date == last


def test_the_session_is_the_same_whatever_zone_the_moment_is_written_in():
    assert mc.session_at(utc("2026-11-11T15:00:00Z")) == mc.session_at(at("2026-11-11 10:00"))


def test_the_clocks_change_in_november_and_march():
    # 2026-11-01: daylight time ended. 2026-03-08: it began. The Eastern 09:30 open is 14:30 UTC in winter and 13:30 UTC in summer.
    assert mc.session_at(utc("2026-11-02T14:30:00Z")).state == "regular" and mc.session_at(utc("2026-11-02T14:29:00Z")).state == "pre_market"
    assert mc.session_at(utc("2026-03-09T13:30:00Z")).state == "regular" and mc.session_at(utc("2026-03-09T13:29:00Z")).state == "pre_market"


def test_the_eastern_date_is_not_the_utc_date_in_the_evening():
    moment = utc("2026-11-12T00:30:00Z")           # 19:30 Eastern on the 11th
    assert mc.eastern_date(moment) == date(2026, 11, 11) and moment.date() == date(2026, 11, 12)


def test_a_naive_moment_raises():
    with pytest.raises(ValueError, match="timezone"):
        mc.session_at(datetime(2026, 11, 11, 10, 0))
    with pytest.raises(ValueError, match="timezone"):
        mc.eastern_date(datetime(2026, 11, 11, 10, 0))


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
