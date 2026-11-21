"""Timing: has the recommendation expired? (Critic spec 8.3.) The calendar itself is tested in test_market_calendar.py; here is what the Critic makes of it."""
from datetime import datetime, timedelta, timezone

import pytest
from critic_helpers import CONFIG, NOW, at, captured, rec_of, tamper

from prometheus.agents.critic import timing as tm

T = CONFIG.timing


def quote(price=167.1, now=NOW, minutes=1, **kw):
    stamp = (now - timedelta(minutes=minutes)).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {"price": price, "previous_close": price, "as_of_utc": stamp, **kw}


def close(value=167.1, as_of="2026-11-05"):
    return {"value": value, "as_of": as_of}


def rec(**changes):
    order = captured("PG_open")
    if changes:
        order.recommendations[0] = tamper(order.recommendations[0], **changes)
    return rec_of(order)


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


def test_the_time_stop_passing_is_a_blocker():
    r = tm.assess(rec(**{"trade.time_stop_date": "2026-11-04"}), NOW, quote(), close(), T)
    assert "time_stop_passed" in codes(r, "blocker")
    assert "time_stop_passed" not in codes(tm.assess(rec(**{"trade.time_stop_date": "2026-11-05"}), NOW, quote(), close(), T))       # the day itself is still inside


def test_a_price_above_the_entry_zone_is_chasing_and_a_blocker():
    r = tm.assess(rec(), NOW, quote(169.0), close(), T)
    [f] = [f for f in r.findings if f.code == "price_left_zone"]
    assert f.severity == "blocker" and "169.00" in f.message and r.detail.zone_position > 1 and r.score == 50       # session 1.0, zone 0
    assert "price_left_zone" not in codes(tm.assess(rec(), NOW, quote(168.7441), close(), T))      # the top itself is inside


@pytest.mark.parametrize("stamp, day", [("2026-11-05T00:00:00Z", "2026-11-05"), ("2026-11-06T01:30:00Z", "2026-11-05"), ("2026-11-05T15:00:00Z", "2026-11-05"), ("garbage", None)])
def test_the_record_day_is_the_date_it_was_given_or_the_easterns_date_of_its_moment(stamp, day):
    r = rec(**{"identity.as_of_utc": stamp})
    assert (str(tm.record_day(r, NOW)) if tm.record_day(r, NOW) else None) == day


def test_ignore_market_hours_skips_the_closed_market_blocker_and_nothing_else(monkeypatch):
    """IGNORE_MARKET_HOURS (Jacob, 2026-11-12): a closed market is noted, not a denial. A recommendation from an earlier trading day is still expired."""
    now = at("2026-11-12 01:40")                                          # a Thursday, before the pre-market open: the record of that day is still the one that can be acted on
    closed = tm.assess(rec(**{"identity.as_of_utc": "2026-11-12T00:00:00Z"}), now, quote(now=now), close(as_of="2026-11-11"), T)
    assert "market_closed" in codes(closed, "blocker")
    skipped = tm.assess(rec(**{"identity.as_of_utc": "2026-11-12T00:00:00Z"}), now, quote(now=now), close(as_of="2026-11-11"), T, ignore_market_hours=True)
    assert "market_closed" not in codes(skipped) and codes(skipped, "info") == ["market_hours_ignored"] and "IGNORE_MARKET_HOURS" in skipped.findings[0].message
    weekend = at("2026-11-07 12:00")
    still = tm.assess(rec(), weekend, quote(now=weekend), close(), T, ignore_market_hours=True)
    assert "market_closed" not in codes(still) and "record_expired" in codes(still, "blocker")
    for value, ignored in (("true", True), ("YES", True), ("1", True), ("on", True), ("false", False), ("no", False)):
        monkeypatch.setenv("IGNORE_MARKET_HOURS", value)
        assert tm.market_hours_ignored() is ignored, value
