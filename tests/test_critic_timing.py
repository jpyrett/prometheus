"""Timing: has the recommendation expired? (Critic spec 8.3.) The calendar itself is tested in test_market_calendar.py; here is what the Critic makes of it."""
from datetime import datetime, timedelta, timezone

import pytest
from critic_helpers import CONFIG, NOW, at, captured, rec_of, tamper

from prometheus.agents.critic import timing as tm
from prometheus.common.market_calendar import session_at

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


@pytest.mark.parametrize("as_of, stale", [("2026-11-04", False), ("2026-11-03", True), ("2026-10-20", True)])
def test_the_stored_bars_must_reach_the_session_before_the_last_one(as_of, stale):
    assert ("bars_stale" in codes(tm.assess(rec(), NOW, quote(), close(as_of=as_of), T), "blocker")) is stale


def test_no_bars_at_all_is_stale_bars():
    r = tm.assess(rec(), NOW, quote(), {"value": None, "as_of": None}, T)
    assert "bars_stale" in codes(r, "blocker") and r.deal_price == 167.1
    assert "bars_stale" in codes(tm.assess(rec(), NOW, quote(), None, T), "blocker")


def test_no_quote_and_no_bars_leave_no_deal_price_at_all():
    r = tm.assess(rec(), NOW, None, None, T)
    assert r.deal_price is None and r.price_basis is None and r.score < 100


def test_outside_the_regular_session_only_the_date_of_the_quote_matters():
    now = at("2026-11-05 05:00")                                  # pre-market: a quote from yesterday's close is fine, one from last week is not
    assert "quote_stale" not in codes(tm.assess(rec(), now, quote(now=now, minutes=600), close(as_of="2026-11-04"), T))
    assert "quote_stale" in codes(tm.assess(rec(), now, quote(now=now, minutes=60 * 24 * 7), close(as_of="2026-11-04"), T), "blocker")


def test_a_price_above_the_entry_zone_is_chasing_and_a_blocker():
    r = tm.assess(rec(), NOW, quote(169.0), close(), T)
    [f] = [f for f in r.findings if f.code == "price_left_zone"]
    assert f.severity == "blocker" and "169.00" in f.message and r.detail.zone_position > 1 and r.score == 50       # session 1.0, zone 0
    assert "price_left_zone" not in codes(tm.assess(rec(), NOW, quote(168.7441), close(), T))      # the top itself is inside


def test_a_price_under_the_zone_is_a_warning_and_at_the_stop_a_blocker():
    r = tm.assess(rec(), NOW, quote(160.0), close(), T)
    assert codes(r) == ["price_below_zone"] and r.findings[0].severity == "warning"
    r = tm.assess(rec(), NOW, quote(157.2351), close(), T)
    assert codes(r) == ["stop_already_hit"] and r.findings[0].severity == "blocker"


def test_extended_hours_are_noted_and_score_lower():
    r = tm.assess(rec(), at("2026-11-05 05:00"), quote(now=at("2026-11-05 05:00")), close(as_of="2026-11-04"), T)
    assert codes(r) == ["extended_hours"] and r.findings[0].severity == "info" and r.score == pytest.approx(100 * (0.5 * 0.7 + 0.5))
    r = tm.assess(rec(), at("2026-11-05 17:00"), quote(now=at("2026-11-05 17:00")), close(), T)
    assert r.session.state == "after_hours" and "extended_hours" in codes(r)


@pytest.mark.parametrize("price, factor", [(165.0, T.below_zone_factor), (165.4559, 1.0), (167.1, 1.0), (167.9220, 0.75), (168.7441, T.top_of_zone_factor), (170.0, 0.0), (None, 0.0)])
def test_the_zone_factor_falls_from_the_mid_to_the_top_and_is_zero_above_it(price, factor):
    assert tm.zone_factor(price, [165.4559, 168.7441], 167.1, T) == pytest.approx(factor, abs=1e-3)


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
