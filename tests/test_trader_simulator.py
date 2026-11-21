"""The Trader's fill rules and exit rules (prometheus/agents/trader/simulator.py, Trader spec sections 6 and 10): every rule, each at its boundary. Pure functions: no network, no model, no clock."""
from decimal import Decimal
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from prometheus.agents.trader import simulator as sim
from prometheus.agents.trader.record import Outcome

ET = ZoneInfo("America/New_York")
LOW, HIGH = Decimal("95.0"), Decimal("100.05")          # the limit range of the tests below: 100.05 is exactly 100 plus 5 basis points of slippage, so a market buy at 100 lands ON the top of the range


def bar(day, o, h, l, c, v=1000):
    return {"date": day, "open": o, "high": h, "low": l, "close": c, "volume": v}


def at(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ET)


# ---------------------------------------------------------------- no order fills outside the range, and bad input

@pytest.mark.parametrize("side, order_type, limit_price", [("buy", "limit", Decimal("94.99")), ("buy", "limit", Decimal("100.06")), ("sell", "limit", Decimal("94.99")), ("sell", "limit", Decimal("100.06"))], ids=str)
def test_a_limit_price_outside_the_range_is_rejected_by_the_rules_even_if_a_tool_let_it_through(side, order_type, limit_price):
    for run in (lambda: sim.fill_paper(side, order_type, limit_price, Decimal("97.0"), LOW, HIGH, 5), lambda: sim.fill_backtest(side, order_type, limit_price, bar("2026-11-06", 97, 99, 96, 98), LOW, HIGH, 5)):
        e = run()
        assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "outside the range" in e.reason


@pytest.mark.parametrize("arguments", [("short", "market", None, Decimal("100.0"), LOW, HIGH, 5), ("buy", "stop", None, Decimal("100.0"), LOW, HIGH, 5), ("buy", "limit", None, Decimal("100.0"), LOW, HIGH, 5), ("buy", "limit", Decimal("0.0"), Decimal("100.0"), LOW, HIGH, 5)])
def test_a_malformed_order_is_an_internal_error_not_an_exception(arguments):
    e = sim.fill_paper(*arguments)
    assert (e.status, e.reject_code) == ("rejected", "internal_error")


@pytest.mark.parametrize("low, high", [(Decimal("100.0"), Decimal("99.0")), (None, Decimal("100.0")), (Decimal("95.0"), 0), (Decimal("NaN"), Decimal("100.0"))], ids=["100.0-99.0", "None-100.0", "95.0-0", "nan-100.0"])
def test_a_trade_with_no_price_range_is_rejected_outside_limits(low, high):
    assert sim.fill_paper("buy", "market", None, Decimal("97.0"), low, high, 5).reject_code == "price_outside_limits"
    assert sim.fill_backtest("buy", "market", None, bar("2026-11-06", 97, 99, 96, 98), low, high, 5).reject_code == "price_outside_limits"


# ---------------------------------------------------------------- backtest: a sale


@pytest.mark.parametrize("entry", [None, {}, {"date": "2026-11-06", "open": 0, "high": 1, "low": 1}, {"date": "2026-11-06", "open": 100, "high": None, "low": 99}, "bar"])
def test_an_unusable_entry_bar_is_price_unavailable(entry):
    assert sim.fill_backtest("buy", "market", None, entry, LOW, HIGH, 5).reject_code == "price_unavailable"


# ---------------------------------------------------------------- the entry date

@pytest.mark.parametrize("now, expected", [
    ("2026-11-05 10:30", date(2026, 11, 6)),        # a decision at 10:30 on a Thursday: Friday
    ("2026-11-05 08:00", date(2026, 11, 5)),        # at 08:00 on a Thursday: that Thursday's open is still ahead
    ("2026-11-05 09:29", date(2026, 11, 5)),
    ("2026-11-05 09:30", date(2026, 11, 6)),        # exactly the open: the open is not STRICTLY after the decision
    ("2026-11-05 23:59", date(2026, 11, 6)),
    ("2026-11-06 10:30", date(2026, 11, 9)),        # a Friday: Monday
    ("2026-11-07 12:00", date(2026, 11, 9)),        # a Saturday
    ("2026-11-08 12:00", date(2026, 11, 9)),        # a Sunday
    ("2026-11-26 08:00", date(2026, 11, 27)),       # Thanksgiving: the market is closed all day
    ("2026-11-25 15:00", date(2026, 11, 27)),       # the day before: the next open is after the holiday
    ("2026-11-27 12:00", date(2026, 11, 30)),       # the day after Thanksgiving closes early, the open was 09:30 all the same
])
def test_the_calendar_entry_day_is_the_first_trading_day_whose_open_is_strictly_after_now(now, expected):
    assert sim.calendar_entry_day(at(now)) == expected


# ---------------------------------------------------------------- following a purchase

def series(*rows, start="2026-11-06"):
    """Bars on consecutive weekdays from `start`: (open, high, low, close) rows."""
    day, out = date.fromisoformat(start), []
    for row in rows:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        out.append(bar(day.isoformat(), *row))
        day += timedelta(days=1)
    return out


QUIET = (100.0, 101.0, 99.0, 100.0)
PLAN = dict(stop_price=Decimal("95.0"), target_price=Decimal("110.0"), time_stop_date="2026-12-31", slippage_bps=5)


def follow(*rows, fill=Decimal("100.0"), shares=Decimal(10), **changes):
    return sim.follow(fill, shares, changes.pop("entry", "2026-11-06"), series(*rows), **{**PLAN, **changes})


@pytest.mark.parametrize("bars", [[], None, [bar("2026-11-09", *QUIET)], [bar("2026-11-05", *QUIET)], [{"date": "2026-11-06"}], ["x"]])
def test_a_series_that_does_not_start_with_the_entry_bar_cannot_be_followed_and_says_nothing(bars):
    o = sim.follow(Decimal("100.0"), Decimal(10), "2026-11-06", bars, **PLAN)
    assert o == Outcome(status="open")
