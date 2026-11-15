"""The Trader's fill rules and exit rules (prometheus/agents/trader/simulator.py, Trader spec sections 6 and 10): every rule, each at its boundary. Pure functions: no network, no model, no clock."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from prometheus.agents.trader import simulator as sim
from prometheus.agents.trader.record import REJECT_CODES, Outcome

ET = ZoneInfo("America/New_York")
LOW, HIGH = 95.0, 100.05          # the limit range of the tests below: 100.05 is exactly 100 plus 5 basis points of slippage, so a market buy at 100 lands ON the top of the range


def bar(day, o, h, l, c, v=1000):
    return {"date": day, "open": o, "high": h, "low": l, "close": c, "volume": v}


def at(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ET)


# ---------------------------------------------------------------- paper: a buy


# ---------------------------------------------------------------- paper: a sale (the mirror image)


# ---------------------------------------------------------------- no order fills outside the range, and bad input

@pytest.mark.parametrize("side, order_type, limit_price", [("buy", "limit", 94.99), ("buy", "limit", 100.06), ("sell", "limit", 94.99), ("sell", "limit", 100.06)])
def test_a_limit_price_outside_the_range_is_rejected_by_the_rules_even_if_a_tool_let_it_through(side, order_type, limit_price):
    for run in (lambda: sim.fill_paper(side, order_type, limit_price, 97.0, LOW, HIGH, 5), lambda: sim.fill_backtest(side, order_type, limit_price, bar("2026-11-06", 97, 99, 96, 98), LOW, HIGH, 5)):
        e = run()
        assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "outside the range" in e.reason


@pytest.mark.parametrize("arguments", [("short", "market", None, 100.0, LOW, HIGH, 5), ("buy", "stop", None, 100.0, LOW, HIGH, 5), ("buy", "limit", None, 100.0, LOW, HIGH, 5), ("buy", "limit", 0.0, 100.0, LOW, HIGH, 5)])
def test_a_malformed_order_is_an_internal_error_not_an_exception(arguments):
    e = sim.fill_paper(*arguments)
    assert (e.status, e.reject_code) == ("rejected", "internal_error")


@pytest.mark.parametrize("low, high", [(100.0, 99.0), (None, 100.0), (95.0, 0), (float("nan"), 100.0)])
def test_a_trade_with_no_price_range_is_rejected_outside_limits(low, high):
    assert sim.fill_paper("buy", "market", None, 97.0, low, high, 5).reject_code == "price_outside_limits"
    assert sim.fill_backtest("buy", "market", None, bar("2026-11-06", 97, 99, 96, 98), low, high, 5).reject_code == "price_outside_limits"


def test_every_code_the_rules_use_is_in_the_closed_list():
    codes = {sim.fill_paper("buy", "market", None, 150.0, LOW, HIGH, 5).reject_code, sim.fill_paper("buy", "market", None, None, LOW, HIGH, 5).reject_code, sim.fill_paper("buy", "stop", None, 1.0, LOW, HIGH, 5).reject_code,
             sim.fill_backtest("buy", "limit", 96.0, bar("d", 99, 100, 98, 99), LOW, HIGH, 5).reject_code}
    assert codes <= set(REJECT_CODES) and "limit_not_reached" in codes


# ---------------------------------------------------------------- backtest: a buy against the entry bar


def test_a_backtest_market_buy_is_rejected_when_the_open_gapped_above_the_range_even_if_the_day_came_back():
    e = sim.fill_backtest("buy", "market", None, bar("2026-11-06", 101.0, 101.5, 96.0, 97.0), LOW, HIGH, 5)
    assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "the opening price" in e.reason


def test_a_backtest_limit_buy_above_the_open_fills_at_its_limit_when_the_low_touches_it_exactly_and_not_a_cent_short():
    e = sim.fill_backtest("buy", "limit", 100.0, bar("2026-11-06", 101.0, 102.0, 100.0, 101.5), LOW, HIGH, 5)
    assert (e.status, e.price, e.price_source, e.slippage_bps) == ("filled", 100.0, "bar_limit", 0.0) and "traded down to 100.0000" in e.reason
    e = sim.fill_backtest("buy", "limit", 100.0, bar("2026-11-06", 101.0, 102.0, 100.01, 101.5), LOW, HIGH, 5)
    assert (e.status, e.reject_code, e.price) == ("rejected", "limit_not_reached", None) and "day order" in e.reason


# ---------------------------------------------------------------- backtest: a sale

def test_a_backtest_market_sale_fills_at_the_open_less_slippage_and_is_rejected_when_the_open_gapped_below_the_range():
    e = sim.fill_backtest("sell", "market", None, bar("2026-11-06", 100.0, 103.0, 99.0, 102.0), 99.95, 103.0, 5)
    assert (e.status, e.price, e.price_source) == ("filled", 99.95, "bar_open")
    e = sim.fill_backtest("sell", "market", None, bar("2026-11-06", 99.0, 103.0, 98.0, 102.0), 99.95, 103.0, 5)
    assert (e.status, e.reject_code) == ("rejected", "price_outside_limits")


def test_a_backtest_limit_sale_fills_at_the_open_when_it_is_at_or_over_the_limit_and_at_the_limit_when_the_high_touches_it():
    e = sim.fill_backtest("sell", "limit", 100.0, bar("2026-11-06", 101.0, 102.0, 99.0, 100.0), 99.0, 103.0, 5)
    assert (e.status, e.price, e.price_source) == ("filled", 100.9495, "bar_open")                                 # 101 less 5 bps is 100.9495, over the limit
    e = sim.fill_backtest("sell", "limit", 100.0, bar("2026-11-06", 99.0, 100.0, 98.0, 99.5), 99.0, 103.0, 5)
    assert (e.status, e.price, e.price_source) == ("filled", 100.0, "bar_limit")                                    # the high touched the limit exactly
    e = sim.fill_backtest("sell", "limit", 100.0, bar("2026-11-06", 99.0, 99.99, 98.0, 99.5), 99.0, 103.0, 5)
    assert (e.status, e.reject_code) == ("rejected", "limit_not_reached")


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


def test_the_entry_day_uses_the_easterns_clock_not_utc_and_a_naive_moment_is_a_programming_mistake():
    assert sim.calendar_entry_day(datetime(2026, 11, 5, 15, 30, tzinfo=timezone.utc)) == date(2026, 11, 6)       # 15:30 UTC is 10:30 Eastern
    assert sim.calendar_entry_day(datetime(2026, 11, 5, 13, 0, tzinfo=timezone.utc)) == date(2026, 11, 5)        # 13:00 UTC is 08:00 Eastern: the same day
    assert sim.calendar_entry_day(datetime(2026, 11, 6, 3, 0, tzinfo=timezone.utc)) == date(2026, 11, 6)         # 03:00 UTC on the 6th is 22:00 Eastern on the 5th: the 6th is the next session
    with pytest.raises(ValueError, match="timezone"):
        sim.calendar_entry_day(datetime(2026, 11, 5, 10, 30))
    with pytest.raises(ValueError, match="timezone"):
        sim.entry_date(datetime(2026, 11, 5, 10, 30), {})


def test_with_no_bar_on_or_after_the_calendar_day_there_is_no_entry_date_and_the_future_is_not_guessed():
    now = at("2026-11-05 10:30")
    assert sim.entry_date(now, {}) is None
    assert sim.entry_date(now, {"PG": [], "SPY": []}) is None
    assert sim.entry_date(now, {"PG": [bar("2026-11-05", 1, 1, 1, 1), bar("2026-11-04", 1, 1, 1, 1)]}) is None    # bars from before the entry day are the past, not an entry
    assert sim.entry_date(now, {"PG": [{"date": "garbage"}, "x", {"open": 1}]}) is None


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
PLAN = dict(stop_price=95.0, target_price=110.0, time_stop_date="2026-12-31", slippage_bps=5)


def follow(*rows, fill=100.0, shares=10, **changes):
    return sim.follow(fill, shares, changes.pop("entry", "2026-11-06"), series(*rows), **{**PLAN, **changes})


def test_a_stop_touched_exactly_on_a_later_bar_exits_at_the_stop_less_slippage_and_a_cent_above_does_not():
    o = follow(QUIET, (100.0, 101.0, 95.0, 97.0))
    assert (o.status, o.exit_reason, o.exit_date, o.exit_price, o.sessions_held) == ("closed", "stop", "2026-11-09", 94.9525, 1)
    assert follow(QUIET, (100.0, 101.0, 95.01, 97.0)).status == "open"


def test_a_target_touched_exactly_exits_at_the_target_and_a_gap_over_it_exits_at_the_open_without_slippage():
    o = follow(QUIET, (101.0, 110.0, 100.0, 108.0))
    assert (o.exit_reason, o.exit_price, o.exit_date, o.sessions_held) == ("target", 110.0, "2026-11-09", 1)
    assert follow(QUIET, (101.0, 109.99, 100.0, 108.0)).status == "open"
    o = follow(QUIET, (111.0, 112.0, 110.5, 111.5))
    assert (o.exit_reason, o.exit_price) == ("target", 111.0)


def test_the_time_stop_exits_at_the_close_less_slippage_on_the_date_and_on_any_bar_after_it_but_not_before():
    o = follow(QUIET, (100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0), time_stop_date="2026-11-10")
    assert (o.status, o.exit_reason, o.exit_date, o.exit_price, o.sessions_held) == ("closed", "time_stop", "2026-11-10", 99.95, 2)
    assert follow(QUIET, QUIET, QUIET, QUIET, time_stop_date="2026-11-11").exit_date == "2026-11-11"
    assert follow(QUIET, QUIET, time_stop_date="2026-11-11").status == "open"                                    # the date has not come in the stored bars
    skipped = sim.follow(100.0, 10, "2026-11-06", [bar("2026-11-06", *QUIET), bar("2026-11-13", *QUIET)], **{**PLAN, "time_stop_date": "2026-11-10"})
    assert skipped.exit_date == "2026-11-13" and skipped.exit_reason == "time_stop"                              # a bar after the date still ends it (a missing day does not skip the rule)


def test_on_the_entry_bar_only_the_stop_can_fire():
    big_day = (100.0, 120.0, 99.0, 118.0)                                                                        # the high is over the target
    assert follow(big_day).status == "open"                                                                       # a target on the entry bar is not taken: the order of the day is not known
    assert follow(QUIET, time_stop_date="2026-11-06").status == "open"                                            # nor the time stop
    o = follow((100.0, 105.0, 94.0, 99.0))
    assert (o.status, o.exit_reason, o.exit_date, o.sessions_held, o.exit_price) == ("closed", "stop", "2026-11-06", 0, 94.9525)
    assert follow((94.0, 96.0, 93.0, 95.0)).exit_price == 93.953                                                 # an entry bar that opened under the stop is stopped at its open


def test_the_first_rule_that_holds_wins_on_one_bar_target_before_the_time_stop_and_stop_before_both():
    o = follow(QUIET, (111.0, 112.0, 110.0, 111.0), time_stop_date="2026-11-09")
    assert o.exit_reason == "target"
    o = follow(QUIET, (94.0, 112.0, 93.0, 111.0), time_stop_date="2026-11-09")
    assert o.exit_reason == "stop"


def test_slippage_is_taken_off_the_stop_and_the_time_stop_exits_only():
    assert follow(QUIET, (100.0, 101.0, 94.0, 96.0), slippage_bps=0).exit_price == 95.0
    assert follow(QUIET, (100.0, 101.0, 94.0, 96.0), slippage_bps=100).exit_price == 94.05
    assert follow(QUIET, (111.0, 112.0, 110.5, 111.5), slippage_bps=100).exit_price == 111.0                      # a target is taken at the price, no slippage
    assert follow(QUIET, QUIET, time_stop_date="2026-11-09", slippage_bps=100).exit_price == 99.0


@pytest.mark.parametrize("bars", [[], None, [bar("2026-11-09", *QUIET)], [bar("2026-11-05", *QUIET)], [{"date": "2026-11-06"}], ["x"]])
def test_a_series_that_does_not_start_with_the_entry_bar_cannot_be_followed_and_says_nothing(bars):
    o = sim.follow(100.0, 10, "2026-11-06", bars, **PLAN)
    assert o == Outcome(status="open")


# ---------------------------------------------------------------- the benchmark


