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

def test_a_paper_market_buy_fills_at_the_quote_plus_slippage_and_says_where_the_price_came_from():
    e = sim.fill_paper("buy", "market", None, 100.0, LOW, HIGH, 5)
    assert (e.status, e.price, e.price_source, e.slippage_bps, e.reject_code) == ("filled", 100.05, "quote", 5.0, None)
    assert sim.fill_paper("buy", "market", None, 100.0, LOW, HIGH, 0).price == 100.0 and sim.fill_paper("buy", "market", None, 100.0, LOW, HIGH, 0).slippage_bps == 0.0


def test_a_paper_market_buy_is_rejected_when_its_price_is_even_a_hundredth_of_a_cent_above_the_range():
    assert sim.fill_paper("buy", "market", None, 100.0, LOW, 100.05, 5).status == "filled"                      # the price equal to the top of the range is inside it
    e = sim.fill_paper("buy", "market", None, 100.0, LOW, 100.0499, 5)
    assert (e.status, e.reject_code, e.price) == ("rejected", "price_outside_limits", None) and "100.0500" in e.reason and "100.0499" in e.reason and "new review" in e.reason
    assert sim.fill_paper("buy", "market", None, 150.0, LOW, HIGH, 5).reject_code == "price_outside_limits"      # a quote far above: no order fills there


def test_a_paper_market_buy_far_below_the_range_still_fills_a_better_price_is_not_outside_what_the_critic_judged():
    e = sim.fill_paper("buy", "market", None, 90.0, LOW, HIGH, 5)
    assert e.status == "filled" and e.price == 90.045


def test_a_paper_limit_buy_fills_when_the_quote_equals_the_limit_and_rests_a_cent_above_it():
    e = sim.fill_paper("buy", "limit", 100.0, 100.0, LOW, HIGH, 5)
    assert (e.status, e.price, e.price_source) == ("filled", 100.0, "quote")                                    # min(limit, quote + slippage): the limit is the ceiling
    e = sim.fill_paper("buy", "limit", 100.0, 100.01, LOW, HIGH, 5)
    assert (e.status, e.price, e.reject_code) == ("active", None, None) and "above the limit" in e.reason


def test_a_paper_limit_buy_never_pays_more_than_its_limit_and_takes_the_better_price_when_there_is_one():
    assert sim.fill_paper("buy", "limit", 100.04, 100.0, LOW, HIGH, 5).price == 100.04                           # slippage would make it 100.05: the limit holds
    assert sim.fill_paper("buy", "limit", 100.0, 99.0, LOW, HIGH, 5).price == 99.0495                             # the quote is below the limit: the quote plus slippage
    e = sim.fill_paper("buy", "limit", 100.0, 99.0, LOW, HIGH, 5)
    assert e.slippage_bps == 5.0
    capped = sim.fill_paper("buy", "limit", 100.0, 99.99, LOW, HIGH, 5)                                            # 99.99 plus 5 bps is 100.04: over the limit, so the limit is the price
    assert (capped.price, capped.slippage_bps) == (100.0, 1.0)                                                     # and the slippage it really had is the 1 basis point between the quote and the limit


# ---------------------------------------------------------------- paper: a sale (the mirror image)

def test_a_paper_market_sale_fills_at_the_quote_minus_slippage_and_is_rejected_below_the_range():
    e = sim.fill_paper("sell", "market", None, 100.0, 99.95, 103.0, 5)
    assert (e.status, e.price, e.slippage_bps) == ("filled", 99.95, 5.0)                                          # exactly the bottom of the range: inside
    e = sim.fill_paper("sell", "market", None, 100.0, 99.9501, 103.0, 5)
    assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "below the lowest price allowed" in e.reason


def test_a_paper_limit_sale_fills_when_the_quote_reaches_the_limit_and_rests_below_it():
    e = sim.fill_paper("sell", "limit", 100.0, 100.0, 99.0, 103.0, 5)
    assert (e.status, e.price) == ("filled", 100.0)                                                                # max(limit, quote less slippage): the limit is the floor
    assert sim.fill_paper("sell", "limit", 100.0, 101.0, 99.0, 103.0, 5).price == 100.9495                         # the quote is over the limit: the quote less slippage
    e = sim.fill_paper("sell", "limit", 100.0, 99.99, 99.0, 103.0, 5)
    assert e.status == "active" and "below the limit" in e.reason


# ---------------------------------------------------------------- no order fills outside the range, and bad input

@pytest.mark.parametrize("side, order_type, limit_price", [("buy", "limit", 94.99), ("buy", "limit", 100.06), ("sell", "limit", 94.99), ("sell", "limit", 100.06)])
def test_a_limit_price_outside_the_range_is_rejected_by_the_rules_even_if_a_tool_let_it_through(side, order_type, limit_price):
    for run in (lambda: sim.fill_paper(side, order_type, limit_price, 97.0, LOW, HIGH, 5), lambda: sim.fill_backtest(side, order_type, limit_price, bar("2026-11-06", 97, 99, 96, 98), LOW, HIGH, 5)):
        e = run()
        assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "outside the range" in e.reason


@pytest.mark.parametrize("quote", [None, 0, -1.0, float("nan"), float("inf"), "100", True])
def test_a_quote_that_cannot_be_traded_on_is_price_unavailable(quote):
    assert sim.fill_paper("buy", "market", None, quote, LOW, HIGH, 5).reject_code == "price_unavailable"


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

def test_a_backtest_market_buy_fills_at_the_open_plus_slippage_on_the_bars_date():
    e = sim.fill_backtest("buy", "market", None, bar("2026-11-06", 100.0, 103.0, 99.0, 102.0), LOW, HIGH, 5)
    assert (e.status, e.price, e.price_source, e.slippage_bps) == ("filled", 100.05, "bar_open", 5.0) and "2026-11-06" in e.reason


def test_a_backtest_market_buy_is_rejected_when_the_open_gapped_above_the_range_even_if_the_day_came_back():
    e = sim.fill_backtest("buy", "market", None, bar("2026-11-06", 101.0, 101.5, 96.0, 97.0), LOW, HIGH, 5)
    assert (e.status, e.reject_code) == ("rejected", "price_outside_limits") and "the opening price" in e.reason


def test_a_backtest_limit_buy_when_the_open_is_at_or_under_the_limit_fills_at_the_open_price_but_never_above_the_limit():
    e = sim.fill_backtest("buy", "limit", 100.0, bar("2026-11-06", 99.0, 101.0, 98.0, 100.0), LOW, HIGH, 5)
    assert (e.status, e.price, e.price_source) == ("filled", 99.0495, "bar_open")
    e = sim.fill_backtest("buy", "limit", 100.0, bar("2026-11-06", 100.0, 101.0, 98.0, 100.0), LOW, HIGH, 5)       # open == limit: reached; open plus slippage would be 100.05, so the limit caps it
    assert (e.status, e.price, e.price_source) == ("filled", 100.0, "bar_limit")
    e = sim.fill_backtest("buy", "limit", 100.0, bar("2026-11-06", 99.99, 101.0, 98.0, 100.0), LOW, HIGH, 5)       # the open is under the limit, but 99.99 plus 5 bps is over it: the limit is the price
    assert (e.price, e.price_source, e.slippage_bps) == (100.0, "bar_limit", 1.0)


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


def test_the_entry_date_is_the_earliest_stored_bar_on_or_after_that_day_among_every_symbol():
    now = at("2026-11-05 10:30")                                                                                  # the calendar says Friday the 6th
    both = {"PG": [bar("2026-11-05", 1, 1, 1, 1), bar("2026-11-06", 1, 1, 1, 1)], "SPY": [bar("2026-11-06", 1, 1, 1, 1)]}
    assert sim.entry_date(now, both) == date(2026, 11, 6)
    only_one = {"PG": [bar("2026-11-09", 1, 1, 1, 1)], "SPY": [bar("2026-11-06", 1, 1, 1, 1), bar("2026-11-09", 1, 1, 1, 1)]}
    assert sim.entry_date(now, only_one) == date(2026, 11, 6)                                                     # a symbol whose Friday file is missing does not move the run, another symbol has it
    missing = {"PG": [bar("2026-11-09", 1, 1, 1, 1)], "SPY": [bar("2026-11-09", 1, 1, 1, 1)]}
    assert sim.entry_date(now, missing) == date(2026, 11, 9)                                                      # the day's file is missing for everybody: the next stored day


def test_with_no_bar_on_or_after_the_calendar_day_there_is_no_entry_date_and_the_future_is_not_guessed():
    now = at("2026-11-05 10:30")
    assert sim.entry_date(now, {}) is None
    assert sim.entry_date(now, {"PG": [], "SPY": []}) is None
    assert sim.entry_date(now, {"PG": [bar("2026-11-05", 1, 1, 1, 1), bar("2026-11-04", 1, 1, 1, 1)]}) is None    # bars from before the entry day are the past, not an entry
    assert sim.entry_date(now, {"PG": [{"date": "garbage"}, "x", {"open": 1}]}) is None


def test_bar_on_finds_a_day_by_date_or_by_text():
    series = [bar("2026-11-05", 1, 2, 0.5, 1.5), bar("2026-11-06", 2, 3, 1.5, 2.5)]
    assert sim.bar_on(series, "2026-11-06")["open"] == 2 and sim.bar_on(series, date(2026, 11, 5))["open"] == 1 and sim.bar_on(series, "2026-11-07") is None and sim.bar_on(None, "2026-11-05") is None


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


def test_a_gap_through_the_stop_exits_at_the_open_less_slippage_not_at_the_stop():
    o = follow(QUIET, (94.0, 96.0, 93.0, 95.5))
    assert (o.exit_reason, o.exit_price) == ("stop", 93.953)                                                    # 94 less 5 bps is 93.953: worse than the stop, as a gap is
    o = follow(QUIET, (95.0, 96.0, 94.0, 95.5))                                                                  # an open exactly at the stop counts as through it
    assert (o.exit_reason, o.exit_price) == ("stop", 94.9525)
    assert sim.follow(100.0, 10, "2026-11-06", series(QUIET, (96.0, 97.0, 95.01, 96.0)), **PLAN).status == "open"


def test_a_target_touched_exactly_exits_at_the_target_and_a_gap_over_it_exits_at_the_open_without_slippage():
    o = follow(QUIET, (101.0, 110.0, 100.0, 108.0))
    assert (o.exit_reason, o.exit_price, o.exit_date, o.sessions_held) == ("target", 110.0, "2026-11-09", 1)
    assert follow(QUIET, (101.0, 109.99, 100.0, 108.0)).status == "open"
    o = follow(QUIET, (111.0, 112.0, 110.5, 111.5))
    assert (o.exit_reason, o.exit_price) == ("target", 111.0)


def test_a_day_that_touches_both_the_stop_and_the_target_is_a_stop_the_bad_case_is_assumed():
    o = follow(QUIET, (100.0, 112.0, 94.0, 100.0))
    assert o.exit_reason == "stop" and o.exit_price == 94.9525 and o.realized_gain < 0


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


def test_a_trade_with_no_exit_by_the_last_stored_bar_is_open_and_marked_at_the_last_close():
    o = follow(QUIET, (100.0, 103.0, 99.0, 102.5), (102.0, 104.0, 101.0, 103.25))
    assert (o.status, o.exit_reason, o.exit_date, o.exit_price, o.realized_gain) == ("open", None, None, None, None)
    assert (o.last_date, o.last_close, o.sessions_held, o.unrealized_gain, o.return_pct) == ("2026-11-10", 103.25, 2, 32.5, 0.0325)


def test_a_one_bar_series_is_an_open_trade_with_no_sessions_held():
    o = follow(QUIET)
    assert (o.status, o.sessions_held, o.last_date, o.last_close, o.unrealized_gain) == ("open", 0, "2026-11-06", 100.0, 0.0)


def test_the_gain_is_counted_in_cents_the_way_the_account_book_rounds():
    o = sim.follow(500.125, 1, "2026-11-06", series(QUIET, (101.0, 600.0, 100.0, 500.0)), stop_price=None, target_price=600.0, time_stop_date=None, slippage_bps=0)
    assert o.exit_price == 600.0 and o.realized_gain == round(600.0 - 500.13, 2) == 99.87                         # 500.125 is 500.13 in the book (half up), not 500.12


def test_a_capital_return_plan_has_no_target_and_a_plan_without_a_stop_or_a_time_stop_skips_those_rules():
    o = follow(QUIET, (100.0, 500.0, 99.0, 300.0), target_price=None)
    assert o.status == "open" and o.last_close == 300.0
    assert follow(QUIET, (50.0, 51.0, 49.0, 50.0), stop_price=None).status == "open"
    assert follow(QUIET, QUIET, QUIET, time_stop_date=None).status == "open"


def test_slippage_is_taken_off_the_stop_and_the_time_stop_exits_only():
    assert follow(QUIET, (100.0, 101.0, 94.0, 96.0), slippage_bps=0).exit_price == 95.0
    assert follow(QUIET, (100.0, 101.0, 94.0, 96.0), slippage_bps=100).exit_price == 94.05
    assert follow(QUIET, (111.0, 112.0, 110.5, 111.5), slippage_bps=100).exit_price == 111.0                      # a target is taken at the price, no slippage
    assert follow(QUIET, QUIET, time_stop_date="2026-11-09", slippage_bps=100).exit_price == 99.0


def test_the_outcome_of_a_closed_trade_has_the_gain_the_return_and_the_sessions():
    o = sim.follow(167.18, 44, "2026-11-06", series((167.0, 168.0, 166.0, 167.5), (167.0, 169.0, 166.5, 168.0), start="2026-11-06"), stop_price=157.2351, target_price=168.3783, time_stop_date="2026-12-31", slippage_bps=5)
    assert (o.exit_reason, o.exit_price, o.sessions_held, o.realized_gain, o.return_pct) == ("target", 168.3783, 1, 52.73, 0.007168)
    assert isinstance(o, Outcome) and o.benchmark_return_pct is None and o.excess_return_pct is None


@pytest.mark.parametrize("bars", [[], None, [bar("2026-11-09", *QUIET)], [bar("2026-11-05", *QUIET)], [{"date": "2026-11-06"}], ["x"]])
def test_a_series_that_does_not_start_with_the_entry_bar_cannot_be_followed_and_says_nothing(bars):
    o = sim.follow(100.0, 10, "2026-11-06", bars, **PLAN)
    assert o == Outcome(status="open")


def test_bars_before_the_entry_day_are_ignored_and_the_entry_may_be_given_as_a_date():
    before = [bar("2026-11-04", 1, 1, 1, 1), bar("2026-11-05", 1, 1, 1, 1)] + series(QUIET, (100.0, 101.0, 94.0, 96.0))
    assert sim.follow(100.0, 10, date(2026, 11, 6), before, **PLAN).exit_date == "2026-11-09"


# ---------------------------------------------------------------- the benchmark

def test_the_benchmark_return_runs_from_the_entry_bars_open_to_the_exit_dates_close_and_the_excess_is_the_difference():
    bench = series((400.0, 402.0, 399.0, 401.0), (401.0, 405.0, 400.0, 404.0), (404.0, 410.0, 403.0, 408.0))
    o = sim.follow(100.0, 10, "2026-11-06", series(QUIET, (101.0, 110.0, 100.0, 109.0), QUIET), benchmark_bars=bench, **PLAN)
    assert (o.exit_reason, o.exit_date, o.exit_price, o.return_pct) == ("target", "2026-11-09", 110.0, 0.1)
    assert o.benchmark_return_pct == 0.01 and o.excess_return_pct == 0.09                                          # 404 over 400: the bar of the exit date, not the later one


def test_the_benchmark_of_an_open_trade_runs_to_the_last_close_and_none_is_none_when_its_bars_are_not_stored():
    bench = series((400.0, 402.0, 399.0, 401.0), (401.0, 405.0, 400.0, 404.0))
    o = sim.follow(100.0, 10, "2026-11-06", series(QUIET, (100.0, 103.0, 99.0, 102.0)), benchmark_bars=bench, **PLAN)
    assert (o.status, o.return_pct, o.benchmark_return_pct, o.excess_return_pct) == ("open", 0.02, 0.01, 0.01)
    for missing in (None, [], series((1, 1, 1, 1), start="2026-11-09")):
        none = sim.follow(100.0, 10, "2026-11-06", series(QUIET, (100.0, 103.0, 99.0, 102.0)), benchmark_bars=missing, **PLAN)
        assert none.benchmark_return_pct is None and none.excess_return_pct is None and none.return_pct == 0.02


def test_follow_changes_nothing_it_is_given():
    rows = series(QUIET, (100.0, 101.0, 94.0, 96.0))
    snapshot = [dict(b) for b in rows]
    sim.follow(100.0, 10, "2026-11-06", rows, **PLAN)
    assert rows == snapshot
