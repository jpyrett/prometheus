"""Tests for prometheus.common.portfolio and the sample portfolio file (design choices entry 42).

Run:  pytest tests/test_portfolio.py -q

No network, no key, no cost.
"""
import json
from datetime import date

import pytest
from pydantic import ValidationError

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO
from prometheus.common import portfolio as pf


def make(**changes):
    data = {"user": "Test User", "positions": [{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}]}
    data.update(changes)
    return pf.Portfolio.model_validate(data)


# ---------------------------------------------------------------- the sample file is Jacob's portfolio

def test_the_sample_portfolio_is_the_one_jacob_described():
    p = pf.load_portfolio(DEFAULT_PORTFOLIO)
    assert p.user == "Jacob Pyrett" and p.currency == "USD"
    assert [(x.symbol, x.shares, x.cost_basis) for x in p.positions] == [("MSFT", 50, 150.0), ("TSLA", 123, 300.0), ("GM", 60, 10.0)]
    assert p.cash_avail == 51.23  # the file writes it as the string "51.23"; it loads as a number


def test_the_sample_portfolio_is_all_sellable_today():
    p = pf.load_portfolio(DEFAULT_PORTFOLIO)
    assert {s: p.sellable_shares(s, date(2026, 11, 3)) for s in p.symbols} == {"MSFT": 50, "TSLA": 123, "GM": 60}


# ---------------------------------------------------------------- the T+1 rule

def test_shares_bought_today_are_not_sellable_until_tomorrow():
    p = make(positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0, "acquired": "2026-11-03"}])
    assert p.sellable_shares("MSFT", date(2026, 11, 3)) == 0
    assert p.sellable_shares("MSFT", date(2026, 11, 4)) == 10


def test_a_position_with_no_purchase_date_is_all_sellable_and_a_symbol_not_held_has_none():
    p = make()
    assert p.sellable_shares("MSFT", date(2026, 11, 3)) == 10 and p.sellable_shares("NVDA", date(2026, 11, 3)) == 0


def test_market_value_uses_the_prices_given():
    p = make(positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}, {"symbol": "GM", "shares": 4, "cost_basis": 10.0}])
    assert p.market_value({"MSFT": 400.0, "GM": 50.0}) == 4200.0


# ---------------------------------------------------------------- bad portfolios are refused

@pytest.mark.parametrize("position", [
    {"symbol": "MSFT", "shares": 0, "cost_basis": 100.0},
    {"symbol": "MSFT", "shares": -3, "cost_basis": 100.0},
    {"symbol": "MSFT", "shares": 1.5, "cost_basis": 100.0},
    {"symbol": "MSFT", "shares": 10, "cost_basis": 0},
    {"symbol": "../etc", "shares": 10, "cost_basis": 100.0},
    {"symbol": "MSFT", "shares": 10, "cost_basis": 100.0, "acquired": "yesterday"},
])
def test_a_position_that_makes_no_sense_is_refused(position):
    with pytest.raises(ValidationError):
        make(positions=[position])


def test_a_symbol_twice_is_refused_and_a_symbol_is_cleaned():
    with pytest.raises(ValidationError, match="appears twice"):
        make(positions=[{"symbol": "MSFT", "shares": 1, "cost_basis": 1.0}, {"symbol": "msft", "shares": 2, "cost_basis": 1.0}])
    assert make(positions=[{"symbol": " $msft ", "shares": 1, "cost_basis": 1.0}]).symbols == ["MSFT"]


@pytest.mark.parametrize("cash", [-1.0, "-0.01", "abc", "nan", "inf", None])
def test_free_cash_that_is_negative_or_not_a_finite_number_is_refused(cash):
    with pytest.raises(ValidationError):
        make(cash_avail=cash)


@pytest.mark.parametrize("cash, expected", [("51.23", 51.23), (51.23, 51.23), (0, 0.0), ("0", 0.0), (" 7.5 ", 7.5)])
def test_free_cash_may_be_a_number_or_a_string_of_one(cash, expected):
    assert make(cash_avail=cash).cash_avail == expected


def test_a_portfolio_without_cash_avail_has_none():
    assert make().cash_avail == 0.0


def test_a_file_that_is_not_json_is_refused_with_its_name(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="p.json: not valid JSON"):
        pf.load_portfolio(path)


def test_a_portfolio_survives_a_round_trip_through_json(tmp_path):
    p = pf.load_portfolio(DEFAULT_PORTFOLIO)
    path = tmp_path / "copy.json"
    path.write_text(json.dumps(json.loads(p.model_dump_json())), encoding="utf-8")
    assert pf.load_portfolio(path) == p


# ---------------------------------------------------------------- shares bought today: only those are locked (T+1)

TODAY = date(2026, 11, 3)


def test_only_the_shares_bought_today_are_locked_when_the_position_says_how_many():
    p = make(positions=[{"symbol": "F", "shares": 560, "cost_basis": 7.36, "acquired": "2026-11-03", "recent_shares": 300}])
    assert p.sellable_shares("F", TODAY) == 260 and p.sellable_shares("F", date(2026, 11, 4)) == 560


def test_without_the_number_the_whole_position_is_locked_as_before():
    p = make(positions=[{"symbol": "F", "shares": 560, "cost_basis": 7.36, "acquired": "2026-11-03"}])
    assert p.sellable_shares("F", TODAY) == 0


@pytest.mark.parametrize("recent", [0, -1])
def test_recent_shares_must_be_positive(recent):
    with pytest.raises(ValidationError):
        make(positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "recent_shares": recent}])


def base():
    return make(cash_avail="100", positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0, "acquired": "2024-01-02"}, {"symbol": "F", "shares": 260, "cost_basis": 2.0}])


def test_after_a_purchase_the_sells_are_out_the_shares_are_in_and_the_cash_is_what_the_plan_says():
    after = pf.after_purchase(base(), "GM", 3, 55.0, [("MSFT", 4)], 7.5, TODAY)
    assert [(p.symbol, p.shares) for p in after.positions] == [("MSFT", 6), ("F", 260), ("GM", 3)] and after.cash_avail == 7.5
    gm = after.position("GM")
    assert gm.cost_basis == 55.0 and gm.acquired == TODAY and after.sellable_shares("GM", TODAY) == 0  # a new position: all of it is locked until tomorrow
    assert after.sellable_shares("MSFT", TODAY) == 6


def test_a_position_sold_out_completely_is_removed():
    assert pf.after_purchase(base(), "GM", 1, 55.0, [("MSFT", 10)], 0.0, TODAY).symbols == ["F", "GM"]


def test_buying_more_of_a_stock_already_held_averages_the_basis_and_locks_only_the_new_shares():
    after = pf.after_purchase(base(), "F", 300, 12.0, [("MSFT", 10)], 3.0, TODAY)
    f = after.position("F")
    assert f.shares == 560 and f.cost_basis == 7.3571 and f.recent_shares == 300 and f.acquired == TODAY  # (260 x 2 + 300 x 12) / 560
    assert after.sellable_shares("F", TODAY) == 260  # the old shares can be sold today, the new ones cannot


def test_a_second_purchase_the_same_day_adds_to_the_locked_shares():
    once = pf.after_purchase(base(), "F", 300, 12.0, [], 0.0, TODAY)
    twice = pf.after_purchase(once, "F", 40, 10.0, [], 0.0, TODAY)
    assert twice.position("F").shares == 600 and twice.position("F").recent_shares == 340 and twice.sellable_shares("F", TODAY) == 260


def test_buying_into_a_position_that_was_wholly_locked_keeps_it_wholly_locked():
    locked = make(positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "acquired": "2026-11-03"}])
    assert pf.after_purchase(locked, "F", 5, 3.0, [], 0.0, TODAY).sellable_shares("F", TODAY) == 0


@pytest.mark.parametrize("sells", [[("NVDA", 1)], [("MSFT", 11)], [("MSFT", 0)], [("MSFT", 1.5)]])
def test_a_sell_the_portfolio_cannot_make_is_refused(sells):
    with pytest.raises(ValueError, match="cannot sell"):
        pf.after_purchase(base(), "GM", 1, 55.0, sells, 0.0, TODAY)


def test_after_a_purchase_the_portfolio_given_is_not_changed():
    p = base()
    before = p.model_dump_json()
    pf.after_purchase(p, "F", 300, 12.0, [("MSFT", 4)], 3.0, TODAY)
    assert p.model_dump_json() == before
