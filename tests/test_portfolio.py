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


# ---------------------------------------------------------------- apply_fill: one filled trade (Trader v1 spec section 13)

def test_a_buy_opens_a_new_position_spends_the_cash_and_locks_all_of_it_until_tomorrow():
    after = pf.apply_fill(base(), "PG", "buy", 4, 12.5, TODAY, name="Procter & Gamble Co")
    assert after.cash_avail == 50.0 and after.symbols == ["MSFT", "F", "PG"]
    pg = after.position("PG")
    assert (pg.shares, pg.cost_basis, pg.acquired, pg.name) == (4, 12.5, TODAY, "Procter & Gamble Co") and after.sellable_shares("PG", TODAY) == 0 and after.sellable_shares("PG", date(2026, 11, 4)) == 4


def test_a_buy_of_a_stock_already_held_averages_the_cost_and_locks_only_the_new_shares_exactly_as_after_purchase_does():
    with_fill = pf.apply_fill(pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": 4000.0}), "F", "buy", 300, 12.0, TODAY)
    f = with_fill.position("F")
    assert (f.shares, f.cost_basis, f.recent_shares, f.acquired) == (560, 7.3571, 300, TODAY) and with_fill.cash_avail == 400.0 and with_fill.sellable_shares("F", TODAY) == 260
    plan = pf.after_purchase(pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": 4000.0}), "F", 300, 12.0, [], 400.0, TODAY)   # the same lock rules, the same code
    assert with_fill.model_dump() == plan.model_dump()


def test_a_second_buy_the_same_day_adds_to_the_locked_shares_and_a_wholly_locked_position_stays_locked():
    rich = pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": 10000.0})
    twice = pf.apply_fill(pf.apply_fill(rich, "F", "buy", 300, 12.0, TODAY), "F", "buy", 40, 10.0, TODAY)
    assert twice.position("F").shares == 600 and twice.position("F").recent_shares == 340 and twice.sellable_shares("F", TODAY) == 260 and twice.cash_avail == 10000.0 - 3600.0 - 400.0
    locked = make(cash_avail="100", positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "acquired": "2026-11-03"}])
    assert pf.apply_fill(locked, "F", "buy", 5, 3.0, TODAY).sellable_shares("F", TODAY) == 0


def test_a_buy_that_the_cash_covers_to_the_cent_goes_through_and_one_cent_more_does_not():
    exact = make(cash_avail="51.23", positions=[])
    assert pf.apply_fill(exact, "PG", "buy", 1, 51.23, TODAY).cash_avail == 0.0
    with pytest.raises(ValueError, match=r"cash \$51.23 does not cover \$51.24"):
        pf.apply_fill(exact, "PG", "buy", 1, 51.24, TODAY)


def test_the_cost_of_a_buy_is_rounded_to_cents():
    after = pf.apply_fill(make(cash_avail="1000", positions=[]), "PG", "buy", 3, 167.1234, TODAY)
    assert after.cash_avail == round(1000 - round(3 * 167.1234, 2), 2) == 498.63 and after.position("PG").cost_basis == 167.1234


def test_an_amount_is_rounded_half_up_to_the_cent_the_way_the_account_book_rounds_it():
    assert round(500.125, 2) == 500.12 and pf.trade_amount(1, 500.125) == 500.13                    # Python's round goes to the even cent; the book (Decimal, ROUND_HALF_UP) does not
    assert pf.trade_amount(3, 167.125) == 501.38 and pf.trade_amount(44, 167.1) == 7352.4 and pf.trade_amount(10, 150.25) == 1502.5 and pf.trade_amount(1, 0.004) == 0.0
    assert pf.apply_fill(make(cash_avail="1000", positions=[]), "PG", "buy", 1, 500.125, TODAY).cash_avail == 499.87
    sale = make(cash_avail="0", positions=[{"symbol": "PG", "shares": 1, "cost_basis": 1.0}])
    assert pf.apply_fill(sale, "PG", "sell", 1, 500.125, TODAY).cash_avail == 500.13


def test_a_sale_brings_in_the_proceeds_and_keeps_the_rest_of_the_position():
    after = pf.apply_fill(base(), "MSFT", "sell", 4, 420.0, TODAY)
    assert after.cash_avail == 1780.0 and after.position("MSFT").shares == 6 and after.position("MSFT").cost_basis == 100.0 and after.symbols == ["MSFT", "F"]


def test_a_sale_keeps_the_locked_shares_inside_the_position():
    held = make(cash_avail="0", positions=[{"symbol": "F", "shares": 10, "cost_basis": 1.0, "acquired": "2026-11-03", "recent_shares": 4}])
    after = pf.apply_fill(held, "F", "sell", 8, 2.0, TODAY)                      # 2 left, and only 2 can be locked
    assert after.position("F").shares == 2 and after.position("F").recent_shares == 2 and after.cash_avail == 16.0
    assert pf.apply_fill(held, "F", "sell", 3, 2.0, TODAY).position("F").recent_shares == 4  # 7 left, all 4 locked shares are still inside


def test_a_position_sold_out_is_removed_and_its_cash_arrives():
    after = pf.apply_fill(base(), "MSFT", "sell", 10, 110.0, TODAY)
    assert after.symbols == ["F"] and after.cash_avail == 1200.0 and after.position("MSFT") is None


def test_a_name_is_kept_unless_a_new_one_is_given():
    named = make(cash_avail="1000", positions=[{"symbol": "PG", "name": "Procter & Gamble Co", "shares": 1, "cost_basis": 100.0}])
    assert pf.apply_fill(named, "PG", "buy", 1, 100.0, TODAY).position("PG").name == "Procter & Gamble Co"
    assert pf.apply_fill(named, "PG", "buy", 1, 100.0, TODAY, name="P&G").position("PG").name == "P&G"
    assert pf.apply_fill(named, "PG", "sell", 1, 100.0, TODAY).position("PG") is None


@pytest.mark.parametrize("args, message", [
    (("MSFT", "hold", 1, 100.0), "side must be buy or sell"),
    (("MSFT", "buy", 0, 100.0), "whole number of at least 1"),
    (("MSFT", "buy", -2, 100.0), "whole number of at least 1"),
    (("MSFT", "buy", 1.5, 100.0), "whole number of at least 1"),
    (("MSFT", "buy", True, 100.0), "whole number of at least 1"),
    (("MSFT", "buy", 1, 0), "finite number above zero"),
    (("MSFT", "buy", 1, -5.0), "finite number above zero"),
    (("MSFT", "buy", 1, float("nan")), "finite number above zero"),
    (("MSFT", "buy", 1, float("inf")), "finite number above zero"),
    (("MSFT", "buy", 1, "100"), "finite number above zero"),
    (("MSFT", "buy", 1, True), "finite number above zero"),
    (("MSFT", "buy", 1, 0.001), "at least one cent"),
    (("../x", "buy", 1, 100.0), "not a valid ticker symbol"),
    ((7, "buy", 1, 100.0), "must be text"),
    (("MSFT", "buy", 20, 100.0), r"cash \$1,000.00 does not cover \$2,000.00"),
    (("NVDA", "sell", 1, 100.0), "cannot sell 1 of NVDA: the portfolio holds 0"),
    (("MSFT", "sell", 11, 100.0), "cannot sell 11 of MSFT: the portfolio holds 10"),
])
def test_a_fill_the_portfolio_cannot_take_is_refused_with_the_reason(args, message):
    p = make(cash_avail="1000", positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}])
    with pytest.raises(ValueError, match=message):
        pf.apply_fill(p, *args, TODAY)


def test_the_date_of_a_fill_must_be_a_date():
    from datetime import datetime

    for bad in ("2026-11-03", datetime(2026, 11, 3, 10, 30), None):
        with pytest.raises(ValueError, match="today must be a date"):
            pf.apply_fill(base(), "MSFT", "sell", 1, 100.0, bad)


def test_apply_fill_does_not_change_the_portfolio_given_and_returns_a_valid_one():
    p = base()
    before = p.model_dump_json()
    after = pf.apply_fill(pf.apply_fill(p, "MSFT", "sell", 4, 120.0, TODAY), "PG", "buy", 2, 100.0, TODAY)
    assert p.model_dump_json() == before and after is not p
    pf.Portfolio.model_validate_json(after.model_dump_json())                                      # still a portfolio: no negative cash, no symbol twice, positive shares


def test_a_run_of_fills_ends_where_the_same_plan_through_after_purchase_does():
    """The broker and the Account Manager both apply fills one by one; a plan's `after_purchase` is the other way to get there. The two must agree, or the portfolios the desk shows would differ."""
    start = pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": 100.0})
    by_fills = pf.apply_fill(pf.apply_fill(start, "MSFT", "sell", 4, 120.0, TODAY), "GM", "buy", 3, 55.0, TODAY)
    by_plan = pf.after_purchase(start, "GM", 3, 55.0, [("MSFT", 4)], 100.0 + 480.0 - 165.0, TODAY)
    assert by_fills.model_dump() == by_plan.model_dump()


# ---------------------------------------------------------------- the cash cut (moved here from test_trader_simulator.py: `affordable_shares` is the Account Manager's arithmetic since design choices entry 81)

@pytest.mark.parametrize("cash, price, shares, commission, expected", [
    (1000.0, 100.05, 44, 0.0, 9),                 # 9 x 100.05 = 900.45; 10 x 100.05 = 1000.50 is over
    (1000.5, 100.05, 44, 0.0, 10),                # 10 x 100.05 = 1000.50: exactly the cash is enough
    (1000.49, 100.05, 44, 0.0, 9),                # a cent short
    (1000.0, 100.05, 5, 0.0, 5),                  # never more than the shares approved
    (100.0, 100.05, 44, 0.0, 0),                  # not even one share
    (100.05, 100.05, 44, 0.0, 1),
    (101.05, 100.05, 44, 1.0, 1),                 # the commission is counted: 100.05 + 1.00
    (101.04, 100.05, 44, 1.0, 0),
    (500.13, 500.125, 3, 0.0, 1),                 # 500.125 is 500.13 in the account book (half up), and a cash of 500.13 pays for exactly one
    (500.12, 500.125, 3, 0.0, 0),
    (99.99, 33.33, 5, 0.0, 3),                    # 3 x 33.33 = 99.99: a quotient that floats to 2.9999999999999996 must not lose the third share
    (0.0, 10.0, 5, 0.0, 0), (-5.0, 10.0, 5, 0.0, 0), (float("nan"), 10.0, 5, 0.0, 0),
    (1000.0, 0.0, 5, 0.0, 0), (1000.0, None, 5, 0.0, 0), (1000.0, 10.0, 0, 0.0, 0), (1000.0, 10.0, 2.5, 0.0, 0),
])
def test_the_cash_pays_for_the_most_shares_that_fit_to_the_cent(cash, price, shares, commission, expected):
    assert pf.affordable_shares(cash, price, shares, commission) == expected
