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


# ---------------------------------------------------------------- the T+1 rule


def test_a_position_with_no_purchase_date_is_all_sellable_and_a_symbol_not_held_has_none():
    p = make()
    assert p.sellable_shares("MSFT", date(2026, 11, 3)) == 10 and p.sellable_shares("NVDA", date(2026, 11, 3)) == 0


def test_market_value_uses_the_prices_given():
    p = make(positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}, {"symbol": "GM", "shares": 4, "cost_basis": 10.0}])
    assert p.market_value({"MSFT": 400.0, "GM": 50.0}) == 4200.0


# ---------------------------------------------------------------- bad portfolios are refused


def test_a_symbol_twice_is_refused_and_a_symbol_is_cleaned():
    with pytest.raises(ValidationError, match="appears twice"):
        make(positions=[{"symbol": "MSFT", "shares": 1, "cost_basis": 1.0}, {"symbol": "msft", "shares": 2, "cost_basis": 1.0}])
    assert make(positions=[{"symbol": " $msft ", "shares": 1, "cost_basis": 1.0}]).symbols == ["MSFT"]


def test_a_file_that_is_not_json_is_refused_with_its_name(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="p.json: not valid JSON"):
        pf.load_portfolio(path)


# ---------------------------------------------------------------- shares bought today: only those are locked (T+1)

TODAY = date(2026, 11, 3)


def base():
    return make(cash_avail="100", positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0, "acquired": "2024-01-02"}, {"symbol": "F", "shares": 260, "cost_basis": 2.0}])


def test_a_second_purchase_the_same_day_adds_to_the_locked_shares():
    once = pf.after_purchase(base(), "F", 300, 12.0, [], 0.0, TODAY)
    twice = pf.after_purchase(once, "F", 40, 10.0, [], 0.0, TODAY)
    assert twice.position("F").shares == 600 and twice.position("F").recent_shares == 340 and twice.sellable_shares("F", TODAY) == 260


def test_buying_into_a_position_that_was_wholly_locked_keeps_it_wholly_locked():
    locked = make(positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "acquired": "2026-11-03"}])
    assert pf.after_purchase(locked, "F", 5, 3.0, [], 0.0, TODAY).sellable_shares("F", TODAY) == 0


# ---------------------------------------------------------------- apply_fill: one filled trade (Trader v1 spec section 13)


def test_a_second_buy_the_same_day_adds_to_the_locked_shares_and_a_wholly_locked_position_stays_locked():
    rich = pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": 10000.0})
    twice = pf.apply_fill(pf.apply_fill(rich, "F", "buy", 300, 12.0, TODAY), "F", "buy", 40, 10.0, TODAY)
    assert twice.position("F").shares == 600 and twice.position("F").recent_shares == 340 and twice.sellable_shares("F", TODAY) == 260 and twice.cash_avail == 10000.0 - 3600.0 - 400.0
    locked = make(cash_avail="100", positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "acquired": "2026-11-03"}])
    assert pf.apply_fill(locked, "F", "buy", 5, 3.0, TODAY).sellable_shares("F", TODAY) == 0


def test_an_amount_is_rounded_half_up_to_the_cent_the_way_the_account_book_rounds_it():
    assert round(500.125, 2) == 500.12 and pf.trade_amount(1, 500.125) == 500.13                    # Python's round goes to the even cent; the book (Decimal, ROUND_HALF_UP) does not
    assert pf.trade_amount(3, 167.125) == 501.38 and pf.trade_amount(44, 167.1) == 7352.4 and pf.trade_amount(10, 150.25) == 1502.5 and pf.trade_amount(1, 0.004) == 0.0
    assert pf.apply_fill(make(cash_avail="1000", positions=[]), "PG", "buy", 1, 500.125, TODAY).cash_avail == 499.87
    sale = make(cash_avail="0", positions=[{"symbol": "PG", "shares": 1, "cost_basis": 1.0}])
    assert pf.apply_fill(sale, "PG", "sell", 1, 500.125, TODAY).cash_avail == 500.13


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
