"""Tests for prometheus.common.portfolio and the sample portfolio file (design choices entry 42).

Run:  pytest tests/test_portfolio.py -q

No network, no key, no cost.
"""
from decimal import Decimal
import json
from datetime import date

import pytest
from pydantic import ValidationError

from prometheus.common import portfolio as pf


def make(**changes):
    data = {"user": "Test User", "positions": [{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}]}
    data.update(changes)
    return pf.Portfolio.model_validate(data)


# ---------------------------------------------------------------- the T+1 rule


def test_a_position_with_no_purchase_date_is_all_sellable_and_a_symbol_not_held_has_none():
    p = make()
    assert p.sellable_shares("MSFT", date(2026, 11, 3)) == 10 and p.sellable_shares("NVDA", date(2026, 11, 3)) == 0


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


# ---------------------------------------------------------------- apply_fill: one filled trade (Trader v1 spec section 13)


def test_a_second_buy_the_same_day_adds_to_the_locked_shares_and_a_wholly_locked_position_stays_locked():
    rich = pf.Portfolio.model_validate({**base().model_dump(), "cash_avail": Decimal("10000.00")})
    twice = pf.apply_fill(pf.apply_fill(rich, "F", "buy", 300, Decimal("12.00"), TODAY), "F", "buy", 40, Decimal("10.00"), TODAY)
    assert twice.position("F").shares == 600 and twice.position("F").recent_shares == 340 and twice.sellable_shares("F", TODAY) == 260 and twice.cash_avail == Decimal("10000.00") - Decimal("3600.00") - Decimal("400.00")
    locked = make(cash_avail="100", positions=[{"symbol": "F", "shares": 5, "cost_basis": 1.0, "acquired": "2026-11-03"}])
    assert pf.apply_fill(locked, "F", "buy", 5, Decimal("3.00"), TODAY).sellable_shares("F", TODAY) == 0


@pytest.mark.parametrize("args, message", [
    (("MSFT", "hold", 1, Decimal("100.00")), "side must be buy or sell"),
    (("MSFT", "buy", 0, Decimal("100.00")), "number above zero"),            # entry 110 amended these four: a quantity may have a fraction (1.5 is a quantity now, so the case is 7 decimals), and the message no longer says "whole"
    (("MSFT", "buy", -2, Decimal("100.00")), "number above zero"),
    (("MSFT", "buy", Decimal("0.1234567"), Decimal("100.00")), "number above zero"),
    (("MSFT", "buy", True, Decimal("100.00")), "number above zero"),
    (("MSFT", "buy", 1, 0), "finite number above zero"),
    (("MSFT", "buy", 1, Decimal("-5.00")), "finite number above zero"),
    (("MSFT", "buy", 1, Decimal("NaN")), "finite number above zero"),
    (("MSFT", "buy", 1, Decimal("Infinity")), "finite number above zero"),
    (("MSFT", "buy", 1, "100"), "finite number above zero"),
    (("MSFT", "buy", 1, True), "finite number above zero"),
    (("MSFT", "buy", 1, Decimal("0.001")), "at least one cent"),
    (("../x", "buy", 1, Decimal("100.00")), "not a valid ticker symbol"),
    ((7, "buy", 1, Decimal("100.00")), "must be text"),
    (("MSFT", "buy", 20, Decimal("100.00")), r"cash \$1,000.00 does not cover \$2,000.00"),
    (("NVDA", "sell", 1, Decimal("100.00")), "cannot sell 1 of NVDA: the portfolio holds 0"),
    (("MSFT", "sell", 11, Decimal("100.00")), "cannot sell 11 of MSFT: the portfolio holds 10"),
])
def test_a_fill_the_portfolio_cannot_take_is_refused_with_the_reason(args, message):
    p = make(cash_avail="1000", positions=[{"symbol": "MSFT", "shares": 10, "cost_basis": 100.0}])
    with pytest.raises(ValueError, match=message):
        pf.apply_fill(p, *args, TODAY)


def test_the_date_of_a_fill_must_be_a_date():
    from datetime import datetime

    for bad in ("2026-11-03", datetime(2026, 11, 3, 10, 30), None):
        with pytest.raises(ValueError, match="today must be a date"):
            pf.apply_fill(base(), "MSFT", "sell", 1, Decimal("100.00"), bad)


# ---------------------------------------------------------------- the cash cut (moved here from test_trader_simulator.py: `affordable_shares` is the Account Manager's arithmetic since design choices entry 81)

@pytest.mark.parametrize("cash, price, shares, commission, expected", [
    (Decimal("1000.00"), Decimal("100.05"), 44, Decimal("0.00"), 9),                 # 9 x 100.05 = 900.45; 10 x 100.05 = 1000.50 is over
    (Decimal("1000.50"), Decimal("100.05"), 44, Decimal("0.00"), 10),                # 10 x 100.05 = 1000.50: exactly the cash is enough
    (Decimal("1000.49"), Decimal("100.05"), 44, Decimal("0.00"), 9),                 # a cent short
    (Decimal("1000.00"), Decimal("100.05"), 5, Decimal("0.00"), 5),                  # never more than the shares approved
    (Decimal("100.00"), Decimal("100.05"), 44, Decimal("0.00"), 0),                  # not even one share
    (Decimal("100.05"), Decimal("100.05"), 44, Decimal("0.00"), 1),
    (Decimal("101.05"), Decimal("100.05"), 44, Decimal("1.00"), 1),                  # the commission is counted: 100.05 + 1.00
    (Decimal("101.04"), Decimal("100.05"), 44, Decimal("1.00"), 0),
    (Decimal("500.13"), Decimal("500.125"), 3, Decimal("0.00"), 1),                  # 500.125 is 500.13 in the account book (half up), and a cash of 500.13 pays for exactly one
    (Decimal("500.12"), Decimal("500.125"), 3, Decimal("0.00"), 0),
    (Decimal("99.99"), Decimal("33.33"), 5, Decimal("0.00"), 3),                     # 3 x 33.33 = 99.99: a quotient that floats to 2.9999999999999996 must not lose the third share
    (Decimal("0.00"), Decimal("10.00"), 5, Decimal("0.00"), 0), (Decimal("-5.00"), Decimal("10.00"), 5, Decimal("0.00"), 0),
    (Decimal("1000.00"), Decimal("0.00"), 5, Decimal("0.00"), 0), (Decimal("1000.00"), Decimal("10.00"), 0, Decimal("0.00"), 0), (Decimal("1000.00"), Decimal("10.00"), Decimal("0.1234567"), Decimal("0.00"), 0),   # entry 110: 2.5 is a quantity now (a fractional order is cut to a fractional quantity); seven decimals is still refused
    (Decimal("NaN"), Decimal("10.00"), 5, Decimal("0.00"), 0), (Decimal("1000.00"), None, 5, Decimal("0.00"), 0),
])
def test_the_cash_pays_for_the_most_shares_that_fit_to_the_cent(cash, price, shares, commission, expected):
    assert pf.affordable_shares(cash, price, shares, commission) == expected
