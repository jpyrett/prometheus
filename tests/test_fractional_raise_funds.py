"""RaiseFunds reads a holding with a fraction of a share and still plans WHOLE shares (design choices entry 110; design/fractional_shares_spec.md section 6.6).

REAL DATA ONLY: the prices are the real prices of the real sessions (`tests/fixtures/session_real/`, through `test_fractional_even_split.real_prices`); the holdings are the sample portfolio's stocks with a fraction of a
share added to them, which is what an even split or a decimal order leaves in an account. RaiseFunds' own tests of whole numbers (`test_raise_funds.py`, `test_green_first.py`, `test_two_purchases.py`) are not changed.
"""
from datetime import date
from decimal import Decimal

import pytest
from test_fractional_even_split import PRICES

from prometheus.common import shares as sh
from prometheus.common.portfolio import Portfolio, Position, trade_amount
from prometheus.reasoning import raise_funds as rf

TODAY = date(2026, 11, 13)


def held() -> Portfolio:
    return Portfolio(user="Fraction Holder", cash_avail=Decimal("51.23"), positions=[
        Position(symbol="MSFT", shares=Decimal("50.75"), cost_basis=Decimal("150"), acquired=date(2024, 3, 15)),        # a fraction held since long ago
        Position(symbol="GM", shares=Decimal("0.5"), cost_basis=Decimal("10")),                                         # less than one share
        Position(symbol="PG", shares=1, cost_basis=Decimal("100")),                                          # exactly one
        Position(symbol="F", shares=60, cost_basis=Decimal("10"), acquired=TODAY, recent_shares=Decimal("0.25")),       # a fractional lot bought today: only it is locked (T+1)
        Position(symbol="XOM", shares=Decimal("2.5"), cost_basis=Decimal("100"))])                                       # the stock to buy more of


def problem(never_liquidate: bool, shares: int = 8):
    return rf.build_problem(held(), "XOM", shares, PRICES, TODAY, only_green=True, never_liquidate=never_liquidate)


def test_a_holding_with_a_fraction_is_read_as_it_is_and_only_whole_shares_of_it_may_be_sold():
    kept, free = problem(True), problem(False)
    assert {s: h.shares for s, h in kept.holdings.items()} == {"MSFT": Decimal("50.75"), "GM": Decimal("0.5"), "PG": 1, "F": 60} and kept.held == Decimal("2.5") and type(kept.holdings["MSFT"].shares) is Decimal and not sh.whole(kept.holdings["MSFT"].shares)
    assert {s: h.sellable for s, h in kept.holdings.items()} == {"MSFT": 49, "GM": 0, "PG": 0, "F": 59}          # one share of every holding is kept; a holding of one share or less has none to spare; never a fraction
    assert {s: h.sellable for s, h in free.holdings.items()} == {"MSFT": 50, "GM": 0, "PG": 1, "F": 59} and all(type(h.sellable) is Decimal and sh.whole(h.sellable) for h in free.holdings.values())      # F: 60 less the lot of 0.25 bought today is 59.75: 59 whole shares
    assert [h.symbol for h in kept.holdings.values() if kept.eligible(h)] == ["MSFT", "F"] and [h.symbol for h in free.holdings.values() if free.eligible(h)] == ["MSFT", "PG", "F"]


def test_the_purchase_stays_a_whole_number_and_a_position_is_closed_only_when_exactly_what_is_held_is_sold():
    with pytest.raises(ValueError, match="whole number of at least 1"):
        problem(False, shares=Decimal("2.5"))
    plan = problem(False)
    assert plan.shares == 8 and plan.cost == trade_amount(8, PRICES["XOM"])
    closing, partial = rf.score_plan(plan, (("PG", 1),)), rf.score_plan(plan, (("MSFT", 50),))
    assert closing[1]["keep"] < 1 and partial[1]["keep"] == 1                                                  # PG 1 of 1 closes it; MSFT 50 of 50.75 does not (a fraction is left)
