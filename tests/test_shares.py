"""Tests for prometheus.common.shares (design choices entry 110): the one door of a share quantity.

Run:  pytest tests/test_shares.py -q

No network, no key, no cost. The prices are the REAL prices of the oil session `tests/fixtures/session_real/auto_oil_no_second_mark.json` (the decisions' own prices).
"""
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.common import shares as sh
from prometheus.common.money import money, price
from prometheus.common.portfolio import trade_amount

OIL = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "auto_oil_no_second_mark.json").read_text(encoding="utf-8"))["order"]["trade_decisions"]
PRICES = {d["symbol"]: price(money(d["est_cost"]) / d["shares"]) for d in OIL if d["shares"]}      # EOG 158.32, CVX 219.89 ... the price each purchase was sized at (the capture's float goes through the door of money, then a Decimal quotient)
SQL = Path(__file__).parent.parent / "sql" / "prometheus_accounts.sql"


def test_the_door_keeps_a_whole_number_an_int_and_gives_a_fraction_as_a_float():
    # entry 112 (Jacob, 2026-11-16): the door gives a Decimal always, a whole quantity with no decimals (it was an int and a float before)
    assert [str(sh.quantity(x)) for x in (81, 81.0, Decimal("15.000000"), Decimal("0E-6"), 0, -0.0)] == ["81", "81", "15", "0", "0", "0"] and all(type(sh.quantity(x)) is Decimal for x in (81, 81.0, Decimal("15.000000")))
    assert sh.quantity(81.004455) == Decimal("81.004455") and type(sh.quantity(Decimal("0.000001"))) is Decimal and sh.quantity(Decimal("0.5")) == Decimal("0.5")
    for refused in (0.1234567, 0.1 + 0.2, -1, -0.5, float("nan"), float("inf"), True, False, "2", None, 1e300, [1]):
        with pytest.raises(ValueError):
            sh.quantity(refused)                         # more than 6 decimals is refused, never rounded; a bool and a string never count


def test_quantities_are_added_and_taken_apart_exactly_and_a_holding_sold_in_full_is_exactly_zero():
    assert sh.plus(81, Decimal("0.004455")) == Decimal("81.004455") and sh.plus(Decimal("0.1"), Decimal("0.2")) == Decimal("0.3") and sh.minus(Decimal("81.004455"), 81) == Decimal("0.004455") and sh.minus(Decimal("81.004455"), Decimal("81.004455")) == 0 and str(sh.minus(Decimal("2.5"), Decimal("2.5"))) == "0"
    assert sh.difference(Decimal("81.004455"), 81) == Decimal("0.004455") and sh.difference(81, Decimal("81.5")) == Decimal("-0.5") and sh.same(81.004455, Decimal("81.004455")) and not sh.same(Decimal("81.004455"), 81)   # `same` also takes the float a JSON file gives, by its digits
    with pytest.raises(ValueError):
        sh.minus(81, Decimal("81.000001"))
    assert (sh.floor(Decimal("81.99")), sh.ceil(Decimal("81.000001")), sh.ceil(81), sh.whole(Decimal("81.0")), sh.whole(Decimal("0.5"))) == (81, 82, 81, True, False) and all(type(x) is Decimal for x in (sh.floor(Decimal("81.99")), sh.ceil(81)))
    assert sh.as_decimal(81) == Decimal("81.000000") and sh.as_decimal(Decimal("81.004455")) == Decimal("81.004455") and sh.as_decimal(81).as_tuple().exponent == -sh.PLACES


def test_buys_is_the_most_shares_whose_cost_to_the_cent_is_not_above_the_dollars_at_every_real_price():
    for price_ in PRICES.values():
        for dollars in (Decimal("5000.00"), Decimal("10000.00"), Decimal("4999.99"), Decimal("3333.34"), Decimal("1234.56"), Decimal("0.50"), Decimal("0.01")):
            n = sh.buys(dollars, price_)
            assert n * price_ <= dollars < sh.plus(n, sh.STEP) * price_      # the most whose price times shares is not above the dollars
            assert trade_amount(n, price_) == dollars                        # a step costs under half a cent here, so the cost IS the amount to the cent
    assert sh.buys(Decimal("0.009"), Decimal("158.32")) == 0 and sh.buys(0, Decimal("158.32")) == 0 and sh.buys(5000, 0) == 0 and sh.buys(5000, None) == 0 and sh.buys(Decimal("NaN"), 5) == 0      # nothing fits: not even a cent
    assert sh.buys(Decimal("1000.00"), Decimal("1000.00")) == 1 and sh.whole(sh.buys(Decimal("1000.00"), Decimal("1000.00"))) and not sh.whole(sh.buys(Decimal("5000.00"), PRICES["CVX"]))
    for price_ in (Decimal("5000.50"), Decimal("7123.45"), Decimal("9999.99")):                                  # a step costs over half a cent: the rounding can leave the cost a cent short, and one more step closes it
        for dollars in (Decimal("5000.00"), Decimal("10000.00"), Decimal("4999.99"), Decimal("1234.56")):
            assert trade_amount(sh.buys(dollars, price_), price_) == dollars


def test_the_four_share_columns_are_numeric_with_the_scale_of_places_in_the_create_table_and_in_the_alter():
    sql = SQL.read_text(encoding="utf-8")
    assert sh.PLACES == 6 and sh.STEP == Decimal(1).scaleb(-sh.PLACES)
    for table, columns in {"positions": ("shares", "recent_shares"), "trades": ("shares",), "recommendation_decisions": ("shares",)}.items():
        create = re.search(rf"CREATE TABLE IF NOT EXISTS trading\.{table} \((.*?)\n\);", sql, re.S).group(1)
        alter = re.search(rf"ALTER TABLE trading\.{table}\n(.*?);", sql, re.S).group(1)                     # the migration of an existing database (the other ALTERs of the table are one line)
        for column in columns:
            for where, text in (("CREATE TABLE", create), ("ALTER TABLE", alter)):
                found = re.search(rf"\b{column}\b(?: TYPE)?\s+(numeric)\((\d+),\s*(\d+)\)", text)
                assert found and int(found.group(3)) == sh.PLACES, f"{table}.{column} is not numeric(p, {sh.PLACES}) in its {where}"
            assert not re.search(rf"\b{column}\b\s+integer", create)
