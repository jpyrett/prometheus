"""Tests for prometheus.common.money (design choices entry 112; design/decimal_money_spec.md): the doors of a dollar amount, a price and a rate, the pydantic types, and the ONE JSON encoder.

Run:  pytest tests/test_decimal_money.py -q

No network, no key, no cost. The numbers are the real ones of the live session `tests/fixtures/session_real/auto_even_split_dollars.json` (its decisions' dollars, prices and shares).
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from prometheus.common import money as m

SESSION = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_dollars.json").read_text(encoding="utf-8"))
DECISIONS = [d for d in SESSION["order"]["trade_decisions"] if d.get("dollars")]       # the five purchases that are orders in dollars


class Row(BaseModel):
    amount: m.Money
    cash: m.NonNegativeMoney
    paid: m.PositiveMoney
    at: m.Price
    chase: m.Rate


def test_jacobs_own_example_is_what_the_functions_do():
    price, tax_rate = Decimal("19.99"), Decimal("0.075")
    tax = (price * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert tax == Decimal("1.50") and price + tax == Decimal("21.49")
    assert m.cents(price * tax_rate) == tax and m.money(price + tax) == Decimal("21.49")


def test_a_float_is_taken_by_the_digits_it_is_written_with_and_never_by_its_binary_expansion():
    assert m.money(19.99) == Decimal("19.99") and m.price(158.32) == Decimal("158.3200") and m.rate(0.01) == Decimal("0.01") and m.digits(0.5) == Decimal("0.5")
    assert m.money(10_000) == Decimal("10000.00") and m.money(Decimal("5000.5")) == Decimal("5000.50") and m.money(-4.5) == Decimal("-4.50")
    with pytest.raises(ValueError):
        m.money(0.1 + 0.2)                         # 0.30000000000000004 is more than two decimals: refused, never rounded
    with pytest.raises(ValueError):
        m.money(Decimal("10.005"))


def test_a_bool_a_string_and_what_is_not_finite_are_refused_at_every_door():
    for door in (m.money, m.positive_money, m.non_negative_money, m.price, m.rate, m.digits):
        for refused in (True, False, "2", "19.99", "", None, [1], float("nan"), float("inf"), Decimal("NaN"), Decimal("-Infinity")):
            with pytest.raises(ValueError):
                door(refused)
    with pytest.raises(ValueError):
        m.positive_money(0)
    with pytest.raises(ValueError):
        m.non_negative_money(-0.01)
    with pytest.raises(ValueError):
        m.price(0)
    with pytest.raises(ValueError):
        m.rate(-1)


def test_a_price_from_outside_is_rounded_to_four_places_half_up_and_nothing_else_is():
    assert m.price(158.32125) == Decimal("158.3213") and m.price(158.32124) == Decimal("158.3212") and m.price(158.32) == Decimal("158.3200")
    assert m.price(Decimal("0.00005")) == Decimal("0.0001") and m.price(171.43) == Decimal("171.4300") and str(m.price(150)) == "150.0000"
    assert m.price_scale(Decimal("12.34565")) == Decimal("12.3457") and m.cents(Decimal("0.125")) == Decimal("0.13")      # the existing named roundings are the same functions


def test_the_pydantic_types_run_the_door_before_pydantic_and_write_numbers_in_json():
    row = Row(amount=-4907.92, cash=0, paid=Decimal("10000.00"), at=158.32, chase=0.01)
    assert (row.amount, row.cash, row.paid, row.at, row.chase) == (Decimal("-4907.92"), Decimal(0), Decimal("10000.00"), Decimal("158.3200"), Decimal("0.01"))
    assert all(type(v) is Decimal for v in (row.amount, row.cash, row.paid, row.at, row.chase))
    assert row.model_dump_json() == '{"amount":-4907.92,"cash":0,"paid":10000,"at":158.32,"chase":0.01}' and row.model_dump(mode="json")["paid"] == 10000
    for field, refused in (("amount", "2"), ("amount", True), ("amount", 0.1 + 0.2), ("cash", -1), ("paid", 0), ("at", "158.32"), ("at", False), ("chase", "0.01")):
        with pytest.raises(ValidationError):
            Row(**{**row.model_dump(), field: refused})


def test_a_decimal_is_written_as_a_number_never_a_string_and_a_digit_is_never_dropped_in_silence():
    amounts = {"whole": Decimal("10000.00"), "cents": Decimal("4907.92"), "shares": Decimal("63.163213"), "price": Decimal("158.3200"), "neg": Decimal("-0.5"), "zero": Decimal("0E-6")}
    text = m.dumps({"amounts": amounts, "list": [Decimal("81"), (Decimal("0.01"),)], "plain": {"when": __import__("datetime").date(2026, 11, 13), "n": 3, "f": 2.5}})
    assert json.loads(text) == {"amounts": {"whole": 10000, "cents": 4907.92, "shares": 63.163213, "price": 158.32, "neg": -0.5, "zero": 0}, "list": [81, [0.01]], "plain": {"when": "2026-11-13", "n": 3, "f": 2.5}}
    assert '"10000' not in text and '"4907.92"' not in text                                                          # a number, not a string
    assert type(m.json_number(Decimal("81.000000"))) is int and type(m.json_number(Decimal("81.000001"))) is float
    for too_long in (Decimal("0.12345678901234567890"), Decimal("1234567890123456.78"), Decimal("63.1632130000000000000001")):
        with pytest.raises(ValueError):
            m.dumps({"x": too_long})                                                                                 # more digits than a float holds: refused, not shortened
    with pytest.raises(ValueError):
        m.json_number(Decimal("NaN"))
    with pytest.raises(ValueError):
        m.json_number(1.5)                                                                                           # only a Decimal goes through here


def test_a_real_sessions_numbers_go_through_the_doors_and_the_encoder_and_come_back_the_same():
    """The live session's decisions are floats once `json` has read them: each money amount, price and share count goes through its door and is written again as the same number."""
    assert len(DECISIONS) == 5
    for d in DECISIONS:
        dollars, at, est = m.money(d["dollars"]), m.price(d["price"]), m.money(d["est_cost"])
        assert dollars == Decimal(str(d["dollars"])) and at == Decimal(str(d["price"])).quantize(Decimal("0.0001")) and est == dollars
        again = json.loads(m.dumps({"dollars": dollars, "price": at, "est_cost": est, "pot": m.money(d["size"]["pot"])}))
        assert again == {"dollars": d["dollars"], "price": d["price"], "est_cost": d["est_cost"], "pot": d["size"]["pot"]}
        assert not isinstance(again["dollars"], str)


def test_a_float_given_to_the_arithmetic_is_a_type_error_and_the_door_is_where_it_is_taken():
    """`Decimal * float` raising is WANTED (spec D4): a float that was missed stops a test instead of putting a wrong cent in the account. The place to take a float is a door, once."""
    from prometheus.common.portfolio import affordable_shares, trade_amount

    with pytest.raises(TypeError):
        trade_amount(3, 158.32)
    with pytest.raises(TypeError):
        affordable_shares(1000.0, Decimal("158.32"), 3)
    with pytest.raises(TypeError):
        Decimal("158.32") * 3.5
    with pytest.raises(TypeError):
        m.exact(158.32)
    with pytest.raises(ValueError):
        m.exact(True)
    assert trade_amount(3, m.price(158.32)) == Decimal("474.96") and trade_amount(Decimal("63.163213"), m.price(158.32)) == Decimal("10000.00")
    assert m.exact(Decimal("1.5")) == Decimal("1.5") and m.exact(3) == Decimal(3)
