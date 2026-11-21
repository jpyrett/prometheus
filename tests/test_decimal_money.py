"""Tests for prometheus.common.money (design choices entry 112; design/decimal_money_spec.md): the doors of a dollar amount, a price and a rate, the pydantic types, and the ONE JSON encoder.

Run:  pytest tests/test_decimal_money.py -q

No network, no key, no cost. The numbers are the real ones of the live session `tests/fixtures/session_real/auto_even_split_dollars.json` (its decisions' dollars, prices and shares).
"""
import json
from decimal import Decimal

import pytest

from prometheus.common import money as m


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


