"""A SHARE COUNT IS A NUMBER, and money from a count and a price is `trade_amount` (design choices entry 111, M5; design/dollar_orders_spec.md section 7).

Jacob (2026-11-15): *"True as a share count is now refused and is an error. Getting any numeric string like "2" is an error also. All the share counts should be decimal and not string when arithmetic is used with counts
and prices."* A bool and a string (a numeric string such as "2" too) are refused at every door that takes a count: RaiseFunds' reading of the model's thoughts (it turned "2" into 2 until now), the purchase the Account Manager
is asked to plan, the how-much answer of the model, the form of a direct order, and a funding sale. TEXT the user typed (`intent.py`'s phrases, the command line's `MSFT:10`) is read into a number once, at the edge, and stays.

REAL DATA ONLY: the real prices of the real sessions (`tests/fixtures/session_real/`), the real sample portfolio's holdings, the real model's reply shape (`parse_thoughts` is the real parser), and the real recorded fill of
`bill_gates_direct_order_company_auto.json`: 50 MSFT at 524.1419, which is $26,207.10 to the cent (half up, as the book works it) and was $26,207.09 as a float product rounded.
"""
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_direct_order import form

from prometheus.agents.account_manager import clarify
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager import sizing
from prometheus.agents.account_manager.account_manager import parse_thoughts
from prometheus.application import session
from prometheus.application.web import views
from prometheus.application.web.replay import Replay, load_capture
from prometheus.common.money import money, price
from prometheus.common.portfolio import Portfolio, Position, trade_amount
from prometheus.reasoning import raise_funds as rf

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"


def test_the_models_how_much_answer_in_shares_is_a_number_and_text_or_true_is_an_error_it_is_told_about():
    assert clarify.SizingChoice(unit="shares", amount=10, scope="each").amount == 10
    for amount in ("10", True):
        with pytest.raises(ValueError, match="a number of shares is a number"):
            clarify.SizingChoice(unit="shares", amount=amount, scope="each")
    assert clarify.SizingChoice(unit="dollars", amount=3000).amount == 3000                              # money is not a count


def test_money_from_a_count_and_a_price_is_decimal_half_up_to_the_cent_in_every_place_the_survey_found():
    data = json.loads((SESSIONS / "bill_gates_direct_order_company_auto.json").read_text(encoding="utf-8"))
    fill = data["order"]["trades"][0]["fill"]
    assert (fill["shares"], price(fill["price"]), money(fill["amount"])) == (50, Decimal("524.1419"), Decimal("26207.10")) and round(50 * 524.1419, 2) == 26207.09 and trade_amount(50, Decimal("524.1419")) == Decimal("26207.10")      # a float product rounded (the float line) was a cent short
    held = Portfolio(user="Bill Gates", cash_avail=Decimal("0.00"), positions=[Position(symbol="MSFT", shares=50, cost_basis=Decimal("524.1419"))])
    assert held.market_value({"MSFT": Decimal("524.1419")}) == Decimal("26207.10") and sizing.equity_of(held, {"MSFT": Decimal("524.1419")})[0] == Decimal("26207.10")
    assert rf.Holding("MSFT", None, Decimal(50), Decimal(49), Decimal("524.1419"), Decimal("500")).value == Decimal("26207.10") and rf.Problem("MSFT", Decimal(50), Decimal("524.1419"), Decimal("0.00"), {}).cost == Decimal("26207.10")
    holder = SimpleNamespace(account_id=1, display_name="Bill Gates", age=70, risk_tolerance="moderate", trade_mode="ask", seed_key=None)
    assert views.account_view(holder, held)["positions"][0]["value"] == Decimal("26207.10") and "$26,207.10" in session.portfolio_lines(held)[-1]
    assert Replay(load_capture(SESSIONS / "bill_gates_direct_order_company_auto.json")).account(final=True)["positions"][0]["value"] == Decimal("26207.10")
