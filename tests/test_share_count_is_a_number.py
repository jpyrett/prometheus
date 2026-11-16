"""A SHARE COUNT IS A NUMBER, and money from a count and a price is `trade_amount` (design choices entry 111, M5; design/dollar_orders_spec.md section 7).

Jacob (2026-11-15): *"True as a share count is now refused and is an error. Getting any numeric string like "2" is an error also. All the share counts should be decimal and not string when arithmetic is used with counts
and prices."* A bool and a string (a numeric string such as "2" too) are refused at every door that takes a count: RaiseFunds' reading of the model's thoughts (it turned "2" into 2 until now), the purchase the Account Manager
is asked to plan, the how-much answer of the model, the form of a direct order, and a funding sale. TEXT the user typed (`intent.py`'s phrases, the command line's `MSFT:10`) is read into a number once, at the edge, and stays.

REAL DATA ONLY: the real prices of the real sessions (`tests/fixtures/session_real/`), the real sample portfolio's holdings, the real model's reply shape (`parse_thoughts` is the real parser), and the real recorded fill of
`bill_gates_direct_order_company_auto.json`: 50 MSFT at 524.1419, which is $26,207.10 to the cent (half up, as the book works it) and was $26,207.09 as a float product rounded.
"""
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_direct_order import form
from test_fractional_even_split import PRICES

from prometheus.agents.account_manager import clarify
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager import sizing
from prometheus.agents.account_manager.account_manager import AccountManager, parse_purchase, parse_thoughts
from prometheus.agents.trader import tickets as tk
from prometheus.application import session
from prometheus.application.web import views
from prometheus.application.web.replay import Replay, load_capture
from prometheus.common.money import money, price
from prometheus.common.portfolio import Portfolio, Position, apply_fill, trade_amount
from prometheus.reasoning import raise_funds as rf

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
SAMPLE = Portfolio(user="Fraction Holder", cash_avail=Decimal("51.23"), positions=[Position(symbol="MSFT", shares=50, cost_basis=Decimal("150")), Position(symbol="GM", shares=60, cost_basis=Decimal("10"))])
THOUGHT = '{"thoughts": [{"symbol": "GM", "shares": %s, "idea": "sell the one that is up"}]}'      # the shape of the reply `AccountManager._generate` reads


def problem():
    return rf.build_problem(SAMPLE, "MU", 2, {**PRICES}, date(2026, 11, 13), only_green=True)


@pytest.mark.parametrize("said", ['"2"', '" 2 "', "true", '"2.0"'])
def test_a_models_thought_that_says_shares_in_text_or_true_is_rejected_and_the_model_is_told_why(said):
    thought = parse_thoughts(THOUGHT % said)[0]
    node, reason = rf.check_step(problem(), rf.root(problem()), thought, "n1")
    assert node is None and "shares must be a whole number of at least 1 (a number, not text or true/false" in reason
    assert rf._clean(thought) == ("GM", None)
    node, reason = rf.check_step(problem(), rf.root(problem()), parse_thoughts(THOUGHT % "2")[0], "n1")
    assert reason is None and node.sells == (("GM", 2),)                                                 # the number itself is as good as ever


@pytest.mark.parametrize("count", [True, "2", "2.0", 0, -1])      # entry 112: 2.0 is not in this list any more (below)
def test_the_purchase_to_plan_is_a_whole_number_and_a_bool_a_string_or_a_count_below_one_is_refused(count):
    with pytest.raises(ValueError, match="whole number of at least 1"):
        AccountManager._check_purchase("MU", count)
    with pytest.raises(ValueError, match="whole number of at least 1"):
        rf.build_problem(SAMPLE, "MU", count, {**PRICES}, date(2026, 11, 13))
    result = AccountManager(model=object(), model_id="x").raise_funds(SAMPLE, "MU", count)
    assert result["status"] == "refused" and "whole number of at least 1" in result["errors"][0]
    assert AccountManager._check_purchase("mu", 2) == "MU" and AccountManager._check_purchase("MU", None) == "MU" and AccountManager._check_purchase("MU", 2.0) == "MU"      # entry 112: a float is taken by its digits at the share door (`shares.quantity`), so 2.0 is the whole number 2; it was refused while a count had to be an int
    assert parse_purchase("MU:10", None) == ("MU", 10)                                                   # the user's own TEXT is read into a number once, at the edge


def test_the_models_how_much_answer_in_shares_is_a_number_and_text_or_true_is_an_error_it_is_told_about():
    assert clarify.SizingChoice(unit="shares", amount=10, scope="each").amount == 10
    for amount in ("10", True):
        with pytest.raises(ValueError, match="a number of shares is a number"):
            clarify.SizingChoice(unit="shares", amount=amount, scope="each")
    assert clarify.SizingChoice(unit="dollars", amount=3000).amount == 3000                              # money is not a count


def test_a_direct_orders_form_and_a_funding_sale_take_a_number_of_shares_and_nothing_else():
    typed = "Buy 50 shares of MSFT"
    assert it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": 50}]), typed).orders[0].shares == 50
    for shares in ('"50"', "true"):
        with pytest.raises(ValueError, match="orders must be a list"):
            it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": json.loads(shares)}]), typed)
    with pytest.raises(ValueError):
        sizing.size_purchases([sizing.Purchase(record_id="r", symbol="MSFT", action="open", price=PRICES["MSFT"])], Decimal("1000.00"), wanted={"r": "2"})
    for count in ("2", True):
        target, legs, note = tk.funding_legs({"status": "plan", "target": "MSFT", "plan": {"sells": [{"symbol": "GM", "shares": count, "price": PRICES["GM"]}]}}, [_ticket()], SAMPLE, date(2026, 11, 13))
        assert legs == [] and "has no whole number of shares" in note
    with pytest.raises(ValueError, match="number above zero"):
        apply_fill(SAMPLE, "GM", "sell", "2", PRICES["GM"], date(2026, 11, 13))


def _ticket():
    from prometheus.agents.trader.record import Trade
    return Trade(trade_id="T1", kind="recommendation", record_id="r", symbol="MSFT", side="buy", shares_approved=1, shares=1, status="new", mode="paper")


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
