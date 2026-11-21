"""Sell the most profitable first, and how many shares to buy (design choices entry 45).

Run:  pytest tests/test_green_first.py -q

Jacob's rules (2026-11-03): "sell the most profitable (stocks most in the green first)"; "buy up to X shares (3 in this example); if X is not provided and the recommendation is a single stock, buy as many as
possible". WHAT TO SELL is still decided by the Tree of Thought: the arithmetic of `trading_math` only measures a plan the tree has built (the green-first component of its score), and the number of shares
is arithmetic (`Problem.max_affordable`), never a plan. The "LLM" is scripted, the prices are the recorded 2026-11-03 quotes, no network, no key, no cost.
"""
from decimal import Decimal

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.prompts import account_manager as prompts
from prometheus.reasoning import raise_funds as rf
from test_sndk_mu import LIVE_RUN_3, SCENARIO, QuoteToolbox, ScriptedByPlan

TODAY = None


@pytest.fixture
def start():
    return pf.load_portfolio(SCENARIO)   # PG 15 @ 60, MCD 120 @ 40, GOOGL 30 @ 100, LLY 25 @ 155, $200


# ---------------------------------------------------------------- the score: green-first replaces the tax component


def test_when_nothing_is_sold_or_every_holding_has_the_same_percent_gain_the_component_is_one():
    holdings = {"A": rf.Holding("A", None, Decimal("10"), Decimal("10"), Decimal("20.0"), Decimal("10.0")), "B": rf.Holding("B", None, Decimal("10"), Decimal("10"), Decimal("40.0"), Decimal("20.0"))}   # both +100%
    p = rf.Problem("T", 1, Decimal("100.0"), Decimal("0.0"), holdings, green_first=True)
    assert rf.green_component(p, Decimal("0.0"), Decimal("0.0")) == 1.0 and rf.green_component(p, Decimal("300.0"), Decimal("150.0")) == 1.0


# ---------------------------------------------------------------- the tree decides, with the new score

def agent(table=None, prices=None):
    toolbox, model = QuoteToolbox(prices), ScriptedByPlan(LIVE_RUN_3 if table is None else table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()))
    manager.toolbox, manager.scripted = toolbox, model
    return manager


def test_the_tree_now_picks_the_most_profitable_holding_and_the_cash_left_buys_a_share_of_mu(start):
    manager = agent()
    result = manager.raise_funds(start, "SNDK", 3, fill="MU", explore=True, green_first=True)
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("LLY", 3)] and result["plan"]["score"] == 98.6
    assert result["plan"]["fill_shares"] == 1 and result["plan"]["left_over"] == Decimal("193.48") and result["weights"]["green"] == 0.4 and result["green_first"] is True
    assert result["after"]["shares"] == {"PG": 15, "MCD": 120, "GOOGL": 30, "LLY": 22, "SNDK": 3, "MU": 1}
    assert result["stats"]["levels"] == 3   # the same tree as before: it is the score that changed, and the answer with it
    text = am.report(result)
    assert "green 1.0" in text and "weights: green 0.4, overshoot 0.25, keep 0.2, orders 0.15" in text and "| BUY   | MU" in text


# ---------------------------------------------------------------- how many shares: up to X, or as many as possible

SMALL = pf.Portfolio(user="x", cash_avail=Decimal("0"), positions=[pf.Position(symbol="GM", shares=10, cost_basis=Decimal("10.0")), pf.Position(symbol="F", shares=40, cost_basis=Decimal("2.0"))])
SMALL_PRICES = {"MU": 410.0, "GM": 55.0, "F": 12.0, "BIG": 5000.0}   # GM 10 x 55 = 550 and F 40 x 12 = 480: 1,030 in all, so 2 MU (820) and not 3 (1,230)


def small_agent(table):
    toolbox, model = QuoteToolbox(SMALL_PRICES), ScriptedByPlan(table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()))
    manager.toolbox, manager.scripted = toolbox, model
    return manager


def test_if_not_even_one_share_can_be_paid_for_nothing_is_sold_and_the_model_is_not_asked():
    for shares in (None, 3):
        manager = small_agent({})
        result = manager.raise_funds(SMALL, "BIG", shares, up_to=True)   # one BIG costs 5,000: more than everything is worth
        assert result["status"] == "infeasible" and result["plan"] is None and result["resolved_shares"] == 0 and manager.scripted.seen == [] and result["stats"]["llm_calls"] == 0
        assert "cannot pay for even one share: nothing is sold" in result["stop"] and "NOTHING IS SOLD" in am.report(result)


def test_several_stocks_with_no_number_of_shares_are_refused_because_as_many_as_possible_needs_one_stock():
    manager = small_agent({})
    result = manager.raise_funds_many(SMALL, [("MU", None), ("F", 2)], today=None)
    assert result["status"] == "refused" and "works for ONE stock" in result["errors"][0] and manager.toolbox.asked == [] and manager.scripted.seen == []
    assert small_agent({}).raise_funds_many(SMALL, [("MU", None)])["status"] != "refused"   # one stock: fine


