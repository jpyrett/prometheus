"""Decisions to tickets (prometheus/agents/trader/tickets.py, Trader spec sections 5 and 7): sides and sizes, the T+1 cut, the limits with and without an entry zone, each rejection code, and the funding plan: accepted, and each
way it is refused. Code only: no network, no model."""
from datetime import date, datetime

import pytest
from trader_helpers import NOW, account, at, config, plan, row, synthetic

from prometheus.agents.trader import tickets
from prometheus.agents.trader.record import REJECT_CODES, Trade

TODAY = date(2026, 11, 5)
CFG = config()


def build(order, portfolio=None, now=NOW, cfg=CFG, mode="paper"):
    return tickets.build_tickets(order, portfolio or account(), now, cfg, mode)


def one(order, **kw) -> Trade:
    found = build(order, **kw)
    assert len(found) == 1, found
    return found[0]


# ---------------------------------------------------------------- the rejections


def test_every_rejection_a_ticket_can_be_built_with_is_in_the_closed_list():
    order = synthetic(row("A", "open", 1, 10.0, approved=False), row("B", "reduce", 1, 10.0), row("C", "open", 1, 10.0, zone=(20.0, 21.0)), row("D", "open", 1, 10.0))
    order.recommendations[2]["trade"]["entry_mid"] = None      # AMENDED (entry 105): B, the sale, is the user's own direct order now and has no recommendation or review, so D is the third of each
    order.reviews[2]["timing"] = {}
    codes = {t.reject_code for t in build(order, cfg=config(limits={"max_trades": 5}))} | {t.reject_code for t in build(order, cfg=config(limits={"max_trades": 1}))}
    assert codes - {None} == {"not_permitted", "not_held", "price_outside_limits", "price_unavailable", "not_attempted"} and codes - {None} <= set(REJECT_CODES)


# ---------------------------------------------------------------- the funding plan

def tickets_for(*rows, portfolio=None):
    return build(synthetic(*rows), portfolio or account(MSFT=(50, 150.0), TSLA=(20, 300.0), GM=(60, 10.0)))


def legs(funding, found, portfolio=None, today=TODAY):
    return tickets.funding_legs(funding, found, portfolio or account(MSFT=(50, 150.0), TSLA=(20, 300.0), GM=(60, 10.0)), today)


@pytest.mark.parametrize("funding, expected", [
    (None, ""),
    ("a plan", "the funding answer is not a record"),
    ({"status": "funded", "target": "PG", "plan": None}, "the cash already covers the purchase"),
    ({"status": "infeasible", "target": "PG", "plan": None}, "'infeasible', not a plan with sells"),
    ({"status": "no_prices", "target": "PG", "plan": None}, "'no_prices', not a plan with sells"),
    ({"status": "plan", "target": "PG", "plan": None}, "not a plan with sells"),
    ({"status": "plan", "target": "PG", "plan": {"sells": []}}, "not a plan with sells"),
    ({"status": "plan", "target": "PG", "plan": {"sells": "MSFT"}}, "not a plan with sells"),
])
def test_a_funding_answer_that_is_not_a_plan_with_sells_adds_none_and_the_run_notes_why(funding, expected):
    target, sells, note = legs(funding, tickets_for(row("PG", "open", 44, 167.1)))
    assert (target, sells) == (None, []) and expected in note


@pytest.mark.parametrize("sells, expected", [
    ([("AAPL", 5, 420.0)], "AAPL, which is not a holding"),
    ([("PG", 5, 420.0)], "PG, which is not a holding"),                                                              # PG is the target and not held here: the holding check comes first
    ([("MSFT", 51, 420.0)], "is for 51 shares but only 50 are sellable today"),
    ([("MSFT", 0, 420.0)], "has no whole number of shares"),
    ([("MSFT", -2, 420.0)], "has no whole number of shares"),
    ([("MSFT", 2.5, 420.0)], "has no whole number of shares"),
    ([("MSFT", True, 420.0)], "has no whole number of shares"),
    ([("MSFT", None, 420.0)], "has no whole number of shares"),
    ([("MSFT", 5, None)], "has no price"),
    ([("MSFT", 5, 0)], "has no price"),
    ([("MSFT", 5, float("nan"))], "has no price"),
    ([("MSFT", 5, 420.0), ("MSFT", 1, 420.0)], "MSFT, which appears twice in the plan"),
    ([("MSFT", 5, 420.0), ("AAPL", 1, 10.0)], "AAPL, which is not a holding"),                                       # one bad sell: no holding is sold at all
])
def test_one_bad_sell_and_no_holding_is_sold_all_or_nothing(sells, expected):
    found = tickets_for(row("PG", "open", 44, 167.1))
    target, got, note = legs(plan("PG", *sells), found)
    assert (target, got) == (None, []) and expected in note and note.endswith("no holdings are sold")


def test_the_stock_being_bought_is_never_sold_to_pay_for_itself_even_when_it_is_held():
    held = account(PG=(10, 100.0), MSFT=(50, 150.0))
    found = tickets_for(row("PG", "open", 44, 167.1), portfolio=held)
    assert "PG, which is the stock being bought" in legs(plan("PG", ("PG", 5, 167.1)), found, portfolio=held)[2]
