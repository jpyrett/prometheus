"""The evaluator only sells holdings in profit and never sells a position out (design choices entry 47), and the success measure (Jacob, 2026-11-03).

Run:  pytest tests/test_strict_evaluator.py -q

"Best score = able to find profitable stocks to sell and was able to buy the requested shares is the success metric. Change the evaluator to only sell profitable shares and never liquidate or take a loss
on shares when selling them." Two hard checks in code, ON BY DEFAULT in the Account Manager: `only_green` (the price is above the cost basis: no sale takes a loss) and `never_liquidate` (one share of every
holding is kept: no position is sold out). A scripted model proposes thoughts that break them; the code rejects them. No network, no key.
"""
from datetime import date

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf
from test_sndk_mu import ROOT, QuoteToolbox, ScriptedByPlan

PRICES = {"MU": 410.0, "GM": 55.0, "F": 12.0, "TSLA": 250.0, "MSFT": 420.0}   # TSLA is below its basis of 300: a loss
HELD = pf.Portfolio(user="x", cash_avail=0.0, positions=[pf.Position(symbol="GM", shares=10, cost_basis=10.0), pf.Position(symbol="F", shares=40, cost_basis=2.0),
                                                       pf.Position(symbol="TSLA", shares=20, cost_basis=300.0), pf.Position(symbol="MSFT", shares=3, cost_basis=150.0)])
TODAY = date(2026, 11, 3)


def agent(table, **options):
    toolbox, model = QuoteToolbox(PRICES), ScriptedByPlan(table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()), **{"only_green": True, "never_liquidate": True, **options})
    manager.toolbox, manager.scripted = toolbox, model
    return manager


def rejected(result):
    return {f"{c['thought']['symbol']}{c['thought']['shares']}": c["reason"] for level in result["trace"] for c in level["children"] if c["status"] == "rejected"}


# ---------------------------------------------------------------- never liquidate: one share of every holding is kept


# ---------------------------------------------------------------- strict by default in the Account Manager


# ---------------------------------------------------------------- the success measure


# ---------------------------------------------------------------- the Account Manager chooses the depth a purchase needs (entry 49)

BIG = pf.Portfolio(user="x", cash_avail=0.0, positions=[pf.Position(symbol=s, shares=n, cost_basis=1.0) for s, n in (("GM", 4), ("F", 40), ("MSFT", 3), ("TSLA", 6))])
BIG_PRICES = {"MU": 600.0, "GM": 55.0, "F": 12.0, "MSFT": 420.0, "TSLA": 250.0}   # kept shares make: GM 3 x 55 = 165, F 39 x 12 = 468, MSFT 2 x 420 = 840, TSLA 5 x 250 = 1,250: 2,723 in all


def big_agent(table, **options):
    toolbox, model = QuoteToolbox(BIG_PRICES), ScriptedByPlan(table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()), **{"only_green": True, "never_liquidate": True, **options})
    manager.toolbox, manager.scripted = toolbox, model
    return manager


def test_the_depth_is_chosen_from_the_purchase_when_it_is_not_asked_for_and_the_report_says_so():
    manager = big_agent({ROOT: [("TSLA", 5)], "TSLA x5": [("MSFT", 2)], "TSLA x5, MSFT x2": [("F", 18)]})   # 3 MU = 1,800: 1,250 + 840 covers it with two stocks; the default depth of 3 would do
    result = manager.raise_funds(BIG, "MU", 3, today=None, up_to=True)
    assert result["auto_depth"] is True and result["fewest_stocks"] == 2 and result["limits"]["max_depth"] == 4 and result["status"] == "plan"   # 2 + 2 = 4 steps allowed, three holdings...
    assert "DEPTH: not asked for, so chosen from the purchase: it needs at least 2 stock(s) sold, so up to 4 steps" in am.report(result)


