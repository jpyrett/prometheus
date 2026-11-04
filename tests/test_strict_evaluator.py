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

def test_with_never_liquidate_the_shares_that_may_be_sold_are_one_fewer_than_held():
    p = rf.build_problem(HELD, "MU", 1, PRICES, TODAY, only_green=True, never_liquidate=True)
    assert {s: (h.shares, h.sellable) for s, h in p.holdings.items()} == {"GM": (10, 9), "F": (40, 39), "TSLA": (20, 19), "MSFT": (3, 2)}
    plain = rf.build_problem(HELD, "MU", 1, PRICES, TODAY)
    assert {s: h.sellable for s, h in plain.holdings.items()} == {"GM": 10, "F": 40, "TSLA": 20, "MSFT": 3}   # off: as before


def test_shares_locked_by_t_plus_one_already_count_among_the_ones_kept():
    held = pf.Portfolio(user="x", positions=[pf.Position(symbol="F", shares=560, cost_basis=7.0, acquired=TODAY, recent_shares=300)])
    p = rf.build_problem(held, "MU", 1, {"MU": 410.0, "F": 12.0}, TODAY, never_liquidate=True)
    assert p.holdings["F"].sellable == 260 and rf.build_problem(held, "MU", 1, {"MU": 410.0, "F": 12.0}, TODAY).holdings["F"].sellable == 260   # the 300 locked shares are kept anyway: no further share is held back
    unlocked = pf.Portfolio(user="x", positions=[pf.Position(symbol="F", shares=5, cost_basis=7.0, acquired=date(2026, 1, 1))])
    assert rf.build_problem(unlocked, "MU", 1, {"MU": 410.0, "F": 12.0}, TODAY, never_liquidate=True).holdings["F"].sellable == 4   # nothing locked: one share is held back
    locked_whole = pf.Portfolio(user="x", positions=[pf.Position(symbol="F", shares=5, cost_basis=7.0, acquired=TODAY)])
    assert rf.build_problem(locked_whole, "MU", 1, {"MU": 410.0, "F": 12.0}, TODAY, never_liquidate=True).holdings["F"].sellable == 0


def test_selling_the_whole_position_is_rejected_and_one_share_fewer_is_allowed():
    p = rf.build_problem(HELD, "MU", 1, PRICES, TODAY, only_green=True, never_liquidate=True)
    node, reason = rf.check_step(p, rf.root(p), {"symbol": "MSFT", "shares": 3}, "n1")
    assert node is None and "only 2 are sellable (3 held, and one share of every holding is always kept: a position is never sold out)" in reason
    assert rf.check_step(p, rf.root(p), {"symbol": "MSFT", "shares": 2}, "n1")[0] is not None


def test_the_most_that_could_be_bought_does_not_count_the_shares_that_are_kept():
    keep = rf.build_problem(HELD, "MU", 1, PRICES, TODAY, only_green=True, never_liquidate=True)
    all_of_it = rf.build_problem(HELD, "MU", 1, PRICES, TODAY, only_green=True)
    assert keep.max_affordable == int((9 * 55 + 39 * 12 + 2 * 420) // 410) == 4 and all_of_it.max_affordable == int((10 * 55 + 40 * 12 + 3 * 420) // 410) == 5


def test_the_prompt_says_one_share_of_every_holding_is_kept():
    from prometheus.prompts import account_manager as prompts
    p = rf.build_problem(HELD, "MU", 1, PRICES, TODAY, only_green=True, never_liquidate=True)
    assert "One share of every holding is always kept." in prompts.user_message(p, rf.root(p), [])
    assert "Never sell a position out" in prompts.system_prompt(3, only_green=True, never_liquidate=True) and "Never sell a position out" not in prompts.system_prompt(3, only_green=True)


# ---------------------------------------------------------------- strict by default in the Account Manager

def test_the_account_manager_is_strict_unless_told_otherwise():
    manager = am.AccountManager(model=object(), clerk=object())
    assert manager.only_green is True and manager.never_liquidate is True
    relaxed = am.AccountManager.classic(model=object(), clerk=object())
    assert relaxed.only_green is False and relaxed.never_liquidate is False


def test_by_default_a_sale_at_a_loss_and_a_sale_that_sells_a_position_out_are_both_rejected():
    manager = agent({ROOT: [("TSLA", 5), ("MSFT", 3), ("GM", 9), ("F", 39)], "GM x9": [("F", 31)]})
    result = manager.raise_funds(HELD, "MU", 1, today=TODAY, branching=5)
    reasons = rejected(result)
    assert "TSLA is not in profit (price $250.00 is not above its basis $300.00)" in reasons["TSLA5"]
    assert "one share of every holding is always kept" in reasons["MSFT3"]
    assert result["status"] == "plan" and result["only_green"] is True and result["never_liquidate"] is True


def test_the_plan_the_strict_evaluator_accepts_meets_the_success_measure():
    manager = agent({ROOT: [("GM", 9)], "GM x9": [("F", 28)]})
    result = manager.raise_funds(HELD, "MU", 2, today=TODAY)       # 2 MU = 820: GM x9 (495), then F x28 (336) finishes it
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]][0] == ("GM", 9)
    ok = rf.success(result)
    assert ok == {"profitable": True, "kept": True, "asked": 2, "wanted": 2, "bought": 2, "all_bought": True, "ok": True}
    assert all(x["gain"] > 0 and not x["closes_position"] for x in result["plan"]["sells"])
    text = am.report(result)
    assert "CHECK    every sale in profit: yes | no position sold out: yes | shares bought: 2 of 2 asked -> SUCCESS" in text


def test_the_relaxed_account_manager_may_take_a_loss_and_sell_a_position_out():
    manager = agent({ROOT: [("TSLA", 4)]}, only_green=False, never_liquidate=False)   # 4 x 250 = 1,000: a loss of 200
    result = manager.raise_funds(HELD, "MU", 2, today=TODAY)
    assert result["status"] == "plan" and result["plan"]["gain"] == -200.0
    assert rf.success(result)["profitable"] is False and "every sale in profit: NO" in am.report(result) and "NOT MET" in am.report(result)


def test_a_method_argument_overrides_the_default():
    manager = agent({ROOT: [("TSLA", 4)]})   # strict by default
    result = manager.raise_funds(HELD, "MU", 2, today=TODAY, only_green=False)
    assert result["status"] == "plan" and result["plan"]["sells"][0]["symbol"] == "TSLA"


def test_with_everything_kept_a_purchase_the_holdings_could_have_paid_for_is_no_longer_possible():
    strict = agent({}).raise_funds(HELD, "MU", 5, today=TODAY, up_to=True)   # the holdings in profit, with one share of each kept, are worth 1,803: 4 MU at most
    loose = agent({}, never_liquidate=False).raise_funds(HELD, "MU", 6, today=TODAY, up_to=True)   # sold out they are worth 2,290: 5 MU
    assert strict["most_affordable"] == 4 and loose["most_affordable"] == 5


# ---------------------------------------------------------------- the success measure

def test_the_success_measure_for_the_cases_it_names():
    plan = {"sells": [{"gain": 100.0, "closes_position": False}], "gain": 100.0}
    base = {"status": "plan", "plan": plan, "shares": 3, "requested": 3}
    assert rf.success(base)["ok"] is True
    assert rf.success({**base, "plan": {"sells": [{"gain": -1.0, "closes_position": False}]}})["ok"] is False        # a loss
    assert rf.success({**base, "plan": {"sells": [{"gain": 5.0, "closes_position": True}]}})["ok"] is False          # a position sold out
    assert rf.success({**base, "shares": 2})["ok"] is False and rf.success({**base, "shares": 2})["all_bought"] is False   # fewer than asked
    assert rf.success({**base, "requested": None, "shares": 7})["ok"] is True                                         # as many as possible: any purchase
    assert rf.success({"status": "infeasible", "plan": None, "shares": 3, "requested": 3})["ok"] is False             # no plan
    assert rf.success({"status": "funded", "plan": None, "shares": 3, "requested": 3})["ok"] is True                  # the cash paid for it
    assert rf.success({**base, "requested": 40, "resolved_shares": 3, "shares": 3})["ok"] is True                    # up to 40, the portfolio could pay for 3: all 3 bought is success
    assert rf.success({**base, "requested": 40, "resolved_shares": 5, "shares": 3})["ok"] is False                   # ... but 3 of the 5 it could pay for is not


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


def test_a_depth_that_was_asked_for_is_never_changed_and_auto_depth_can_be_turned_off():
    manager = big_agent({ROOT: [("TSLA", 5)], "TSLA x5": [("MSFT", 2)]})
    asked = manager.raise_funds(BIG, "MU", 3, up_to=True, max_depth=3)
    assert asked["limits"]["max_depth"] == 3 and "auto_depth" not in asked
    off = big_agent({ROOT: [("TSLA", 5)], "TSLA x5": [("MSFT", 2)]}).raise_funds(BIG, "MU", 3, up_to=True, auto_depth=False)
    assert off["limits"]["max_depth"] == 3 and "auto_depth" not in off


def test_the_call_budget_grows_with_the_depth_that_was_chosen_so_that_the_depth_can_be_reached():
    manager = big_agent({})
    result = manager.raise_funds(BIG, "MU", 3, up_to=True)
    depth = result["limits"]["max_depth"]
    assert result["limits"]["max_calls"] >= 1 + 3 * (depth - 1)   # one call at the root and a beam of three for every level below it, at least
    explicit = big_agent({}).raise_funds(BIG, "MU", 3, up_to=True, max_calls=5)
    assert explicit["limits"]["max_calls"] == 5   # a budget that was asked for is kept
