"""Sell the most profitable first, and how many shares to buy (design choices entry 45).

Run:  pytest tests/test_green_first.py -q

Jacob's rules (2026-11-03): "sell the most profitable (stocks most in the green first)"; "buy up to X shares (3 in this example); if X is not provided and the recommendation is a single stock, buy as many as
possible". WHAT TO SELL is still decided by the Tree of Thought: the arithmetic of `trading_math` only measures a plan the tree has built (the green-first component of its score), and the number of shares
is arithmetic (`Problem.max_affordable`), never a plan. The "LLM" is scripted, the prices are the recorded 2026-11-03 quotes, no network, no key, no cost.
"""
import re

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.prompts import account_manager as prompts
from prometheus.reasoning import raise_funds as rf
from prometheus.tools import trading_math as tm
from test_sndk_mu import LIVE_RUN_3, PRICES, ROOT, SCENARIO, QuoteToolbox, ScriptedByPlan

TODAY = None


@pytest.fixture
def start():
    return pf.load_portfolio(SCENARIO)   # PG 15 @ 60, MCD 120 @ 40, GOOGL 30 @ 100, LLY 25 @ 155, $200


def green_problem(start, **options):
    return rf.build_problem(start, "SNDK", 3, PRICES, fill="MU", green_first=True, **options)


# ---------------------------------------------------------------- the arithmetic that measures a plan

def test_holdings_are_ranked_by_percent_gain_highest_first():
    holdings = [{"symbol": "PG", "shares": 15, "price": 162.06, "basis": 60.0}, {"symbol": "LLY", "shares": 25, "price": 1409.38, "basis": 155.0},
                {"symbol": "GOOGL", "shares": 30, "price": 280.03, "basis": 100.0}, {"symbol": "MCD", "shares": 120, "price": 263.04, "basis": 40.0}]
    ranked = tm.rank_green(holdings)
    assert [h["symbol"] for h in ranked] == ["LLY", "MCD", "GOOGL", "PG"]
    assert [round(tm.pct_gain(h["price"], h["basis"]), 3) for h in ranked] == [8.093, 5.576, 1.8, 1.701]  # +809%, +558%, +180%, +170%


def test_percent_gain_ranks_holdings_in_the_same_order_as_profit_per_dollar_raised():
    holdings = [{"symbol": s, "shares": 1, "price": p, "basis": b} for s, p, b in (("A", 10.0, 2.0), ("B", 100.0, 60.0), ("C", 50.0, 5.0), ("D", 8.0, 7.0))]
    by_pct = [h["symbol"] for h in tm.rank_green(holdings)]
    by_profit_per_dollar = [h["symbol"] for h in sorted(holdings, key=lambda h: -(h["price"] - h["basis"]) / h["price"])]
    assert by_pct == by_profit_per_dollar == ["C", "A", "B", "D"]


def test_only_green_leaves_out_holdings_at_or_below_their_basis_and_empty_ones():
    holdings = [{"symbol": "UP", "shares": 1, "price": 10.0, "basis": 5.0}, {"symbol": "EVEN", "shares": 1, "price": 10.0, "basis": 10.0}, {"symbol": "DOWN", "shares": 1, "price": 8.0, "basis": 10.0},
                {"symbol": "NONE", "shares": 0, "price": 10.0, "basis": 1.0}]
    assert [h["symbol"] for h in tm.rank_green(holdings, only_green=True)] == ["UP"]
    assert [h["symbol"] for h in tm.rank_green(holdings)] == ["UP", "EVEN", "DOWN"]


def test_the_most_and_least_profit_that_selling_an_amount_could_realise(start):
    holdings = rf.build_problem(start, "SNDK", 3, PRICES).holdings_list()
    best, worst = tm.profit_bounds(holdings, 3179.56)
    assert best == pytest.approx(3179.56 * (1409.38 - 155) / 1409.38, abs=0.01)   # all of it from LLY, the most in the green
    assert worst == pytest.approx(2012.21, abs=0.01)                               # PG's $2,430.90 first, the rest from GOOGL
    assert tm.profit_bounds(holdings, 0.0) == (0.0, 0.0)
    everything = sum(h["shares"] * (h["price"] - h["basis"]) for h in holdings)
    assert tm.profit_bounds(holdings, 10 ** 9) == (pytest.approx(everything, abs=0.01), pytest.approx(everything, abs=0.01))   # more than it is worth: all of it, in either order


# ---------------------------------------------------------------- the score: green-first replaces the tax component

def test_green_first_scores_the_plan_that_sells_the_most_profitable_holding_highest(start):
    p = green_problem(start)
    lly, mcd, googl = (rf.score_plan(p, plan) for plan in ((("LLY", 3),), (("MCD", 13),), (("GOOGL", 12),)))
    assert (lly[0], mcd[0], googl[0]) == (98.6, 91.7, 60.1) and lly[1]["green"] == 1.0 and "gain" not in lly[1]
    assert lly[1] == {"green": 1.0, "keep": 1.0, "orders": 1.0, "overshoot": 0.943}


def test_the_tax_aware_score_ranks_the_same_plans_the_other_way_round(start):
    p = rf.build_problem(start, "SNDK", 3, PRICES, fill="MU")   # green_first off: the earlier entries' score
    scores = {plan[0][0]: rf.score_plan(p, plan)[0] for plan in ((("LLY", 3),), (("MCD", 13),), (("GOOGL", 12),))}
    assert scores == {"LLY": 63.0, "MCD": 64.3, "GOOGL": 72.9}   # the least taxable gain first: the opposite of "the most in the green first"
    assert "green" not in rf.score_plan(p, (("LLY", 3),))[1] and p.weights == rf.WEIGHTS


def test_green_first_has_its_own_weights_with_the_same_numbers_and_a_name(start):
    p = green_problem(start)
    assert p.weights == {"green": 0.40, "overshoot": 0.25, "keep": 0.20, "orders": 0.15} and sum(p.weights.values()) == pytest.approx(1.0)


def test_a_plan_with_several_stocks_is_scored_by_how_far_it_departs_from_most_profitable_first(start):
    p = green_problem(start)
    mixed = rf.score_plan(p, (("LLY", 2), ("MCD", 1), ("PG", 1)))   # the live run 1 plan: mostly the most profitable
    worst = rf.score_plan(p, (("PG", 11), ("GOOGL", 5)))              # the least profitable first
    assert mixed[1]["green"] == pytest.approx(0.936, abs=0.001) and worst[1]["green"] == pytest.approx(0.01, abs=0.01) and mixed[0] > 80 > worst[0]


def test_when_nothing_is_sold_or_every_holding_has_the_same_percent_gain_the_component_is_one():
    holdings = {"A": rf.Holding("A", None, 10, 10, 20.0, 10.0), "B": rf.Holding("B", None, 10, 10, 40.0, 20.0)}   # both +100%
    p = rf.Problem("T", 1, 100.0, 0.0, holdings, green_first=True)
    assert rf.green_component(p, 0.0, 0.0) == 1.0 and rf.green_component(p, 300.0, 150.0) == 1.0


def test_with_losses_allowed_the_least_profitable_first_scores_zero():
    holdings = {"WIN": rf.Holding("WIN", None, 10, 10, 20.0, 10.0), "LOSS": rf.Holding("LOSS", None, 10, 10, 8.0, 10.0)}
    p = rf.Problem("T", 1, 100.0, 0.0, holdings, green_first=True)
    assert rf.green_component(p, 80.0, -20.0) == 0.0 and rf.green_component(p, 100.0, 50.0) == 1.0


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
    assert result["plan"]["fill_shares"] == 1 and result["plan"]["left_over"] == 193.48 and result["weights"]["green"] == 0.4 and result["green_first"] is True
    assert result["after"]["shares"] == {"PG": 15, "MCD": 120, "GOOGL": 30, "LLY": 22, "SNDK": 3, "MU": 1}
    assert result["stats"]["levels"] == 3   # the same tree as before: it is the score that changed, and the answer with it
    text = am.report(result)
    assert "green 1.0" in text and "weights: green 0.4, overshoot 0.25, keep 0.2, orders 0.15" in text and "| BUY   | MU" in text


def test_the_prompt_lists_the_holdings_most_in_the_green_first_and_scores_green_first(start):
    manager = agent()
    manager.raise_funds(start, "SNDK", 3, fill="MU", explore=True, green_first=True)
    system, user = manager.scripted.seen[0], None
    p = green_problem(start, explore=True)
    text = prompts.user_message(p, rf.root(p), [])
    assert "Most in the green first (percent gain): LLY +809%, MCD +558%, GOOGL +180%, PG +170%" in text
    assert [m.group(1) for m in re.finditer(r"^- (\w+) \(", text, re.M)] == ["LLY", "MCD", "GOOGL", "PG"]   # the lines are in that order
    assert "percent gain +809%" in text and "taxable share" not in text
    system = prompts.system_prompt(3, green_first=True)
    assert "how green-first the plan is: sell the MOST PROFITABLE holdings first" in system and "taxable gain" not in system
    assert "taxable gain" in prompts.system_prompt(3) and "green-first" not in prompts.system_prompt(3)


# ---------------------------------------------------------------- how many shares: up to X, or as many as possible

SMALL = pf.Portfolio(user="x", cash_avail=0.0, positions=[pf.Position(symbol="GM", shares=10, cost_basis=10.0), pf.Position(symbol="F", shares=40, cost_basis=2.0)])
SMALL_PRICES = {"MU": 410.0, "GM": 55.0, "F": 12.0, "BIG": 5000.0}   # GM 10 x 55 = 550 and F 40 x 12 = 480: 1,030 in all, so 2 MU (820) and not 3 (1,230)


def small_agent(table):
    toolbox, model = QuoteToolbox(SMALL_PRICES), ScriptedByPlan(table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()))
    manager.toolbox, manager.scripted = toolbox, model
    return manager


TWO_STEPS = {ROOT: [("GM", 10)], "GM x10": [("F", 23)]}   # 550, then 270 more: 23 F (276) finishes it


def test_with_no_number_of_shares_the_portfolio_buys_as_many_as_it_can_pay_for_and_the_tree_decides_what_to_sell():
    manager = small_agent(TWO_STEPS)
    result = manager.raise_funds(SMALL, "MU", None)
    assert result["requested"] is None and result["most_affordable"] == 2 and result["resolved_shares"] == 2 and result["shares"] == 2
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("GM", 10), ("F", 23)]
    assert result["limits"]["max_depth"] == 2 and result["stats"]["llm_calls"] == 2   # a plan as deep as the holdings that can be sold; and it was the model that proposed the sells
    assert "QUANTITY: no number of shares was given, so as many as the portfolio can pay for: at most 2 x MU" in am.report(result)


def test_the_most_affordable_is_arithmetic_of_the_cash_and_everything_that_may_be_sold():
    p = rf.build_problem(SMALL, "MU", 1, SMALL_PRICES)
    assert p.max_affordable == 2 and rf.build_problem(SMALL, "MU", 1, SMALL_PRICES, only_green=True).max_affordable == 2
    assert rf.build_problem(SMALL.model_copy(update={"cash_avail": 300.0}), "MU", 1, SMALL_PRICES).max_affordable == 3   # (1,030 + 300) / 410 = 3.24


def test_up_to_x_buys_x_when_it_can_and_the_most_it_can_when_it_cannot():
    big = small_agent({ROOT: [("GM", 10)], "GM x10": [("F", 23)]})
    capped = big.raise_funds(SMALL, "MU", 5, up_to=True)   # 5 asked, 2 affordable
    assert capped["resolved_shares"] == 2 and capped["shares"] == 2 and capped["status"] == "plan" and "QUANTITY: up to 5 x MU; the portfolio can pay for at most 2, so 2 bought" in am.report(capped)
    one = small_agent({ROOT: [("GM", 8)]}).raise_funds(SMALL, "MU", 1, up_to=True)   # 1 asked and 1 affordable: 410 needs 8 GM (440)
    assert one["resolved_shares"] == 1 and one["status"] == "plan" and one["plan"]["proceeds"] == 440.0


def test_an_exact_number_that_cannot_be_paid_for_sells_nothing():
    result = small_agent({ROOT: [("GM", 10), ("F", 40)]}).raise_funds(SMALL, "MU", 5, up_to=False)
    assert result["status"] == "infeasible" and result["plan"] is None and result["after"] == {} and result["resolved_shares"] == 5 and result["most_affordable"] == 2


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


def test_several_stocks_with_a_number_each_use_up_to_and_green_first_in_order():
    manager = small_agent({ROOT: [("GM", 8)]})
    result = manager.raise_funds_many(SMALL, [("MU", 1), ("BIG", 2)], up_to=True, green_first=True)
    first, second = result["purchases"]
    assert first["status"] == "plan" and first["resolved_shares"] == 1 and first["green_first"] is True
    assert second["status"] == "infeasible" and second["resolved_shares"] == 0   # after the first, not even one BIG can be paid for


@pytest.mark.parametrize("shares", [0, -1, 1.5, True, "3"])
def test_a_number_of_shares_that_is_not_a_whole_number_of_at_least_one_is_still_refused(shares):
    result = small_agent({}).raise_funds(SMALL, "MU", shares)
    assert result["status"] == "refused" and "whole number of at least 1" in result["errors"][0]


def test_on_the_command_line_no_number_means_as_many_as_possible_and_a_number_after_a_colon_means_that_many():
    assert am.parse_purchase("MU", None) == ("MU", None) and am.parse_purchase("MU:3", None) == ("MU", 3) and am.parse_purchase("MU", 4) == ("MU", 4)


def test_the_portfolio_after_an_up_to_purchase_holds_the_shares_bought_not_the_shares_asked():
    """Found by the live complex run: 'up to 3 MU' bought 2, and the final portfolio of the run said 3. The cash was right, the shares were the number asked."""
    manager = small_agent({ROOT: [("GM", 10)], "GM x10": [("F", 23)]})
    result = manager.raise_funds_many(SMALL, [("MU", 5)], up_to=True)
    bought = result["purchases"][0]
    assert bought["shares"] == 2 and bought["resolved_shares"] == 2 and bought["requested"] == 5
    final = {p["symbol"]: p["shares"] for p in result["final"]["positions"]}
    assert final == {"F": 17, "MU": 2} and result["final"]["cash_avail"] == round(10 * 55.0 + 23 * 12.0 - 2 * 410.0, 2)   # GM sold out, 23 of 40 F sold, 2 MU bought
