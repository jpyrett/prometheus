"""Buy 3 SNDK, then as many MU as the cash left affords, by selling several cheaper stocks: a tree that goes more than one level deep (design choices entry 44).

Run:  pytest tests/test_sndk_mu.py -q

The portfolio is made up (configs/portfolios/jacob_pyrett_sndk_mu.json): PG 15 @ 60, MCD 120 @ 40, GOOGL 30 @ 100, LLY 25 @ 155 and $200 of free cash. The PRICES are the real Finnhub quotes of the 2026-11-03
close, recorded from the live run and used here as fixed numbers (no call is made). The "LLM" is scripted with the thoughts the real gpt-oss-120b gave in the third live run (in the order it gave them),
so the whole tree of that run can be replayed: 4 model calls, 3 levels, 7 completed plans, and 13 nodes as it ran (entry 45 then made the search drop a plan that is the same sells in another order:
PG x5, GOOGL x4 and LLY x1 reached twice, so 12 nodes and 6 completed plans now). The Data Clerk's quote tool, the checks, the scores, the pruning and the report are the real code.
No network, no key, no cost.

The numbers: 3 SNDK cost $3,379.56, $3,179.56 to raise after the free cash. Whole shares, lean sells (no step sells more than the shortfall needs).
"""
import json
import math
import re

import pytest
from langchain_core.messages import AIMessage

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf

SCENARIO = am.DEFAULT_PORTFOLIO.parent / "jacob_pyrett_sndk_mu.json"
PRICES = {"SNDK": 1126.52, "MU": 855.10, "PG": 162.06, "MCD": 263.04, "GOOGL": 280.03, "LLY": 1409.38}  # Finnhub /quote, the 2026-11-03 close (21:00 UTC)


class QuoteToolbox:
    def __init__(self, prices=None):
        self.prices, self.asked = dict(PRICES if prices is None else prices), []

    def get_quote(self, symbol):
        self.asked.append(symbol)
        if symbol not in self.prices:
            raise RuntimeError(f"No quote returned for {symbol}")
        p = self.prices[symbol]
        return {"symbol": symbol, "price": p, "change": 0.0, "change_percent": 0.0, "high": p, "low": p, "open": p, "previous_close": p,
                "as_of_utc": "2026-11-03T21:00:00Z", "source": "Finnhub /quote (recorded)"}


class ScriptedByPlan:
    """The 'LLM': answers from a table keyed by the plan so far, read from the prompt; a list of replies for one plan is used in order (a re-ask gets the next). Records every prompt."""
    model_name = "scripted/sndk-mu"

    def __init__(self, table):
        self.table, self.seen = {k: list(v) if isinstance(v, list) and v and isinstance(v[0], list) else [v] for k, v in table.items()}, []

    def invoke(self, messages):
        text = messages[-1].content
        self.seen.append(text)
        plan = re.search(r"Plan so far: (.*?)(?: \(raises| \(still short)", text).group(1)
        replies = self.table.get(plan, [[]])
        thoughts = replies.pop(0) if len(replies) > 1 else replies[0]
        return AIMessage(content=json.dumps({"thoughts": [{"symbol": s, "shares": n, "idea": f"idea: sell {n} {s}"} for s, n in thoughts]}),
                         usage_metadata={"input_tokens": 1000, "output_tokens": 900, "total_tokens": 1900})


ROOT = "(nothing sold)"
LIVE_RUN_3 = {  # what gpt-oss-120b answered, node by node, in the third live run (2026-11-03)
    ROOT: [("LLY", 3), ("GOOGL", 12), ("PG", 5)],
    "PG x5": [("MCD", 10), ("GOOGL", 4), ("LLY", 1)],
    "PG x5, GOOGL x4": [("LLY", 1), ("MCD", 5), ("MCD", 2)],
    "PG x5, LLY x1": [("MCD", 4), ("GOOGL", 4), ("MCD", 2)],
}


def agent(table=None, prices=None):
    toolbox = QuoteToolbox(prices)
    model = ScriptedByPlan(LIVE_RUN_3 if table is None else table)
    manager = am.AccountManager.classic(model=model, clerk=DataClerk(toolbox=toolbox, model=object()))
    manager.toolbox, manager.scripted = toolbox, model
    return manager


@pytest.fixture
def start():
    return pf.load_portfolio(SCENARIO)


def run(start, shares=3, table=None, **options):
    manager = agent(table)
    return manager, manager.raise_funds(start, "SNDK", shares, fill="MU", **{"explore": True, **options})


# ---------------------------------------------------------------- calculate first

def test_the_made_up_portfolio_is_the_one_in_the_request(start):
    assert [(p.symbol, p.shares, p.cost_basis) for p in start.positions] == [("PG", 15, 60.0), ("MCD", 120, 40.0), ("GOOGL", 30, 100.0), ("LLY", 25, 155.0)]
    assert start.cash_avail == 200.0


def test_the_numbers_calculated_before_any_search(start):
    p = rf.build_problem(start, "SNDK", 3, PRICES, fill="MU")
    assert p.cost == 3379.56 and p.need == 3179.56 and p.fill == "MU" and p.fill_price == 855.10 and p.lean
    assert set(p.holdings) == {"PG", "MCD", "GOOGL", "LLY"}                        # neither SNDK nor MU can be sold to pay for itself
    assert round(sum(h.value for h in p.holdings.values()) + p.cash, 2) == 77831.10  # everything, liquidated: far more than the 3 SNDK need
    assert p.max_affordable == 69
    assert p.holdings["PG"].value < p.need  # PG alone cannot cover: its 15 shares are worth $2,430.90


def best_by_enumeration(p):
    """The brute-force optimum: every lean plan of one or two sells (the first step any number of shares, the last the covering number), scored by the code's own score."""
    cover = lambda short, h: math.ceil(round(short / h.price, 6))
    plans = {}
    for a in p.holdings.values():
        for n1 in range(1, a.sellable + 1):
            short = round(p.need - n1 * a.price, 2)
            if short <= 0:
                if n1 == cover(p.need, a):
                    plans[((a.symbol, n1),)] = rf.score_plan(p, ((a.symbol, n1),))[0]
                continue
            for b in p.holdings.values():
                if b.symbol != a.symbol and cover(short, b) <= b.sellable:
                    plans[((a.symbol, n1), (b.symbol, cover(short, b)))] = rf.score_plan(p, ((a.symbol, n1), (b.symbol, cover(short, b))))[0]
    return sorted(plans.items(), key=lambda kv: -kv[1])


def test_the_best_plan_by_brute_force_is_one_sale_of_googl_and_the_best_of_several_stocks_loses_to_it(start):
    ranked = best_by_enumeration(rf.build_problem(start, "SNDK", 3, PRICES, fill="MU"))
    assert ranked[0] == ((("GOOGL", 12),), 72.9)
    best_multi = next(kv for kv in ranked if len(kv[0]) == 2)
    assert best_multi == ((("PG", 11), ("GOOGL", 5)), 67.1)  # several cheaper stocks: a lower score, because of the second order
    assert dict(ranked)[(("MCD", 13),)] == 64.3 and dict(ranked)[(("LLY", 3),)] == 63.0


# ---------------------------------------------------------------- the tree of the third live run, replayed

def test_the_tree_goes_three_levels_deep_and_finds_the_brute_force_optimum(start):
    _, result = run(start)
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("GOOGL", 12)] and result["plan"]["score"] == 72.9
    assert {k: result["stats"][k] for k in ("llm_calls", "nodes", "levels", "completed_plans")} == {"llm_calls": 4, "nodes": 12, "levels": 3, "completed_plans": 6}   # 13 and 7 before the search learned that the same sells in another order are one plan
    assert [len(level["expanded"]) for level in result["trace"]] == [1, 1, 2]
    assert "the depth limit (3) was reached" in result["stop"]


def test_the_plans_that_combine_several_stocks_were_built_scored_and_lost(start):
    _, result = run(start)
    plans = {c["plan"]: c for level in result["trace"] for c in level["children"] if "plan" in c}
    several = {p: c["score"] for p, c in plans.items() if p.count(",") >= 2 and c["status"] == "complete"}   # three stocks in one plan
    assert several == {"PG x5, LLY x1, GOOGL x4": 59.1, "PG x5, GOOGL x4, MCD x5": 60.6, "PG x5, LLY x1, MCD x4": 56.8}
    twice = [c for level in result["trace"] for c in level["children"] if c["status"] == "duplicate" and "different order" in c["reason"]]
    assert [c["reason"] for c in twice] == ["the same sells as n8 in a different order"] and twice[0]["thought"]["symbol"] == "LLY" and twice[0]["parent"] == "n5"   # PG x5, GOOGL x4 + LLY x1 is PG x5, LLY x1 + GOOGL x4
    assert plans["PG x5, MCD x10"]["status"] == "complete" and plans["PG x5, MCD x10"]["score"] == 58.7  # two stocks
    assert all(score < 72.9 for score in several.values())
    assert [c["status"] for c in result["trace"][0]["children"]] == ["complete", "complete", "kept"]   # PG x5 is the only unfinished plan: it is the branch that goes down


def test_a_step_that_sells_more_than_the_shortfall_needs_is_rejected_as_not_lean(start):
    p = rf.build_problem(start, "SNDK", 3, PRICES, fill="MU")
    node, reason = rf.check_step(p, rf.root(p), {"symbol": "GOOGL", "shares": 20, "idea": "x"}, "n1")
    assert node is None and "sells more than is needed: 12 share(s) already cover the shortfall of $3,179.56" in reason
    assert rf.check_step(p, rf.root(p), {"symbol": "GOOGL", "shares": 12}, "n1")[0].complete
    assert rf.check_step(p, rf.root(p), {"symbol": "GOOGL", "shares": 11}, "n1")[0].shortfall == 99.23   # fewer: a partial step, allowed


def test_without_a_fill_stock_the_sells_are_not_lean_as_in_the_earlier_entries(start):
    p = rf.build_problem(start, "SNDK", 3, PRICES)
    assert not p.lean and rf.check_step(p, rf.root(p), {"symbol": "GOOGL", "shares": 20}, "n1")[0] is not None


# ---------------------------------------------------------------- MU with the cash that is left

def test_the_cash_left_after_the_winning_plan_buys_no_mu_and_the_report_says_why(start):
    _, result = run(start)
    plan = result["plan"]
    assert plan["fill_shares"] == 0 and plan["left_over"] == 180.8 and result["after"]["cash"] == 180.8 and result["after"]["shares"]["GOOGL"] == 18
    text = am.report(result)
    assert "the cash left, $180.80, buys 0 at $855.10: less than one share" in text
    assert re.search(r"\| MU +\| +0 \| +0 \|", text) and re.search(r"\| cash +\| +\$200\.00 \| +\$180\.80 \|", text)
    assert "then the cash that is LEFT buys as many shares of MU" in text and "if selling everything cannot pay for 3 x SNDK, nothing is sold" in text


def test_a_plan_whose_leftover_buys_a_share_of_mu_buys_it_and_shows_the_order(start):
    _, result = run(start, table={ROOT: [("LLY", 3)]})   # 3 x 1,409.38 = 4,228.14: 1,048.58 left after SNDK, which buys 1 MU (855.10) and leaves 193.48
    plan = result["plan"]
    assert plan["fill_shares"] == 1 and plan["fill_cost"] == 855.10 and plan["left_over"] == 193.48
    assert result["after"]["shares"] == {"PG": 15, "MCD": 120, "GOOGL": 30, "LLY": 22, "SNDK": 3, "MU": 1} and result["after"]["cash"] == 193.48
    assert round(200.0 + plan["proceeds"] - 3379.56 - plan["fill_cost"], 2) == 193.48   # every dollar accounted for
    text = am.report(result)
    assert "| BUY   | MU" in text and "with the cash left" in text and "MU bought" not in text.split("OTHER COMPLETED")[0]


def test_the_score_counts_only_the_cash_that_buys_nothing_as_left_over(start):
    p = rf.build_problem(start, "SNDK", 3, PRICES, fill="MU")
    assert rf.score_plan(p, (("LLY", 3),))[1]["overshoot"] == pytest.approx(1 - 193.48 / 3379.56, abs=0.001)  # not 1 - 1,048.58 / 3,379.56: the MU it buys is not waste
    assert p.fill_with(1048.58) == (1, 193.48) and p.fill_with(180.8) == (0, 180.8) and p.fill_with(-5) == (0, 0.0)


# ---------------------------------------------------------------- "if liquidating cannot pay for it, sell nothing"

def test_if_selling_everything_cannot_pay_for_the_purchase_nothing_is_sold(start):
    manager, result = run(start, shares=100, table={ROOT: [("LLY", 25), ("MCD", 120), ("GOOGL", 30)]})   # 100 x 1,126.52 = 112,652 > 77,831.10
    assert result["status"] == "infeasible" and result["plan"] is None and result["after"] == {} and result["max_affordable"] == 69
    assert all(c["status"] == "rejected" for level in result["trace"] for c in level["children"])   # every step is a dead end: nothing can be completed
    text = am.report(result)
    assert "RESULT   INFEASIBLE" in text and "NOTHING IS SOLD" in text and "FINAL ANSWER" not in text


def test_a_purchase_that_is_infeasible_leaves_the_portfolio_file_untouched(start):
    before = start.model_dump_json()
    run(start, shares=100, table={ROOT: [("LLY", 25)]})
    assert start.model_dump_json() == before


# ---------------------------------------------------------------- the generator: both kinds of step, and asked again when one is missing

def test_the_prices_come_from_the_data_clerk_once_for_the_target_the_fill_stock_and_the_holdings(start):
    manager, result = run(start)
    assert manager.toolbox.asked == ["SNDK", "MU", "PG", "MCD", "GOOGL", "LLY"] and result["prices_order"]["action_type"] == "fetch_quote"


def test_a_generator_that_gives_only_partial_steps_is_asked_again_for_a_cover_step(start):
    # the first live run with explore: all three first steps were partial (12 MCD, 11 GOOGL, 2 LLY); one cover step was never offered
    table = {ROOT: [[("MCD", 12), ("GOOGL", 11), ("LLY", 2)], [("GOOGL", 12)]]}   # the second answer to the same plan has the cover step
    manager, result = run(start, table=table)
    first = result["trace"][0]
    assert first["reasked"] and "none of your steps finishes the plan" in first["reasked"][0]["why"] and "GOOGL (12 shares)" in first["reasked"][0]["why"]
    plans = [c["plan"] for c in first["children"] if "plan" in c]
    assert "GOOGL x12" in plans and len(plans) == 3 and result["stats"]["llm_calls"] >= 2   # the re-ask is a call, and it counts
    assert "asked again: none of your steps finishes the plan" in am.report(result)
    second_prompt = manager.scripted.seen[1]
    assert "The program's note on your last answer for this plan: none of your steps finishes the plan" in second_prompt


def test_a_generator_that_gives_only_cover_steps_is_asked_again_for_a_partial_one(start):
    table = {ROOT: [[("LLY", 3), ("GOOGL", 12), ("MCD", 13)], [("PG", 5)]]}
    _, result = run(start, table=table)
    first = result["trace"][0]
    assert first["reasked"] and "give a PARTIAL step as well" in first["reasked"][0]["why"]
    assert "PG x5" in [c["plan"] for c in first["children"] if "plan" in c] and len(result["trace"]) >= 2


def test_with_both_kinds_present_it_is_not_asked_again(start):
    _, result = run(start)
    assert all(not level["reasked"] for level in result["trace"]) and result["stats"]["llm_calls"] == 4


def test_without_explore_the_generator_is_asked_once_as_before(start):
    _, result = run(start, explore=False, table={ROOT: [("MCD", 12), ("GOOGL", 11), ("LLY", 2)]})
    assert all(not level["reasked"] for level in result["trace"])


def test_the_explore_prompt_has_the_two_kinds_the_taxable_share_and_the_partial_hint_and_the_plain_prompt_has_none(start):
    manager, _ = run(start)
    root_prompt = manager.scripted.seen[0]
    assert "- PG (Procter & Gamble Co): held 15, sellable 15, price $162.06, basis $60.00, gain $102.06 per share, cannot raise the shortfall alone: all 15 sellable shares are worth $2,430.90" in root_prompt
    assert "taxable share of what selling it raises: 64%" in root_prompt and "taxable share of what selling it raises: 89%" in root_prompt   # GOOGL, LLY
    assert "4 stocks can still be sold, so at least one step must be PARTIAL" in root_prompt
    assert "GOOGL (Alphabet Inc): held 30, sellable 30, price $280.03, basis $100.00, gain $180.03 per share, 12 share(s) alone would raise the shortfall" in root_prompt
    from prometheus.prompts import account_manager as prompts
    assert "COVER" in prompts.system_prompt(3, explore=True) and "COVER" not in prompts.system_prompt(3)
    plain = rf.build_problem(start, "SNDK", 3, PRICES)
    assert "taxable share" not in prompts.user_message(plain, rf.root(plain), []) and "PARTIAL step" in prompts.user_message(plain, rf.root(plain), [])  # the PG line still says it cannot cover alone
    lean = prompts.system_prompt(3, fill="MU", lean=True)
    assert "EXACTLY the number of shares shown" in lean and "buys as many shares of MU as it can" in lean and "do not sell MU" in lean


# ---------------------------------------------------------------- the generator logic, in isolation

def probe(start, plan_sells=()):
    p = rf.build_problem(start, "SNDK", 3, PRICES, fill="MU", explore=True)
    node = rf.root(p)
    for symbol, n in plan_sells:
        node, _ = rf.check_step(p, node, {"symbol": symbol, "shares": n}, "x")
    return p, node


def test_which_holdings_could_finish_the_plan_alone_is_a_fact_that_code_decides(start):
    p, root = probe(start)
    assert [h.symbol for h in rf.wanted_kinds(p, root, 3)[0]] == ["MCD", "GOOGL", "LLY"]   # PG is worth $2,430.90, under the shortfall
    assert rf.wanted_kinds(p, root, 3)[1] is True


def test_a_partial_step_is_not_wanted_when_the_plan_has_no_room_for_another_step(start):
    p, node = probe(start, [("PG", 5), ("GOOGL", 4)])   # two steps taken: the next is the last
    assert node.depth == 2 and rf.wanted_kinds(p, node, 3)[1] is False


def test_the_two_kinds_are_told_apart_by_the_hard_checks_not_by_the_models_label(start):
    p, root = probe(start)
    cover, partial = rf.kinds_of(p, root, [{"symbol": "GOOGL", "shares": 12}, {"symbol": "GOOGL", "shares": 11}, {"symbol": "GOOGL", "shares": 20}, {"symbol": "NVDA", "shares": 1}, "x"])
    assert cover == [{"symbol": "GOOGL", "shares": 12}] and partial == [{"symbol": "GOOGL", "shares": 11}]   # 20 is not lean, NVDA is not held, "x" is not a thought


def test_choose_keeps_one_of_each_wanted_kind_first_and_never_more_than_the_branching_factor(start):
    p, root = probe(start)
    first = [{"symbol": "LLY", "shares": 3}, {"symbol": "MCD", "shares": 13}, {"symbol": "GOOGL", "shares": 12}]
    again = [{"symbol": "PG", "shares": 5}, {"symbol": "LLY", "shares": 3}]
    chosen = rf.choose(p, root, [*first, *again], 3, 3)
    assert chosen == [{"symbol": "LLY", "shares": 3}, {"symbol": "PG", "shares": 5}, {"symbol": "MCD", "shares": 13}] and len(chosen) == 3   # the first COVER, the first PARTIAL, then the rest in order
