"""Tests for prometheus.reasoning.raise_funds: the Tree of Thought for what to sell (design choices entry 42).

Run:  pytest tests/test_raise_funds.py -q

The thought generator is a plain function that returns the thoughts a test wants, so each test decides exactly what the "LLM" proposes (a good step, an impossible one, a duplicate, a
score it made up) and checks what the CODE does with it. No network, no key, no cost.

The numbers used throughout (invented round prices, Jacob's real holdings): MSFT 50 @ 150 now 420, TSLA 123 @ 300 now 250, GM 60 @ 10 now 55; buy 2 MU at 410 = 820.
"""
from datetime import date

import pytest

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf

PRICES = {"MSFT": 420.0, "TSLA": 250.0, "GM": 55.0, "MU": 410.0}
TODAY = date(2026, 11, 3)


def no_cash():
    """The sample portfolio with its free cash taken away, so that the numbers below are the round ones. The sample file itself has cash_avail 51.23 (tests further down)."""
    return pf.load_portfolio(DEFAULT_PORTFOLIO).model_copy(update={"cash_avail": 0.0})


def problem(shares=2, prices=None, portfolio=None, target="MU"):
    return rf.build_problem(portfolio or no_cash(), target, shares, prices or PRICES, TODAY)


def thought(symbol, shares, idea="an idea"):
    return {"symbol": symbol, "shares": shares, "idea": idea}


class Gen:
    """A generator that answers from a table keyed by the plan so far (`node.plan()`), and records every call."""

    def __init__(self, table, default=()):
        self.table, self.default, self.calls = table, list(default), []

    def __call__(self, prob, node, rejected):
        self.calls.append((node.id, node.plan(), list(rejected)))
        return list(self.table.get(node.plan(), self.default))


ROOT = "(nothing sold)"


# ---------------------------------------------------------------- the problem


@pytest.mark.parametrize("shares", [0, -1, 1.5, True, "2"])
def test_a_number_of_shares_that_is_not_a_whole_number_of_at_least_one_is_refused(shares):
    with pytest.raises(ValueError, match="whole number"):
        problem(shares=shares)


def test_a_missing_or_zero_price_is_refused_and_named():
    with pytest.raises(ValueError, match="no price for MU, GM"):
        problem(prices={"MSFT": 420.0, "TSLA": 250.0, "GM": 0})


# ---------------------------------------------------------------- the score is calculated, from four components

def score(*sells, prob=None):
    return rf.score_plan(prob or problem(), tuple(sells))


def test_the_weights_add_up_to_one():
    assert sum(rf.WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- the hard checks

def check(t, parent=None, prob=None):
    prob = prob or problem()
    return rf.check_step(prob, parent or rf.root(prob), t, "n9")


def test_a_good_thought_becomes_a_node_with_every_figure_calculated_by_code():
    node, reason = check({"symbol": "tsla", "shares": "4", "idea": "x", "score": 999, "price": 1, "proceeds": 1e9})  # the model's own figures are ignored; "tsla" and "4" are cleaned
    assert reason is None and node.sells == (("TSLA", 4),) and node.proceeds == 1000.0 and node.shortfall == 0.0 and node.complete
    assert node.gain == -200.0 and node.score == rf.score_plan(problem(), node.sells)[0] != 999


@pytest.mark.parametrize("t, message", [
    ("sell TSLA", "not an object"),
    ({"symbol": "TSLA"}, "shares must be a whole number"),
    ({"symbol": "TSLA", "shares": 0}, "shares must be a whole number"),
    ({"symbol": "TSLA", "shares": -2}, "shares must be a whole number"),
    ({"symbol": "TSLA", "shares": 1.5}, "shares must be a whole number"),
    ({"symbol": "TSLA", "shares": True}, "shares must be a whole number"),
    ({"symbol": None, "shares": 2}, "shares must be a whole number"),
    ({"symbol": "MU", "shares": 2}, "cannot be sold to pay for itself"),
    ({"symbol": "NVDA", "shares": 2}, "NVDA is not in the portfolio"),
    ({"symbol": "TSLA", "shares": 124}, "only 123 are sellable"),
    ({"symbol": "GM", "shares": 61}, "only 60 are sellable"),
])
def test_a_thought_that_breaks_a_rule_is_rejected_with_the_reason(t, message):
    node, reason = check(t)
    assert node is None and message in reason


def test_a_symbol_may_appear_once_in_a_plan():
    prob = problem()
    parent, _ = rf.check_step(prob, rf.root(prob), thought("TSLA", 1), "n1")
    node, reason = rf.check_step(prob, parent, thought("TSLA", 1), "n2")
    assert node is None and "already in this plan" in reason


# ---------------------------------------------------------------- the search


def untimed(result):
    """The result without the seconds, which are the only thing that differs between two runs of the same search."""
    result["stats"].pop("seconds", None), result["stats"].pop("first_plan_seconds", None)
    for level in result["trace"]:
        level.pop("seconds", None)
    return result


def test_the_search_is_deterministic_for_the_same_thoughts():
    table = {ROOT: [thought("MSFT", 2), thought("TSLA", 2), thought("GM", 15)], "TSLA x2": [thought("GM", 6), thought("MSFT", 1)]}
    first, second = rf.search(problem(), Gen(table)), rf.search(problem(), Gen(table))
    assert rf.as_json(untimed(first)) == rf.as_json(untimed(second))


def test_the_call_budget_ends_the_search_even_with_unfinished_plans_left():
    gen = Gen({ROOT: [thought("GM", 1), thought("TSLA", 1), thought("MSFT", 1)]}, default=[thought("GM", 1)])
    result = rf.search(problem(), gen, max_calls=2)
    assert result["stats"]["llm_calls"] == 2 and len(gen.calls) == 2 and "the call budget (2) was used" in result["stop"]


def test_the_same_thought_twice_from_one_node_is_counted_once():
    result = rf.search(problem(), Gen({ROOT: [thought("TSLA", 4), thought("tsla", "4"), thought("GM", 15)]}))
    statuses = [c["status"] for c in result["trace"][0]["children"]]
    assert statuses == ["complete", "duplicate", "complete"] and result["stats"]["nodes"] == 3


def test_only_the_latest_rejections_are_told_to_the_generator():
    bad = [thought("NVDA", 1), thought("AAPL", 1), thought("QQQ", 1)]  # none is held: three rejections from each node
    gen = Gen({ROOT: [thought("GM", 1), thought("TSLA", 1), thought("MSFT", 1)],
               "GM x1": [*bad, thought("TSLA", 1)], "TSLA x1": [*bad, thought("GM", 1)], "MSFT x1": [*bad, thought("GM", 1)]})
    result = rf.search(problem(), gen, branching=4)
    every = [c["reason"] for level in result["trace"] for c in level["children"] if c["status"] == "rejected"]
    assert len(every) == 9
    third_level = [c for c in gen.calls if c[1].count("x") == 2]  # the calls on plans of two steps
    assert third_level and all(len(c[2]) == rf.MAX_REJECTIONS_TOLD == 6 and c[2] == every[-6:] for c in third_level)


def test_a_generator_that_fails_is_recorded_and_the_other_nodes_are_still_expanded():
    calls = []

    def flaky(prob, node, rejected):
        calls.append(node.id)
        if node.id == "n0":
            return [thought("TSLA", 1), thought("GM", 1)]
        if node.id == "n1":
            raise RuntimeError("secret detail that must not be printed")
        return [thought("MSFT", 2)]

    result = rf.search(problem(), flaky)
    assert calls == ["n0", "n1", "n2"] and result["status"] == "plan"
    assert result["errors"] == ["expanding n1 failed (RuntimeError)"] and "secret" not in rf.as_json(result)


def test_if_every_call_fails_the_answer_is_infeasible_with_the_errors():
    def broken(prob, node, rejected):
        raise TimeoutError("x")

    result = rf.search(problem(), broken)
    assert result["status"] == "infeasible" and result["plan"] is None and result["errors"] == ["expanding n0 failed (TimeoutError)"]


def test_when_the_cash_already_covers_the_purchase_nothing_is_sold_and_nothing_is_asked():
    portfolio = no_cash().model_copy(update={"cash_avail": 1000.0})
    gen = Gen({})
    result = rf.search(problem(portfolio=portfolio), gen)
    assert result["status"] == "funded" and result["plan"] is None and gen.calls == [] and result["stats"]["llm_calls"] == 0


# ---------------------------------------------------------------- the reasoning trace and the answer, as the user reads them

QUOTES = {s: {"price": p, "change_percent": 0.5, "as_of_utc": "2026-11-03T21:00:00Z", "source": "Finnhub /quote"} for s, p in PRICES.items()}


def test_the_answer_prints_the_goal_the_trace_the_orders_the_score_and_the_portfolio_before_and_after():
    gen = Gen({ROOT: [thought("MSFT", 2), thought("TSLA", 2, "TSLA is below its cost basis"), thought("GM", 15)], "TSLA x2": [thought("GM", 6), thought("MSFT", 1)]})
    text = rf.format_answer(rf.search(problem(), gen), quotes=QUOTES)
    for expected in ("GOAL     buy 2 x MU at $410.00 = $820.00", "QUOTES (from the Data Clerk's get_quote tool)", "Finnhub /quote", "REASONING TRACE", "LEVEL 1", "LEVEL 2",
                     "COMPLETE", "kept", "LLM idea: TSLA is below its cost basis", "FINAL ANSWER", "| SELL  | TSLA", "| SELL  | GM", "| BUY   | MU",
                     "score 84.0 / 100", "OTHER COMPLETED PLANS", "PORTFOLIO (shares)", "| cash   |", "SEARCH   2 LLM calls"):
        assert expected in text, expected


def test_the_answer_for_an_impossible_purchase_says_so_and_prints_no_orders():
    text = rf.format_answer(rf.search(problem(shares=500), Gen({ROOT: [thought("MSFT", 50)]})))
    assert "RESULT   INFEASIBLE" in text and "at most 134 share(s) of MU" in text and "FINAL ANSWER" not in text


def test_the_answer_when_the_cash_covers_the_purchase():
    portfolio = no_cash().model_copy(update={"cash_avail": 1000.0})
    text = rf.format_answer(rf.search(problem(portfolio=portfolio), Gen({})))
    assert "RESULT   funded" in text and "(no search: the free settled cash (cash_avail) already covers the purchase)" in text


def test_a_rejected_and_a_pruned_thought_are_shown_with_their_reasons():
    gen = Gen({ROOT: [thought("TSLA", 1), thought("GM", 2), thought("MSFT", 1), ], "TSLA x1": [thought("GM", 500)]})
    text = rf.format_trace(rf.search(problem(), gen, beam_width=1))
    assert "PRUNED" in text and "below the beam" in text and "REJECTED" in text and "only 60 are sellable" in text


# ---------------------------------------------------------------- "sell only what is in profit" (green), and never the stock being bought

def green_problem(prices=None, shares=2):
    return rf.build_problem(no_cash(), "MU", shares, prices or {**PRICES, "TSLA": 250.0}, TODAY, only_green=True)  # TSLA 250 is below its basis of 300: red


def test_when_the_free_cash_pays_for_the_purchase_the_holdings_after_are_the_same_and_the_cash_is_spent():
    portfolio = no_cash().model_copy(update={"cash_avail": 1000.0})
    result = rf.search(problem(portfolio=portfolio), Gen({}))
    assert result["status"] == "funded" and result["after"] == {"shares": {"MSFT": 50, "TSLA": 123, "GM": 60, "MU": 2}, "cash": 180.0}


# ---------------------------------------------------------------- the same sells in another order are one plan (entry 45)


# ---------------------------------------------------------------- the size of the tree, and what the beam ranks by (entry 45): pure functions on the complex portfolio, no model

COMPLEX = DEFAULT_PORTFOLIO.parent / "jacob_pyrett_complex.json"
COMPLEX_PRICES = {"LLY": 1409.38, "SNDK": 1126.52, "MU": 855.10, "F": 12.31, "GM": 88.34, "FITB": 53.93, "PG": 162.06, "MCD": 263.04, "GOOGL": 280.03, "TSLA": 398.50, "MSFT": 520.99}  # the 2026-11-03 close


def complex_problem(**options):
    return rf.build_problem(pf.load_portfolio(COMPLEX), "LLY", 3, COMPLEX_PRICES, only_green=True, lean=True, **options)


def test_with_explore_the_beam_keeps_the_plan_that_has_raised_more_even_if_its_score_is_lower():
    thoughts = [thought("F", 40), thought("MCD", 6)]   # F x40 raises $492.40 (and closes F); MCD x6 raises $1,578.24 (and closes MCD)
    plain = rf.search(complex_problem(), Gen({ROOT: thoughts}), beam_width=1)
    smart = rf.search(complex_problem(explore=True), Gen({ROOT: thoughts}), beam_width=1)
    kept = lambda result: [c["plan"] for c in result["trace"][0]["children"] if c["status"] == "kept"]
    pruned = lambda result: [c["reason"] for c in result["trace"][0]["children"] if c["status"] == "pruned"]
    assert len(kept(plain)) == 1 and len(kept(smart)) == 1 and kept(smart) == ["MCD x6"]       # the one that raised $1,578.24, not $492.40
    assert "priority" in pruned(smart)[0] and "of the money raised" in pruned(smart)[0] and "priority" not in pruned(plain)[0]


# ---------------------------------------------------------------- the plans of a level, expanded at the same time (entry 46)

import threading
import time


class SlowGen(Gen):
    """A table generator that takes a moment per call and records how many calls were running at once."""

    def __init__(self, table, delay=0.05):
        super().__init__(table)
        self.delay, self.running, self.most, self.lock = delay, 0, 0, threading.Lock()

    def __call__(self, prob, node, rejected):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(self.delay)
        try:
            return super().__call__(prob, node, rejected)
        finally:
            with self.lock:
                self.running -= 1


THREE_BRANCHES = {ROOT: [thought("GM", 1), thought("TSLA", 1), thought("MSFT", 1)], "GM x1": [thought("TSLA", 1)], "TSLA x1": [thought("GM", 2)], "MSFT x1": [thought("GM", 3)]}


def test_parallel_gives_exactly_the_same_result_and_trace_as_sequential():
    results = [rf.search(problem(), SlowGen(THREE_BRANCHES, 0.01), parallel=flag) for flag in (False, True)]
    for r in results:   # the seconds are the only thing that differs between two runs
        r["stats"].pop("seconds"), r["stats"].pop("first_plan_seconds")
        for level in r["trace"]:
            level.pop("seconds")
    assert rf.as_json(results[0]) == rf.as_json(results[1])
    assert [e["node"] for e in results[1]["trace"][1]["expanded"]] == [e["node"] for e in results[0]["trace"][1]["expanded"]] and len(results[1]["trace"][1]["expanded"]) == 3


def test_parallel_keeps_to_the_call_budget_that_the_plans_of_a_level_share():
    gen = SlowGen(THREE_BRANCHES)
    result = rf.search(problem(), gen, parallel=True, max_calls=3)   # the root takes one call: two of the three plans of level 2 can be expanded
    assert result["stats"]["llm_calls"] == 3 and len(gen.calls) == 3 and len(result["trace"][1]["expanded"]) == 2


def test_a_plan_that_fails_in_parallel_is_recorded_in_its_place_and_the_others_still_run():
    def flaky(prob, node, rejected):
        time.sleep(0.02)
        if node.id == "n2":
            raise TimeoutError("secret")
        return THREE_BRANCHES.get(node.plan(), [])
    result = rf.search(problem(), flaky, parallel=True)
    assert result["errors"] == ["expanding n2 failed (TimeoutError)"] and len(result["trace"][1]["expanded"]) == 2 and "secret" not in rf.as_json(result)


# ---------------------------------------------------------------- the time limit (entry 48): what is sold when it passes (the tests that only measured seconds were deleted, entry 86)

def test_once_the_time_limit_has_passed_no_new_level_starts_and_the_best_completed_plan_so_far_is_the_answer():
    gen = SlowGen({ROOT: [thought("TSLA", 4), thought("GM", 1)], "GM x1": [thought("MSFT", 2)]}, delay=0.1)
    result = rf.search(problem(), gen, deadline=0.05)   # level 1 takes 0.1 s: the limit has passed when it ends, with TSLA x4 completed and GM x1 still open
    assert [c[1] for c in gen.calls] == [ROOT] and result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("TSLA", 4)]
    assert "the time limit (0.05 s) was reached; the best completed plan is the answer" in result["stop"] and result["limits"]["deadline"] == 0.05


def test_with_nothing_completed_when_the_time_is_up_nothing_is_sold():
    result = rf.search(problem(), SlowGen({ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 4)]}, delay=0.1), deadline=0.05)
    assert result["status"] == "infeasible" and result["plan"] is None and result["after"] == {} and "the time limit (0.05 s) was reached" in result["stop"]


def test_in_parallel_a_level_already_out_is_waited_for_after_the_limit():
    gen = SlowGen(THREE_BRANCHES, delay=0.15)
    result = rf.search(problem(), gen, deadline=0.2, parallel=True)   # level 1 ends at 0.15 s (in time); level 2's three calls start together and finish at 0.30 s: after the limit, waited for
    assert len(result["trace"]) == 2 and gen.most == 3 and "the time limit" in result["stop"] and result["stats"]["seconds"] >= 0.29


# ---------------------------------------------------------------- the tree raises its own depth (entry 49)

def strict_complex(target="LLY", shares=3, **options):
    return rf.build_problem(pf.load_portfolio(COMPLEX), target, shares, COMPLEX_PRICES, only_green=True, lean=True, never_liquidate=True, **options)


# ---------------------------------------------------------------- whole plans as thoughts (entry 50): tried, measured slower, kept as an option

def plan(*sells, idea="idea"):
    return {"sells": [{"symbol": s, "shares": n} for s, n in sells], "idea": idea}


def whole_problem(**options):
    return rf.build_problem(no_cash(), "MU", 2, PRICES, TODAY, whole_plans=True, lean=True, **options)   # 820 to raise; TSLA 250, GM 55, MSFT 420


def test_a_whole_plan_is_one_node_one_level_below_its_parent_with_every_sell_checked_in_order():
    p = whole_problem()
    node, reason = rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("TSLA", 2)), "n1")   # 420, then 2 x 250 = 920 >= 820
    assert reason is None and node.sells == (("MSFT", 1), ("TSLA", 2)) and node.depth == 1 and node.proceeds == 920.0 and node.complete and node.idea == "idea"
    assert node.score == rf.score_plan(p, node.sells)[0]


def test_a_whole_plan_that_breaks_a_rule_is_rejected_with_the_sell_that_breaks_it():
    p = whole_problem(only_green=True)
    assert "sell 2 of the plan (TSLA x2): TSLA is not in profit" in rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("TSLA", 2)), "n1")[1]
    assert "sell 1 of the plan (MU x1): MU is the stock being bought" in rf.check_step(p, rf.root(p), plan(("MU", 1)), "n1")[1]
    assert "already in this plan" in rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("MSFT", 1)), "n1")[1]


def test_a_whole_plan_may_not_keep_selling_after_it_has_raised_the_money():
    p = whole_problem()
    reason = rf.check_step(p, rf.root(p), plan(("TSLA", 4), ("GM", 1)), "n1")[1]   # TSLA x4 = 1,000 already raises 820
    assert "sell 2 (GM) comes after the plan has already raised the money" in reason


def test_the_last_sell_of_a_lean_whole_plan_may_not_sell_a_share_too_many():
    p = whole_problem()
    assert "sells more than is needed: 2 share(s)" in rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("TSLA", 3)), "n1")[1]   # after MSFT 420, 400 more: 2 TSLA are enough, 3 are not


@pytest.mark.parametrize("bad", [{"sells": []}, {"sells": "MSFT"}, {"sells": ["MSFT"]}, {"sells": [1]}])
def test_a_whole_plan_that_is_not_a_list_of_sells_is_rejected(bad):
    p = whole_problem()
    node, reason = rf.check_step(p, rf.root(p), bad, "n1")
    assert node is None and "not a non-empty list of sells" in reason


def test_with_whole_plans_no_partial_step_is_wanted_and_the_search_asks_again_from_the_start_when_every_plan_was_rejected():
    p = whole_problem()
    assert rf.wanted_kinds(p, rf.root(p), 3)[1] is False
    gen = Gen({ROOT: [plan(("GM", 99)), plan(("NVDA", 1))]})
    bad = rf.search(p, gen)                     # both rejected, every time: asked again from the start until the depth is used
    assert bad["status"] == "infeasible" and [c[1] for c in gen.calls] == [ROOT, ROOT, ROOT] and [lv["retry"] for lv in bad["trace"]] == [True, True, False]
    assert gen.calls[1][2] and "only 60 are sellable" in gen.calls[1][2][0] or "only 10 are sellable" in gen.calls[1][2][0]   # the second call is told why


def test_a_retry_that_finds_a_good_plan_ends_the_search():
    calls = []

    def generate(prob, node, rejected):
        calls.append(list(rejected))
        return [plan(("GM", 99))] if len(calls) == 1 else [plan(("MSFT", 1), ("TSLA", 2))]
    result = rf.search(whole_problem(), generate)
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("MSFT", 1), ("TSLA", 2)] and result["stats"]["levels"] == 2 and calls[1] != []


def test_a_search_with_whole_plans_finishes_in_one_round_when_the_first_plans_are_good():
    gen = Gen({ROOT: [plan(("MSFT", 1), ("TSLA", 2)), plan(("TSLA", 4)), plan(("GM", 10), ("MSFT", 1))]})
    result = rf.search(whole_problem(), gen)
    assert result["stats"]["levels"] == 1 and len(gen.calls) == 1 and result["stats"]["completed_plans"] >= 2


def test_the_trace_shows_a_whole_plan_as_its_list_of_sells():
    text = rf.format_trace(rf.search(whole_problem(), Gen({ROOT: [plan(("MSFT", 1), ("TSLA", 2))]})))
    assert "plan: MSFT x1, TSLA x2" in text
