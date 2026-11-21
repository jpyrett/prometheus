"""Tests for prometheus.reasoning.raise_funds: the Tree of Thought for what to sell (design choices entry 42).

Run:  pytest tests/test_raise_funds.py -q

The thought generator is a plain function that returns the thoughts a test wants, so each test decides exactly what the "LLM" proposes (a good step, an impossible one, a duplicate, a
score it made up) and checks what the CODE does with it. No network, no key, no cost.

The numbers used throughout (invented round prices, Jacob's real holdings): MSFT 50 @ 150 now 420, TSLA 123 @ 300 now 250, GM 60 @ 10 now 55; buy 2 MU at 410 = 820.
"""
from datetime import date
from decimal import Decimal

import pytest

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf

PRICES = {"MSFT": Decimal("420.00"), "TSLA": Decimal("250.00"), "GM": Decimal("55.00"), "MU": Decimal("410.00")}
TODAY = date(2026, 11, 3)


def no_cash():
    """The sample portfolio with its free cash taken away, so that the numbers below are the round ones. The sample file itself has cash_avail 51.23 (tests further down)."""
    return pf.load_portfolio(DEFAULT_PORTFOLIO).model_copy(update={"cash_avail": Decimal("0.00")})


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


@pytest.mark.parametrize("shares", [0, -1, Decimal("1.5"), True, "2"])
def test_a_number_of_shares_that_is_not_a_whole_number_of_at_least_one_is_refused(shares):
    with pytest.raises(ValueError, match="whole number"):
        problem(shares=shares)


def test_a_missing_or_zero_price_is_refused_and_named():
    with pytest.raises(ValueError, match="no price for MU, GM"):
        problem(prices={"MSFT": Decimal("420.00"), "TSLA": Decimal("250.00"), "GM": Decimal("0")})


# ---------------------------------------------------------------- the score is calculated, from four components

def score(*sells, prob=None):
    return rf.score_plan(prob or problem(), tuple(sells))


def test_the_weights_add_up_to_one():
    assert sum(rf.WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- the hard checks

def check(t, parent=None, prob=None):
    prob = prob or problem()
    return rf.check_step(prob, parent or rf.root(prob), t, "n9")


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


# ---------------------------------------------------------------- the search


def test_the_same_thought_twice_from_one_node_is_counted_once():
    result = rf.search(problem(), Gen({ROOT: [thought("TSLA", 4), thought("tsla", 4), thought("GM", 15)]}))      # AMENDED (entry 111, M5): the second one said "4" (a string); the same thought now differs only in the case of the symbol
    statuses = [c["status"] for c in result["trace"][0]["children"]]
    assert statuses == ["complete", "duplicate", "complete"] and result["stats"]["nodes"] == 3


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


# ---------------------------------------------------------------- the reasoning trace and the answer, as the user reads them


def test_the_answer_when_the_cash_covers_the_purchase():
    portfolio = no_cash().model_copy(update={"cash_avail": Decimal("1000.00")})
    text = rf.format_answer(rf.search(problem(portfolio=portfolio), Gen({})))
    assert "RESULT   funded" in text and "(no search: the free settled cash (cash_avail) already covers the purchase)" in text


# ---------------------------------------------------------------- the size of the tree, and what the beam ranks by (entry 45): pure functions on the complex portfolio, no model

COMPLEX = DEFAULT_PORTFOLIO.parent / "jacob_pyrett_complex.json"
COMPLEX_PRICES = {s: Decimal(p) for s, p in {"LLY": "1409.38", "SNDK": "1126.52", "MU": "855.10", "F": "12.31", "GM": "88.34", "FITB": "53.93", "PG": "162.06", "MCD": "263.04", "GOOGL": "280.03", "TSLA": "398.50", "MSFT": "520.99"}.items()}  # the 2026-11-03 close


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


def test_parallel_keeps_to_the_call_budget_that_the_plans_of_a_level_share():
    gen = SlowGen(THREE_BRANCHES)
    result = rf.search(problem(), gen, parallel=True, max_calls=3)   # the root takes one call: two of the three plans of level 2 can be expanded
    assert result["stats"]["llm_calls"] == 3 and len(gen.calls) == 3 and len(result["trace"][1]["expanded"]) == 2


# ---------------------------------------------------------------- the time limit (entry 48): what is sold when it passes (the tests that only measured seconds were deleted, entry 86)

def test_once_the_time_limit_has_passed_no_new_level_starts_and_the_best_completed_plan_so_far_is_the_answer():
    gen = SlowGen({ROOT: [thought("TSLA", 4), thought("GM", 1)], "GM x1": [thought("MSFT", 2)]}, delay=0.1)
    result = rf.search(problem(), gen, deadline=0.05)   # level 1 takes 0.1 s: the limit has passed when it ends, with TSLA x4 completed and GM x1 still open
    assert [c[1] for c in gen.calls] == [ROOT] and result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("TSLA", 4)]
    assert "the time limit (0.05 s) was reached; the best completed plan is the answer" in result["stop"] and result["limits"]["deadline"] == 0.05


# ---------------------------------------------------------------- whole plans as thoughts (entry 50): tried, measured slower, kept as an option

def plan(*sells, idea="idea"):
    return {"sells": [{"symbol": s, "shares": n} for s, n in sells], "idea": idea}


def whole_problem(**options):
    return rf.build_problem(no_cash(), "MU", 2, PRICES, TODAY, whole_plans=True, lean=True, **options)   # 820 to raise; TSLA 250, GM 55, MSFT 420


def test_a_whole_plan_that_breaks_a_rule_is_rejected_with_the_sell_that_breaks_it():
    p = whole_problem(only_green=True)
    assert "sell 2 of the plan (TSLA x2): TSLA is not in profit" in rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("TSLA", 2)), "n1")[1]
    assert "sell 1 of the plan (MU x1): MU is the stock being bought" in rf.check_step(p, rf.root(p), plan(("MU", 1)), "n1")[1]
    assert "already in this plan" in rf.check_step(p, rf.root(p), plan(("MSFT", 1), ("MSFT", 1)), "n1")[1]


def test_a_whole_plan_may_not_keep_selling_after_it_has_raised_the_money():
    p = whole_problem()
    reason = rf.check_step(p, rf.root(p), plan(("TSLA", 4), ("GM", 1)), "n1")[1]   # TSLA x4 = 1,000 already raises 820
    assert "sell 2 (GM) comes after the plan has already raised the money" in reason


@pytest.mark.parametrize("bad", [{"sells": []}, {"sells": "MSFT"}, {"sells": ["MSFT"]}, {"sells": [1]}])
def test_a_whole_plan_that_is_not_a_list_of_sells_is_rejected(bad):
    p = whole_problem()
    node, reason = rf.check_step(p, rf.root(p), bad, "n1")
    assert node is None and "not a non-empty list of sells" in reason


def test_a_retry_that_finds_a_good_plan_ends_the_search():
    calls = []

    def generate(prob, node, rejected):
        calls.append(list(rejected))
        return [plan(("GM", 99))] if len(calls) == 1 else [plan(("MSFT", 1), ("TSLA", 2))]
    result = rf.search(whole_problem(), generate)
    assert result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("MSFT", 1), ("TSLA", 2)] and result["stats"]["levels"] == 2 and calls[1] != []
