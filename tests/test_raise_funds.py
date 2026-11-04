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

def test_the_problem_is_the_cost_of_the_purchase_and_what_can_be_sold():
    p = problem()
    assert p.cost == 820.0 and p.need == 820.0 and p.price == 410.0 and p.cash == 0.0
    assert set(p.holdings) == {"MSFT", "TSLA", "GM"}  # the target is never a holding that can be sold
    assert p.holdings["TSLA"].sellable == 123 and p.holdings["TSLA"].basis == 300.0 and p.holdings["TSLA"].price == 250.0


def test_the_most_that_could_be_bought_is_calculated_from_everything_sellable():
    assert problem().max_affordable == 134  # (50*420 + 123*250 + 60*55) / 410 = 134.27


def test_the_free_settled_cash_of_the_sample_portfolio_reduces_what_has_to_be_raised():
    p = problem(portfolio=pf.load_portfolio(DEFAULT_PORTFOLIO))  # cash_avail 51.23, as the file says
    assert p.cash == 51.23 and p.cost == 820.0 and p.need == 768.77
    assert p.max_affordable == 134  # (55,050 + 51.23) / 410 = 134.39: the cash counts in the most that could be bought


def test_the_cash_is_spent_first_so_a_plan_that_could_not_pay_alone_is_complete_with_it():
    sample = pf.load_portfolio(DEFAULT_PORTFOLIO)
    asked = rf.search(problem(portfolio=sample), Gen({ROOT: [thought("GM", 14)]}))  # 14 x 55 = 770: short of 820 alone, enough with 51.23 of cash
    assert asked["status"] == "plan" and asked["plan"]["proceeds"] == 770.0 and asked["plan"]["left_over"] == 1.23 and asked["after"]["cash"] == 1.23
    assert asked["before"]["cash"] == 51.23 and asked["cash"] == 51.23 and asked["need"] == 768.77
    without = rf.search(problem(), Gen({ROOT: [thought("GM", 14)]}))
    assert without["status"] == "infeasible"  # the same step, with no cash, does not reach 820


def test_the_overshoot_component_counts_the_cash_too():
    sample = problem(portfolio=pf.load_portfolio(DEFAULT_PORTFOLIO))
    assert rf.score_plan(sample, (("GM", 14),))[1]["overshoot"] == pytest.approx(1 - 1.23 / 820, abs=0.001)  # 770 + 51.23 - 820 left over
    assert "overshoot" not in rf.score_plan(sample, (("GM", 13),))[1]  # 715 + 51.23 is still short of 820


def test_cash_already_in_the_account_reduces_what_has_to_be_raised():
    portfolio = no_cash().model_copy(update={"cash_avail": 300.0})
    p = problem(portfolio=portfolio)
    assert p.need == 520.0 and p.max_affordable == 135  # (55,050 + 300) / 410


def test_shares_bought_today_are_not_counted_as_sellable():
    portfolio = pf.Portfolio(user="x", positions=[pf.Position(symbol="GM", shares=60, cost_basis=10.0, acquired=TODAY), pf.Position(symbol="TSLA", shares=10, cost_basis=300.0)])
    p = problem(portfolio=portfolio)
    assert p.holdings["GM"].sellable == 0 and p.holdings["TSLA"].sellable == 10 and p.max_affordable == 6  # only TSLA: 10 * 250 / 410


def test_a_target_already_held_is_kept_in_the_shares_after_the_purchase():
    portfolio = pf.Portfolio(user="x", positions=[pf.Position(symbol="MU", shares=3, cost_basis=100.0), pf.Position(symbol="GM", shares=60, cost_basis=10.0)])
    p = problem(portfolio=portfolio)
    assert "MU" not in p.holdings and p.held == 3
    result = rf.search(p, Gen({ROOT: [thought("GM", 15)]}))
    assert result["before"]["shares"] == {"GM": 60, "MU": 3} and result["after"]["shares"] == {"GM": 45, "MU": 5}


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


def test_the_score_of_a_plan_is_calculated_from_its_components_by_hand():
    s, parts = score(("TSLA", 2), ("GM", 6))  # raises 830, gain 2*(250-300) + 6*(55-10) = 170
    assert parts == {"gain": 0.795, "keep": 1.0, "orders": 0.5, "overshoot": 0.988}
    assert s == 84.0  # 100 * (0.40*0.795 + 0.25*0.988 + 0.20*1.0 + 0.15*0.5)


def test_a_plan_that_sells_at_a_loss_has_the_best_possible_gain_component():
    assert score(("TSLA", 4))[1]["gain"] == 1.0  # 4 * (250 - 300) is a loss


def test_selling_every_share_of_a_position_costs_the_keep_component():
    assert score(("GM", 60))[1]["keep"] == pytest.approx(0.667, abs=0.001)  # 1 of 3 positions closed
    assert score(("GM", 59))[1]["keep"] == 1.0


def test_more_orders_score_lower_on_the_orders_component():
    assert [score(*sells)[1]["orders"] for sells in ((("GM", 1),), (("GM", 1), ("TSLA", 1)), (("GM", 1), ("TSLA", 1), ("MSFT", 1)))] == [1.0, 0.5, 0.333]


def test_overshoot_is_only_known_for_a_plan_that_covers_the_cost_and_the_weights_are_renormalised_without_it():
    assert "overshoot" not in score(("TSLA", 2))[1]                       # raises 500 of 820
    s, parts = score(("TSLA", 2))
    assert s == round(100 * (0.40 * parts["gain"] + 0.20 * parts["keep"] + 0.15 * parts["orders"]) / 0.75, 1)
    assert score(("TSLA", 4))[1]["overshoot"] == pytest.approx(1 - 180 / 820, abs=0.001)


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


def test_shares_bought_today_cannot_be_sold_even_though_they_are_held():
    portfolio = pf.Portfolio(user="x", positions=[pf.Position(symbol="GM", shares=60, cost_basis=10.0, acquired=TODAY), pf.Position(symbol="TSLA", shares=10, cost_basis=300.0)])
    node, reason = check(thought("GM", 1), prob=problem(portfolio=portfolio))
    assert node is None and "only 0 are sellable (60 held)" in reason


def test_a_branch_that_can_no_longer_reach_the_cost_is_a_dead_end():
    portfolio = pf.Portfolio(user="x", positions=[pf.Position(symbol="GM", shares=10, cost_basis=10.0), pf.Position(symbol="TSLA", shares=1, cost_basis=300.0)])
    prob = problem(portfolio=portfolio)  # GM 10 * 55 = 550, TSLA 1 * 250: together 800 < 820, the target can never be bought
    node, reason = rf.check_step(prob, rf.root(prob), thought("GM", 1), "n1")
    assert node is None and reason.startswith("dead end")


def test_a_thought_that_does_not_cover_the_cost_is_an_open_node_with_the_shortfall():
    node, _ = check(thought("TSLA", 2))
    assert not node.complete and node.proceeds == 500.0 and node.shortfall == 320.0


# ---------------------------------------------------------------- the search

def test_the_best_completed_plan_by_calculated_score_is_the_answer():
    gen = Gen({ROOT: [thought("MSFT", 2), thought("TSLA", 2), thought("GM", 15)], "TSLA x2": [thought("GM", 6), thought("MSFT", 1)]})
    result = rf.search(problem(), gen)
    assert result["status"] == "plan"
    assert [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("TSLA", 2), ("GM", 6)] and result["plan"]["score"] == 84.0
    assert result["plan"]["proceeds"] == 830.0 and result["plan"]["left_over"] == 10.0 and result["plan"]["gain"] == 170.0
    assert [(a["node"], a["score"]) for a in result["alternatives"]] == [("n5", 82.1), ("n1", 73.7), ("n3", 67.1)]  # the others, best first
    assert {k: result["stats"][k] for k in ("llm_calls", "nodes", "levels", "completed_plans")} == {"llm_calls": 2, "nodes": 6, "levels": 2, "completed_plans": 4}
    assert result["before"] == {"shares": {"MSFT": 50, "TSLA": 123, "GM": 60}, "cash": 0.0} and result["after"] == {"shares": {"MSFT": 50, "TSLA": 121, "GM": 54, "MU": 2}, "cash": 10.0}


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


def test_a_completed_plan_is_never_expanded():
    gen = Gen({ROOT: [thought("TSLA", 4)]})
    rf.search(problem(), gen)
    assert [c[1] for c in gen.calls] == [ROOT]  # n1 (TSLA x4) is complete: the generator is not asked to extend it


def test_the_beam_keeps_only_the_best_unfinished_plans_and_the_trace_says_why_the_others_were_pruned():
    gen = Gen({ROOT: [thought("TSLA", 1), thought("GM", 2), thought("MSFT", 1)], "TSLA x1": [thought("GM", 12)]})
    result = rf.search(problem(), gen, beam_width=1)
    level1 = result["trace"][0]["children"]
    assert [c["status"] for c in level1] == ["kept", "pruned", "pruned"] and level1[0]["plan"] == "TSLA x1"
    assert "below the beam" in level1[1]["reason"] and "top 1" in level1[1]["reason"]
    assert [c[1] for c in gen.calls] == [ROOT, "TSLA x1"]  # only the kept node was expanded; the pruned ones were not
    assert result["status"] == "plan"


def test_a_pruned_plan_can_never_come_back():
    gen = Gen({ROOT: [thought("TSLA", 1), thought("GM", 2)], "TSLA x1": [thought("GM", 12)], "GM x2": [thought("TSLA", 3)]})
    result = rf.search(problem(), gen, beam_width=1)
    assert [c[1] for c in gen.calls] == [ROOT, "TSLA x1"]  # GM x2 was pruned at level 1: it is never expanded, so its good continuation is never found
    assert [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("TSLA", 1), ("GM", 12)]


def test_the_depth_limit_ends_the_search_and_says_so():
    gen = Gen({ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 1)], "GM x1, TSLA x1": [thought("MSFT", 1)]})
    result = rf.search(problem(), gen, max_depth=2)
    assert result["status"] == "infeasible" and len(result["trace"]) == 2 and "the depth limit (2) was reached" in result["stop"]


def test_the_search_never_goes_deeper_than_the_number_of_holdings():
    portfolio = pf.Portfolio(user="x", positions=[pf.Position(symbol="GM", shares=60, cost_basis=10.0), pf.Position(symbol="TSLA", shares=123, cost_basis=300.0)])
    gen = Gen({ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 1)], "GM x1, TSLA x1": [thought("GM", 2)]})
    result = rf.search(problem(portfolio=portfolio), gen)
    assert len(result["trace"]) == 2  # two holdings, two levels: a third step could only repeat a symbol


def test_the_call_budget_ends_the_search_even_with_unfinished_plans_left():
    gen = Gen({ROOT: [thought("GM", 1), thought("TSLA", 1), thought("MSFT", 1)]}, default=[thought("GM", 1)])
    result = rf.search(problem(), gen, max_calls=2)
    assert result["stats"]["llm_calls"] == 2 and len(gen.calls) == 2 and "the call budget (2) was used" in result["stop"]


def test_at_most_the_branching_factor_of_thoughts_is_considered_per_node():
    gen = Gen({ROOT: [thought("GM", i) for i in range(1, 8)]})
    result = rf.search(problem(), gen)
    assert len(result["trace"][0]["children"]) == rf.BRANCHING == 3


def test_the_same_thought_twice_from_one_node_is_counted_once():
    result = rf.search(problem(), Gen({ROOT: [thought("TSLA", 4), thought("tsla", "4"), thought("GM", 15)]}))
    statuses = [c["status"] for c in result["trace"][0]["children"]]
    assert statuses == ["complete", "duplicate", "complete"] and result["stats"]["nodes"] == 3


def test_a_rejected_thought_is_kept_in_the_trace_and_its_reason_is_told_to_the_next_call():
    gen = Gen({ROOT: [thought("TSLA", 500), thought("GM", 2)], "GM x2": [thought("TSLA", 3)]})
    result = rf.search(problem(), gen)
    rejected = result["trace"][0]["children"][0]
    assert rejected["status"] == "rejected" and "only 123 are sellable" in rejected["reason"]
    assert gen.calls[1][2] == ["TSLA: 500 shares asked but only 123 are sellable (123 held)"]  # the second call heard about the first thought's failure


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


def test_an_impossible_purchase_is_infeasible_and_says_how_many_could_be_bought():
    gen = Gen({ROOT: [thought("MSFT", 50), thought("TSLA", 123), thought("GM", 60)]})
    result = rf.search(problem(shares=500), gen)
    assert result["status"] == "infeasible" and result["plan"] is None and result["max_affordable"] == 134 and result["after"] == {}


def test_when_the_cash_already_covers_the_purchase_nothing_is_sold_and_nothing_is_asked():
    portfolio = no_cash().model_copy(update={"cash_avail": 1000.0})
    gen = Gen({})
    result = rf.search(problem(portfolio=portfolio), gen)
    assert result["status"] == "funded" and result["plan"] is None and gen.calls == [] and result["stats"]["llm_calls"] == 0


def test_a_tie_in_score_goes_to_the_earlier_node():
    prices = {**PRICES, "TSLA": 300.0, "MSFT": 150.0, "GM": 10.0}  # at cost: no gain or loss anywhere
    result = rf.search(problem(shares=1, prices={**prices, "MU": 300.0}), Gen({ROOT: [thought("TSLA", 1), thought("MSFT", 2)]}))
    assert result["plan"]["node"] == "n1" and result["alternatives"][0]["node"] == "n2"


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


def test_a_holding_at_a_loss_is_not_green_and_may_not_be_sold_when_only_green_is_on():
    prob = green_problem()
    assert [s for s, h in prob.holdings.items() if h.green] == ["MSFT", "GM"] and not prob.holdings["TSLA"].green
    node, reason = rf.check_step(prob, rf.root(prob), thought("TSLA", 1), "n1")
    assert node is None and "TSLA is not in profit (price $250.00 is not above its basis $300.00)" in reason
    assert rf.check_step(prob, rf.root(prob), thought("GM", 1), "n1")[0] is not None


def test_a_holding_priced_exactly_at_its_basis_is_not_green():
    assert not rf.Holding("X", None, 1, 1, 10.0, 10.0).green


def test_without_the_rule_the_same_holding_may_be_sold():
    prob = rf.build_problem(no_cash(), "MU", 2, PRICES, TODAY)  # only_green defaults to off
    assert rf.check_step(prob, rf.root(prob), thought("TSLA", 4), "n1")[0] is not None


def test_the_most_that_could_be_bought_counts_only_what_may_be_sold():
    assert green_problem().max_affordable == 59   # (50 x 420 + 60 x 55) / 410 = 59.3: the 30,750 of TSLA, at a loss, is not counted
    assert problem().max_affordable == 134        # with the rule off it is


def test_a_branch_is_a_dead_end_when_only_the_green_holdings_left_cannot_raise_the_rest():
    prob = green_problem(shares=60)  # 60 x 410 = 24,600; MSFT + GM = 24,300 of green holdings: short by 300 whatever is sold
    node, reason = rf.check_step(prob, rf.root(prob), thought("GM", 1), "n1")
    assert node is None and reason.startswith("dead end")


def test_the_search_goes_no_deeper_than_the_holdings_that_may_be_sold():
    gen = Gen({ROOT: [thought("GM", 1)], "GM x1": [thought("MSFT", 1)], "GM x1, MSFT x1": [thought("TSLA", 1)]})
    result = rf.search(green_problem(), gen)  # two green holdings: two levels
    assert len(result["trace"]) == 2


def test_the_goal_line_and_the_prompt_say_that_only_profitable_holdings_may_be_sold():
    from prometheus.prompts import account_manager as prompts
    prob = green_problem()
    text = rf.format_answer(rf.search(prob, Gen({ROOT: [thought("TSLA", 4), thought("GM", 15)]})))
    assert "only holdings in profit (green) may be sold, and never MU itself" in text and "REJECTED" in text and "is not in profit" in text
    user = prompts.user_message(prob, rf.root(prob), [])
    assert "Only holdings that show a profit may be sold." in user and "- TSLA (Tesla Inc): held 123" in user and "(at a loss: may not be sold)" in user
    assert "Only holdings that show a profit" in prompts.system_prompt(3, only_green=True) and "Only holdings that show a profit" not in prompts.system_prompt(3)
    assert "OTHER symbols" in prompts.system_prompt(3)  # the rule that the money comes from other symbols is always in the prompt


def test_when_the_free_cash_pays_for_the_purchase_the_holdings_after_are_the_same_and_the_cash_is_spent():
    portfolio = no_cash().model_copy(update={"cash_avail": 1000.0})
    result = rf.search(problem(portfolio=portfolio), Gen({}))
    assert result["status"] == "funded" and result["after"] == {"shares": {"MSFT": 50, "TSLA": 123, "GM": 60, "MU": 2}, "cash": 180.0}


# ---------------------------------------------------------------- the same sells in another order are one plan (entry 45)

def test_the_same_sells_reached_in_another_order_are_one_plan_and_take_no_second_place_in_the_beam():
    gen = Gen({ROOT: [thought("GM", 5), thought("TSLA", 1)], "GM x5": [thought("TSLA", 1)], "TSLA x1": [thought("GM", 5), thought("MSFT", 1)]})
    result = rf.search(problem(), gen)
    level2 = result["trace"][1]["children"]
    duplicate = [c for c in level2 if c["status"] == "duplicate"]
    assert len(duplicate) == 1 and duplicate[0]["reason"] == "the same sells as n3 in a different order" and duplicate[0]["thought"] == thought("TSLA", 1) and duplicate[0]["parent"] == "n1"
    assert result["stats"]["nodes"] == 5   # n0, GM x5, TSLA x1, TSLA x1 + GM x5 (n3), TSLA x1 + MSFT x1 (n4): GM x5 + TSLA x1 would have been a sixth node, the same plan as n3


def test_a_plan_with_the_same_symbols_but_other_share_counts_is_a_different_plan():
    gen = Gen({ROOT: [thought("GM", 5), thought("TSLA", 1)], "GM x5": [thought("TSLA", 1)], "TSLA x1": [thought("GM", 6)]})
    result = rf.search(problem(), gen)
    assert not any(c["status"] == "duplicate" for level in result["trace"] for c in level["children"])


# ---------------------------------------------------------------- the size of the tree, and what the beam ranks by (entry 45): pure functions on the complex portfolio, no model

COMPLEX = DEFAULT_PORTFOLIO.parent / "jacob_pyrett_complex.json"
COMPLEX_PRICES = {"LLY": 1409.38, "SNDK": 1126.52, "MU": 855.10, "F": 12.31, "GM": 88.34, "FITB": 53.93, "PG": 162.06, "MCD": 263.04, "GOOGL": 280.03, "TSLA": 398.50, "MSFT": 520.99}  # the 2026-11-03 close


def complex_problem(**options):
    return rf.build_problem(pf.load_portfolio(COMPLEX), "LLY", 3, COMPLEX_PRICES, only_green=True, lean=True, **options)


def test_no_three_holdings_can_pay_for_the_first_complex_purchase_so_at_least_four_levels_are_needed():
    p = complex_problem()
    values = sorted((h.sellable * h.price for h in p.holdings.values() if p.eligible(h)), reverse=True)
    assert p.need == 4128.14 and round(sum(values), 2) == 7121.94 and "FITB" in p.holdings and not p.eligible(p.holdings["FITB"])   # FITB (at a loss) is not counted
    assert sum(values[:3]) < p.need <= sum(values[:4])


def test_the_depth_is_never_more_than_the_holdings_that_can_be_sold_and_the_problem_carries_the_size_of_the_tree():
    assert rf.effective_limits(complex_problem(), max_depth=50)["max_depth"] == 7   # eight holdings, one at a loss: seven can be sold
    sized = complex_problem(max_depth=5, branching=4)
    assert (sized.max_depth, sized.branching) == (5, 4) and rf.effective_limits(sized) == {"beam_width": 3, "branching": 4, "max_depth": 5, "max_calls": 12}
    assert rf.effective_limits(complex_problem()) == {"beam_width": 3, "branching": 3, "max_depth": 3, "max_calls": 12}


def test_the_priority_of_an_unfinished_plan_is_half_its_score_and_half_its_progress_with_explore_and_its_score_without():
    plain, smart = complex_problem(), complex_problem(explore=True)
    node, _ = rf.check_step(smart, rf.root(smart), {"symbol": "MCD", "shares": 6}, "n1")    # raises $1,578.24 of the $4,128.14
    assert rf.progress(smart, node) == pytest.approx(1578.24 / 4128.14, abs=1e-4) and rf.progress(smart, rf.root(smart)) == 0.0
    assert rf.priority(plain, node) == node.score
    assert rf.priority(smart, node) == pytest.approx(0.5 * node.score + 50 * 1578.24 / 4128.14, abs=0.01)
    assert rf.progress(smart, rf.Node("x", None, 0, shortfall=0.0)) == 1.0


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


def test_in_parallel_the_plans_of_a_level_call_the_generator_at_the_same_time_and_one_after_the_other_otherwise():
    parallel, sequential = SlowGen(THREE_BRANCHES), SlowGen(THREE_BRANCHES)
    rf.search(problem(), parallel, parallel=True)
    rf.search(problem(), sequential)
    assert parallel.most == 3 and sequential.most == 1   # level 2 has three plans to expand


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


# ---------------------------------------------------------------- a search a person is waiting for: the time limit (entry 48)

def test_the_search_records_the_seconds_at_the_end_of_each_level_and_when_the_first_plan_completed():
    gen = SlowGen({ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 4)]}, delay=0.03)   # level 1: an open plan; level 2: TSLA x4 completes it
    result = rf.search(problem(), gen)
    seconds = [level["seconds"] for level in result["trace"]]
    assert len(seconds) == 2 and 0.03 <= seconds[0] < seconds[1] and result["stats"]["first_plan_seconds"] == seconds[1] and result["stats"]["seconds"] >= seconds[1]


def test_with_no_plan_completed_the_first_plan_time_is_none():
    assert rf.search(problem(), Gen({ROOT: [thought("GM", 1)]}))["stats"]["first_plan_seconds"] is None


def test_once_the_time_limit_has_passed_no_new_level_starts_and_the_best_completed_plan_so_far_is_the_answer():
    gen = SlowGen({ROOT: [thought("TSLA", 4), thought("GM", 1)], "GM x1": [thought("MSFT", 2)]}, delay=0.1)
    result = rf.search(problem(), gen, deadline=0.05)   # level 1 takes 0.1 s: the limit has passed when it ends, with TSLA x4 completed and GM x1 still open
    assert [c[1] for c in gen.calls] == [ROOT] and result["status"] == "plan" and [(x["symbol"], x["shares"]) for x in result["plan"]["sells"]] == [("TSLA", 4)]
    assert "the time limit (0.05 s) was reached; the best completed plan is the answer" in result["stop"] and result["limits"]["deadline"] == 0.05


def test_with_nothing_completed_when_the_time_is_up_nothing_is_sold():
    result = rf.search(problem(), SlowGen({ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 4)]}, delay=0.1), deadline=0.05)
    assert result["status"] == "infeasible" and result["plan"] is None and result["after"] == {} and "the time limit (0.05 s) was reached" in result["stop"]


def test_without_a_limit_the_search_runs_as_before_and_a_limit_not_reached_changes_nothing():
    table = {ROOT: [thought("GM", 1)], "GM x1": [thought("TSLA", 4)]}
    free, limited = rf.search(problem(), Gen(table)), rf.search(problem(), Gen(table), deadline=60)
    assert free["limits"]["deadline"] is None and limited["status"] == free["status"] == "plan" and free["stop"] == limited["stop"]
    assert [(x["symbol"], x["shares"]) for x in limited["plan"]["sells"]] == [(x["symbol"], x["shares"]) for x in free["plan"]["sells"]]


def test_in_parallel_a_level_already_out_is_waited_for_after_the_limit():
    gen = SlowGen(THREE_BRANCHES, delay=0.15)
    result = rf.search(problem(), gen, deadline=0.2, parallel=True)   # level 1 ends at 0.15 s (in time); level 2's three calls start together and finish at 0.30 s: after the limit, waited for
    assert len(result["trace"]) == 2 and gen.most == 3 and "the time limit" in result["stop"] and result["stats"]["seconds"] >= 0.29


# ---------------------------------------------------------------- the tree raises its own depth (entry 49)

def strict_complex(target="LLY", shares=3, **options):
    return rf.build_problem(pf.load_portfolio(COMPLEX), target, shares, COMPLEX_PRICES, only_green=True, lean=True, never_liquidate=True, **options)


def test_the_fewest_stocks_that_could_pay_for_a_purchase_is_arithmetic_on_the_biggest_holdings_first():
    p = strict_complex()   # 3 LLY: $4,128.14 to raise; with one share of each kept, the biggest sellable lots are 1,315.2, 840.09, 797, 795.06, 648.24 ...
    assert p.fewest_stocks() == 5
    assert strict_complex("SNDK", 1).fewest_stocks() == 1 and strict_complex("MU", 3).fewest_stocks() == 3   # one lot pays for 1 SNDK; 3 MU ($2,465.30 to raise) take three lots
    assert strict_complex("MU", 10).fewest_stocks() is None                                                # 10 MU cost more than everything that may be sold


def test_the_depth_a_purchase_needs_is_the_fewest_stocks_plus_two_between_the_default_and_the_holdings_that_can_be_sold():
    assert strict_complex().auto_depth() == 7                     # 5 + 2, and seven holdings can be sold
    assert strict_complex("SNDK", 1).auto_depth() == 3            # 1 + 2 = 3: the default
    assert strict_complex("MU", 3).auto_depth() == 5              # 3 + 2
    assert strict_complex("MU", 10).auto_depth() == 7             # cannot be paid for: the depth of all the holdings (the search will find nothing)
    assert strict_complex().auto_depth(slack=0) == 5 and strict_complex().auto_depth(slack=9) == 7   # never more than the holdings that can be sold


def test_the_default_depth_of_three_cannot_find_a_plan_for_a_purchase_that_needs_five_stocks_and_the_automatic_depth_can():
    table = {ROOT: [thought("MCD", 5)], "MCD x5": [thought("GOOGL", 3)], "MCD x5, GOOGL x3": [thought("TSLA", 2)], "MCD x5, GOOGL x3, TSLA x2": [thought("GM", 9)],
             "MCD x5, GOOGL x3, TSLA x2, GM x9": [thought("PG", 3)]}
    short = rf.search(strict_complex(), Gen(table))                                         # depth 3
    deep = rf.search(strict_complex(max_depth=strict_complex().auto_depth()), Gen(table))   # depth 7
    assert short["status"] == "infeasible" and "the depth limit (3) was reached" in short["stop"]
    assert deep["status"] == "plan" and [s for s, _ in [(x["symbol"], x["shares"]) for x in deep["plan"]["sells"]]] == ["MCD", "GOOGL", "TSLA", "GM", "PG"] and deep["stats"]["levels"] == 5


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
