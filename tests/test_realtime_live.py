"""Are the funding plans RIGHT on a harder portfolio? The portfolio of "claude", four purchases, LIVE (design choices entries 48 and 86).

Run:  PROMETHEUS_LIVE_TESTS=1 pytest tests/test_realtime_live.py -q -s

THIS TEST CALLS THE PAID APIs (Finnhub quotes and about 100 gpt-oss-120b calls in all, a few cents). Skipped unless PROMETHEUS_LIVE_TESTS=1. No fake: the quotes and the model are real. It was written
to ask whether the Tree of Thought was quick enough for a person to wait for; Jacob decided (2026-11-15, entry 86) that how long a session takes is not what these tests are for, so every test that
measured time was deleted. What is left asks only whether the plans are correct.

THE MADE-UP SCENARIO (`configs/portfolios/claude.json`, user "claude", chosen to be harder than the earlier ones): eleven holdings across price ranges ($20 to $620 a share) worth about $27,000 at the
2026-11-03 close, $800 of cash, two holdings at a loss (PFE and NKE: never sold). Four purchases in a row, with the rules of entry 47 (only holdings in profit, one share of each kept), lean, green-first:
  1. 10 NFLX     $634: the CASH pays for it, nothing is sold and no model is asked;
  2. 9 COST      $9,173: needs at least THREE stocks (no holding is worth more than $3,600), so the tree must go three levels deep;
  3. 10 AVGO     $3,321: two or three stocks, from what is left;
  4. up to 40 NVDA: "as many as the portfolio can pay for": 29 at these prices, which takes nearly everything that is left, the heaviest purchase.

THE TWO SESSIONS: `realtime` (a limit of 30 seconds a purchase: when it passes no new level starts and the best plan COMPLETED so far is the answer) and `hurried` (a limit of 1 second, far too short, to see
that when time runs out before a plan exists nothing is sold, and nothing half planned is).
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("PROMETHEUS_LIVE_TESTS") != "1",
                                                   reason="calls the paid Finnhub and OpenRouter APIs: set PROMETHEUS_LIVE_TESTS=1 to run it")]

PORTFOLIO = am.DEFAULT_PORTFOLIO.parent / "claude.json"
PURCHASES = [("NFLX", 10), ("COST", 9), ("AVGO", 10), ("NVDA", 40)]
MODEL = "openai/gpt-oss-120b"   # the larger model, by name: not whatever a role's setting says
EFFORT = "low"                  # its reasoning effort in the tests below: see the guide, section 9m, for low against medium and high
SIZE = dict(branching=3, beam_width=3)   # the depth and the call budget are chosen from each purchase (entry 49)
DEADLINE = 30.0          # seconds a purchase may take before the best plan so far is taken


def claude_session(deadline: float | None = None, effort: str | None = EFFORT, model: str = MODEL, **options) -> dict:
    """The four purchases of `claude`, with the real model and quotes, strict, lean, green-first, the plans of a level in parallel. `deadline`: seconds a purchase may take; `effort`: the model's reasoning
    effort (low, medium, high); `options` override the size of the tree (beam_width, branching, max_depth) or the rules."""
    import time
    manager = am.AccountManager(model_id=model, effort=effort, deadline=deadline)   # every other setting is the default of entry 50: strict, green-first, lean, explore, parallel, up to
    started = time.perf_counter()
    result = manager.raise_funds_many(pf.load_portfolio(PORTFOLIO), list(PURCHASES), **{**SIZE, **options})
    return {**result, "seconds": round(time.perf_counter() - started, 1), "deadline": deadline}


@pytest.fixture(scope="module")
def sessions() -> dict[str, dict]:
    """Two sessions at the same time: one with a limit of 30 seconds a purchase, and one with a limit that is far too short. On the REAL clock: this module-scoped fixture runs before conftest's per-test
    `live_clock_on`, so without this it would inherit Jacob's .env (USE_LIVE_TRADING=false: the simulated clock and the stored close) and the quotes would not be Finnhub's."""
    with pytest.MonkeyPatch.context() as live:
        live.setenv("USE_LIVE_TRADING", "true")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {"realtime": pool.submit(claude_session, DEADLINE), "hurried": pool.submit(claude_session, 1.0)}
            return {name: future.result() for name, future in futures.items()}


@pytest.fixture
def realtime(sessions):
    return sessions["realtime"]


def by_target(session) -> dict[str, dict]:
    return {p["target"]: p for p in session["purchases"]}


def sells(purchase):
    return [(x["symbol"], x["shares"]) for x in purchase["plan"]["sells"]]


def prices_of(session):
    return {symbol: quote["price"] for symbol, quote in session["quotes"].items()}


# ---------------------------------------------------------------- the scenario is what it says

def test_the_user_is_claude_with_eleven_holdings_two_of_them_at_a_loss(realtime):
    assert realtime["start"]["user"] == "claude" and len(realtime["start"]["positions"]) == 11 and realtime["start"]["cash_avail"] == 800.0
    prices = prices_of(realtime)
    assert sorted(p["symbol"] for p in realtime["start"]["positions"] if prices[p["symbol"]] <= p["cost_basis"]) == ["NKE", "PFE"]
    assert all(quote["source"] == "Finnhub /quote" for quote in realtime["quotes"].values()) and realtime["status"] == "done"


def test_the_second_purchase_needs_at_least_three_stocks_so_the_tree_has_to_go_three_levels_deep(realtime):
    start = pf.Portfolio.model_validate(realtime["start"])
    p = rf.build_problem(start, "COST", 9, prices_of(realtime), only_green=True, lean=True, never_liquidate=True)
    values = sorted((h.sellable * h.price for h in p.holdings.values() if p.eligible(h)), reverse=True)
    assert values[0] < p.need and values[0] + values[1] < p.need <= sum(values[:3])   # two stocks cannot pay for it, three can


# ---------------------------------------------------------------- what it found is right

def test_the_purchase_the_cash_pays_for_needs_no_model_call(realtime):
    nflx = by_target(realtime)["NFLX"]
    assert nflx["status"] == "funded" and nflx["stats"]["llm_calls"] == 0 and nflx["plan"] is None


def test_every_purchase_that_was_made_meets_the_success_measure(realtime):
    made = [p for p in realtime["purchases"] if p["status"] in ("plan", "funded")]
    assert [p["target"] for p in made][:3] == ["NFLX", "COST", "AVGO"] and all(rf.success(p)["ok"] for p in made)
    assert rf.success(by_target(realtime)["COST"])["bought"] == 9 and rf.success(by_target(realtime)["AVGO"])["bought"] == 10


def test_the_heaviest_purchase_either_finds_a_plan_or_sells_nothing(realtime):
    nvda = by_target(realtime)["NVDA"]
    assert (nvda["status"] == "plan" and rf.success(nvda)["ok"]) or (nvda["status"] == "infeasible" and nvda["plan"] is None and nvda["after"] == {})   # a plan, or nothing sold


def test_every_node_of_every_tree_is_valid_by_an_independent_calculation(realtime):
    """Recomputes proceeds, shortfall and every rule from the portfolio file and the live quotes, without the search's own functions."""
    prices, holdings, cash = prices_of(realtime), {p["symbol"]: p for p in realtime["start"]["positions"]}, realtime["start"]["cash_avail"]
    for purchase in realtime["purchases"]:
        target, shares = purchase["target"], purchase["shares"]
        cost = round(shares * prices[target], 2)
        for level in purchase["trace"]:
            for n in (c for c in level["children"] if "plan" in c):
                plan = [(m.group(1), int(m.group(2))) for m in re.finditer(r"(\w+) x(\d+)", n["plan"])]
                assert len({s for s, _ in plan}) == len(plan) and target not in {s for s, _ in plan}
                assert all(k < holdings[s]["shares"] and prices[s] > holdings[s]["cost_basis"] for s, k in plan)   # one share kept, every sale in profit
                proceeds = round(sum(k * prices[s] for s, k in plan), 2)
                assert proceeds == n["proceeds"] and round(max(0.0, cost - cash - proceeds), 2) == n["shortfall"]
        if purchase["status"] in ("plan", "funded"):
            for symbol, k in (sells(purchase) if purchase["plan"] else []):
                holdings[symbol] = {**holdings[symbol], "shares": holdings[symbol]["shares"] - k}
            cash = purchase["after"]["cash"]


def test_shares_and_dollars_add_up_and_no_position_was_sold_out(realtime):
    final = {p["symbol"]: p["shares"] for p in realtime["final"]["positions"]}
    expected = {p["symbol"]: p["shares"] for p in realtime["start"]["positions"]}
    cash = realtime["start"]["cash_avail"]
    for purchase in realtime["purchases"]:
        if purchase["status"] in ("plan", "funded"):
            for s, k in (sells(purchase) if purchase["plan"] else []):
                expected[s] -= k
            expected[purchase["target"]] = expected.get(purchase["target"], 0) + purchase["shares"]
            cash = round(cash + (purchase["plan"]["proceeds"] if purchase["plan"] else 0.0) - purchase["cost"], 2)
    assert final == {s: k for s, k in expected.items() if k > 0} and realtime["final"]["cash_avail"] == pytest.approx(cash, abs=0.01)
    assert all(final[p["symbol"]] >= 1 for p in realtime["start"]["positions"]) and final["PFE"] == 80 and final["NKE"] == 20   # nothing sold out; the two losses untouched


# ---------------------------------------------------------------- when time runs out before a plan exists

def test_a_limit_that_is_far_too_short_sells_nothing_rather_than_something_half_planned(sessions):
    hurried = by_target(sessions["hurried"])
    for target in ("COST", "AVGO", "NVDA"):
        purchase = hurried[target]
        assert (purchase["status"] == "plan" and rf.success(purchase)["ok"]) or (purchase["status"] == "infeasible" and purchase["plan"] is None and purchase["after"] == {})
    assert any(p["status"] == "infeasible" and "the time limit (1 s) was reached" in p["stop"] for p in sessions["hurried"]["purchases"])   # at least one ran out of time and sold nothing
