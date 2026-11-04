"""Is the Tree of Thought quick enough to trade in real time? The portfolio of "claude", four purchases, LIVE (design choices entry 48).

Run:  PROMETHEUS_LIVE_TESTS=1 pytest tests/test_realtime_live.py -q -s

THIS TEST CALLS THE PAID APIs (Finnhub quotes and about 100 gpt-oss-120b calls in all, a few cents) and takes about 4 minutes: three sessions run at the same time. Skipped unless PROMETHEUS_LIVE_TESTS=1
(AGENTS.md: a paid call needs Jacob's yes). No fake: the quotes and the model are real.

THE QUESTION (Jacob, 2026-11-03): "Focus on checking if this will be quick enough for the user to run this trading in real time. It can't be too slow to be able to find some stocks to sell."

THE MADE-UP SCENARIO (`configs/portfolios/claude.json`, user "claude", chosen to be harder than the earlier ones): eleven holdings across price ranges ($20 to $620 a share) worth about $27,000 at the
2026-11-03 close, $800 of cash, two holdings at a loss (PFE and NKE: never sold). Four purchases in a row, with the rules of entry 47 (only holdings in profit, one share of each kept), lean, green-first:
  1. 10 NFLX     $634: the CASH pays for it, nothing is sold and no model is asked;
  2. 9 COST      $9,173: needs at least THREE stocks (no holding is worth more than $3,600), so the tree must go three levels deep;
  3. 10 AVGO     $3,321: two or three stocks, from what is left;
  4. up to 40 NVDA: "as many as the portfolio can pay for": 29 at these prices, which takes nearly everything that is left, the heaviest purchase.

THE THREE SESSIONS: `realtime` (a limit of 30 seconds a purchase: when it passes no new level starts and the best plan COMPLETED so far is the answer), `unlimited` (no limit: what the tree finds when it is
left alone) and `hurried` (a limit of 1 second, far too short, to see what happens when time runs out before a plan exists: nothing is sold).

WHAT "REAL TIME" MEANS HERE, as thresholds (set from the measured runs, with room for a slow call): a purchase answers within 60 seconds (the 30-second limit plus the level that is in flight when it passes: 35 seconds measured alone, 44 with three sessions at once);
the first valid plan of an ordinary purchase (2 and 3) is there within 30 seconds; the whole session of four purchases is done within 150 seconds; and the answer under the limit is at least 90% as good as
the unlimited one.

MEASURED, entry 48 (OpenRouter's own choice of provider for gpt-oss-120b, about 55 tokens a second): COST 34 s and AVGO 36 s with the first plan at 18 s and 15 s, NVDA 35 s, the session 108 s against
193 s without the limit, the same plans and scores either way (89.6, 88.4, 87.7). MEASURED, entry 50 (the default: Cerebras, then Groq, about 650 and 370 tokens a second, effort low, beam 3, depth
chosen): COST 3 s, AVGO 3 to 5 s, NVDA 2 to 4 s, the first plan at 1 to 4 s, **the session 10 to 14 s**, the same plans and scores (89.6, 88.4, 87.7). With the fast providers the 30-second limit never
binds, which is why `hurried` is 1 second.
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
SLACK = 30.0             # the level that is in flight when the limit passes: 10 to 15 seconds alone, more when three sessions share the connection
FIRST_PLAN = 30.0        # seconds to the first valid plan of an ordinary purchase
SESSION = 150.0          # seconds for the four purchases
KEEP_QUALITY = 0.90      # the score under the limit, as a share of the unlimited score


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
    """Three sessions at the same time: the limited one, the unlimited one, and one with a limit that is far too short."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {"realtime": pool.submit(claude_session, DEADLINE), "unlimited": pool.submit(claude_session), "hurried": pool.submit(claude_session, 1.0)}
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


# ---------------------------------------------------------------- is it quick enough?

def test_every_purchase_answers_within_the_time_a_person_will_wait(realtime):
    for purchase in realtime["purchases"]:
        assert purchase["stats"]["seconds"] <= DEADLINE + SLACK, (purchase["target"], purchase["stats"]["seconds"])


def test_the_purchase_the_cash_pays_for_needs_no_model_call_and_no_time(realtime):
    nflx = by_target(realtime)["NFLX"]
    assert nflx["status"] == "funded" and nflx["stats"]["llm_calls"] == 0 and nflx["stats"]["seconds"] < 1.0 and nflx["plan"] is None


def test_a_first_valid_plan_is_there_quickly_for_the_ordinary_purchases(realtime):
    for target in ("COST", "AVGO"):
        purchase = by_target(realtime)[target]
        assert purchase["status"] == "plan" and 0 < purchase["stats"]["first_plan_seconds"] <= FIRST_PLAN, (target, purchase["stats"])


def test_the_whole_session_of_four_purchases_fits_in_real_time_and_the_limit_makes_it_faster(sessions):
    realtime, unlimited = sessions["realtime"], sessions["unlimited"]
    assert realtime["seconds"] <= SESSION, realtime["seconds"]
    assert sum(p["stats"]["seconds"] for p in realtime["purchases"]) < 0.9 * sum(p["stats"]["seconds"] for p in unlimited["purchases"])   # the limit saves time


def test_nothing_is_given_up_for_the_speed(sessions):
    realtime, unlimited = by_target(sessions["realtime"]), by_target(sessions["unlimited"])
    for target in ("COST", "AVGO", "NVDA"):
        if realtime[target]["plan"] and unlimited[target]["plan"]:
            assert realtime[target]["plan"]["score"] >= KEEP_QUALITY * unlimited[target]["plan"]["score"], (target, realtime[target]["plan"]["score"], unlimited[target]["plan"]["score"])


# ---------------------------------------------------------------- what it found is right

def test_every_purchase_that_was_made_meets_the_success_measure(realtime):
    made = [p for p in realtime["purchases"] if p["status"] in ("plan", "funded")]
    assert [p["target"] for p in made][:3] == ["NFLX", "COST", "AVGO"] and all(rf.success(p)["ok"] for p in made)
    assert rf.success(by_target(realtime)["COST"])["bought"] == 9 and rf.success(by_target(realtime)["AVGO"])["bought"] == 10


def test_the_heaviest_purchase_either_finds_a_plan_in_time_or_sells_nothing(realtime):
    nvda = by_target(realtime)["NVDA"]
    assert nvda["stats"]["seconds"] <= DEADLINE + SLACK
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
        assert purchase["stats"]["seconds"] <= 1.0 + SLACK
        assert (purchase["status"] == "plan" and rf.success(purchase)["ok"]) or (purchase["status"] == "infeasible" and purchase["plan"] is None and purchase["after"] == {})
    assert sessions["hurried"]["seconds"] < sessions["realtime"]["seconds"]
    assert any(p["status"] == "infeasible" and "the time limit (1 s) was reached" in p["stop"] for p in sessions["hurried"]["purchases"])   # at least one ran out of time and sold nothing


# ---------------------------------------------------------------- the numbers, side by side

def test_the_timings_of_the_three_sessions(sessions, capsys):
    with capsys.disabled():
        print("\n\nIS IT QUICK ENOUGH? the portfolio of claude, four purchases (seconds; first completed plan in brackets; score)")
        for name in ("realtime", "unlimited", "hurried"):
            s = sessions[name]
            limit = f"limit {s['deadline']:g} s" if s["deadline"] else "no limit"
            parts = []
            for p in s["purchases"]:
                st = p["stats"]
                first = f" [{st['first_plan_seconds']:.0f}]" if st.get("first_plan_seconds") is not None else ""
                parts.append(f"{p['target']} {st['seconds']:.0f}s{first} {st['llm_calls']}c " + (f"{p['plan']['score']}" if p["plan"] else p["status"]))
            print(f"  {name:<9} {limit:<12} session {s['seconds']:>5.0f} s   " + " | ".join(parts))
        print()
