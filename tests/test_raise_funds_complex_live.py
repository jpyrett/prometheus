"""raise_funds_complex: the complex RaiseFunds scenario run LIVE with a large and a smaller model, and compared (design choices entry 45).

Run:  PROMETHEUS_LIVE_TESTS=1 pytest tests/test_raise_funds_complex_live.py -q -s

THIS TEST CALLS THE PAID APIs: 11 Finnhub quotes and about 20 to 25 model calls for each of the two models, for about 3 minutes in all (the two runs go at the same time, and the plans of a level are
expanded at the same time: entry 46; it was about 6 minutes each before), a few cents. It writes no file and
changes no portfolio. It is skipped unless PROMETHEUS_LIVE_TESTS=1, so that a plain `pytest` does not spend money (AGENTS.md: a paid call needs Jacob's yes); there is no other offline fake here: the Data
Clerk's quotes and the model are real.

THE SCENARIO (the made-up portfolio configs/portfolios/jacob_pyrett_complex.json): eight small holdings worth $7,553.38 at the 2026-11-03 close, $100 of cash, FITB at a loss; three purchases in a row,
3 LLY, 1 SNDK and up to 3 MU, selling only holdings in profit and never a whole position (the evaluator of entry 47, on by default), lean, the most profitable first (`--green-first`), with a tree up to 5
levels deep, 4 thoughts per plan and a beam of 3. LLY needs at least
four stocks, so a tree that stops at level 3 finds nothing.

THE TWO MODELS are the `account_manager` role (ACCOUNT_MANAGER_MODEL, openai/gpt-oss-120b) and the `fast` role (FAST_MODEL, qwen/qwen3-30b-a3b): `raise_funds_complex(model_id)` runs the scenario with either.
What the tests check is what must hold for ANY model, because the code, not the model, checks every thought: every node of every tree is valid by a calculation that does not use the search's own functions,
the stock being bought and the holding at a loss are never sold, money and shares add up, LLY needs a plan of at least four stocks. What may differ between models, and is printed and compared, is the
tree: how many nodes, how many calls and tokens, how long, and which plan wins. Prices are live, so no number of shares or dollars is asserted.
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_UP, Decimal

import pytest

from prometheus.agents.account_manager import account_manager as am
from prometheus.common import portfolio as pf
from prometheus.common.money import money, price
from prometheus.models import chat
from prometheus.reasoning import raise_funds as rf

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.environ.get("PROMETHEUS_LIVE_TESTS") != "1",
                                                   reason="calls the paid Finnhub and OpenRouter APIs: set PROMETHEUS_LIVE_TESTS=1 to run it")]

SCENARIO = am.DEFAULT_PORTFOLIO.parent / "jacob_pyrett_complex.json"
PURCHASES = [("LLY", 3), ("SNDK", 1), ("MU", 3)]
SIZE = dict(max_depth=5, branching=4, beam_width=3, max_calls=30)
MODELS = {"large": chat.model_name("account_manager"), "small": chat.model_name("fast")}   # gpt-oss-120b and qwen3-30b-a3b unless .env says otherwise


def raise_funds_complex(model_id: str | None = None, portfolio=None, purchases=PURCHASES, **options) -> dict:
    """The complex scenario with a real model (`model_id`, or the account_manager role's): the three purchases in a row, from the made-up portfolio, green-first, lean, only holdings in profit, with a
    deep and wide tree. Returns the result of `raise_funds_many` plus `seconds` and `model`. `options` override the size of the tree or the rules."""
    model_id = model_id or MODELS["large"]
    manager = am.AccountManager(model_id=model_id, deadline=None)   # the defaults of entry 50 (the reasoning effort is low for gpt-oss and the model's own for any other), no time limit   # "low" reasoning effort: gpt-oss thinks for thousands of tokens before a short JSON answer
    started = time.perf_counter()
    result = manager.raise_funds_many(portfolio or pf.load_portfolio(SCENARIO), list(purchases), **{**SIZE, **options})
    return {**result, "seconds": round(time.perf_counter() - started, 1), "model": manager.run["model"]}


@pytest.fixture(scope="module")
def runs() -> dict[str, dict]:
    """Both models, run once, in parallel, on the REAL clock: this module-scoped fixture runs before conftest's per-test `live_clock_on`, so without this it would inherit Jacob's .env (USE_LIVE_TRADING=false:
    the simulated clock and the stored close) and the quotes would not be Finnhub's."""
    with pytest.MonkeyPatch.context() as live:
        live.setenv("USE_LIVE_TRADING", "true")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {kind: pool.submit(raise_funds_complex, model) for kind, model in MODELS.items()}
            return {kind: future.result() for kind, future in futures.items()}


@pytest.fixture(params=sorted(MODELS))
def run(request, runs):
    return runs[request.param]


def nodes(purchase):
    return [c for level in purchase["trace"] for c in level["children"] if "plan" in c]


def sells(purchase):
    return [(x["symbol"], x["shares"]) for x in purchase["plan"]["sells"]]


def prices_of(result):
    """The live quotes as prices: the Data Clerk's quote is a float (what Finnhub's JSON gives), and a price crosses into the account's code through the price door (entry 112)."""
    return {symbol: price(quote["price"]) for symbol, quote in result["quotes"].items()}


def cents(shares, each: Decimal) -> Decimal:
    """What `shares` at `each` come to, in cents, half up: the independent calculation of the book's rounding, written here and not taken from the search."""
    return (Decimal(shares) * each).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fewest_stocks_needed(result) -> int:
    """From the live prices: how many of the biggest holdings in profit have to be sold to raise what the first purchase needs (calculated, not searched)."""
    start = pf.Portfolio.model_validate(result["start"])
    p = rf.build_problem(start, "LLY", 3, prices_of(result), only_green=True, lean=True, never_liquidate=True)
    total, count = Decimal("0.00"), 0
    for value in sorted((h.sellable * h.price for h in p.holdings.values() if p.eligible(h)), reverse=True):
        total, count = total + value, count + 1
        if total >= p.need:
            return count
    return 99


# ---------------------------------------------------------------- what must hold for any model

def test_the_run_finished_with_quotes_for_every_symbol_from_the_data_clerk(run):
    assert run["status"] == "done" and run["prices_order"]["action_type"] == "fetch_quote" and run["prices_order"]["status"] == "data_ready"
    assert set(run["quotes"]) == {"LLY", "SNDK", "MU", "F", "GM", "FITB", "PG", "MCD", "GOOGL", "TSLA", "MSFT"}
    assert all(quote["price"] > 0 and quote["source"] == "Finnhub /quote" for quote in run["quotes"].values())


def test_the_first_purchase_needs_at_least_four_stocks_and_the_tree_went_that_deep(run):
    first, k = run["purchases"][0], fewest_stocks_needed(run)
    assert k >= 4   # the scenario is built so that no three holdings can pay for 3 LLY: if the prices ever change that, the scenario has to be rebuilt
    assert first["status"] == "plan" and len(first["plan"]["sells"]) >= k and first["stats"]["levels"] >= k


def test_the_first_purchase_is_paid_for_and_no_purchase_ended_in_an_error(run):
    statuses = [p["status"] for p in run["purchases"]]
    assert statuses[0] == "plan" and set(statuses) <= {"plan", "infeasible"} and "error" not in statuses
    assert not run["purchases"][0]["errors"]   # no model call failed in the first purchase


def test_every_node_of_every_tree_is_valid_by_an_independent_calculation(run):
    """Recomputes proceeds, shortfall and every rule from the portfolio file and the live quotes, without the search's own functions."""
    prices, holdings, cash = prices_of(run), {p["symbol"]: p for p in run["start"]["positions"]}, money(run["start"]["cash_avail"])      # the start is JSON: its numbers go through the doors
    for purchase in run["purchases"]:
        target, shares = purchase["target"], purchase["shares"]
        cost = cents(shares, prices[target])
        for n in nodes(purchase):
            plan = [(m.group(1), int(m.group(2))) for m in re.finditer(r"(\w+) x(\d+)", n["plan"])]
            symbols = [s for s, _ in plan]
            assert len(set(symbols)) == len(symbols) and target not in symbols and "FITB" not in symbols
            assert all(k < holdings[s]["shares"] and prices[s] > price(holdings[s]["cost_basis"]) for s, k in plan)   # one share of every holding kept, and every sale in profit
            proceeds = sum((cents(k, prices[s]) for s, k in plan), Decimal("0.00"))
            assert proceeds == n["proceeds"] and max(Decimal("0.00"), cost - cash - proceeds) == n["shortfall"]
            if n["status"] == "complete":   # lean: the last step could not have sold one share fewer
                assert proceeds - prices[plan[-1][0]] + cash < cost
        if purchase["plan"]:
            for symbol, n in sells(purchase):
                holdings[symbol] = {**holdings[symbol], "shares": holdings[symbol]["shares"] - n}
            cash = purchase["after"]["cash"]      # a Decimal


def test_every_purchase_that_was_made_meets_the_success_measure(run):
    """Jacob's measure: profitable stocks were found to sell (every sale shows a gain), no position was sold out, and the requested shares were bought."""
    made = [p for p in run["purchases"] if p["status"] == "plan"]
    assert made and all(rf.success(p)["ok"] for p in made)
    first = rf.success(run["purchases"][0])
    assert first["asked"] == 3 and first["bought"] == 3   # LLY: all three asked were bought


def test_the_holding_at_a_loss_and_the_stock_being_bought_were_never_sold(run):
    for purchase in run["purchases"]:
        if purchase["plan"]:
            sold = [s for s, _ in sells(purchase)]
            assert "FITB" not in sold and purchase["target"] not in sold
    assert {p["symbol"]: p["shares"] for p in run["final"]["positions"]}["FITB"] == 8


def test_shares_and_dollars_add_up_across_the_three_purchases(run):
    final = {p["symbol"]: p["shares"] for p in run["final"]["positions"]}
    expected = {p["symbol"]: p["shares"] for p in run["start"]["positions"]}
    cash = money(run["start"]["cash_avail"])
    for purchase in run["purchases"]:
        if purchase["plan"]:
            for s, k in sells(purchase):
                expected[s] -= k
            expected[purchase["target"]] = expected.get(purchase["target"], 0) + purchase["shares"]   # the shares BOUGHT: an 'up to 3' purchase may buy 2
            cash = cash + purchase["plan"]["proceeds"] - purchase["cost"]
    assert final == {s: k for s, k in expected.items() if k > 0} and money(run["final"]["cash_avail"]) == cash      # exactly, to the cent: the sums are Decimal
    assert all(final[p["symbol"]] >= 1 for p in run["start"]["positions"])   # no position was sold out


def test_the_tree_is_wide_and_deep_and_the_search_dropped_repeats(run):
    first = run["purchases"][0]
    assert first["stats"]["nodes"] >= 15 and first["stats"]["completed_plans"] >= 3 and max(len(level["expanded"]) for level in first["trace"]) >= 2
    kinds = {c["status"] for level in first["trace"] for c in level["children"]}
    assert {"complete", "kept"} <= kinds


# ---------------------------------------------------------------- the two models compared

def test_the_smaller_model_against_the_larger_one(runs, capsys):
    """Prints both runs side by side. Asserts only that the smaller model, like the larger, paid for the first purchase and met the success measure: whether its answers are the same is the finding."""
    large, small = runs["large"], runs["small"]
    rows = []
    for label, r in (("large", large), ("small", small)):
        rows.append(f"{label:>5}  {r['model']:<26} {r['seconds']:>6.0f}s {r['run']['calls']:>3} calls {r['run']['tokens_in']:>6} in {r['run']['tokens_out']:>6} out   "
                    + " | ".join(f"{p['target']}: {p['stats']['nodes']}n/{p['stats']['levels']}l/{p['stats']['completed_plans']}c " + ("+".join(f"{s}{k}" for s, k in sells(p)) if p["plan"] else p["status"]) for p in r["purchases"]))
    answers_match = [sells(a) == sells(b) if a["plan"] and b["plan"] else a["status"] == b["status"] for a, b in zip(large["purchases"], small["purchases"])]
    with capsys.disabled():
        print("\n\nTHE SAME COMPLEX SCENARIO, TWO MODELS (nodes / levels / completed plans per purchase, then what was sold)")
        print("\n".join(rows))
        print(f"same sells in purchase 1, 2, 3: {answers_match}   same final portfolio: "
              f"{ {p['symbol']: p['shares'] for p in large['final']['positions']} == {p['symbol']: p['shares'] for p in small['final']['positions']} }\n")
    for r in (large, small):
        assert r["purchases"][0]["status"] == "plan" and all(rf.success(p)["ok"] for p in r["purchases"] if p["status"] == "plan")
