"""A complex, made-up example: two purchases in a row, F and then FITB, paid for by selling what is in profit (design choices entry 43).

Run:  pytest tests/test_two_purchases.py -q

The portfolio is made up (configs/portfolios/jacob_pyrett_f_fitb.json): Jacob's MSFT 50 @ 150, TSLA 123 @ 300 and GM 60 @ 10, plus F 260 @ 2 and FITB 52 @ 15, and 51.23 of free cash. The prices are made up too
(a fake toolbox behind the REAL Data Clerk tool and work order). The "LLM" is scripted by what it is asked: for each purchase and each plan so far it answers with the thoughts the test wants, among them
thoughts that break the rules (sell the stock being bought, sell a holding at a loss, sell shares bought a minute ago). The Data Clerk's quote tool, the tree search, the hard checks, the scores and the
report are the real code. No network, no key, no cost.

The made-up prices: MSFT 420 (basis 150, in profit), TSLA 250 (basis 300, AT A LOSS), GM 55 (basis 10), F 12 (basis 2), FITB 45 (basis 15).
Purchase 1: 300 F = $3,600 (to raise $3,548.77 after the free cash).  Purchase 2: 40 FITB = $1,800.
"""
from decimal import ROUND_HALF_UP, Decimal
import json
import re
from datetime import date

import pytest
from langchain_core.messages import AIMessage

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf

SCENARIO = am.DEFAULT_PORTFOLIO.parent / "jacob_pyrett_f_fitb.json"
PRICES = {"MSFT": Decimal("420.00"), "TSLA": Decimal("250.00"), "GM": Decimal("55.00"), "F": Decimal("12.00"), "FITB": Decimal("45.00")}
TODAY = date(2026, 11, 3)


class QuoteToolbox:
    def __init__(self, prices=None):
        self.prices, self.asked = dict(PRICES if prices is None else prices), []

    def get_quote(self, symbol):
        self.asked.append(symbol)
        if symbol not in self.prices:
            raise RuntimeError(f"No quote returned for {symbol}")
        p = float(self.prices[symbol])      # the Data Clerk's quote is what an API's JSON gives, a float: the Account Manager turns it into a Decimal price itself
        return {"symbol": symbol, "price": p, "change": 0.0, "change_percent": 0.0, "high": p, "low": p, "open": p, "previous_close": p,
                "as_of_utc": "2026-11-03T21:00:00Z", "source": "MADE UP (test)"}


class ScriptedByPlan:
    """The 'LLM': answers each call from a table keyed by (the stock being bought, the plan so far), read from the prompt it is given. Records what it was asked."""
    model_name = "scripted/two-purchases"

    def __init__(self, table):
        self.table, self.seen = table, []

    def invoke(self, messages):
        text = messages[-1].content
        self.seen.append(text)
        target = re.search(r"buy \d+ share\(s\) of (\w+)", text).group(1)
        plan = re.search(r"Plan so far: (.*?)(?: \(raises| \(still short)", text).group(1)
        thoughts = self.table.get((target, plan), [])
        return AIMessage(content=json.dumps({"thoughts": [{"symbol": s, "shares": n, "idea": f"idea: sell {n} {s}"} for s, n in thoughts]}),
                         usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


ROOT = "(nothing sold)"
LIMITS = dict(branching=5, beam_width=2)  # the test script gives up to five thoughts per node (three of them break rules); a beam of 2 makes it prune
TABLE = {
    # PURCHASE 1: buy 300 F, $3,548.77 to raise. Thoughts 1 to 3 break rules: F is the stock being bought, TSLA is at a loss, FITB has 52 shares (so 60 is too many).
    ("F", ROOT): [("F", 100), ("TSLA", 20), ("FITB", 60), ("MSFT", 8), ("GM", 50)],  # the last two are fine and so is a third below: MSFT x8, GM x50 (and FITB x52)
    ("F", "MSFT x8"): [("F", 1), ("TSLA", 1), ("FITB", 5), ("GM", 4)],                 # short $188.77: FITB x5 or GM x4 completes it
    ("F", "FITB x52"): [("MSFT", 3), ("GM", 22)],                                       # short $1,208.77: MSFT x3 or GM x22 completes it
    # PURCHASE 2: buy 40 FITB, $1,763.77 to raise (the cash is what purchase 1 left).
    ("FITB", ROOT): [("FITB", 10), ("F", 300), ("TSLA", 8), ("F", 150), ("GM", 33)],   # FITB is bought; 300 F is more than the 260 sellable (300 were bought a minute ago); TSLA is red; F x150, GM x33 are fine
}
# Purchase 1's root also needs FITB x52 as a valid thought; the table keeps the five-thought limit, so it replaces the rule-breaking FITB x60 with it:
TABLE[("F", ROOT)] = [("F", 100), ("TSLA", 20), ("MSFT", 8), ("GM", 50), ("FITB", 52)]


def agent(table=None, prices=None):
    toolbox = QuoteToolbox(prices)
    model = ScriptedByPlan(TABLE if table is None else table)
    clerk = DataClerk(toolbox=toolbox, model=object())
    manager = am.AccountManager.classic(model=model, clerk=clerk)
    manager.toolbox, manager.scripted = toolbox, model
    return manager


@pytest.fixture
def start():
    return pf.load_portfolio(SCENARIO)


def run(start, purchases=(("F", 300), ("FITB", 40)), table=None, **options):
    manager = agent(table)
    result = manager.raise_funds_many(start, list(purchases), today=TODAY, **{**LIMITS, "only_green": True, **options})
    return manager, result


def final_of(result):
    """The run's final portfolio as the typed doors read it (the JSON of `final` holds floats once `json` has read it)."""
    return pf.Portfolio.model_validate(result["final"])


def sold(purchase):
    return [(x["symbol"], x["shares"]) for x in purchase["plan"]["sells"]]


def rejected(purchase):
    return {c["thought"]["symbol"] + str(c["thought"]["shares"]): c["reason"] for level in purchase["trace"] for c in level["children"] if c["status"] == "rejected"}


# ---------------------------------------------------------------- the scenario file

def test_the_made_up_portfolio_has_the_five_holdings_and_the_free_cash(start):
    assert [(p.symbol, p.shares, p.cost_basis) for p in start.positions] == [("MSFT", 50, Decimal("150")), ("TSLA", 123, Decimal("300")), ("GM", 60, Decimal("10")), ("F", 260, Decimal("2")), ("FITB", 52, Decimal("15"))]
    assert start.cash_avail == Decimal("51.23")


# ---------------------------------------------------------------- purchase 1: buy 300 F, selling only what is in profit and never F

def test_the_first_purchase_sells_other_symbols_in_profit_and_never_the_stock_it_buys(start):
    _, result = run(start)
    first = result["purchases"][0]
    assert first["status"] == "plan" and first["target"] == "F" and first["cost"] == Decimal("3600.00") and first["need"] == Decimal("3548.77") and first["only_green"] is True
    assert sold(first) == [("MSFT", 8), ("FITB", 5)] and first["plan"]["score"] == 66.5   # 8 x 420 + 5 x 45 = 3,585; 3,585 + 51.23 - 3,600 = 36.23 left
    assert first["plan"]["proceeds"] == Decimal("3585.00") and first["plan"]["left_over"] == Decimal("36.23") and first["after"]["cash"] == Decimal("36.23")
    assert all(symbol != "F" for symbol, _ in sold(first)) and all(symbol != "TSLA" for symbol, _ in sold(first))


def test_selling_the_stock_being_bought_and_selling_at_a_loss_are_rejected_with_their_reasons(start):
    _, result = run(start)
    reasons = rejected(result["purchases"][0])
    assert "cannot be sold to pay for itself" in reasons["F100"]                     # the user's rule: sell OTHER symbols to buy this one
    assert "TSLA is not in profit (price $250.00 is not above its basis $300.00)" in reasons["TSLA20"]
    assert "F1" in reasons and "TSLA1" in reasons                                      # the same two mistakes at level 2, again rejected


def test_the_same_thought_without_the_green_rule_is_allowed_and_sells_at_a_loss(start):
    manager = agent({("F", ROOT): [("TSLA", 15)]})   # 15 x 250 = 3,750: covers the 3,548.77, at a loss of 15 x 50
    result = manager.raise_funds_many(start, [("F", 300)], today=TODAY, only_green=False)
    first = result["purchases"][0]
    assert first["status"] == "plan" and sold(first) == [("TSLA", 15)] and first["plan"]["gain"] == Decimal("-750.00")


def test_the_beam_prunes_the_weakest_partial_plan_and_the_trace_says_why(start):
    _, result = run(start)
    level1 = {c["plan"]: c for c in result["purchases"][0]["trace"][0]["children"] if "plan" in c}
    assert {p: c["status"] for p, c in level1.items()} == {"MSFT x8": "kept", "GM x50": "pruned", "FITB x52": "kept"}  # scores 65.7, 56.4 and 57.8: the beam is 2
    assert "below the beam" in level1["GM x50"]["reason"] and level1["MSFT x8"]["score"] > level1["FITB x52"]["score"] > level1["GM x50"]["score"]


def test_the_plans_that_completed_and_lost_are_listed_with_their_scores(start):
    _, result = run(start)
    first = result["purchases"][0]
    others = {tuple(x["symbol"] for x in a["sells"]): a["score"] for a in first["alternatives"]}
    assert others == {("MSFT", "GM"): 66.1, ("FITB", "MSFT"): 60.8, ("FITB", "GM"): 58.8}  # keep counts 4 holdings (F, being bought, is not one): selling FITB out costs 1 of 4
    assert first["stats"]["completed_plans"] == 4 and first["stats"]["levels"] == 2


# ---------------------------------------------------------------- purchase 2: buy 40 FITB, against what purchase 1 left

def test_the_second_purchase_is_planned_against_the_portfolio_the_first_one_left(start):
    manager, result = run(start)
    second = result["purchases"][1]
    prompt = manager.scripted.seen[[i for i, t in enumerate(manager.scripted.seen) if "buy 40 share(s) of FITB" in t][0]]
    assert "Free settled cash (spent first): $36.23. Money to raise: $1,763.77." in prompt     # the cash purchase 1 left, not the 51.23 of the file
    assert "- MSFT (Microsoft Corp): held 42, sellable 42" in prompt and "- FITB" not in prompt  # MSFT was sold down by 8; FITB is the stock being bought: not offered at all
    assert "- F (Ford Motor Co): held 560, sellable 260" in prompt                              # 260 old shares + 300 bought a minute ago, which are locked (T+1)
    assert second["status"] == "plan" and second["cost"] == Decimal("1800.00") and second["need"] == Decimal("1763.77")


def test_the_shares_just_bought_cannot_be_sold_to_pay_for_the_next_purchase(start):
    _, result = run(start)
    second = result["purchases"][1]
    assert "F: 300 shares asked but only 260 are sellable (560 held)" in rejected(second)["F300"]
    assert "cannot be sold to pay for itself" in rejected(second)["FITB10"] and "not in profit" in rejected(second)["TSLA8"]


def test_the_second_purchase_sells_older_f_shares_that_are_in_profit_against_the_weighted_cost_basis(start):
    _, result = run(start)
    second = result["purchases"][1]
    assert sold(second) == [("F", 150)] and second["plan"]["proceeds"] == Decimal("1800.00") and second["plan"]["score"] == 84.0
    f = second["plan"]["sells"][0]
    assert f["basis"] == Decimal("7.3571") and f["gain"] == pf.trade_amount(150, Decimal("12.00")) - pf.trade_amount(150, Decimal("7.3571"))  # (260 x 2 + 300 x 12) / 560: a profit, so F is green again
    assert {tuple(x["symbol"] for x in a["sells"]): a["score"] for a in second["alternatives"]} == {("GM",): 66.6}


# ---------------------------------------------------------------- the whole run

def test_the_final_portfolio_the_cash_and_the_conservation_of_money(start):
    _, result = run(start)
    end = final_of(result)
    final = {p.symbol: (p.shares, p.cost_basis) for p in end.positions}
    assert final == {"MSFT": (42, Decimal("150")), "TSLA": (123, Decimal("300")), "GM": (60, Decimal("10")), "F": (410, Decimal("7.3571")), "FITB": (87, final["FITB"][1])}  # FITB: 52 - 5 + 40
    assert end.cash_avail == Decimal("36.23")
    first, second = result["purchases"]
    assert Decimal("51.23") + first["plan"]["proceeds"] - first["cost"] + second["plan"]["proceeds"] - second["cost"] == end.cash_avail  # every dollar accounted for, exactly
    assert final["FITB"][1] == ((47 * Decimal("15") + 40 * Decimal("45")) / 87).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)  # the new FITB shares at 45 join the 47 older ones at 15


def test_the_prices_are_asked_of_the_data_clerk_once_for_both_purchases(start):
    manager, result = run(start)
    assert manager.toolbox.asked == ["F", "FITB", "MSFT", "TSLA", "GM"]  # the two targets, then the other holdings, each once: not once per purchase
    assert result["prices_order"]["action_type"] == "fetch_quote" and result["prices_order"]["status"] == "data_ready" and set(result["quotes"]) == set(PRICES)


def test_the_portfolio_given_is_not_changed_by_the_run(start):
    before = start.model_dump_json()
    run(start)
    assert start.model_dump_json() == before


def test_a_purchase_that_cannot_be_paid_for_changes_nothing_and_the_next_one_is_still_attempted(start):
    table = {("F", ROOT): [("MSFT", 50), ("GM", 60), ("FITB", 52)], ("FITB", ROOT): [("GM", 9)]}  # 100,000 F can never be paid for; 10 FITB (450) can
    _, result = run(start, purchases=(("F", 100000), ("FITB", 10)), table=table)
    first, second = result["purchases"]
    assert first["status"] == "infeasible" and first["plan"] is None and first["max_affordable"] < 100000 and first["after"] == {}
    assert second["status"] == "plan" and sold(second) == [("GM", 9)] and second["need"] == Decimal("398.77")
    final = {p["symbol"]: p["shares"] for p in result["final"]["positions"]}
    assert final == {"MSFT": 50, "TSLA": 123, "GM": 51, "F": 260, "FITB": 62}  # only the second purchase happened: F is untouched
    assert final_of(result).cash_avail == Decimal("51.23") + Decimal("495.00") - Decimal("450.00")


def test_a_purchase_the_free_cash_alone_pays_for_sells_nothing_and_spends_the_cash(start):
    _, result = run(start, purchases=(("F", 4),))  # 4 x 12 = 48 <= 51.23
    only = result["purchases"][0]
    assert only["status"] == "funded" and only["plan"] is None and only["after"]["cash"] == Decimal("3.23")
    assert final_of(result).cash_avail == Decimal("3.23") and {p["symbol"]: p["shares"] for p in result["final"]["positions"]}["F"] == 264


def test_a_missing_quote_stops_everything_before_any_model_call(start):
    manager = agent(prices={k: v for k, v in PRICES.items() if k != "FITB"})
    result = manager.raise_funds_many(start, [("F", 300), ("FITB", 40)], today=TODAY, only_green=True)
    assert result["status"] == "no_prices" and any(e.startswith("FITB: no quote") for e in result["errors"]) and manager.scripted.seen == [] and result["purchases"] == []


@pytest.mark.parametrize("purchases", [[], [("F", 0)], [("F", 2), ("../etc", 1)], [("F", True)], [(None, 1)]])
def test_a_bad_purchase_refuses_the_whole_request_before_anything_is_asked(start, purchases):
    manager = agent()
    result = manager.raise_funds_many(start, purchases, today=TODAY)
    assert result["status"] == "refused" and result["errors"] and manager.toolbox.asked == [] and manager.scripted.seen == []


def test_the_report_has_a_section_for_each_purchase_a_summary_and_the_portfolio_before_and_after(start):
    _, result = run(start)
    text = am.report_many(result)
    for expected in ("PURCHASE 1 of 2: buy 300 x F", "PURCHASE 2 of 2: buy 40 x FITB", "only holdings in profit (green) may be sold, and never F itself", "TSLA is not in profit",
                     "cannot be sold to pay for itself", "only 260 are sellable (560 held)", "SUMMARY OF 2 PURCHASES", "MSFT x8, FITB x5", "F x150",
                     "PORTFOLIO, start of the run and end of it"):
        assert expected in text, expected
    assert re.search(r"\| 1 +\| F x300 +\| PLAN +\| MSFT x8, FITB x5 +\|", text) and re.search(r"\| 2 +\| FITB x40 +\| PLAN +\| F x150 +\|", text)
    assert re.search(r"\| cash +\| +\$51\.23 \| +\$36\.23 \|", text) and re.search(r"\| F +\| +260 \| +410 \| +\$7\.36 \|", text)  # cash and F, start and end (F's basis is the weighted $7.3571)
    assert text.count("QUOTES (from the Data Clerk's get_quote tool)") == 1  # the quotes once, at the top


def test_one_purchase_on_the_command_line_means_the_old_report_and_several_mean_the_summary():
    assert am.parse_purchase("F", 7) == ("F", 7) and am.parse_purchase("FITB:40", 7) == ("FITB", 40) and am.parse_purchase(" f : 3", 1) == ("f", 3)
    with pytest.raises(ValueError, match="whole number"):
        am.parse_purchase("F:two", 1)
