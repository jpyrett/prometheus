"""A complex, made-up example: two purchases in a row, F and then FITB, paid for by selling what is in profit (design choices entry 43).

Run:  pytest tests/test_two_purchases.py -q

The portfolio is made up (configs/portfolios/jacob_pyrett_f_fitb.json): Jacob's MSFT 50 @ 150, TSLA 123 @ 300 and GM 60 @ 10, plus F 260 @ 2 and FITB 52 @ 15, and 51.23 of free cash. The prices are made up too
(a fake toolbox behind the REAL Data Clerk tool and work order). The "LLM" is scripted by what it is asked: for each purchase and each plan so far it answers with the thoughts the test wants, among them
thoughts that break the rules (sell the stock being bought, sell a holding at a loss, sell shares bought a minute ago). The Data Clerk's quote tool, the tree search, the hard checks, the scores and the
report are the real code. No network, no key, no cost.

The made-up prices: MSFT 420 (basis 150, in profit), TSLA 250 (basis 300, AT A LOSS), GM 55 (basis 10), F 12 (basis 2), FITB 45 (basis 15).
Purchase 1: 300 F = $3,600 (to raise $3,548.77 after the free cash).  Purchase 2: 40 FITB = $1,800.
"""
from decimal import Decimal
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


def sold(purchase):
    return [(x["symbol"], x["shares"]) for x in purchase["plan"]["sells"]]


def rejected(purchase):
    return {c["thought"]["symbol"] + str(c["thought"]["shares"]): c["reason"] for level in purchase["trace"] for c in level["children"] if c["status"] == "rejected"}


# ---------------------------------------------------------------- the whole run


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


