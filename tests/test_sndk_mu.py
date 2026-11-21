"""Buy 3 SNDK, then as many MU as the cash left affords, by selling several cheaper stocks: a tree that goes more than one level deep (design choices entry 44).

Run:  pytest tests/test_sndk_mu.py -q

The portfolio is made up (configs/portfolios/jacob_pyrett_sndk_mu.json): PG 15 @ 60, MCD 120 @ 40, GOOGL 30 @ 100, LLY 25 @ 155 and $200 of free cash. The PRICES are the real Finnhub quotes of the 2026-11-03
close, recorded from the live run and used here as fixed numbers (no call is made). The "LLM" is scripted with the thoughts the real gpt-oss-120b gave in the third live run (in the order it gave them),
so the whole tree of that run can be replayed: 4 model calls, 3 levels, 7 completed plans, and 13 nodes as it ran (entry 45 then made the search drop a plan that is the same sells in another order:
PG x5, GOOGL x4 and LLY x1 reached twice, so 12 nodes and 6 completed plans now). The Data Clerk's quote tool, the checks, the scores, the pruning and the report are the real code.
No network, no key, no cost.

The numbers: 3 SNDK cost $3,379.56, $3,179.56 to raise after the free cash. Whole shares, lean sells (no step sells more than the shortfall needs).
"""
from decimal import Decimal
import json
import re

import pytest
from langchain_core.messages import AIMessage

from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.reasoning import raise_funds as rf

SCENARIO = am.DEFAULT_PORTFOLIO.parent / "jacob_pyrett_sndk_mu.json"
PRICES = {"SNDK": Decimal("1126.52"), "MU": Decimal("855.10"), "PG": Decimal("162.06"), "MCD": Decimal("263.04"), "GOOGL": Decimal("280.03"), "LLY": Decimal("1409.38")}  # Finnhub /quote, the 2026-11-03 close (21:00 UTC)


class QuoteToolbox:
    def __init__(self, prices=None):
        self.prices, self.asked = dict(PRICES if prices is None else prices), []

    def get_quote(self, symbol):
        self.asked.append(symbol)
        if symbol not in self.prices:
            raise RuntimeError(f"No quote returned for {symbol}")
        p = float(self.prices[symbol])      # the Data Clerk's quote is what Finnhub's JSON gives, a float: the Account Manager turns it into a Decimal price itself
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


# ---------------------------------------------------------------- the generator: both kinds of step, and asked again when one is missing


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
