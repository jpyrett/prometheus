"""Named LangSmith spans for the agents' own steps (design choices entry 115; design/langsmith_traceable_spec.md): `prometheus/common/trace.py`, the `traced` decorator.

A decorated function changes nothing about what it does. With no trace open it IS the plain function (nothing is summarized, no run is made, so a decorated step never makes a root trace of its own); inside a node of a
LangGraph it is a child run of the node's. These tests run the real thing: a real one-node `StateGraph`, with tracing ON for the test only and aimed at a dead local address with a sampling rate of 0 (nothing
reaches any network), the real work orders of `tests/fixtures/` and the real handshake of the Account Manager and the Trader over the stored closes of Friday 2026-11-13 (`handshake_helpers`).
"""
import copy
import inspect
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import pytest
from handshake_helpers import BOB, MOMENT, friday_close, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from langgraph.graph import END, START, StateGraph
from langsmith import run_trees, utils
from langsmith.run_helpers import get_current_run_tree

from prometheus.agents.account_manager import handshake
from prometheus.agents.account_manager.sizing import size_purchases
from prometheus.common import langsmith_tracing, trace
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.common.trace import summarize, traced
from prometheus.persistence.accounts import AccountStore

FIXTURES = Path(__file__).parent / "fixtures"
DEAD = "http://127.0.0.1:9"
NAMES = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_TRACING_SAMPLING_RATE", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT")


class Watched(dict):
    """A real dict that notes every time `summarize` reads it (`items` is how it reads one): a way to know that nothing was summarized, with nothing patched."""

    def __init__(self, **values):
        super().__init__(**values)
        self.reads = 0

    def items(self):
        self.reads += 1
        return super().items()


@pytest.fixture
def tracing_on():
    """Tracing ON for this test only (conftest turns it off for every test), aimed at a dead local address, sampling rate 0: runs are made and nested as usual and none is queued or sent. `configure()` is NOT called."""
    saved = {name: os.environ.get(name) for name in NAMES}
    os.environ.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="x", LANGSMITH_ENDPOINT=DEAD, LANGSMITH_TRACING_SAMPLING_RATE="0")
    for name in ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT"):
        os.environ.pop(name, None)

    def forget():
        langsmith_tracing._clear_sdk_cache()
        utils.get_tracer_project.cache_clear()

    forget()
    client = run_trees._CLIENT
    run_trees._CLIENT = None
    try:
        yield
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
        run_trees._CLIENT = client
        forget()


def in_a_node(work):
    """Run `work()` inside the only node of a real LangGraph; return what it returned and the node's own run."""
    seen = {}

    class State(TypedDict):
        value: int

    def node(state):
        seen["node"] = get_current_run_tree()
        seen["result"] = work()
        return {"value": 1}

    graph = StateGraph(State)
    graph.add_node("only", node)
    graph.add_edge(START, "only")
    graph.add_edge("only", END)
    graph.compile().invoke({"value": 0})
    return seen["result"], seen["node"]


def tree(run) -> list[str]:
    return [run.name, *[f"  {line}" for child in run.child_runs for line in tree(child)]]


# ---------------------------------------------------------------- rule R3: nest only, never a root

@traced("test.add")
def add(a, b=2, *, config=None):
    return a + b


@traced("test.fail")
def fail(argument):
    raise ValueError("this is the function's own error")


def test_with_no_trace_open_a_decorated_function_is_the_plain_function_and_nothing_is_summarized():
    watched = Watched(a=1)
    assert get_current_run_tree() is None
    assert add(1, 3) == 4 and add(a=5, config=watched) == 7             # a parameter called `config` is the function's own
    with pytest.raises(ValueError, match="the function's own error"):
        fail(watched)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        add()                                                             # the function's own refusal of a bad call
    assert watched.reads == 0 and get_current_run_tree() is None          # no summary was made, no run was made
    assert inspect.signature(add) == inspect.signature(add.__wrapped__) and add.__name__ == "add"
    assert list(inspect.signature(size_purchases).parameters)[:3] == ["purchases", "cash", "config"]      # the real decorated functions keep their signatures too


def test_inside_a_node_a_decorated_function_is_a_child_of_the_node_and_changes_nothing(tracing_on):
    runs, watched, store = [], Watched(a=1), AccountStore({"host": "db.invalid", "port": "1", "database": "d", "app_user": "u", "app_password": "SECRETWORD"})

    @traced("test.child")
    def child(x, kept, handle, *, config=None):
        runs.append(get_current_run_tree())
        return {"doubled": x * 2, "price": Decimal("19.99")}

    @traced(lambda x: f"test.named_{x}", run_type="tool")
    def named(x):
        return x

    @traced("test.fails")
    def fails():
        raise KeyError("its own error")

    def work():
        with pytest.raises(KeyError, match="its own error"):
            fails()
        return child(21, watched, store, config={"k": 1}), named("by_argument")

    (result, name), node = in_a_node(work)
    assert result == {"doubled": 42, "price": Decimal("19.99")} and name == "by_argument"       # the same value, not a copy of a summary
    assert runs[0].parent_run_id == node.id and runs[0].name == "test.child" and runs[0].run_type == "chain"
    assert tree(node) == ["only", "  test.fails", "  test.child", "  test.named_by_argument"]
    assert node.child_runs[2].run_type == "tool" and node.child_runs[0].error and "its own error" in node.child_runs[0].error
    assert watched.reads > 0 and runs[0].inputs["handle"] == "<AccountStore>" and "SECRETWORD" not in json.dumps(runs[0].inputs, default=str)       # a store is its type name, never its repr
    assert runs[0].outputs == {"doubled": 42, "price": Decimal("19.99")} and runs[0].inputs["config"] == {"k": 1}
    assert get_current_run_tree() is None and child(1, {}, None) == {"doubled": 2, "price": Decimal("19.99")} and runs[-1] is None      # outside the graph: the function runs, and no run was made for it (its own current run is None)


def test_run_type_is_a_chain_or_a_tool_and_never_an_llm():
    with pytest.raises(ValueError, match="chain"):
        traced("test.model", run_type="llm")


# ---------------------------------------------------------------- the real handshake, traced

def test_the_handshake_spans_show_the_approved_shares_and_the_cash_before_and_after(friday_close, tracing_on):
    order, account = real_order(), manager()
    _, node = in_a_node(lambda: account.perform_trades(order, BOB, trader(), MOMENT))
    spans = {}
    pending = [*node.child_runs]
    while pending:
        run = pending.pop(0)
        spans.setdefault(run.name, []).append(run)
        pending += run.child_runs
    assert {"trader.execute_trades", "trader.execute", "account_manager.trading_portfolio", "data_clerk.request_quotes", "account_manager.approve_trade", "account_manager.decide_approval",
            "account_manager.record_fill", "trader.verify_fill", "trader.verify_trades"} <= set(spans)
    assert all(run.run_type != "llm" for runs in spans.values() for run in runs) and spans["data_clerk.request_quotes"][0].run_type == "tool"
    first, second = spans["account_manager.approve_trade"]
    assert (first.outputs["approved"], first.outputs["shares"], first.outputs["cash_avail"]) == (True, Decimal("8"), Decimal("15000.00"))
    assert (second.outputs["shares"], second.outputs["cash_avail"]) == (Decimal("8"), Decimal("13635.28"))             # the cash the first fill left
    booked = spans["account_manager.record_fill"]
    assert [(b.outputs["status"], b.outputs["cash_avail"]) for b in booked] == [("applied", Decimal("13635.28")), ("applied", Decimal("12368.72"))]
    assert first.inputs["trade"]["symbol"] == "XOM" and first.inputs["price"] == Decimal("170.5900") and first.inputs["order"]["counts"]["trade_decisions"] == 3
    assert [t["status"] for t in order.trades] == ["filled", "filled"] and order.status == "traded"                # the traced run is the run


def test_a_refusal_is_summarized_with_its_code_and_its_reason():
    refused = handshake.refusal("insufficient_cash", "the cash does not pay for one share", Decimal("12.50"))
    assert summarize(refused) == {"approved": False, "shares": Decimal("0"), "cash_avail": Decimal("12.50"), "code": "insufficient_cash", "reason": "the cash does not pay for one share"}


# ---------------------------------------------------------------- summarize, on real data

def captured(name: str) -> wo.WorkOrder:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))


def test_a_work_order_is_its_symbols_and_its_counts_never_its_articles():
    order = captured("advisor_real/PG_open.json")                      # the real order the Data Clerk answered: 17 chunks of stored news, press releases and filings, each with its text
    summary = summarize(order)
    assert order.data and len(order.data.sources) == 17 and all(source.get("text") for source in order.data.sources)
    assert summary["id"] == order.request_id and summary["status"] == order.status and summary["symbols"] == ([order.subject] if isinstance(order.subject, str) else list(order.subject))
    assert summary["counts"]["sources"] == 17 and summary["counts"]["recommendations"] == len(order.recommendations)
    text = json.dumps(summary, default=str)
    assert len(text) < 2500 and not [s for s in order.data.sources if s["text"][:60] in text] and order.data.answer[:60] not in text


def test_a_work_order_that_traded_is_counted_and_its_decimals_stay_decimals():
    order = captured("session_real/auto_even_split_decimal.json")
    summary = summarize(order)
    assert summary["counts"] == {"companies": 0, "sources": 0, "quotes": 0, "bars": 0, "recommendations": 3, "reviews": 3, "trade_decisions": len(order.trade_decisions), "direct_orders": 0, "trades": 3, "errors": 0}
    assert summary["symbols"] == list(order.subject) and len(summary["symbols"]) == 15 and summary["action"] == order.action_type
    assert len(json.dumps(summary, default=str)) < 2500 and "thesis" not in json.dumps(summary)
    cash = summarize(Portfolio(user="Bob Jones", cash_avail=Decimal("15000.00")))["cash_avail"]
    assert cash == Decimal("15000.00") and isinstance(cash, Decimal)
    assert summarize(BOB)["cash_avail"] == Decimal("15000.00") and isinstance(summarize(BOB)["cash_avail"], Decimal)


def test_what_is_not_data_is_its_type_name_a_long_list_is_cut_and_nothing_raises():
    store = AccountStore({"host": "db.invalid", "port": "1", "database": "d", "app_user": "u", "app_password": "SECRETWORD"})
    assert repr(store).count("db.invalid") == 1                                       # its repr names the server: that is why it is never used
    assert summarize(store) == "<AccountStore>" and summarize(print) == "<builtin_function_or_method>" and summarize(AccountStore) == "<AccountStore (class)>"
    assert summarize({"model": store, "n": 3}) == {"model": "<AccountStore>", "n": 3}
    long = summarize(list(range(25)))
    assert long[:20] == list(range(20)) and long[20] == {trace.MORE: 5} and len(long) == 21
    assert summarize("x" * 2500).endswith("… [500 more characters]") and len(summarize("x" * 2500)) < 2100
    assert summarize(Decimal("0.10")) == Decimal("0.10") and summarize(None) is None and summarize(True) is True
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}
    assert "<dict>" in json.dumps(summarize(deep))                                     # a record in a record in a record ... stops
