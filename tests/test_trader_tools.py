"""The Trader's four tools (prometheus/tools/trader.py, Trader spec section 8): each refusal of each tool, the budget, sells before buys, and nothing after the notify. The tools are called the way the loop's tools node calls
them: through the budget, with the model's arguments, and a refusal comes back as the message the model reads. No network, no model."""
from decimal import Decimal
import json

import pytest
from trader_helpers import FakeNotifier, account, acct_now, decided, make_broker, row, synthetic

from prometheus.tools.trader import Budget, TraderTools, budgeted, call_tool


class Rig:
    """The tools over a broker, behind a budget. `rig("place_order", trade_id="T1")` returns (outcome, what the model reads): outcome is ok, invalid, limit or error, as the loop's log says it."""

    def __init__(self, order, portfolio=None, quotes=None, **kw):
        self.order, self.kw = order, kw
        self.broker, self.found = make_broker(order, portfolio or account(5000.0), quotes=quotes if quotes is not None else {"PG": 100.0, "MSFT": 400.0}, **kw)
        self.notifier = FakeNotifier(self.broker)
        self.tools = TraderTools(self.broker, self.notifier)
        self.budget = Budget([t.trade_id for t in self.found if t.kind == "recommendation"])
        self.by_name = {t.name: t for t in (budgeted(x, self.budget) for x in self.tools.langchain_tools())}

    def __call__(self, name, **args):
        outcome, text = call_tool(self.by_name, name, args)
        return outcome, (json.loads(text) if outcome == "ok" else text)


def rig_for(*rows, **kw):
    return Rig(synthetic(*rows), **kw)


# ---------------------------------------------------------------- place_order


@pytest.mark.parametrize("args, message", [
    ({"trade_id": "T9"}, "there is no trade 'T9' on this work order"),
    ({"trade_id": "T1", "order_type": "stop"}, "order_type must be 'market' or 'limit'"),
    ({"trade_id": "T1", "order_type": "limit", "limit_price": 105.0}, "limit_price 105.0000 is outside the range allowed for T1, 99.00 to 101.00"),
    ({"trade_id": "T1", "order_type": "limit", "limit_price": 90.0}, "99.00 to 101.00"),
    ({"trade_id": "T1", "order_type": "market", "limit_price": 100.0}, "a market order has no limit_price"),
    ({"trade_id": "T1", "limit_price": 100.0}, 'give an order_type of "limit"'),
    ({"trade_id": "T1", "limit_price": "abc"}, "limit_price: Input should be a valid number"),
    ({"trade_id": 5}, "trade_id: Input should be a valid string"),
    ({}, "trade_id: Field required"),
])
def test_each_refusal_of_place_order_comes_back_as_a_message_that_says_how_to_fix_it_and_places_nothing(args, message):
    rig = rig_for(row("PG", "open", 10, 100.0))
    outcome, text = rig("place_order", **args)
    assert outcome == "invalid" and text.startswith("Invalid arguments: ") and message in text and text.endswith("Fix the arguments and call the tool again.")
    assert (rig.found[0].status, rig.broker.placed, acct_now(rig.broker).cash_avail) == ("new", [], Decimal("5000.0"))


def test_a_refused_placing_may_be_corrected_but_only_a_few_times():
    rig = rig_for(row("PG", "open", 10, 100.0), row("MSFT", "reduce", 2, 400.0), portfolio=account(5000.0, MSFT=5))
    assert [rig("place_order", trade_id="T1")[0] for _ in range(3)] == ["invalid", "invalid", "invalid"]
    outcome, text = rig("place_order", trade_id="T1")
    assert outcome == "limit" and "was refused 3 times: leave it, the program places it" in text
    rig("place_order", trade_id="T2")
    assert rig.broker.place("T1", placed_by="code").status == "filled"                                                    # the program (the sweep) still places it


def test_nothing_can_be_placed_after_the_run_was_closed():
    rig = rig_for(row("PG", "open", 10, 100.0), row("KO", "open", 1, 60.0), quotes={"PG": 100.0, "KO": 60.0})
    rig("place_order", trade_id="T1")
    rig.notifier.done, rig.broker.closed = True, True
    outcome, text = rig("place_order", trade_id="T2")
    assert outcome == "invalid" and "the run was closed (notify_account_manager was called): nothing more can be placed" in text and rig.found[1].status == "new"


# ---------------------------------------------------------------- check_trade_status


def test_check_trade_status_gives_the_fill_of_a_filled_trade_and_refuses_an_unknown_id():
    rig = rig_for(row("PG", "open", 10, 100.0))
    rig("place_order", trade_id="T1", order_type="market")
    assert rig("check_trade_status", trade_id="T1")[1] == {"trades": [{"trade_id": "T1", "status": "filled", "fill_price": Decimal("100.0"), "fill_shares": 10}]}      # AMENDED (entry 111): 100.05 was the quote plus 5 bps
    outcome, text = rig("check_trade_status", trade_id="T7")
    assert outcome == "invalid" and "there is no trade 'T7'" in text


# ---------------------------------------------------------------- the messages


def test_a_tool_that_raises_something_unexpected_comes_back_as_a_failure_message_not_an_exception():
    rig = rig_for(row("PG", "open", 10, 100.0))
    rig.broker.place = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))                                            # a bug in the code behind a tool
    outcome, text = rig("place_order", trade_id="T1")
    assert outcome == "error" and text.startswith("The tool failed (KeyError)") and "Do not repeat the same call" in text
