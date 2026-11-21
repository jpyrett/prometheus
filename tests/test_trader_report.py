"""The TRADER section of a report (prometheus/agents/trader/report.py, Trader spec section 16): plain text from the records, no model. Every line is checked against a run made over fakes."""

from test_trader import go
from trader_helpers import ScriptedModel, answer, call, decided, say

from prometheus.agents.trader.report import outcome_line, trade_line, trade_section


def section(order):
    return "\n".join(trade_section(order))


def test_a_trades_note_is_shown_under_it_in_the_models_words():
    model = ScriptedModel([call("place_order", trade_id="T1", note="inside the range"), call("notify_account_manager"), say(answer("ok", ("T1", "filled")))])
    order, *_ = go(decided(), {"PG": 167.0}, model=model)
    lines = trade_section(order)
    assert lines[1].startswith("  T1") and lines[2] == "      note: inside the range"


def test_an_outcome_with_no_later_bar_says_so():
    assert outcome_line({"trade_id": "T1", "outcome": {"status": "open"}}) == "  outcome  T1  open: no bar after the entry is stored" and outcome_line({"trade_id": "T1", "outcome": None}) is None


def test_the_line_of_a_trade_that_was_never_placed_says_so():
    assert trade_line({"trade_id": "T1", "side": "buy", "symbol": "PG", "shares": 5, "shares_approved": 5, "status": "new", "order_type": "market"}).strip().startswith("T1     buy  5 of 5 PG") and "NOT PLACED" in trade_line(
        {"trade_id": "T1", "side": "buy", "symbol": "PG", "shares": 5, "shares_approved": 5, "status": "new", "order_type": "market"})


