"""THE REAL LIVE SESSION OF JACOB'S REQUEST (design choices entry 142): "Buy 2 shares of ANT and raise funds if needed", Ask mode, on a THROWAWAY account that is the same as his account 6553 (its cash $60.14 and its twelve holdings), 2026-11-21,
with the real models and the real Finnhub quotes (`session.py`, a scripted user who answers "y" to the plan RaiseFunds showed; no fetch). The saved work order (`tests/fixtures/session_real/ant_raise_funds_ask.json`, written by `--save`) is what the
desk made: the Account Manager read the words, the cash did not cover 2 x $88.91, RaiseFunds took a profit (one share of DPZ, bought at $341.72, sold at $348.61) and the Trader sold it first and then bought.
"""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from prometheus.agents.account_manager.direct import direct_section
from prometheus.agents.trader.report import trade_section
from prometheus.common import work_order as wo
from prometheus.common.money import digits

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "ant_raise_funds_ask.json").read_text(encoding="utf-8"))
ORDER = wo.WorkOrder.model_validate(CAPTURE["order"])


def test_the_real_session_went_account_manager_direct_decision_funds_node_trader_and_the_words_were_read_as_permission():
    assert CAPTURE["path"] == ["account_manager", "account_manager_direct", "account_manager_funds", "trader"] and ORDER.status == "traded" and ORDER.errors == []
    assert CAPTURE["intent"]["raise_funds"] is True and CAPTURE["intent"]["orders"][0]["symbol"] == "ANT" and [o["symbol"] for o in ORDER.direct_orders] == ["ANT"] and ORDER.recommendations == [] and ORDER.reviews == []


def test_the_decision_is_the_users_own_order_as_typed_with_its_funding_and_it_was_permitted_to_trade():
    (d,) = ORDER.trade_decisions
    assert (d["will_trade"], d["origin"], d["shares"], d["mode"], digits(d["price"])) == ("yes_user", "direct", 2, "ask", Decimal("88.91")) and d["funding"]["by"] == "words" and d["funding"]["plan_id"] == f"{ORDER.request_id}-funds"
    assert digits(d["funding"]["shortfall"]) == Decimal("177.82") - Decimal("60.14") == Decimal("117.68")                              # what the cash lacked: 2 x $88.91 less $60.14
    assert "the cash $60.14 does not pay for 2 shares of ANT at $88.91 ($177.82)" in d["reason"] and "raises the money by selling holdings first" in d["reason"]
    assert wo.verify_trade(ORDER, d["record_id"], datetime(2026, 11, 21, 4, 50, tzinfo=timezone.utc)) == []


def test_the_sale_is_a_funding_sale_for_the_purchase_it_paid_for_at_a_profit_and_the_cash_never_went_below_what_it_was_and_adds_up_to_the_cent():
    buy, sale = ORDER.trades
    assert (buy["trade_id"], buy["side"], buy["status"], buy["shares"], digits(buy["fill"]["price"]), digits(buy["fill"]["amount"]), buy["booked"]) == ("T1", "buy", "filled", 2, Decimal("88.91"), Decimal("177.82"), True)
    assert (sale["trade_id"], sale["kind"], sale["funds_for"], sale["symbol"], sale["side"], sale["shares"], digits(sale["fill"]["price"]), digits(sale["fill"]["amount"]), sale["placed_by"]) == ("T1-F1", "funding", "T1", "DPZ", "sell", 1, Decimal("348.61"), Decimal("348.61"), "code")
    assert digits(sale["fill"]["amount"]) - Decimal("341.72") == Decimal("6.89")                                                       # a gain of $6.89 on a share bought at $341.72
    run = ORDER.trader_runs[-1]
    assert digits(run["portfolio_before"]["cash"]) == Decimal("60.14") and digits(run["portfolio_after"]["cash"]) == Decimal("60.14") + Decimal("348.61") - Decimal("177.82") == Decimal("230.93")
    assert run["funding"]["status"] == "plan" and run["funding"]["approved"] is True and wo.verify_trades(ORDER) == []
    after = {p["symbol"]: digits(p["shares"]) for p in run["portfolio_after"]["positions"]}
    assert after["ANT"] == 2 and after["DPZ"] == Decimal("3.389558") and "ANT" not in {p["symbol"] for p in run["portfolio_before"]["positions"]}


def test_the_reports_say_it_in_words():
    lines = "\n".join([*direct_section(ORDER), *trade_section(ORDER)])
    assert "buy 2 shares of Anthropic Pbc-Cl A (ANT) at $88.91 (about $177.82)   APPROVED" in lines and "T1-F1  sell 1 of 1 DPZ" in lines and "(funds T1)" in lines and "funding  RaiseFunds said plan" in lines and "cash $60.14 -> $230.93" in lines
