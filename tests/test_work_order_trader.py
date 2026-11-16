"""The work order's Trader v1 additions (spec section 11): the status `traded`, `trades` and `trader_runs`, `TRADE_STATUSES`, the `fetch_bars` action and `verify_bar_request`, `trade_of`, `trades_of`, `verify_trades`.

(Kept apart from tests/test_work_order.py, which builds real stores for the Data Clerk: none of this needs one. No network, no model.)
"""
import copy

import pytest

from prometheus.common import work_order as wo


def bars_order(symbols=("PG",), start="2026-11-05", end="2026-11-30", **changes):
    order = wo.new_work_order("bars", requester="account_manager", action_type="fetch_bars", subject=list(symbols), bars={"start": start, "end": end})
    for key, value in changes.items():
        setattr(order, key, value)
    return order


# ---------------------------------------------------------------- the status, the new fields and the new action


def test_the_trade_statuses_and_the_trader_s_record_say_the_same_thing():
    from prometheus.agents.trader import record

    assert wo.TRADE_STATUSES == ("new", "active", "filled", "rejected")
    assert (record.STATUSES, record.KINDS, record.SIDES) == (wo.TRADE_STATUSES, wo.TRADE_KINDS, wo.TRADE_SIDES) == (wo.TRADE_STATUSES, ("recommendation", "funding"), ("buy", "sell"))
    for name, allowed in (("status", wo.TRADE_STATUSES), ("kind", wo.TRADE_KINDS), ("side", wo.TRADE_SIDES)):
        assert set(record.Trade.model_fields[name].annotation.__args__) == set(allowed), name


def test_data_bars_travel_with_the_data_result():
    result = wo.DataResult(bars={"PG": [{"date": "2026-11-05", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}], "XYZ": []})
    assert wo.DataResult.model_validate_json(result.model_dump_json()).bars == result.bars


# ---------------------------------------------------------------- verify_bar_request


@pytest.mark.parametrize("changes, message", [
    ({"action_type": "fetch_quote"}, "is not 'fetch_bars'"),
    ({"subject": None}, "needs one or more ticker symbols"),
    ({"subject": []}, "needs one or more ticker symbols"),
    ({"subject": ["PG", ""]}, "needs one or more ticker symbols"),
    ({"subject": ["PG", 7]}, "needs one or more ticker symbols"),
    ({"subject": [f"S{i}" for i in range(31)]}, "at most 30 symbols"),
    ({"bars": None}, 'needs `bars` with exactly "start" and "end"'),
    ({"bars": "2026-11-05"}, 'needs `bars` with exactly "start" and "end"'),
    ({"bars": {"start": "2026-11-05"}}, 'needs `bars` with exactly "start" and "end"'),
    ({"bars": {"start": "2026-11-05", "end": "2026-11-30", "step": 1}}, 'needs `bars` with exactly "start" and "end"'),
    ({"bars": {"start": "2026-11-05", "finish": "2026-11-30"}}, 'needs `bars` with exactly "start" and "end"'),
    ({"bars": {"start": "2026-11-31", "end": "2026-12-05"}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": "20261105", "end": "2026-12-05"}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": "2026-W45-4", "end": "2026-12-05"}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": "2026-11-05T10:00", "end": "2026-12-05"}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": "2026-11-05", "end": None}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": 20261105, "end": "2026-12-05"}}, "each be a date written YYYY-MM-DD"),
    ({"bars": {"start": "2026-11-10", "end": "2026-11-09"}}, "is after end"),
    ({"bars": {"start": "2025-09-30", "end": "2026-11-05"}}, "at most 400 days"),
])
def test_the_bar_requests_the_data_clerk_may_not_answer(changes, message):
    reasons = wo.verify_bar_request(bars_order(**changes))
    assert any(message in r for r in reasons), reasons


# ---------------------------------------------------------------- trade_of and trades_of

def trade(trade_id, **changes):
    base = {"trade_id": trade_id, "kind": "recommendation", "record_id": "r1", "review_id": "rv1", "funds_for": None, "symbol": "PG", "side": "buy", "shares_approved": 44, "shares": 44,
            "status": "filled", "reject_code": None, "mode": "paper", "fill": {"price": 167.1, "shares": 44, "amount": 7352.4, "commission": 0.0, "date": "2026-11-05", "price_source": "quote", "slippage_bps": 5.0}}
    return {**base, **changes}


def good_order():
    """Two decisions: r1 (buy 44 PG, yes_user), r2 (denied). Trades: T1 filled for r1, T1-F1 the sale of MSFT that paid for it, T2 rejected because r2 is not permitted. Every test below changes ONE thing."""
    order = wo.new_work_order("x", subject=["PG"])
    order.status = "decided"
    order.trade_decisions = [{"record_id": "r1", "symbol": "PG", "action": "add", "shares": 44, "will_trade": "yes_user"},
                             {"record_id": "r2", "symbol": "XYZ", "action": "add", "shares": 10, "will_trade": "no_system_denied"}]
    order.trades = [trade("T1"), trade("T1-F1", kind="funding", funds_for="T1", symbol="MSFT", side="sell", shares_approved=5, shares=5,
                                       fill={"price": 420.0, "shares": 5, "amount": 2100.0, "commission": 0.0, "date": "2026-11-05", "price_source": "quote", "slippage_bps": 5.0}),
                    trade("T2", record_id="r2", symbol="XYZ", shares_approved=10, shares=10, status="rejected", reject_code="not_permitted", fill=None)]
    return order


def test_a_trade_is_found_by_its_id_and_a_recommendations_trades_by_its_record_id():
    order = good_order()
    assert wo.trade_of(order, "T1-F1")["symbol"] == "MSFT" and wo.trade_of(order, "T9") is None
    assert [t["trade_id"] for t in wo.trades_of(order, "r1")] == ["T1", "T1-F1"] and [t["trade_id"] for t in wo.trades_of(order, "r2")] == ["T2"] and wo.trades_of(order, "r9") == []
    order.trades.append("not a record")
    assert wo.trade_of(order, "T1")["trade_id"] == "T1" and len(wo.trades_of(order, "r1")) == 2             # a damaged entry is skipped, not an exception


# ---------------------------------------------------------------- verify_trades: the output guard

def changed(edit):
    order = good_order()
    edit(order)
    return order


@pytest.mark.parametrize("edit, message", [
    (lambda o: o.trades[0].update(record_id="r2", symbol="XYZ"), "recommendation r2 has the decision 'no_system_denied'"),             # a filled trade of a denied recommendation
    (lambda o: o.trades[0].update(record_id="nobody"), "recommendation nobody has no trade decision"),
    (lambda o: o.trades[0].update(record_id=None), "recommendation None has no trade decision"),
    (lambda o: o.trade_decisions[0].update(will_trade="pending_user"), "has the decision 'pending_user'"),
    (lambda o: o.trade_decisions[0].update(will_trade="no_user_denied"), "has the decision 'no_user_denied'"),
    (lambda o: o.trades[0]["fill"].update(shares=45), "filled 45 shares but the decision gave 44"),
    (lambda o: o.trade_decisions[0].update(shares=40), "filled 44 shares but the decision gave 40"),
    (lambda o: o.trades[0].update(symbol="MSFT"), "is for 'MSFT' but the decision is for 'PG'"),
    (lambda o: o.trades[0].update(side="sell"), "is a sell but the decision's action is 'add'"),
    (lambda o: o.trade_decisions[0].update(action="reduce"), "is a buy but the decision's action is 'reduce'"),
    (lambda o: o.trades.append(trade("T3")), "second filled trade for recommendation r1 (the first is T1)"),
    (lambda o: o.trades[0].update(fill=None), "T1 is filled but has no fill"),
    (lambda o: o.trades[0].pop("fill"), "T1 is filled but has no fill"),
    # AMENDED (entry 110): these seven rows said "whole number of shares", the reason now says "a number of shares above zero"; the row with shares=2.5 became 0.1234567 (2.5 is a valid fill now, seven decimals is refused)
    (lambda o: o.trades[0]["fill"].update(price=0), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(price=-1.0), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(price=None), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(price=float("nan")), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(price=True), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(shares=0), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[0]["fill"].update(shares=0.1234567), "the fill needs a price and a number of shares above zero"),
    (lambda o: o.trades[1].update(funds_for="T9"), "T1-F1 funds 'T9', which is not a purchase on this order"),
    (lambda o: o.trades[1].update(funds_for=None), "T1-F1 funds None, which is not a purchase on this order"),
    (lambda o: o.trades[1].update(funds_for="T1-F1"), "T1-F1 funds 'T1-F1'"),                                                           # a sale cannot fund itself
    (lambda o: o.trades[1].update(side="buy"), "T1-F1 is a funding trade but is not a sale"),
    (lambda o: o.trades[0].update(side="sell"), "T1-F1 funds 'T1', which is not a purchase on this order"),                              # the purchase it pays for is a sale now
    (lambda o: o.trades[0].update(status="rejected", reject_code="insufficient_cash", fill=None), "T1-F1 is a filled funding sale but the purchase it pays for, T1, is rejected, not filled"),   # holdings sold, nothing bought
    (lambda o: o.trades[0].update(status="active", fill=None), "T1-F1 is a filled funding sale but the purchase it pays for, T1, is active, not filled"),
    (lambda o: o.trades[0].update(status="new", fill=None), "T1-F1 is a filled funding sale but the purchase it pays for, T1, is new, not filled"),
    (lambda o: o.trades[0].update(status="done"), "status 'done' is not one of new, active, filled, rejected"),
    (lambda o: o.trades[0].pop("status"), "status None is not one of"),
    (lambda o: o.trades[2].update(reject_code=None), "T2 is rejected but has no reject_code"),
    (lambda o: o.trades[2].update(reject_code=""), "T2 is rejected but has no reject_code"),
    (lambda o: o.trades[2].pop("reject_code"), "T2 is rejected but has no reject_code"),
    (lambda o: o.trades.append(trade("T2", status="active", fill=None)), "T2: the trade_id appears more than once"),
    (lambda o: o.trades[0].update(trade_id=""), "has no trade_id"),
    (lambda o: o.trades.append("T3"), "trade #4 is not a record"),
])
def test_each_way_the_trades_can_disagree_with_the_decisions_is_a_reason(edit, message):
    reasons = wo.verify_trades(changed(edit))
    assert any(message in r for r in reasons), reasons


