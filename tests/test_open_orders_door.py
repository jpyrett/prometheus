"""EVERY reader of what may be spent or sold reads what is AVAILABLE (design choices entry 140; design/limit_orders_spec.md W3), run on the real parts: a real account on the real local Postgres (throwaway) with a waiting order that holds most of its
cash, then each reader in turn: the pot of the recommendation path (`decide_trades`, on the real review capture of a real session), the sizing of a direct order, RaiseFunds' holdings, the handshake's approval, the withdrawal and the portfolio shown to the user.
(The scan of the source for a reader that bypasses the door is in `test_open_orders_plan.py`.)
"""
import copy
import json
from decimal import Decimal

import pytest
from handshake_helpers import CAPTURE, MOMENT
from open_orders_helpers import MONDAY, clerk_with_directory, db, eastern, stored_close, store  # noqa: F401  (db, store and stored_close are fixtures)

from prometheus.agents.account_manager import intent as it
from prometheus.common import work_order as wo
from prometheus.reasoning import raise_funds as rf

NOW = eastern(MONDAY, 10, 30)


def hold(db, account_id, cash: str, n: int = 1):
    """A waiting purchase that sets aside `cash` dollars (a ceiling of $100 a share, so the order is `cash / 100` shares): the rest of the account's cash is what is available."""
    return db.store.place_open_order(account_id, request_id="door", record_id=f"direct-door-{n}", symbol="XOM", side="buy", limit_price=Decimal("100"), request="Buy shares of XOM at $100 or less", placed_at=NOW,
                                     expires_at=eastern(MONDAY, 16, 0), shares=Decimal(cash) / 100)


def reviewed_order() -> wo.WorkOrder:
    """The REAL session's order of `tests/fixtures/session_real/bob_jones_oil.json` (2026-11-14), before the Account Manager decided: the real recommendations and the Critic's real reviews."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(json.loads(CAPTURE.read_text(encoding="utf-8"))["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def test_the_pot_of_a_recommendation_is_the_available_cash_not_the_whole_balance(db, stored_close):
    account = db.account("10000.00", "Bob Jones", "auto")
    hold(db, account, "9000")
    am = db.manager(account)
    portfolio = am.list_portfolio()
    assert (portfolio.cash_avail, portfolio.cash_total) == (Decimal("1000.00"), Decimal("10000.00"))
    order = am.decide_trades(reviewed_order(), portfolio, None, MOMENT, "auto")
    trading = [d for d in order.trade_decisions if d["will_trade"] in wo.TRADING]
    assert trading and all(wo.decision_pot(d) == Decimal("1000.00") for d in trading)      # the money of the request is the $1,000 that is available, never the $10,000 balance
    assert sum((d["est_cost"] for d in trading), Decimal("0.00")) <= Decimal("1000.00")


def test_a_direct_order_with_no_number_spends_the_available_cash_only_and_one_that_the_available_cash_cannot_pay_for_is_not_placed(db, stored_close):
    account = db.account("10000.00", "Door Holder", "auto")
    hold(db, account, "9000")
    am = db.manager(account, clerk=clerk_with_directory())
    order = wo.new_work_order("Buy MSFT", requester="desk", action_type="trade")
    assert am.name_the_stocks(order, [it.DirectOrder(side="buy", symbol="MSFT")], "Buy MSFT", NOW) == []
    decided = am.decide_direct(order, am.list_portfolio(), NOW, "auto")
    decision = decided.trade_decisions[0]
    assert decision["will_trade"] == "yes_user" and wo.decision_pot(decision) == Decimal("1000.00") and decision["est_cost"] <= Decimal("1000.00")
    typed = wo.new_work_order("Buy 15 shares of MSFT", requester="desk", action_type="trade")
    assert am.name_the_stocks(typed, [it.DirectOrder(side="buy", symbol="MSFT", shares=15)], "Buy 15 shares of MSFT", NOW) == []
    refused = am.decide_direct(typed, am.list_portfolio(), NOW, "auto").trade_decisions[0]      # 15 x 517.99 is $7,769.85: the account HAS it, but $9,000 of it is set aside
    assert refused["will_trade"] == "no_funds" and "cash $1,000.00 does not pay for 15 shares" in refused["reason"]


def test_raise_funds_cannot_plan_to_sell_the_shares_a_waiting_sale_holds(db, stored_close):
    account = db.account("10000.00", "Door Holder")
    db.buy(account, "MSFT", 10, "400.00")
    db.store.place_open_order(account, request_id="door", record_id="direct-door-9", symbol="MSFT", side="sell", limit_price=Decimal("600"), request="Sell 8 shares of MSFT at $600 or more", placed_at=NOW, expires_at=eastern(MONDAY, 16, 0), shares=Decimal(8))
    portfolio = db.manager(account).list_portfolio()
    problem = rf.build_problem(portfolio, "XOM", 1, {"XOM": Decimal("166.87"), "MSFT": Decimal("517.99")}, MONDAY)
    assert problem.holdings["MSFT"].sellable == Decimal(2) and problem.holdings["MSFT"].shares == Decimal(10)      # 10 held, 8 set aside: 2 may be planned for sale
    assert problem.cash == portfolio.cash_avail


def test_the_handshake_approves_what_is_available_to_a_trade_and_the_orders_own_hold_to_its_own_fill_only(db, stored_close):
    from open_orders_helpers import trader
    account = db.account("10000.00", "Door Holder", "auto")
    hold(db, account, "9000", n=1)
    waiting = db.store.open_orders(account)[0]
    am = db.manager(account, clerk=clerk_with_directory())
    # another order of the user's, one share of MSFT, is paid from the cash that is available and from nothing else
    order = wo.new_work_order("Buy 1 share of MSFT", requester="desk", action_type="trade")
    assert am.name_the_stocks(order, [it.DirectOrder(side="buy", symbol="MSFT", shares=1)], "Buy 1 share of MSFT", NOW) == []
    decided = am.decide_direct(order, am.list_portfolio(), NOW, "auto")
    assert decided.trade_decisions[0]["will_trade"] == "yes_user"
    am.perform_trades(decided, None, trader(), NOW, book=True)
    assert decided.trades[0]["status"] == "filled" and db.store.available_cash(account) < Decimal("500.00")      # $1,000 less one share of MSFT at its stored close (517.99): the $9,000 was never touched
    assert db.store.get_open_order(account, waiting.order_id).status == "open" and db.store.list_portfolio(account).reserved_cash == Decimal("9000.00")


def test_a_withdrawal_and_the_screen_see_the_available_cash(db):
    account = db.account("10000.00", "Door Holder")
    hold(db, account, "9000")
    am = db.manager(account)
    with pytest.raises(ValueError, match=r"\$9,000.00 of it is set aside for waiting orders: only \$1,000.00 can be removed"):
        am.remove_funds(Decimal("1000.01"))
    assert am.available_cash() == Decimal("1000.00") and am.account_balance() == Decimal("10000.00")
    lines = []
    am.screen = lines.append
    am.show_portfolio()
    assert lines[:4] == ["      the account as it stands now:", "        cash 10,000.00", "        set aside for waiting orders 9,000.00", "        available 1,000.00"]
