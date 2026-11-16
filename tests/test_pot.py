"""A RUN NEVER SPENDS MORE THAN ITS MONEY (design choices entry 111, M1; design/dollar_orders_spec.md section 3).

Jacob (2026-11-15), on a real session that sized $30,000.00 and filled $30,015.00: *"It is impossible to spend more than the amount in the portfolio and budget. ... It is OK to have unused money left over but never is
allowed to spend more than is available."* The POT of a run is the budget the user gave, else the cash, never more than the cash; the Account Manager writes it on every decision it sized from money (`size.pot`), the handshake
never approves a purchase past what is left of it, and a program fails a work order whose fills pass it.

REAL DATA ONLY, as `handshake_helpers.py`: the real session's order (`bob_jones_oil.json`: XOM and EOG decided by the real Account Manager over a budget of $2,750, so each carries `size.pot` 2750.0), the stored closes of
Friday 2026-11-13 through the real Data Clerk, the real Trader with no model. The only change to a real decision is its number of shares, made too big for the pot on purpose, and a pot made too small for the fills.
"""
import copy
from decimal import Decimal

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, Recorder, close_of, friday_close, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_handshake import tickets_of

from prometheus.agents.account_manager.handshake import decide_approval
from prometheus.common import work_order as wo
from prometheus.common.portfolio import trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")


def run(order, cash=Decimal("15000.00")):
    am = manager()
    am.perform_trades(order, BOB.model_copy(update={"cash_avail": cash}), trader(), MOMENT)
    return am, {t["trade_id"]: t for t in order.trades}


def test_every_purchase_the_account_manager_sized_carries_the_pot_of_the_request():
    order = real_order()
    pots = {d["symbol"]: d["size"]["pot"] for d in order.trade_decisions if d["size"]}
    assert set(pots) >= {"XOM", "EOG"} and set(pots.values()) == {Decimal("2750.00")}                                      # the budget of the request, which is less than the $15,000 of cash


def test_the_handshake_cuts_a_purchase_that_would_pass_the_pot_while_the_cash_would_cover_it_and_refuses_the_next_when_the_money_is_used_up():
    order = real_order()
    xom = next(d for d in order.trade_decisions if d["symbol"] == "XOM")
    xom["shares"] = 20                                                                                         # $3,411.80 at the close: more than the pot of $2,750.00, far less than the cash of $15,000
    am, t = run(order)
    price = t["T1"]["fill"]["price"]
    assert t["T1"]["status"] == "filled" and t["T1"]["shares"] == Decimal("2750.00") // price < 20 and "the money of this request" in t["T1"]["reason"] and "cut from 20 to" in t["T1"]["reason"]
    assert (t["T2"]["status"], t["T2"]["reject_code"]) == ("rejected", "insufficient_cash") and "the money of this request is used up" in t["T2"]["reason"] and "is not the limit" in t["T2"]["reason"]
    assert sum((x["fill"]["amount"] for x in order.trades if x["status"] == "filled"), Decimal("0.00")) <= Decimal("2750.00") and wo.verify_trades(order) == []
    assert order.trader_runs[-1]["portfolio_after"]["cash"] > Decimal("11000.00")                                          # the cash was never the limit: money is left over, which is fine


def test_a_run_that_is_resumed_counts_what_was_filled_before_from_the_work_order_and_not_from_memory():
    order = real_order()
    run(order)
    spent = sum((t["fill"]["amount"] for t in order.trades if t["status"] == "filled"), Decimal("0.00"))
    assert 0 < spent <= Decimal("2750.00")
    fresh = manager()                                                                                          # a new Account Manager: it remembers nothing of the first run
    assert fresh.spent_of_pot(order, Decimal("2750.00")) == spent and fresh.spent_of_pot(order, Decimal("1000.00")) == 0     # another pot is another request
    left = decide_approval(BOB, tickets_of(real_order())["XOM"], close_of("XOM"), FRIDAY, spendable=max(Decimal("0.00"), Decimal("2750.00") - spent), pot=Decimal("2750.00"))
    assert left.approved is (Decimal("2750.00") - spent >= close_of("XOM")) and (left.approved or "the money of this request is used up" in left.reason)


def test_a_program_fails_a_work_order_whose_fills_pass_the_pot_or_whose_purchases_carry_two_pots():
    order = real_order()
    run(order)
    assert wo.verify_trades(order) == []
    spent = sum((t["fill"]["amount"] for t in order.trades if t["status"] == "filled"), Decimal("0.00"))
    smaller = copy.deepcopy(order)
    for d in smaller.trade_decisions:
        if d["size"]:
            d["size"]["pot"] = Decimal("2000.00")
    reasons = wo.verify_trades(smaller)
    assert reasons == [f"the fills of the purchases cost ${spent:,.2f}, more than the money of the request, $2,000.00 (the pot): a run never spends more than its money"]
    mixed = copy.deepcopy(order)
    next(d for d in mixed.trade_decisions if d["size"])["size"]["pot"] = Decimal("9999.00")
    assert wo.verify_trades(mixed) == ["the purchases of this work order carry different pots: $2,750.00, $9,999.00: a run has one"]
    unsized = copy.deepcopy(order)
    for d in unsized.trade_decisions:
        d["size"] = None
    assert wo.verify_trades(unsized) == []                                                                     # a decision that carries no pot has the cash for its limit, as before


def test_without_a_pot_the_cash_is_the_limit_exactly_as_before_and_a_direct_order_with_typed_shares_is_never_cut():
    order = real_order()
    for d in order.trade_decisions:
        d["size"] = None
    xom = next(d for d in order.trade_decisions if d["symbol"] == "XOM")
    xom["shares"] = 20
    ticket = {**tickets_of(order)["XOM"], "shares": 20}
    am, t = run(order)
    assert t["T1"]["shares"] == 20 and t["T1"]["status"] == "filled" and wo.verify_trades(order) == []
    answer = decide_approval(BOB, ticket, close_of("XOM"), FRIDAY)
    assert answer.approved and answer.shares == 20 and trade_amount(20, close_of("XOM")) > Decimal("2750.00")
