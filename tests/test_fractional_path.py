"""A purchase or a sale with a fraction of a share goes through the path from a decision to the account, and each upper-bound guard still fires for it (design choices entry 110; design/fractional_shares_spec.md sections 6 and 9).

REAL DATA ONLY, as `handshake_helpers.py`: the real session's order (`tests/fixtures/session_real/bob_jones_oil.json`: XOM and EOG, decided by the Account Manager), the stored closes of Friday 2026-11-13 read by the real Data Clerk, the
real Account Manager and the real Trader with no model (the sweep places the orders). The only change to a real record is the NUMBER OF SHARES of a decision: the whole shares the session bought are replaced by the fraction that
$5,000.00 buys at the real close (`shares.buys`), which is what the even split gives. The book (Postgres) is in `test_fractional_book.py`.
"""
from decimal import Decimal
import copy

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, close_of, friday_close, jacob_as_bob, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_handshake import tickets_of
from test_direct_order import direct_order

from prometheus.agents.account_manager.handshake import decide_approval
from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.config import load_config
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.portfolio import affordable_shares, apply_fill, trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")


def fractional_order(dollars: Decimal = Decimal("5000.00")) -> wo.WorkOrder:
    """The real session's order with each decision for the fraction of a share that `dollars` buys at the real close."""
    order = real_order()
    for decision in order.trade_decisions:
        if decision["will_trade"] in wo.TRADING:
            price = close_of(decision["symbol"])
            decision["shares"] = sh.buys(dollars, price)
            decision["est_cost"] = trade_amount(decision["shares"], price)
    pot = dollars * len([d for d in order.trade_decisions if d["will_trade"] in wo.TRADING])
    for decision in order.trade_decisions:
        if decision["will_trade"] in wo.TRADING:
            decision["size"] = {**decision["size"], "pot": pot}      # entry 111, M1: the real decisions were sized over a pot of $2,750; these are $5,000 each, so the pot is the sum of the two
    return order


def run(order, cash: Decimal = Decimal("15000.00")):
    am = manager()
    am.perform_trades(order, BOB.model_copy(update={"cash_avail": cash}), trader(), MOMENT)
    return am, {t["trade_id"]: t for t in order.trades}


# ---------------------------------------------------------------- a decision with a fraction, to the working portfolio

def test_a_real_decision_with_a_fraction_is_ticketed_approved_filled_and_applied_to_the_working_portfolio():
    order = fractional_order()
    wanted = {d["symbol"]: d["shares"] for d in order.trade_decisions if d["will_trade"] in wo.TRADING}
    assert all(type(n) is Decimal and not sh.whole(n) for n in wanted.values()) and all(trade_amount(n, close_of(s)) == Decimal("5000.00") for s, n in wanted.items())
    assert wo.verify_trade(order, order.trade_decisions[0]["record_id"], MOMENT) == []                       # the permission holds for a fraction
    am, t = run(order)
    assert [(x["symbol"], x["status"], x["shares"], x["fill"]["shares"]) for x in t.values()] == [(s, "filled", n, n) for s, n in wanted.items()] and wo.verify_trades(order) == []
    after = order.trader_runs[-1]["portfolio_after"]
    assert {p["symbol"]: p["shares"] for p in after["positions"]} == wanted and after["cash"] == Decimal("15000.00") - sum((x["fill"]["amount"] for x in t.values()), Decimal("0.00"))
    assert {p.symbol: p.shares for p in am.current_portfolio().positions} == wanted and not [s for s in wanted if am.current_portfolio().sellable_shares(s, FRIDAY)]      # all bought today: all of it is locked (T+1)


def test_a_purchase_the_cash_cuts_is_cut_to_a_fractional_quantity_the_cash_pays_for_to_the_cent():
    order = fractional_order()
    am, t = run(order, cash=Decimal("3000.00"))
    price = t["T1"]["fill"]["price"]
    assert t["T1"]["shares"] == sh.buys(Decimal("3000.00"), price) and 0 < t["T1"]["shares"] < t["T1"]["shares_approved"] and type(t["T1"]["shares"]) is Decimal and not sh.whole(t["T1"]["shares"]) and t["T1"]["fill"]["amount"] == Decimal("3000.00")
    assert f"cut from {sh.text(t['T1']['shares_approved'])} to {sh.text(t['T1']['shares'])} shares for the cash" in t["T1"]["reason"]
    assert order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("0.00") and (t["T2"]["status"], t["T2"]["reject_code"]) == ("rejected", "insufficient_cash") and "any of the shares" in t["T2"]["reason"]
    assert wo.verify_trades(order) == []


def test_a_whole_order_is_still_cut_to_whole_shares_and_a_fractional_one_to_a_fraction():
    price = close_of("XOM")
    assert affordable_shares(Decimal("3000.00"), price, 100) == Decimal("3000.00") // price and type(affordable_shares(Decimal("3000.00"), price, 100)) is Decimal and sh.whole(affordable_shares(Decimal("3000.00"), price, 100))
    assert affordable_shares(Decimal("3000.00"), price, Decimal("100.5")) == sh.buys(Decimal("3000.00"), price) and affordable_shares(Decimal("3000.00"), price, Decimal("2.5")) == Decimal("2.5") and affordable_shares(Decimal("3000.00"), price, Decimal("0.5")) == Decimal("0.5")
    assert affordable_shares(Decimal("0.00"), price, Decimal("0.5")) == 0 and affordable_shares(Decimal("3000.00"), price, Decimal("0.1234567")) == 0


# ---------------------------------------------------------------- the three guards that went silent for a value that was not an int

def test_the_three_guards_that_ran_for_an_int_only_each_fire_for_a_fraction():
    order = fractional_order()
    approved = {t.symbol: t.shares_approved for t in tk.build_tickets(order, BOB, MOMENT, load_config(), "paper")}
    assert approved == {d["symbol"]: d["shares"] for d in order.trade_decisions if d["will_trade"] in wo.TRADING} and all(type(n) is Decimal and not sh.whole(n) for n in approved.values())      # 3: the ticket used to turn it into 0 shares approved
    am, t = run(order)
    more = copy.deepcopy(order)
    more.trades[0]["fill"]["shares"] = sh.plus(more.trades[0]["fill"]["shares"], sh.STEP)
    assert any("filled" in r and "shares but the decision gave" in r for r in wo.verify_trades(more))                                                   # 2: a fill of one step more than the decision gave
    one, _ = direct_order(shares=38)
    record = one.direct_orders[0]["record_id"]
    one.direct_orders[0]["shares"], one.trade_decisions[0]["shares"] = Decimal("38.5"), Decimal("38.5")
    assert wo.verify_trade(one, record, MOMENT) == []
    one.trade_decisions[0]["shares"] = Decimal("38.500001")
    assert "the decision gives 38.500001 shares but the user typed 38.5" in "; ".join(wo.verify_trade(one, record, MOMENT))                           # 1: a direct decision above what the user typed
    one.trade_decisions[0]["shares"] = Decimal("0.1234567")
    assert "a number of shares above zero" in "; ".join(wo.verify_trade(one, record, MOMENT))


# ---------------------------------------------------------------- the portfolio: a fraction sold in full closes the position, and a fractional lot is locked for a day (T+1)

def test_a_fractional_holding_sold_in_full_closes_the_position_and_a_fractional_lot_is_locked_until_tomorrow():
    price, next_day = close_of("MSFT"), FRIDAY.replace(day=16)
    held = apply_fill(jacob_as_bob().model_copy(update={"cash_avail": Decimal("1000.00")}), "MSFT", "buy", Decimal("0.222222"), price, FRIDAY)
    msft = held.position("MSFT")
    assert (msft.shares, msft.recent_shares) == (Decimal("50.222222"), Decimal("0.222222")) and held.sellable_shares("MSFT", FRIDAY) == 50 and held.sellable_shares("MSFT", next_day) == Decimal("50.222222")   # only the new lot is locked
    sale = {"symbol": "MSFT", "side": "sell", "shares": Decimal("50.222222"), "shares_approved": Decimal("50.222222")}
    assert (decide_approval(held, sale, price, FRIDAY).shares, decide_approval(held, sale, price, next_day).shares) == (50, Decimal("50.222222"))                  # cut to what is sellable today, then all of it
    closed = apply_fill(held, "MSFT", "sell", Decimal("50.222222"), price, next_day)
    assert closed.position("MSFT") is None and closed.cash_avail == held.cash_avail + trade_amount(Decimal("50.222222"), price)
    new = apply_fill(jacob_as_bob().model_copy(update={"cash_avail": Decimal("1000.00")}), "PG", "buy", Decimal("0.333333"), close_of("PG"), FRIDAY)
    assert new.position("PG").shares == Decimal("0.333333") and new.sellable_shares("PG", FRIDAY) == 0 and new.sellable_shares("PG", next_day) == Decimal("0.333333")
    assert apply_fill(new, "PG", "sell", Decimal("0.333333"), close_of("PG"), next_day).position("PG") is None and apply_fill(new, "PG", "sell", Decimal("0.1"), close_of("PG"), next_day).position("PG").shares == Decimal("0.233333")
    with pytest.raises(ValueError, match="cannot sell"):
        apply_fill(new, "PG", "sell", Decimal("0.333334"), close_of("PG"), next_day)


def test_the_tickets_of_a_fractional_order_are_fractions_and_a_trade_of_nothing_or_of_seven_decimals_is_not_approved():
    tickets = tickets_of(fractional_order())
    assert all(not sh.whole(t["shares"]) for t in tickets.values())      # the tickets are dicts in JSON mode: a JSON number, read through the share door
    for bad in (0, Decimal("0.0"), Decimal("-0.5"), Decimal("0.1234567"), True, "2"):
        assert decide_approval(BOB, {**tickets["XOM"], "shares": bad}, Decimal("100.00"), FRIDAY).approved is False
