"""The account book and the WAITING ORDERS (design choices entry 140; design/limit_orders_spec.md W3 and W4), on the real local Postgres with THROWAWAY accounts (deleted at the end; the three samples are checked not to move).

What is tested: an order SETS ASIDE its cash or its shares and nothing else can use them (the one door, `available_cash`, `available_shares`, `list_portfolio`); a placement that does not fit is refused by the book itself; the book refuses a purchase,
a sale or a withdrawal that would use what is set aside; a fill books the trade and marks the order filled in ONE transaction and can never book twice; an expiry, a cancel and a rejection release the hold; and two placements racing for the same
cash place only what fits.
"""
import threading
from datetime import timedelta
from decimal import Decimal
from datetime import date

import pytest
from open_orders_helpers import db, eastern, store, MONDAY  # noqa: F401  (db and store are fixtures)

NOW = eastern(MONDAY, 10, 30)
LATER = eastern(MONDAY, 16, 0)


def place(store, account_id, n, symbol="MU", side="buy", limit="90", shares=None, dollars=None, company=None):
    """One waiting order on the account (the record id says which of the test's orders it is)."""
    return store.place_open_order(account_id, request_id="req1", record_id=f"direct-req1-{n}", symbol=symbol, side=side, limit_price=Decimal(limit), request=f"{side} {symbol} at {limit}", placed_at=NOW, expires_at=LATER,
                                  shares=None if dollars is not None else Decimal(shares if shares is not None else 5), dollars=None if dollars is None else Decimal(dollars), company=company)


def test_a_purchase_sets_aside_its_most_possible_cost_and_nothing_else_can_spend_it(db):
    account = db.account("1000.00")
    order = place(db.store, account, 1, shares=5, limit="90")
    assert (order.status, order.reserved_cash, order.reserved_shares, order.shares, order.dollars) == ("open", Decimal("450.00"), Decimal(0), Decimal(5), None)
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.cash_total) == (Decimal("550.00"), Decimal("450.00"), Decimal("1000.00"))
    assert db.store.available_cash(account) == Decimal("550.00") and db.store.account_balance(account) == Decimal("1000.00")
    with pytest.raises(ValueError, match=r"\$550.00 can be removed|only \$550.00"):
        db.store.remove_funds(account, Decimal("600.00"))
    assert db.store.remove_funds(account, Decimal("550.00")) == Decimal("450.00")      # exactly what is available can go; what is set aside stays
    assert db.store.available_cash(account) == Decimal("0.00")


def test_an_order_in_dollars_sets_aside_its_dollars_and_a_sale_sets_aside_its_shares(db):
    account = db.account("2000.00")
    db.buy(account, "MSFT", 3, "100.00")      # $300: the account now holds 3 MSFT and $1,700
    dollars = place(db.store, account, 1, dollars="500.00", limit="90")
    sale = place(db.store, account, 2, symbol="MSFT", side="sell", limit="150", shares=2)
    assert (dollars.reserved_cash, dollars.shares, dollars.dollars) == (Decimal("500.00"), None, Decimal("500.00"))
    assert (sale.reserved_cash, sale.reserved_shares) == (Decimal("0.00"), Decimal(2))
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.reserved_shares) == (Decimal("1200.00"), Decimal("500.00"), {"MSFT": Decimal(2)})
    assert held.free_shares("MSFT") == Decimal(1) and held.sellable_shares("MSFT", date(2030, 1, 1)) == Decimal(1) and db.store.available_shares(account, "MSFT") == Decimal(1)
    assert held.position("MSFT").shares == Decimal(3)      # the whole holding is still shown: only the sale sees what is free


def test_the_book_refuses_a_placement_the_available_cash_or_shares_do_not_cover_and_writes_nothing(db):
    account = db.account("1000.00")
    db.buy(account, "MSFT", 4, "100.00")      # 4 MSFT, $600 of cash
    place(db.store, account, 1, shares=5, limit="100")      # holds $500
    with pytest.raises(ValueError, match=r"\$100.00 is available.*set aside \$150.00"):
        place(db.store, account, 2, shares=3, limit="50")
    place(db.store, account, 3, symbol="MSFT", side="sell", limit="200", shares=3)
    with pytest.raises(ValueError, match=r"holds 4 of MSFT, 3 of them set aside.*1 can be sold.*set aside 2"):
        place(db.store, account, 4, symbol="MSFT", side="sell", limit="200", shares=2)
    assert [o.record_id for o in db.store.open_orders(account)] == ["direct-req1-1", "direct-req1-3"]
    with pytest.raises(ValueError, match="shares or dollars"):
        db.store.place_open_order(account, request_id="r", record_id="x", symbol="MU", side="buy", limit_price=Decimal(1), request="x", placed_at=NOW, expires_at=LATER)
    with pytest.raises(ValueError, match="expire after"):
        db.store.place_open_order(account, request_id="r", record_id="y", symbol="MU", side="buy", limit_price=Decimal(1), request="x", placed_at=NOW, expires_at=NOW, shares=Decimal(1))


def test_the_book_will_not_book_a_purchase_a_sale_or_a_withdrawal_that_uses_what_an_order_set_aside(db):
    account = db.account("1000.00")
    db.buy(account, "MSFT", 4, "100.00")      # $600 of cash
    place(db.store, account, 1, shares=5, limit="100")      # $500 set aside: $100 available
    place(db.store, account, 2, symbol="MSFT", side="sell", limit="200", shares=3)      # 3 of 4 MSFT set aside
    with pytest.raises(ValueError, match=r"less \$500.00 set aside for waiting orders, does not cover \$150.00"):
        db.store.book_trade(account, "AAPL", "buy", Decimal(3), Decimal("50.00"), MONDAY)
    assert db.store.book_trade(account, "AAPL", "buy", Decimal(2), Decimal("50.00"), MONDAY).cash_after == Decimal("500.00")      # $100 was available: exactly that
    with pytest.raises(ValueError, match=r"holds 4 and 3 of them are set aside"):
        db.store.book_trade(account, "MSFT", "sell", Decimal(2), Decimal("150.00"), MONDAY)
    assert db.store.book_trade(account, "MSFT", "sell", Decimal(1), Decimal("150.00"), MONDAY).held_after == Decimal(3)
    assert db.store.available_cash(account) == Decimal("150.00") and db.store.available_shares(account, "MSFT") == Decimal(0)


def test_a_fill_books_the_trade_and_marks_the_order_filled_in_one_transaction_and_never_books_twice(db):
    account = db.account("1000.00")
    order = place(db.store, account, 1, shares=5, limit="90")
    other = place(db.store, account, 2, symbol="XOM", shares=2, limit="100")      # a second waiting order: its $200 stays set aside
    booked = db.store.book_trade(account, "MU", "buy", Decimal(5), Decimal("89.50"), MONDAY, record_id="direct-req1-1", order_id=order.order_id)
    assert (booked.order_id, booked.cash_before, booked.cash_after) == (order.order_id, Decimal("1000.00"), Decimal("552.50"))
    filled = db.store.get_open_order(account, order.order_id)
    assert (filled.status, filled.trade_id) == ("filled", booked.trade_row) and filled.closed_at is not None
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.position("MU").shares) == (Decimal("352.50"), Decimal("200.00"), Decimal(5))      # the first order's hold is gone, the other's stays
    with pytest.raises(ValueError, match="is filled, not open"):
        db.store.book_trade(account, "MU", "buy", Decimal(5), Decimal("89.50"), MONDAY, order_id=order.order_id)
    assert db.store.number_trades(account, date(2026, 1, 1), date(2026, 12, 31)) == 1      # one trade in the book: the buy of MU (a deposit is not a trade, and the other order has not filled)
    assert other.status == "open" and db.store.get_open_order(account, other.order_id).status == "open"


def test_a_fill_may_not_cost_more_than_was_set_aside_nor_be_for_another_stock_or_side_nor_for_another_account(db):
    account, stranger = db.account("1000.00"), db.account("1000.00", "Another Holder")
    order = place(db.store, account, 1, shares=5, limit="90")      # sets aside $450
    with pytest.raises(ValueError, match="never costs more than was set aside"):
        db.store.book_trade(account, "MU", "buy", Decimal(5), Decimal("90.01"), MONDAY, order_id=order.order_id)
    with pytest.raises(ValueError, match="is to buy MU, not to buy XOM"):
        db.store.book_trade(account, "XOM", "buy", Decimal(1), Decimal("10.00"), MONDAY, order_id=order.order_id)
    with pytest.raises(ValueError, match="no open order"):
        db.store.book_trade(stranger, "MU", "buy", Decimal(5), Decimal("89.00"), MONDAY, order_id=order.order_id)
    assert db.store.available_cash(account) == Decimal("550.00") and db.store.available_cash(stranger) == Decimal("1000.00")      # nothing moved


def test_cancel_expire_and_reject_each_release_the_hold_in_one_transaction_and_only_for_an_order_that_is_open(db):
    account = db.account("1000.00")
    db.buy(account, "MSFT", 4, "100.00")
    a, b, c = (place(db.store, account, 1, shares=1, limit="10"), place(db.store, account, 2, shares=2, limit="10"), place(db.store, account, 3, symbol="MSFT", side="sell", limit="200", shares=2))
    assert db.store.available_cash(account) == Decimal("570.00") and db.store.available_shares(account, "MSFT") == Decimal(2)
    cancelled = db.store.cancel_open_order(account, a.order_id)
    assert (cancelled.status, db.store.available_cash(account)) == ("cancelled", Decimal("580.00"))
    with pytest.raises(ValueError, match="is cancelled: it cannot be cancelled"):
        db.store.cancel_open_order(account, a.order_id)
    assert db.store.expire_open_orders(account, NOW) == []      # nothing is over yet
    expired = db.store.expire_open_orders(account, LATER + timedelta(seconds=1))
    assert sorted(o.order_id for o in expired) == [b.order_id, c.order_id] and all(o.status == "expired" for o in expired)
    assert (db.store.available_cash(account), db.store.available_shares(account, "MSFT"), db.store.open_orders(account)) == (Decimal("600.00"), Decimal(4), [])
    d = place(db.store, account, 4, shares=1, limit="10")
    assert db.store.reject_open_order(account, d.order_id, "the booking failed").status == "rejected" and db.store.reject_open_order(account, d.order_id, "again") is None
    assert db.store.available_cash(account) == Decimal("600.00")
    assert {o.status for o in db.store.open_orders(account, ("open", "filled", "expired", "cancelled", "rejected"))} == {"cancelled", "expired", "rejected"}


def test_two_placements_racing_for_the_same_cash_place_only_what_fits(db):
    """Two threads each place an order that holds $600 of $1,000 at the same moment: the account row lock makes one wait for the other, and the second finds the cash gone."""
    account = db.account("1000.00")
    gate, results = threading.Barrier(2), []

    def attempt(n):
        gate.wait()
        try:
            results.append(("placed", place(db.store, account, n, shares=6, limit="100").order_id))
        except ValueError as error:
            results.append(("refused", str(error)))

    threads = [threading.Thread(target=attempt, args=(n,)) for n in (1, 2)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    assert sorted(kind for kind, _ in results) == ["placed", "refused"], results
    assert next(text for kind, text in results if kind == "refused").startswith("the account has $1,000.00 in cash, $600.00 of it set aside for other waiting orders")
    assert (db.store.available_cash(account), len(db.store.open_orders(account))) == (Decimal("400.00"), 1)


def test_a_fill_and_a_cancel_racing_for_one_order_leave_exactly_one_outcome_and_a_consistent_book(db):
    account = db.account("1000.00")
    order = place(db.store, account, 1, shares=5, limit="90")
    gate, results = threading.Barrier(2), {}

    def fill():
        gate.wait()
        try:
            results["fill"] = db.store.book_trade(account, "MU", "buy", Decimal(5), Decimal("89.00"), MONDAY, order_id=order.order_id).trade_row
        except ValueError as error:
            results["fill"] = str(error)

    def cancel():
        gate.wait()
        try:
            results["cancel"] = db.store.cancel_open_order(account, order.order_id).status
        except ValueError as error:
            results["cancel"] = str(error)

    threads = [threading.Thread(target=fill), threading.Thread(target=cancel)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    final = db.store.get_open_order(account, order.order_id)
    held = db.store.list_portfolio(account)
    if final.status == "filled":
        assert isinstance(results["fill"], int) and "cannot be cancelled" in results["cancel"] and (held.cash_avail, held.position("MU").shares) == (Decimal("555.00"), Decimal(5))
    else:
        assert final.status == "cancelled" and results["cancel"] == "cancelled" and "cannot be filled" in results["fill"] and (held.cash_avail, held.position("MU")) == (Decimal("1000.00"), None)
    assert held.reserved_cash == Decimal("0.00")      # either way the hold is gone


def test_only_one_watcher_holds_the_advisory_lock_for_the_database(store):
    first, second = store.watcher_lock(), store.watcher_lock()
    try:
        assert first.try_acquire() is True and first.try_acquire() is True      # again: it still holds it
        assert second.try_acquire() is False and store.watcher_running() is True
    finally:
        first.release()
        second.release()
    assert store.watcher_running() is False
    third = store.watcher_lock()
    try:
        assert third.try_acquire() is True      # released: another may take it
    finally:
        third.release()


def test_a_book_with_no_waiting_orders_reads_exactly_as_it_did(db):
    account = db.account("750.00")
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.reserved_shares, held.cash_total) == (Decimal("750.00"), Decimal("0.00"), {}, Decimal("750.00"))
    assert "reserved" not in held.model_dump_json()      # the reservations are not part of any dump of a portfolio
    assert db.store.open_orders(account) == [] and db.store.accounts_with_open_orders().count(account) == 0
