"""The account book holds a fraction of a share (design choices entry 110; design/fractional_shares_spec.md sections 7 and 9).

The REAL local Postgres on THROWAWAY accounts that the tests create and delete (the three sample accounts are read before and after every test and must not move), as `test_trader_handshake.py`, whose `Book` and fixtures
these tests use. The tests that book a fraction are skipped, with the reason, when the database still has the integer share columns (`AccountStore.holds_fractions()`: run `bash scripts/setup_postgres.sh`); the test of
the refusal runs against whatever the database is. The prices are the stored closes of Friday 2026-11-13 through the real Data Clerk, the order is the real session's (`bob_jones_oil.json`).
"""
from datetime import date, timedelta
from decimal import Decimal

import pytest
from handshake_helpers import FRIDAY, MOMENT, Recorder, close_of, friday_close, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from test_fractional_path import fractional_order
from test_trader_handshake import YEAR, db, store, by_id, last  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager.handshake import check_booked
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.portfolio import trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")


@pytest.fixture
def fractional_db(db):
    if not db.store.holds_fractions():
        pytest.skip("the account book still has integer share columns: run `bash scripts/setup_postgres.sh` (design choices entry 110)")
    return db


def test_the_book_says_it_holds_fractions_exactly_when_its_share_columns_are_numeric(store):
    def column(cur):
        cur.execute("SELECT data_type FROM information_schema.columns WHERE table_schema = 'trading' AND table_name = 'positions' AND column_name = 'shares'")
        return cur.fetchone()["data_type"]

    assert (store._run(None, column) == "numeric") == store.holds_fractions() == store.holds_fractions()


def test_a_fractional_buy_is_booked_and_read_back_exactly_part_is_sold_and_the_rest_sold_leaves_no_position(fractional_db):
    db = fractional_db
    account_id, price, day = db.account("10000.00"), close_of("XOM"), FRIDAY
    n = sh.buys(Decimal("5000.00"), price)
    buy = db.store.book_trade(account_id, "XOM", "buy", n, price, day, record_id="frac-buy")
    assert type(n) is Decimal and not sh.whole(n) and buy.shares == buy.held_after == n and buy.held_before == 0 and buy.cash_before - buy.cash_after == trade_amount(n, price) == Decimal("5000.00")
    row = db.store.read_booked(account_id, buy.trade_row)
    assert (row["symbol"], row["side"], row["shares"], row["record_id"]) == ("XOM", "buy", sh.as_decimal(n), "frac-buy")
    held = db.store.list_portfolio(account_id)
    assert [(p.symbol, p.shares, p.recent_shares, type(p.shares).__name__) for p in held.positions] == [("XOM", n, n, "Decimal")] and held.cash_avail == Decimal("5000.00")
    before = held.model_copy(update={"cash_avail": Decimal("10000.00"), "positions": []})
    assert check_booked(before, held, row, "XOM", "buy", n) == [] and len(check_booked(held, held, row, "XOM", "buy", n)) == 2          # confirmed with the row as the book returns it and the fill's quantity, both Decimals; the same read twice is not
    part = Decimal("0.222222")
    sold = db.store.book_trade(account_id, "XOM", "sell", part, price, day + timedelta(days=3), record_id="frac-part")
    assert sold.held_before == n and sold.held_after == sh.minus(n, part) and db.held(account_id) == {"XOM": sh.minus(n, part)} and sold.gain == trade_amount(part, price) - trade_amount(part, buy.price)
    rest = db.store.book_trade(account_id, "XOM", "sell", sh.minus(n, part), price, day + timedelta(days=3), record_id="frac-rest")
    assert rest.held_after == 0 and db.held(account_id) == {} and db.store.list_portfolio(account_id).positions == []                 # exactly 0 is left: the position row is gone
    assert db.cash(account_id) == Decimal("10000.00") - trade_amount(n, price) + trade_amount(part, price) + trade_amount(sh.minus(n, part), price) and db.store.number_trades(account_id, *YEAR) == 3
    with pytest.raises(ValueError, match="cannot sell"):
        db.store.book_trade(account_id, "XOM", "sell", Decimal("0.000001"), price, day, record_id="frac-none")
    for bad in (0, Decimal("0.1234567"), True, "2", Decimal("-1.5")):
        with pytest.raises(ValueError, match="number above zero"):
            db.store.book_trade(account_id, "XOM", "buy", bad, price, day)


def test_a_second_fractional_buy_adds_exactly_averages_the_cost_and_locks_only_the_new_lot(fractional_db):
    db = fractional_db
    account_id, first_price, second_price = db.account("5000.00"), Decimal("500.0000"), Decimal("520.5000")
    db.store.book_trade(account_id, "MSFT", "buy", Decimal("4.5"), first_price, date(2026, 11, 10))
    two = db.store.book_trade(account_id, "MSFT", "buy", Decimal("0.333333"), second_price, FRIDAY)
    held = db.store.list_portfolio(account_id).position("MSFT")
    basis = ((Decimal("4.5") * first_price + Decimal("0.333333") * second_price) / Decimal("4.833333")).quantize(Decimal("0.0001"))
    assert (held.shares, held.recent_shares, held.cost_basis, held.acquired) == (Decimal("4.833333"), Decimal("0.333333"), basis, FRIDAY) and two.held_before == Decimal("4.5") and two.held_after == Decimal("4.833333")
    assert sh.plus(Decimal("4.5"), Decimal("0.333333")) == Decimal("4.833333") and db.cash(account_id) == Decimal("5000.00") - Decimal("2250.00") - trade_amount(Decimal("0.333333"), second_price)


def test_the_trader_books_two_fractional_purchases_and_the_book_shows_them_exactly(fractional_db, monkeypatch):
    db = fractional_db
    account_id, order = db.account("15000.00"), fractional_order()
    am = db.manager(account_id)
    rec = Recorder(monkeypatch, am)
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    t = by_id(order)
    wanted = {d["symbol"]: d["shares"] for d in order.trade_decisions if d["will_trade"] in wo.TRADING}
    assert [(x["status"], x["booked"]) for x in t.values()] == [("filled", True)] * 2 and {x["symbol"]: x["shares"] for x in t.values()} == wanted and wo.verify_trades(order) == []
    assert db.held(account_id) == wanted and all(type(n) is Decimal and not sh.whole(n) for n in db.held(account_id).values()) and db.cash(account_id) == Decimal("15000.00") - sum((x["fill"]["amount"] for x in t.values()), Decimal("0.00"))
    assert [a.status for k, _, a in rec.answers if k == "record"] == ["booked", "booked"] and all(a.verified for k, _, a in rec.answers if k == "record") and last(order)["booked"] is True and order.errors == []
    assert last(order)["portfolio_after"]["positions"] == [{"symbol": p.symbol, "shares": p.shares, "cost_basis": p.cost_basis, "acquired": p.acquired.isoformat()} for p in db.store.list_portfolio(account_id).positions]


def test_a_fractional_decision_is_saved_and_counted(fractional_db):
    db = fractional_db
    account_id = db.account("1000.00")
    decision = {"record_id": "frac-decision", "symbol": "XOM", "action": "open", "shares": Decimal("31.581606"), "mode": "auto", "will_trade": "yes_auto", "reason": "an even split", "decided_utc": "2026-11-13T20:30:00Z"}
    db.store.save_decisions(account_id, [decision, {**decision, "record_id": "none", "shares": 0, "will_trade": "no_funds"}])
    assert db.store.number_approved_recommendations(account_id, *YEAR) == 1 and db.store.number_denied_recommendations(account_id, *YEAR) == 1


def test_a_fraction_on_a_book_that_does_not_hold_fractions_is_refused_before_anything_is_written_and_a_whole_trade_still_goes_through(db, monkeypatch):
    """Run against whatever the database is: on a migrated one the book is made to answer as the old one did (`holds_fractions` False: the one real method, answered the other way), on an old one it is the real thing."""
    monkeypatch.setattr(db.store, "holds_fractions", lambda: False)
    account_id, price = db.account("1000.00"), close_of("XOM")
    day = FRIDAY
    for call in (lambda: db.store.book_trade(account_id, "XOM", "buy", Decimal("0.5"), price, day), lambda: db.store.record_trade(account_id, "XOM", "buy", Decimal("0.5"), price, day),
                 lambda: db.store.save_decisions(account_id, [{"record_id": "f", "symbol": "XOM", "action": "open", "shares": Decimal("0.5"), "mode": "auto", "will_trade": "yes_auto", "reason": "x", "decided_utc": "2026-11-13T20:30:00Z"}])):
        with pytest.raises(ValueError, match=r"holds whole shares only.*bash scripts/setup_postgres\.sh"):
            call()
    assert db.cash(account_id) == Decimal("1000.00") and db.held(account_id) == {} and db.store.number_trades(account_id, *YEAR) == 0 and db.store.number_approved_recommendations(account_id, *YEAR) == 0
    whole = db.store.book_trade(account_id, "XOM", "buy", 2, price, day)
    assert (whole.shares, type(whole.shares).__name__, str(whole.shares), whole.held_after) == (2, "Decimal", "2", 2) and db.held(account_id) == {"XOM": 2}
