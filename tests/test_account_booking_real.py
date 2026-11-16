"""The booking of one fill, on the REAL local Postgres (design choices entry 82; design/trader_handshake_spec.md, amendment 7).

Jacob, 2026-11-15: the Trader reports a completed trade; the Account Manager SELECTs the portfolio, updates it in ONE transaction under row-level locks on the user's portfolio, commits, SELECTs again to confirm the
trade is in and the totals are different, and only then lets the Trader go on. A trade that fails is not committed, is shown to the user by the Account Manager and is never retried.

Real Postgres on THROWAWAY accounts that the tests create and delete (the three sample accounts are read before and after and must not move); skipped when Postgres is not reachable. No model, no quote, no
network. The only faults are one real call made to misbehave (a position that is not written, a holding changed by somebody else) to make the in-transaction check run, and a lock another connection really holds.
"""
import threading
import time
from contextlib import suppress
from datetime import date
from decimal import Decimal

import psycopg
import pytest
from test_account_manager_trades import filled

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.persistence import accounts as accounts_module
from prometheus.persistence.accounts import SECRETS, BookingCheckError, open_account_store

DAY = date(2026, 11, 13)
SAMPLES = ("jacob_sample", "avery_chen", "blake_okonkwo")


@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
    except Exception as error:
        pytest.skip(f"Postgres is configured but not reachable ({type(error).__name__})")
    return opened


class Accounts:
    """Throwaway accounts for one test; deleted at the end, and the samples are checked."""

    def __init__(self, store):
        self.store, self.made = store, []
        self.before = self.samples()

    def samples(self):
        return {seed: (self.store.account_balance(self.store.account_id_for_seed(seed)), self.store.list_portfolio(self.store.account_id_for_seed(seed)).model_dump_json()) for seed in SAMPLES}

    def account(self, cash: str, **holdings) -> int:
        """A new account with this much cash and these holdings (symbol=shares, each bought at 10.00 on an earlier day)."""
        account_id = self.store.create_account("Booking Holder", "moderate")
        self.made.append(account_id)
        self.store.add_funds(account_id, Decimal(cash) + sum(Decimal("10.00") * n for n in holdings.values()), date(2026, 11, 1))
        for symbol, shares in holdings.items():
            self.store.record_trade(account_id, symbol, "buy", shares, Decimal("10.00"), date(2026, 11, 2), source="user")
        return account_id

    def manager(self, account_id: int, screen=None) -> AccountManager:
        am = AccountManager(model=object(), model_id="booking", store=self.store, account_id=account_id, screen=screen)
        am.trading_portfolio(None, book=True)
        return am

    def held(self, account_id: int) -> dict[str, int]:
        return {p.symbol: p.shares for p in self.store.list_portfolio(account_id).positions}

    def trades(self, account_id: int) -> int:
        return self.store.number_trades(account_id, date(2026, 1, 1), date(2026, 12, 31))


@pytest.fixture
def db(store):
    book = Accounts(store)
    try:
        yield book
    finally:
        for account_id in book.made:
            with suppress(ValueError):
                store.delete_account(account_id)
    assert book.samples() == book.before, "a sample account moved"


# ---------------------------------------------------------------- what book_trade says, and the read back on a new connection

def test_book_trade_says_what_it_did_inside_the_locked_transaction_and_the_trade_row_reads_back_on_a_new_connection(db):
    account_id = db.account("1000.00", KO=3)
    booked = db.store.book_trade(account_id, "XOM", "buy", 4, Decimal("100.00"), DAY, source="bot", record_id="rec-1")
    assert (booked.cash_before, booked.cash_after, booked.held_before, booked.held_after, booked.gain) == (Decimal("1000.00"), Decimal("600.00"), 0, 4, Decimal("0.00"))
    row = db.store.read_booked(account_id, booked.trade_row)
    assert (row["symbol"], row["side"], row["shares"], row["price"], row["record_id"], row["source"]) == ("XOM", "buy", 4, Decimal("100.0000"), "rec-1", "bot")
    assert db.store.read_booked(account_id, booked.trade_row + 10_000) is None and db.held(account_id) == {"KO": 3, "XOM": 4}
    sold = db.store.book_trade(account_id, "KO", "sell", 3, Decimal("12.00"), DAY)
    assert (sold.cash_before, sold.cash_after, sold.held_before, sold.held_after, sold.gain) == (Decimal("600.00"), Decimal("636.00"), 3, 0, Decimal("6.00"))
    assert db.held(account_id) == {"XOM": 4} and db.store.record_trade(account_id, "XOM", "sell", 1, Decimal("100.00"), DAY) == Decimal("0.00")      # record_trade is book_trade and its gain


def test_the_account_manager_reads_the_account_books_the_trade_reads_the_account_again_and_the_second_read_confirms_it(db, monkeypatch):
    account_id = db.account("1000.00")
    am = db.manager(account_id)
    log = []
    for name in ("list_portfolio", "book_trade", "read_booked"):
        real = getattr(db.store, name)
        monkeypatch.setattr(db.store, name, lambda *a, _n=name, _r=real, **k: log.append(_n) or _r(*a, **k))
    booking = am.record_fill(filled("T1", "XOM", "buy", 4, Decimal("100.0"), record_id="rec-1", fill={"price": Decimal("100.0"), "shares": 4, "amount": Decimal("400.0"), "date": DAY.isoformat()}))
    assert log == ["list_portfolio", "book_trade", "list_portfolio", "read_booked"]                                    # SELECT, the one transaction, SELECT again, the trade row
    assert (booking.status, booking.verified, booking.booked, booking.cash_before, booking.cash_avail) == ("booked", True, True, Decimal("1000"), Decimal("600"))
    assert db.held(account_id) == {"XOM": 4} and db.store.account_balance(account_id) == Decimal("600.00") and db.trades(account_id) == 1


# ---------------------------------------------------------------- the row-level locks on the user's portfolio

def hold(store, account_id, sql, release: threading.Event, locked: threading.Event):
    """A second connection that takes row locks with `sql` and keeps them until `release` is set (then rolls back)."""
    with store._connection() as conn:
        conn.execute("SELECT set_config('trading.account_id', %s, true)", (str(account_id),))
        conn.execute(sql, (account_id,))
        locked.set()
        release.wait(20)
        conn.rollback()


@pytest.mark.parametrize("sql, what", [
    ("SELECT cash_balance FROM trading.accounts WHERE account_id = %s FOR UPDATE", "the account row (the cash)"),
    ("SELECT shares FROM trading.positions WHERE account_id = %s AND symbol = 'KO' FOR UPDATE", "a holding that is NOT the one being traded: the whole portfolio is locked, not one symbol"),
])
def test_a_booking_waits_for_the_locks_on_the_portfolio_and_fails_with_nothing_committed_when_it_cannot_have_them(db, monkeypatch, sql, what):
    account_id = db.account("1000.00", KO=3)
    monkeypatch.setattr(accounts_module, "BOOKING_LOCK_SECONDS", 1)
    release, locked = threading.Event(), threading.Event()
    holder = threading.Thread(target=hold, args=(db.store, account_id, sql, release, locked))
    holder.start()
    try:
        assert locked.wait(10)
        started = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            db.store.book_trade(account_id, "XOM", "buy", 4, Decimal("100.00"), DAY)
        assert 0.8 < time.monotonic() - started < 8, what                                                       # it waited for the lock, a bounded time, and did not just fail at once
    finally:
        release.set()
        holder.join(10)
    assert db.held(account_id) == {"KO": 3} and db.store.account_balance(account_id) == Decimal("1000.00") and db.trades(account_id) == 1   # the one trade is the setup's own purchase of KO


def test_a_booking_that_waits_for_a_lock_goes_ahead_when_the_lock_is_let_go_and_sees_the_cash_as_the_other_transaction_left_it(db):
    account_id = db.account("1000.00", KO=3)
    release, locked = threading.Event(), threading.Event()
    holder = threading.Thread(target=hold, args=(db.store, account_id, "SELECT shares FROM trading.positions WHERE account_id = %s FOR UPDATE", release, locked))
    holder.start()
    assert locked.wait(10)
    result = {}
    worker = threading.Thread(target=lambda: result.update(booked=db.store.book_trade(account_id, "XOM", "buy", 4, Decimal("100.00"), DAY)))
    worker.start()
    time.sleep(0.6)
    assert worker.is_alive() and "booked" not in result                                                             # it is waiting on the portfolio lock
    release.set()
    holder.join(10)
    worker.join(10)
    assert result["booked"].cash_after == Decimal("600.00") and db.held(account_id) == {"KO": 3, "XOM": 4}


def test_two_bookings_at_once_are_one_after_the_other_the_second_starts_from_the_cash_the_first_left(db):
    account_id = db.account("1000.00")
    results = []
    barrier = threading.Barrier(2)

    def book(symbol):
        barrier.wait()
        results.append(db.store.book_trade(account_id, symbol, "buy", 4, Decimal("100.00"), DAY))

    threads = [threading.Thread(target=book, args=(s,)) for s in ("XOM", "EOG")]
    [t.start() for t in threads]
    [t.join(20) for t in threads]
    first, second = sorted(results, key=lambda b: b.cash_before, reverse=True)
    assert (first.cash_before, first.cash_after, second.cash_before, second.cash_after) == (Decimal("1000.00"), Decimal("600.00"), Decimal("600.00"), Decimal("200.00"))
    assert db.held(account_id) == {"XOM": 4, "EOG": 4} and db.trades(account_id) == 2


# ---------------------------------------------------------------- the check inside the transaction: a difference rolls it all back

def test_a_trade_the_account_does_not_show_when_it_is_read_again_before_the_commit_is_rolled_back_and_nothing_is_committed(db, monkeypatch):
    account_id = db.account("1000.00", KO=3)
    monkeypatch.setattr(accounts_module, "_buy_position", lambda *a, **k: None)           # the cash moves through the ledger but the position is never written: the account will not show the trade
    with pytest.raises(BookingCheckError, match="read again before the commit.*rolled back and nothing was committed"):
        db.store.book_trade(account_id, "XOM", "buy", 4, Decimal("100.00"), DAY)
    monkeypatch.undo()
    assert db.held(account_id) == {"KO": 3} and db.store.account_balance(account_id) == Decimal("1000.00") and db.trades(account_id) == 1


def test_a_holding_that_changed_during_the_booking_is_a_difference_and_rolls_it_back(db, monkeypatch):
    account_id = db.account("1000.00", KO=3)
    real = accounts_module._buy_position

    def also_touches_another_holding(cur, account, symbol, *rest):
        real(cur, account, symbol, *rest)
        cur.execute("UPDATE trading.positions SET shares = shares + 1 WHERE account_id = %s AND symbol = 'KO'", (account,))

    monkeypatch.setattr(accounts_module, "_buy_position", also_touches_another_holding)
    with pytest.raises(BookingCheckError, match="another holding changed"):
        db.store.book_trade(account_id, "XOM", "buy", 4, Decimal("100.00"), DAY)
    monkeypatch.undo()
    assert db.held(account_id) == {"KO": 3} and db.store.account_balance(account_id) == Decimal("1000.00") and db.trades(account_id) == 1


def test_the_account_manager_says_a_rolled_back_trade_is_refused_shows_the_user_the_error_and_the_account_and_does_not_retry(db, monkeypatch):
    account_id = db.account("1000.00", KO=3)
    shown = []
    am = db.manager(account_id, screen=shown.append)
    monkeypatch.setattr(accounts_module, "_buy_position", lambda *a, **k: None)
    calls = []
    real = db.store.book_trade
    monkeypatch.setattr(db.store, "book_trade", lambda *a, **k: calls.append(1) or real(*a, **k))
    booking = am.record_fill(filled("T1", "XOM", "buy", 4, Decimal("100.0"), fill={"price": Decimal("100.0"), "shares": 4, "amount": Decimal("400.0"), "date": DAY.isoformat()}))
    monkeypatch.undo()
    assert (booking.status, booking.ok, booking.verified) == ("refused", False, False) and "rolled back and nothing was committed" in booking.reason and len(calls) == 1
    said = "\n".join(shown)
    assert "TRADE NOT DONE: the buy of 4 XOM" in said and "will not be tried again" in said and "cash 1,000.00" in said and "KO" in said      # the error, then the account as it stands: nothing moved
    assert db.held(account_id) == {"KO": 3} and db.store.account_balance(account_id) == Decimal("1000.00")


def test_a_purchase_the_cash_does_not_cover_is_refused_with_the_books_sentence_and_shown_to_the_user(db):
    account_id = db.account("100.00")
    shown = []
    am = db.manager(account_id, screen=shown.append)
    booking = am.record_fill(filled("T1", "XOM", "buy", 4, Decimal("100.0"), fill={"price": Decimal("100.0"), "shares": 4, "amount": Decimal("400.0"), "date": DAY.isoformat()}))
    assert (booking.status, booking.reason, booking.cash_before) == ("refused", "cash $100.00 does not cover $400.00", Decimal("100"))
    assert "TRADE NOT DONE" in "\n".join(shown) and "cash $100.00 does not cover $400.00" in "\n".join(shown) and db.held(account_id) == {} and db.trades(account_id) == 0
