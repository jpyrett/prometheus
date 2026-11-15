"""The Account Manager as the Trader's only door (Trader v1 spec section 13): `connect`, `trading_portfolio`, `request_bars`, `perform_trades` and `notify_trades`, and the funding card's own question in `terminal_approver`.

No network, no model and no database: a fake store stands in for Postgres (the real one is tested in test_account_book.py), a fake clerk records what it is asked, and a fake trader records what it is handed.
"""
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from bar_helpers import weekdays, write_day_files
from prometheus.agents.account_manager import account_manager as am
from prometheus.agents.account_manager.decisions import terminal_approver
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import portfolio as pf
from prometheus.common import work_order as wo
from prometheus.common.user_profile import UserProfile
from prometheus.persistence.accounts import Booked

TODAY = date(2026, 11, 5)


class FakeStore:
    """Postgres, as far as the Account Manager calls it. Records every call; `fail` names a symbol whose booking is refused (a ValueError, like the real store's), `crash` one that raises a database error; a booking moves its portfolio as the real book does (entry 82)."""

    def __init__(self, portfolio=None, fail=(), crash=()):
        self.portfolio = portfolio or pf.Portfolio(user="Test Holder", cash_avail=1000.0, positions=[])
        self.calls, self.booked, self.fail, self.crash = [], [], set(fail), set(crash)

    def list_portfolio(self, account_id):
        self.calls.append(("list_portfolio", account_id))
        return self.portfolio

    def load_profile(self, account_id):
        self.calls.append(("load_profile", account_id))
        return UserProfile(user="Test Holder", risk_tolerance="aggressive", trade_mode="auto")

    def get_user(self, account_id):      # the status `approve_trade` reads of a saved account (the real store's UserAccount has it): this one is active
        self.calls.append(("get_user", account_id))
        return SimpleNamespace(display_name="Test Holder", status="active")

    def book_trade(self, account_id, symbol, side, shares, price, traded_on, *, name=None, source="bot", record_id=None):
        """The real store's `book_trade` as far as the Account Manager can tell: a refusal (ValueError), a database error, or the trade applied to the portfolio (the cash and the position move together) and a `Booked`."""
        self.calls.append(("book_trade", account_id))
        if symbol in self.fail:
            raise ValueError(f"cash $1.00 does not cover ${shares * price:,.2f}" if side == "buy" else f"cannot sell {shares} of {symbol}: the portfolio holds 0")
        if symbol in self.crash:
            raise ConnectionError("connection to server at 127.0.0.1 failed: password authentication failed for user app")
        before = self.portfolio
        held_before = (before.position(symbol).shares if before.position(symbol) else 0)
        self.portfolio = pf.apply_fill(before, symbol, side, shares, float(price), traded_on)
        held_after = (self.portfolio.position(symbol).shares if self.portfolio.position(symbol) else 0)
        row = {"account_id": account_id, "symbol": symbol, "side": side, "shares": shares, "price": price, "traded_on": traded_on, "name": name, "source": source, "record_id": record_id}
        self.booked.append(row)
        return Booked(trade_row=len(self.booked), gain=Decimal("0.00"), symbol=symbol, side=side, shares=shares, price=price, cash_before=Decimal(str(before.cash_avail)),
                      cash_after=Decimal(str(self.portfolio.cash_avail)), held_before=held_before, held_after=held_after)

    def record_trade(self, account_id, symbol, side, shares, price, traded_on, *, name=None, source="bot", record_id=None):
        self.calls.append(("record_trade", account_id))
        return self.book_trade(account_id, symbol, side, shares, price, traded_on, name=name, source=source, record_id=record_id).gain

    def read_booked(self, account_id, trade_row):
        self.calls.append(("read_booked", account_id))
        return {**self.booked[trade_row - 1], "trade_id": trade_row} if 0 < trade_row <= len(self.booked) else None


def manager(**options):
    return am.AccountManager(model=object(), model_id="account-book", **options)


def holder(cash=1000.0, **positions):
    return pf.Portfolio(user="Test Holder", cash_avail=cash, positions=[{"symbol": s, "shares": n, "cost_basis": 100.0, "acquired": "2026-01-02"} for s, n in positions.items()])


def filled(trade_id, symbol, side, shares, price, record_id="r1", **changes):
    return {"trade_id": trade_id, "kind": "recommendation", "record_id": record_id, "symbol": symbol, "side": side, "shares": shares, "status": "filled",
            "fill": {"price": price, "shares": shares, "amount": round(shares * price, 2), "commission": 0.0, "date": TODAY.isoformat(), "price_source": "quote", "slippage_bps": 0.0}, **changes}


# ---------------------------------------------------------------- connect


@pytest.mark.parametrize("store, account_id, message", [
    (None, None, "needs both"), (FakeStore(), None, "needs both"), (None, 7, "needs both"),
    (FakeStore(), 0, "positive whole number"), (FakeStore(), -3, "positive whole number"), (FakeStore(), True, "positive whole number"), (FakeStore(), "7", "positive whole number"), (FakeStore(), 2.0, "positive whole number"),
])
def test_connect_refuses_what_the_constructor_refuses_and_both_arguments_are_needed(store, account_id, message):
    m = manager()
    with pytest.raises(ValueError, match=message):
        m.connect(store, account_id)
    assert (m.store, m.account_id) == (None, None)


def test_connect_forgets_a_profile_read_from_the_default_file_so_the_account_row_is_used():
    store, m = FakeStore(), manager()
    assert m.profile.user == "Jacob Pyrett" and m.profile.trade_mode == "ask"        # read from configs/profiles/jacob_pyrett.json
    m.connect(store, 7)
    assert m.profile.user == "Test Holder" and m.profile.trade_mode == "auto" and store.calls == [("load_profile", 7)]


# ---------------------------------------------------------------- trading_portfolio


def test_there_is_no_trading_portfolio_without_a_book_and_without_one_given():
    with pytest.raises(ValueError, match="no portfolio to trade"):
        manager().trading_portfolio()
    with pytest.raises(ValueError, match="no portfolio to trade"):
        manager().trading_portfolio(None)


# ---------------------------------------------------------------- the per-trade handshake against the store (design choices entry 81): what the Account Manager asks the book, and when

def real_ticket():
    """A REAL ticket of the real session's order (XOM, 8 shares) and the order it came from: what `approve_trade` is given."""
    from handshake_helpers import BOB, MOMENT, real_order
    from prometheus.agents.trader import tickets as tk
    from prometheus.agents.trader.config import load_config
    order = real_order()
    return order, tk.build_tickets(order, BOB, MOMENT, load_config(), "paper")[0].model_dump(mode="json"), MOMENT


def test_an_account_that_is_not_active_is_not_approved_and_a_database_failure_is_not_approved_with_the_kind_only():
    order, xom, now = real_ticket()
    store = FakeStore(portfolio=holder(5000.0))
    m = manager(store=store, account_id=7)
    m.trading_portfolio(None, book=True)
    store.get_user = lambda account_id: SimpleNamespace(status="suspended")
    refused = m.approve_trade(xom, 170.6753, order=order, now=now, on=TODAY)
    assert (refused.approved, refused.code, refused.reason) == (False, "account_refused", "the account is suspended: it cannot take a trade")

    def down(account_id):
        raise ConnectionError("connection to server at 127.0.0.1 failed: password authentication failed for user app")

    store.list_portfolio = down
    store.get_user = lambda account_id: SimpleNamespace(status="active")
    failed = m.approve_trade(xom, 170.6753, order=order, now=now, on=TODAY)
    assert (failed.approved, failed.code) == (False, "account_refused") and "ConnectionError" in failed.reason and "password" not in failed.reason and "127.0.0.1" not in failed.reason


def test_a_fill_the_book_refuses_is_refused_with_its_sentence_and_shown_to_the_user_with_the_account_as_it_stands():
    store = FakeStore(portfolio=holder(5000.0, KO=2), fail={"KO"})
    shown = []
    m = manager(store=store, account_id=7, screen=shown.append)
    m.trading_portfolio(None, book=True)
    refused = m.record_fill(filled("T1", "KO", "buy", 5, 100.0))
    assert (refused.status, refused.reason, refused.cash_avail, refused.booked, refused.ok, refused.verified) == ("refused", "cash $1.00 does not cover $500.00", None, False, False, False) and store.booked == []
    said = "\n".join(shown)
    assert "TRADE NOT DONE: the buy of 5 KO" in said and "will not be tried again" in said and "cash $1.00 does not cover $500.00" in said and "cash 5,000.00" in said and "KO" in said.split("cash 5,000.00")[1]


def test_a_lost_connection_is_settled_by_reading_the_account_again_unchanged_is_refused_and_anything_else_is_unconfirmed_and_never_a_connection_detail():
    class Dies(FakeStore):                      # the connection is gone for good once the booking failed: the account cannot be read again
        def book_trade(self, *a, **kw):
            self.dead = True
            return super().book_trade(*a, **kw)

        def list_portfolio(self, account_id):
            if getattr(self, "dead", False):
                raise ConnectionError("connection to server at 127.0.0.1 failed: password authentication failed for user app")
            return super().list_portfolio(account_id)

    class Moves(FakeStore):                     # the account is different when it is read again (somebody else's booking, or the trade itself): nobody can say which
        def book_trade(self, *a, **kw):
            self.portfolio = self.portfolio.model_copy(update={"cash_avail": self.portfolio.cash_avail - 1.0})
            return super().book_trade(*a, **kw)

    def attempt(store):
        m = manager(store=store, account_id=7)
        m.trading_portfolio(None, book=True)
        return m.record_fill(filled("T1", "PG", "buy", 3, 100.0))

    same = attempt(FakeStore(portfolio=holder(5000.0), crash={"PG"}))
    assert (same.status, same.ok, same.cash_avail) == ("refused", False, 5000.0) and "the booking failed (ConnectionError)" in same.reason and "unchanged: nothing was committed" in same.reason
    dead = attempt(Dies(portfolio=holder(5000.0), crash={"PG"}))
    assert (dead.status, dead.ok) == ("unconfirmed", False) and "the booking could not be confirmed (ConnectionError)" in dead.reason and "may or may not hold this trade" in dead.reason
    moved = attempt(Moves(portfolio=holder(5000.0), crash={"PG"}))
    assert (moved.status, moved.ok) == ("unconfirmed", False) and "is not what it was before" in moved.reason
    for answer in (same, dead, moved):
        assert "password" not in answer.reason and "127.0.0.1" not in answer.reason


def test_a_committed_booking_that_the_second_read_does_not_show_is_unconfirmed_and_says_what_is_missing_and_one_that_cannot_be_read_again_still_stands():
    class Forgets(FakeStore):                   # the book says yes, and the second read finds no such trade row
        def read_booked(self, account_id, trade_row):
            return None

    class Stuck(FakeStore):                     # the book says yes, and the cash and the shares are the same as before: the trade is not in the totals
        def book_trade(self, *a, **kw):
            booked = super().book_trade(*a, **kw)
            self.portfolio = holder(5000.0)
            return booked

    def attempt(store):
        m = manager(store=store, account_id=7)
        m.trading_portfolio(None, book=True)
        return m.record_fill(filled("T1", "PG", "buy", 3, 100.0))

    missing = attempt(Forgets(portfolio=holder(5000.0)))
    assert (missing.status, missing.ok) == ("unconfirmed", False) and "the trade is not in the book's list of trades" in missing.reason and "has to be checked" in missing.reason
    stuck = attempt(Stuck(portfolio=holder(5000.0)))
    assert stuck.status == "unconfirmed" and "the cash is the same as before the trade" in stuck.reason and "the shares of PG are the same as before the trade" in stuck.reason
    store, shown = FakeStore(portfolio=holder(5000.0)), []
    m = manager(store=store, account_id=7, screen=shown.append)
    m.trading_portfolio(None, book=True)
    real_read = store.list_portfolio
    reads = []

    def second_read_fails(account_id):
        reads.append(1)
        if len(reads) == 2:
            raise ConnectionError("gone")
        return real_read(account_id)

    store.list_portfolio = second_read_fails
    booking = m.record_fill(filled("T1", "PG", "buy", 3, 100.0))
    assert (booking.status, booking.booked, booking.verified) == ("booked", True, False) and booking.cash_avail == store.portfolio.cash_avail
    assert "is booked, but the account could not be read again to confirm it (ConnectionError)" in booking.note and any("could not be read again" in line for line in shown) and len(store.booked) == 1


# ---------------------------------------------------------------- request_bars

class BarClerk:
    """Answers `request_bars` with a canned order and records the order it was handed."""

    def __init__(self):
        self.orders = []

    def request_bars(self, order):
        self.orders.append(order.model_copy(deep=True))
        order.status = "data_ready"
        order.data = wo.DataResult(bars={s: [] for s in order.subject})
        return order


@pytest.mark.parametrize("start, end", [("2026-11-05", date(2026, 11, 6)), (date(2026, 11, 5), "2026-11-06"), (None, date(2026, 11, 6))])
def test_request_bars_wants_dates_and_says_so_before_asking_anybody(start, end):
    clerk = BarClerk()
    with pytest.raises(ValueError, match="must be dates"):
        manager(clerk=clerk).request_bars(["PG"], start, end)
    assert clerk.orders == []


def test_a_refused_bars_request_comes_back_as_a_failed_order_not_an_exception():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"the model was used ({name})")

    result = manager(clerk=DataClerk(toolbox=object(), model=Forbidden())).request_bars(["PG"], date(2026, 12, 4), date(2026, 11, 5))
    assert result.status == "data_failed" and any("is after end" in e for e in result.errors)


# ---------------------------------------------------------------- perform_trades

class FakeTrader:
    def __init__(self, returns=None):
        self.calls, self.returns = [], returns

    def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book=False):
        self.calls.append({"order": order, "account": account, "portfolio": portfolio, "now": now, "funding": funding, "approver": approver, "book": book})
        order.status = "traded"
        return self.returns or order


# ---------------------------------------------------------------- notify_trades, in memory


def test_only_filled_trades_are_applied():
    trades = [{**filled("T1", "PG", "buy", 4, 100.0), "status": s} for s in ("new", "active", "rejected")] + [{"trade_id": "T9", "status": "rejected", "reject_code": "not_permitted"}]
    after, problems = manager().notify_trades(holder(1000.0), trades, TODAY)
    assert problems == [] and after.cash_avail == 1000.0 and after.positions == []


def test_a_funding_sale_before_its_purchase_pays_for_it_in_the_order_given_and_the_purchase_alone_is_not_covered():
    given = holder(100.0, MSFT=50)
    sale, buy = filled("T1-F1", "MSFT", "sell", 5, 420.0, kind="funding", funds_for="T1"), filled("T1", "PG", "buy", 13, 167.1)
    after, problems = manager().notify_trades(given, [sale, buy], TODAY)
    assert problems == [] and after.cash_avail == round(100.0 + 2100.0 - round(13 * 167.1, 2), 2) == 27.7 and after.position("MSFT").shares == 45 and after.position("PG").shares == 13
    alone, problems = manager().notify_trades(given, [buy], TODAY)
    assert alone.positions == given.positions and alone.cash_avail == 100.0 and len(problems) == 1 and problems[0].startswith("T1: buy 13 PG was not applied: cash $100.00 does not cover")
    wrong_way, problems = manager().notify_trades(given, [buy, sale], TODAY)           # the order given is the order applied: a purchase before the sale that pays for it is not covered
    assert len(problems) == 1 and "T1: buy 13 PG was not applied" in problems[0] and wrong_way.position("MSFT").shares == 45


def test_a_fill_that_cannot_be_applied_is_named_and_the_others_still_go_through():
    given = holder(1000.0, MSFT=5)
    trades = [filled("T1", "PG", "buy", 4, 100.0), filled("T2", "XYZ", "sell", 1, 10.0, record_id="r2"), filled("T3", "MSFT", "sell", 5, 400.0, record_id="r3")]
    after, problems = manager().notify_trades(given, trades, TODAY)
    assert problems == ["T2: sell 1 XYZ was not applied: cannot sell 1 of XYZ: the portfolio holds 0"]
    assert after.symbols == ["PG"] and after.cash_avail == 1000.0 - 400.0 + 2000.0


def test_a_filled_trade_with_no_fill_or_a_damaged_entry_is_a_problem_and_not_an_exception():
    after, problems = manager().notify_trades(holder(1000.0), [{"trade_id": "T1", "status": "filled", "symbol": "PG", "side": "buy"}, "junk", {"status": "filled", "fill": None}], TODAY)
    assert after.cash_avail == 1000.0 and problems == ["T1: filled, but it has no fill", "trade #2: not a record", "#3: filled, but it has no fill"]


@pytest.mark.parametrize("change", [{"side": "short"}, {"symbol": "../x"}, {"fill": {"price": 0, "shares": 4}}, {"fill": {"price": 100.0, "shares": 0}}, {"fill": {"price": "100", "shares": 4}}])
def test_a_fill_of_nonsense_is_a_problem_naming_the_trade(change):
    after, problems = manager().notify_trades(holder(1000.0), [{**filled("T7", "PG", "buy", 4, 100.0), **change}], TODAY)
    assert after.cash_avail == 1000.0 and len(problems) == 1 and problems[0].startswith("T7: ")


def test_the_date_of_the_trades_must_be_a_date():
    from datetime import datetime

    for bad in ("2026-11-05", datetime(2026, 11, 5, 10, 30), None):
        with pytest.raises(ValueError, match="traded_on must be a date"):
            manager().notify_trades(holder(), [], bad)


# ---------------------------------------------------------------- notify_trades, booked

def test_with_book_and_a_connected_account_each_fill_is_booked_as_the_bot_with_its_recommendation():
    store = FakeStore(portfolio=holder(1000.0, MSFT=5))
    m = manager(store=store, account_id=7)
    trades = [filled("T1-F1", "MSFT", "sell", 2, 400.0, kind="funding", funds_for="T1", record_id="r1"), filled("T1", "PG", "buy", 4, 167.1, record_id="r1"), filled("T2", "F", "buy", 3, 10.0, record_id=None)]
    after, problems = m.notify_trades(store.portfolio, trades, TODAY, book=True)
    assert problems == []
    assert store.booked == [
        {"account_id": 7, "symbol": "MSFT", "side": "sell", "shares": 2, "price": Decimal("400.0"), "traded_on": TODAY, "name": None, "source": "bot", "record_id": "r1"},
        {"account_id": 7, "symbol": "PG", "side": "buy", "shares": 4, "price": Decimal("167.1"), "traded_on": TODAY, "name": None, "source": "bot", "record_id": "r1"},
        {"account_id": 7, "symbol": "F", "side": "buy", "shares": 3, "price": Decimal("10.0"), "traded_on": TODAY, "name": None, "source": "bot", "record_id": None}]
    assert after.cash_avail == round(1000.0 + 800.0 - 668.4 - 30.0, 2) and after.symbols == ["MSFT", "PG", "F"]


def test_book_without_a_connected_account_is_one_problem_and_nothing_is_booked_but_the_fills_apply_in_memory():
    trades = [filled("T1", "PG", "buy", 3, 100.0), filled("T2", "F", "buy", 2, 10.0, record_id="r2")]
    after, problems = manager().notify_trades(holder(1000.0), trades, TODAY, book=True)
    assert len(problems) == 1 and problems[0].startswith("book: the fills were not booked") and "no saved account" in problems[0]
    assert after.cash_avail == 680.0 and after.symbols == ["PG", "F"]


def test_a_fill_the_book_refuses_is_named_and_left_out_of_the_portfolio_and_the_others_are_still_booked():
    store = FakeStore(fail=("PG",))
    m = manager(store=store, account_id=7)
    trades = [filled("T1", "PG", "buy", 3, 100.0), filled("T2", "F", "buy", 2, 10.0, record_id="r2")]
    after, problems = m.notify_trades(store.portfolio, trades, TODAY, book=True)
    assert problems == ["T1: buy 3 PG was not booked: cash $1.00 does not cover $300.00"]
    assert [b["symbol"] for b in store.booked] == ["F"] and after.symbols == ["F"] and after.cash_avail == 980.0       # the portfolio returned is what was booked


def test_a_database_error_names_its_kind_and_never_its_message():
    store = FakeStore(crash=("PG",))
    after, problems = manager(store=store, account_id=7).notify_trades(store.portfolio, [filled("T1", "PG", "buy", 3, 100.0)], TODAY, book=True)
    assert problems == ["T1: buy 3 PG was not booked (ConnectionError)"] and "password" not in " ".join(problems) and "127.0.0.1" not in " ".join(problems)
    assert after.positions == [] and after.cash_avail == 1000.0


def test_a_fill_that_cannot_be_applied_is_not_sent_to_the_book():
    store = FakeStore()
    after, problems = manager(store=store, account_id=7).notify_trades(store.portfolio, [filled("T1", "NVDA", "sell", 1, 100.0)], TODAY, book=True)
    assert len(problems) == 1 and "was not applied" in problems[0] and store.booked == [] and ("record_trade", 7) not in store.calls


# ---------------------------------------------------------------- record_trade and the funding card's question


class Prompt:
    def __init__(self, *answers):
        self.answers, self.asked, self.shown = list(answers), [], []

    def ask(self, question):
        self.asked.append(question)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def show(self, text):
        self.shown.append(text)


FUNDING = {"text": "SELL MSFT 5", "symbol": "PG", "shares": 44, "est_cost": 7352.40, "kind": "funding", "question": "Sell these to pay for 44 shares of PG (about $7,352)? [y/N] "}
RECOMMENDATION = {"text": "PG: approve", "symbol": "PG", "shares": 44, "est_cost": 7352.40}


def test_nobody_there_or_the_end_of_input_is_no_answer_at_all():
    prompt = Prompt(EOFError())
    assert terminal_approver(ask=prompt.ask, show=prompt.show, interactive=False)(FUNDING) is None and prompt.asked == [] and prompt.shown == []
    assert terminal_approver(ask=prompt.ask, show=prompt.show, interactive=True)(FUNDING) is None
