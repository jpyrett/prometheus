"""A direct order whose limit is not met WAITS, and the watcher fills it, expires it or leaves it alone (design choices entry 140; design/limit_orders_spec.md W1 to W7), on the REAL parts: the real Account Manager over the real Data Clerk (its
symbols from Finnhub's recorded directory, its prices the STORED CLOSES of real sessions), the real Trader with no model, the real local Postgres on throwaway accounts, and the market calendar's real sessions for the fake clock.

The "price that falls" is the real stored close of a later (or earlier) session of the same stock: MSFT closed 517.99 on Monday 2026-11-16 and 506.24 on Tuesday the 17th; 523.00 on Thursday the 12th. Nothing here is invented stock data.
"""
import json
import time
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from open_orders_helpers import FRIDAY, MONDAY, clerk, clerk_with_directory, close_of, db, eastern, stored_close, store, trader  # noqa: F401  (db, store and stored_close are fixtures)

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager import orders
from prometheus.common.work_order import new_work_order

TUESDAY = date(2026, 11, 17)
THURSDAY = date(2026, 11, 12)


def direct(db, account_id, typed: str, *wanted: it.DirectOrder, now, mode: str = "auto", approver=None):
    """The user's request as the desk makes it: the Account Manager asks the real Data Clerk for each company and symbol, then decides the orders at `now`. Returns (the decided work order, the Account Manager)."""
    am = db.manager(account_id, clerk=clerk_with_directory())
    order = new_work_order(typed, requester="desk", action_type="trade")
    problems = am.name_the_stocks(order, list(wanted), typed, now)
    assert problems == [], problems
    return am.decide_direct(order, am.list_portfolio(), now, mode, approver), am


def watcher(db, account_id: int, box: dict, events: list, tmp_path, **options):
    """A watcher over the real book with a fake clock (`box["now"]`) and the real Trader with no model; its events go to `events` and its audit log to a temporary file. It looks at THIS account's orders only: a test never expires or
    fills an order of an account it did not make (the dev book may hold real ones)."""
    return orders.Watcher(db.store, clerk=options.pop("clerk", clerk()), clock=lambda: box["now"], trader_factory=trader, notify=events.append, audit_path=tmp_path / "audit.jsonl", accounts=[account_id], **options)


def msft_buy(shares=10, ceiling="510"):
    return it.DirectOrder(side="buy", symbol="MSFT", shares=shares, max_price=Decimal(ceiling))


def test_a_purchase_above_its_ceiling_waits_with_its_cash_set_aside_and_the_watcher_fills_it_when_the_quote_reaches_the_ceiling(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    now = eastern(MONDAY, 10, 30)
    order, am = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=now)
    price = close_of("MSFT", MONDAY)      # 517.99: above the $510 ceiling
    decision = order.trade_decisions[0]
    assert price == Decimal("517.99") and decision["will_trade"] == "no_system_denied" and decision["origin"] == "direct" and order.trades == []      # nothing trades now
    assert decision["reason"] == "Waiting: buy 10 MSFT at $510.00 or less (now $517.99), $5,100.00 set aside, until 4:00 pm ET today."
    waiting = db.store.open_orders(account)
    assert [(o.order_id, o.symbol, o.side, o.shares, o.limit_price, o.reserved_cash, o.status) for o in waiting] == [(decision["open_order_id"], "MSFT", "buy", Decimal(10), Decimal("510.0000"), Decimal("5100.00"), "open")]
    assert waiting[0].expires_at == datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc) and waiting[0].company == "Microsoft Corp"
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.cash_total) == (Decimal("4900.00"), Decimal("5100.00"), Decimal("10000.00"))
    box, events = {"now": eastern(MONDAY, 11, 0)}, []
    mine = watcher(db, account, box, events, tmp_path)
    assert mine.tick() == [] and db.store.get_open_order(account, waiting[0].order_id).status == "open"      # the quote is still 517.99: it waits
    stored_close["day"] = TUESDAY      # the real close of the next session, 506.24: at or under the ceiling
    box["now"] = eastern(MONDAY, 11, 5)
    done = mine.tick()
    assert [(e["type"], e["order_id"], e["symbol"]) for e in done] == [("filled", waiting[0].order_id, "MSFT")] and done == events
    assert done[0]["text"] == f"Order #{waiting[0].order_id} filled: bought 10 MSFT at $506.24 ($5,062.40), the waiting order you left: buy 10 MSFT at $510.00 or less."
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_cash, held.position("MSFT").shares, held.position("MSFT").cost_basis) == (Decimal("4937.60"), Decimal("0.00"), Decimal(10), Decimal("506.2400"))
    filled = db.store.get_open_order(account, waiting[0].order_id)
    assert filled.status == "filled" and filled.trade_id is not None and db.store.read_booked(account, filled.trade_id)["price"] == Decimal("506.2400")
    assert mine.tick() == [] and db.store.number_trades(account, date(2026, 1, 1), date(2026, 12, 31)) == 1      # nothing is booked twice
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["filled"]      # the fill is a line of the audit log (the placement's own line went to the test's temporary log)


def test_a_sale_under_its_floor_waits_with_its_shares_set_aside_which_nothing_else_can_sell_and_the_watcher_fills_it_at_the_floor_or_over(db, stored_close, tmp_path):
    account = db.account("20000.00", mode="auto")
    db.buy(account, "MSFT", 20, "400.00")
    now = eastern(MONDAY, 10, 30)
    order, am = direct(db, account, "Sell 5 shares of MSFT at $520 or more", it.DirectOrder(side="sell", symbol="MSFT", shares=5, min_price=Decimal("520")), now=now)
    decision = order.trade_decisions[0]
    assert decision["reason"] == "Waiting: sell 5 MSFT at $520.00 or more (now $517.99), 5 shares set aside, until 4:00 pm ET today." and decision["open_order_id"] is not None
    held = db.store.list_portfolio(account)
    assert (held.reserved_shares, held.free_shares("MSFT"), held.position("MSFT").shares, held.cash_avail) == ({"MSFT": Decimal(5)}, Decimal(15), Decimal(20), Decimal("12000.00"))
    again, _ = direct(db, account, "Sell 18 shares of MSFT", it.DirectOrder(side="sell", symbol="MSFT", shares=18), now=now)      # 18 are held, 5 of them set aside: only 15 can be sold
    assert again.trade_decisions[0]["will_trade"] == "no_funds" and "5 of them set aside for waiting orders" in again.trade_decisions[0]["reason"]
    box, events = {"now": eastern(MONDAY, 11, 0)}, []
    mine = watcher(db, account, box, events, tmp_path)
    assert mine.tick() == []
    stored_close["day"] = THURSDAY      # 523.00: at or over the floor
    done = mine.tick()
    assert [e["type"] for e in done] == ["filled"] and "sold 5 MSFT at $523.00 ($2,615.00)" in done[0]["text"]
    held = db.store.list_portfolio(account)
    assert (held.cash_avail, held.reserved_shares, held.position("MSFT").shares) == (Decimal("14615.00"), {}, Decimal(15))


def test_an_order_in_dollars_sets_aside_its_dollars_and_buys_what_they_pay_for_at_the_fill_price_never_more(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    now = eastern(MONDAY, 10, 30)
    order, _ = direct(db, account, "Buy $5,000 of MSFT at $510 or less", it.DirectOrder(side="buy", symbol="MSFT", dollars=Decimal("5000"), max_price=Decimal("510")), now=now)
    decision = order.trade_decisions[0]
    assert decision["reason"] == "Waiting: buy $5,000.00 of MSFT at $510.00 or less (now $517.99), $5,000.00 set aside, until 4:00 pm ET today."
    assert db.store.available_cash(account) == Decimal("5000.00")
    stored_close["day"] = TUESDAY
    box, events = {"now": eastern(MONDAY, 11, 5)}, []
    done = watcher(db, account, box, events, tmp_path).tick()
    assert [e["type"] for e in done] == ["filled"]
    bought = db.store.list_portfolio(account)
    spent = Decimal("10000.00") - bought.cash_avail
    assert spent == Decimal("5000.00") and bought.reserved_cash == Decimal("0.00")      # at most its dollars (never more), to the cent: 5000 / 506.24 is 9.8767383..., so 9.876738 shares cost 4999.9998 and round to 5000.00
    assert bought.position("MSFT").shares == Decimal("9.876738") and db.store.holds_fractions()


def test_an_order_placed_after_the_close_is_for_the_next_session_and_is_told_so(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    stored_close["day"] = FRIDAY
    now = eastern(FRIDAY, 17, 0)      # an hour after the close of Friday 2026-11-13
    order, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=now)
    assert order.trade_decisions[0]["reason"] == "Waiting: buy 10 MSFT at $510.00 or less (now $523.88), $5,100.00 set aside, until 4:00 pm ET Monday 2026-11-16."
    waiting = db.store.open_orders(account)[0]
    assert waiting.expires_at == datetime(2026, 11, 16, 21, 0, tzinfo=timezone.utc)
    box, events = {"now": eastern(MONDAY, 9, 0)}, []      # Monday before the open: nothing is filled or expired
    mine = watcher(db, account, box, events, tmp_path)
    stored_close["day"] = TUESDAY
    assert mine.tick() == [] and db.store.get_open_order(account, waiting.order_id).status == "open"
    box["now"] = eastern(MONDAY, 9, 31)      # the session is open and the quote (506.24) is under the ceiling
    assert [e["type"] for e in mine.tick()] == ["filled"]


def test_a_day_order_expires_at_its_sessions_close_releases_its_hold_and_is_never_carried_to_another_day(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    order, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    waiting = db.store.open_orders(account)[0]
    box, events = {"now": eastern(MONDAY, 15, 59)}, []
    mine = watcher(db, account, box, events, tmp_path)
    assert mine.tick() == [] and db.store.available_cash(account) == Decimal("4900.00")      # one minute before the close it still waits
    box["now"] = eastern(MONDAY, 16, 0)
    done = mine.tick()
    assert [(e["type"], e["order_id"]) for e in done] == [("expired", waiting.order_id)] and "available again" in done[0]["text"]
    assert (db.store.available_cash(account), db.store.open_orders(account), db.store.get_open_order(account, waiting.order_id).status) == (Decimal("10000.00"), [], "expired")
    stored_close["day"] = TUESDAY      # a price that would have filled it: too late
    box["now"] = eastern(TUESDAY, 10, 0)
    assert mine.tick() == [] and db.store.list_portfolio(account).position("MSFT") is None


def test_a_server_that_starts_after_the_session_expires_what_is_stale_before_it_does_anything_else(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    box, events = {"now": eastern(TUESDAY, 8, 0)}, []      # the server was off overnight; it starts before the next open
    first = watcher(db, account, box, events, tmp_path).tick()
    assert [e["type"] for e in first] == ["expired"] and db.store.available_cash(account) == Decimal("10000.00")


def test_a_quote_that_failed_is_tried_again_at_the_next_tick_and_an_outage_never_fills_or_expires_an_order_early(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    stored_close["day"] = TUESDAY
    real, down = clerk(), {"on": True}

    class Outage:
        """The real Data Clerk with one real failure injected: while `down` is on, its quote call fails the way an API outage does."""
        def request_quotes(self, order):
            if down["on"]:
                raise ConnectionError("the quote API is down")
            return real.request_quotes(order)

    box, events = {"now": eastern(MONDAY, 11, 0)}, []
    mine = watcher(db, account, box, events, tmp_path, clerk=Outage())
    assert mine.tick() == [] and mine.errors == ["the quotes could not be fetched (ConnectionError)"]
    assert db.store.open_orders(account)[0].status == "open"      # never filled, never expired
    down["on"] = False
    box["now"] = eastern(MONDAY, 11, 1)
    assert [e["type"] for e in mine.tick()] == ["filled"]


def test_a_fill_that_fails_is_rejected_shown_never_retried_and_its_hold_is_released(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    db.store.update_user(account, status="suspended")      # the account cannot take a trade any more: the approval is refused
    stored_close["day"] = TUESDAY
    box, events = {"now": eastern(MONDAY, 11, 5)}, []
    mine = watcher(db, account, box, events, tmp_path)
    done = mine.tick()
    assert [e["type"] for e in done] == ["rejected"] and "will not be tried again" in done[0]["text"] and "suspended" in done[0]["text"]
    assert db.store.open_orders(account) == [] and db.store.available_cash(account) == Decimal("10000.00") and db.store.list_portfolio(account).position("MSFT") is None
    assert mine.tick() == []      # a rejected order is not tried again


def test_a_waiting_order_is_a_direct_order_only_and_a_limit_that_is_met_trades_at_once_as_ever(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    stored_close["day"] = TUESDAY      # 506.24: already at or under the $510 ceiling
    order, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    decision = order.trade_decisions[0]
    assert decision["will_trade"] == "yes_user" and decision.get("open_order_id") is None and db.store.open_orders(account) == []      # nothing waits: the Trader fills it at the quote, inside the limit


def test_what_cannot_wait_is_not_placed_and_says_why(db, stored_close, tmp_path):
    account = db.account("1000.00", mode="auto")
    now = eastern(MONDAY, 10, 30)
    no_cash, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=now)      # $5,100 would be set aside and $1,000 is there
    assert no_cash.trade_decisions[0]["will_trade"] == "no_funds" and "$1,000.00" in no_cash.trade_decisions[0]["reason"] and "cannot wait" in no_cash.trade_decisions[0]["reason"]
    no_amount, _ = direct(db, account, "Buy MSFT at $510 or less", it.DirectOrder(side="buy", symbol="MSFT", max_price=Decimal("510")), now=now)
    assert no_amount.trade_decisions[0]["will_trade"] == "no_system_denied" and "needs an amount to set aside" in no_amount.trade_decisions[0]["reason"]
    under, _ = direct(db, account, "Buy 1 share of MSFT if it is over $600", it.DirectOrder(side="buy", symbol="MSFT", shares=1, min_price=Decimal("600")), now=now)      # a buy that waits for the price to RISE is a stop order
    assert under.trade_decisions[0]["will_trade"] == "no_system_denied" and "under your floor of $600.00" in under.trade_decisions[0]["reason"] and "stop order" in under.trade_decisions[0]["reason"]
    assert db.store.open_orders(account) == []


def test_in_ask_mode_a_named_company_is_confirmed_before_anything_is_set_aside(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="ask")
    asked = []

    def user(card):
        asked.append(card)
        return False

    now = eastern(MONDAY, 10, 30)
    order, _ = direct(db, account, "buy 10 shares of Microsoft at $510 or less", it.DirectOrder(side="buy", company="Microsoft", shares=10, max_price=Decimal("510")), now=now, mode="ask", approver=user)
    assert len(asked) == 1 and "WAITS" in asked[0]["text"] and "$510.00 or less" in asked[0]["question"]
    assert order.trade_decisions[0]["will_trade"] == "no_user_denied" and db.store.open_orders(account) == [] and db.store.available_cash(account) == Decimal("10000.00")
    yes, _ = direct(db, account, "buy 10 shares of Microsoft at $510 or less", it.DirectOrder(side="buy", company="Microsoft", shares=10, max_price=Decimal("510")), now=now, mode="ask", approver=lambda card: True)
    assert yes.trade_decisions[0]["open_order_id"] is not None and db.store.available_cash(account) == Decimal("4900.00")


def test_the_user_cancels_a_waiting_order_and_what_it_set_aside_is_available_again(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    order, am = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    number = order.trade_decisions[0]["open_order_id"]
    closed = am.cancel_open_order(number)
    assert (closed.status, db.store.available_cash(account), am.open_orders()) == ("cancelled", Decimal("10000.00"), [])
    with pytest.raises(ValueError, match="is cancelled"):
        am.cancel_open_order(number)


def test_the_account_listing_and_the_run_message_show_the_cash_the_orders_hold(db, stored_close, tmp_path):
    from prometheus.agents.account_manager.account_manager import format_account
    account = db.account("10000.00", mode="auto")
    order, am = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    shown = format_account(am, date(2026, 1, 1), date(2026, 12, 31))
    assert "cash 10,000.00" in shown and "WAITING ORDERS  (cash set aside 5,100.00; available 4,900.00)" in shown and f"#{order.trade_decisions[0]['open_order_id']} buy 10 MSFT at $510.00 or less, $5,100.00 set aside, open" in shown
    lines = []
    am.screen = lines.append
    am.show_portfolio("the account now")
    assert any("set aside for waiting orders 5,100.00" in line for line in lines) and any("available 4,900.00" in line for line in lines)


def test_only_the_watcher_that_holds_the_lock_ticks_and_a_second_one_waits_for_it(db, stored_close, tmp_path):
    account = db.account("10000.00", mode="auto")
    direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    stored_close["day"] = TUESDAY
    box, events = {"now": eastern(MONDAY, 11, 5)}, []
    other = db.store.watcher_lock()      # another server's watcher holds the lock
    assert other.try_acquire()
    mine = watcher(db, account, box, events, tmp_path, seconds=1)
    mine.start()
    try:
        deadline = time.monotonic() + 15
        while mine.loops < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mine.loops >= 2 and mine.owns_lock is False and events == [] and db.store.open_orders(account)[0].status == "open"      # it went round and did nothing
        other.release()
        deadline = time.monotonic() + 15
        while not events and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mine.owns_lock is True and [e["type"] for e in events] == ["filled"]
    finally:
        mine.stop()
        other.release()
    assert db.store.watcher_running() is False      # stopped: the lock is let go


def test_a_watcher_is_never_started_by_building_one_only_by_start_and_it_stops_with_stop(db, stored_close, tmp_path):
    mine = watcher(db, db.account("100.00"), {"now": eastern(MONDAY, 11, 0)}, [], tmp_path, seconds=1)
    assert mine._thread is None and db.store.watcher_running() is False
    mine.start()
    try:
        assert mine._thread.is_alive()
    finally:
        mine.stop()
    assert mine._thread is None and db.store.watcher_running() is False


def test_the_watchers_own_modules_name_no_size_limit_and_no_risk_rule():
    """Nothing in the waiting-order code limits a trade (AGENTS.md: the Account Manager does no risk management): the only checks are the cash, the shares and the limit price the user wrote."""
    import inspect
    from prometheus.agents.account_manager import orders as module
    source = inspect.getsource(module).lower()
    assert not any(word in source for word in ("max_position", "max_order", "liquidity", "exposure", "max_loss", "risk_level"))


def test_the_command_lists_and_cancels_the_accounts_waiting_orders_and_says_when_no_server_is_watching(db, stored_close, capsys):
    account = db.account("10000.00", mode="auto")
    order, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    number = order.trade_decisions[0]["open_order_id"]
    assert orders.main(["--account", str(account), "--list"]) == 0
    listed = capsys.readouterr().out
    assert f"account {account}: 1 waiting order" in listed and f"#{number} buy 10 MSFT at $510.00 or less, $5,100.00 set aside, open" in listed
    assert "watched only while the page's server runs" in listed and "NO SERVER IS RUNNING a watcher now" in listed
    lock = db.store.watcher_lock()
    assert lock.try_acquire()
    try:
        assert orders.main(["--account", str(account), "--list"]) == 0
        assert "NO SERVER IS RUNNING" not in capsys.readouterr().out      # a server's watcher holds the lock: the order is being watched
    finally:
        lock.release()
    assert orders.main(["--account", str(account), "--cancel", str(number)]) == 0
    assert "is cancelled" in capsys.readouterr().out and db.store.open_orders(account) == [] and db.store.available_cash(account) == Decimal("10000.00")
    assert orders.main(["--account", str(account), "--cancel", str(number)]) == 1
    assert "is cancelled: it cannot be cancelled" in capsys.readouterr().err


def test_the_session_tells_the_user_a_waiting_order_is_watched_only_while_the_pages_server_runs(db, stored_close):
    """`session.watcher_note` (the terminal session's last words about a waiting order) and the sentence of each waiting order: the real book's advisory lock says whether a server is running its watcher."""
    from types import SimpleNamespace

    from prometheus.application import session
    account = db.account("10000.00", mode="auto")
    before = db.store.list_portfolio(account)
    order, _ = direct(db, account, "Buy 10 shares of MSFT at $510 or less", msft_buy(), now=eastern(MONDAY, 10, 30))
    assert session.waiting_lines_of(order) == ["Waiting: buy 10 MSFT at $510.00 or less (now $517.99), $5,100.00 set aside, until 4:00 pm ET today."]
    parts = SimpleNamespace(store=db.store)      # `run_session` only asks its parts for the account book here
    assert "NO SERVER IS RUNNING a watcher now" in session.watcher_note(parts)[0]
    lock = db.store.watcher_lock()
    assert lock.try_acquire()
    try:
        assert session.watcher_note(parts) == ["  A waiting order is watched only while the page's server runs (./run_ui.sh)."]
    finally:
        lock.release()
    held = db.store.list_portfolio(account)
    assert before.cash_avail != held.cash_avail and session.changed(before, held) is False      # cash set aside is not a change of the account (the whole cash and the holdings are as they were)
    assert any("set aside" in line for line in session.portfolio_lines(held))
