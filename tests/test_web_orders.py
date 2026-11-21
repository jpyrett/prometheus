"""The waiting limit orders on the web bridge (design choices entry 140; design/limit_orders_spec.md W5 and W6): the account panel's list, the Cancel route, the page's notices, and the WATCHER that is tied to the app's lifespan and that
no test starts unless it asks for one. REAL local Postgres on THROWAWAY accounts; the real market calendar's sessions for the fake clock; the real Data Clerk's STORED closes for the quotes. Skipped when the book is not reachable.
"""
import json
import threading
import time
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from open_orders_helpers import MONDAY, clerk, db, eastern, stored_close, store, trader  # noqa: F401  (db, store and stored_close are fixtures)

from prometheus.application.web.app import create_app
from prometheus.application.web.watch import Notices, OrderWatch

NOW = eastern(MONDAY, 10, 30)


@pytest.fixture
def web(db):
    return TestClient(create_app())


def waiting(db, account_id, n=1, shares=10, limit="510"):
    return db.store.place_open_order(account_id, request_id="webreq", record_id=f"direct-webreq-{n}", symbol="MSFT", side="buy", limit_price=Decimal(limit), request="Buy 10 shares of MSFT at $510 or less", placed_at=NOW,
                                     expires_at=eastern(MONDAY, 16, 0), shares=Decimal(shares), company="Microsoft Corp")


def test_the_account_panel_lists_the_waiting_orders_and_the_cash_they_hold(db, web):
    account = db.account("10000.00", mode="ask")
    order = waiting(db, account)
    listed = web.get(f"/api/accounts/{account}/orders").json()
    assert listed["reserved_cash"] == 5100 and listed["available"] == 4900 and listed["watching"] is False
    assert [(o["order_id"], o["symbol"], o["side"], o["shares"], o["limit_price"], o["reserved_cash"], o["status"], o["expires_utc"]) for o in listed["orders"]] == [(order.order_id, "MSFT", "buy", 10, 510, 5100, "open", "2026-11-16T21:00:00Z")]
    view = web.get(f"/api/accounts/{account}").json()
    assert (view["cash"], view["available"], view["reserved_cash"], view["invested"], view["total"]) == (10000, 4900, 5100, 0, 10000)      # the cash is all of it, and the part that may be spent is told apart
    refused = web.post(f"/api/accounts/{account}/funds", json={"action": "remove", "amount": "9000.00"})
    assert refused.status_code == 422 and "$5,100.00 of it is set aside for waiting orders: only $4,900.00 can be removed" in refused.json()["error"]


def test_a_holding_shows_the_shares_a_waiting_sale_has_set_aside(db, web):
    account = db.account("10000.00")
    db.buy(account, "MSFT", 4, "100.00")
    db.store.place_open_order(account, request_id="webreq", record_id="direct-webreq-9", symbol="MSFT", side="sell", limit_price=Decimal("600"), request="Sell 3 shares of MSFT at $600 or more", placed_at=NOW, expires_at=eastern(MONDAY, 16, 0), shares=Decimal(3))
    view = web.get(f"/api/accounts/{account}").json()
    assert [(p["symbol"], p["shares"], p["reserved"]) for p in view["positions"]] == [("MSFT", 4, 3)] and view["cash"] == 9600


def test_cancel_releases_what_the_order_set_aside_and_tells_the_page(db, web):
    account = db.account("10000.00")
    order = waiting(db, account)
    cancelled = web.delete(f"/api/orders/{order.order_id}")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled" and cancelled.json()["order_id"] == order.order_id
    assert web.get(f"/api/accounts/{account}/orders").json() == {"orders": [], "reserved_cash": 0, "available": 10000, "watching": False}
    notices = web.get("/api/orders/events?once=true")
    assert notices.status_code == 200 and notices.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[len("data: "):]) for line in notices.text.splitlines() if line.startswith("data: ")]
    assert [(e["type"], e["order_id"], e["symbol"]) for e in events] == [("cancelled", order.order_id, "MSFT")] and "available again" in events[0]["text"]
    again = web.delete(f"/api/orders/{order.order_id}")
    assert again.status_code == 409 and "cancelled" in again.json()["error"]
    assert web.delete("/api/orders/999999999").status_code == 404


def test_a_sample_account_is_read_only_and_a_replay_has_no_orders_to_change(db, store):
    from prometheus.application.web.replay import Replay, load_capture
    from pathlib import Path
    played = TestClient(create_app(replay=Replay(load_capture(Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"), speed=0)))
    assert played.get("/api/accounts/1/orders").json() == {"orders": [], "reserved_cash": 0, "available": 0, "watching": False}
    assert played.delete("/api/orders/1").status_code == 409


def test_no_watcher_is_started_by_an_app_that_was_not_given_one_and_one_that_is_given_starts_and_stops_with_the_app(db, stored_close):
    names = lambda: [t.name for t in threading.enumerate() if t.name == "open-orders-watcher"]
    before = len(names())
    with TestClient(create_app()):
        assert len(names()) == before      # a test client's app has no watcher unless the test builds one
    account = db.account("10000.00", mode="auto")
    waiting(db, account)
    watch = OrderWatch(db.store, clerk=clerk(), clock=lambda: eastern(MONDAY, 16, 0), trader_factory=trader, seconds=1, accounts=[account])      # a fake clock at the close: the order is stale and the first tick expires it
    app = create_app(watch=watch)
    assert len(names()) == before      # building the app starts nothing
    with TestClient(app) as running:
        deadline = time.monotonic() + 15
        while db.store.open_orders(account) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(names()) == before + 1 and db.store.open_orders(account) == []      # the thread runs while the app runs, and its first check expired what was stale
        assert running.get(f"/api/accounts/{account}/orders").json()["watching"] is True      # the page can tell that a watcher holds the lock
        events = [json.loads(line[len("data: "):]) for line in running.get("/api/orders/events?once=true").text.splitlines() if line.startswith("data: ")]
        assert [e["type"] for e in events] == ["expired"]      # the page is told
    assert len(names()) == before and db.store.watcher_running() is False      # stopped with the app, and the advisory lock is let go


def test_the_notices_are_numbered_kept_in_order_and_read_after_the_last_one_seen():
    notices = Notices(keep=3)
    for n in range(5):
        notices.add({"type": "filled", "order_id": n})
    batch, over = notices.read_from(0)
    assert [e["seq"] for e in batch] == [3, 4, 5] and over is False and [e["data"]["order_id"] for e in batch] == [2, 3, 4]
    assert [e["seq"] for e in notices.read_from(4)[0]] == [5] and notices.read_from(5)[0] == []


def test_a_run_that_places_a_waiting_order_ends_with_its_sentence_as_the_last_message():
    """`views.run_verdict`: a direct order that now waits is not "the portfolio changed" (the cash set aside is not a change of the account) and is not "nothing was traded": its last message is the sentence."""
    from prometheus.application.web import views
    from prometheus.common.work_order import new_work_order
    order = new_work_order("Buy 10 shares of MSFT at $510 or less", requester="desk", action_type="trade")
    order.trade_decisions = [{"record_id": "direct-x-1", "symbol": "MSFT", "will_trade": "no_system_denied", "open_order_id": 7, "reason": "Waiting: buy 10 MSFT at $510.00 or less (now $517.99), $5,100.00 set aside, until 4:00 pm ET today."}]
    account = {"cash": Decimal("10000.00"), "positions": []}
    verdict = views.run_verdict(account, {**account}, action="order", order=order)
    assert verdict == {"status": "waiting", "headline": "Waiting: buy 10 MSFT at $510.00 or less (now $517.99), $5,100.00 set aside, until 4:00 pm ET today.", "reasons": []}
