"""A WAITING LIMIT ORDER THE CASH DOES NOT COVER IS PAID FOR BY SELLING WHEN ITS PRICE IS REACHED (design choices entry 143, W-F1 to W-F5; Jacob: "Sell when the price is reached").

"Buy 2 shares of ANT at $85 or less and raise funds if needed" on a THROWAWAY copy of Jacob's real account 6553 (cash $60.14, twelve holdings), the REAL Finnhub quotes of 2026-11-20 (ANT $88.91, so the order WAITS; its recorded day LOW, $83.05, is the price
"later in the session" that the watcher meets), the REAL replies of the Account Manager's model (the request's form, and RaiseFunds' thought generator recorded by the content of the question), the real Trader with no model and the real Postgres book.
What the tests pin: it waits with the cash that is FREE set aside and NOTHING is sold while it waits; when the price is reached the watcher raises the shortfall at THAT price with RaiseFunds, sells first and buys after, one trade at a time, and
the order is filled in the book's own transaction; it needs the user's words (or ALWAYS_RAISE_FUNDS), never a question (nobody is there to approve a plan at that moment); and an order that expires sold nothing.
"""
from datetime import date
from decimal import Decimal

import open_orders_helpers as OH
import pytest
import raise_funds_helpers as H
from raise_funds_helpers import ANT, CASH, WAITING, WAITING_PLAIN, Cards, Model
from test_raise_funds_direct import DPZ, Book, bookings, db, store  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager import funding
from prometheus.agents.account_manager.orders import Watcher
from prometheus.common import work_order as wo
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import trade_amount

LOW = price_door(Decimal(str(H.QUOTES["ANT"]["low"])))      # $83.05, the recorded day low of ANT
COST_LOW = trade_amount(2, LOW)                               # $166.10
LIMIT_COST = trade_amount(2, Decimal("85.00"))                # $170.00: the most it can cost
MONDAY = date(2026, 11, 23)                                   # the order was placed after Friday's close: it is for Monday's session
SENTENCE = ("Waiting: buy 2 ANT at $85.00 or less (now $88.91), $60.14 set aside, until 4:00 pm ET Monday 2026-11-23. If the price is reached I will sell holdings to pay for the rest "
            "(up to $109.86 at your limit; you said to raise funds) and then buy; nothing is sold while it waits.")


def place(db, mode="auto", typed=WAITING, **options):
    account_id = db.account(mode, **options.pop("account", {}))
    model = Model(typed)
    state = H.run_desk(db.store, account_id, typed, approver=Cards(), model=model, **options)
    return account_id, state, model


def watcher(db, account_id, model, tmp_path, quotes=None):
    return Watcher(db.store, clerk=H.clerk(quotes), trader_factory=lambda: H.trader(), model=model, accounts=[account_id], audit_path=tmp_path / "audit.jsonl")


def test_it_waits_with_the_free_cash_set_aside_and_nothing_is_sold_while_it_waits(db):
    account_id, state, _ = place(db)
    order = state["order"]
    (decision,) = order.trade_decisions
    assert state["path"] == ["account_manager", "account_manager_direct"] and order.trades == [] and decision["will_trade"] == "no_system_denied" and decision["reason"] == SENTENCE      # nothing trades now, and no funds node
    assert LIMIT_COST - CASH == Decimal("109.86") and "up to $109.86 at your limit" in decision["reason"] and decision["open_order_id"] is not None
    (waiting,) = db.store.open_orders(account_id)
    assert (waiting.symbol, waiting.shares, waiting.limit_price, waiting.reserved_cash, waiting.raise_funds_by) == ("ANT", 2, Decimal("85.0000"), CASH, "words")
    assert db.cash(account_id) == CASH and db.store.available_cash(account_id) == Decimal("0.00") and db.held(account_id) == {p["symbol"]: Decimal(str(p["shares"])) for p in H.ACCOUNT["positions"]}      # the free cash is set aside; not one share sold


def test_a_tick_before_the_price_is_reached_sells_nothing(db, tmp_path):
    account_id, _, model = place(db)
    events = watcher(db, account_id, model, tmp_path).tick(OH.eastern(MONDAY, 11))                   # ANT is still $88.91
    assert events == [] and model.asked == [] and db.cash(account_id) == CASH and "ANT" not in db.held(account_id) and db.held(account_id)["DPZ"] == Decimal("4.389558")
    assert [o.status for o in db.store.open_orders(account_id)] == ["open"]


def test_when_the_price_is_reached_the_watcher_raises_the_shortfall_at_that_price_sells_first_buys_after_and_fills_the_order(db, tmp_path, monkeypatch):
    account_id, _, model = place(db)
    log = bookings(monkeypatch, db.store)
    events = watcher(db, account_id, model, tmp_path, H.FALLEN).tick(OH.eastern(MONDAY, 11))      # ANT is at its recorded day low
    (event,) = events
    assert event["type"] == "filled" and f"bought 2 ANT at ${LOW:,.2f} (${COST_LOW:,.2f})" in event["text"] and len(model.asked) >= 1      # RaiseFunds' model was asked, at this price
    (done,) = db.store.open_orders(account_id, ("filled",))
    assert done.raise_funds_by == "words" and db.store.open_orders(account_id) == []
    assert [(side, symbol) for side, symbol, _ in log] == [("sell", "DPZ"), ("buy", "ANT")]               # the sale is booked first, and then the purchase, which fills the order in its own transaction
    assert db.cash(account_id) == CASH + DPZ - COST_LOW == Decimal("242.65") and db.held(account_id)["ANT"] == 2 and db.held(account_id)["DPZ"] == Decimal("3.389558") and db.store.available_cash(account_id) == db.cash(account_id)
    assert not any(e.startswith("T1-F1") for e in watcher(db, account_id, model, tmp_path).errors)


def test_an_order_that_expires_sold_nothing_and_what_it_set_aside_is_free_again(db, tmp_path):
    account_id, _, model = place(db)
    events = watcher(db, account_id, model, tmp_path).tick(OH.eastern(MONDAY, 16, 1))                 # after Monday's close
    assert [e["type"] for e in events] == ["expired"] and model.asked == []
    assert db.cash(account_id) == CASH and db.store.available_cash(account_id) == CASH and "ANT" not in db.held(account_id) and db.held(account_id)["DPZ"] == Decimal("4.389558")


@pytest.mark.parametrize("mode", ["auto", "ask"])
def test_without_the_words_it_is_refused_for_want_of_cash_as_before_and_says_what_to_add_and_nothing_is_asked(db, mode):
    account_id = db.account(mode)
    cards = Cards()
    state = H.run_desk(db.store, account_id, WAITING_PLAIN, approver=cards, model=Model(WAITING_PLAIN))
    (decision,) = state["order"].trade_decisions
    assert cards.cards == [] and decision["will_trade"] == "no_funds" and "the cash available is $60.14" in decision["reason"] and funding.NOT_SAID in decision["reason"]      # nobody is there at the moment of the sale: only the words permit it
    assert "waiting" not in decision and db.store.open_orders(account_id) == [] and db.cash(account_id) == CASH


def test_the_setting_always_raise_funds_is_the_same_as_the_words(db, monkeypatch):
    monkeypatch.setenv("ALWAYS_RAISE_FUNDS", "true")
    account_id, state, _ = place(db, typed=WAITING_PLAIN)
    (decision,) = state["order"].trade_decisions
    assert "the setting ALWAYS_RAISE_FUNDS is on" in decision["reason"] and [o.raise_funds_by for o in db.store.open_orders(account_id)] == ["setting"]


def test_when_the_holdings_could_not_pay_for_it_at_its_limit_it_is_refused_and_names_what_they_could_raise(db):
    account_id = db.account("auto", drop=("DPZ", "MCD"))                                                  # the only two holdings that show a profit at these quotes
    state = H.run_desk(db.store, account_id, WAITING, approver=Cards(), model=Model(WAITING))
    (decision,) = state["order"].trade_decisions
    assert decision["will_trade"] == "no_funds" and "raising funds could not pay for it" in decision["reason"] and "$0.00 could be raised from holdings in profit" in decision["reason"] and db.store.open_orders(account_id) == []


def test_an_account_with_no_free_cash_at_all_still_waits_holding_nothing_and_the_fill_sells_and_buys(db, tmp_path):
    account_id, state, model = place(db, account={"cash": Decimal("0.00")})
    (decision,) = state["order"].trade_decisions
    (waiting,) = db.store.open_orders(account_id)
    assert waiting.reserved_cash == Decimal("0.00") and waiting.raise_funds_by == "words" and "$0.00 set aside" in decision["reason"] and "up to $170.00 at your limit" in decision["reason"]      # the book takes an order that holds nothing, because it carries the permission
    (event,) = watcher(db, account_id, model, tmp_path, H.FALLEN).tick(OH.eastern(MONDAY, 11))
    assert event["type"] == "filled" and db.held(account_id)["ANT"] == 2 and db.cash(account_id) >= 0


def test_the_book_refuses_the_permission_on_a_sale_and_an_unknown_one_and_still_refuses_an_unfunded_order_the_cash_does_not_cover(db):
    account_id = db.account("auto")
    start = OH.eastern(date(2026, 11, 20), 21, 10)
    common = dict(request_id="r1", record_id="rec1", symbol="ANT", limit_price=Decimal("85.00"), request="x", placed_at=start, expires_at=OH.eastern(MONDAY, 16))
    with pytest.raises(ValueError, match="words or setting, and only for a purchase"):
        db.store.place_open_order(account_id, side="sell", shares=1, raise_funds_by="words", **common)
    with pytest.raises(ValueError, match="words or setting"):
        db.store.place_open_order(account_id, side="buy", shares=2, raise_funds_by="asked", **common)
    with pytest.raises(ValueError, match="would set aside"):
        db.store.place_open_order(account_id, side="buy", shares=2, **common)                              # no permission: refused for want of cash, as it always was
    assert db.store.open_orders(account_id) == []
