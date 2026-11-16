"""A DIRECT ORDER with a decimal number of shares ("sell 0.5 shares of MSFT"): design choices entry 110, design/fractional_shares_spec.md section 5.

Without it a holding of less than one share could never be sold, and neither could the fraction left after "sell 81 shares" of 81.004455. REAL DATA ONLY, as `test_direct_order.py` (whose helpers these tests use): the
Account Manager's model is a REPLAY of its REAL replies to four sentences with a decimal number of shares, recorded on 2026-11-15 by ONE live run of `AccountManager.understand_request` (no fetch, no account:
`tests/fixtures/session_real/direct_order_decimal_forms.json`); the symbol directory is Finnhub's recorded answers; the prices are the stored closes of Friday 2026-11-13 through the real Data Clerk; the Trader is the real
Trader with no model; the book is the real local Postgres on throwaway accounts (skipped when it is not reachable or does not hold fractions).
"""
import json
from datetime import date
from decimal import Decimal

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_direct_order import BILL, BILL_PROFILE, FIXTURES, Replay, account, clerk, direct_order, form, typed_words, wanted
from test_trader_handshake import YEAR, db, store  # noqa: F401  (db and store are fixtures)

from prometheus.agents.account_manager import direct
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position, trade_amount

pytestmark = pytest.mark.usefixtures("friday_close")

DECIMALS = json.loads((FIXTURES / "session_real" / "direct_order_decimal_forms.json").read_text(encoding="utf-8"))["requests"]


def reading(typed: str) -> AccountManager:
    return AccountManager(model=Replay(DECIMALS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)


def holding(shares, cash: Decimal = Decimal("1000.00")) -> Portfolio:
    return Portfolio(user="Bill Gates", cash_avail=cash, positions=[Position(symbol="MSFT", shares=shares, cost_basis=Decimal("400"))])


def after(order) -> Portfolio:
    """The account as the Trader left it: the snapshot of its run, as a portfolio."""
    run = order.trader_runs[-1]["portfolio_after"]
    return Portfolio(user="Bill Gates", cash_avail=run["cash"], positions=[Position(**p) for p in run["positions"]])


# ---------------------------------------------------------------- the form: the number of shares may have a decimal part

def test_the_real_models_replies_to_requests_with_a_decimal_number_of_shares_are_orders_the_words_support():
    got = {typed: reading(typed).understand_request(typed) for typed in list(DECIMALS)[:3]}
    assert [[(o.side, o.symbol, o.company, o.shares) for o in i.orders] for i in got.values()] == [[("sell", "MSFT", None, Decimal("0.5"))], [("sell", "DPZ", None, Decimal("81.004455"))],
                                                                                                    [("buy", None, "Microsoft Corp", Decimal("2.25")), ("sell", "TSLA", None, Decimal("0.222222"))]]
    assert all(i.action == "order" and i.said_how_much() and i.describe().endswith("(your own order: no recommendation is made and the Critic is not asked)") for i in got.values())
    assert got["sell 81.004455 shares of DPZ"].describe().startswith("sell 81.004455 shares of DPZ") and "sell 0.5 shares of MSFT" in got["Sell 0.5 shares of MSFT"].describe()
    assert it.HOW_MUCH.search("buy 0.5 shares of MSFT") and type(got["Sell 0.5 shares of MSFT"].orders[0].shares) is Decimal and not sh.whole(got["Sell 0.5 shares of MSFT"].orders[0].shares)
    typed = "Buy 50.5 shares of MSFT"                                                                       # entry 110 amended the old refusal of this: it is an order for 50.5 shares
    assert it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": 50.5}]), typed).orders[0].shares == Decimal("50.5")
    with pytest.raises(ValueError, match="50 shares of MSFT is not what you typed"):
        it.parse_intent(form(orders=[{"side": "buy", "symbol": "MSFT", "shares": 50}]), typed)               # the wrong number is still refused


def test_more_than_six_decimals_is_refused_with_the_reason_and_never_rounded():
    typed = "Sell 0.1234567 shares of MSFT"
    am = reading(typed)
    with pytest.raises(ValueError, match="0.1234567 shares of MSFT has more than 6 decimals: a number of shares has at most 6"):
        am.understand_request(typed)
    assert am.intent_run["calls"] == 2 and am.intent_run["retries"] == 2                                      # the model's own 0.123457 on the second try was refused too: it is not what the user typed
    for shares in (Decimal("0.1234567"), True, "0.5", 0):
        with pytest.raises(ValueError):
            wanted("sell", "MSFT", shares)
    assert wo.verify_direct_orders([{"record_id": "d-1", "side": "sell", "symbol": "MSFT", "company": "Microsoft Corp", "shares": Decimal("0.1234567"), "request": "x", "requested_utc": "2026-11-13T20:30:00Z"}])[0].endswith(
        "or None while the user has given no number and the Account Manager has not sized the order (design choices entry 98)")


# ---------------------------------------------------------------- decided and traded: placed as typed or not at all

def test_a_fraction_left_after_a_sale_can_be_sold_and_a_sale_of_exactly_the_holding_closes_the_position():
    price = close_of("MSFT")
    held = holding(Decimal("81.004455"))
    first, am = direct_order("sell", "MSFT", 81, portfolio=held)
    am.perform_trades(first, held, trader(), MOMENT)
    assert [(t["status"], t["shares"], str(t["shares"])) for t in first.trades] == [("filled", 81, "81")] and [p.shares for p in after(first).positions] == [Decimal("0.004455")]       # 81.004455 less 81 is exactly 0.004455
    left = after(first)
    second, am = direct_order("sell", "MSFT", Decimal("0.004455"), portfolio=left)
    assert [(d["will_trade"], d["shares"], d["est_cost"], d["action"]) for d in second.trade_decisions] == [("yes_user", Decimal("0.004455"), Decimal("0.00"), "reduce")] and wo.verify_trade(second, second.direct_orders[0]["record_id"], MOMENT) == []
    am.perform_trades(second, left, trader(), MOMENT)
    t = second.trades[0]
    assert (t["status"], t["shares"], t["fill"]["shares"]) == ("filled", Decimal("0.004455"), Decimal("0.004455")) and after(second).positions == [] and wo.verify_trades(second) == []        # exactly 0 is left: the position is gone
    assert after(second).cash_avail == left.cash_avail + t["fill"]["amount"] and t["fill"]["amount"] == trade_amount(Decimal("0.004455"), t["fill"]["price"]) and price > 0
    more, _ = direct_order("sell", "MSFT", Decimal("0.004456"), portfolio=left)                                            # one step more than the holding: placed as typed or not at all
    assert [(d["will_trade"], d["shares"]) for d in more.trade_decisions] == [("no_funds", Decimal("0.004456"))] and "the account holds 0.004455 shares of MSFT and you asked to sell 0.004456" in more.trade_decisions[0]["reason"]


def test_a_fraction_of_a_real_holding_is_sold_and_what_is_left_is_a_fraction():
    """"Sell 0.222222 shares of MSFT" from 50 shares held long ago: decided `yes_user`, filled, and the account holds 49.777778 (a fraction: the whole 50 became one)."""
    held = holding(50)
    order, am = direct_order("sell", "MSFT", Decimal("0.222222"), portfolio=held)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["shares"], d["action"], d["origin"]) == ("yes_user", Decimal("0.222222"), "reduce", "direct") and wo.verify_trade(order, order.direct_orders[0]["record_id"], MOMENT) == []
    am.perform_trades(order, held, trader(), MOMENT)
    t = order.trades[0]
    assert (t["status"], t["shares"], t["fill"]["shares"], type(t["fill"]["shares"]).__name__) == ("filled", Decimal("0.222222"), Decimal("0.222222"), "Decimal") and [p.shares for p in after(order).positions] == [Decimal("49.777778")] and wo.verify_trades(order) == []
    assert after(order).cash_avail == held.cash_avail + t["fill"]["amount"] and t["fill"]["amount"] == trade_amount(Decimal("0.222222"), t["fill"]["price"])


def test_a_purchase_of_a_fraction_is_placed_as_typed_when_the_cash_pays_for_all_of_it_and_not_at_all_when_it_does_not():
    price = close_of("MSFT")
    rich = Portfolio(user="Bill Gates", cash_avail=Decimal("3000.00"))
    order, am = direct_order("buy", "MSFT", Decimal("2.25"), portfolio=rich)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["shares"], d["est_cost"], d["action"]) == ("yes_user", Decimal("2.25"), trade_amount(Decimal("2.25"), price), "open") and "buy 2.25 shares of MSFT" in d["reason"]
    am.perform_trades(order, rich, trader(), MOMENT)
    assert order.trades[0]["status"] == "filled" and order.trades[0]["fill"]["shares"] == Decimal("2.25") and [p.shares for p in after(order).positions] == [Decimal("2.25")] and wo.verify_trades(order) == []
    poor = Portfolio(user="Bill Gates", cash_avail=Decimal("1000.00"))
    short, _ = direct_order("buy", "MSFT", Decimal("2.25"), portfolio=poor)
    assert short.trade_decisions[0]["will_trade"] == "no_funds" and f"the most it pays for is {sh.text(sh.buys(Decimal("1000.00"), price))} shares" in short.trade_decisions[0]["reason"]
    nothing, _ = direct_order("buy", "MSFT", Decimal("2.25"), portfolio=Portfolio(user="Bill Gates", cash_avail=Decimal("0.00")))
    assert nothing.trade_decisions[0]["will_trade"] == "no_funds" and "does not pay for any of it" in nothing.trade_decisions[0]["reason"]


def test_an_order_for_a_fraction_on_a_book_that_cannot_hold_one_is_refused_before_anything_is_placed_and_never_rounded():
    order = wo.new_work_order(typed_words("buy", "MSFT", Decimal("0.5")), requester="desk", action_type="trade")
    am = account(BILL_PROFILE)
    assert am.name_the_stocks(order, [wanted("buy", "MSFT", Decimal("0.5"))], typed_words("buy", "MSFT", Decimal("0.5")), MOMENT) == []
    out = direct.decide_direct(order, BILL, BILL_PROFILE, None, am._direct_prices(order), MOMENT, 30, fractions=False)
    d = out.trade_decisions[0]
    assert (d["will_trade"], d["shares"], d["est_cost"]) == ("no_system_denied", 0, Decimal("0.00")) and "holds whole shares only" in d["reason"] and "bash scripts/setup_postgres.sh" in d["reason"] and "not rounded" in d["reason"]
    assert wo.verify_trade(out, out.direct_orders[0]["record_id"], MOMENT) != []                               # not a trade the Trader may place


# ---------------------------------------------------------------- the real Postgres: a fractional buy and sales

def test_a_fractional_holding_is_sold_in_two_orders_through_the_trader_into_the_account_book_and_the_last_leaves_no_position(db, store):
    """The holding is a fractional buy booked on an EARLIER day (a share cannot be sold the day it was bought: T+1), then the Trader sells 0.25 and then the 2 that are left, each booked in one transaction."""
    if not store.holds_fractions():
        pytest.skip("the account book still has integer share columns: run `bash scripts/setup_postgres.sh` (design choices entry 110)")
    account_id, price = db.account("2000.00", name="Bill Gates"), close_of("MSFT")
    db.store.book_trade(account_id, "MSFT", "buy", Decimal("2.25"), price, date(2026, 11, 10), record_id="held-before")
    assert db.held(account_id) == {"MSFT": Decimal("2.25")} and type(db.held(account_id)["MSFT"]) is Decimal
    spent = db.cash(account_id)
    part, am = direct_order("sell", "MSFT", Decimal("0.25"), portfolio=db.store.list_portfolio(account_id), store=db.store, account_id=account_id)
    am.perform_trades(part, None, trader(), MOMENT, book=True)
    assert (part.trades[0]["status"], part.trades[0]["booked"]) == ("filled", True) and db.held(account_id) == {"MSFT": 2} and str(db.held(account_id)["MSFT"]) == "2" and db.cash(account_id) > spent       # 2.25 less 0.25 is exactly 2: a whole quantity again, with no decimals
    rest, am = direct_order("sell", "MSFT", 2, portfolio=db.store.list_portfolio(account_id), store=db.store, account_id=account_id)
    am.perform_trades(rest, None, trader(), MOMENT, book=True)
    assert (rest.trades[0]["status"], rest.trades[0]["booked"]) == ("filled", True) and db.held(account_id) == {} and db.store.number_trades(account_id, *YEAR) == 3 and wo.verify_trades(rest) == [] and rest.errors == []
    assert db.cash(account_id) == Decimal("2000.00") - trade_amount(Decimal("2.25"), price) + part.trades[0]["fill"]["amount"] + rest.trades[0]["fill"]["amount"]


def test_with_the_book_made_to_answer_as_the_old_one_a_decimal_order_is_refused_by_the_real_account_manager_and_nothing_is_written(db, store, monkeypatch):
    monkeypatch.setattr(db.store, "holds_fractions", lambda: False)
    account_id = db.account("2000.00", name="Bill Gates")
    order, am = direct_order("buy", "MSFT", Decimal("0.5"), portfolio=db.store.list_portfolio(account_id), store=db.store, account_id=account_id)
    assert [(d["will_trade"], d["shares"]) for d in order.trade_decisions] == [("no_system_denied", 0)] and "bash scripts/setup_postgres.sh" in order.trade_decisions[0]["reason"]
    am.perform_trades(order, None, trader(), MOMENT, book=True)
    assert order.trades == [] and db.held(account_id) == {} and db.cash(account_id) == 2000 and db.store.number_trades(account_id, *YEAR) == 0
    assert order.errors == [] or all("fraction" not in e for e in order.errors)
