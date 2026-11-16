"""The Account Manager's side of the per-trade handshake (design choices entry 81): `approve_trade`, `preview_trade`, `record_fill`, `apply_earlier_fills`, `current_portfolio` and the two records `Approval` and `Booking`
(prometheus/agents/account_manager/handshake.py).

REAL DATA ONLY: the real Account Manager, the real session's order (`tests/fixtures/session_real/bob_jones_oil.json`: real recommendations, real reviews, the Account Manager's own decisions: XOM and EOG, 8 shares each),
the real sample portfolio, and the stored closes of 2026-11-13 through the real Data Clerk (see `handshake_helpers.py`). Nothing is replaced. The same protocol on the real Postgres is in test_trader_handshake.py.
"""
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, close_of, friday_close, jacob_as_bob, manager, real_order, trader  # noqa: F401  (friday_close is a fixture)
from pydantic import ValidationError

from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO
from prometheus.agents.account_manager.handshake import APPROVAL_CODES, NO_STORE_NOTE, Approval, Booking, decide_approval, unbooked_fills
from prometheus.agents.trader import tickets as tk
from prometheus.agents.trader.config import load_config
from prometheus.agents.trader.record import REJECT_CODES
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import apply_fill, load_portfolio, trade_amount


def tickets_of(order, portfolio=BOB, now=MOMENT):
    """The REAL tickets the Trader builds from the real order: XOM and EOG, 8 shares each (as dicts, the shape `approve_trade` and `record_fill` are given)."""
    return {t.symbol: t.model_dump(mode="json") for t in tk.build_tickets(order, portfolio, now, load_config(), "paper")}


def filled(ticket: dict, price: Decimal, shares: int | None = None, on: date = FRIDAY, **changes) -> dict:
    """The ticket as the Trader reports it once it filled: the same record with a fill."""
    n, price = shares or ticket["shares"], price_door(price)                                                  # a price comes in through the price door (a Decimal passes as it is, a float by its written digits)
    fill = {"price": price, "shares": n, "amount": trade_amount(n, price), "commission": 0.0, "date": on.isoformat(), "price_source": "quote", "slippage_bps": 5.0}
    return {**ticket, "status": "filled", "shares": n, "fill": fill, **changes}


def ask(am, ticket, price, order, **kw):
    return am.approve_trade(ticket, price, order=order, now=kw.pop("now", MOMENT), on=kw.pop("on", FRIDAY), **kw)


# ---------------------------------------------------------------- the two records

def test_the_codes_the_account_manager_gives_are_words_of_the_traders_closed_list_of_reject_codes():
    assert set(APPROVAL_CODES) <= set(REJECT_CODES) and "account_refused" in APPROVAL_CODES and "insufficient_cash" in APPROVAL_CODES


def test_an_approval_and_a_booking_are_frozen_records_that_refuse_unknown_fields():
    ok = Approval(approved=True, shares=8, cash_avail=Decimal("15000"))
    with pytest.raises(ValidationError):
        ok.shares = 9
    with pytest.raises(ValidationError):
        Approval(approved=True, shares=8, colour="red")
    booking = Booking(status="booked", cash_avail=1.0)
    with pytest.raises(ValidationError):
        booking.status = "refused"
    with pytest.raises(ValidationError):
        Booking(status="maybe")


@pytest.mark.parametrize("fields", [{"approved": True, "shares": 0}, {"approved": True, "shares": 3, "code": "insufficient_cash"}, {"approved": False, "shares": 2, "code": "insufficient_cash"},
                                    {"approved": False, "shares": 0}, {"approved": False, "shares": 0, "code": "price_unavailable"}])
def test_an_approval_that_contradicts_itself_is_refused_when_it_is_made(fields):
    with pytest.raises(ValidationError):
        Approval(**fields)


def test_a_booking_says_whether_the_fill_stands_and_whether_it_is_in_the_book():
    assert (Booking(status="booked").ok, Booking(status="booked").booked) == (True, True)
    assert (Booking(status="applied").ok, Booking(status="applied").booked) == (True, False)
    assert (Booking(status="refused").ok, Booking(status="unconfirmed").ok, Booking(status="unconfirmed").booked) == (False, False, False)


# ---------------------------------------------------------------- approve_trade: a purchase

def test_a_purchase_is_approved_for_all_its_shares_when_the_cash_pays_for_them_and_nothing_is_reserved(friday_close):
    order, am = real_order(), manager()
    xom, eog = tickets_of(order)["XOM"], tickets_of(order)["EOG"]
    am.trading_portfolio(BOB)
    price = price_door(close_of("XOM") * Decimal("1.0005"))
    first = ask(am, xom, price, order)
    assert first == Approval(approved=True, shares=8, cash_avail=Decimal("15000"), reason="") and first.code is None
    assert ask(am, eog, price_door(close_of("EOG") * Decimal("1.0005")), order).cash_avail == Decimal("15000")                      # an approval changes nothing: the next one meets the same funds, until a fill is recorded
    assert am.current_portfolio() == BOB


def test_a_purchase_is_cut_to_the_whole_shares_the_available_cash_pays_for():
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("700")}))
    answer = ask(am, xom, Decimal("170.6753"), order)
    assert (answer.approved, answer.shares, answer.cash_avail, answer.code) == (True, 4, Decimal("700"), None) and answer.reason == "cut from 8 to 4 shares for the cash"
    exact = BOB.model_copy(update={"cash_avail": Decimal("1365.4")})                                                      # 8 x Decimal("170.6753") = 1365.40 to the cent: the cash that just pays for the shares buys them
    am.trading_portfolio(exact)
    assert ask(am, xom, Decimal("170.6753"), order).shares == 8
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("1365.39")}))
    assert ask(am, xom, Decimal("170.6753"), order).shares == 7


def test_a_purchase_the_cash_does_not_pay_for_even_one_share_of_is_not_approved_insufficient_cash():
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(jacob_as_bob())     # the real sample: $51.23
    answer = ask(am, xom, Decimal("170.6753"), order)
    assert (answer.approved, answer.shares, answer.code, answer.cash_avail) == (False, 0, "insufficient_cash", Decimal("51.23")) and answer.reason == "cash $51.23 does not pay for one share at 170.6753"


def test_the_approval_never_raises_the_shares_the_decision_gave_and_the_commission_setting_is_counted():
    order, am = real_order(), manager()
    xom = {**tickets_of(order)["XOM"], "shares": 50}                                                           # more asked for than the decision allowed (shares_approved is 8)
    am.trading_portfolio(BOB)
    assert ask(am, xom, Decimal("170.6753"), order).shares == 8
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("1366.4")}))
    assert ask(am, tickets_of(order)["XOM"], Decimal("170.6753"), order, commission=Decimal("1.0")).shares == 8 and ask(am, tickets_of(order)["XOM"], Decimal("170.6753"), order, commission=Decimal("1.01")).shares == 7


# ---------------------------------------------------------------- approve_trade: a sale (the rules of the account: held, and sellable today under T+1)

SALE = {"symbol": "MSFT", "side": "sell", "shares": 10, "shares_approved": 10}


def test_a_sale_is_approved_for_the_shares_held_and_sellable_up_to_the_decisions():
    sample = load_portfolio(DEFAULT_PORTFOLIO)      # the real sample: MSFT 50, bought in 2024
    assert decide_approval(sample, SALE, Decimal("523.88"), FRIDAY) == Approval(approved=True, shares=10, cash_avail=Decimal("51.23"), reason="")
    assert decide_approval(sample, {**SALE, "shares": 80, "shares_approved": 80}, Decimal("523.88"), FRIDAY).shares == 50      # only 50 are held
    assert decide_approval(sample, {**SALE, "shares": 80, "shares_approved": 20}, Decimal("523.88"), FRIDAY).shares == 20      # never more than the decision


def test_a_sale_of_what_is_not_held_or_was_bought_today_is_not_approved_and_the_next_day_it_is():
    held = apply_fill(BOB, "XOM", "buy", 8, Decimal("170.6753"), FRIDAY)                                                    # a REAL fill of the real run, applied: XOM was bought today
    sale = {"symbol": "XOM", "side": "sell", "shares": 8, "shares_approved": 8}
    gone = decide_approval(BOB, sale, Decimal("170.0"), FRIDAY)
    assert (gone.approved, gone.code, gone.reason) == (False, "not_held", "the portfolio does not hold XOM, and there is no short sale")
    today = decide_approval(held, sale, Decimal("170.0"), FRIDAY)
    assert (today.approved, today.code) == (False, "not_sellable_today") and "T+1" in today.reason
    assert decide_approval(held, sale, Decimal("170.0"), FRIDAY + timedelta(days=3)).shares == 8
    older = apply_fill(held, "XOM", "buy", 2, Decimal("171.0"), FRIDAY + timedelta(days=3))                                  # two more bought on a later day lock only those two
    assert decide_approval(older, {**sale, "shares": 10, "shares_approved": 10}, Decimal("170.0"), FRIDAY + timedelta(days=3)).shares == 8


# AMENDED (entry 110): the row with "shares": 2.5 became 0.1234567: 2.5 shares is a trade now (an even split or a decimal the user typed); seven decimals is not
@pytest.mark.parametrize("trade, price", [({"symbol": "XOM", "side": "hold", "shares": 1}, Decimal("1")), ({"symbol": "XOM", "side": "buy", "shares": 0}, Decimal("1")), ({"symbol": "XOM", "side": "buy", "shares": 0.1234567}, Decimal("1")),
                                          ({"symbol": 7, "side": "buy", "shares": 1}, Decimal("1")), ({"symbol": "XOM", "side": "buy", "shares": 1}, 0), ({"symbol": "XOM", "side": "buy", "shares": 1}, None),
                                          ({"symbol": "XOM", "side": "buy", "shares": 1}, Decimal("NaN")), ({"symbol": "XOM", "side": "buy", "shares": 1}, Decimal("Infinity"))])
def test_a_trade_that_is_not_a_trade_is_not_approved_whatever_the_cash(trade, price):
    answer = decide_approval(BOB, trade, price, FRIDAY)
    assert (answer.approved, answer.shares, answer.code) == (False, 0, "account_refused")


# ---------------------------------------------------------------- approve_trade: the decision must still permit it, and a run must be open

def test_the_decision_must_still_permit_the_trade_an_expired_approval_and_a_denied_decision_are_not_approved(friday_close):
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(BOB)
    late = ask(am, xom, Decimal("170.6753"), order, now=datetime(2026, 11, 13, 21, 1, tzinfo=timezone.utc))               # the real approval lasts to 21:00 UTC
    assert (late.approved, late.code) == (False, "approval_expired") and "the approval ran out at 2026-11-13T21:00:00Z" in late.reason
    denied = next(d["record_id"] for d in order.trade_decisions if d["will_trade"] == "no_system_denied")      # APA, denied by the real Critic for risk
    refused = ask(am, {**xom, "record_id": denied}, Decimal("170.6753"), order)
    assert (refused.approved, refused.code) == (False, "not_permitted") and "the Critic did not approve it" in refused.reason
    order.trade_decisions[0]["will_trade"] = "no_user_denied"                                                   # the decision changes after the ticket was made
    changed = ask(am, xom, Decimal("170.6753"), order)
    assert (changed.approved, changed.code) == (False, "not_permitted") and "no_user_denied" in changed.reason


def test_without_an_open_trading_run_nothing_is_approved_or_recorded_and_nothing_raises():
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    answer = ask(am, xom, Decimal("170.6753"), order)
    assert (answer.approved, answer.code) == (False, "account_refused") and "ValueError" in answer.reason
    booking = am.record_fill(filled(xom, Decimal("170.6753")))
    assert booking.status == "refused" and "no trading run is open" in booking.reason
    with pytest.raises(ValueError, match="no trading run is open"):
        am.current_portfolio()
    with pytest.raises(ValueError, match="no trading run is open"):
        am.apply_earlier_fills([])


def test_a_preview_is_a_question_it_counts_the_expected_proceeds_as_cash_for_that_answer_only():
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("51.23")}))
    assert ask(am, xom, Decimal("170.6753"), order).code == "insufficient_cash"
    answer = am.preview_trade(xom, Decimal("170.6753"), order=order, now=MOMENT, on=FRIDAY, expected_proceeds=Decimal("2098.95"))    # what the funding sales would bring
    assert (answer.approved, answer.shares, answer.cash_avail) == (True, 8, Decimal("51.23"))                              # cash_avail is the cash the account has, not the cash plus the question
    assert am.current_portfolio().cash_avail == Decimal("51.23") and ask(am, xom, Decimal("170.6753"), order).code == "insufficient_cash"      # nothing was reserved or changed


# ---------------------------------------------------------------- record_fill, with nothing booked (the Account Manager's working portfolio)

def test_a_fill_is_applied_to_the_working_portfolio_with_apply_fill_and_nothing_is_booked(friday_close):
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    start = am.trading_portfolio(BOB)
    price = price_door(close_of("XOM") * Decimal("1.0005"))
    booking = am.record_fill(filled(xom, price))
    after = apply_fill(BOB, "XOM", "buy", 8, price, FRIDAY)
    assert booking == Booking(status="applied", cash_avail=after.cash_avail, reason="", note="") and not booking.booked and booking.ok
    assert am.current_portfolio() == after and start is BOB and BOB.cash_avail == Decimal("15000") and BOB.positions == []   # the portfolio given is never changed


def test_the_next_approval_meets_the_funds_the_earlier_fill_left_so_two_purchases_cannot_overdraw_the_cash(friday_close):
    order, am = real_order(), manager()
    tickets = tickets_of(order)
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("2000")}))
    xom_price, eog_price = price_door(close_of("XOM") * Decimal("1.0005")), price_door(close_of("EOG") * Decimal("1.0005"))
    first = ask(am, tickets["XOM"], xom_price, order)
    booking = am.record_fill(filled(tickets["XOM"], xom_price, first.shares))
    second = ask(am, tickets["EOG"], eog_price, order)
    assert first.shares == 8 and booking.cash_avail == second.cash_avail == Decimal("2000") - trade_amount(8, xom_price)
    assert second.approved and second.shares == 4 and second.reason == "cut from 8 to 4 shares for the cash"       # what is left ($634.60) pays for 4, not the 8 the decision gave
    am.record_fill(filled(tickets["EOG"], eog_price, second.shares))
    assert 0 <= am.current_portfolio().cash_avail < eog_price                                                   # and it is never below zero
    third = ask(am, tickets["EOG"], eog_price, order)
    assert (third.approved, third.code) == (False, "insufficient_cash")


def test_a_fill_the_portfolio_cannot_take_is_refused_with_its_own_sentence_and_changes_nothing(friday_close):
    order, am = real_order(), manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("1000")}))
    too_dear = am.record_fill(filled(xom, Decimal("170.6753")))                                                           # 8 shares cost $1,365.40: the cash does not cover it
    assert (too_dear.status, too_dear.cash_avail, too_dear.reason) == ("refused", Decimal("1000"), "cash $1,000.00 does not cover $1,365.40")
    assert am.current_portfolio().cash_avail == Decimal("1000") and am.current_portfolio().positions == []
    sale = am.record_fill(filled({**xom, "side": "sell"}, Decimal("170.0")))
    assert (sale.status, sale.reason) == ("refused", "cannot sell 8 of XOM: the portfolio holds 0")


@pytest.mark.parametrize("change, why", [({"status": "active"}, "'active', not filled"), ({"fill": None}, "no fill"), ({"fill": {"price": 1.0, "shares": 1, "date": "soon"}}, "is not a date"),
                                         ({"fill": {"price": 0, "shares": 1, "date": "2026-11-13"}}, "price must be a finite number above zero"), ({"fill": {"price": 1.0, "shares": 0, "date": "2026-11-13"}}, "number above zero")])   # entry 110: the message no longer says "whole"
def test_something_that_is_not_a_usable_fill_is_refused_and_never_applied(change, why):
    am = manager()
    xom = tickets_of(real_order())["XOM"]
    am.trading_portfolio(BOB)
    booking = am.record_fill({**filled(xom, Decimal("170.0")), **change})
    assert booking.status == "refused" and why in booking.reason and am.current_portfolio() == BOB
    assert am.record_fill("junk").status == "refused" and am.record_fill(None).status == "refused"


def test_when_booking_was_asked_for_and_no_account_is_connected_the_first_fill_says_so_once_and_is_applied_in_memory():
    am = manager()
    tickets = tickets_of(real_order())
    am.trading_portfolio(BOB, book=True)
    first = am.record_fill(filled(tickets["XOM"], Decimal("170.0")))
    second = am.record_fill(filled(tickets["EOG"], Decimal("158.0")))
    assert (first.status, first.note) == ("applied", NO_STORE_NOTE) and (second.status, second.note) == ("applied", "") and not first.booked
    am.trading_portfolio(BOB, book=True)                                                                         # a new run says it again
    assert am.record_fill(filled(tickets["XOM"], Decimal("170.0"))).note == NO_STORE_NOTE


def test_a_new_trading_run_forgets_what_the_last_one_left_and_needs_a_portfolio_when_no_account_is_connected():
    am = manager()
    xom = tickets_of(real_order())["XOM"]
    am.trading_portfolio(BOB)
    am.record_fill(filled(xom, Decimal("170.0")))
    assert am.current_portfolio().cash_avail < Decimal("15000")
    assert am.trading_portfolio(BOB) is BOB and am.current_portfolio() == BOB
    with pytest.raises(ValueError, match="no portfolio to trade"):
        am.trading_portfolio(None)


# ---------------------------------------------------------------- the fills of an earlier run that were never booked

def a_real_first_run(friday_close_fixture=None):
    """The real Trader's first run over the real order, on a working portfolio: the trades it leaves on the order (T1 XOM and T2 EOG, filled, not booked)."""
    order = real_order()
    manager().perform_trades(order, BOB, trader(), MOMENT)
    return order


def test_the_fills_of_an_earlier_run_that_were_never_booked_are_applied_to_the_working_portfolio_sales_first(friday_close):
    order = a_real_first_run()
    assert [(t["trade_id"], t["status"], t["booked"]) for t in order.trades] == [("T1", "filled", False), ("T2", "filled", False)]
    am = manager()
    am.trading_portfolio(BOB)
    assert am.apply_earlier_fills(order.trades) == []
    expected = apply_fill(apply_fill(BOB, "XOM", "buy", 8, order.trades[0]["fill"]["price"], FRIDAY), "EOG", "buy", 8, order.trades[1]["fill"]["price"], FRIDAY)
    assert am.current_portfolio() == expected and BOB.cash_avail == Decimal("15000")
    ordered = unbooked_fills([order.trades[1], {**order.trades[0], "side": "sell"}, {"trade_id": "x", "status": "rejected"}, "junk"])
    assert [t["trade_id"] for t in ordered] == ["T1", "T2"]                                                    # the sale first, then the purchase; what is not a fill is left out


def test_a_trade_that_was_booked_is_already_in_the_account_and_is_not_applied_again(friday_close):
    order = a_real_first_run()
    order.trades[0]["booked"] = True
    am = manager()
    am.trading_portfolio(BOB)
    assert am.apply_earlier_fills(order.trades) == []
    assert am.current_portfolio().position("XOM") is None and am.current_portfolio().position("EOG").shares == 8


def test_an_earlier_fill_that_cannot_be_applied_is_a_problem_and_the_others_still_go_through(friday_close):
    order = a_real_first_run()
    order.trades[0]["fill"]["date"] = "never"
    am = manager()
    am.trading_portfolio(BOB)
    problems = am.apply_earlier_fills(order.trades)
    assert problems == ["T1: its earlier fill was not applied to the portfolio: its fill date 'never' is not a date"] and am.current_portfolio().position("EOG").shares == 8
    poor, again = manager(), a_real_first_run()
    poor.trading_portfolio(BOB.model_copy(update={"cash_avail": Decimal("100")}))
    assert poor.apply_earlier_fills(again.trades)[0].startswith("T1: its earlier fill was not applied to the portfolio: cash $100.00 does not cover")


def test_the_handshake_records_nothing_of_the_order_it_is_not_given_and_never_edits_it(friday_close):
    order = real_order()
    before = json.dumps(order.model_dump(mode="json"), sort_keys=True)
    am = manager()
    xom = tickets_of(order)["XOM"]
    am.trading_portfolio(BOB)
    ask(am, xom, Decimal("170.6753"), order)
    am.record_fill(filled(xom, Decimal("170.6753")))
    assert json.dumps(order.model_dump(mode="json"), sort_keys=True) == before
