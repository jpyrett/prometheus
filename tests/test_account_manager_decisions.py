"""The Account Manager's trade decisions from the Critic's verdicts (Critic spec 12.2 and 12.3): every row of the table, the cash across a review, the mode, and the Trader's precondition."""
import copy
import functools
import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from critic_helpers import NOW, captured, make_critic, profile, tamper, view

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.rag.config import REPO_ROOT


@functools.lru_cache(maxsize=None)
def _reviewed(own_key="own", records=1):
    order = captured("PG_open")
    make_critic(order, **({"listed": {"PG": None}} if own_key == "no_own" else {})).review_recommendations(order, view(), NOW)      # "no_own": the real review while the symbol directory cannot be read (`identity_unverified`, which asks for a person: entry 103; `value_not_assessable` no longer does)
    for i in range(1, records):      # AMENDED (entry 105): these were second RECORDS of PG with the action `add`; the Critic reviews recommendations to buy only (and two records of one stock fail its integrity check), so the second purchase is a copy of the real PG record and its real review under another id: the decisions read the review and the record, nothing else
        order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": f"extra{i:07d}"}))
        order.reviews.append(tamper(order.reviews[0], **{"record_id": f"extra{i:07d}", "review_id": f"review{i:07d}"}))
    return order


def reviewed(own_key="own", records=1, **changes):
    """A reviewed order (PG_open, approved at 88 for 44 shares); `changes` edit its first review, as `tamper` does."""
    order = _reviewed(own_key, records).model_copy(deep=True)
    if changes:
        order.reviews[0] = tamper(order.reviews[0], **changes)
    return order


def cash(amount=Decimal("20000")):
    return Portfolio(user="Jacob Pyrett", cash_avail=amount)


def decide(order, p=None, mode=None, approver=None, now=NOW, pf=None):
    return decisions.decide_trades(order, pf or cash(), p or profile(), approver, now, mode)


class Asked:
    """An approver that records the cards it was shown and gives a fixed answer."""

    def __init__(self, answer):
        self.answer, self.cards = answer, []

    def __call__(self, card):
        self.cards.append(card)
        return self.answer


def test_a_traded_order_is_returned_unchanged_as_a_decided_one_is_and_nobody_is_asked_again():
    """Trader v1 spec section 11 (phase 3): `traded` follows `decided`. `decide_trades` returns such an order as it is: no message, no new decision, no question to the user."""
    ask = Asked(True)
    order = decide(reviewed(), approver=ask)
    assert order.status == "decided" and len(ask.cards) == 1
    order.status = "traded"
    before, again = json.dumps(order.model_dump(), sort_keys=True, default=str), Asked(False)
    out = decide(order, approver=again, mode="auto", pf=cash(Decimal("5")))
    assert out is order and out.status == "traded" and out.errors == [] and again.cards == [] and json.dumps(out.model_dump(), sort_keys=True, default=str) == before


# ---------------------------------------------------------------- the table


@pytest.fixture(scope="module")
def sample_prices():
    """The real closes of 2026-11-13 of the sample portfolio's three holdings, from the real stored bars."""
    from datetime import date
    from prometheus.tools.data_clerk import DataClerkTools
    try:
        found = DataClerkTools().daily_bars(["MSFT", "TSLA", "GM"], date(2026, 11, 13), date(2026, 11, 13))["bars"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    if not all(found.get(s) for s in ("MSFT", "TSLA", "GM")):
        pytest.skip("the stored bars do not hold 2026-11-13 for MSFT, TSLA and GM")
    return {s: price_door(found[s][-1]["close"]) for s in found}                  # a stored bar is a float: it comes in through the price door, once


def test_a_cost_the_cash_does_not_cover_is_checked_with_the_holdings_prices_for_raise_funds(sample_prices):
    """`decisions._funds` is the seam to `sizing.funds_check`: holdings in profit that are sellable could raise the money (a note), or could not (`no_funds`). The sizing never gives a cost over the cash, so this is the
    check for a size that was given more money than the portfolio holds."""
    from datetime import date
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    today = date(2026, 11, 13)
    fresh = lambda: {"will_trade": "yes_user", "funds_avail": None, "funds_note": None, "reason": "x"}
    covered, d = Decimal("40"), fresh()
    decisions._funds(d, covered, sample, sample_prices, today)
    assert d == {**fresh(), "funds_avail": True}                                                      # $51.23 covers $40: yes, and no note
    d = fresh()
    decisions._funds(d, Decimal("5000"), sample, sample_prices, today)
    assert (d["will_trade"], d["funds_avail"]) == ("yes_user", True) and d["funds_note"].startswith("needs RaiseFunds: the cash does not cover it, but holdings in profit could (short by $4,948.77;")
    d = fresh()
    decisions._funds(d, Decimal("10000000"), sample, sample_prices, today)
    assert (d["will_trade"], d["funds_avail"]) == ("no_funds", False) and "$51.23 does not cover $10,000,000.00 and no holding in profit could raise the difference" in d["reason"]


# ---------------------------------------------------------------- Ask and Auto


def test_a_prompt_that_fails_leaves_the_decision_pending_and_says_so():
    def broken(card):
        raise EOFError("no terminal")
    order = decide(reviewed(), approver=broken)
    assert order.trade_decisions[0]["will_trade"] == "pending_user" and any("could not be asked about PG (EOFError)" in e for e in order.errors)


def test_the_mode_argument_overrides_the_profile_for_one_run():
    assert decide(reviewed(), p=profile(trade_mode="ask"), mode="auto").trade_decisions[0]["will_trade"] == "yes_auto"
    assert decide(reviewed(), p=profile(trade_mode="auto"), mode="ask", approver=Asked(True)).trade_decisions[0]["mode"] == "ask"
    order = decide(reviewed(), mode="manual")
    assert order.status == "reviewed" and order.trade_decisions == [] and "mode must be one of ask, auto" in order.errors[-1]


# ---------------------------------------------------------------- refusals

def test_a_profile_for_another_user_is_refused_before_anything_is_decided():
    order = decide(reviewed(), p=profile(user="Someone Else"))
    assert order.trade_decisions == [] and order.status == "reviewed" and "is for 'Someone Else' but the portfolio belongs to 'Jacob Pyrett'" in order.errors[-1]


def test_a_clock_without_a_timezone_is_refused():
    from datetime import datetime
    with pytest.raises(ValueError, match="timezone"):
        decide(reviewed(), now=datetime(2026, 11, 5, 10, 30))


# ---------------------------------------------------------------- the user answers later

def test_a_pending_decision_is_answered_later_and_the_order_is_decided_when_none_is_left():
    order = decide(_reviewed(records=2).model_copy(deep=True), pf=cash(Decimal("20000")))
    first, second = [d["record_id"] for d in order.trade_decisions]
    decisions.record_user_decision(order, first, True, NOW + timedelta(minutes=5))
    assert order.trade_decisions[0]["will_trade"] == "yes_user" and order.trade_decisions[0]["asked_user"] and order.status == "reviewed"
    decisions.record_user_decision(order, second, False, NOW + timedelta(minutes=6))
    assert order.trade_decisions[1]["will_trade"] == "no_user_denied" and order.status == "decided"


def test_an_answer_after_the_approval_ran_out_is_no_expired_whatever_it_says():
    order = decide(reviewed())
    decisions.record_user_decision(order, order.trade_decisions[0]["record_id"], True, NOW + timedelta(minutes=61))
    assert order.trade_decisions[0]["will_trade"] == "no_expired" and order.status == "decided"


# ---------------------------------------------------------------- the AccountManager's own part

def manager(**kw):
    return AccountManager(model=object(), **kw)


def test_the_account_manager_decides_from_the_critics_verdict_alone_with_no_mark_of_its_own():
    """AMENDED (entry 104): the profile used to carry an automatic-trading mark (`auto_min_score`) that could make an Auto approval wait for the user; it is gone, and the verdict is the only thing the decision reads."""
    assert manager(profile=profile(trade_mode="auto")).decide_trades(reviewed(), cash(), now=NOW).trade_decisions[0]["will_trade"] == "yes_auto"
    assert manager(profile=profile(trade_mode="auto")).decide_trades(reviewed("no_own"), cash(), now=NOW).trade_decisions[0]["will_trade"] == "no_system_denied"
    order = manager().decide_trades(reviewed(), cash(), approver=Asked(True), now=NOW)
    assert order.trade_decisions[0]["will_trade"] == "yes_user"
    manager().record_user_decision(order, order.trade_decisions[0]["record_id"], False, NOW)
    assert any("no pending decision" in e for e in order.errors)


# ---------------------------------------------------------------- the prices of the holdings: the Account Manager asks the Data Clerk, `decisions.decide_trades` stays pure

def real_clerk(monkeypatch, day=date(2026, 11, 13)):
    """The REAL Data Clerk with USE_LIVE_TRADING=false (design choices entry 76): its quotes are the stored closes of `day`, from the real stored daily bars; Finnhub is not asked and nothing is written. No stand-in."""
    from prometheus.agents.data_clerk import data_clerk_agent as dca
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    monkeypatch.setattr(dca, "market_day", lambda: day)
    return dca.DataClerk(model=object())


def spying(clerk):
    """The same Data Clerk, with a note of each quote order it is given (a spy around the real call, not a replacement for it)."""
    asked, original = [], clerk.request_quotes
    clerk.request_quotes = lambda order: asked.append((order.requester, order.action_type, list(order.subject))) or original(order)
    return asked


def test_a_portfolio_of_only_cash_and_an_order_with_nothing_approved_ask_for_no_price(monkeypatch):
    clerk = real_clerk(monkeypatch)
    asked = spying(clerk)
    cash_only = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(), cash(), approver=Asked(True), now=NOW)
    assert asked == [] and cash_only.trade_decisions[0]["size"]["equity_basis"] == "cash only (no holdings)"
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    denied = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(approved=False), sample, approver=Asked(True), now=NOW)
    assert asked == [] and denied.trade_decisions[0]["will_trade"] == "no_system_denied"                                    # nothing to size: no quote is bought for it
    waiting = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(), sample, approver=Asked(True), now=NOW + timedelta(minutes=61))
    assert asked == [] and waiting.trade_decisions[0]["will_trade"] == "no_expired"                                          # an approval that ran out is not sized, so no quote is asked for it either


