"""The Account Manager's trade decisions from the Critic's verdicts (Critic spec 12.2 and 12.3): every row of the table, the cash across a review, the mode, and the Trader's precondition."""
import copy
import functools
import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from critic_helpers import NOW, FakeClerk, captured, make_critic, profile, tamper, view

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import work_order as wo
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, Position, load_portfolio
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


def test_an_approval_that_has_run_out_is_not_traded():
    ask = Asked(True)
    order = decide(reviewed(), approver=ask, now=NOW + timedelta(minutes=61))
    assert order.trade_decisions[0]["will_trade"] == "no_expired" and "ran out at 2026-11-05T16:30:00Z" in order.trade_decisions[0]["reason"] and ask.cards == []
    assert decide(reviewed(), approver=Asked(True), now=NOW + timedelta(minutes=60)).trade_decisions[0]["will_trade"] == "yes_user"       # the last instant is inside
    assert decide(reviewed(valid_until_utc="soon"), now=NOW).trade_decisions[0]["will_trade"] == "no_expired"                              # an approval with no readable end is not valid


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


def test_a_review_that_asked_for_a_person_is_denied_so_in_auto_mode_nobody_is_asked_and_nothing_is_traded():
    """AMENDED (entry 104). Before, this review was APPROVED with a finding that asked for a person, and in Auto mode the Account Manager asked the user (`pending_user`, `escalated`). Now the Critic denies it
    (`needs_human_review`, entry 103), and the Account Manager's row is `no_system_denied` whatever the mode: the reason is the Critic's own summary, and the approver is never called."""
    asked = Asked(True)
    order = decide(reviewed("no_own"), p=profile(trade_mode="auto"), approver=asked)
    d = order.trade_decisions[0]
    assert order.reviews[0]["denial_reasons"] == ["needs_human_review"] and d["will_trade"] == "no_system_denied" and not d["asked_user"] and asked.cards == []
    assert d["reason"] == order.reviews[0]["summary"] and d["reason"].startswith("DENY PG: a person must look before this can trade: identity_unverified") and "escalated" not in d
    ask_mode = Asked(True)
    assert decide(reviewed("no_own"), p=profile(trade_mode="ask"), approver=ask_mode).trade_decisions[0]["will_trade"] == "no_system_denied" and ask_mode.cards == []      # in Ask mode too: a denial is final


def test_in_auto_mode_an_approved_purchase_trades_with_the_critics_reason_and_nobody_is_asked_at_any_score():
    """Entry 104: no second mark. Auto mode trades what the Critic approved, at the score it approved (PG_open's 88 here, and a review that approved at 70, the Critic's own mark, the same)."""
    for changes in ({}, {"score": 70, "grade": "C"}):
        asked = Asked(False)
        d = decide(reviewed(**changes), p=profile(trade_mode="auto"), approver=asked).trade_decisions[0]
        assert d["will_trade"] == "yes_auto" and not d["asked_user"] and asked.cards == [] and d["shares"] > 0
        assert d["reason"].startswith(f"approved by the Critic (score {changes.get('score', 88)}): Auto mode trades what the Critic approves") and "escalated" not in d


def test_a_recommendation_to_reduce_is_never_traded_or_asked_about_in_either_mode():
    """AMENDED (entry 104; the sale rows of the old table are gone). The Advisor recommends purchases only and a sale never comes from a recommendation (Jacob: *"The Advisor never recommends reducing a holding"*): an
    approved review whose action is `reduce` (a stored one, or a tampered one: this is the real PG_open review with the action changed) has nothing to trade. The sample portfolio holds 60 shares of GM and 123 of TSLA, and
    neither is sold: no shares, no card, no `yes`."""
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    for mode, symbol in (("auto", "GM"), ("ask", "TSLA")):
        order, ask = reviewed(action="reduce", symbol=symbol), Asked(True)
        d = decide(order, p=profile(trade_mode=mode), approver=ask, pf=sample).trade_decisions[0]
        assert (d["will_trade"], d["shares"], d["est_cost"], d["funds_avail"], d["size"]) == ("no_funds", 0, 0, False, None) and ask.cards == []
        assert "a reduce recommendation is not a purchase and is never traded" in d["reason"] and wo.verify_trade(order, d["record_id"], NOW) != []


def test_a_recommendation_to_reduce_a_symbol_the_portfolio_does_not_hold_is_not_traded_either():
    d = decide(reviewed(action="reduce"), approver=Asked(True)).trade_decisions[0]                   # PG, in a portfolio of cash only
    assert (d["will_trade"], d["shares"], d["funds_avail"]) == ("no_funds", 0, False) and "is not a purchase and is never traded" in d["reason"]


def test_the_mode_argument_overrides_the_profile_for_one_run():
    assert decide(reviewed(), p=profile(trade_mode="ask"), mode="auto").trade_decisions[0]["will_trade"] == "yes_auto"
    assert decide(reviewed(), p=profile(trade_mode="auto"), mode="ask", approver=Asked(True)).trade_decisions[0]["mode"] == "ask"
    order = decide(reviewed(), mode="manual")
    assert order.status == "reviewed" and order.trade_decisions == [] and "mode must be one of ask, auto" in order.errors[-1]


# ---------------------------------------------------------------- the cash: nothing is reserved (Jacob, 2026-11-14: "any available funds in the portfolio can be used")

def _two_purchases():
    return _reviewed(records=2).model_copy(deep=True)


def with_holdings(amount):
    """`amount` in cash and $1,000 of a holding (50 shares of GM at a cost of $20): an equity over the cash,."""
    return Portfolio(user="Jacob Pyrett", cash_avail=amount, positions=[Position(symbol="GM", shares=50, cost_basis=Decimal("20.0"), acquired=date(2026, 1, 5))])


def test_two_purchases_share_the_cash_a_share_at_a_time_and_no_cash_is_reserved_for_either():
    """AMENDED (entry 106). With no split asked each of the two purchases used to be sized against ALL the cash (5 shares each of the $1,000: $1,671 in all, more than the account holds) and the Trader met the cash at each trade.
    The money is spread round-robin in both modes: one share of each in turn. By hand at 167.10: a round of both costs $334.20, two rounds $668.40, a third round $1,002.60 is over the $1,000, so the first
    takes one more share and the split is 3 and 2 ($835.50 together). Nothing is reserved: each one's caps still say the cash alone would allow 5."""
    first, second = decide(_two_purchases(), pf=with_holdings(Decimal("1000")), approver=Asked(True)).trade_decisions
    assert [d["will_trade"] for d in (first, second)] == ["yes_user", "yes_user"]
    assert (first["shares"], second["shares"]) == (3, 2) and first["est_cost"] + second["est_cost"] == Decimal("835.50") <= Decimal("1000") and first["size"]["allocation"] == second["size"]["allocation"] == "round_robin"
    assert first["size"]["caps"]["money"] == second["size"]["caps"]["money"] == 5 == Decimal("1000") // Decimal("167.1")
    short = decide(_two_purchases(), pf=with_holdings(Decimal("100")), approver=Asked(True))
    assert [d["will_trade"] for d in short.trade_decisions] == ["no_funds", "no_funds"] and all("not even one share fits" in d["reason"] for d in short.trade_decisions)


# ---------------------------------------------------------------- refusals

def test_a_profile_for_another_user_is_refused_before_anything_is_decided():
    order = decide(reviewed(), p=profile(user="Someone Else"))
    assert order.trade_decisions == [] and order.status == "reviewed" and "is for 'Someone Else' but the portfolio belongs to 'Jacob Pyrett'" in order.errors[-1]


def test_an_order_that_is_not_reviewed_is_refused_and_a_decided_one_is_returned_as_it_is():
    fresh = captured("PG_open")
    assert decide(fresh).errors[-1].endswith("(status recommended)") and fresh.trade_decisions == []
    done = decide(reviewed(), approver=Asked(True))
    before = copy.deepcopy(done.trade_decisions)
    assert decide(done, approver=Asked(False)).trade_decisions == before


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


def test_answering_a_decision_that_is_not_pending_is_a_message_not_a_change():
    order = decide(reviewed(), approver=Asked(True))
    before = copy.deepcopy(order.trade_decisions)
    decisions.record_user_decision(order, order.trade_decisions[0]["record_id"], False, NOW)
    decisions.record_user_decision(order, "no-such-record", True, NOW)
    assert order.trade_decisions == before and sum("no pending decision" in e for e in order.errors) == 2


# ---------------------------------------------------------------- the Trader's precondition reads these decisions


# ---------------------------------------------------------------- the AccountManager's own part

def manager(**kw):
    return AccountManager(model=object(), **kw)


def test_a_profile_file_that_is_wrong_is_a_value_error_naming_it(tmp_path):
    (tmp_path / "p.json").write_text('{"user": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="p.json"):
        manager(profile=tmp_path / "p.json").profile


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


def test_the_account_manager_asks_the_data_clerk_once_for_the_prices_of_what_the_portfolio_holds_and_sizes_on_that_equity(monkeypatch, sample_prices):
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    clerk = real_clerk(monkeypatch)
    asked = spying(clerk)
    order = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(), sample, approver=Asked(True), now=NOW)
    assert asked == [("account_manager", "fetch_quote", ["MSFT", "TSLA", "GM"])] and order.errors == []                  # ONE quote order, for the holdings only (PG is not held)
    size = order.trade_decisions[0]["size"]
    assert size["equity"] == Decimal("51.23") + 50 * sample_prices["MSFT"] + 123 * sample_prices["TSLA"] + 60 * sample_prices["GM"] and size["equity_basis"] == "cash and holdings at the prices given"
    assert order.trade_decisions[0]["will_trade"] == "no_funds" and size["caps"]["money"] == 0                              # the real equity is large and the sample's $51.23 buys no share of PG


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


def test_a_manager_with_no_clerk_and_a_portfolio_with_holdings_counts_them_at_cost_basis_and_says_so():
    """The conftest guard makes a Data Clerk that was never given fail the way a missing key does; the Account Manager meets it as it meets any outage: a message on the order and the cost basis."""
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    order = manager(profile=profile()).decide_trades(reviewed(), sample, approver=Asked(True), now=NOW)
    assert any("the current prices of the holdings could not be had (RuntimeError)" in e and "cost basis" in e for e in order.errors)
    size = order.trade_decisions[0]["size"]
    assert size["equity"] == Decimal("51.23") + 50 * Decimal("150") + 123 * Decimal("300") + 60 * Decimal("10") and size["equity_basis"].endswith("MSFT, TSLA, GM at cost basis (no price)") and order.status == "decided"


def test_a_price_the_data_clerk_cannot_give_is_a_message_and_that_holding_counts_at_cost_basis(monkeypatch):
    clerk = real_clerk(monkeypatch, day=date(2026, 11, 14))                              # a Saturday: no daily bar is stored for it, so no quote can be had
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    order = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(), sample, approver=Asked(True), now=NOW)
    assert any(e.startswith("no current price for MSFT, TSLA, GM: counted at cost basis") for e in order.errors)
    assert order.trade_decisions[0]["size"]["equity_basis"].endswith("MSFT, TSLA, GM at cost basis (no price)")


