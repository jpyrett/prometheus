"""The Account Manager's trade decisions from the Critic's verdicts (Critic spec 12.2 and 12.3): every row of the table, the cash across a review, the mode, and the Trader's precondition."""
import copy
import functools
import json
from datetime import date, timedelta

import pytest
from critic_helpers import NOW, FakeClerk, captured, make_critic, profile, tamper, view

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position, load_portfolio
from prometheus.rag.config import REPO_ROOT

MARK = 80


@functools.lru_cache(maxsize=None)
def _reviewed(own_key="own", records=1):
    order = captured("PG_open")
    for i in range(1, records):
        order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": f"extra{i:07d}", "action.action": "add"}))
    make_critic(order, **({"own": {}} if own_key == "no_own" else {})).review_recommendations(order, view(), NOW)
    return order


def reviewed(own_key="own", records=1, **changes):
    """A reviewed order (PG_open, approved at 88 for 44 shares); `changes` edit its first review, as `tamper` does."""
    order = _reviewed(own_key, records).model_copy(deep=True)
    if changes:
        order.reviews[0] = tamper(order.reviews[0], **changes)
    return order


def cash(amount=20000.0):
    return Portfolio(user="Jacob Pyrett", cash_avail=amount)


def decide(order, p=None, mode=None, approver=None, now=NOW, pf=None, mark=MARK):
    return decisions.decide_trades(order, pf or cash(), p or profile(), mark, approver, now, mode)


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
    out = decide(order, approver=again, mode="auto", pf=cash(5.0))
    assert out is order and out.status == "traded" and out.errors == [] and again.cards == [] and json.dumps(out.model_dump(), sort_keys=True, default=str) == before


# ---------------------------------------------------------------- the table

def test_a_denial_is_final_and_the_user_is_never_asked_to_override_it():
    ask = Asked(True)
    order = decide(reviewed(approved=False, summary="DENY PG: the market is closed (market_closed)."), approver=ask)
    [d] = order.trade_decisions
    assert (d["will_trade"], d["reason"]) == ("no_system_denied", "DENY PG: the market is closed (market_closed).") and ask.cards == [] and order.status == "decided" and d["asked_user"] is False


def test_an_approval_that_has_run_out_is_not_traded():
    ask = Asked(True)
    order = decide(reviewed(), approver=ask, now=NOW + timedelta(minutes=61))
    assert order.trade_decisions[0]["will_trade"] == "no_expired" and "ran out at 2026-11-05T16:30:00Z" in order.trade_decisions[0]["reason"] and ask.cards == []
    assert decide(reviewed(), approver=Asked(True), now=NOW + timedelta(minutes=60)).trade_decisions[0]["will_trade"] == "yes_user"       # the last instant is inside
    assert decide(reviewed(valid_until_utc="soon"), now=NOW).trade_decisions[0]["will_trade"] == "no_expired"                              # an approval with no readable end is not valid


def test_without_the_money_a_purchase_sizes_to_no_share_and_is_not_traded_or_asked_about():
    """Design choices entry 80: the Account Manager sizes the purchase from the money in the account, so with $100 not even one share of PG at 167.10 fits. The decision is `no_funds` (the closed list has no other code), says
    why in the sizing's own words, and the user is never asked to approve a trade of nothing."""
    ask = Asked(True)
    order = decide(reviewed(), pf=cash(100.0), approver=ask)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["funds_avail"], d["shares"], d["est_cost"], d["escalated"]) == ("no_funds", False, 0, 0.0, False) and order.status == "decided" and ask.cards == [] and d["asked_user"] is False
    assert "not even one share fits" in d["reason"] and d["size"]["shares"] == 0 and d["size"]["binding"] == "no_room"


def test_holdings_that_could_be_sold_do_not_make_money_the_size_is_cut_by_the_cash():
    """The sample portfolio of Jacob Pyrett holds $51.23 and three holdings that are in profit. The Account Manager's money cap is the cash that is there, never what a RaiseFunds sale could raise: no share fits."""
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    d = decide(reviewed(), pf=sample, approver=Asked(True), p=profile(max_position_pct=1.0, max_new_exposure_pct=1.0, max_loss_per_trade_pct=1.0)).trade_decisions[0]
    assert (d["will_trade"], d["shares"], d["size"]["binding"], d["size"]["caps"]["money"]) == ("no_funds", 0, "no_room", 0) and d["size"]["equity_basis"].startswith("cash and holdings at the prices given")


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
    return {s: found[s][-1]["close"] for s in found}


def test_a_cost_the_cash_does_not_cover_is_checked_with_the_holdings_prices_for_raise_funds(sample_prices):
    """`decisions._funds` is the seam to `sizing.funds_check`: holdings in profit that are sellable could raise the money (a note), or could not (`no_funds`). The sizing never gives a cost over the cash, so this is the
    check for a size that was given more money than the portfolio holds."""
    from datetime import date
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    today = date(2026, 11, 13)
    fresh = lambda: {"will_trade": "yes_user", "funds_avail": None, "funds_note": None, "escalated": True, "reason": "x"}
    covered, d = 40.0, fresh()
    decisions._funds(d, covered, sample, sample_prices, today)
    assert d == {**fresh(), "funds_avail": True}                                                      # $51.23 covers $40: yes, and no note
    d = fresh()
    decisions._funds(d, 5000.0, sample, sample_prices, today)
    assert (d["will_trade"], d["funds_avail"]) == ("yes_user", True) and d["funds_note"].startswith("needs RaiseFunds: the cash does not cover it, but holdings in profit could (short by $4,948.77;")
    d = fresh()
    decisions._funds(d, 10_000_000.0, sample, sample_prices, today)
    assert (d["will_trade"], d["funds_avail"], d["escalated"]) == ("no_funds", False, False) and "$51.23 does not cover $10,000,000.00 and no holding in profit could raise the difference" in d["reason"]


def test_enough_cash_is_funds_with_no_note_and_the_decision_carries_how_it_was_sized():
    d = decide(reviewed(), approver=Asked(True)).trade_decisions[0]
    assert (d["funds_avail"], d["funds_note"], d["shares"], d["est_cost"]) == (True, None, 11, round(11 * 167.1, 2))        # $20,000: the position limit is 10%, $2,000, or 11 shares at 167.10
    assert set(d["size"]) == {"shares", "caps", "binding", "equity", "equity_basis", "notes", "allocation"}
    assert (d["size"]["shares"], d["size"]["binding"], d["size"]["equity"], d["size"]["equity_basis"], d["size"]["allocation"]) == (11, "position", 20000.0, "cash only (no holdings)", "as_much_as_possible")
    assert d["size"]["caps"]["position"] == 11 == int(2000 // 167.1) and d["size"]["caps"]["money"] == int(20000 // 167.1) and d["size"]["caps"]["exposure"] == int(6000 // 167.1) and d["size"]["caps"]["budget"] is None


# ---------------------------------------------------------------- Ask and Auto

def test_ask_mode_asks_with_the_scorecard_and_the_answer_is_the_decision():
    ask = Asked(True)
    order = decide(reviewed(), approver=ask)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["mode"], d["asked_user"], d["escalated"]) == ("yes_user", "ask", True, False) and order.status == "decided"
    [card] = ask.cards
    assert card["symbol"] == "PG" and card["approved"] and card["score"] == 88 and card["shares"] == 11 and "APPROVE" in card["text"] and "accuracy" in card["text"] and card["est_cost"] == pytest.approx(1838.1)
    assert "WOULD BE PLACED  buy 11 shares of PG, about $1,838.10" in card["text"] and "share" not in card["text"].split("WOULD BE PLACED")[0].splitlines()[0]        # the Critic's own card names no size; the Account Manager adds what would be placed
    assert decide(reviewed(), approver=Asked(False)).trade_decisions[0]["will_trade"] == "no_user_denied"


def test_with_nobody_to_ask_the_decision_waits_and_is_never_a_yes():
    for approver in (None, Asked(None)):
        order = decide(reviewed(), approver=approver)
        assert order.trade_decisions[0]["will_trade"] == "pending_user" and order.status == "reviewed" and order.trade_decisions[0]["reason"].endswith("waiting for the user")


def test_a_prompt_that_fails_leaves_the_decision_pending_and_says_so():
    def broken(card):
        raise EOFError("no terminal")
    order = decide(reviewed(), approver=broken)
    assert order.trade_decisions[0]["will_trade"] == "pending_user" and any("could not be asked about PG (EOFError)" in e for e in order.errors)


def test_auto_mode_trades_what_clears_the_mark_without_asking():
    ask = Asked(False)
    order = decide(reviewed(), p=profile(trade_mode="auto"), approver=ask)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["asked_user"], d["escalated"], d["mode"]) == ("yes_auto", False, False, "auto") and ask.cards == [] and "at or above the automatic-trading mark 80" in d["reason"]


def test_in_auto_mode_a_score_under_the_mark_goes_to_the_user():
    order = decide(reviewed(), p=profile(trade_mode="auto", auto_min_score=90), approver=Asked(True))
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["escalated"], d["asked_user"]) == ("yes_user", True, True) and "under the automatic-trading mark 90" in d["reason"]
    assert decide(reviewed(), p=profile(trade_mode="auto", auto_min_score=88)).trade_decisions[0]["will_trade"] == "yes_auto"       # exactly the mark


def test_in_auto_mode_a_finding_that_asks_for_a_person_goes_to_the_user():
    order = decide(reviewed("no_own"), p=profile(trade_mode="auto", auto_min_score=10), approver=Asked(None))
    d = order.trade_decisions[0]
    assert d["will_trade"] == "pending_user" and d["escalated"] and "value_not_assessable" in d["reason"]


def test_a_sale_is_never_traded_automatically_and_is_the_whole_holding_of_the_real_sample_portfolio():
    """The real sample portfolio of Jacob Pyrett holds 60 shares of GM. A recommendation to reduce GM is the whole holding (the Account Manager's own knowledge: the Critic and the Advisor size no sale), and it goes to
    the user even in Auto mode with a mark of 0."""
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    order = reviewed(action="reduce", symbol="GM")
    d = decide(order, p=profile(trade_mode="auto", auto_min_score=0), approver=Asked(True), pf=sample).trade_decisions[0]
    assert (d["will_trade"], d["escalated"], d["shares"], d["est_cost"], d["funds_avail"], d["size"]) == ("yes_user", True, 60, 0.0, True, None) and "never traded automatically" in d["reason"] and "whole holding" in d["reason"]
    assert wo.verify_trade(order, d["record_id"], NOW) == []                                          # the Trader's precondition: a whole number of shares of at least 1, and the decision's own


def test_the_card_of_a_sale_says_the_whole_holding_and_its_shares():
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    ask = Asked(None)
    decide(reviewed(action="reduce", symbol="TSLA"), approver=ask, pf=sample)
    [card] = ask.cards
    assert card["action"] == "reduce" and card["shares"] == 123 and "WOULD BE PLACED  sell the whole holding of TSLA (123 shares)" in card["text"]


def test_a_sale_of_a_symbol_the_portfolio_does_not_hold_is_not_traded():
    d = decide(reviewed(action="reduce"), approver=Asked(True)).trade_decisions[0]                   # PG, in a portfolio of cash only
    assert (d["will_trade"], d["shares"], d["funds_avail"]) == ("no_funds", 0, False) and "the portfolio holds no PG: there is nothing to sell" in d["reason"]


def test_the_mode_argument_overrides_the_profile_for_one_run():
    assert decide(reviewed(), p=profile(trade_mode="ask"), mode="auto").trade_decisions[0]["will_trade"] == "yes_auto"
    assert decide(reviewed(), p=profile(trade_mode="auto"), mode="ask", approver=Asked(True)).trade_decisions[0]["mode"] == "ask"
    order = decide(reviewed(), mode="manual")
    assert order.status == "reviewed" and order.trade_decisions == [] and "mode must be one of ask, auto" in order.errors[-1]


# ---------------------------------------------------------------- the cash: nothing is reserved (Jacob, 2026-11-14: "any available funds in the portfolio can be used")

WIDE = dict(max_position_pct=1.0, max_new_exposure_pct=1.0, max_loss_per_trade_pct=0.2)            # a user whose own limits are not what holds a purchase back: the money is


def _two_purchases():
    return _reviewed(records=2).model_copy(deep=True)


def with_holdings(amount):
    """`amount` in cash and $1,000 of a holding (50 shares of GM at a cost of $20): an equity over the cash, so that the exposure room (a limit of the profile shared by the purchases) is not what runs out first."""
    return Portfolio(user="Jacob Pyrett", cash_avail=amount, positions=[Position(symbol="GM", shares=50, cost_basis=20.0, acquired=date(2026, 1, 5))])


def test_each_purchase_is_checked_against_all_the_cash_nothing_is_reserved_for_another():
    first, second = decide(_two_purchases(), pf=with_holdings(1000.0), p=profile(**WIDE), approver=Asked(True)).trade_decisions
    assert [d["will_trade"] for d in (first, second)] == ["yes_user", "yes_user"]
    assert first["shares"] == second["shares"] == 5 == int(1000 // 167.1) and first["est_cost"] == second["est_cost"] == 835.5            # $1,671 in all, each one covered by the $1,000
    assert first["size"]["binding"] == second["size"]["binding"] == "money" and first["size"]["caps"]["money"] == second["size"]["caps"]["money"] == 5
    short = decide(_two_purchases(), pf=with_holdings(100.0), p=profile(**WIDE), approver=Asked(True))
    assert [d["will_trade"] for d in short.trade_decisions] == ["no_funds", "no_funds"] and all("not even one share fits" in d["reason"] for d in short.trade_decisions)


def test_a_purchase_the_user_denied_does_not_change_what_the_other_may_use():
    answers = iter([False, True])
    order = decide(_two_purchases(), pf=with_holdings(1000.0), p=profile(**WIDE), approver=lambda card: next(answers))
    assert [d["will_trade"] for d in order.trade_decisions] == ["no_user_denied", "yes_user"] and [d["shares"] for d in order.trade_decisions] == [5, 5]       # a "no" re-splits nothing


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
    order = decide(_reviewed(records=2).model_copy(deep=True), pf=cash(20000.0))
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

def test_the_trader_may_trade_only_what_was_approved_and_permitted_and_not_after_the_approval_ran_out():
    for will, expect in (("yes_user", True), ("pending_user", False), ("no_user_denied", False)):
        order = decide(reviewed(), approver=Asked({"yes_user": True, "pending_user": None, "no_user_denied": False}[will]))
        record_id = order.reviews[0]["record_id"]
        assert (wo.verify_trade(order, record_id, NOW) == []) is expect
    order = decide(reviewed(), p=profile(trade_mode="auto"))
    record_id = order.reviews[0]["record_id"]
    assert wo.verify_trade(order, record_id, NOW) == [] and any("ran out" in r for r in wo.verify_trade(order, record_id, NOW + timedelta(minutes=61)))
    denied = decide(reviewed(approved=False), approver=Asked(True))
    assert any("did not approve" in r for r in wo.verify_trade(denied, denied.reviews[0]["record_id"], NOW))


# ---------------------------------------------------------------- the AccountManager's own part

def manager(**kw):
    return AccountManager(model=object(), **kw)


def test_the_account_manager_holds_the_profile_and_hands_the_critic_only_the_limits():
    am = manager()
    assert am.profile.user == "Jacob Pyrett" and am.profile.trade_mode == "ask"
    assert am.risk_view() == am.profile.risk_view() and "trade_mode" not in json.dumps(am.risk_view().model_dump())


def test_the_profile_can_be_given_as_an_object_or_a_path_or_replaced():
    assert manager(profile=profile(risk_tolerance="aggressive")).risk_view().risk_tolerance == "aggressive"
    assert manager(profile=REPO_ROOT / "configs" / "profiles" / "jacob_pyrett.json").profile.risk_tolerance == "moderate"
    am = manager()
    am.profile = profile(risk_tolerance="conservative")
    assert am.risk_view().risk_tolerance == "conservative"


def test_a_profile_file_that_is_wrong_is_a_value_error_naming_it(tmp_path):
    (tmp_path / "p.json").write_text('{"user": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="p.json"):
        manager(profile=tmp_path / "p.json").profile


def test_the_account_manager_decides_with_the_critics_own_mark_and_the_profiles_when_it_has_one():
    assert manager(profile=profile(trade_mode="auto")).decide_trades(reviewed(), cash(), now=NOW).trade_decisions[0]["will_trade"] == "yes_auto"          # the Critic's mark is 80, the score 88
    assert manager(profile=profile(trade_mode="auto", auto_min_score=99)).decide_trades(reviewed(), cash(), now=NOW).trade_decisions[0]["will_trade"] == "pending_user"
    order = manager().decide_trades(reviewed(), cash(), approver=Asked(True), now=NOW)
    assert order.trade_decisions[0]["will_trade"] == "yes_user"
    manager().record_user_decision(order, order.trade_decisions[0]["record_id"], False, NOW)
    assert any("no pending decision" in e for e in order.errors)


def test_the_managers_decisions_need_no_model_and_no_network():
    class Raising:
        def invoke(self, *a, **k):
            raise AssertionError("a decision must not call a model")
    order = AccountManager(model=Raising(), clerk=FakeClerk(reviewed()), profile=profile()).decide_trades(reviewed(), cash(), now=NOW)
    assert order.trade_decisions[0]["will_trade"] == "pending_user"


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
    assert size["equity"] == pytest.approx(51.23 + 50 * sample_prices["MSFT"] + 123 * sample_prices["TSLA"] + 60 * sample_prices["GM"]) and size["equity_basis"] == "cash and holdings at the prices given"
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
    assert size["equity"] == pytest.approx(51.23 + 50 * 150.0 + 123 * 300.0 + 60 * 10.0) and size["equity_basis"].endswith("MSFT, TSLA, GM at cost basis (no price)") and order.status == "decided"


def test_a_price_the_data_clerk_cannot_give_is_a_message_and_that_holding_counts_at_cost_basis(monkeypatch):
    clerk = real_clerk(monkeypatch, day=date(2026, 11, 14))                              # a Saturday: no daily bar is stored for it, so no quote can be had
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    order = manager(profile=profile(), clerk=clerk).decide_trades(reviewed(), sample, approver=Asked(True), now=NOW)
    assert any(e.startswith("no current price for MSFT, TSLA, GM: counted at cost basis") for e in order.errors)
    assert order.trade_decisions[0]["size"]["equity_basis"].endswith("MSFT, TSLA, GM at cost basis (no price)")


def test_the_pure_decision_step_takes_the_prices_it_is_given_and_calls_nothing(sample_prices):
    sample = load_portfolio(REPO_ROOT / "configs" / "portfolios" / "jacob_pyrett.json")
    priced = decisions.decide_trades(reviewed(), sample, profile(), MARK, Asked(True), NOW, None, None, None, sample_prices).trade_decisions[0]["size"]
    unpriced = decisions.decide_trades(reviewed(), sample, profile(), MARK, Asked(True), NOW).trade_decisions[0]["size"]
    assert priced["equity_basis"] == "cash and holdings at the prices given" and priced["equity"] > unpriced["equity"] and unpriced["equity_basis"].endswith("at cost basis (no price)")
