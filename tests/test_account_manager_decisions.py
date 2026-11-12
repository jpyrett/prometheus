"""The Account Manager's trade decisions from the Critic's verdicts (Critic spec 12.2 and 12.3): every row of the table, the cash across a review, the mode, and the Trader's precondition."""
import copy
import functools
import json
from datetime import timedelta

import pytest
from critic_helpers import NOW, FakeClerk, captured, make_critic, portfolio, profile, tamper, view

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.rag.config import REPO_ROOT

MARK = 80


@functools.lru_cache(maxsize=None)
def _reviewed(own_key="own", records=1):
    order = captured("PG_open")
    for i in range(1, records):
        order.recommendations.append(tamper(order.recommendations[0], **{"identity.record_id": f"extra{i:07d}", "action.action": "add"}))
    make_critic(order, **({"own": {}} if own_key == "no_own" else {})).review_recommendations(order, view(max_new_exposure_pct=0.30), portfolio(), NOW)
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


def test_without_the_money_it_is_not_traded_and_with_holdings_that_could_raise_it_it_says_so():
    order = decide(reviewed(), pf=cash(100.0), approver=Asked(True))
    d = order.trade_decisions[0]
    assert d["will_trade"] == "yes_user" and d["funds_avail"] is True and "needs RaiseFunds" in d["funds_note"]               # the record's own plan found sellable holdings in profit
    broke = reviewed()
    broke.recommendations[0] = tamper(broke.recommendations[0], **{"trade.funds_feasible": False})
    order = decide(broke, pf=cash(100.0), approver=Asked(True))
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["funds_avail"]) == ("no_funds", False) and "$100.00 does not cover $7,352.40" in d["reason"] and order.status == "decided"


def test_enough_cash_is_funds_with_no_note():
    d = decide(reviewed(), approver=Asked(True)).trade_decisions[0]
    assert (d["funds_avail"], d["funds_note"], d["shares"], d["est_cost"]) == (True, None, 44, round(44 * 167.1, 2))


# ---------------------------------------------------------------- Ask and Auto

def test_ask_mode_asks_with_the_scorecard_and_the_answer_is_the_decision():
    ask = Asked(True)
    order = decide(reviewed(), approver=ask)
    d = order.trade_decisions[0]
    assert (d["will_trade"], d["mode"], d["asked_user"], d["escalated"]) == ("yes_user", "ask", True, False) and order.status == "decided"
    [card] = ask.cards
    assert card["symbol"] == "PG" and card["approved"] and card["score"] == 88 and card["shares"] == 44 and "APPROVE" in card["text"] and "accuracy" in card["text"] and card["est_cost"] == pytest.approx(7352.4)
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


def test_a_sale_is_never_traded_automatically():
    order = reviewed()
    order.reviews[0].update(action="reduce")
    d = decide(order, p=profile(trade_mode="auto", auto_min_score=0), approver=Asked(True)).trade_decisions[0]
    assert (d["will_trade"], d["escalated"], d["est_cost"], d["funds_avail"]) == ("yes_user", True, 0.0, True) and "never traded automatically" in d["reason"]


def test_the_mode_argument_overrides_the_profile_for_one_run():
    assert decide(reviewed(), p=profile(trade_mode="ask"), mode="auto").trade_decisions[0]["will_trade"] == "yes_auto"
    assert decide(reviewed(), p=profile(trade_mode="auto"), mode="ask", approver=Asked(True)).trade_decisions[0]["mode"] == "ask"
    order = decide(reviewed(), mode="manual")
    assert order.status == "reviewed" and order.trade_decisions == [] and "mode must be one of ask, auto" in order.errors[-1]


# ---------------------------------------------------------------- the cash across a review

def test_two_purchases_cannot_spend_the_same_dollars():
    order = _reviewed(records=2).model_copy(deep=True)
    for i, r in enumerate(order.recommendations):
        order.recommendations[i] = tamper(r, **{"trade.funds_feasible": False})
    order = decide(order, pf=cash(10000.0), approver=Asked(True))
    first, second = order.trade_decisions
    assert [d["will_trade"] for d in (first, second)] == ["yes_user", "no_funds"] and "$2,647.60 does not cover" in second["reason"]
    both = decide(_reviewed(records=2).model_copy(deep=True), pf=cash(20000.0), approver=Asked(True))
    assert [d["will_trade"] for d in both.trade_decisions] == ["yes_user", "yes_user"]


def test_a_purchase_the_user_denied_does_not_keep_the_cash():
    answers = iter([False, True])
    order = _reviewed(records=2).model_copy(deep=True)
    for i, r in enumerate(order.recommendations):
        order.recommendations[i] = tamper(r, **{"trade.funds_feasible": False})
    order = decide(order, pf=cash(8000.0), approver=lambda card: next(answers))
    assert [d["will_trade"] for d in order.trade_decisions] == ["no_user_denied", "yes_user"]


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
