"""The work order's Critic v1 additions (Critic spec section 4.3): the new statuses, `HAS_DATA`, `verify_verification_request`, `verify_trade`.

(Kept apart from tests/test_work_order.py, which builds real stores for the Data Clerk: none of this needs one.)
"""
from datetime import datetime, timedelta, timezone

import pytest

from prometheus.common import work_order as wo

NOW = datetime(2026, 11, 5, 15, 30, tzinfo=timezone.utc)


def verify_order(**spec):
    return wo.new_work_order("verify", requester="critic", action_type="verify_data", subject=["PG"], verify=spec)


def reasons_of(**spec):
    return wo.verify_verification_request(verify_order(**spec))


# ---------------------------------------------------------------- statuses and the order's new fields

def test_the_statuses_run_on_to_reviewed_decided_and_traded():
    assert wo.STATUSES == ("new", "data_ready", "data_failed", "recommended", "reviewed", "decided", "traded")
    assert wo.HAS_DATA == ("data_ready", "recommended", "reviewed", "decided", "traded")


@pytest.mark.parametrize("status", ["data_ready", "recommended", "reviewed", "decided", "traded"])
def test_every_check_refuses_an_order_that_already_has_its_data(status):
    search = wo.new_work_order("x", subject=["PG"])
    quote = wo.new_work_order("q", action_type="fetch_quote", subject=["PG"])
    metric = wo.new_work_order("m", action_type="calculate_financials", subject=["PG"], metrics=[{"name": "rsi_14"}])
    verify = verify_order(symbols=["PG"])
    bars = wo.new_work_order("b", action_type="fetch_bars", subject=["PG"], bars={"start": "2026-11-05", "end": "2026-11-06"})
    for o, check in ((search, wo.verify_request), (quote, wo.verify_quote_request), (metric, wo.verify_metric_request), (verify, wo.verify_verification_request), (bars, wo.verify_bar_request)):
        assert not any("already has its data" in r for r in check(o))
        o.status = status
        assert any(f"already has its data (status {status})" in r for r in check(o))


# ---------------------------------------------------------------- verify_verification_request


@pytest.mark.parametrize("spec, message", [
    ({}, "nothing to verify"),
    ({"chunks": [], "symbols": []}, "nothing to verify"),
    ({"symbols": ["PG"], "colour": 1}, "unknown verify keys: colour"),
    ({"chunks": "finnhub:1#0"}, "chunks must be a list"),
    ({"chunks": ["has a space"]}, "chunks must be a list"),
    ({"chunks": [""]}, "chunks must be a list"),
    ({"chunks": ["x" * 201]}, "chunks must be a list"),
    ({"chunks": [7]}, "chunks must be a list"),
    ({"chunks": [f"c{i}" for i in range(61)]}, "at most 60 chunks"),
    ({"evidence": ["E1"]}, "evidence must be"),
    ({"evidence": {"PG": ["X1"]}}, "evidence must be"),
    ({"evidence": {"pg": ["E1"]}}, "evidence must be"),
    ({"evidence": {"PG": "E1"}}, "evidence must be"),
    ({"symbols": "PG"}, "symbols must be a list"),
    ({"symbols": ["../x"]}, "symbols must be a list"),
    ({"symbols": [f"S{i}" for i in range(31)]}, "at most 30 symbols"),
    ({"symbols": ["PG"], "company": ["PG"]}, "company must be"),
    ({"symbols": ["PG"], "company": {"PG": 5}}, "company must be"),
    ({"symbols": ["PG"], "fresh": {"since_days": 0}}, "fresh needs since_days from 1 to 30"),
    ({"symbols": ["PG"], "fresh": {"since_days": 31}}, "fresh needs since_days from 1 to 30"),
    ({"symbols": ["PG"], "fresh": {"k": 21}}, "k from 1 to 20"),
    ({"symbols": ["PG"], "fresh": {"k": True}}, "k from 1 to 20"),
    ({"symbols": ["PG"], "fresh": {"days": 3}}, "fresh must be"),
    ({"symbols": ["PG"], "fresh": 14}, "fresh must be"),
    ({"chunks": ["finnhub:1#0"], "fresh": {"since_days": 3}}, "give at least one"),
])
def test_the_requests_the_data_clerk_may_not_answer(spec, message):
    assert any(message in r for r in reasons_of(**spec)), reasons_of(**spec)


def test_a_verification_request_must_be_a_verify_order_with_a_verify_dict_and_no_data_yet():
    assert any("is not 'verify_data'" in r for r in wo.verify_verification_request(wo.new_work_order("x", subject=["PG"], verify={"symbols": ["PG"]})))
    assert any("needs `verify`" in r for r in wo.verify_verification_request(wo.new_work_order("x", action_type="verify_data", subject=["PG"])))
    done = verify_order(symbols=["PG"])
    done.status = "reviewed"
    assert any("already has its data (status reviewed)" in r for r in wo.verify_verification_request(done))


# ---------------------------------------------------------------- lookups and parse_utc


# ---------------------------------------------------------------- verify_trade: the Trader's precondition

def reviewed(will_trade="yes_user", approved=True, valid_until=None, shares=44, action="open"):
    order = wo.new_work_order("x", subject=["PG"])
    order.status = "decided"
    order.reviews = [{"record_id": "r1", "approved": approved, "valid_until_utc": valid_until or (NOW + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")}]
    order.trade_decisions = [{"record_id": "r1", "will_trade": will_trade, "shares": shares, "action": action}]      # AMENDED (entry 105): the decision of a recommendation says its action, and only `open` is a purchase a recommendation may be
    return order


def test_no_review_no_decision_a_denial_an_expired_approval_or_no_shares_each_stop_the_trade():
    order = wo.new_work_order("x", subject=["PG"])
    reasons = wo.verify_trade(order, "r1", NOW)
    assert any("no Critic review" in r for r in reasons) and any("no trade decision" in r for r in reasons)
    assert any("did not approve" in r for r in wo.verify_trade(reviewed(approved=False), "r1", NOW))
    assert any("ran out" in r for r in wo.verify_trade(reviewed(), "r1", NOW + timedelta(minutes=31)))
    assert wo.verify_trade(reviewed(), "r1", NOW + timedelta(minutes=30)) == []                        # the last instant is still inside
    assert any("no valid_until_utc" in r for r in wo.verify_trade(reviewed(valid_until="soon"), "r1", NOW))
    # AMENDED (entry 110): the reason says "a number of shares above zero" (not "whole"), and 2.5 is a quantity now (the decision of an even split may be a fraction): it became 0.1234567, seven decimals, still refused
    assert any("a number of shares above zero" in r for r in wo.verify_trade(reviewed(shares=0), "r1", NOW))
    assert any("a number of shares above zero" in r for r in wo.verify_trade(reviewed(shares=True), "r1", NOW))
    assert any("a number of shares above zero" in r for r in wo.verify_trade(reviewed(shares=0.1234567), "r1", NOW)) and wo.verify_trade(reviewed(shares=2.5), "r1", NOW) == []
    assert wo.verify_trade(reviewed(shares=1), "r1", NOW) == [] and wo.verify_trade(reviewed(shares=45), "r1", NOW) == []      # the number of shares is the Account Manager's (the decision's): the Critic approves no size to compare it with


def test_a_decision_tied_to_a_recommendation_that_is_not_a_purchase_to_open_is_refused_a_sale_never_comes_from_a_recommendation():
    """Entry 105 (Jacob: *"Only the AccountManager can raise funds and recommend this"*): the Trader may sell for the user's own direct order or for a RaiseFunds plan, never because of a recommendation. A decision that
    answers a Critic review and whose action is `reduce`, `add` or `hold` is refused by the Trader's precondition, however it is approved."""
    for action in ("reduce", "add", "hold"):
        reasons = wo.verify_trade(reviewed(action=action), "r1", NOW)
        assert any(f"the decision's action is '{action}': a recommendation is a purchase to open, and a sale never comes from a recommendation" in r for r in reasons), (action, reasons)
    assert wo.verify_trade(reviewed(action="open"), "r1", NOW) == []


def test_verify_trade_refuses_a_naive_clock():
    with pytest.raises(ValueError, match="timezone"):
        wo.verify_trade(reviewed(), "r1", datetime(2026, 11, 5, 15, 30))
