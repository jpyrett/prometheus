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


def test_a_v1_order_still_validates_and_the_new_fields_are_empty():
    order = wo.new_work_order("What is the news on PG?", subject=["PG"])
    assert order.schema_version == 1 and order.reviews == [] and order.critic_runs == [] and order.trade_decisions == [] and order.verify is None
    assert order.trades == [] and order.trader_runs == [] and order.bars is None             # Trader v1 (spec section 11): the fields are there and empty
    wo.WorkOrder.model_validate(order.model_dump())
    assert wo.DataResult().verification == {}


def test_a_verify_order_is_a_version_2_order_and_round_trips_as_json():
    order = verify_order(symbols=["PG"])
    assert order.schema_version == 2 and order.action_type == "verify_data" and order.verify == {"symbols": ["PG"]}
    assert wo.WorkOrder.model_validate_json(order.model_dump_json()).verify == {"symbols": ["PG"]}


def test_the_search_check_refuses_a_verify_order_and_names_the_new_action():
    reasons = wo.verify_request(verify_order(symbols=["PG"]))
    assert any("not handled by the Data Clerk's request_data yet" in r for r in reasons)
    assert any("is not one of" in r and "verify_data" in r for r in wo.verify_request(wo.new_work_order("x", action_type="launch_rocket")))


# ---------------------------------------------------------------- verify_verification_request

def test_a_minimal_request_is_allowed():
    assert reasons_of(symbols=["PG"]) == [] and reasons_of(chunks=["finnhub:1#0"]) == [] and reasons_of(evidence={"PG": ["E15"]}) == []


def test_a_full_request_is_allowed():
    assert reasons_of(chunks=["finnhub:1#0", "pr:ab12#0"], evidence={"PG": ["E15", "E21"]}, symbols=["PG", "SNDK"], company={"PG": "Procter & Gamble"},
                      fresh={"since_days": 14, "k": 8}) == []


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


def test_the_limits_are_inclusive():
    assert reasons_of(chunks=[f"c{i}" for i in range(60)]) == [] and reasons_of(symbols=[f"S{i}" for i in range(30)]) == []
    assert reasons_of(symbols=["PG"], fresh={"since_days": 30, "k": 20}) == [] and reasons_of(symbols=["PG"], fresh={"since_days": 1, "k": 1}) == []


def test_a_verification_request_must_be_a_verify_order_with_a_verify_dict_and_no_data_yet():
    assert any("is not 'verify_data'" in r for r in wo.verify_verification_request(wo.new_work_order("x", subject=["PG"], verify={"symbols": ["PG"]})))
    assert any("needs `verify`" in r for r in wo.verify_verification_request(wo.new_work_order("x", action_type="verify_data", subject=["PG"])))
    done = verify_order(symbols=["PG"])
    done.status = "reviewed"
    assert any("already has its data (status reviewed)" in r for r in wo.verify_verification_request(done))


# ---------------------------------------------------------------- lookups and parse_utc

def test_a_review_and_a_decision_are_found_by_record_id():
    order = wo.new_work_order("x", subject=["PG"])
    order.reviews, order.trade_decisions = [{"record_id": "a"}, {"record_id": "b", "n": 2}], [{"record_id": "b", "will_trade": "yes_user"}]
    assert wo.review_of(order, "b")["n"] == 2 and wo.review_of(order, "zz") is None
    assert wo.decision_of(order, "b")["will_trade"] == "yes_user" and wo.decision_of(order, "a") is None


@pytest.mark.parametrize("text, expected", [("2026-11-05T15:30:00Z", NOW), ("2026-11-05T10:30:00-05:00", NOW), ("2026-11-05T15:30:00", None), ("garbage", None), (None, None)])
def test_parse_utc(text, expected):
    assert wo.parse_utc(text) == expected


# ---------------------------------------------------------------- verify_trade: the Trader's precondition

def reviewed(will_trade="yes_user", approved=True, valid_until=None, shares=44):
    order = wo.new_work_order("x", subject=["PG"])
    order.status = "decided"
    order.reviews = [{"record_id": "r1", "approved": approved, "valid_until_utc": valid_until or (NOW + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")}]
    order.trade_decisions = [{"record_id": "r1", "will_trade": will_trade, "shares": shares}]
    return order


def test_a_review_an_approval_a_permission_and_time_left_let_the_trader_trade():
    assert wo.verify_trade(reviewed(), "r1", NOW) == [] and wo.verify_trade(reviewed("yes_auto"), "r1", NOW) == []


@pytest.mark.parametrize("will_trade", ["pending_user", "no_system_denied", "no_expired", "no_funds", "no_user_denied"])
def test_only_yes_auto_and_yes_user_are_permission(will_trade):
    assert any(will_trade in r for r in wo.verify_trade(reviewed(will_trade), "r1", NOW))


def test_no_review_no_decision_a_denial_an_expired_approval_or_no_shares_each_stop_the_trade():
    order = wo.new_work_order("x", subject=["PG"])
    reasons = wo.verify_trade(order, "r1", NOW)
    assert any("no Critic review" in r for r in reasons) and any("no trade decision" in r for r in reasons)
    assert any("did not approve" in r for r in wo.verify_trade(reviewed(approved=False), "r1", NOW))
    assert any("ran out" in r for r in wo.verify_trade(reviewed(), "r1", NOW + timedelta(minutes=31)))
    assert wo.verify_trade(reviewed(), "r1", NOW + timedelta(minutes=30)) == []                        # the last instant is still inside
    assert any("no valid_until_utc" in r for r in wo.verify_trade(reviewed(valid_until="soon"), "r1", NOW))
    assert any("whole number of shares" in r for r in wo.verify_trade(reviewed(shares=0), "r1", NOW))
    assert any("whole number of shares" in r for r in wo.verify_trade(reviewed(shares=True), "r1", NOW))
    assert any("whole number of shares" in r for r in wo.verify_trade(reviewed(shares=2.5), "r1", NOW))
    assert wo.verify_trade(reviewed(shares=1), "r1", NOW) == [] and wo.verify_trade(reviewed(shares=45), "r1", NOW) == []      # the number of shares is the Account Manager's (the decision's): the Critic approves no size to compare it with


def test_a_decision_without_a_matching_review_for_a_different_record_does_not_help():
    assert any("no Critic review of recommendation r2" in r for r in wo.verify_trade(reviewed(), "r2", NOW))


def test_verify_trade_refuses_a_naive_clock():
    with pytest.raises(ValueError, match="timezone"):
        wo.verify_trade(reviewed(), "r1", datetime(2026, 11, 5, 15, 30))
