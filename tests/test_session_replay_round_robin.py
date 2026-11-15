"""A second real user session, replayed offline: the first one run through the whole NEW flow (design choices entries 80 to 83).

`tests/fixtures/session_real/bob_jones_oil_round_robin.json` is a capture of
`./test_prometheus.sh --session --name "Bob Jones" --age 60 --risk conservative --cash 15000 --mode auto --yes --no-fetch --auto-min-score 70 --cleanup --save ... --prompt "I want to buy oil stocks. Use as much of the
money in my account as you can to buy shares of the 3 most recommended oil companies."` on 2026-11-15, with the real Account Manager, Data Clerk, Advisor, Critic (GLM 5.3) and Trader (gpt-oss-120b) on the real stored data,
the simulated clock (the close of Friday 2026-11-13) and the real local Postgres account book. It is the first capture made AFTER the Advisor and the Critic stopped sizing (the Account Manager sizes), after Auto mode began
spending the funds round-robin (no split was asked for), and after the Trader and the Account Manager began shaking hands on every trade (each fill SELECTed, booked in one locked transaction and SELECTed again).
Nothing in it was migrated: it was written by the code as it is.

What the real run showed: CVX and EOG were approved (Critic 87 and 77), PSX was denied for `risk_above_profile` (Bob is conservative); the round-robin spend bought one share of CVX, one of EOG, and so on until each
reached the profile's limit of 10% of the $15,000 (CVX 6 shares, EOG 9), which is $2,745 of the money; the Trader placed both and each was booked.

The tests re-run the DECISION step over the real reviews and must reproduce what the live run decided, and check the real fills against the real decisions and the account's own arithmetic. No stand-in agent, nothing invented.
"""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil_round_robin.json"
NODES = ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)
BOB = Portfolio(user="Bob Jones", cash_avail=15000.0)
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto", auto_min_score=70)


@pytest.fixture(scope="module")
def capture():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def reviewed(capture) -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def kept(decision):
    return {k: decision[k] for k in ("record_id", "symbol", "action", "shares", "est_cost", "mode", "will_trade", "funds_avail") if k in decision}


def test_the_capture_is_one_real_session_through_every_node_and_the_run_booked_what_it_traded(capture):
    assert capture["path"] == NODES and capture["request"].startswith("I want to buy oil stocks. Use as much of the money")
    assert capture["intent"]["action"] == "buy" and capture["intent"]["count"] == 3 and capture["intent"]["allocation"] == "default" and capture["intent"]["budget"] is None
    order = wo.WorkOrder.model_validate(capture["order"])
    assert order.status == "traded" and order.schema_version == 4 and order.data is None
    assert [r["identity"]["record_id"] for r in order.recommendations] == [r["record_id"] for r in order.reviews] == [d["record_id"] for d in order.trade_decisions]
    run = order.trader_runs[0]
    assert run["model"] == "openai/gpt-oss-120b" and run["mode"] == "paper" and run["booked"] is True and run["errors"] == [] and [t["booked"] for t in order.trades] == [True, True]


def test_the_real_reviews_are_two_approvals_and_a_denial_for_risk_and_the_advisor_recommended_no_size(capture):
    reviews = {r["symbol"]: r for r in capture["order"]["reviews"]}
    assert {s for s, r in reviews.items() if r["approved"]} == {"CVX", "EOG"} and [b["code"] for b in reviews["PSX"]["blockers"]] == ["risk_above_profile"]
    for record in capture["order"]["recommendations"]:
        assert not {"suggested_shares", "equity", "position_pct", "funds_feasible", "risk_budget"} & set(record.get("trade") or {})      # entry 80: the Advisor holds no size
    for review in capture["order"]["reviews"]:
        assert not {"approved_shares", "suggested_shares", "allowed_shares", "caps", "equity"} & (set(review) | set(review["risk"]))     # and neither does the Critic


def test_the_decisions_of_the_live_run_are_reproduced_from_the_real_reviews_round_robin_each_stopped_by_its_own_position_limit(capture):
    live = [kept(d) for d in capture["order"]["trade_decisions"]]
    again = decisions.decide_trades(reviewed(capture), BOB, PROFILE, 80, None, MOMENT, "auto", None, None)
    assert again.status == "decided" and again.errors == [] and [kept(d) for d in again.trade_decisions] == live
    bought = [d for d in again.trade_decisions if d["will_trade"] == "yes_auto"]
    assert [(d["symbol"], d["shares"], d["size"]["binding"], d["size"]["allocation"]) for d in bought] == [("CVX", 6, "position", "round_robin"), ("EOG", 9, "position", "round_robin")]
    assert all("spent round-robin" in d["reason"] for d in bought) and not any("balance" in d for d in bought)                          # no split was asked: not an even split
    assert [d["will_trade"] for d in again.trade_decisions if d["symbol"] == "PSX"] == ["no_system_denied"]


def test_each_purchase_is_the_most_its_own_limit_allows_and_one_share_more_would_have_broken_it(capture):
    """The profile's limit is 10% of the $15,000 in one stock: 6 CVX and 9 EOG are the most whole shares at the real deal prices, and every decision's cost is below the limit."""
    prices = {r["symbol"]: r["timing"]["deal_price"] for r in capture["order"]["reviews"]}
    for d in capture["order"]["trade_decisions"]:
        if d["will_trade"] == "yes_auto":
            assert d["est_cost"] <= 1500.0 < d["est_cost"] + prices[d["symbol"]] and d["shares"] == int(1500.0 // prices[d["symbol"]])


def test_the_real_fills_agree_with_the_decisions_pass_the_output_guard_and_the_account_arithmetic_adds_up(capture):
    order = wo.WorkOrder.model_validate(capture["order"])
    assert wo.verify_trades(order) == []
    decided = {d["record_id"]: d for d in order.trade_decisions if d["will_trade"] == "yes_auto"}
    assert [(t["symbol"], t["shares"], t["status"]) for t in order.trades] == [("CVX", 6, "filled"), ("EOG", 9, "filled")]
    for trade in order.trades:
        assert trade["shares"] <= decided[trade["record_id"]]["shares"] and wo.verify_fill(order, trade, order.trades) == []
    spent = round(sum(t["fill"]["amount"] for t in order.trades), 2)
    after = order.trader_runs[0]["portfolio_after"]
    assert after["cash"] == round(15000.0 - spent, 2) == 12254.41 and {p["symbol"]: p["shares"] for p in after["positions"]} == {"CVX": 6, "EOG": 9}


def test_the_session_is_the_new_flow_the_handshake_booked_each_fill_and_nothing_was_spent_above_the_limit(capture):
    run = capture["order"]["trader_runs"][0]
    assert run["booked"] is True and run["placed_by_model"] + run["placed_by_code"] == 2
    cash_before = run["portfolio_before"]["cash"]
    assert cash_before == 15000.0 and run["portfolio_after"]["cash"] >= 15000.0 - 2 * 1500.0                                         # no purchase above the 10% limit, and nothing reserved
