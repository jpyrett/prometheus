"""A second real user session, replayed offline: the first one run through the whole NEW flow (design choices entries 80 to 83).

`tests/fixtures/session_real/bob_jones_oil_round_robin.json` is a capture of
`./test_prometheus.sh --session --name "Bob Jones" --age 60 --risk conservative --cash 15000 --mode auto --yes --no-fetch --cleanup --save ... --prompt "I want to buy oil stocks. Use as much of the
money in my account as you can to buy shares of the 3 most recommended oil companies."` on 2026-11-15, with the real Account Manager, Data Clerk, Advisor, Critic (GLM 5.3) and Trader (gpt-oss-120b) on the real stored data,
the simulated clock (the close of Friday 2026-11-13) and the real local Postgres account book. It is the first capture made AFTER the Advisor and the Critic stopped sizing (the Account Manager sizes), after Auto mode began
spending the funds round-robin (no split was asked for), and after the Trader and the Account Manager began shaking hands on every trade (each fill SELECTed, booked in one locked transaction and SELECTed again).
Nothing in it was migrated: it was written by the code as it is.

What the real run showed: CVX and EOG were approved (Critic 87 and 77), PSX was denied for `risk_above_profile` (Bob is conservative); the round-robin spend bought one share of CVX, one of EOG, and so on until each
reached the profile's limit of 10% of the $15,000 (CVX 6 shares, EOG 9), which is $2,745 of the money; the Trader placed both and each was booked. THAT LIMIT NO LONGER EXISTS (design choices entry 91): the capture is the
record of a real run under the old rule, so the fills and the Trader's work are checked as they were, and the Account Manager's decision step is re-run under the new one (the cash is the only check), over the same real reviews.

The tests re-run the DECISION step over the real reviews and must reproduce what the live run decided, and check the real fills against the real decisions and the account's own arithmetic. No stand-in agent, nothing invented.
"""
import copy
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.common import work_order as wo
from prometheus.common.money import money as money_door
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil_round_robin.json"
NODES = ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)
BOB = Portfolio(user="Bob Jones", cash_avail=Decimal("15000"))
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto")


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


def robin(prices, pot):
    """The round robin worked out by hand, in cents: one share of each in turn while the share fits what is left of the pot; a stock that cannot take another share is closed."""
    from prometheus.common.portfolio import trade_amount
    shares, spent, open_ = [0] * len(prices), Decimal(0), list(range(len(prices)))
    while open_:
        for i in list(open_):
            step = trade_amount(shares[i] + 1, prices[i]) - trade_amount(shares[i], prices[i])
            if spent + step <= pot:
                shares[i], spent = shares[i] + 1, spent + step
            else:
                open_.remove(i)
    return shares


def test_the_decisions_are_reproduced_from_the_real_reviews_by_the_round_robin_with_only_the_cash_to_stop_it(capture):
    live = {d["symbol"]: d for d in capture["order"]["trade_decisions"]}
    again = decisions.decide_trades(reviewed(capture), BOB, PROFILE, None, MOMENT, "auto", None, None)
    assert again.status == "decided" and again.errors == [] and [(d["symbol"], d["will_trade"]) for d in again.trade_decisions] == [(d["symbol"], d["will_trade"]) for d in capture["order"]["trade_decisions"]]
    bought = [d for d in again.trade_decisions if d["will_trade"] == "yes_auto"]
    prices = [price_door(next(r["timing"]["deal_price"] for r in capture["order"]["reviews"] if r["symbol"] == d["symbol"])) for d in bought]      # the Critic's deal price is a float of the review: it comes into the account through the price door
    assert [d["shares"] for d in bought] == robin(prices, Decimal("15000")) and [(d["symbol"], d["size"]["binding"], d["size"]["allocation"]) for d in bought] == [("CVX", "money", "round_robin"), ("EOG", "money", "round_robin")]
    assert all("spent round-robin" in d["reason"] for d in bought) and not any("balance" in d for d in bought)                          # no split was asked: not an even split
    assert sum(d["est_cost"] for d in bought) <= Decimal("15000") and Decimal("15000") - sum(d["est_cost"] for d in bought) < min(prices)               # the money is used until it buys no share of either
    assert [d["will_trade"] for d in again.trade_decisions if d["symbol"] == "PSX"] == ["no_system_denied"] and live["PSX"]["will_trade"] == "no_system_denied"


def test_the_real_fills_agree_with_the_decisions_pass_the_output_guard_and_the_account_arithmetic_adds_up(capture):
    order = wo.WorkOrder.model_validate(capture["order"])
    assert wo.verify_trades(order) == []
    decided = {d["record_id"]: d for d in order.trade_decisions if d["will_trade"] == "yes_auto"}
    assert [(t["symbol"], t["shares"], t["status"]) for t in order.trades] == [("CVX", 6, "filled"), ("EOG", 9, "filled")]
    for trade in order.trades:
        assert trade["shares"] <= decided[trade["record_id"]]["shares"] and wo.verify_fill(order, trade, order.trades) == []
    spent = sum(money_door(t["fill"]["amount"]) for t in order.trades)
    after = order.trader_runs[0]["portfolio_after"]
    assert money_door(after["cash"]) == Decimal("15000") - spent == Decimal("12254.41") and {p["symbol"]: p["shares"] for p in after["positions"]} == {"CVX": 6, "EOG": 9}


def test_the_session_is_the_new_flow_the_handshake_booked_each_fill(capture):
    run = capture["order"]["trader_runs"][0]
    assert run["booked"] is True and run["placed_by_model"] + run["placed_by_code"] == 2
    cash_before = run["portfolio_before"]["cash"]
    assert money_door(cash_before) == Decimal("15000") and money_door(run["portfolio_after"]["cash"]) == Decimal("12254.41")                                                     # as the live run left it (it was sized under the old limit)
