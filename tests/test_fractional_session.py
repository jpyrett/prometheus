"""A REAL Auto session that split $30,000 evenly: $10,000.00 of each of three stocks, bought as fractions of a share (design choices entry 110).

`tests/fixtures/session_real/auto_even_split_fractions.json` is a capture of `session --name "Fraction Check" --age 60 --risk conservative --cash 31000 --mode auto --prompt "I want to buy oil stocks. Split $30,000 evenly
among the 3 most recommended companies." --yes --cleanup --save` (2026-11-15; NO fetch: the stored data; the real Account Manager, Data Clerk, Advisor (gpt-oss-120b), Critic (GLM 5.3) and Trader (gpt-oss-120b) over the
real local Postgres, whose share columns are `numeric(18,6)`; the simulated close of Friday 2026-11-13; a throwaway account that the run deleted). The Critic approved EOG (79), CVX (87) and XOM (85); the Account Manager sized
them at $10,000.00 each, the Trader bought and booked them at the quote of that moment, which is five basis points above the close, so each FILL cost $10,005.00 (design/fractional_shares_spec.md, open question 2). The tests
decide the same order again with the real Account Manager, trade it again with the real Trader (the stored closes), and read the reports and the page's views from the real records.
"""
import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_neutral_recommendation import account

from prometheus.agents.trader.report import trade_section
from prometheus.application import session
from prometheus.application.web import views
from prometheus.application.web.replay import Replay, load_capture
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import jsonable, money, price
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_fractions.json"
PROFILE = UserProfile(user="Fraction Check", risk_tolerance="conservative", age=60, trade_mode="auto")
CASH = Portfolio(user="Fraction Check", cash_avail=Decimal("31000.00"))
SHARES = {"EOG": Decimal("63.163213"), "CVX": Decimal("45.477284"), "XOM": Decimal("58.620083")}


@pytest.fixture(scope="module")
def data():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def reviewed(data) -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def test_the_session_split_30000_dollars_evenly_into_three_purchases_of_exactly_10000_dollars_of_fractional_shares(data):
    assert data["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
    assert (data["intent"]["action"], data["intent"]["count"], data["intent"]["allocation"], money(data["intent"]["budget"])) == ("buy", 3, "even", Decimal("30000.00"))
    order = wo.WorkOrder.model_validate(data["order"])
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == []
    by = {d["symbol"]: d for d in order.trade_decisions}
    assert {s: sh.quantity(d["shares"]) for s, d in by.items()} == SHARES and all(not sh.whole(d["shares"]) for d in by.values()) and {money(d["est_cost"]) for d in by.values()} == {Decimal("10000.00")}      # a capture's numbers are floats after json: read through the doors
    assert {d["will_trade"] for d in by.values()} == {"yes_auto"} and not any(d["asked_user"] for d in by.values()) and {d["size"]["binding"] for d in by.values()} == {"budget"}
    assert {d["size"]["fractional"] for d in by.values()} == {True} and {(money(d["balance"]["target_dollars"]), d["balance"]["of"]) for d in by.values()} == {(Decimal("10000.00"), 3)} and "all of it is spent" in by["EOG"]["size"]["notes"][1]
    prices = {r["symbol"]: price(r["timing"]["deal_price"]) for r in order.reviews}      # the Critic's deal price is a float in the record: it goes through the price door
    assert {s: sh.buys(Decimal("10000.00"), prices[s]) for s in SHARES} == SHARES and {s: trade_amount(SHARES[s], prices[s]) for s in SHARES} == {s: Decimal("10000.00") for s in SHARES}      # the shares are what $10,000.00 buys at the price each was sized at
    assert {s: (sh.quantity(t["shares"]), sh.quantity(t["fill"]["shares"]), t["booked"]) for t in order.trades for s in [t["symbol"]]} == {s: (n, n, True) for s, n in SHARES.items()}
    assert {money(t["fill"]["amount"]) for t in order.trades} == {Decimal("10005.00")} and money(order.trader_runs[-1]["portfolio_after"]["cash"]) == Decimal("985.00") and order.trader_runs[-1]["booked"] is True      # THE RECORDING WAS MADE WITH 5 bps OF SLIPPAGE (before entry 111): $10,005.00 each, $30,015.00 in all on a budget of $30,000, $985.00 left of the $31,000


def test_the_same_order_decided_again_by_the_real_account_manager_gives_the_same_three_decisions(data, monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    out = account(profile=PROFILE, fractional_even_split=None).decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", Decimal("30000.00"), count=3)
    live = {d["symbol"]: d for d in data["order"]["trade_decisions"]}      # the capture's own numbers are floats after json: they are read through the doors below
    assert {d["symbol"]: (d["shares"], d["est_cost"], d["will_trade"]) for d in out.trade_decisions if d["will_trade"] == "yes_auto"} == {s: (SHARES[s], Decimal("10000.00"), "yes_auto") for s in SHARES}
    assert [(d["symbol"], d["shares"]) for d in out.trade_decisions] == [(d["symbol"], sh.quantity(d["shares"])) for d in data["order"]["trade_decisions"]] and live["EOG"]["balance"] == jsonable(out.trade_decisions[0]["balance"])
    off = account(profile=PROFILE, fractional_even_split=False).decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", Decimal("30000.00"), count=3)
    assert all(type(d["shares"]) is Decimal and sh.whole(d["shares"]) for d in off.trade_decisions) and sum((d["est_cost"] for d in off.trade_decisions), Decimal("0.00")) <= Decimal("30000.00")           # the switch off: the whole-share split of entry 108


def test_the_same_order_traded_again_by_the_real_trader_fills_the_same_fractions_and_the_account_shows_them(data, friday_close, monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    am = account(profile=PROFILE, fractional_even_split=None)
    order = am.decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", Decimal("30000.00"), count=3)
    am.perform_trades(order, CASH, trader(), MOMENT)
    got = {t["symbol"]: t for t in order.trades}
    # AMENDED (entry 111): the recording filled $10,005.00 of each (5 bps of slippage, $30,015.00 on a budget of $30,000: Jacob: "It is impossible to spend more than the amount in the portfolio and budget"); traded again with none
    # (`costs.slippage_bps` is 0) it fills $10,000.00 of each and $30,000.00 in all
    assert {s: (t["status"], t["shares"], t["fill"]["amount"]) for s, t in got.items()} == {s: ("filled", n, Decimal("10000.00")) for s, n in SHARES.items()} and wo.verify_trades(order) == []
    after = order.trader_runs[-1]["portfolio_after"]
    assert after["cash"] == Decimal("1000.00") and {p["symbol"]: p["shares"] for p in after["positions"]} == SHARES


def test_the_reports_and_the_pages_views_print_the_fractions_and_the_changes_do_not_crash(data):
    order = wo.WorkOrder.model_validate(data["order"])
    before, after = Portfolio(user="Fraction Check", cash_avail=Decimal("31000.00")), Portfolio(user="Fraction Check", cash_avail=Decimal("985.00"), positions=order.trader_runs[-1]["portfolio_after"]["positions"])
    said = "\n".join(session.use_of_money_lines(order, before, after))
    assert "EOG    63.163213 shares ($10,000.00) of the $31,000.00: stopped by the amount you asked to spend: its equal share was spent to the cent" in said and "equal to the cent" in said and "as far as whole shares allow" not in said
    assert any("EOG" in line and "63.163213 shares, about $10,000.00" in line for line in session.decision_lines(order, "auto"))
    trades = "\n".join(trade_section(order))
    assert "T1     buy  63.163213 of 63.163213 EOG" in trades and "FILLED 63.163213 at 158.3992 = $10,005.00" in trades and "CVX 0 -> 45.477284 shares; EOG 0 -> 63.163213 shares; XOM 0 -> 58.620083 shares" in trades
    assert [line for line in session.portfolio_lines(after) if "EOG" in line][0].startswith("  EOG       63.163213 shares at an average cost of")
    replay = Replay(load_capture(CAPTURE))
    start, end = replay.account(), replay.account(final=True)
    assert views.portfolio_changes(start, end) == {"changes": SHARES | {}, "cash_change": Decimal("-30015.00"), "cash_before": Decimal("31000.00")} and {p["symbol"]: p["shares"] for p in end["positions"]} == SHARES
    verdict = views.run_verdict(start, end, action="buy", order=order)
    assert verdict["status"] == "traded" and verdict["headline"].startswith("The portfolio changed: +45.477284 CVX, +63.163213 EOG, +58.620083 XOM.")
