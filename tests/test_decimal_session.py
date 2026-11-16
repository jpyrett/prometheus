"""A REAL Auto session whose money is Decimal from the request to the book, to the cent (design choices entry 112; design/decimal_money_spec.md section 7).

`tests/fixtures/session_real/auto_even_split_decimal.json` is a capture of `session --name "Decimal Check" --age 60 --risk conservative --cash 40000.50 --mode auto --prompt "I want to buy oil stocks. Split $12,345.67 evenly among the 3 most recommended
companies." --yes --cleanup --save` (2026-11-16; NO fetch: the stored data; the real Account Manager, Data Clerk, Advisor (gpt-oss-120b), Critic (GLM 5.3) and Trader (gpt-oss-120b) over the real local Postgres, the simulated close of Friday 2026-11-13; a
throwaway account that the run deleted, and the three samples did not move). Every number the real agents made went through the doors and the encoder: a budget of $12,345.67 and a cash of $40,000.50 with cents, an even split of three that shares the odd cents
(EOG $4,115.23, XOM $4,115.22, META $4,115.22: the Critic approved EOG and XOM and denied APA, so the Advisor's best neutral company, META, made the three), fractional shares at six decimals, a fill at the quote of each (no slippage), and a cash of $27,654.83 left in the
book. The tests read the recording through the doors (its numbers are the floats `json` gave), decide its real reviews again with the real Account Manager and trade them again with the real Trader at the stored closes: the same dollars, the same shares, exactly.
"""
import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, friday_close, manager, trader  # noqa: F401  (friday_close is a fixture)

from prometheus.agents.trader.report import trade_section
from prometheus.application import session
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_decimal.json"
PROFILE = UserProfile(user="Decimal Check", risk_tolerance="conservative", age=60, trade_mode="auto")
CASH = Portfolio(user="Decimal Check", cash_avail=Decimal("40000.50"))
BUDGET, LEFT = Decimal("12345.67"), Decimal("27654.83")
DOLLARS = {"EOG": Decimal("4115.23"), "XOM": Decimal("4115.22"), "META": Decimal("4115.22")}        # the odd cent went to the first purchase, in rank order
SIZED_AT = {"EOG": Decimal("158.3200"), "XOM": Decimal("170.5900"), "META": Decimal("641.6900")}


@pytest.fixture(scope="module")
def data():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")


def reviewed(data) -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def test_the_recording_says_what_the_real_agents_made_and_every_number_passes_its_door_exactly(data):
    assert data["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
    assert (data["intent"]["action"], data["intent"]["count"], data["intent"]["allocation"]) == ("buy", 3, "even") and money(data["intent"]["budget"]) == BUDGET
    order = wo.WorkOrder.model_validate(data["order"])
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []
    by = {d["symbol"]: d for d in order.trade_decisions}
    assert {s for s, d in by.items() if d["will_trade"] == "yes_auto"} == set(DOLLARS) and by["APA"]["will_trade"] == "no_system_denied" and by["META"]["origin"] == "neutral"
    live = [by[s] for s in DOLLARS]
    assert {s: money(by[s]["dollars"]) for s in DOLLARS} == DOLLARS and {money(d["size"]["pot"]) for d in live} == {BUDGET}
    assert sum((money(d["dollars"]) for d in live), Decimal("0.00")) == BUDGET                           # an even split of three over $12,345.67: the amounts add up to the pot, to the cent
    assert {s: price(by[s]["price"]) for s in DOLLARS} == SIZED_AT
    for s, d in by.items():
        if s in DOLLARS:
            assert money(d["est_cost"]) == money(d["dollars"]) == trade_amount(sh.quantity(d["shares"]), price(d["price"])) and sh.quantity(d["shares"]) == sh.buys(money(d["dollars"]), price(d["price"]))
    fills = {t["symbol"]: t for t in order.trades}
    assert {s: (t["status"], money(t["fill"]["amount"]), price(t["fill"]["price"]), money(t["dollars"]), t["fill"]["slippage_bps"], t["booked"]) for s, t in fills.items()} \
        == {s: ("filled", DOLLARS[s], SIZED_AT[s], DOLLARS[s], 0, True) for s in DOLLARS}
    assert all(sh.quantity(t["fill"]["shares"]) == sh.quantity(by[s]["shares"]) and money(t["fill"]["amount"]) == trade_amount(sh.quantity(t["fill"]["shares"]), price(t["fill"]["price"])) for s, t in fills.items())
    run = order.trader_runs[-1]
    assert money(run["portfolio_before"]["cash"]) == CASH.cash_avail and money(run["portfolio_after"]["cash"]) == LEFT and run["booked"] is True
    assert CASH.cash_avail - sum((money(t["fill"]["amount"]) for t in fills.values()), Decimal("0.00")) == LEFT        # the cash after is the cash before less the fills, exactly


def test_the_same_reviews_decided_again_and_traded_again_at_the_stored_closes_give_the_same_dollars_the_same_shares_and_the_same_cash(data, allowed, friday_close):
    am = manager(profile=PROFILE, fractional_even_split=None)
    order = am.decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", BUDGET, count=3)
    made = [d for d in order.trade_decisions if d["will_trade"] == "yes_auto"]
    assert [(d["symbol"], d["dollars"], d["shares"], d["price"]) for d in made] == [(s, DOLLARS[s], sh.quantity(next(x for x in data["order"]["trade_decisions"] if x["symbol"] == s)["shares"]), SIZED_AT[s]) for s in DOLLARS]
    assert all(isinstance(d["dollars"], Decimal) and isinstance(d["shares"], Decimal) and isinstance(d["est_cost"], Decimal) and d["size"]["pot"] == BUDGET for d in made)
    am.perform_trades(order, CASH, trader(), MOMENT)
    fills = {t["symbol"]: t["fill"] for t in order.trades}
    assert {s: f["amount"] for s, f in fills.items()} == DOLLARS and {s: f["price"] for s, f in fills.items()} == SIZED_AT
    assert wo.verify_trades(order) == [] and order.errors == [] and order.trader_runs[-1]["portfolio_after"]["cash"] == LEFT == am.current_portfolio().cash_avail


def test_the_reports_and_the_page_read_the_same_numbers_as_text_and_as_json_numbers(data):
    order = wo.WorkOrder.model_validate(data["order"])
    decided = "\n".join(session.decision_lines(order, "auto"))
    assert "EOG    yes (AUTO)  $4,115.23, about 25.993115 shares at $158.32" in decided and "META   yes (AUTO)  $4,115.22, about 6.413096 shares at $641.69" in decided
    trades = "\n".join(trade_section(order))
    assert "T1     buy  $4,115.23 of EOG" in trades and "FILLED 25.993115 at 158.3200 = $4,115.23" in trades and "FILLED 6.413096 at 641.6900 = $4,115.22" in trades
    assert "cash $40,000.50 -> $27,654.83" in trades
    again = json.loads(order.model_dump_json())
    assert all(isinstance(t["fill"]["amount"], (int, float)) and not isinstance(t["fill"]["amount"], str) for t in again["trades"])
    assert [t["fill"]["amount"] for t in again["trades"]] == [4115.23, 4115.22, 4115.22]                    # numbers, the same ones the recording holds
