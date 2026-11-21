"""A REAL Auto session whose money is spent to the cent: $30,000.00 of $40,000.00, five stocks, $6,000.00 of each, and nothing above the budget (design choices entry 111).

`tests/fixtures/session_real/auto_even_split_dollars.json` is a capture of `session --name "Dollar Check" --age 60 --risk conservative --cash 40000 --mode auto --prompt "I want to buy oil stocks. Split $30,000 evenly among the 5 most
recommended companies." --yes --cleanup --save` (2026-11-16; NO fetch: the stored data; the real Account Manager, Data Clerk, Advisor (gpt-oss-120b), Critic (GLM 5.3) and Trader (gpt-oss-120b) over the real local Postgres, the simulated
close of Friday 2026-11-13; a throwaway account that the run deleted). The request asked for MORE companies than the Critic approves: it approved EOG (74), XOM (85) and CVX (82) and DENIED APA and COP (risk above the profile, and a finding that
asks for a person), so the Account Manager counted the Advisor's two best NEUTRAL companies, OXY and META, as approved (ALLOW_NEUTRAL_RECOMMENDATIONS) to make the five. The budget is smaller than the cash, so the pot is $30,000.00 and not $40,000.00: each
purchase is an order in dollars of $6,000.00, the Trader filled each at the quote, which is the price it was sized at (there is no slippage), and the account is left with $10,000.00. The recording that started entry 111 filled $30,015.00 of $30,000.00.
The tests decide the same order again, trade it again at the next session's REAL opens (some above the sizing price, some below), and read the reports.
"""
import copy
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_neutral_recommendation import account

from prometheus.application import session
from prometheus.application.web import views
from prometheus.agents.trader.report import trade_section
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price, rate
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_dollars.json"
PROFILE = UserProfile(user="Dollar Check", risk_tolerance="conservative", age=60, trade_mode="auto")
CASH = Portfolio(user="Dollar Check", cash_avail=Decimal("40000.00"))
EVENING = datetime(2026, 11, 12, 20, 30, tzinfo=timezone.utc)
OPENS = {"EOG": Decimal("157.25"), "XOM": Decimal("171.43"), "CVX": Decimal("218.275"), "OXY": Decimal("59.00"), "META": Decimal("627.21")}      # the real opens of 2026-11-13: XOM and OXY are above the price they were sized at, EOG, CVX and META below
SIZED_AT = {"EOG": Decimal("158.32"), "XOM": Decimal("170.59"), "CVX": Decimal("219.89"), "OXY": Decimal("58.75"), "META": Decimal("641.69")}

pytestmark = pytest.mark.usefixtures("friday_close")


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


def test_the_session_spent_30000_dollars_of_a_budget_of_30000_and_left_10000_dollars_of_the_cash(data):
    assert data["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
    assert (data["intent"]["action"], data["intent"]["count"], data["intent"]["allocation"], money(data["intent"]["budget"])) == ("buy", 5, "even", Decimal("30000.00"))
    order = wo.WorkOrder.model_validate(data["order"])
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []
    by = {d["symbol"]: d for d in order.trade_decisions}
    assert {s for s, d in by.items() if d["will_trade"] == "yes_auto"} == set(SIZED_AT) and {s for s, d in by.items() if d["will_trade"] == "no_system_denied"} == {"APA", "COP"}
    assert {s for s, d in by.items() if d.get("origin") == "neutral"} == {"OXY", "META"} and all(not d.get("origin") for s, d in by.items() if s in ("EOG", "XOM", "CVX"))
    live = [d for d in by.values() if d["will_trade"] == "yes_auto"]
    assert {money(d["dollars"]) for d in live} == {money(d["est_cost"]) for d in live} == {Decimal("6000.00")} and {money(d["size"]["pot"]) for d in live} == {Decimal("30000.00")} and not any(d["asked_user"] for d in live)      # a pot of $30,000.00, not the cash of $40,000.00 (a capture's numbers are floats after json: read through the doors)
    assert {s: price(d["price"]) for s, d in by.items() if s in SIZED_AT} == SIZED_AT and all(sh.quantity(d["shares"]) == sh.buys(Decimal("6000.00"), price(d["price"])) and "fractional shares" in d["size"]["notes"][0] for d in live)
    fills = {t["symbol"]: t for t in order.trades}
    assert {s: (t["status"], money(t["dollars"]), price(t["fill"]["price"]), money(t["fill"]["amount"]), rate(t["fill"]["slippage_bps"]), t["booked"]) for s, t in fills.items()} == {s: ("filled", Decimal("6000.00"), p, Decimal("6000.00"), 0, True) for s, p in SIZED_AT.items()}
    assert sum((money(t["fill"]["amount"]) for t in fills.values()), Decimal("0.00")) == Decimal("30000.00") and money(order.trader_runs[-1]["portfolio_after"]["cash"]) == Decimal("10000.00") and order.trader_runs[-1]["booked"] is True


def test_the_same_order_decided_again_by_the_real_account_manager_and_traded_again_by_the_real_trader_spends_exactly_the_budget(data, allowed):
    am = account(profile=PROFILE, fractional_even_split=None)
    order = am.decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", Decimal("30000.00"), count=5)
    assert [(d["symbol"], d["shares"], d["will_trade"]) for d in order.trade_decisions] == [(d["symbol"], sh.quantity(d["shares"]), d["will_trade"]) for d in data["order"]["trade_decisions"]]
    am.perform_trades(order, CASH, trader(), MOMENT)
    assert {t["symbol"]: t["fill"]["amount"] for t in order.trades} == {s: Decimal("6000.00") for s in SIZED_AT} and wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("10000.00")


def test_traded_at_the_next_sessions_real_opens_each_fill_is_still_its_dollars_more_shares_where_the_open_is_lower_and_fewer_where_it_is_higher(data, allowed):
    am = account(profile=PROFILE, fractional_even_split=None)
    order = am.decide_trades(reviewed(data), CASH, None, MOMENT, "auto", "even", Decimal("30000.00"), count=5)
    estimate = {d["symbol"]: d["shares"] for d in order.trade_decisions if d["will_trade"] == "yes_auto"}
    account(profile=PROFILE).perform_trades(order, CASH, trader(mode="backtest"), EVENING)
    fills = {t["symbol"]: t for t in order.trades}
    assert {s: (t["status"], t["fill"]["price"], t["fill"]["price_source"]) for s, t in fills.items()} == {s: ("filled", p, "bar_open") for s, p in OPENS.items()}
    assert {s: t["fill"]["amount"] for s, t in fills.items()} == {s: Decimal("6000.00") for s in OPENS} and sum((t["fill"]["amount"] for t in fills.values()), Decimal("0.00")) == Decimal("30000.00") <= Decimal("30000.00")
    assert {s: t["shares"] > estimate[s] for s, t in fills.items()} == {"EOG": True, "XOM": False, "CVX": True, "OXY": False, "META": True} and wo.verify_trades(order) == [] and order.errors == []


def test_the_reports_say_the_dollars_first_and_the_shares_as_an_estimate(data):
    order = wo.WorkOrder.model_validate(data["order"])
    before, after = CASH, Portfolio(user="Dollar Check", cash_avail=Decimal("10000.00"), positions=order.trader_runs[-1]["portfolio_after"]["positions"])
    decided = "\n".join(session.decision_lines(order, "auto"))
    assert "EOG    yes (AUTO)  $6,000.00, about 37.897928 shares at $158.32" in decided and "META   yes (AUTO)  $6,000.00, about 9.350309 shares at $641.69" in decided
    assert "EOG    $6,000.00 (about 37.897928 shares at $158.32) of the $40,000.00: stopped by the amount you asked to spend: its equal share was spent to the cent" in "\n".join(session.use_of_money_lines(order, before, after))
    trades = "\n".join(trade_section(order))
    assert "T1     buy  $6,000.00 of EOG" in trades and "FILLED 37.897928 at 158.3200 = $6,000.00" in trades and "T5     buy  $6,000.00 of META" in trades
    view = views.work_order_view(views.cut_order(data["order"], data["path"]), understood="x")
    rows = {r["key"]: r for r in view["rows"]}
    assert {s: (rows[s]["group"], money(rows[s]["decision"]["dollars"]), price(rows[s]["decision"]["price"]), money(rows[s]["trade"]["amount"])) for s in SIZED_AT} == {s: ("neutral" if s in ("OXY", "META") else "recommended", Decimal("6000.00"), p, Decimal("6000.00")) for s, p in SIZED_AT.items()}      # the page's rows carry the capture's numbers: through the doors
    assert money(rows["EOG"]["detail"]["order"]["decision"]["dollars"]) == Decimal("6000.00") and money(rows["OXY"]["detail"]["order"]["trade"]["amount"]) == Decimal("6000.00")
