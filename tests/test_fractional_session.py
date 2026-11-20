"""A REAL Auto session that split $30,000 evenly: $10,000.00 of each of three stocks, bought as fractions of a share, nothing above the budget (design choices entries 110 and 111).

`tests/fixtures/session_real/auto_even_split_fractions.json` is a capture of `session --name "Fraction Check" --age 60 --risk conservative --cash 31000 --mode auto --prompt "I want to buy oil stocks. Split $30,000 evenly
among the 3 most recommended companies." --fetch --yes --cleanup --save` (2026-11-20, 11:20 Eastern, the market open; LIVE quotes (`USE_LIVE_TRADING=true`, Finnhub) after a fetch of the news; the real Account Manager, Data
Clerk, Advisor (gpt-oss-120b) and Trader (gpt-oss-120b) over the real local Postgres, whose share columns are `numeric(18,6)`; a throwaway account that the run deleted). It RE-RECORDS the session of 2026-11-15 (EOG, CVX and
XOM at the simulated close of Friday 2026-11-13, recorded with 5 basis points of slippage: it spent $30,015.00 of a $30,000 budget, which started entry 111; that capture is in git history).
THE ADVISOR RECOMMENDED NONE of the oil companies it scored (all neutral, "no family passed": COP 64, CVX 63, APA 62, XOM 61, ...), so the Critic had nothing to review (the path has no `critic`, `reviews` is empty) and the
Account Manager counted the three best neutral companies as approved (`ALLOW_NEUTRAL_RECOMMENDATIONS`, `origin: "neutral"`; the theme check of entry 120 kept PG and META out of the candidates). It sized them at $10,000.00 each at the
live quotes (COP 137.15, CVX 209.53, APA 45.29, orders in dollars), and the Trader bought and booked them at those quotes: NO SLIPPAGE, each FILL cost exactly $10,000.00, $30,000.00 in all, and the account was left with
$1,000.00 of its $31,000.00. The tests decide the same order again at the quotes it was sized at, decide it and trade it again with the real Account Manager, Data Clerk and Trader at the stored closes of Thursday 2026-11-19
(the last complete stored daily bar before the session's day: the live quotes of 11:20 are in no stored bar), and read the reports and the page's views from the real records.
"""
import copy
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import clerk, trader
from test_neutral_recommendation import account

from prometheus.agents.account_manager import decisions, neutral
from prometheus.agents.trader.report import trade_section
from prometheus.application import session
from prometheus.application.web import views
from prometheus.application.web.replay import Replay, load_capture
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import jsonable, money, price, rate
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_fractions.json"
PROFILE = UserProfile(user="Fraction Check", risk_tolerance="conservative", age=60, trade_mode="auto")
CASH = Portfolio(user="Fraction Check", cash_avail=Decimal("31000.00"))
SHARES = {"COP": Decimal("72.912869"), "CVX": Decimal("47.725862"), "APA": Decimal("220.799293")}
QUOTES = {"COP": Decimal("137.15"), "CVX": Decimal("209.53"), "APA": Decimal("45.29")}      # the live quotes of 2026-11-20, 11:20 Eastern: what each decision was sized at and each fill was made at
SESSION_MOMENT = datetime(2026, 11, 20, 16, 20, 36, tzinfo=timezone.utc)      # the moment of the Account Manager's decision (`decided_utc` of the capture)

DAY = date(2026, 11, 19)      # the last COMPLETE stored daily bar before the session's day (the bar of the 20th is the partial day the fetch saw)
DAY_MOMENT = datetime(2026, 11, 19, 20, 30, tzinfo=timezone.utc)      # 15:30 Eastern on Thursday 2026-11-19
CLOSES = {"COP": Decimal("134.99"), "CVX": Decimal("206.50"), "APA": Decimal("44.62")}      # the real stored closes of that day
AT_CLOSES = {"COP": Decimal("74.079561"), "CVX": Decimal("48.42615"), "APA": Decimal("224.114746")}      # what $10,000.00 buys of each at those closes, six decimals, never more


@pytest.fixture(scope="module")
def data():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


@pytest.fixture
def stored_closes(monkeypatch):
    """The real Data Clerk's price is the stored close of Thursday 2026-11-19 (the simulated clock: `USE_LIVE_TRADING=false`, design choices entry 76, and the day pinned): read from the stored daily bars (`data/historical`,
    read only, the day files and the Massive per-ticker bars the last fetch added), marked `simulated`; Finnhub is never asked. Skips when those bars are not on this machine."""
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: DAY)
    try:
        for symbol in CLOSES:
            clerk().tools.close_quote(symbol, DAY)
    except Exception:
        pytest.skip("the stored daily bars of COP, CVX and APA for 2026-11-19 are not on this machine (data/historical)")


def advised(data) -> wo.WorkOrder:
    """The session's order as the Advisor left it, before the Account Manager decided: the Advisor's candidates (all neutral), no review, no decision, no trade."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "recommended", [], [], []
    return order


def decided_at_the_sessions_quotes(data, *, fractional: bool) -> wo.WorkOrder:
    """The Account Manager's decision step over the session's own recorded quotes: the neutral companies it chose (`neutral.choose`, the 3 asked for) as purchases at the price each was sized at, one sizing, $30,000.00 split evenly."""
    order = advised(data)
    chosen, _ = neutral.choose(neutral.neutral_candidates(order), 0, 3)
    purchases = neutral.purchases_for(order, chosen, QUOTES)
    return decisions.decide_trades(order, CASH, PROFILE, None, SESSION_MOMENT, "auto", "even", Decimal("30000.00"), None, None, neutral_purchases=list(zip(chosen, purchases)), fractional=fractional)


def test_the_session_split_30000_dollars_evenly_into_three_purchases_of_exactly_10000_dollars_of_fractional_shares(data):
    assert data["path"] == ["account_manager", "data_clerk", "advisor", "account_manager_decision", "trader"]      # the Advisor recommended nothing: no Critic node ran
    assert (data["intent"]["action"], data["intent"]["count"], data["intent"]["allocation"], money(data["intent"]["budget"])) == ("buy", 3, "even", Decimal("30000.00"))
    order = wo.WorkOrder.model_validate(data["order"])
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []      # nothing above the pot
    assert order.recommendations == [] and order.reviews == [] and {c["stance"] for c in order.candidates} == {"neutral"}
    assert [(c["symbol"], c["score"]) for c in order.candidates[:3]] == [("COP", 64), ("CVX", 63), ("APA", 62)] and not {"PG", "META"} & {c["symbol"] for c in order.candidates}      # entry 120: PG and META are not oil companies
    by = {d["symbol"]: d for d in order.trade_decisions}
    assert list(by) == ["COP", "CVX", "APA"] and {s: sh.quantity(d["shares"]) for s, d in by.items()} == SHARES and all(not sh.whole(d["shares"]) for d in by.values()) and {money(d["est_cost"]) for d in by.values()} == {Decimal("10000.00")}      # a capture's numbers are floats after json: read through the doors
    assert {d["will_trade"] for d in by.values()} == {"yes_auto"} and {d["origin"] for d in by.values()} == {"neutral"} and not any(d["asked_user"] for d in by.values()) and {d["size"]["binding"] for d in by.values()} == {"budget"}
    assert {d["size"]["fractional"] for d in by.values()} == {True} and {(money(d["balance"]["target_dollars"]), d["balance"]["of"]) for d in by.values()} == {(Decimal("10000.00"), 3)} and "all of it is spent" in by["COP"]["size"]["notes"][1]
    assert {money(d["dollars"]) for d in by.values()} == {Decimal("10000.00")} and {money(d["size"]["pot"]) for d in by.values()} == {Decimal("30000.00")}      # orders in dollars, under a pot of $30,000.00 (the budget; the cash is $31,000.00)
    prices = {s: price(d["price"]) for s, d in by.items()}      # a neutral purchase has no Critic deal price: the decision's own price is the live quote it was sized at, through the price door
    assert prices == QUOTES and {s: sh.buys(Decimal("10000.00"), prices[s]) for s in SHARES} == SHARES and {s: trade_amount(SHARES[s], prices[s]) for s in SHARES} == {s: Decimal("10000.00") for s in SHARES}      # the shares are what $10,000.00 buys at the price each was sized at
    assert {t["symbol"]: (sh.quantity(t["shares"]), sh.quantity(t["fill"]["shares"]), t["booked"]) for t in order.trades} == {s: (n, n, True) for s, n in SHARES.items()}
    assert {s: (price(t["fill"]["price"]), money(t["fill"]["amount"]), rate(t["fill"]["slippage_bps"]), t["fill"]["price_source"]) for t in order.trades for s in [t["symbol"]]} == {s: (QUOTES[s], Decimal("10000.00"), 0, "quote") for s in SHARES}      # NO SLIPPAGE: each fill is at the quote and costs exactly its $10,000.00
    assert sum((money(t["fill"]["amount"]) for t in order.trades), Decimal("0.00")) == Decimal("30000.00") and money(order.trader_runs[-1]["portfolio_after"]["cash"]) == Decimal("1000.00") and order.trader_runs[-1]["booked"] is True


def test_the_same_order_decided_again_at_the_quotes_it_was_sized_at_gives_the_same_three_decisions(data, monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    out = decided_at_the_sessions_quotes(data, fractional=True)
    live = {d["symbol"]: d for d in data["order"]["trade_decisions"]}      # the capture's own numbers are floats after json: they are read through the doors below
    assert out.status == "decided" and out.errors == [] and {d["symbol"]: (d["shares"], d["est_cost"], d["dollars"], d["will_trade"], d["origin"]) for d in out.trade_decisions} == {s: (SHARES[s], Decimal("10000.00"), Decimal("10000.00"), "yes_auto", "neutral") for s in SHARES}
    assert [(d["symbol"], d["shares"]) for d in out.trade_decisions] == [(d["symbol"], sh.quantity(d["shares"])) for d in data["order"]["trade_decisions"]] and live["COP"]["balance"] == jsonable(out.trade_decisions[0]["balance"])
    assert [jsonable(d["size"]) for d in out.trade_decisions] == [d["size"] for d in data["order"]["trade_decisions"]] and wo.verify_trades(out) == [] and wo.pot_reasons(out) == []      # the sizing record is the same, caps and notes too
    off = decided_at_the_sessions_quotes(data, fractional=False)
    assert all(type(d["shares"]) is Decimal and sh.whole(d["shares"]) for d in off.trade_decisions) and sum((d["est_cost"] for d in off.trade_decisions), Decimal("0.00")) <= Decimal("30000.00")           # the switch off: the whole-share split of entry 108
    assert not any(d.get("dollars") for d in off.trade_decisions)      # with fractions off nothing is an order in dollars (entry 111)


def test_the_same_order_decided_and_traded_again_by_the_real_account_manager_data_clerk_and_trader_at_the_stored_closes_spends_exactly_the_budget(data, stored_closes):
    am = account(profile=PROFILE, fractional_even_split=None)
    order = am.decide_trades(advised(data), CASH, None, DAY_MOMENT, "auto", "even", Decimal("30000.00"), count=3)
    assert [(d["symbol"], d["will_trade"], d["origin"], d["shares"], d["est_cost"], d["dollars"], d["price"]) for d in order.trade_decisions] == [(s, "yes_auto", "neutral", AT_CLOSES[s], Decimal("10000.00"), Decimal("10000.00"), CLOSES[s]) for s in SHARES]      # the same three companies, in the same order, each $10,000.00: the Data Clerk's stored close is the price the Account Manager sized at
    am.perform_trades(order, CASH, trader(), DAY_MOMENT)
    got = {t["symbol"]: t for t in order.trades}
    assert {s: (t["status"], t["shares"], t["fill"]["price"], t["fill"]["amount"], t["fill"]["slippage_bps"]) for s, t in got.items()} == {s: ("filled", AT_CLOSES[s], CLOSES[s], Decimal("10000.00"), 0) for s in SHARES} and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []
    after = order.trader_runs[-1]["portfolio_after"]
    assert sum((t["fill"]["amount"] for t in got.values()), Decimal("0.00")) == Decimal("30000.00") and after["cash"] == Decimal("1000.00") and {p["symbol"]: p["shares"] for p in after["positions"]} == AT_CLOSES      # $30,000.00 of the budget of $30,000, $1,000.00 of the cash left


def test_the_reports_and_the_pages_views_print_the_fractions_and_the_changes_do_not_crash(data):
    order = wo.WorkOrder.model_validate(data["order"])
    before, after = Portfolio(user="Fraction Check", cash_avail=Decimal("31000.00")), Portfolio(user="Fraction Check", cash_avail=Decimal("1000.00"), positions=order.trader_runs[-1]["portfolio_after"]["positions"])
    neutrals = "\n".join(session.neutral_lines(order))      # a run of neutral companies has no review, so the session prints no "HOW MUCH WAS BOUGHT" block (it needs `order.reviews`): the neutral block says what was ordered
    assert "COP                             Advisor score 64 neutral   BUY (treated as approved, Auto): $10,000.00, about 72.912869 shares at $137.15" in neutrals and "APA                             Advisor score 62 neutral   BUY (treated as approved, Auto): $10,000.00, about 220.799293 shares at $45.29" in neutrals
    assert any("COP" in line and "$10,000.00, about 72.912869 shares at $137.15" in line for line in session.decision_lines(order, "auto"))
    trades = "\n".join(trade_section(order))
    assert "T1     buy  $10,000.00 of COP" in trades and "FILLED 72.912869 at 137.1500 = $10,000.00" in trades and "FILLED 220.799293 at 45.2900 = $10,000.00" in trades and "cash $31,000.00 -> $1,000.00; APA 0 -> 220.799293 shares; COP 0 -> 72.912869 shares; CVX 0 -> 47.725862 shares" in trades
    assert [line for line in session.portfolio_lines(after) if "COP" in line][0].startswith("  COP        72.912869 shares at an average cost of")
    replay = Replay(load_capture(CAPTURE))
    start, end = replay.account(), replay.account(final=True)
    assert views.portfolio_changes(start, end) == {"changes": SHARES, "cash_change": Decimal("-30000.00"), "cash_before": Decimal("31000.00")} and {p["symbol"]: p["shares"] for p in end["positions"]} == SHARES
    verdict = views.run_verdict(start, end, action="buy", order=order)
    assert verdict["status"] == "traded" and verdict["headline"].startswith("The portfolio changed: +220.799293 APA, +72.912869 COP, +47.725862 CVX.")


def test_a_run_of_neutral_companies_alone_still_reports_how_much_was_bought(data):
    """Entry 123: the run has no Critic review (the Advisor recommended none; the three are neutral, counted as approved), and the HOW MUCH WAS BOUGHT section still lists each purchase and the even split."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    assert order.reviews == [] and [symbol for symbol, _ in session.sized_purchases(order)] == ["COP", "CVX", "APA"]
    said = "\n".join(session.use_of_money_lines(order, CASH, Portfolio(user="Fraction Check", cash_avail=Decimal("1000.00"))))
    assert "HOW MUCH WAS BOUGHT  spent $30,000.00 of $31,000.00" in said and "equal to the cent" in said
    assert "COP    $10,000.00 (about 72.912869 shares at $137.15) of the $31,000.00: stopped by the amount you asked to spend: its equal share was spent to the cent" in said
