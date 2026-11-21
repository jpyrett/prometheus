"""An exit plan for a neutral purchase (Advisor config version 2, design/critic_v2_evidence_spec.md section 7): a stop 2 ATR under the last close and a time stop 20 trading days on, no target, nothing about the entry. On the REAL captures of
the Advisor (`tests/fixtures/advisor_real`), the REAL neutral session of `tests/fixtures/session_real/pam_smith_neutral.json` put through the real Trader in backtest, and the REAL stored bars.

What is proved: the numbers (and that the calendar knows Thanksgiving); who gets a plan (a neutral company no gate stopped, and nobody else); that the Trader's tickets of a neutral purchase are byte for byte what they were; that the Proving Ground follows a
neutral position by it (`record.plan_of`, `SimPortfolio`, `outcomes.score`) and the Live view says so; and that the engine's one-exit-rule check does not fail a neutral position the account sold by it (the Trader's ticket has no plan for it).
"""
import copy
import json
from datetime import date
from decimal import Decimal

from advisor_helpers import MemoryLedger, Scripted
from handshake_helpers import MOMENT, clerk, manager, trader
from proving_helpers import CAPTURES, real_bars, stored_bars_available
from test_advisor_sndk_pg import FIXTURES, RecordedClerk

import pytest

from prometheus.agents.advisor import trade_plan as tp
from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.config import load_config
from prometheus.agents.advisor.record import Candidate
from prometheus.common import trading_clock
from prometheus.common import work_order as wo
from prometheus.common.money import dumps
from prometheus.common.portfolio import Portfolio
from prometheus.proving import engine, outcomes, recs, record
from prometheus.proving.portfolio import LotInfo, SimPortfolio

CONFIG = load_config()
PG = json.loads((FIXTURES / "PG.json").read_text(encoding="utf-8"))
needs_bars = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars are not on this machine")


def test_the_stop_is_two_atr_under_the_last_close_and_the_time_stop_twenty_trading_days_on_with_no_target():
    """The real PG numbers of 2026-11-05 (last close 167.10, ATR 3.2883): the stop is 167.10 - 2 x 3.2883. Twenty TRADING days after the 5th is 2026-12-04, not the 3rd: the exchange is closed on Thanksgiving, which a count of weekdays does not know."""
    metrics = PG["metrics"]["PG"]
    plan = tp.neutral_exit_plan(metrics, date(2026, 11, 5), CONFIG)
    assert plan.stop_price == round(metrics["last_close"]["value"] - 2.0 * metrics["atr_14"]["value"], 4) == 160.5234
    assert plan.time_stop_date == "2026-12-04" and tp.add_weekdays(date(2026, 11, 5), 20).isoformat() == "2026-12-03"
    assert set(plan.model_dump()) == {"stop_price", "time_stop_date", "basis"} and "no target" in plan.basis and "2 x atr_14 (3.29)" in plan.basis and "20 trading days after 2026-11-05" in plan.basis
    gone = {k: v for k, v in metrics.items() if k != "atr_14"}
    blind = tp.neutral_exit_plan(gone, date(2026, 11, 5), CONFIG)
    assert blind.stop_price is None and blind.time_stop_date == "2026-12-04" and blind.basis.startswith("no stop: atr_14 is missing") and "time stop alone stands" in blind.basis
    tight = {**metrics, "atr_14": {**metrics["atr_14"], "value": 100.0}}                     # an ATR so large that 2 ATR is under zero: no stop above zero
    assert tp.neutral_exit_plan(tight, date(2026, 11, 5), CONFIG).stop_price is None
    assert (CONFIG.config_version, CONFIG.neutral.stop_atr_multiple, CONFIG.neutral.horizon_bars) == ("2", 2.0, 20)


def advised_neutral():
    """The real PG capture with ONE change, so that no family passes (its RSI is 80, over the 75 the trend and conviction strategies allow) and the model says nothing: PG is a company the Advisor scores NEUTRAL, no gate stopped it."""
    fx = copy.deepcopy(PG)
    fx["metrics"]["PG"]["rsi_14"]["value"] = 80.0
    order = wo.WorkOrder.model_validate(fx["order"])
    replies = [json.dumps({"companies": []}), *(r for r in fx["replies"] if '"theses"' in r)]
    Advisor(model=Scripted(*replies), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger()).generate_recommendations(order, "funds", date.fromisoformat(fx["today"]))
    return order


def test_the_advisor_gives_a_neutral_company_a_plan_and_nobody_else_one():
    order = advised_neutral()
    by_symbol = {c["symbol"]: c for c in order.candidates}
    pg, addyy = by_symbol["PG"], by_symbol["ADDYY"]
    assert pg["stance"] == "neutral" and pg["gate"] is None and wo.is_neutral_candidate(pg) and order.recommendations == []
    assert pg["exit_plan"] == {"stop_price": 160.5234, "time_stop_date": "2026-12-04", "basis": pg["exit_plan"]["basis"]} and Candidate.model_validate(pg).exit_plan.stop_price == 160.5234
    assert addyy["stance"] == "neutral" and addyy["gate"] == "freshness" and addyy["exit_plan"] is None             # a gate stopped it: it is never a neutral purchase, so no plan
    order2 = PG_open_order()
    assert order2.recommendations and all("exit_plan" not in r for r in order2.recommendations)                      # a recommendation has its trade plan and no exit plan


def PG_open_order():
    from test_advisor_sndk_pg import replay
    return replay("PG_open")[0]


def real_neutral_order():
    """The real neutral session (MU and AMD scored neutral by the Advisor on 2026-11-13, bought as approved) with an exit plan on each of its candidates, made by the Advisor's own function from the REAL stored bars of that day through the Data Clerk's metric door."""
    data = json.loads((CAPTURES / "pam_smith_neutral.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    with trading_clock.as_of(date(2026, 11, 13)):
        asked = wo.new_work_order("m", requester="test", action_type="calculate_financials", subject=["MU", "AMD"], metrics=[{"name": "last_close"}, {"name": "atr_14"}])
        clerk().request_metrics(asked, date(2026, 11, 13))
    for candidate in order.candidates:
        candidate["exit_plan"] = tp.neutral_exit_plan(asked.data.metrics[candidate["symbol"]], date(2026, 11, 13), CONFIG).model_dump()
    return order


def traded(order, mutate=None):
    order = order.model_copy(deep=True)
    if mutate:
        mutate(order)
    order.status, order.trades, order.trader_runs = "decided", [], []
    manager().perform_trades(order, Portfolio(user="Bob Jones", cash_avail=Decimal("40000.00")), trader(mode="backtest"), MOMENT)
    return order


@needs_bars
def test_the_traders_tickets_of_a_neutral_purchase_are_byte_for_byte_what_they_were():
    """The same real neutral decisions through the real Trader in backtest, with the candidates' exit plans and without them: the same trades, field for field (a market order at the next open, the limit range of the quote). The plan is an EXIT plan and nothing else."""
    with_plan = real_neutral_order()
    assert all(c["exit_plan"]["stop_price"] and c["exit_plan"]["time_stop_date"] > "2026-11-13" for c in with_plan.candidates)
    without = with_plan.model_copy(deep=True)
    for c in without.candidates:
        c.pop("exit_plan")
    a, b = traded(with_plan), traded(without)
    runs = lambda order: json.loads(dumps([{k: v for k, v in r.items() if not k.endswith("_order")} for r in order.trader_runs]))      # the ids of the Data Clerk's requests are new each time; nothing else of a run is
    assert dumps(a.trades) == dumps(b.trades) and runs(a) == runs(b) and [t["order_type"] for t in a.trades] == ["market", "market"] and len(a.trades) == 2


@needs_bars
def test_the_proving_ground_follows_a_neutral_position_by_its_stop_and_its_time_stop():
    order = real_neutral_order()
    mu = recs.neutral_record_of(order.candidates, "MU", "neutral-d76d39703eed-0")
    plan = record.plan_of(mu)
    candidate = next(c for c in order.candidates if c["symbol"] == "MU")
    assert plan.stop == Decimal(str(candidate["exit_plan"]["stop_price"])).quantize(Decimal("0.0001")) and plan.target is None and plan.time_stop_date.isoformat() == candidate["exit_plan"]["time_stop_date"]
    assert (plan.entry_low, plan.entry_high, plan.entry_mid, plan.limit_price, plan.order_style) == (None, None, None, None, None)           # nothing about the entry
    trade_plan_wins = {**mu, "trade": {"stop_price": 1.5, "target_price": 9.0, "time_stop_date": "2026-12-01"}}
    assert record.plan_of(trade_plan_wins).stop == Decimal("1.5") and record.plan_of({**mu, "exit_plan": None}).stop is None                    # a trade plan keeps its own; no exit plan, no plan (as before)
    bars = real_bars(["MU", "AMD", "SPY"])
    day = date(2026, 11, 13)
    followed = outcomes.score(plan, bars["MU"], bars["SPY"], None, day, [5, 10, 20], symbol="MU")
    assert followed.plan.stop == plan.stop and followed.result in ("open", "lost", "won") and followed.hypothetical
    # a stop above the price (one change): the position leaves at once, by its stop, which the Trader's own follower knows nothing about
    high = copy.deepcopy(order)
    for c in high.candidates:
        c["exit_plan"]["stop_price"] = 2000.0
    bought = traded(high)
    account = SimPortfolio(Decimal("40000.00"), user="Bob Jones")
    info = {t["record_id"]: LotInfo(plan=record.plan_of(recs.neutral_record_of(high.candidates, t["symbol"], t["record_id"])), key=f"{day}:{t['record_id']}", name=t["symbol"], decision_day=day) for t in bought.trades}
    account.apply_fills(bought.trades, info)
    exits = []
    for session in sorted(b["date"] for b in bars["SPY"] if "2026-11-16" <= b["date"] <= "2026-11-19"):
        exits += account.step(date.fromisoformat(session), bars)
    assert {e.symbol: e.reason for e in exits} == {"MU": "stop", "AMD": "stop"} and account.lots == [] and all(t["outcome"]["status"] == "open" for t in bought.trades)
    notes = engine.check_exits(bought.trades, account, {})                     # the one-exit-rule check: a neutral purchase is followed by its exit plan, which the Trader's follower does not have
    assert any("neutral purchase" in n and "exit plan" in n for n in notes)
    with pytest.raises(engine.ExitRuleError):                                   # and for any other purchase the same disagreement is still a failed run
        engine.check_exits([{**t, "record_id": "abc123"} for t in bought.trades], account, {})


def test_the_live_view_says_a_neutral_purchase_has_a_stop_and_a_time_stop_and_an_older_one_says_the_advisor_made_none():
    from prometheus.proving import live
    order = real_neutral_order()
    candidates = [c for c in order.candidates]
    capture = recs.Capture("fixture", "x.json", {"order": json.loads(order.model_dump_json(exclude={"data"}))}, 0)
    index = {}
    for decision in capture.payload["order"]["trade_decisions"]:
        if decision.get("origin") == "neutral":
            index[decision["record_id"]] = (recs.neutral_record_of(capture.payload["order"]["candidates"], decision["symbol"], decision["record_id"]), capture)
    buys = [{"symbol": "MU", "side": "buy", "traded_on": date(2026, 11, 16), "trade_id": 1, "record_id": "neutral-d76d39703eed-0"}]
    plan, record_id, since, notes = live._plan_for("MU", buys, index)
    assert plan is not None and plan.target is None and plan.stop == Decimal(str(candidates[0]["exit_plan"]["stop_price"])) and plan.time_stop_date.isoformat() == candidates[0]["exit_plan"]["time_stop_date"]
    assert record_id == "neutral-d76d39703eed-0" and notes == ["bought as a company the Advisor scored neutral: it has a stop and a time stop, no target"]
    old_plan, _, _, old_notes = live._plan_for("MU", buys, {})                           # a purchase of a run saved before: no exit plan is found
    assert old_plan is None and old_notes == ["bought as a company the Advisor scored neutral: it made no plan (no stop or target) for it"]
