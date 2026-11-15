"""SNDK (Sandisk) and PG (Procter & Gamble) through the Advisor on REAL data, judged by a very simple Critic (tests/simple_critic.py).

The inputs are not invented. `scripts/capture_advisor_fixture.py` ran the real step for each symbol on 2026-11-05 and saved what went in (tests/fixtures/advisor_real/<SYMBOL>.json):
the work order the Data Clerk produced from the stored news and 10-Q filings (real sources, real filing evidence), the metrics the Data Clerk calculated (real daily bars of 276 sessions,
Finnhub's real figures for the day, the real 10-Q statements) and every raw reply of the Advisor's model (the plan call and the thesis call, and a retry of either when there was one). These tests feed them back through the real Advisor code with a
model that returns the recorded replies: the same numbers and words, offline and free. A new capture (`--yes`: three paid APIs) changes the fixture and may change what a test pins:
PG sits near the line (a strategy passed or not depending on small differences): in five live runs it was a recommendation twice and neutral three times, depending on how
the model worded its quotes. So there are two captures of PG, `PG.json` (neutral) and `PG_open.json` (a recommendation), and a test for each.

What is asked of each: does the symbol score as a recommendation, and does the record carry everything a Critic needs to judge it?
"""
import json
from datetime import date
from pathlib import Path

import pytest
import simple_critic as critic
from advisor_helpers import MemoryLedger, Scripted

from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.calls import parse_list
from prometheus.agents.advisor.record import Recommendation
from prometheus.common import work_order as wo
from prometheus.common.portfolio import load_portfolio
from prometheus.rag.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures" / "advisor_real"
SIZE_FIELDS = {"equity", "equity_basis", "risk_budget", "suggested_shares", "position_pct", "oversized", "size_note", "funds_feasible", "funds_note"}     # what the plan had before entry 80 removed the size


class RecordedClerk:
    """The Data Clerk's metrics of the capture: what `request_metrics` returned for each symbol."""
    tools = None

    def __init__(self, metrics):
        self.metrics = metrics

    def request_metrics(self, order, today=None):
        order.data = wo.DataResult(metrics={s: self.metrics[s] for s in order.subject if s in self.metrics})
        order.status = "data_ready"
        return order


def replay(name: str, plan_reply=None):
    """The step on the captured inputs. `name` is the fixture (SNDK, PG, PG_open); `plan_reply` replaces the model's plan reply (the thesis replies stay)."""
    fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(fx["order"])
    replies = fx["replies"] if plan_reply is None else [plan_reply, *(r for r in fx["replies"] if '"theses"' in r)]
    advisor = Advisor(model=Scripted(*replies), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger())
    advisor.generate_recommendations(order, load_portfolio(REPO_ROOT / fx["portfolio"]), "funds", date.fromisoformat(fx["today"]))
    return order, fx


@pytest.fixture(scope="module")
def sndk():
    return replay("SNDK")


@pytest.fixture(scope="module")
def pg():
    return replay("PG")


@pytest.fixture(scope="module")
def pg_open():
    return replay("PG_open")


def scrub(record):
    record = json.loads(json.dumps(record))
    record["identity"]["record_id"] = record["identity"]["request_id"] = record["audit"]["metric_order"] = None
    return record


# ---------------------------------------------------------------- the real inputs are what they say they are

def test_the_fixtures_are_real_captures_of_the_stored_data():
    for name in ("SNDK", "PG", "PG_open"):
        fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        symbol = fx["symbol"]
        order = wo.WorkOrder.model_validate(fx["order"])
        assert fx["today"] == "2026-11-05" and order.status == "data_ready" and len(order.data.sources) >= 10 and order.data.filing_evidence[symbol]["balance_sheet"] is not None
        assert fx["metrics"][symbol]["last_close"]["as_of"] == "2026-11-05" and len(fx["metrics"][symbol]["recent_bars"]["value"]) == 35     # bars through the capture day, from the REST refresh
        assert fx["metrics"][symbol]["market_cap"]["source"].startswith("finnhub_metric:2026-11-0") and fx["replies"] and all(isinstance(r, str) for r in fx["replies"])


# ---------------------------------------------------------------- SNDK

def test_sndk_scores_as_a_recommendation_and_the_critic_finds_it_complete_and_arithmetically_sound(sndk):
    order, _ = sndk
    assert order.status == "recommended" and order.errors == []
    new = [r for r in order.recommendations if r["symbol"] == "SNDK"]
    assert len(new) == 1
    record = new[0]
    Recommendation.model_validate(record)
    assert record["action"]["action"] == "open" and record["stance"] == "bullish" and record["supported"] is True and record["score"] > 50 and record["identity"]["case"] == "funds"
    result = critic.critique(record, order)
    assert result.failures == [] and len(result.checked) >= 55                                       # a defect would be in `failures`; many separate checks were made
    assert result.verdict == "review"                                                                # the position itself is what the Critic questions, below


def test_the_critic_questions_a_stock_that_fell_by_half_and_is_recommended_on_a_buyback(sndk):
    order, _ = sndk
    result = critic.critique(order.recommendations[0], order)
    text = " | ".join(result.warnings)
    assert "below its 200-day mean" in text and "far under its 52-week high (34% of it)" in text and "a -57% drawdown in 63 days" in text and "the stop is 40% away" in text
    assert "counter-evidence on file" in text                                                         # the model's own bearish quote ("down 45% from its high") is in the record


def test_sndks_numbers_are_the_real_ones_with_their_sources(sndk):
    record = sndk[0].recommendations[0]
    f = record["features"]
    assert f["net_payout_yield_ttm"]["value"] == pytest.approx(0.1355, abs=1e-3) and f["net_payout_yield_ttm"]["source"].startswith("derived:filing:0001628280-26-068601+finnhub_metric:2026-11-0")
    assert f["share_count_change"]["value"] == pytest.approx(-0.0137) and f["cash_balance"]["value"] == 5136e6 and f["last_close"]["value"] == 806.43 and f["market_cap"]["unit"] == "USD"
    assert f["rsi_14"]["source"].startswith("day_aggs:2025-") and f["rsi_14"]["as_of"] == "2026-11-05" and record["identity"]["regime"] == "trend" and record["identity"]["sector_etf"] == "XLK"
    assert f["sector_ret_5d"]["value"] is not None and f["residual_5d"]["value"] == pytest.approx(-0.254, abs=1e-3)       # the sector ETF's bars, from the order's `sector`


def test_every_family_says_what_it_did_with_the_real_numbers(sndk):
    families = {f["family"]: f for f in sndk[0].recommendations[0]["families"]}
    assert families["capital_return"]["passed"] and families["capital_return"]["features_used"]["net_payout_yield_ttm"] > 0.13 and families["capital_return"]["basis"] == "prior"
    assert families["trend"]["computable"] and not families["trend"]["passed"] and any("200-day" in n for n in families["trend"]["notes"])        # 276 bars: the long indicators exist now
    assert families["reversion"]["computable"] and not families["reversion"]["passed"] and any("not in an uptrend" in n for n in families["reversion"]["notes"])
    assert not families["event"]["computable"] and "verified bullish or bearish event" in families["event"]["missing"][0]
    gates = {g["gate"]: g for g in sndk[0].recommendations[0]["gates"]}
    assert all(g["passed"] for g in gates.values()) and gates["liquidity"]["value"]["adv_20d_dollars"] > 1e9


def test_sndks_trade_plan_is_checkable_by_hand(sndk):
    t = sndk[0].recommendations[0]["trade"]
    assert t["family"] == "capital_return" and t["target_price"] is None and t["entry_zone"][0] < t["entry_mid"] < t["entry_zone"][1] and t["entry_mid"] == 806.43
    assert t["stop_price"] == pytest.approx(t["entry_mid"] - t["stop_distance"], abs=0.01) and t["stop_distance"] == pytest.approx(3 * 108.6778, rel=1e-3)   # 3 ATR of the real ATR
    assert t["time_stop_date"] == "2027-04-30" and t["order_style"] == "limit" and not SIZE_FIELDS & set(t)                          # prices and dates: the record says nothing of how much (entry 80)


def test_the_holdings_of_the_sample_portfolio_were_reviewed_and_left_alone(sndk):
    held = {c["symbol"]: c for c in sndk[0].candidates}
    assert set(held) >= {"MSFT", "TSLA", "GM"} and all(held[s]["held"] and held[s]["reason"].startswith("hold:") for s in ("MSFT", "TSLA", "GM"))


# ---------------------------------------------------------------- PG

def test_pg_in_the_neutral_capture_is_a_recommendation_on_conviction_because_the_blend_is_high(pg):
    order, fx = pg
    record = next(r for r in order.recommendations if r["symbol"] == "PG")
    assert order.status == "recommended" and order.errors == [] and record["action"]["action"] == "open" and record["stance"] == "bullish"
    families = {f["family"]: f for f in record["families"]}
    assert families["capital_return"]["passed"] and families["capital_return"]["direction"] == "bullish" and 0 < families["capital_return"]["edge_20d"] < 0.002   # its edge alone is smaller than the 0.20% cost
    assert not families["trend"]["passed"] and not families["reversion"]["passed"]
    conviction = families["conviction"]                                                    # Amendment 5: the blend (0.55, score 78) is over 0.30 with news, so the blend is a strategy of its own
    assert conviction["passed"] and conviction["score"] == 0.5 and conviction["basis_ref"] == "borrowed_from_trend" and conviction["features_used"] == {"rsi_14": pytest.approx(conviction["features_used"]["rsi_14"])}
    assert conviction["edge_20d"] == pytest.approx(0.006 * conviction["regime_multiplier"] * 0.5 * 0.85, rel=1e-6) and record["trade"]["family"] == "conviction" and record["trade"]["horizon_bars"] == 40
    ranking, blend = record["ranking"], record["blend"]
    assert ranking["composite"] == conviction["edge_20d"] and "excess_20d_after_cost" not in ranking and "cost_pct" not in ranking and "scale_factor" not in ranking     # Amendment 7: no expected return on the record
    assert blend["score"] == 78 and blend["components"]["news"]["n"] == 3 and record["score"] == 78 and record["agreement"]["confidence"] == pytest.approx((1 + blend["value"]) / 2, abs=1e-3)
    result = critic.critique(record, order)
    assert result.failures == []
    with pytest.raises(ValueError):                                                                          # the real model's first plan reply was cut short (no closing brace): not valid JSON
        parse_list(fx["replies"][0], "companies")
    assert order.advisor_runs[0]["calls"] == 2 and order.advisor_runs[0]["error"] is None                    # the retry, also recorded, worked


def test_pgs_real_filing_numbers_reach_the_capital_return_family(pg):
    _, fx = pg
    m = fx["metrics"]["PG"]
    assert m["net_payout_yield_ttm"]["value"] == pytest.approx(0.0392, abs=1e-3) and m["payout_coverage"]["value"] == pytest.approx(0.547, abs=1e-3) and m["cfo_negative_with_profit"]["value"] is False
    assert m["market_cap"]["value"] == pytest.approx(393.3e9, rel=1e-3) and m["net_payout_yield_ttm"]["source"].startswith("derived:filing:0000080424-26-000256")   # the 10-Q filed 2026-10-22
    assert m["accrual_ratio"]["value"] == pytest.approx(-0.0076, abs=1e-3) and m["inventory_vs_sales"]["value"] == pytest.approx(-0.1157, abs=1e-3)               # P&G's own layout, read since the first real run


def test_pg_in_the_other_capture_is_a_recommendation_and_the_critic_finds_it_complete(pg_open):
    order, _ = pg_open
    assert order.status == "recommended" and order.errors == []
    records = [r for r in order.recommendations if r["symbol"] == "PG"]
    assert len(records) == 1
    record = records[0]
    Recommendation.model_validate(record)
    assert record["action"]["action"] == "open" and record["stance"] == "bullish" and record["score"] > 50 and record["trade"]["family"] == "conviction"
    result = critic.critique(record, order)
    assert result.failures == [] and len(result.checked) >= 55 and result.verdict in ("review", "approve")
    assert record["thesis"]["thesis"] and record["thesis"]["thesis_error"] is None and "unmeasured" in record["thesis"]["thesis"].lower()                  # the real model's thesis passed the number and ref checks
    families = {f["family"]: f for f in record["families"]}
    assert families["capital_return"]["passed"] and families["capital_return"]["direction"] == "bullish" and families["trend"]["computable"] and not families["trend"]["passed"]
    t = record["trade"]
    assert t["entry_mid"] == 167.1 and t["stop_price"] == pytest.approx(t["entry_mid"] - t["stop_distance"], abs=0.01) and not SIZE_FIELDS & set(t)


def test_a_dilution_passage_of_the_10q_does_not_turn_pgs_capital_return_bearish(pg_open):
    """The captured Procter & Gamble record stays bullish on capital return. Issuance is no longer a family vote (entry 70); this checks the stored record, which was written before that."""
    order, fx = pg_open
    evidence = order.data.filing_evidence["PG"]["evidence"]
    note = next(i for i in evidence if i["topic"] == "dilution" and not i["derived"] and i["source"]["form"] == "10-Q")
    assert "dilutive" in note["quote"].lower()
    record = next(r for r in order.recommendations if r["symbol"] == "PG")
    assert {f["family"]: f["direction"] for f in record["families"] if f["passed"]}["capital_return"] == "bullish"


def test_the_simple_critic_catches_each_kind_of_defect(sndk):
    order, _ = sndk
    good = order.recommendations[0]
    assert critic.critique(good, order).failures == []
    quote = good["evidence"]["signals"][0]
    cases = [
        ({"trade.stop_price": 900.0}, "trade.stop"), ({"trade.stop_price": 400.0}, "recompute.stop"), ({"trade.entry_zone": None}, "trade.entry"), ({"trade.order_style": None}, "trade.order_style"),
        ({"trade.invalidation": []}, "trade.invalidation"), ({"trade.time_stop_date": "2026-11-01"}, "trade.time_stop"), ({"score": 99}, "recompute.score"),
        ({"ranking.composite": 0.5}, "recompute.composite"), ({"agreement.confidence": 0.99}, "recompute.confidence"),
        ({"gates": [{"gate": "liquidity", "passed": False}]}, "gates"), ({"audit.metric_order": None}, "audit"),
        ({"thesis.bull_case": [{"text": "x", "ref": "made_up"}]}, "thesis.refs"),
        ({"features.roe.unit": None}, "complete.features"), ({"risk": None}, "complete"), ({"features.rsi_14.as_of": "2026-10-01"}, "features.fresh"),
    ]
    for change, name in cases:
        failures = critic.critique(critic.tampered(good, **change), order).failures
        assert any(f.startswith(name) for f in failures), (change, failures)
    forged = critic.tampered(good, **{"evidence.signals": [{**quote, "quote": "Sandisk will triple its profit by Friday"}]})
    assert any(f.startswith("evidence.quote") for f in critic.critique(forged, order).failures)                           # the Critic reads the order's own sources
    used = critic.tampered(good, **{"families": [{**good["families"][3], "features_used": {**good["families"][3]["features_used"], "net_payout_yield_ttm": 0.5}}]})
    assert any(f.startswith("features.used") for f in critic.critique(used, order).failures)


def of(order, symbol):
    return next(r for r in order.recommendations if r["symbol"] == symbol)


def test_the_replay_is_deterministic_and_the_model_owns_only_its_own_fields(sndk):
    again, _ = replay("SNDK")
    assert scrub(again.recommendations[0]) == scrub(sndk[0].recommendations[0])
    changed, _ = replay("SNDK", plan_reply=json.dumps({"companies": []}))                        # a model that says nothing: the numbers of the record are the same, the text evidence is not
    one, two = scrub(of(sndk[0], "SNDK")), scrub(of(changed, "SNDK"))                            # (by symbol: a held name can be recommended too, see the next test)
    for section in ("features", "gates", "quality", "risk", "identity"):
        assert one[section] == two[section]
    assert two["signals"] == [] and one["signals_verified"] >= 1 and two["trade"]["stop_price"] == one["trade"]["stop_price"]


def test_a_held_name_as_good_as_the_new_ideas_is_an_add_with_no_amount_and_the_user_limits_are_the_account_managers(sndk):
    """Entry 80: the holdings pass no longer asks whether the position after adding stays within 20% of the equity (it needed a size and the user's money). With the model saying nothing (no text evidence, so
    SNDK's blend falls to 56 and the bar for an add with it), the held GM, bullish at 60, is an `add` on the REAL capture; with the model's evidence (SNDK 60) GM is below the new idea and stays a hold.
    The `add` carries a plan of prices and no amount. Whether the user's own limits leave room for more GM is decided by `sizing.size_purchases`, which counts the shares held against the position limit."""
    base_gm = next(c for c in sndk[0].candidates if c["symbol"] == "GM")
    assert base_gm["held"] and base_gm["reason"] == "hold: bullish, but below this run's new ideas"
    changed, _ = replay("SNDK", plan_reply=json.dumps({"companies": []}))
    gm = of(changed, "GM")
    assert gm["action"]["action"] == "add" and gm["action"]["held_shares"] > 0 and "as good as this run's new ideas" in gm["action"]["reason"] and gm["score"] == 60
    assert gm["trade"]["family"] and gm["trade"]["stop_price"] and not SIZE_FIELDS & set(gm["trade"]) and "position_pct_after" not in gm["portfolio_fit"]
    Recommendation.model_validate(gm)
