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
import tempfile
from datetime import date
from pathlib import Path

import pytest
import simple_critic as critic
from advisor_helpers import MemoryLedger, Scripted

from prometheus.agents.advisor.advisor import Advisor
from prometheus.agents.advisor.calls import parse_list
from prometheus.agents.advisor.record import Recommendation
from prometheus.common import work_order as wo
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


_RECORDED = tempfile.TemporaryDirectory(prefix="advisor_recorded_plan_")      # removed when the test run ends


def recorded_plan_config() -> Path:
    """The Advisor's settings with the TWO EXIT CONVENTIONS OF CONFIG VERSION 3 switched off (`trade.min_reward_to_risk` 0 and no `trailing_families`: the plan of config version 2, the target by the family's edge alone). The recorded model replies of the captures were
    written when the digest the model reads held THAT plan, and its thesis may name the target; the program's number check refuses a thesis that names a number the record does not have. So `replay` (and every test that builds on it: the Critic's, the Trader's,
    the Account Manager's) runs a recorded reply with the plan the reply was written for, the settings of one run changed and nothing else, and the target the tests pin (PG: 168.3783) is that plan's. The new plan is tested on the same captures in
    `tests/test_trade_plan_real.py` and by `replay(name, config_path=REPOSITORY_CONFIG)` below. A new capture (`scripts/capture_advisor_fixture.py`, paid) would make this unnecessary."""
    path = Path(_RECORDED.name) / "advisor_v1_recorded_plan.json"
    if not path.exists():
        settings = json.loads((REPO_ROOT / "configs" / "advisor_v1.json").read_text(encoding="utf-8"))
        settings["trade"] = {**settings["trade"], "min_reward_to_risk": 0, "trailing_families": []}
        path.write_text(json.dumps(settings), encoding="utf-8")
    return path


REPOSITORY_CONFIG = REPO_ROOT / "configs" / "advisor_v1.json"      # the Advisor's own settings: pass it to `replay` for the plan of the current config


def replay(name: str, plan_reply=None, config_path=None):
    """The step on the captured inputs. `name` is the fixture (SNDK, PG, PG_open); `plan_reply` replaces the model's plan reply (the thesis replies stay); `config_path` the Advisor's settings: when left out, the plan the recorded replies were written for (`recorded_plan_config`); `REPOSITORY_CONFIG` is the current one."""
    fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(fx["order"])
    replies = fx["replies"] if plan_reply is None else [plan_reply, *(r for r in fx["replies"] if '"theses"' in r)]
    advisor = Advisor(model=Scripted(*replies), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger(), config_path=config_path or recorded_plan_config())
    advisor.generate_recommendations(order, "funds", date.fromisoformat(fx["today"]))      # the Advisor is given NO portfolio (entry 105). The captures were made with the sample portfolio (MSFT, TSLA, GM); replayed without it the GM `add` and the two `hold` candidates are gone (see `test_the_sample_portfolios_holdings_are_not_in_the_run_any_more`)
    return order, fx


@pytest.fixture(scope="module")
def sndk():
    return replay("SNDK")


@pytest.fixture(scope="module")
def pg():
    return replay("PG")


# ---------------------------------------------------------------- the real inputs are what they say they are

def test_the_fixtures_are_real_captures_of_the_stored_data():
    for name in ("SNDK", "PG", "PG_open"):
        fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        symbol = fx["symbol"]
        order = wo.WorkOrder.model_validate(fx["order"])
        assert fx["today"] == "2026-11-05" and order.status == "data_ready" and len(order.data.sources) >= 10 and order.data.filing_evidence[symbol]["balance_sheet"] is not None
        assert fx["metrics"][symbol]["last_close"]["as_of"] == "2026-11-05" and len(fx["metrics"][symbol]["recent_bars"]["value"]) == 35     # bars through the capture day, from the REST refresh
        assert fx["replies"] and all(isinstance(r, str) for r in fx["replies"])


# ---------------------------------------------------------------- SNDK


def test_sndks_numbers_are_the_real_ones_with_their_sources(sndk):
    record = of(sndk[0], "SNDK")
    f = record["features"]
    assert f["share_count_change"]["value"] == pytest.approx(-0.0137) and f["cash_balance"]["value"] == 5136e6 and f["last_close"]["value"] == 806.43
    assert f["rsi_14"]["source"].startswith("day_aggs:2025-") and f["rsi_14"]["as_of"] == "2026-11-05" and record["identity"]["regime"] == "trend" and record["identity"]["sector_etf"] == "XLK"
    assert f["sector_ret_5d"]["value"] is not None and f["residual_5d"]["value"] == pytest.approx(-0.254, abs=1e-3)       # the sector ETF's bars, from the order's `sector`


def test_every_family_says_what_it_did_with_the_real_numbers(sndk):
    families = {f["family"]: f for f in of(sndk[0], "SNDK")["families"]}
    assert families["capital_return"]["passed"] and families["capital_return"]["features_used"]["share_count_change"] == pytest.approx(-0.0137) and families["capital_return"]["basis"] == "prior"
    assert families["trend"]["computable"] and not families["trend"]["passed"] and any("200-day" in n for n in families["trend"]["notes"])        # 276 bars: the long indicators exist now
    assert families["reversion"]["computable"] and not families["reversion"]["passed"] and any("not in an uptrend" in n for n in families["reversion"]["notes"])
    assert not families["event"]["computable"] and "verified bullish or bearish event" in families["event"]["missing"][0]
    gates = {g["gate"]: g for g in of(sndk[0], "SNDK")["gates"]}
    assert all(g["passed"] for g in gates.values()) and set(gates) == {"no_metrics", "freshness", "red_flag", "event_risk"}       # no liquidity gate (entry 93)


def test_sndks_trade_plan_is_checkable_by_hand(sndk):
    t = of(sndk[0], "SNDK")["trade"]
    assert t["family"] == "capital_return" and t["target_price"] is None and t["entry_zone"][0] < t["entry_mid"] < t["entry_zone"][1] and t["entry_mid"] == 806.43
    assert t["stop_price"] == pytest.approx(t["entry_mid"] - t["stop_distance"], abs=0.01) and t["stop_distance"] == pytest.approx(3 * 108.6778, rel=1e-3)   # 3 ATR of the real ATR
    assert t["time_stop_date"] == "2027-04-30" and t["order_style"] == "limit" and not SIZE_FIELDS & set(t)                          # prices and dates: the record says nothing of how much (entry 80)


# ---------------------------------------------------------------- PG


def test_pgs_real_filing_numbers_reach_the_capital_return_family(pg):
    _, fx = pg
    m = fx["metrics"]["PG"]
    assert m["payout_coverage"]["value"] == pytest.approx(0.547, abs=1e-3) and m["cfo_negative_with_profit"]["value"] is False and m["payout_coverage"]["source"].startswith("filing:0000080424-26-000256")      # the 10-Q filed 2026-10-22
    assert m["accrual_ratio"]["value"] == pytest.approx(-0.0076, abs=1e-3) and m["inventory_vs_sales"]["value"] == pytest.approx(-0.1157, abs=1e-3)               # P&G's own layout, read since the first real run


def test_the_simple_critic_catches_each_kind_of_defect(sndk):
    order, _ = sndk
    good = of(order, "SNDK")
    assert critic.critique(good, order).failures == []
    quote = good["evidence"]["signals"][0]
    cases = [
        ({"trade.stop_price": 900.0}, "trade.stop"), ({"trade.stop_price": 400.0}, "recompute.stop"), ({"trade.entry_zone": None}, "trade.entry"), ({"trade.order_style": None}, "trade.order_style"),
        ({"trade.invalidation": []}, "trade.invalidation"), ({"trade.time_stop_date": "2026-11-01"}, "trade.time_stop"), ({"score": 99}, "recompute.score"),
        ({"ranking.composite": 0.5}, "recompute.composite"), ({"agreement.confidence": 0.99}, "recompute.confidence"),
        ({"gates": [{"gate": "red_flag", "passed": False}]}, "gates"), ({"audit.metric_order": None}, "audit"),
        ({"thesis.bull_case": [{"text": "x", "ref": "made_up"}]}, "thesis.refs"),
        ({"features.roe.unit": None}, "complete.features"), ({"risk": None}, "complete"), ({"features.rsi_14.as_of": "2026-10-01"}, "features.fresh"),
    ]
    for change, name in cases:
        failures = critic.critique(critic.tampered(good, **change), order).failures
        assert any(f.startswith(name) for f in failures), (change, failures)
    forged = critic.tampered(good, **{"evidence.signals": [{**quote, "quote": "Sandisk will triple its profit by Friday"}]})
    assert any(f.startswith("evidence.quote") for f in critic.critique(forged, order).failures)                           # the Critic reads the order's own sources
    capital = next(i for i, f in enumerate(good["families"]) if f["family"] == "capital_return")
    used = critic.tampered(good, **{"families": [{**good["families"][capital], "features_used": {**good["families"][capital]["features_used"], "share_count_change": 0.5}}]})
    assert any(f.startswith("features.used") for f in critic.critique(used, order).failures)


def of(order, symbol):
    return next(r for r in order.recommendations if r["symbol"] == symbol)


