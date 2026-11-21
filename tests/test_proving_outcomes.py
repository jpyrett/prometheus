"""`outcomes.score`, `splits` and the readers of a recommendation (design/proving_ground_spec.md sections 5, 8 and 14; design choices entry 126; P4, P5, P8, P10).

REAL DATA ONLY. The recommendations are the 28 distinct records of the real captures (`tests/fixtures/session_real/`) and of the real ledger (`data/ledger/recommendations.jsonl`), made on 2026-11-06, 2026-11-12 and 2026-11-13, and the bars are the stored daily
bars (`data/historical/day_aggs_v1`, read only) to the finished session of 2026-11-19 (`proving_helpers.real_bars`). A machine without those bars skips these tests. Nothing here calls a model or a paid API.

THE FEW DAYS SINCE NOVEMBER DO NOT EXERCISE EVERY RULE, so where a case needs a longer time (a time stop, a limit the day never reached, a gap through a stop or a target, a split) the test takes a REAL plan of those records and puts its OWN
distances (its stop, target, entry zone and limit price as fractions of its entry price, and its horizon in sessions, by the Advisor's own `add_weekdays`) on an EARLIER REAL entry day of the same symbol in the stored bars: `restated()`. Every
price that is then compared with is a real stored bar; only the plan's distances are carried back. The splits are real too (Netflix 10 for 1 on 2025-11-17 is in the stored bars: a close of 1112.17 and an open of 110.75); the table is written in the
shape `splits.py` documents, from those public facts, and the test checks it against the bars.

Each exit is read THREE ways: by `outcomes.score`, by `simulator.follow` called by hand on the same bars, and by a ten-line reading of the plan's rules written in this file (the stop before the target, only the stop on the entry bar, the time stop
at the first bar on or after its date).
"""
import json
from datetime import date
from decimal import Decimal

import pytest
from proving_helpers import FIRST, LAST, close_on, real_bars, real_records, restated, stored_bars_available

from prometheus.agents.trader import simulator
from prometheus.common.money import price as price_door
from prometheus.proving import outcomes, record, splits
from prometheus.proving.config import load_config

pytestmark = pytest.mark.skipif(not stored_bars_available(), reason="the stored daily bars (data/historical/day_aggs_v1) are not on this machine")

HORIZONS = [5, 10, 20]


def d(x) -> Decimal:
    return Decimal(str(x))


def score(real_plan, bars, symbol, day: date, sector=None, **more):
    return outcomes.score(real_plan, bars[symbol], bars["SPY"], bars.get(sector) if sector else None, day, HORIZONS, symbol=symbol, **more)


def test_no_bar_no_outcome_and_a_price_is_never_invented():
    """A symbol with no stored bar (a London listing) and a decision on the last stored day (the future is not stored) are `no_data`, each with its reason, no entry, no exit and no number; and a decision whose next session is the last stored one
    is `open` with the one day it has."""
    reals = {r.record_id: r for r in real_records()}
    plan = record.plan_of(reals["8b78a7afdbcc"].rec)
    bars = real_bars({"XOM", "MSFT", "SHEL.L"})
    assert bars["SHEL.L"] == []
    nothing = outcomes.score(plan, bars["SHEL.L"], bars["SPY"], None, date(2026, 11, 13), HORIZONS, symbol="SHEL.L")
    assert (nothing.result, nothing.entry, nothing.exit, nothing.return_pct, nothing.path, nothing.notes) == ("no_data", None, None, None, [], ["no daily bar of SHEL.L is stored"])
    last = outcomes.score(plan, bars["XOM"], bars["SPY"], None, LAST, HORIZONS, symbol="XOM")
    assert (last.result, last.entry, last.return_pct, last.notes) == ("no_data", None, None, [f"no daily bar is stored after {LAST.isoformat()} yet: the future is not stored"])
    one = outcomes.score(record.plan_of(reals["be3e391cac5f"].rec), bars["MSFT"], bars["SPY"], None, date(2026, 11, 18), HORIZONS, symbol="MSFT")
    assert one.result == "open" and one.entry.date == LAST and len(one.path) == 1 and one.path[0].date == LAST


def test_a_stock_split_is_adjusted_from_the_stored_table_and_a_suspected_split_with_no_table_is_not_scored(tmp_path):
    """Netflix split 10 for 1 on 2025-11-17: the stored bars close at 1112.17 on the 14th and open at 110.75 on the 17th. NO REAL PLAN EXISTS for Netflix, so the real AMD plan's distances are put on its real bars (see the top of this file).
    With the table (read from a file in the documented shape) a decision made the day before the split and bought ON the split day reads at 1107.50, the plan's own scale, and the target of that scale ends it the next day; read raw, the same bars
    are a stop at the open and a loss of 90%; with no table the outcome is `no_data` and says why."""
    reals = {r.record_id: r for r in real_records()}
    bars = real_bars({"NFLX", "ORLY"})
    day = date(2025, 11, 14)
    net = bars["NFLX"]
    before, after = next(b for b in net if b["date"] == "2025-11-14"), next(b for b in net if b["date"] == "2025-11-17")
    assert round(before["close"] / after["open"], 1) == 10.0                                              # the table below is checked against the bars: a tenth
    (tmp_path / "NFLX.json").write_text(json.dumps({"symbol": "NFLX", "fetched_utc": "2026-11-20T18:00:00Z", "provider": "massive", "splits": [{"execution_date": "2025-11-17", "split_from": 1, "split_to": 10, "id": "x"}]}), encoding="utf-8")
    table, why = splits.read("NFLX", tmp_path)
    assert why == "" and table.symbol == "NFLX" and splits.load("NFLX", tmp_path) == table and table.splits[0].factor == 10
    assert splits.read("ORLY", tmp_path) == (None, "no split table is stored for ORLY") and splits.load("ORLY", tmp_path) is None
    (tmp_path / "ORLY.json").write_text("{not json", encoding="utf-8")
    assert splits.read("ORLY", tmp_path) == (None, "the split table of ORLY could not be read (JSONDecodeError)")
    (tmp_path / "ORLY.json").write_text(json.dumps({"symbol": "NFLX", "splits": []}), encoding="utf-8")
    assert splits.read("ORLY", tmp_path) == (None, "the split table of ORLY is for NFLX")
    plan = restated(record.plan_of(reals["83910656bece"].rec), close_on(net, "2025-11-14"), day)
    assert (plan.limit_price, plan.target, plan.stop) == (Decimal("1112.1700"), Decimal("1122.1794"), Decimal("850.1847"))
    out = outcomes.score(plan, net, bars["SPY"], None, day, HORIZONS, symbol="NFLX", splits=table)
    assert (out.result, out.entry.date.isoformat(), str(out.entry.price), out.exit.date.isoformat(), str(out.exit.price), out.exit.reason) == ("won", "2025-11-17", "1107.5000", "2025-11-18", "1122.1794", "target")
    assert out.split_adjusted and out.notes == ["adjusted for the 10-for-1 split of 2025-11-17"] and out.return_pct == 0.013255 and out.path[0].close == price_door(d(after["close"]) * 10)
    raw = simulator.follow(price_door(after["open"]), Decimal(1), "2025-11-17", net, plan.stop, plan.target, plan.time_stop_date.isoformat(), Decimal(0))
    assert (raw.exit_reason, raw.exit_date, raw.return_pct) == ("stop", "2025-11-17", 0.0) and plan.stop > price_door(after["open"])        # the plan's scale against a post-split price: stopped at the open without a loss of its own
    blind = outcomes.score(plan, net, bars["SPY"], None, day, HORIZONS, symbol="NFLX", table_missing="no split table is stored for NFLX")
    assert blind.result == "no_data" and not blind.split_adjusted and blind.entry is None and blind.notes == ["NFLX moved -90.0% from one close to the next open on 2025-11-17, which looks like a split, and no split table is stored for NFLX: it is not scored across it"]
    found = splits.suspect_move(net, day, date(2025, 12, 31), load_config().split_suspect_pct)
    assert found[0] == date(2025, 11, 17) and abs(found[1] + Decimal("0.9")) < Decimal("0.01")
    assert splits.suspect_move(net, date(2025, 11, 17), date(2025, 12, 31), load_config().split_suspect_pct) is None                    # the move is the entry's own night, not on the way
    shares, prices = table.adjust(Decimal(8), {"stop": plan.stop, "target": plan.target, "cost_basis": None}, date(2025, 11, 17))
    assert (shares, prices) == (Decimal(80), {"stop": Decimal("85.0185"), "target": Decimal("112.2179"), "cost_basis": None}) and table.adjust(Decimal(8), {}, date(2025, 11, 18)) == (Decimal(8), {})
    assert table.factor_between(date(2025, 11, 14), date(2025, 11, 20)) == 10 and table.factor_between(date(2025, 11, 17), date(2025, 11, 20)) == 1


def test_a_recommendation_is_read_as_the_record_has_it_in_schema_2_and_3_and_the_verdict_is_the_critics_or_the_advisors():
    """`plan_of` on a schema 3 record of a work order (MSFT of the quoted-theme session) and a schema 2 record of the ledger (Quick Service's), through the price door; the verdicts of three real orders: the Critic approved or denied, the Advisor's neutral
    companies the Account Manager decided on, and what nobody reviewed."""
    reals = {r.record_id: r for r in real_records()}
    msft, qsr = reals["be3e391cac5f"], reals["a497639f3127"]
    assert (msft.rec["identity"]["schema_version"], qsr.rec["identity"]["schema_version"]) == (3, 2)
    assert record.plan_of(msft.rec) == record.PlanView(entry_low=Decimal("516.2709"), entry_high=Decimal("531.4891"), entry_mid=Decimal("523.88"), limit_price=Decimal("523.88"), stop=Decimal("478.2255"), target=Decimal("528.5949"),
                                                       time_stop_date=date(2027, 1, 8), horizon_bars=40, order_style="limit")
    plan = record.plan_of(qsr.rec)
    assert (plan.stop, plan.target, plan.time_stop_date, plan.horizon_bars) == (Decimal("74.7294"), Decimal("79.5762"), date(2027, 1, 1), 40)
    sndk = record.plan_of(reals["816aea79fcda"].rec)
    assert sndk.target is None and sndk.stop == Decimal("564.6116") and record.plan_of({"symbol": "MU", "stance": "neutral"}) == record.PlanView() and record.plan_of(None) == record.PlanView()      # a capital-return plan has no target; a neutral company has no plan
    who = record.identity_of(msft.order, msft.rec, msft.day)
    assert (who["key"], who["symbol"], who["verdict"], who["critic"].approved, who["critic"].grade, who["advisor"].family) == ("2026-11-13:be3e391cac5f", "MSFT", "approved", True, who["critic"].grade, "conviction") and who["critic"].score == 81
    assert [record.verdict_of(reals[r].order, r) for r in ("937bfd681be3", "8b78a7afdbcc", "816aea79fcda", "6d8959fcf54e")] == ["denied", "approved", "denied", "denied"]     # APA (risk), XOM, SNDK (risk), PG (the market was closed)
    assert [record.verdict_of(reals["816aea79fcda"].order, d["record_id"]) for d in reals["816aea79fcda"].order.trade_decisions] == ["denied", "neutral", "neutral"]
    assert record.verdict_of(reals["816aea79fcda"].order, "not-a-record") == "unreviewed"
    assert record.caveat_sentences(["small_sample"]) == [record.CAVEATS["small_sample"]] and len(record.CAVEATS) == 12 and "fast_models" in record.CAVEATS and "analysts_not_point_in_time" in record.CAVEATS
    with pytest.raises(ValueError):
        record.caveat_sentences(["everything_is_fine"])
    first = FIRST
    assert first == date(2025, 5, 1) and load_config().bucket_of(73) == (70, 75) and load_config().bucket_of(100) == (75, 101) and load_config().bucket_of(101) is None and load_config().bucket_of(59) == (0, 60)      # the buckets of the config since the forward excess: <60, 60-65, 65-70, 70-75, 75+
