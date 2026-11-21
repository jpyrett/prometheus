"""The Track Record and the saved sessions (prometheus/proving/track.py, sessions.py, recs.py; design/proving_ground_spec.md section 10; P4, P5, P8). Real data only: the real captures of `tests/fixtures/session_real/` and
`tests/fixtures/proving_real/live_session_oil.json` (a live Auto run of 2026-11-20 on real Finnhub quotes), scored against the real stored bars, read independently in the test."""
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from prometheus.agents.trader import simulator
from prometheus.proving import bars as bars_door
from prometheus.proving import recs, sessions, track
from prometheus.proving.record import ProvingRun
from prometheus.proving.store import ProvingStore

HERE = Path(__file__).parent
SESSION_REAL, REAL = HERE / "fixtures" / "session_real", HERE / "fixtures" / "proving_real"


def outcome_of(run: ProvingRun, key: str):
    return next(o for o in run.outcomes if o.key == key)


def test_a_real_fill_is_the_entry_and_a_denied_recommendation_is_one_share_at_the_next_open():
    """EOG was bought for real on 2026-11-13 (fill 158.32, 37.897928 shares): that fill is the entry. APA was denied by the Critic: it is hypothetical, one share-equivalent at the next stored open (2026-11-16, read here from the bars)."""
    run = track.build()
    capture = json.loads((SESSION_REAL / "auto_even_split_dollars.json").read_text(encoding="utf-8"))
    fill = next(t["fill"] for t in capture["order"]["trades"] if t["record_id"] == "d415bba65829")
    bought = outcome_of(run, "2026-11-13:d415bba65829")
    assert (bought.traded, bought.hypothetical, bought.verdict, bought.not_traded_reason) == (True, False, "approved", None)
    assert (bought.entry.date, bought.entry.price, bought.entry.shares) == (date(2026, 11, 13), Decimal("158.32"), Decimal("37.897928")) and fill["price"] == 158.32
    assert bought.result == "lost" and bought.exit.reason == "stop" and bought.gain < 0
    denied = outcome_of(run, "2026-11-13:85ca6d9aadcd")
    opened = next(b for b in bars_door.read_bars(["APA"], date(2026, 11, 13), date(2026, 11, 19))["APA"] if b["date"] == "2026-11-16")
    assert (denied.traded, denied.hypothetical, denied.verdict, denied.not_traded_reason, denied.gain) == (False, True, "denied", "denied", None)
    assert (denied.entry.date, denied.entry.price, denied.entry.shares) == (date(2026, 11, 16), simulator.bar_price(opened, "open"), Decimal(1))


def test_a_neutral_purchase_has_no_plan_and_is_followed_to_the_last_close():
    """OXY was bought as a NEUTRAL company (no plan: no stop, no target, no time stop): its real fill is the entry and nothing closes it, so it is open and marked at the last usable close, never a win or a loss by a rule that was never made."""
    run = track.build()
    oxy = outcome_of(run, "2026-11-13:neutral-ce4c2f395399-1")
    assert (oxy.verdict, oxy.traded, oxy.result, oxy.exit) == ("neutral", True, "open", None)
    assert oxy.plan.stop is None and oxy.plan.target is None and oxy.plan.time_stop_date is None
    assert oxy.entry.price == Decimal("58.75") and oxy.path[-1].date <= run.range.last_bar_day      # OXY has no per-ticker file: its last bar is the last day file, which can be a day behind


def test_nothing_after_the_last_usable_bar_is_read_and_the_run_says_so():
    """P2/P4: the last bar that may be read is the last trading day that has CLOSED (entry 129: today only after its close), so no path goes past it. A decision of that day has `no_data` for what it cannot
    know yet: a recommendation that was not bought needs the NEXT open. A purchase really filled that day is known and is marked at that day's close."""
    run = track.build()
    last = bars_door.usable_last_day()
    assert run.range.last_bar_day == last and run.coverage["last_bar_day"] == last.isoformat()
    assert all(p.date <= last for o in run.outcomes for p in o.path) and all(point.date <= last for point in run.equity)
    assert all(o.result == "no_data" for o in run.outcomes if o.decision_day > last or (o.decision_day == last and o.hypothetical))
    assert all(o.entry.date == last and [p.date for p in o.path] in ([], [last]) for o in run.outcomes if o.decision_day == last and not o.hypothetical and o.entry)
    assert run.stats.portfolio is None and run.kind == "track" and set(run.coverage["sources"].values()) == {"fixture"}      # the Track Record has no equity of its own: its curve is the $1 index


def test_the_dollar_index_and_the_statistics_are_recomputed_from_the_outcomes():
    run = track.build()
    block = run.stats.outcomes
    assert block.n == len(run.outcomes) == block.won + block.lost + block.open + block.not_filled + block.no_data and block.closed == block.won + block.lost
    traded = [o for o in run.outcomes if o.traded and o.path]
    first = run.equity[0]
    day = [o for o in traded if o.entry.date == first.date]      # on the first day each real entry is one dollar: the index is the mean of 1 + the return of the day
    expected = (Decimal(100) * sum(Decimal(1) + Decimal(str(next(p for p in o.path if p.date == first.date).return_pct)) for o in day) / len(day)).quantize(Decimal("0.01"))
    assert first.equity == expected and first.cash == 0 and first.benchmark is not None
    assert not any(o.hypothetical for o in traded)


def test_the_saved_real_track_is_a_proving_run_that_adds_up():
    run = ProvingRun.model_validate(json.loads((REAL / "track.json").read_text(encoding="utf-8")))
    assert run.kind == "track" and run.stats.outcomes.n == len(run.outcomes) and run.decision_days and run.equity


def test_a_live_run_is_saved_and_counts_once_with_its_fixture_copy(tmp_path):
    """The real live session (a saved capture) goes into the store as a run: its recommendations are scored once even though a copy of the same capture is a fixture, and the saved one is the one that counts."""
    store = ProvingStore(tmp_path)
    capture = json.loads((REAL / "live_session_oil.json").read_text(encoding="utf-8"))
    path = sessions.save_capture(capture, store=store)
    assert path == tmp_path / "sessions" / "4435c024e662.json" and store.load_session("4435c024e662")["order"]["request_id"] == "4435c024e662"
    run = track.build(store)
    mine = [o for o in run.outcomes if o.key.startswith("2026-11-20:") and o.record_id in ("653dfbe9e70f", "4f81adfe2f80", "27f0880b8b96")]
    assert len(mine) == 3 and {run.coverage["sources"][o.key] for o in mine} == {"run"} and all(o.verdict == "denied" for o in mine)


def test_a_failed_save_is_a_log_line_and_a_direct_order_is_not_saved(tmp_path):
    """`save_session` never raises: a store that cannot write gives None; a direct order has no recommendation to score and is not kept; a run that never reached the Advisor is not kept."""
    capture = json.loads((REAL / "live_session_oil.json").read_text(encoding="utf-8"))
    from prometheus.common.work_order import WorkOrder
    order = WorkOrder.model_validate(capture["order"])

    class Broken(ProvingStore):
        def save_session(self, request_id, payload):
            raise OSError("disk full")
    assert sessions.save_session(capture["request"], None, {"order": order, "plan": {"intent": capture["intent"]}, "path": capture["path"]}, store=Broken(tmp_path)) is None
    good = {"order": order, "plan": {"intent": capture["intent"]}, "path": capture["path"]}
    assert sessions.save_session(capture["request"], None, good, store=ProvingStore(tmp_path)) == tmp_path / "sessions" / "4435c024e662.json"
    assert sessions.save_session("Buy 5 MSFT", None, {"order": order, "plan": {"direct": True}, "path": []}, store=ProvingStore(tmp_path / "other")) is None
    assert sessions.save_session("x", None, {"order": None, "plan": None, "path": []}, store=ProvingStore(tmp_path / "other")) is None
    assert not (tmp_path / "other").exists()


def test_every_outcome_of_the_track_record_says_where_its_recommendation_came_from():
    """`RecOutcome.source` is "run" (a saved live run) or "fixture" (a real capture kept with the tests), the same word the run's `coverage["sources"]` has kept for each outcome."""
    run = track.build()
    assert run.outcomes and all(o.source in ("run", "fixture") and run.coverage["sources"][o.key] == o.source for o in run.outcomes)
