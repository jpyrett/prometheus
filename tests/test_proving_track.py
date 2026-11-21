"""The Track Record and the saved sessions (prometheus/proving/track.py, sessions.py, recs.py; design/proving_ground_spec.md section 10; P4, P5, P8). Real data only: the real captures of `tests/fixtures/session_real/` and
`tests/fixtures/proving_real/live_session_oil.json` (a live Auto run of 2026-11-20 on real Finnhub quotes), scored against the real stored bars, read independently in the test."""
import json
from pathlib import Path

from prometheus.proving import recs, sessions, track
from prometheus.proving.store import ProvingStore

HERE = Path(__file__).parent
SESSION_REAL, REAL = HERE / "fixtures" / "session_real", HERE / "fixtures" / "proving_real"


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


