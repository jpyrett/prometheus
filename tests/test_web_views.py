"""What the page shows, from the REAL captures of real sessions (UI v1 spec 4.3.1, 4.4, 4.5; design choices entry 99). `views.py` is pure: it reads a work order and writes rows, stages and an account.

No stand-in anywhere: the captures are `tests/fixtures/session_real/` (`bob_jones_oil.json`, `pam_smith_neutral.json`, `bill_gates_direct_order_*.json`), made by real runs of the real agents.
"""
import json
from pathlib import Path


from prometheus.application.web import views
from prometheus.common.work_order import WorkOrder

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"


def capture(name: str) -> dict:
    return json.loads((SESSIONS / f"{name}.json").read_text(encoding="utf-8"))


def test_stage_states_follow_the_path_and_say_skipped_failed_waiting_and_not_reached():
    bob = capture("bob_jones_oil")
    every = lambda states: [s["state"] for s in states]
    done = views.stage_states(path=bob["path"], seconds={"data_clerk": 37.2, "critic": 45.3})
    assert every(done) == ["done"] * 6 and done[1]["seconds"] == 37.2 and [s["key"] for s in done] == ["work_order", "gather", "recommend", "critique", "prepare", "trade"]
    assert every(views.stage_states(path=bob["path"][:3], running="critic")) == ["done", "done", "done", "busy", "pending", "pending"]
    assert every(views.stage_states(path=bob["path"][:3], running="critic", waiting=True))[3] == "waiting"
    direct = capture("bill_gates_direct_order_company_auto")
    assert every(views.stage_states(path=direct["path"])) == ["done", "skipped", "skipped", "skipped", "done", "done"]
    neutral = ["account_manager", "data_clerk", "advisor", "account_manager_decision", "trader"]          # a neutral pass: no Critic
    assert every(views.stage_states(path=neutral))[3] == "skipped"
    stopped = WorkOrder.model_validate({**bob["order"], "status": "data_failed"})
    assert every(views.stage_states(path=["account_manager", "data_clerk"], order=stopped))[1] == "failed"
    could_not = WorkOrder.model_validate({**bob["order"], "errors": ["the Advisor could not start: a missing key"]})
    assert every(views.stage_states(path=["account_manager", "data_clerk", "advisor"], order=could_not))[2] == "failed"
    assert every(views.stage_states(path=[], request_errors=["the request could not be read"]))[0] == "failed"
    assert every(views.stage_states(path=["account_manager", "data_clerk"], failed={"advisor"}, running=None))[2] == "failed"


def test_the_records_of_a_row_are_the_raw_records_and_a_row_that_is_not_there_is_none():
    data = capture("bob_jones_oil")
    found = views.records_of(data["order"], "XOM")
    assert set(found) == {"recommendation", "review", "decision", "trade"} and found["review"]["score"] == 86 and found["trade"]["fill"]["price"] == 170.6753
    assert set(views.records_of(data["order"], "CVX")) == {"candidate"}
    assert views.records_of(data["order"], "ZZZZ") is None
    neutral = capture("pam_smith_neutral")
    assert set(views.records_of(neutral["order"], "MU")) == {"candidate", "decision", "trade"}


