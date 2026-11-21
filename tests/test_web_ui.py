"""The web bridge of the desk (UI v1 spec; design choices entry 99): the REAL app (`fastapi.testclient`) playing the REAL captures of real sessions, the run's questions, the two checkboxes as settings, and a read of the source.

No model, no quote, no key: the replay plays what `session --save` recorded. No stand-in agent: nothing here builds an agent. The account routes against the local Postgres are in `tests/test_web_accounts.py`.
"""
import ast
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prometheus.application.web import settings
from prometheus.application.web.app import create_app
from prometheus.application.web.replay import Replay, load_capture
from prometheus.application.web.runs import QuestionClosed, RunBusy, RunManager

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
WEB = Path(__file__).parents[1] / "prometheus" / "application" / "web"


def client(name: str, **options) -> TestClient:
    return TestClient(create_app(replay=Replay(load_capture(SESSIONS / f"{name}.json"), speed=0, **options)))


def events_of(http: TestClient, run_id: str, after: int = 0) -> list[tuple[int, str, dict]]:
    found, kind, seq = [], None, None
    with http.stream("GET", f"/api/runs/{run_id}/events", params={"after": after}) as stream:
        for line in stream.iter_lines():
            if line.startswith("id:"):
                seq = int(line[4:])
            elif line.startswith("event:"):
                kind = line[7:]
            elif line.startswith("data:"):
                found.append((seq, kind, json.loads(line[6:])))
    return found


def play(http: TestClient) -> list[tuple[int, str, dict]]:
    run_id = http.post("/api/runs", json={"prompt": "ignored", "account_id": 1}).json()["run_id"]
    return events_of(http, run_id)


def test_a_late_reader_gets_every_event_and_a_reconnect_gets_the_rest():
    http = client("bob_jones_oil")
    run_id = http.post("/api/runs", json={}).json()["run_id"]
    everything = events_of(http, run_id)
    assert events_of(http, run_id) == everything                              # a page reloaded after the run draws the same
    assert events_of(http, run_id, after=40) == everything[40:]
    resumed = []
    with http.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "45"}) as stream:
        resumed = [line for line in stream.iter_lines() if line.startswith("id:")]
    assert resumed == [f"id: {n}" for n in range(46, len(everything) + 1)]
    assert http.get("/api/runs/nope/events").status_code == 404 and http.get(f"/api/runs/{run_id}/records/XOM").json()["review"]["score"] == 86


def test_a_direct_order_replay_skips_the_research_and_asks_to_confirm_in_ask_mode():
    http = client("bill_gates_direct_order_company_auto")                    # Auto mode: nothing is asked
    events = play(http)
    states = {}
    for _, kind, data in events:
        if kind == "stage":
            states[data["key"]] = data["state"]
    assert states == {"work_order": "done", "gather": "skipped", "recommend": "skipped", "critique": "skipped", "prepare": "done", "trade": "done"}
    assert events[-1][2]["status"] == "traded" and [d for _, k, d in events if k == "portfolio"][-1]["positions"][0]["symbol"] == "MSFT"
    ask = client("bill_gates_direct_order_funded")                           # Ask mode: the same confirmation the live desk asks
    run_id = ask.post("/api/runs", json={}).json()["run_id"]
    for _ in range(100):
        if ask.app.state.runs.get(run_id).open_questions():
            break
        time.sleep(0.05)
    assert ask.post("/api/runs", json={}).status_code == 409                  # one run at a time
    question_id = ask.app.state.runs.get(run_id).open_questions()[0]
    assert ask.post(f"/api/runs/{run_id}/answers", json={"question_id": question_id, "answer": "yes"}).status_code == 422
    assert ask.post(f"/api/runs/{run_id}/answers", json={"question_id": question_id, "answer": True}).status_code == 204
    assert ask.post(f"/api/runs/{run_id}/answers", json={"question_id": question_id, "answer": True}).status_code == 409      # a second answer is refused
    events = events_of(ask, run_id)
    asked = next(d for _, k, d in events if k == "question")
    assert asked["kind"] == "confirm" and asked["text"].startswith("I understood: buy 50 shares of MSFT") and asked["question_id"] == question_id
    assert [d["by"] for _, k, d in events if k == "question_closed"] == ["you"] and events[-1][2]["status"] == "traded"
    states = {d["key"]: d["state"] for _, k, d in events if k == "stage"}
    assert states["trade"] == "done"
    again = ask.post("/api/runs", json={})
    assert again.status_code == 202 and ask.post(f"/api/runs/{again.json()['run_id']}/stop").status_code == 202           # a new run is allowed once the last one is over (and this one is stopped, not left waiting)


# ---------------------------------------------------------------- the run's questions, with no agent anywhere

def asked_in_a_thread(run, kind="confirm", timeout=5.0):
    box = {}
    thread = threading.Thread(target=lambda: box.update(answer=run.ask(kind, {"title": "t", "text": "t"}, timeout)))
    thread.start()
    for _ in range(100):
        if run.open_questions():
            break
        time.sleep(0.02)
    return thread, box


def test_a_question_waits_for_the_answer_times_out_to_none_closes_on_stop_and_refuses_a_second_answer():
    manager = RunManager()
    release = threading.Event()
    run = manager.start(lambda r: release.wait(10), kind="test")
    with pytest.raises(RunBusy):
        manager.start(lambda r: None)
    thread, box = asked_in_a_thread(run)
    question_id = run.open_questions()[0]
    run.answer(question_id, True)
    thread.join(2)
    assert box["answer"] is True and run.open_questions() == []
    with pytest.raises(QuestionClosed):
        run.answer(question_id, False)
    with pytest.raises(KeyError):
        run.answer("nothing", True)
    assert run.ask("confirm", {}, timeout=0.3) is None                       # no answer is never a yes
    closed = [e["data"] for e in run.events if e["type"] == "question_closed"]
    assert [(c["answer"], c["by"]) for c in closed] == [(True, "you"), (None, "timeout")]
    thread, box = asked_in_a_thread(run)
    run.stop()
    thread.join(2)
    assert box["answer"] is None and [e["data"]["by"] for e in run.events if e["type"] == "question_closed"][-1] == "stopped"
    assert run.ask("approve", {}, timeout=5) is None                         # a run that is stopped asks nobody
    release.set()


def test_a_worker_that_raises_ends_the_run_failed_with_the_kind_only_and_releases_the_lock():
    manager = RunManager()

    def boom(run):
        raise KeyError("a secret that must not be shown")

    run = manager.start(boom)
    for _ in range(100):
        if run.status == "finished":
            break
        time.sleep(0.02)
    finished = run.events[-1]
    assert finished["type"] == "run_finished" and finished["data"]["status"] == "failed" and finished["data"]["headline"] == "the run failed (KeyError)" and "secret" not in json.dumps(run.events)
    assert manager.active_id() is None and manager.start(lambda r: None).status in ("running", "finished")


# ---------------------------------------------------------------- the source

def test_the_web_package_never_opens_dotenv_and_imports_no_scoring_code_of_an_agent():
    allowed_critic = {"prometheus.agents.critic.report", "prometheus.agents.critic.config"}               # the Critic's own words for a verdict, and its marks
    forbidden = ("prometheus.agents.advisor",)
    for path in sorted(WEB.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "dotenv" not in source and ".env\"" not in source and ".env'" not in source.replace(".env.example", ""), f"{path.name} touches .env"
        for node in ast.walk(ast.parse(source)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"
                if name.startswith("prometheus.agents.critic"):
                    assert name in allowed_critic, f"{path.name} imports {name}"
