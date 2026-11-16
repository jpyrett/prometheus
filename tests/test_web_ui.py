"""The web bridge of the desk (UI v1 spec; design choices entry 99): the REAL app (`fastapi.testclient`) playing the REAL captures of real sessions, the run's questions, the two checkboxes as settings, and a read of the source.

No model, no quote, no key: the replay plays what `session --save` recorded. No stand-in agent: nothing here builds an agent. The account routes against the local Postgres are in `tests/test_web_accounts.py`.
"""
import ast
import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prometheus.application.web import settings
from prometheus.application.web.app import DEFAULT_PROMPT, create_app
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


def test_the_replay_of_bob_jones_goes_through_six_stages_and_ends_traded_with_the_account_of_the_capture():
    http = client("bob_jones_oil")
    health = http.get("/api/health").json()
    assert health["mode"] == "replay" and health["book"] == "ok" and health["defaults"] == {"prompt": DEFAULT_PROMPT, "strict": False, "neutral": True, "fetch": True} and health["limits"] == {"prompt_chars": 600}
    assert DEFAULT_PROMPT == "Buy an even mixture of 3 most recommended stocks in the pizza restaurants using $3000 of my available money"
    events = play(http)
    kinds = [kind for _, kind, _ in events]
    assert [seq for seq, _, _ in events] == list(range(1, len(events) + 1)) and kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert {"stage", "log", "work_order", "portfolio", "activity"} <= set(kinds) and "question" not in kinds
    started = events[0][2]
    assert started["mode"] == "replay" and [s["key"] for s in started["stages"]] == ["work_order", "gather", "recommend", "critique", "prepare", "trade"] and started["trade_mode"] == "auto"
    last = {}
    for _, kind, data in events:
        if kind == "stage":
            last[data["key"]] = data["state"]
    assert set(last.values()) == {"done"} and len(last) == 6
    portfolios = [d for _, k, d in events if k == "portfolio"]
    assert (portfolios[0]["cash"], portfolios[0]["positions"], portfolios[0]["total"]) == (15000.0, [], 15000.0)
    assert (portfolios[-1]["cash"], {p["symbol"]: p["shares"] for p in portfolios[-1]["positions"]}, portfolios[-1]["changes"], portfolios[-1]["cash_change"]) == (12367.41, {"XOM": 8, "EOG": 8}, {"EOG": 8, "XOM": 8}, -2632.59)
    assert [d["cash"] for d in portfolios] == [15000.0, 13634.6, 12367.41]                      # each fill lands as it is booked
    finished = events[-1][2]
    assert finished["status"] == "traded" and finished["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"] and "$12,367.41" in finished["headline"]
    orders = [d for _, k, d in events if k == "work_order"]
    assert orders[0]["rows"] == [] and orders[1]["rows"][0]["group"] == "found" and orders[-1]["status"] == "traded" and orders[-1]["rows"][0]["trade"]["status"] == "filled"
    texts = [d["text"] for _, k, d in events if k == "log"]
    assert any(t.startswith("understood: buy the 3 most recommended 'oil' companies") for t in texts) and any("XOM" in t and "yes (AUTO)" in t for t in texts)
    assert texts[-1] == finished["headline"] and kinds[-2] == "log" and [d for _, k, d in events if k == "log"][-1]["level"] == "good"      # the feed's last line is how the run ended


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


def test_the_account_routes_of_a_replay_show_its_account_and_refuse_every_write():
    http = client("bob_jones_oil")
    view = http.get("/api/accounts/7").json()
    assert view["account"]["name"] == "Bob Jones" and view["account"]["read_only"] is True and view["cash"] == 15000.0
    for call in (http.post("/api/accounts", json={}), http.patch("/api/accounts/7", json={"name": "x"}), http.post("/api/accounts/7/funds", json={"action": "add", "amount": "5.00"})):
        assert call.status_code == 409 and "read-only" in call.json()["error"]


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


# ---------------------------------------------------------------- the two checkboxes

def test_run_settings_for_the_four_combinations_only_say_whether_a_neutral_recommendation_passes():
    """AMENDED (entry 104): Strict used to lower the automatic-trading mark too; that mark is gone, so the settings of a run are only whether a neutral recommendation counts as approved."""
    assert settings.run_settings(strict=False, neutral=True) == settings.RunSettings(allow_neutral=True)
    assert settings.run_settings(strict=False, neutral=False) == settings.RunSettings(allow_neutral=False)
    for neutral in (True, False):                                            # strict: only what the Critic approves trades, so neutral is not a pass whatever the other box says
        assert settings.run_settings(strict=True, neutral=neutral) == settings.RunSettings(allow_neutral=False)
    assert [f.name for f in dataclasses.fields(settings.RunSettings)] == ["allow_neutral"]


def test_the_neutral_variable_is_set_for_the_run_and_put_back_even_when_the_run_raises(monkeypatch):
    monkeypatch.setenv(settings.NEUTRAL_SETTING, "false")
    with settings.applied(settings.run_settings(False, True)):
        assert os.environ[settings.NEUTRAL_SETTING] == "true"
    assert os.environ[settings.NEUTRAL_SETTING] == "false"
    with pytest.raises(RuntimeError):
        with settings.applied(settings.run_settings(False, True)):
            raise RuntimeError("the run broke")
    assert os.environ[settings.NEUTRAL_SETTING] == "false"
    monkeypatch.delenv(settings.NEUTRAL_SETTING)
    with settings.applied(settings.run_settings(True, True)):
        assert os.environ[settings.NEUTRAL_SETTING] == "false"
    assert settings.NEUTRAL_SETTING not in os.environ


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


def test_the_replays_of_jacobs_request_say_what_was_found_and_what_was_left_out_and_the_table_keeps_only_the_stocks_that_stayed():
    """The two live sessions of Jacob's own request (a throwaway account each, 2026-11-15; design choices entries 101 and 102): Ask mode, where Chick-fil-a was left out at the user's word, and Auto mode, where it was left out
    without a question. The replay says what the Data Clerk found (Google as the listed owner, Alphabet Inc.) and did not find, what the Account Manager left out, and the work order lists the four stocks that stayed."""
    for name, note in (("d_pyrett_names_ask", "dropped at the user's word: Chick-fil-a"), ("d_pyrett_names_auto", "dropped in Auto mode: Chick-fil-a")):
        http = client(name)
        run_id = http.post("/api/runs", json={"prompt": "ignored", "account_id": 1}).json()["run_id"]
        run = http.app.state.runs.get(run_id)

        def answer(run=run):
            for _ in range(200):                                       # the Ask-mode replay asks to confirm what was understood; the Auto-mode one asks nothing
                for question in run.open_questions():
                    run.answer(question, True)
                if run.status == "finished":
                    return
                time.sleep(0.05)

        thread = threading.Thread(target=answer)
        thread.start()
        events = events_of(http, run_id)
        thread.join(5)
        lines = [d["text"] for _, k, d in events if k == "log"]
        assert any("found: you typed 'Google': Alphabet Inc. (GOOGL), the listed owner of what you typed" in line for line in lines), name
        assert any("not found: you typed 'Chick-fil-a': privately held, or not known to be listed" in line for line in lines) and any(line.startswith(note) for line in lines), name
        assert [r["symbol"] for r in [d for _, k, d in events if k == "work_order"][-1]["rows"]] == ["F", "GM", "PG", "GOOGL"] and events[-1][2]["status"] == "traded", name
        assert ("question" in [k for _, k, _ in events]) == (name == "d_pyrett_names_ask"), name
