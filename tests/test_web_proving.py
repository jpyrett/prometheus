"""The Proving Ground's routes (prometheus/application/web/proving_run.py, proving_views.py, app.py; design/proving_ground_spec.md section 9) over the SAVED REAL runs of `tests/fixtures/proving_real/` in replay: free, no model, no quote, no
database. The shapes are the page's contract: the routes answer with the run's own fields and the events of a replay are the events of a live run."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from proving_helpers import as_before_the_forward_excess

from prometheus.application.web.app import create_app
from prometheus.application.web.proving_run import Proving
from prometheus.application.web.replay import Replay, load_capture
from prometheus.proving.record import ProvingRun, Stats as RunStats
from prometheus.proving.store import ProvingStore

HERE = Path(__file__).parent
REAL = HERE / "fixtures" / "proving_real"
SAVED = ProvingRun.model_validate_json((REAL / "oil_weekly_short.json").read_text(encoding="utf-8"))


@pytest.fixture
def http(tmp_path):
    replay = Replay(load_capture(HERE / "fixtures" / "session_real" / "bob_jones_oil.json"), speed=0)
    client = TestClient(create_app(replay=replay, proving=Proving(store=ProvingStore(tmp_path), fixtures=REAL, paid=False)))
    return client


def events_of(http, run_id: str) -> list[dict]:
    """Every event of a run, read to its end (the SSE stream closes when the run finishes)."""
    body = http.get(f"/api/runs/{run_id}/events").text
    out = []
    for block in body.split("\n\n"):
        lines = [l for l in block.splitlines() if l and not l.startswith(":")]
        if lines:
            out.append({"type": lines[1].split(": ", 1)[1], "data": json.loads(lines[2].split(": ", 1)[1])})
    return out


def wait_until_idle(http) -> None:
    for _ in range(200):
        if not http.get("/api/health").json()["active_run"]:
            return
        time.sleep(0.05)


def test_the_saved_real_run_is_listed_and_read_with_its_tiles_caveats_and_days(http):
    rows = http.get("/api/proving/runs").json()
    row = next(r for r in rows if r["run_id"] == SAVED.run_id)
    assert set(row) == {"run_id", "kind", "status", "created_utc", "prompt", "range", "headline", "return_pct", "excess_return_pct", "speed"} and row["speed"] in (None, "faithful") and row["headline"] == SAVED.stats.headline.sentence
    view = http.get(f"/api/proving/runs/{SAVED.run_id}").json()
    assert view["kpis"]["recommendations"]["total"] == len(SAVED.outcomes) and view["kpis"]["win_rate"]["n"] == SAVED.stats.outcomes.closed
    assert [c["code"] for c in view["caveats"]] == list(SAVED.caveats) and all(c["sentence"] for c in view["caveats"]) and "desks" in view and "desks" not in view["coverage"]
    assert view["equity"][-1]["equity"] == float(SAVED.equity[-1].equity) and len(view["outcomes"]) == len(SAVED.outcomes)
    day = view["desks"][0]
    desk = http.get(f"/api/proving/runs/{SAVED.run_id}/days/{day}").json()
    assert desk["title"] == f"The desk on {day}" and desk["view"]["rows"]
    key = desk["view"]["rows"][0]["key"]
    assert http.get(f"/api/proving/runs/{SAVED.run_id}/days/{day}/records/{key}").status_code == 200
    assert http.get(f"/api/proving/runs/{SAVED.run_id}/days/2020-01-02").status_code == 404 and http.get("/api/proving/runs/nope").status_code == 404


def test_the_replay_of_a_saved_run_sends_the_events_of_a_live_run_and_ends_in_the_saved_stats(http):
    started = http.post(f"/api/proving/runs/{SAVED.run_id}/replay?speed=0")
    assert started.status_code == 202
    wait_until_idle(http)
    events = events_of(http, started.json()["run_id"])
    types = [e["type"] for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished" and types.count("pv_started") == 1 and types.count("pv_finished") == 1
    assert types.count("pv_day") == len(SAVED.equity) and types.count("pv_research") == len(SAVED.decision_days) and types.count("pv_outcome") == len(SAVED.outcomes)
    assert types.index("pv_started") < types.index("pv_research") < types.index("pv_day")
    finished = next(e["data"] for e in events if e["type"] == "pv_finished")
    assert finished["stats"] == json.loads(SAVED.stats.model_dump_json()) and finished["headline"]["sentence"] == SAVED.stats.headline.sentence and len(finished["equity"]) == len(SAVED.equity)
    assert next(e["data"] for e in events if e["type"] == "pv_started")["speed"] == "faithful" and finished["speed"] == "faithful" and "timing" in finished and "timing" in next(e["data"] for e in events if e["type"] == "pv_started")
    assert all("stages" in e["data"] for e in events if e["type"] == "pv_research")
    decisions = [e["data"] for e in events if e["type"] == "pv_decision"]
    assert decisions and all(d["view"]["rows"] for d in decisions)
    assert [e["data"]["point"]["date"] for e in events if e["type"] == "pv_day"] == [p.date.isoformat() for p in SAVED.equity]


def test_a_paid_run_is_refused_in_replay_and_the_plan_says_why_a_request_cannot_run(http):
    body = {"prompt": "Buy the 3 most recommended oil stocks with $3,000, split evenly", "cash": "10000", "start": "2026-01-05", "end": "2026-11-13", "cadence": "weekly"}
    refused = http.post("/api/proving/runs", json=body)
    assert refused.status_code == 409 and "recorded session" in refused.json()["error"]
    plan = http.post("/api/proving/plan", json=body).json()
    assert plan["problems"] and "usable range is" in plan["problems"][0] and plan["decision_days"] == []
    good = http.post("/api/proving/plan", json={**body, "start": "2026-10-19", "end": "2026-11-13"}).json()
    assert good["problems"] == [] and good["decision_days"][0] == "2026-10-19" and {c["code"] for c in good["caveats"]} >= {"fundamentals_not_point_in_time", "model_knowledge_cutoff"}
    assert good["speed"] == "fast" and good["timing"]["speed"] == "fast" and "fast_models" in {c["code"] for c in good["caveats"]}      # no speed in the request is the config's own: fast, with its caveat
    slow = http.post("/api/proving/plan", json={**body, "start": "2026-10-19", "end": "2026-11-13", "speed": "faithful"}).json()
    assert slow["speed"] == "faithful" and "fast_models" not in {c["code"] for c in slow["caveats"]} and http.post("/api/proving/plan", json={**body, "speed": "quick"}).status_code == 422
    assert http.post("/api/proving/plan", json={**body, "prompt": ""}).status_code == 422 and http.post("/api/proving/plan", json={**body, "cash": "-5"}).status_code == 422
    assert http.get("/api/proving/coverage").json()["usable"]["last_day"]


def test_one_run_at_a_time_across_kinds_and_a_recorded_run_is_not_deleted(http):
    started = http.post(f"/api/proving/runs/{SAVED.run_id}/replay?speed=5")      # slow enough to still be running
    assert started.status_code == 202
    assert http.get("/api/health").json()["active_kind"] == "proving"
    assert http.post("/api/runs", json={"prompt": "x"}).status_code == 409 and http.post(f"/api/proving/runs/{SAVED.run_id}/replay").status_code == 409
    http.post(f"/api/runs/{started.json()['run_id']}/stop")
    wait_until_idle(http)
    assert http.delete(f"/api/proving/runs/{SAVED.run_id}").status_code == 404      # the user's own file only: a recorded run is a fixture


def test_the_track_record_and_the_recorded_live_snapshots_are_served(http):
    track = http.get("/api/proving/track").json()
    assert track["kind"] == "track" and track["outcomes"] and track["kpis"]["recommendations"]["total"] == len(track["outcomes"])
    oil = http.get("/api/proving/live/8277").json()
    assert oil["account_id"] == 8277 and {p["status"] for p in oil["positions"]} == {"no_plan"}
    plan = http.get("/api/proving/live/1").json()
    assert {p["status"] for p in plan["positions"]} == {"stop_hit"} and plan["alerts"]
    stream = http.get("/api/proving/live/8277/events?once=true").text
    assert stream.startswith("event: snapshot") and json.loads(stream.split("data: ", 1)[1].split("\n\n")[0])["account_id"] == 8277


def test_a_time_machine_run_from_the_real_cache_streams_the_events_and_saves_the_run(tmp_path, monkeypatch):
    """The live path of `POST /api/proving/runs` over the real cached research (no model is called, nothing is fetched): the run's events in the order a live run sends them, the saved run in the store, and `active_kind` while it runs."""
    import shutil

    from prometheus.proving import engine
    store = ProvingStore(tmp_path)
    (tmp_path / "cache").mkdir()
    for path in (REAL / "oil_weekly_short_cache").glob("*.json"):
        shutil.copy2(path, tmp_path / "cache" / path.name)
    monkeypatch.setattr(engine, "data_stamp", lambda report: SAVED.coverage["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: SAVED.models)
    monkeypatch.setattr(engine, "versions", lambda: SAVED.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare
    monkeypatch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: SAVED.range.last_bar_day)      # the saved run's horizon: the stored bars have moved on since it was recorded

    def refuse(*args, **kwargs):
        raise AssertionError("a test must not research a day")
    monkeypatch.setattr(engine, "research_day", refuse)
    web = TestClient(create_app(proving=Proving(store=store, paid=True)))
    body = {"prompt": SAVED.request["prompt"], "cash": "10000", "start": SAVED.request["start"], "end": SAVED.request["end"], "cadence": "weekly", "neutral": True, "speed": "faithful"}      # the saved research was made with the live settings
    started = web.post("/api/proving/runs", json=body)
    assert started.status_code == 202
    wait_until_idle(web)
    events = events_of(web, started.json()["run_id"])
    types = [e["type"] for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished" and types.count("pv_finished") == 1 and types.count("pv_research") == len(SAVED.decision_days) and types.count("pv_day") == len(SAVED.equity)
    assert all(e["data"]["status"] == "cached" for e in events if e["type"] == "pv_research") and any(e["type"] == "pv_decision" and e["data"]["view"]["rows"] for e in events)
    saved_id = next(e["data"]["saved_id"] for e in events if e["type"] == "pv_finished")
    assert [r["run_id"] for r in web.get("/api/proving/runs").json()] == [saved_id] and web.get("/api/health").json()["active_kind"] is None
    served = web.get(f"/api/proving/runs/{saved_id}").json()["stats"]
    assert served == json.loads(RunStats.model_validate(served).model_dump_json())                                    # the page is sent exactly the statistics the run computed (and nothing it computes itself, P12)
    assert as_before_the_forward_excess(RunStats.model_validate(served)) == as_before_the_forward_excess(SAVED.stats)      # and in everything the saved run recorded they are the same; the forward excess is new
    assert web.delete(f"/api/proving/runs/{saved_id}").status_code == 204 and web.get("/api/proving/runs").json() == []
