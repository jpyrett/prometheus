"""The Time Machine's speed (prometheus/proving/speed.py, engine.py, design/proving_ground_spec.md P14): the fast mode, the faithful mode and what they share. FREE: the research is the REAL cached research of the real run
`tests/fixtures/proving_real/oil_weekly_short.json` (made with the live settings: it is a FAITHFUL capture), no model is called, nothing is fetched. A test that would need a model fails loudly instead of paying for one."""
import json
import shutil
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.proving import engine, speed as speed_module, tracing
from prometheus.proving.config import load_config
from prometheus.proving.record import CAVEATS, ProvingRun
from prometheus.proving.store import ProvingStore
from prometheus.rag.config import REPO_ROOT

REAL = Path(__file__).parent / "fixtures" / "proving_real"
SAVED = ProvingRun.model_validate_json((REAL / "oil_weekly_short.json").read_text(encoding="utf-8"))
FIELDS = {k: v for k, v in SAVED.request.items() if k in engine.TimeMachineRequest.model_fields}
FAITHFUL = engine.TimeMachineRequest(**{**FIELDS, "speed": "faithful"})
FAST = engine.TimeMachineRequest(**{**FIELDS, "speed": "fast"})
INTENT = TradeIntent.model_validate(SAVED.request["intent"])
CAPTURES = {entry["day"]: path for path, entry in ((p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted((REAL / "oil_weekly_short_cache").glob("*.json"))) if "order" in entry}      # the four real days (the folder also keeps the stored reading of the request)


def entry_of(day: str) -> dict:
    """A day of the real capture, read afresh (a run works on what it is given: no test may hand it the same objects twice)."""
    return json.loads(CAPTURES[day].read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def the_saved_runs_data(monkeypatch):
    """The data's stamp, the models and the config versions are those of the saved real run, and a day that is not in the cache cannot be researched (no model is ever called)."""
    monkeypatch.setattr(engine, "data_stamp", lambda report: SAVED.coverage["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: SAVED.models)
    monkeypatch.setattr(engine, "versions", lambda: SAVED.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare

    def refuse(*args, **kwargs):
        raise AssertionError("a test must not research a day: it is not in the cache")
    monkeypatch.setattr(engine, "research_day", refuse)
    monkeypatch.setattr(engine, "warm_stores", lambda: None)      # the stores are opened for a cold run; these tests read the cache and the stored bars only


def stored(tmp_path, request) -> ProvingStore:
    """A store in a temporary folder holding the real cached research under the keys `request` looks it up by: the faithful capture's entries are copied to the key a request of that speed would ask for."""
    store = ProvingStore(tmp_path)
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    for day, path in CAPTURES.items():
        key = engine.research_key(INTENT, date.fromisoformat(day), request, SAVED.coverage["stamp"])
        shutil.copy2(path, tmp_path / "cache" / f"{key}.json")
    return store


# ---------------------------------------------------------------- the cache key


def test_a_request_with_no_speed_is_the_configs_and_the_plan_says_which_and_shows_the_caveat(tmp_path):
    config = load_config()
    assert FIELDS.get("speed") is None and speed_module.speed_of(None, config) == config.time_machine.mode == "fast" and speed_module.speed_of("faithful", config) == "faithful"
    with pytest.raises(ValueError):
        speed_module.speed_of("quick", config)
    store = stored(tmp_path, FAITHFUL)
    report = {"usable": {"first_day": "2026-06-03", "last_day": "2026-11-19"}, "last_bar_day": "2026-11-19", "months": {}}
    fast, faithful = engine.plan(FAST, store, report, INTENT), engine.plan(FAITHFUL, store, report, INTENT)
    assert fast["speed"] == "fast" and faithful["speed"] == "faithful" and "fast_models" in [c["code"] for c in fast["caveats"]] and "fast_models" not in [c["code"] for c in faithful["caveats"]]
    assert {c["code"]: c["sentence"] for c in fast["caveats"]}["fast_models"] == CAVEATS["fast_models"] == "The Advisor's model ran with its reasoning switched off and the Critic's at reduced reasoning effort to make this test fast; a faithful run uses the live settings and takes longer."
    assert faithful["cached_days"] == faithful["decision_days"] and faithful["paid_days"] == 0 and faithful["timing"]["estimate_seconds"] == 0      # the real capture answers a faithful run
    assert fast["cached_days"] == [] and fast["paid_days"] == len(fast["decision_days"]) and fast["timing"]["speed"] == "fast"      # and not a fast one
    assert {"problems", "decision_days", "cached_days", "cache_known", "paid_days", "cost_estimate_usd", "caveats"} <= set(fast)
    assert fast["cost_estimate_usd"] == config.time_machine.fast.cost_note_per_day_usd * fast["paid_days"] and faithful["cost_estimate_usd"] == 0      # the measured cost of a day in THAT speed; a cached run costs nothing
    assert fast["timing"]["estimate_seconds"] == config.time_machine.fast.run_seconds_note and faithful["timing"]["estimate_seconds"] == 0


# ---------------------------------------------------------------- fast mode changes no computed rule


def test_the_faithful_settings_are_the_live_desks_and_the_old_engine_note_is_gone():
    """`faithful` asks the models exactly what the live desk asks: a config that gives it a role, a provider sort or a call limit is refused. The settings of the pool are fields of the config, not a note."""
    config = load_config()
    assert config.time_machine.faithful.advisor is config.time_machine.faithful.critic is config.time_machine.faithful.data_clerk is None and config.time_machine.faithful.provider_sort is None
    with pytest.raises(ValueError, match="live desk"):
        config.time_machine.model_validate({**config.time_machine.model_dump(), "faithful": {**config.time_machine.faithful.model_dump(), "provider_sort": "throughput"}})
    assert "_engine" not in json.loads((REPO_ROOT / "configs" / "proving_v1.json").read_text(encoding="utf-8")) and engine.engine_settings(config) == {"parallel_days": config.time_machine.parallel_days, "retry_429": config.time_machine.retry_429,
                                                                                                                                                           "pause_429_seconds": config.time_machine.pause_429_seconds}


def test_the_models_of_a_speed_are_built_with_its_settings_and_a_faithful_run_builds_none(monkeypatch):
    """Faithful: no model is built here (the agents build their own, as live). Fast: the same models asked for less thinking, a capped reply, a shorter wait, fewer retries, and the providers sorted by throughput."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key-and-never-used")
    config = load_config()
    faithful = speed_module.build_models("faithful", config)
    assert faithful.advisor is faithful.critic is faithful.data_clerk is None and faithful.call_limit is None
    fast = speed_module.build_models("fast", config)
    settings = config.time_machine.fast
    for role in ("advisor", "critic"):
        model, wanted = getattr(fast, role), getattr(settings, role)
        assert model.max_tokens == wanted.max_output_tokens and model.max_retries == wanted.max_retries and model.request_timeout == wanted.timeout_seconds
        assert model.extra_body == {"reasoning": {"effort": wanted.effort}, "provider": {"sort": "throughput", "require_parameters": True}}
    assert fast.data_clerk.extra_body == {"provider": {"sort": "throughput"}} and fast.call_limit == settings.call_limit_seconds and fast.parallel_records == settings.parallel_records
    assert speed_module.request_body(settings.advisor.model_copy(update={"effort": None, "max_output_tokens": None}), None) == {}


def test_one_model_call_waits_for_the_limit_of_its_context_and_the_live_limit_otherwise():
    """`advisor.calls.invoke_with_limit` waits the context's `CALL_LIMIT` (the fast mode's) and, with none, the live 300 seconds; the limit follows the context into the thread the call runs in and never outlives its block."""
    from prometheus.agents.advisor import calls

    class Slow:
        def invoke(self, messages):
            time.sleep(0.4)
            return "late"
    assert calls.CALL_LIMIT.get() is None and calls.MODEL_TIMEOUT == 300
    with speed_module.limit_of(0.05):
        with pytest.raises(TimeoutError, match="0.05"):
            calls.invoke_with_limit(Slow(), [])
    assert calls.invoke_with_limit(Slow(), []) == "late" and calls.CALL_LIMIT.get() is None


# ---------------------------------------------------------------- the Critic's records, at once

def test_the_critic_checks_its_records_at_once_and_puts_the_review_back_in_the_sequential_order(monkeypatch):
    """`review._check_all` with `parallel` above 1 gives the run record the sequential loop gives: counters added, drops and errors appended record by record, the last record's `independent_model`; the records finish in the
    reverse order here. The limit of model calls keeps it sequential when it could bind."""
    from prometheus.agents.critic import review
    from prometheus.agents.critic.critic import Critic
    order = sorted(CAPTURES)
    records = entry_of(order[0])["order"]["recommendations"]
    assert len(records) == 3
    finished: list[int] = []

    def check(w, critic, model, order_, view, gathered, now, run):
        time.sleep(0.2 * (len(records) - w.index))
        run["records_called"] += 1
        run["calls"] += 1
        run["tokens_in"] += 100 * (w.index + 1)
        run["controls_failed"] += w.index % 2
        run["dropped"].append(f"{w.raw['symbol']}: dropped")
        run["errors"].append(f"{w.raw['symbol']}: a note")
        run["independent_model"], run["advisor_model"] = w.raw["symbol"], f"advisor-{w.index}"
        finished.append(w.index)
    monkeypatch.setattr(review, "_check", check)
    seen = {}
    for parallel in (1, 8):
        finished.clear()
        critic = Critic(parallel=parallel)
        works = [review.Work(i, raw) for i, raw in enumerate(records)]
        run = {"records_called": 0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "controls_failed": 0, "labels_from_store": 0, "dropped": ["earlier"], "errors": ["gathered"], "independent_model": None, "advisor_model": None}      # `labels_from_store`: the counter Critic v2 added to the run record
        review._check_all(works, critic, None, None, None, None, None, run)
        seen[parallel] = run
        assert finished == ([0, 1, 2] if parallel == 1 else [2, 1, 0])
    assert seen[1] == seen[8] and seen[8]["errors"] == ["gathered", *(f"{r['symbol']}: a note" for r in records)] and seen[8]["independent_model"] == records[-1]["symbol"]
    assert len(records) <= Critic(parallel=8).config.limits.max_model_calls      # three records never reach the review's limit of model calls: the parallel path ran
    with pytest.raises(ValueError):
        Critic(parallel=0)


# ---------------------------------------------------------------- the result is out before LangSmith is told

def test_the_result_is_emitted_and_saved_before_the_feedback_is_posted_and_posting_does_not_block(tmp_path, monkeypatch):
    """`pv_finished` is out and the run is saved when the feedback is queued; queueing returns at once, the posts run in the background and `wait_for_posts` is how a command waits for them."""
    emitted: list[str] = []
    queued: list = []
    monkeypatch.setattr(tracing, "post_in_background", lambda jobs, **target: queued.append(({"types": list(emitted), "saved": (tmp_path / "runs").is_dir() and any((tmp_path / "runs").glob("tm-*.json")), "jobs": len(jobs)}, target)) or 0)
    engine.run_time_machine(FAITHFUL, store=stored(tmp_path, FAITHFUL), intent=INTENT, emit=lambda type_, data: emitted.append(type_))
    state, target = queued[0]
    assert state["types"][-1] == "pv_finished" and state["saved"] and state["jobs"] == len(SAVED.decision_days) and set(target) <= {"root_trace_id", "address"}
    monkeypatch.undo()
    monkeypatch.setattr(tracing, "tracing_is_enabled", lambda: True)
    started = threading.Event()

    def slow(trace_id, outcomes, client=None, **target):
        started.set()
        time.sleep(0.6)
        return 3
    monkeypatch.setattr(tracing, "post_outcomes", slow)
    began = time.perf_counter()
    assert tracing.post_in_background([("run-a", []), ("run-b", []), (None, []), ("run-c", [])]) == 3      # a day with no span (tracing was off for it) is not posted
    assert time.perf_counter() - began < 0.3 and started.wait(2)
    waited = tracing.wait_for_posts(timeout=10)
    assert waited["jobs"] == 3 and waited["rows"] == 9 and 0.4 < waited["seconds"] < 3 and tracing.wait_for_posts()["jobs"] == 0      # three posts at once took one post's time


def test_feedback_is_posted_with_the_trace_and_the_address_the_sdk_asks_for():
    """The SDK's run-level feedback wants the run's address (or session) and, for the background queue, the trace id: both go with every row, and a call with neither is the call it always was."""
    from prometheus.proving.record import RecOutcome
    outcomes = ProvingRun.model_validate(json.loads((REAL / "track.json").read_text(encoding="utf-8"))).outcomes
    day = [o for o in outcomes if o.decision_day.isoformat() == "2026-11-13"]
    sent: list[dict] = []

    class Recorder:
        def create_feedback(self, **row):
            sent.append(row)
    import prometheus.proving.tracing as module
    original = module.tracing_is_enabled
    module.tracing_is_enabled = lambda: True
    try:
        module.post_outcomes("6b0e1d3e-55f2-4d0b-9d37-0b5e2f7f2a11", day, client=Recorder(), root_trace_id="trace-1", address="the-address")
        module.post_outcomes("6b0e1d3e-55f2-4d0b-9d37-0b5e2f7f2a11", day, client=Recorder())
    finally:
        module.tracing_is_enabled = original
    half = len(sent) // 2
    assert half > 0 and all(r["trace_id"] == "trace-1" and r["address"] == "the-address" for r in sent[:half]) and all("trace_id" not in r and "address" not in r for r in sent[half:])
    assert [r["feedback_id"] for r in sent[:half]] == [r["feedback_id"] for r in sent[half:]] and isinstance(day[0], RecOutcome)


# ---------------------------------------------------------------- the real 6-month run, made in the fast mode

FAST_6M = ProvingRun.model_validate_json((REAL / "oil_weekly_6m_fast.json").read_text(encoding="utf-8"))
FAST_6M_REQUEST = engine.TimeMachineRequest(**{k: v for k, v in FAST_6M.request.items() if k in engine.TimeMachineRequest.model_fields})
FAST_6M_INTENT = TradeIntent.model_validate(FAST_6M.request["intent"])


def six_month_store(tmp_path) -> ProvingStore:
    store = ProvingStore(tmp_path)
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    for path in (REAL / "oil_weekly_6m_fast_cache").glob("*.json"):
        shutil.copy2(path, tmp_path / "cache" / path.name)
    return store


def test_the_timing_of_the_real_run_is_what_its_days_measured():
    """`timing_of` is arithmetic over the days' own rows: the seconds of each desk node and of each model, and the tokens, recomputed here from the calls of the recorded days."""
    per_day = FAST_6M.timing["per_day"]
    calls = [c for row in per_day.values() for c in row["calls"]]
    again = engine.timing_of(per_day, speed="fast", research_seconds=FAST_6M.timing["research_seconds"], walk_seconds=FAST_6M.timing["walk_seconds"], cached_days=0, parallel_days=23, link_seconds=FAST_6M.timing["link_seconds"], setup_seconds=FAST_6M.timing["setup_seconds"])
    assert again == FAST_6M.timing
    for model, row in FAST_6M.timing["models"].items():
        mine = [c for c in calls if c["model"] == model]
        assert row["calls"] == len(mine) and row["tokens_out"] == sum(c["tokens_out"] for c in mine) and abs(row["seconds"] - sum(c["seconds"] for c in mine)) < 0.05 and row["failed"] == sum(1 for c in mine if not c["ok"])
    assert FAST_6M.coverage["usage"]["total"]["calls"] == sum(row["calls"] for row in FAST_6M.timing["models"].values())      # the token ledger of the run and the timed calls count the same calls
    for day, row in per_day.items():
        assert row["seconds"] >= max(row["stages"].values()) and set(row["stages"]) <= {"account_manager", "data_clerk", "advisor", "critic"} and all(c["stage"] for c in row["calls"])
