"""The Time Machine's speed (prometheus/proving/speed.py, engine.py, design/proving_ground_spec.md P14): the fast mode, the faithful mode and what they share. FREE: the research is the REAL cached research of the real run
`tests/fixtures/proving_real/oil_weekly_short.json` (made with the live settings: it is a FAITHFUL capture), no model is called, nothing is fetched. A test that would need a model fails loudly instead of paying for one."""
import json
import shutil
import threading
import time
from datetime import date
from pathlib import Path

import pytest
from proving_helpers import as_before_the_forward_excess, outcomes_before_the_forward_return

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


def stable(run: ProvingRun) -> dict:
    """A run without what changes between two runs of the same thing (its id, clock and timings) and without what a SPEED says of itself."""
    body = run.model_dump(mode="json")
    for key in ("run_id", "created_utc", "seconds", "langsmith", "timing", "speed", "caveats"):
        body.pop(key)
    body["request"] = {k: v for k, v in body["request"].items() if k != "speed"}
    body["coverage"] = {k: v for k, v in body["coverage"].items() if k not in ("timing", "memo", "desks", "cache")}
    body["desks"] = {day: ([(d["symbol"], d["will_trade"], d["shares"]) for d in desk["trade_decisions"]], [(t["symbol"], t["status"], t["fill"]) for t in desk["trades"]]) for day, desk in run.coverage["desks"].items()}
    return body


# ---------------------------------------------------------------- the cache key

def test_the_cache_key_keeps_a_fast_day_apart_from_a_faithful_one_and_every_real_capture_is_faithful():
    """A fast day never answers a faithful run (and the other way round); a changed fast setting is another research; and the FAITHFUL key is the key the real captures were stored under, so they still answer."""
    day, stamp, config = date(2026, 10, 19), SAVED.coverage["stamp"], load_config()
    faithful, fast = engine.research_key(INTENT, day, FAITHFUL, stamp), engine.research_key(INTENT, day, FAST, stamp)
    assert faithful != fast
    harder = config.model_copy(update={"time_machine": config.time_machine.model_copy(update={"fast": config.time_machine.fast.model_copy(update={"advisor": config.time_machine.fast.advisor.model_copy(update={"effort": "high"})})})})
    assert engine.research_key(INTENT, day, FAST, stamp, harder) != fast and engine.research_key(INTENT, day, FAITHFUL, stamp, harder) == faithful
    assert engine.research_key(INTENT, day, engine.TimeMachineRequest(**FIELDS), stamp) == fast      # a request with no speed is the config's own (fast)
    for found in CAPTURES:
        assert (REAL / "oil_weekly_short_cache" / f"{engine.research_key(INTENT, date.fromisoformat(found), FAITHFUL, stamp)}.json").is_file()


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

def test_the_same_research_walked_fast_and_faithful_gives_the_same_run_and_only_the_fast_one_says_so(tmp_path):
    """The speed is only about how the models were asked: the walk, the sizing, the fills, the exits and every statistic are the same code, so the same research gives the same run; the fast run carries the caveat."""
    faithful = engine.run_time_machine(FAITHFUL, store=stored(tmp_path / "a", FAITHFUL), intent=INTENT)
    fast = engine.run_time_machine(FAST, store=stored(tmp_path / "b", FAST), intent=INTENT)
    assert faithful.speed == "faithful" and fast.speed == "fast" and fast.request["speed"] == "fast" and faithful.request["speed"] == "faithful"
    assert "fast_models" in fast.caveats and "fast_models" not in faithful.caveats
    assert stable(fast) == stable(faithful)
    assert set(fast.caveats) - {"fast_models"} == set(faithful.caveats)


def test_the_speed_reaches_nothing_that_a_program_computes():
    """Only the engine (and the web bridge, to NAME the run's speed) imports the speed module, and what a mode holds is how the models are asked and nothing else: no score, gate, threshold, verdict, size or fill is read from it."""
    importers = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "prometheus").rglob("*.py") if "proving import speed" in p.read_text(encoding="utf-8") or "proving.speed import" in p.read_text(encoding="utf-8"))
    assert importers == ["prometheus/application/web/proving_run.py", "prometheus/proving/engine.py"]
    assert set(load_config().time_machine.fast.__class__.model_fields) == {"provider_sort", "parallel_records", "call_limit_seconds", "run_seconds_note", "cost_note_per_day_usd", "advisor", "critic", "data_clerk"}
    assert set(load_config().time_machine.fast.advisor.__class__.model_fields) == {"effort", "max_output_tokens", "timeout_seconds", "max_retries"}


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


# ---------------------------------------------------------------- the days come back in order, whoever finishes first

def test_days_researched_in_a_pool_are_put_back_in_order_with_one_thread_or_many(tmp_path, monkeypatch):
    """The pool's results are collected by day: the days finish in the REVERSE of their order (the earliest sleeps longest) and the run is the one the cache gives and the one a single thread gives."""
    order = sorted(CAPTURES)
    finished: list[str] = []

    def research(day, request, intent, memo, guard, run_id, audit_path, models=None):
        time.sleep(0.3 * (len(order) - order.index(day.isoformat())))
        finished.append(day.isoformat())
        entry = entry_of(day.isoformat())
        return engine.Day(day, entry["order"], entry["plan"], "done", 0.0, False, None, entry.get("usage"), None, {"seconds": 0.0, "stages": {}, "calls": []}, None)
    monkeypatch.setattr(engine, "research_day", research)
    by_cache = engine.run_time_machine(FAITHFUL, store=stored(tmp_path / "cache", FAITHFUL), intent=INTENT)
    runs = {}
    for pool in (1, 8):
        finished.clear()
        monkeypatch.setattr(engine, "engine_settings", lambda pool=pool: {"parallel_days": pool, "retry_429": 0})
        runs[pool] = engine.run_time_machine(FAITHFUL, store=ProvingStore(tmp_path / f"pool{pool}"), intent=INTENT, use_cache=False)
        assert sorted(finished) == order and (finished == order if pool == 1 else finished == order[::-1])
    assert stable(runs[1]) == stable(runs[8]) == stable(by_cache)
    assert runs[8].timing["parallel_days"] == len(order) and runs[1].timing["parallel_days"] == 1 and [d.day.isoformat() for d in runs[8].decision_days] == order


def test_a_cached_rerun_needs_no_model_and_no_key(tmp_path, monkeypatch):
    """The models of a run are made only when a day has to be researched: a fast run whose days are all in the cache starts no model (the key of the provider is never read)."""
    def must_not_build(*args, **kwargs):
        raise AssertionError("a run that is all cache must not build a model")
    monkeypatch.setattr(speed_module, "build_model", must_not_build)
    run = engine.run_time_machine(FAST, store=stored(tmp_path, FAST), intent=INTENT)
    assert run.coverage["timing"]["cached_days"] == len(run.decision_days) and all(row["cached"] for row in run.timing["per_day"].values())


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


def test_the_real_six_month_run_was_fast_and_its_research_answers_a_fast_request_only(monkeypatch):
    """The recorded run (23 weekly decision days, researched all at once): it says it is fast, carries the caveat, its research is cached under the key of a FAST request and under no faithful one, and its time is measured per day."""
    stamp = FAST_6M.coverage["stamp"]
    assert FAST_6M.speed == "fast" and FAST_6M.request["speed"] == "fast" and "fast_models" in FAST_6M.caveats and len(FAST_6M.decision_days) == 23
    cache = REAL / "oil_weekly_6m_fast_cache"
    monkeypatch.setattr(engine, "models_used", lambda: FAST_6M.models)
    monkeypatch.setattr(engine, "versions", lambda: FAST_6M.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare
    faithful = FAST_6M_REQUEST.model_copy(update={"speed": "faithful"})
    for row in FAST_6M.decision_days:
        assert (cache / f"{engine.research_key(FAST_6M_INTENT, row.day, FAST_6M_REQUEST, stamp)}.json").is_file()
        assert not (cache / f"{engine.research_key(FAST_6M_INTENT, row.day, faithful, stamp)}.json").exists()
    timing = FAST_6M.timing
    assert timing["speed"] == "fast" and timing["parallel_days"] == 23 and timing["cached_days"] == 0 and set(timing["per_day"]) == {r.day.isoformat() for r in FAST_6M.decision_days}
    assert 0 < timing["research_seconds"] < 300 and FAST_6M.coverage["timing"] == timing      # five minutes is the target; the page reads `coverage.timing` and the run's own `timing` is the same


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


def test_the_walk_over_the_real_fast_research_is_the_saved_run_and_a_cached_rerun_is_quick(tmp_path, monkeypatch):
    """Cached, a fast run is the walk alone: the same run as the recorded one, from the cache, in seconds, with no model and no key. The `cooldown` reason and the `source` of every outcome are the engine's own."""
    monkeypatch.setattr(engine, "data_stamp", lambda report: FAST_6M.coverage["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: FAST_6M.models)
    monkeypatch.setattr(engine, "versions", lambda: FAST_6M.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare
    monkeypatch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: FAST_6M.range.last_bar_day)      # the recorded horizon: the stored bars have moved on since
    monkeypatch.setenv("OPENROUTER_API_KEY", "")      # a cached run needs no key
    again = engine.run_time_machine(FAST_6M_REQUEST, store=six_month_store(tmp_path), intent=FAST_6M_INTENT)
    assert again.coverage["timing"]["cached_days"] == 23 and again.speed == "fast" and "fast_models" in again.caveats
    assert [(o.key, o.result, o.return_pct, o.not_traded_reason, o.source) for o in again.outcomes] == [(o.key, o.result, o.return_pct, o.not_traded_reason, o.source) for o in FAST_6M.outcomes]
    assert as_before_the_forward_excess(again.stats) == as_before_the_forward_excess(FAST_6M.stats) and outcomes_before_the_forward_return(again) == outcomes_before_the_forward_return(FAST_6M) and [p.equity for p in again.equity] == [p.equity for p in FAST_6M.equity]      # the fixture's last day was walked again once the bars of 2026-11-20 had settled (entry 129)
    assert again.seconds < 30 and again.timing["research_seconds"] < 3      # the recorded run took more than a minute in its research; from the cache it is the reading of 23 files
    held_back = [o for o in again.outcomes if o.notes and "Advisor's cooldown" in " ".join(o.notes)]
    assert held_back and {o.not_traded_reason for o in held_back} == {"cooldown"} and {o.source for o in again.outcomes} == {"time_machine"}


def test_a_day_with_few_news_articles_is_thin_and_the_run_says_so(tmp_path, monkeypatch):
    """`news_coverage_thin`: a decision day whose research found fewer news articles for its companies than the config's `news_coverage_min_sources` (the count was kept with the research). The recorded run was made with a lower mark and has none."""
    monkeypatch.setattr(engine, "data_stamp", lambda report: FAST_6M.coverage["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: FAST_6M.models)
    monkeypatch.setattr(engine, "versions", lambda: FAST_6M.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare
    monkeypatch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: FAST_6M.range.last_bar_day)
    mark = load_config().time_machine.news_coverage_min_sources
    counts = {day: found for day, found in FAST_6M.coverage["news_sources"].items()}
    thin = sorted(day for day, found in counts.items() if found < mark)
    assert thin and len(thin) < len(counts) and "news_coverage_thin" not in FAST_6M.caveats and FAST_6M.coverage["news_thin_days"] == []
    again = engine.run_time_machine(FAST_6M_REQUEST, store=six_month_store(tmp_path), intent=FAST_6M_INTENT)
    assert again.coverage["news_thin_days"] == thin and "news_coverage_thin" in again.caveats
