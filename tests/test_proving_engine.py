"""The Time Machine's walk (prometheus/proving/engine.py; design/proving_ground_spec.md section 7; P2, P4, P5, P6, P7) over the REAL research of a real run, cached: FREE, no model is called. The saved real run is
`tests/fixtures/proving_real/oil_weekly_short.json` (the oil request, 4 weekly decision days, cash 10000) and its research cache is beside it (`oil_weekly_short_cache/`). A test that would need a model fails loudly instead of paying for one."""
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.proving import engine
from prometheus.proving.memo import NAMES, Memo
from prometheus.proving.record import ProvingRun
from prometheus.proving.store import ProvingStore
from prometheus.rag.config import REPO_ROOT

REAL = Path(__file__).parent / "fixtures" / "proving_real"
SAVED = ProvingRun.model_validate_json((REAL / "oil_weekly_short.json").read_text(encoding="utf-8"))
REQUEST = engine.TimeMachineRequest(**{k: v for k, v in SAVED.request.items() if k in engine.TimeMachineRequest.model_fields}, **({} if "speed" in SAVED.request else {"speed": "faithful"}))      # the real captures were made with the live settings: they are FAITHFUL (a run saved before the speeds existed has no `speed`)
INTENT = TradeIntent.model_validate(SAVED.request["intent"])
REAL_RESEARCH_DAY = engine.research_day


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store in a temporary folder holding the real cached research; the data's stamp and the models are those of the saved run, and a day that is not cached cannot be researched (no model is ever called)."""
    kept = ProvingStore(tmp_path)
    (tmp_path / "cache").mkdir()
    for path in (REAL / "oil_weekly_short_cache").glob("*.json"):
        shutil.copy2(path, tmp_path / "cache" / path.name)
    monkeypatch.setattr(engine, "data_stamp", lambda report: SAVED.coverage["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: SAVED.models)
    monkeypatch.setattr(engine, "versions", lambda: SAVED.config_versions)
    monkeypatch.setattr(engine, "spare_of", lambda config=None: 0)      # the saved research asked the Advisor for no spare recommendations (Critic v2 spec section 8): its keys are the keys of before the spare
    monkeypatch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: SAVED.range.last_bar_day)      # the saved run's horizon: the stored bars have moved on since it was recorded (a later close is stored every day)

    def refuse(*args, **kwargs):
        raise AssertionError("a test must not research a day: it is not in the cache")
    monkeypatch.setattr(engine, "research_day", refuse)
    return kept


def run(store, **changes):
    request = REQUEST.model_copy(update=changes)
    return engine.run_time_machine(request, store=store, intent=INTENT)


def test_a_stopped_run_is_saved_as_stopped_with_what_it_has(store):
    calls = iter(range(100))
    stopped = engine.run_time_machine(REQUEST, store=store, intent=INTENT, should_stop=lambda: next(calls) > 12)
    assert stopped.status == "stopped" and 0 < len(stopped.equity) < len(SAVED.equity) and store.load_run(stopped.run_id).status == "stopped"


def test_a_day_that_cannot_be_researched_fails_and_the_walk_goes_on(store, monkeypatch):
    """With one day out of the cache and no model that can start, that day is `failed` with its reason, trades nothing, and the other days are walked."""
    key = engine.research_key(INTENT, SAVED.decision_days[1].day, REQUEST, SAVED.coverage["stamp"])
    (store.root / "cache" / f"{key}.json").unlink()

    def no_model(*args, **kwargs):
        raise RuntimeError("OPENROUTER_API_KEY is not set (a test must not build a real model)")

    def research_with_no_model(*args, **kwargs):
        with monkeypatch.context() as inner:      # the research of the day finds no model that can start; the walk's own clerk is built outside it
            inner.setattr("prometheus.agents.data_clerk.data_clerk_agent.get_chat_model", no_model)
            return REAL_RESEARCH_DAY(*args, **kwargs)
    monkeypatch.setattr(engine, "research_day", research_with_no_model)
    result = run(store)
    bad = next(d for d in result.decision_days if d.day == SAVED.decision_days[1].day)
    assert result.status == "finished" and bad.status == "failed" and "OPENROUTER_API_KEY" in bad.reason and bad.traded == 0 and len(result.equity) == len(SAVED.equity)
    assert [d.status for d in result.decision_days if d is not bad] == [d.status for d in SAVED.decision_days if d.day != bad.day]


def test_the_memo_keeps_only_what_is_the_same_whatever_the_day_and_asks_the_call_once():
    """P2: the memo refuses a dated call and any name that is not on its list; two threads asking for the same key make ONE call."""
    import threading
    memo, calls = Memo(), []
    with pytest.raises(ValueError):
        memo.get("financials", ("XOM", date(2026, 11, 9)), lambda: 1)
    with pytest.raises(ValueError):
        memo.get("bars", "XOM", lambda: 1)
    answers = []
    threads = [threading.Thread(target=lambda: answers.append(memo.get("financials", "XOM", lambda: calls.append(1) or {"metrics": {}}))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and len(answers) == 8 and memo.hits == 7 and {name for name, _ in memo.asked} <= set(NAMES)
    answers[0]["metrics"]["x"] = 1
    assert "x" not in memo.get("financials", "XOM", lambda: None)["metrics"]      # a copy: one day's reading never changes another's


def test_nothing_is_written_outside_the_store_and_paper_trading_off_refuses(store, monkeypatch):
    """P6: the real ledgers, the news files and the vector store are the same before and after a run; PAPER_TRADING off refuses it."""
    watched = [*sorted((REPO_ROOT / "data" / "ledger").glob("*.jsonl")), *sorted((REPO_ROOT / "news").glob("*.txt"))[:10], REPO_ROOT / "data" / "processed" / "vectorstore" / "chroma.sqlite3"]
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched if p.exists()}
    run(store)
    assert before == {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched if p.exists()}
    monkeypatch.setenv("PAPER_TRADING", "false")
    with pytest.raises(engine.RefusedRequest, match="PAPER_TRADING"):
        run(store)
