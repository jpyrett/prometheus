"""The Time Machine's walk (prometheus/proving/engine.py; design/proving_ground_spec.md section 7; P2, P4, P5, P6, P7) over the REAL research of a real run, cached: FREE, no model is called. The saved real run is
`tests/fixtures/proving_real/oil_weekly_short.json` (the oil request, 4 weekly decision days, cash 10000) and its research cache is beside it (`oil_weekly_short_cache/`). A test that would need a model fails loudly instead of paying for one."""
import json
import os
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from proving_helpers import as_before_the_forward_excess, outcomes_before_the_forward_return

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.common.work_order import WorkOrder
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


def stable(saved: ProvingRun) -> dict:
    """A run without what changes between two runs of the same thing: its id, its clock and its timings."""
    body = saved.model_dump(mode="json")
    for key in ("run_id", "created_utc", "seconds", "langsmith", "timing"):
        body.pop(key)
    body["coverage"] = {k: v for k, v in body["coverage"].items() if k not in ("timing", "memo", "desks")}      # a desk holds the Trader's own ids of a run: its decisions and fills are compared below
    body["desks"] = {day: ([(d["symbol"], d["will_trade"], d["shares"]) for d in desk["trade_decisions"]], [(t["symbol"], t["status"], t["fill"]) for t in desk["trades"]]) for day, desk in saved.coverage["desks"].items()}
    return body


def test_the_walk_over_the_cached_research_is_the_saved_run_and_the_same_twice(store, monkeypatch):
    """No model: the same cached research, the same stored bars, the same result: as the saved real run, as itself a second time, and whatever the pool's size is (the pool only matters to phase 1)."""
    first = run(store)
    assert first.status == "finished" and [(d.day, d.status) for d in first.decision_days] == [(d.day, d.status) for d in SAVED.decision_days]
    assert [(o.key, o.result, o.return_pct) for o in first.outcomes] == [(o.key, o.result, o.return_pct) for o in SAVED.outcomes]
    assert as_before_the_forward_excess(first.stats) == as_before_the_forward_excess(SAVED.stats) and [p.equity for p in first.equity] == [p.equity for p in SAVED.equity]      # the saved run (made before the forward excess) is reproduced in everything it recorded
    assert outcomes_before_the_forward_return(first) == outcomes_before_the_forward_return(SAVED) and SAVED.outcomes[0].forward == [] and len(first.outcomes[0].forward) == 4      # E8: old records are followed exactly as before; only the new `forward` block is added
    assert first.coverage["timing"]["cached_days"] == len(first.decision_days)
    monkeypatch.setattr(engine, "engine_settings", lambda: {"parallel_days": 1, "retry_429": 0})
    second = run(store)
    assert stable(first) == stable(second)


def test_the_cooldown_taken_is_the_advisors_own_function_on_the_same_records():
    """A recommendation of a later day is taken off exactly when `ranking.cooldown_of` says an earlier one of this run still stands (same symbol and family, time stop not passed, no invalidation fired)."""
    from types import SimpleNamespace

    from prometheus.agents.advisor import ranking
    cached = sorted((REAL / "oil_weekly_short_cache").glob("*.json"))
    days = sorted((d for d in (json.loads(p.read_text(encoding="utf-8")) for p in cached) if "day" in d), key=lambda d: d["day"])
    earlier: list[dict] = []
    for found in days:
        order = WorkOrder.model_validate(found["order"])
        day = date.fromisoformat(found["day"])
        expected = []
        for rec in order.recommendations:
            seen = SimpleNamespace(metrics=rec["features"], events=rec["evidence"]["events"])
            if ranking.cooldown_of(seen, rec["trade"]["family"], [r for r in earlier if r["symbol"] == rec["symbol"]], day.isoformat()):
                expected.append(rec["identity"]["record_id"])
        taken = engine.cooldown_filter(order, day, earlier)
        assert [c["identity"]["record_id"] for c in taken] == expected
        assert all(r["identity"]["record_id"] not in expected for r in order.recommendations)
        earlier.extend(order.recommendations)


def test_another_amount_of_cash_changes_the_sizes_and_not_the_recommendations(store):
    """The research does not depend on the money: with $1,000 instead of $10,000 the same recommendations are scored, and what is bought is smaller: a day never spends more than the account's cash THAT day or than the pot (P7)."""
    rich, poor = run(store), run(store, cash=Decimal("1000.00"))
    assert [o.key for o in rich.outcomes if o.verdict != "neutral"] == [o.key for o in poor.outcomes if o.verdict != "neutral"]
    spent = lambda r: sum(d.spent for d in r.decision_days)
    assert 0 < spent(poor) < spent(rich)
    for saved in (rich, poor):
        cash_on = {p.date: p.cash for p in saved.equity}      # the point of a decision day is marked after the decision and before its fill: its cash is the cash the Account Manager sized with
        assert all(d.spent <= cash_on[d.day] and d.spent <= Decimal("3000.00") for d in saved.decision_days)
        assert all(p.cash >= 0 for p in saved.equity)


def test_a_run_never_spends_more_than_its_money_and_its_books_balance(store):
    """P7 on the real walk: every day the equity is the cash plus the positions and the cash is never negative; the cash only moves by the real fills and exits; a position's exit is never before its entry."""
    saved = run(store)
    paid = sum(e.amount for e in saved.events if e.kind == "buy")
    got = sum(e.amount for e in saved.events if e.kind == "exit")
    assert all(p.equity == p.cash + p.invested and p.cash >= 0 for p in saved.equity)
    assert saved.equity[-1].cash == REQUEST.cash - paid + got and paid > REQUEST.cash > paid - got      # the cash from the exits bought again: more was paid in all than the account started with, never more than it had
    assert all(e.date >= next(o.entry.date for o in saved.outcomes if o.key == e.key) for e in saved.events if e.kind == "exit")


def test_the_engines_exits_are_the_traders_followers_exits(store):
    """One exit rule: the run checked every filled purchase's `Trade.outcome` against the account's own sale (or open lot) and did not fail; a position that was sold has the exit date, price and reason `follow` gave."""
    saved = run(store)
    sold = [e for e in saved.events if e.kind == "exit"]
    for event in sold:
        outcome = next(o for o in saved.outcomes if o.key == event.key)
        assert (outcome.exit.date, outcome.exit.price, outcome.exit.reason) == (event.date, event.price, event.reason)


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
