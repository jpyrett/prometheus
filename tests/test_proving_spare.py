"""The cooldown's replacement (design/critic_v2_evidence_spec.md section 8; design choices entry 135): phase 1 asks the Advisor for `time_machine.spare_recommendations` MORE than the number asked for, and the walk takes them in rank order, skips what the Advisor's own
cooldown catches and keeps the first `find`; the ones beyond it are `beyond_count`. On the REAL research of nine weekly decision days (2026-09-14 to 2026-11-09) of a real fast Time Machine run made with Critic version 2 and the spare
(`tests/fixtures/critic_v2_real/`: the oil request, cash 10000): FREE, no model is called. A test that would need a model fails loudly instead of paying for one."""
import copy
import json
import shutil
from datetime import date
from pathlib import Path

import pytest
from advisor_helpers import MemoryLedger, Scripted
from test_advisor_sndk_pg import FIXTURES, RecordedClerk

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.agents.advisor.advisor import Advisor
from prometheus.common import work_order as wo
from prometheus.proving import engine
from prometheus.proving.config import load_config
from prometheus.proving.store import ProvingStore

REAL = Path(__file__).parent / "fixtures" / "critic_v2_real"
META = json.loads((REAL / "oil_weekly_9d.json").read_text(encoding="utf-8"))
REQUEST = engine.TimeMachineRequest(**{**META["request"], "start": "2026-09-14", "end": "2026-11-09"})
INTENT = TradeIntent.model_validate(META["intent"])
ASKED = INTENT.count


def put_research(root: Path, mutate=None, reverse=False) -> ProvingStore:
    kept = ProvingStore(root)
    (root / "cache").mkdir(parents=True)
    files = sorted((REAL / "oil_weekly_9d_cache").glob("*.json"), reverse=reverse)
    for path in files:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if mutate:
            mutate(entry)
        (root / "cache" / path.name).write_text(json.dumps(entry), encoding="utf-8")
    return kept


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(engine, "data_stamp", lambda report: META["stamp"])
    monkeypatch.setattr(engine, "models_used", lambda: META["models"])
    monkeypatch.setattr(engine, "versions", lambda: META["versions"])
    monkeypatch.setattr("prometheus.proving.bars.usable_last_day", lambda now=None: date(2026, 11, 19))      # the bars the tests were written against: a session stored later is not read

    def refuse(*args, **kwargs):
        raise AssertionError("a test must not research a day: it is not in the cache")
    monkeypatch.setattr(engine, "research_day", refuse)


def walk(store) -> "engine.ProvingRun":
    return engine.run_time_machine(REQUEST, store=store, intent=INTENT)


def traded_per_day(run) -> dict:
    out = {d.day.isoformat(): 0 for d in run.decision_days}
    for o in run.outcomes:
        if o.traded:
            out[o.decision_day.isoformat()] += 1
    return out


def cut_to_find(entry: dict) -> None:
    """The same research, cut to the first `find` recommendations before the walk: how a walk with no spare meets the same days (the cooldown takes what it takes and nothing takes its place)."""
    find = (entry.get("plan") or {}).get("find")
    order = entry["order"]
    kept = order["recommendations"][:find]
    ids = {r["identity"]["record_id"] for r in kept}
    order["recommendations"], order["reviews"] = kept, [r for r in order["reviews"] if r["record_id"] in ids]


def test_the_cache_key_holds_the_spare_number_and_the_real_research_was_made_with_three():
    config = load_config()
    assert config.time_machine.spare_recommendations == 3 and engine.spare_of() == 3
    day = date(2026, 10, 19)
    with_none = config.model_copy(update={"time_machine": config.time_machine.model_copy(update={"spare_recommendations": 0})})
    with_more = config.model_copy(update={"time_machine": config.time_machine.model_copy(update={"spare_recommendations": 4})})
    keys = {engine.research_key(INTENT, day, REQUEST, META["stamp"], c) for c in (config, with_none, with_more)}
    assert len(keys) == 3                                                                                          # a research that asked for more is another research


def test_the_keys_of_the_real_research_are_the_keys_the_engine_makes(no_model):
    for day, key in META["keys"].items():
        assert engine.research_key(INTENT, date.fromisoformat(day), REQUEST, META["stamp"]) == key
        assert (REAL / "oil_weekly_9d_cache" / f"{key}.json").is_file()


def test_phase_1_asks_the_advisor_for_the_number_plus_the_spare_and_never_past_its_own_limit():
    """The real Advisor on the real PG capture, wrapped as phase 1 wraps it: the number it is asked for is `top_k` plus the spare, and never over the 20 the Advisor itself keeps."""
    fx = json.loads((FIXTURES / "PG_open.json").read_text(encoding="utf-8"))

    class Spy(Advisor):
        asked: list = []

        def generate_recommendations(self, order, case=None, today=None, top_k=None, only_subject=False):
            self.asked.append(top_k)
            return super().generate_recommendations(order, case, today, top_k=top_k, only_subject=only_subject)
    spy = Spy(model=Scripted(*fx["replies"]), clerk=RecordedClerk(fx["metrics"]), ledger=MemoryLedger())
    order = wo.WorkOrder.model_validate(fx["order"])
    wrapped = engine.SpareAdvisor(spy, 3)
    out = wrapped.generate_recommendations(order, "funds", date.fromisoformat(fx["today"]), top_k=3, only_subject=True)
    assert spy.asked == [6] and [r["symbol"] for r in out.recommendations] == ["PG"] and wrapped.config is spy.config      # the Advisor's own work, and its own attributes
    engine.SpareAdvisor(spy, 3).generate_recommendations(wo.WorkOrder.model_validate(fx["order"]), "funds", date.fromisoformat(fx["today"]), top_k=19, only_subject=True)
    assert spy.asked == [6, wo.MAX_TOP_K]


def real_order(day: str):
    entry = json.loads((REAL / "oil_weekly_9d_cache" / f"{META['keys'][day]}.json").read_text(encoding="utf-8"))
    return wo.WorkOrder.model_validate(entry["order"]), entry["plan"]


def test_the_recommendations_after_the_first_find_are_taken_off_with_their_reviews_and_nothing_is_cut_when_the_number_is_not_known():
    order, plan = real_order("2026-09-14")                       # the first day: no cooldown; the Advisor was asked for 3 + 3 and returned four
    assert plan["find"] == 3 == ASKED and len(order.recommendations) == 4 and len(order.reviews) == 4
    fourth = order.recommendations[3]["identity"]["record_id"]
    gone = engine.beyond_count(order, plan["find"])
    assert [g["identity"]["record_id"] for g in gone] == [fourth] and len(order.recommendations) == 3 and fourth not in [r["record_id"] for r in order.reviews] and len(order.reviews) == 3
    again, _ = real_order("2026-09-14")
    for find in (None, 0, True, "3"):
        assert engine.beyond_count(again, find) == [] and len(again.recommendations) == 4
    assert engine.beyond_count(again, 9) == [] and len(again.recommendations) == 4


def test_over_the_real_weeks_a_day_trades_the_number_asked_for_where_the_cooldown_alone_would_have_left_it_short(tmp_path, no_model):
    with_spare = walk(put_research(tmp_path / "a"))
    without = walk(put_research(tmp_path / "b", mutate=cut_to_find))
    mine, theirs = traded_per_day(with_spare), traded_per_day(without)
    assert with_spare.status == without.status == "finished" and list(mine) == list(theirs) and len(mine) == 9
    assert all(mine[day] >= theirs[day] for day in mine) and any(mine[day] > theirs[day] for day in mine) and sum(mine.values()) > sum(theirs.values())      # the spare puts a company in the place of the one the cooldown took
    assert all(n <= ASKED for n in mine.values()), mine                                                               # never more than the number asked for
    reasons = [o.not_traded_reason for o in with_spare.outcomes if not o.traded]
    assert "beyond_count" in reasons and "cooldown" in reasons
    beyond = [o for o in with_spare.outcomes if o.not_traded_reason == "beyond_count"]
    assert all(o.hypothetical and o.verdict in ("approved", "denied") and any("ranked beyond the 3 the request asked for" in n for n in o.notes) for o in beyond)       # a hypothetical outcome, with the reason said
    assert all(o.result in ("won", "lost", "open", "not_filled", "no_data") for o in beyond)
    assert [o.not_traded_reason for o in without.outcomes if o.not_traded_reason == "beyond_count"] == []


def test_the_walk_gives_the_same_result_whatever_order_the_days_were_researched_and_written_in(tmp_path, monkeypatch, no_model):
    """The selection is made in date order in the walk, from each day's own cached research: the order the files were written in (the order the days finished in) and the pool's size change nothing."""
    first = walk(put_research(tmp_path / "a"))
    monkeypatch.setattr(engine, "engine_settings", lambda: {"parallel_days": 1, "retry_429": 0})
    second = walk(put_research(tmp_path / "b", reverse=True))
    strip = lambda run: ([(o.key, o.result, o.return_pct, o.not_traded_reason, o.traded) for o in run.outcomes], [(d.day, d.status, d.recommended, d.traded, d.spent) for d in run.decision_days], [p.equity for p in run.equity])
    assert strip(first) == strip(second) and first.stats == second.stats
