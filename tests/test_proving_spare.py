"""The cooldown's replacement (design/critic_v2_evidence_spec.md section 8; design choices entry 135): phase 1 asks the Advisor for `time_machine.spare_recommendations` MORE than the number asked for, and the walk takes them in rank order, skips what the Advisor's own
cooldown catches and keeps the first `find`; the ones beyond it are `beyond_count`. On the REAL research of nine weekly decision days (2026-09-14 to 2026-11-09) of a real fast Time Machine run made with Critic version 2 and the spare
(`tests/fixtures/critic_v2_real/`: the oil request, cash 10000): FREE, no model is called. A test that would need a model fails loudly instead of paying for one."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest
from advisor_helpers import MemoryLedger, Scripted
from test_advisor_sndk_pg import FIXTURES, RecordedClerk

from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.agents.advisor.advisor import Advisor
from prometheus.common import work_order as wo
from prometheus.proving import engine
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


