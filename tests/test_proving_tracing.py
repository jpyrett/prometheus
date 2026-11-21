"""LangSmith for the Proving Ground (prometheus/common/trace.py `root` and `span`, prometheus/proving/tracing.py; design/proving_ground_spec.md section 12). Real data: the real Track Record outcomes
(`tests/fixtures/proving_real/track.json`) and a real saved session. Tracing is forced off by `tests/conftest.py`: nothing is posted and nothing fails."""
import json
from pathlib import Path

from prometheus.common import trace
from prometheus.common.work_order import WorkOrder
from prometheus.proving import tracing
from prometheus.proving.record import ProvingRun

REAL = Path(__file__).parent / "fixtures" / "proving_real"


def real_outcomes():
    return ProvingRun.model_validate(json.loads((REAL / "track.json").read_text(encoding="utf-8"))).outcomes


def test_with_tracing_off_nothing_is_made_posted_or_failed():
    """`root` and `span` yield None, `set_outputs` is a no-op, and `post_outcomes` posts nothing and does not raise, even for a trace id."""
    with trace.root("proving.time_machine", metadata={"cash": 10000}, tags=["proving"]) as root:
        assert root is None
        with trace.span("proving.decision_day", metadata={"as_of": "2026-11-09"}) as day:
            assert day is None
            trace.set_outputs(day, {"headline": "x"})
    class Refusing:
        def create_feedback(self, **row):
            raise AssertionError("nothing may be posted with tracing off")
    assert tracing.post_outcomes("6b0e1d3e-55f2-4d0b-9d37-0b5e2f7f2a11", real_outcomes(), client=Refusing()) == 0


def test_the_feedback_of_a_day_is_deterministic_and_carries_figures_only():
    """The same decision day always gives the same feedback ids (a second scoring updates, never duplicates), one `rec_outcome` for each scored recommendation, the day's mean excess and win rate over its TRADED ones, and no text of an article."""
    day = [o for o in real_outcomes() if o.decision_day.isoformat() == "2026-11-13"]
    first, second = tracing.day_feedback("run-1", day), tracing.day_feedback("run-1", day)
    assert first == second and len({r["feedback_id"] for r in first}) == len(first)
    assert {r["feedback_id"] for r in first}.isdisjoint({r["feedback_id"] for r in tracing.day_feedback("run-2", day)})
    traded = [o for o in day if o.traded and o.excess_return_pct is not None]
    excess = next(r for r in first if r["key"] == "outcome_excess_return")
    assert excess["score"] == round(sum(o.excess_return_pct for o in traded) / len(traded), 6)
    assert len([r for r in first if r["key"] == "rec_outcome"]) == len([o for o in day if o.result in ("won", "lost", "open")])
    assert all(set(r) == {"run_id", "key", "score", "value", "comment", "feedback_id"} for r in first)
    assert all(o.symbol in r["comment"] for r, o in zip([r for r in first if r["key"] == "rec_outcome"], [o for o in day if o.result in ("won", "lost", "open")]))


def test_a_work_order_in_a_proving_span_is_ids_and_counts_never_article_text():
    """`summarize` is what every proving span's inputs and outputs go through: a real work order with its articles is its id, its symbols and the COUNT of what it holds."""
    capture = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "auto_even_split_dollars.json").read_text(encoding="utf-8"))
    summary = trace.summarize(WorkOrder.model_validate(capture["order"]))
    text = json.dumps(summary)
    assert summary["type"] == "WorkOrder" and summary["counts"]["recommendations"] == 5 and len(text) < 3000
    thesis = json.dumps(capture["order"]["recommendations"][0]["thesis"])
    assert len(thesis) > 100 and thesis[20:80] not in text
