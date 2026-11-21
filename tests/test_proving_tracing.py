"""LangSmith for the Proving Ground (prometheus/common/trace.py `root` and `span`, prometheus/proving/tracing.py; design/proving_ground_spec.md section 12). Real data: the real Track Record outcomes
(`tests/fixtures/proving_real/track.json`) and a real saved session. Tracing is forced off by `tests/conftest.py`: nothing is posted and nothing fails."""
import json
from pathlib import Path

from prometheus.common import trace
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


