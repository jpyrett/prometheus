"""Integrity: the record against itself and against the order (Critic spec 8.1). The three REAL captures must pass every check; each check must fail on its own tampering."""
import pytest
from critic_helpers import ADVISOR_CONFIG, captured, rec_of

from prometheus.agents.critic import integrity as ig


def run(order, **kw):
    return ig.check(rec_of(order), order, ADVISOR_CONFIG, **kw)


@pytest.mark.parametrize("name", ["SNDK", "PG", "PG_open"])
def test_a_real_record_passes_every_check(name):
    result = run(captured(name))
    assert result.failures == [] and len(result.checked) > 50 and not result.no_thesis


# each tampering of PG_open's record, and the check that must notice it


def test_close_treats_two_nones_as_the_same_and_one_as_different():
    assert ig.close(None, None) and not ig.close(None, 0.0) and ig.close(1.0, 1.0000001, 1e-6) and not ig.close(1.0, 1.1, 1e-6)


def test_the_composite_is_the_best_bullish_edge_less_the_largest_bearish():
    from prometheus.agents.advisor.record import FamilyResult

    def fam(direction, edge, passed=True):
        return FamilyResult(family="trend", computable=True, passed=passed, direction=direction, edge_20d=edge, horizon_bars=10, prior_20d=0.01)

    assert ig.composite_of([fam("bullish", 0.01), fam("bullish", 0.02), fam("bearish", -0.005)]) == pytest.approx(0.015)
    assert ig.composite_of([fam("bullish", 0.01)]) == 0.01 and ig.composite_of([fam("bearish", -0.004)]) == -0.004
    assert ig.composite_of([fam("bullish", 0.01, passed=False)]) is None and ig.composite_of([]) is None
