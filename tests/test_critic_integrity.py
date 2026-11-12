"""Integrity: the record against itself and against the order (Critic spec 8.1). The three REAL captures must pass every check; each check must fail on its own tampering."""
import pytest
from critic_helpers import ADVISOR_CONFIG, captured, rec_of, tamper, tampered_order

from prometheus.agents.critic import integrity as ig


def run(order, **kw):
    return ig.check(rec_of(order), order, ADVISOR_CONFIG, **kw)


@pytest.mark.parametrize("name", ["SNDK", "PG", "PG_open"])
def test_a_real_record_passes_every_check(name):
    result = run(captured(name))
    assert result.failures == [] and len(result.checked) > 60 and not result.no_thesis


# each tampering of PG_open's record, and the check that must notice it
CASES = [
    ({"trade.stop_price": 1.0}, "recompute.stop"),
    ({"trade.stop_price": 1.0}, "trade.invalidation"),
    ({"trade.suggested_shares": 1000}, "recompute.position"),
    ({"trade.entry_zone": [170.0, 171.0]}, "trade.entry"),
    ({"trade.time_stop_date": "2026-11-01"}, "trade.time_stop"),
    ({"trade.order_style": None}, "trade.size"),
    ({"trade.invalidation": []}, "trade.invalidation"),
    ({"ranking.composite": 0.05}, "recompute.composite"),
    ({"blend.value": 0.0}, "blend.value"),
    ({"blend.score": 1}, "blend.score"),
    ({"blend.components.news.value": 0.9}, "blend.news"),
    ({"blend.components.performance.value": -0.9}, "blend.performance"),
    ({"blend.components.news.parts.0.weight": 0.01}, "blend.recency"),
    ({"blend.components.news.weight": 0.5}, "blend.weights"),
    ({"blend.half_life_days": 30}, "blend.half_life"),
    ({"agreement.confidence": 0.1}, "recompute.confidence"),
    ({"score": 1}, "recompute.score"),
    ({"evidence.signals.0.quote": "words that appear in no source anywhere in this order"}, "evidence.quote"),
    ({"evidence.events.0.quote": "an event quote that appears in no source anywhere"}, "evidence.event"),
    ({"features.last_close.as_of": "2026-10-01"}, "features.fresh"),
    ({"features.last_close.unit": ""}, "complete.features"),
    ({"families.0.basis_ref": ""}, "complete.families"),
    ({"gates.1.passed": False}, "gates"),
    ({"thesis.bull_case.0.ref": "[99]"}, "thesis.refs"),
    ({"audit.metric_order": None}, "audit"),
    ({"identity.request_id": "someone-elses-request"}, "identity.request"),
    ({"audit.config_version": "99"}, "config.version"),
    ({"ranking.stance": "neutral"}, "stance.action"),
]


@pytest.mark.parametrize("changes, check", CASES, ids=[f"{list(c[0])[0]}->{c[1]}" for c in CASES])
def test_each_check_fails_on_its_own_tampering(changes, check):
    assert check in run(tampered_order("PG_open", **changes)).failed


def test_a_feature_a_family_says_it_used_must_be_the_feature():
    order = captured("PG_open")
    family = next(i for i, f in enumerate(order.recommendations[0]["families"]) if f["family"] == "capital_return")
    name = next(n for n, v in order.recommendations[0]["families"][family]["features_used"].items() if isinstance(v, float) and n in order.recommendations[0]["features"])
    order.recommendations[0] = tamper(order.recommendations[0], **{f"families.{family}.features_used.{name}": 123456.789})
    assert "features.used" in run(order).failed


def test_two_records_of_the_same_symbol_and_action_are_flagged_as_duplicates():
    assert "order.unique" in run(captured("PG_open"), duplicate=True).failed and "order.unique" not in run(captured("PG_open")).failed


def test_the_constants_come_from_the_advisors_config_not_from_this_module():
    from prometheus.agents.advisor.config import BlendWeights
    shifted = ADVISOR_CONFIG.model_copy(update={"blend": ADVISOR_CONFIG.blend.model_copy(update={"weights": BlendWeights(news=0.25, performance=0.25, historical=0.25, filings=0.25)})})
    assert "blend.weights" in ig.check(rec_of(captured("PG_open")), captured("PG_open"), shifted).failed


def test_the_filing_evidence_is_looked_up_for_this_symbol_only():
    order = captured("PG_open")
    other = order.data.filing_evidence.pop("PG")
    order.data.filing_evidence["MSFT"] = other        # the E-ids exist, but not for PG
    assert "evidence.quote" in run(order).failed


def test_a_record_with_no_thesis_and_no_reason_is_reported_apart_not_as_a_defect():
    order = tampered_order("PG_open", **{"thesis.thesis": None, "thesis.thesis_error": None})
    result = run(order)
    assert result.no_thesis and "thesis.refs" not in result.failed


def test_close_treats_two_nones_as_the_same_and_one_as_different():
    assert ig.close(None, None) and not ig.close(None, 0.0) and ig.close(1.0, 1.0000001, 1e-6) and not ig.close(1.0, 1.1, 1e-6)


def test_the_composite_is_the_best_bullish_edge_less_the_largest_bearish():
    from prometheus.agents.advisor.record import FamilyResult

    def fam(direction, edge, passed=True):
        return FamilyResult(family="trend", computable=True, passed=passed, direction=direction, edge_20d=edge, horizon_bars=10, prior_20d=0.01)

    assert ig.composite_of([fam("bullish", 0.01), fam("bullish", 0.02), fam("bearish", -0.005)]) == pytest.approx(0.015)
    assert ig.composite_of([fam("bullish", 0.01)]) == 0.01 and ig.composite_of([fam("bearish", -0.004)]) == -0.004
    assert ig.composite_of([fam("bullish", 0.01, passed=False)]) is None and ig.composite_of([]) is None
