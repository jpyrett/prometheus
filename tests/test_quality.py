"""Tests for prometheus.agents.advisor.quality (Advisor v1 spec section 9.3, phase 3)."""
from advisor_helpers import CONFIG, TODAY, healthy, metric

from prometheus.agents.advisor.quality import quality_multiplier

Q = CONFIG.quality


def quality(metrics):
    return quality_multiplier(metrics, TODAY, Q)


def test_the_earnings_quality_checks_take_their_factor_once_and_turn_reversion_off():
    result = quality(healthy(accrual_ratio=0.5, receivables_vs_sales=30.0, inventory_vs_sales=40.0))
    assert result.multiplier == 0.5 and [h.factor for h in result.haircuts] == [0.5, 1.0, 1.0]   # the group's factor once; the others are recorded, not applied again
    assert result.turned_off == ["reversion"]
    assert quality(healthy(cfo_negative_with_profit=True)).turned_off == ["reversion"]
    assert quality(healthy(roe=1.0)).turned_off == []


def test_a_statement_older_than_120_days_skips_the_checks_that_read_it():
    metrics = healthy(accrual_ratio=0.5, cfo_negative_with_profit=True, roe=1.0)
    for name in ("accrual_ratio", "cfo_negative_with_profit"):
        metrics[name] = metric(metrics[name]["value"], as_of="2026-06-22")        # 123 days before the run
    result = quality(metrics)
    assert [h.check for h in result.haircuts] == ["roe"] and result.turned_off == []
    assert any(s.startswith("accrual_ratio: the statement is from 2026-06-22, more than 120 days old") for s in result.haircuts_skipped)
    metrics["accrual_ratio"] = metric(0.5, as_of="2026-06-25")                   # 120 days exactly is still allowed
    assert "accrual_ratio" in [h.check for h in quality(metrics).haircuts]
