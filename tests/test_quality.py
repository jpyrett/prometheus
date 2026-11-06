"""Tests for prometheus.agents.advisor.quality (Advisor v1 spec section 9.3, phase 3)."""
import pytest
from advisor_helpers import CONFIG, TODAY, healthy, metric

from prometheus.agents.advisor.quality import quality_multiplier

Q = CONFIG.quality


def quality(metrics):
    return quality_multiplier(metrics, TODAY, Q)


def test_a_healthy_company_has_multiplier_one_and_no_haircuts():
    result = quality(healthy())
    assert result.multiplier == 1.0 and result.haircuts == [] and result.haircuts_skipped == [] and result.turned_off == []


@pytest.mark.parametrize("override, check, factor", [
    (dict(operating_margin=0.0), "operating_margin", 0.6), (dict(roe=7.99), "roe", 0.85), (dict(revenue_growth=-0.01), "revenue_growth", 0.85),
    (dict(debt_to_equity=2.01), "debt_to_equity", 0.7), (dict(current_ratio=0.99), "current_ratio", 0.85), (dict(share_count_change=0.011), "share_count_change", 0.8),
    (dict(dilution_current=True), "dilution_current", 0.6), (dict(sbc_to_revenue=0.16), "sbc_to_revenue", 0.7), (dict(accrual_ratio=0.11), "accrual_ratio", 0.5),
    (dict(receivables_vs_sales=15.1), "receivables_vs_sales", 0.5), (dict(inventory_vs_sales=15.1), "inventory_vs_sales", 0.5), (dict(cfo_negative_with_profit=True), "cfo_negative_with_profit", 0.5),
    (dict(risk_factor_change=0.36), "risk_factor_change", 0.75),
])
def test_each_haircut_applies_just_past_its_threshold(override, check, factor):
    result = quality(healthy(**override))
    assert [(h.check, h.factor) for h in result.haircuts] == [(check, factor)] and result.multiplier == pytest.approx(factor)


@pytest.mark.parametrize("override", [dict(operating_margin=0.01), dict(roe=8.0), dict(revenue_growth=0.0), dict(debt_to_equity=2.0), dict(current_ratio=1.0), dict(share_count_change=0.01),
                                      dict(sbc_to_revenue=0.15), dict(accrual_ratio=0.10), dict(receivables_vs_sales=15.0), dict(risk_factor_change=0.35)])
def test_no_haircut_exactly_at_the_threshold(override):
    assert quality(healthy(**override)).haircuts == []


def test_factors_multiply_and_the_floor_is_0_2():
    result = quality(healthy(operating_margin=-5.0, roe=1.0, revenue_growth=-3.0))
    assert result.multiplier == pytest.approx(0.6 * 0.85 * 0.85)
    terrible = quality(healthy(operating_margin=-5.0, roe=1.0, revenue_growth=-3.0, debt_to_equity=5.0, current_ratio=0.5, share_count_change=0.05, dilution_current=True, sbc_to_revenue=0.5,
                               accrual_ratio=0.5, cfo_negative_with_profit=True))
    assert terrible.multiplier == 0.2 and len(terrible.haircuts) > 6


def test_the_earnings_quality_checks_take_their_factor_once_and_turn_reversion_off():
    result = quality(healthy(accrual_ratio=0.5, receivables_vs_sales=30.0, inventory_vs_sales=40.0))
    assert result.multiplier == 0.5 and [h.factor for h in result.haircuts] == [0.5, 1.0, 1.0]   # the group's factor once; the others are recorded, not applied again
    assert result.turned_off == ["reversion"]
    assert quality(healthy(cfo_negative_with_profit=True)).turned_off == ["reversion"]
    assert quality(healthy(roe=1.0)).turned_off == []


def test_a_missing_input_applies_no_factor_and_is_listed_with_its_reason():
    result = quality(healthy(roe=None, accrual_ratio=None))
    assert result.multiplier == 1.0 and result.haircuts == []
    assert "roe: needs bars for roe" in result.haircuts_skipped and "accrual_ratio: needs bars for accrual_ratio" in result.haircuts_skipped
    nothing = quality(None)
    assert nothing.multiplier == 1.0 and len(nothing.haircuts_skipped) == len(Q.checks)


def test_a_statement_older_than_120_days_skips_the_checks_that_read_it():
    metrics = healthy(accrual_ratio=0.5, cfo_negative_with_profit=True, roe=1.0)
    for name in ("accrual_ratio", "cfo_negative_with_profit"):
        metrics[name] = metric(metrics[name]["value"], as_of="2026-06-22")        # 123 days before the run
    result = quality(metrics)
    assert [h.check for h in result.haircuts] == ["roe"] and result.turned_off == []
    assert any(s.startswith("accrual_ratio: the statement is from 2026-06-22, more than 120 days old") for s in result.haircuts_skipped)
    metrics["accrual_ratio"] = metric(0.5, as_of="2026-06-25")                   # 120 days exactly is still allowed
    assert "accrual_ratio" in [h.check for h in quality(metrics).haircuts]
