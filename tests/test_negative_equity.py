"""Tests for the negative-equity waiver (design choices entry 54): a company with a stockholders' DEFICIT (Domino's, Papa John's, Yum, McDonald's) has no meaningful debt / equity, so the
Advisor's four debt-to-equity rules are not applied to it, and every result says so. The numbers are literals that put a rule at its threshold; they are not market data."""
import pytest
from advisor_helpers import CONFIG, TODAY, family_input, healthy, metrics_of, scored

from prometheus.agents.advisor import families as fam
from prometheus.agents.advisor import gates as gate_module
from prometheus.agents.advisor import ranking as rk
from prometheus.agents.advisor.quality import quality_multiplier

REV = dict(ret_5d=-0.09, zscore_21d=-2.5, rsi_14=28.0, atr_pct=0.02, roe=18.0, debt_to_equity=0.9, current_ratio=1.4, revenue_growth=6.0, residual_5d=-0.06, sector_ret_5d=-0.01, spy_ret_5d=0.0,
           above_sma_200=True, above_sma_50=True, bollinger_pct_b=-0.1, volume_ratio_1d=2.2, candle_reversal_1d=True)
CR = dict(earnings_yield=0.04, debt_to_equity=0.9, share_count_change=-0.02, payout_coverage=0.6, revenue_growth=4.0, cash_balance=10e9)


def run(family, metrics):
    return fam.evaluate(family, family_input(metrics_of(**metrics)))


# ---------------------------------------------------------------- the red-flag gate

def red_flag(**override):
    return {g.gate: g for g in gate_module.run_gates(healthy(**override), [], TODAY, CONFIG.gates).gates}["red_flag"]


# ---------------------------------------------------------------- the families


def test_capital_return_with_negative_equity_does_not_require_or_test_debt_to_equity():
    assert any("debt_to_equity" in n for n in run("capital_return", {**CR, "debt_to_equity": 30.0}).notes)
    waived = run("capital_return", {**CR, "debt_to_equity": 30.0, "negative_equity": True})
    assert waived.passed and waived.direction == "bullish" and waived.waived == ["debt_to_equity"] and gate_module.WAIVED_NOTE in waived.notes
    missing = run("capital_return", {**{k: v for k, v in CR.items() if k != "debt_to_equity"}, "negative_equity": True})
    assert missing.computable and missing.passed
    assert not run("capital_return", {k: v for k, v in CR.items() if k != "debt_to_equity"}).computable
    still = run("capital_return", {**CR, "debt_to_equity": 30.0, "negative_equity": True, "share_count_change": 0.001})
    assert not still.passed and not any("debt_to_equity" in n for n in still.notes if n != gate_module.WAIVED_NOTE)         # its other rules still fail it


# ---------------------------------------------------------------- the quality haircut

def test_the_debt_to_equity_haircut_is_skipped_with_the_reason_for_negative_equity():
    plain = quality_multiplier(healthy(debt_to_equity=33.6), TODAY, CONFIG.quality)
    assert [h.check for h in plain.haircuts] == ["debt_to_equity"] and plain.multiplier == pytest.approx(0.7)
    waived = quality_multiplier(healthy(debt_to_equity=33.6, negative_equity=True), TODAY, CONFIG.quality)
    assert waived.haircuts == [] and waived.multiplier == 1.0 and f"debt_to_equity: {gate_module.WAIVED_NOTE}" in waived.haircuts_skipped
    other = quality_multiplier(healthy(debt_to_equity=33.6, negative_equity=True, roe=5.0), TODAY, CONFIG.quality)
    assert [h.check for h in other.haircuts] == ["roe"]                                                                      # the other checks still apply


# ---------------------------------------------------------------- the ranking: the least negative first


def fr_bearish(edge):
    from advisor_helpers import fr
    return fr("event", "bearish", edge=edge)
