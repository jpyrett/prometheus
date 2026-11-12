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
CR = dict(earnings_yield=0.04, debt_to_equity=0.9, net_payout_yield_ttm=0.05, share_count_change=-0.02, payout_coverage=0.6, revenue_growth=4.0, market_cap=300e9, cash_balance=10e9)


def run(family, metrics):
    return fam.evaluate(family, family_input(metrics_of(**metrics)))


# ---------------------------------------------------------------- the red-flag gate

def red_flag(**override):
    return {g.gate: g for g in gate_module.run_gates(healthy(**override), [], False, TODAY, CONFIG.gates).gates}["red_flag"]


def test_high_debt_to_equity_with_a_weak_current_ratio_is_a_red_flag_unless_the_equity_is_negative():
    flagged = red_flag(debt_to_equity=33.6, current_ratio=0.83)
    assert not flagged.passed and flagged.value == ["debt_to_equity 33.6 with current_ratio 0.83"]
    waived = red_flag(debt_to_equity=33.6, current_ratio=0.83, negative_equity=True)
    assert waived.passed and waived.value == [] and waived.note == gate_module.WAIVED_NOTE
    assert red_flag(debt_to_equity=33.6, current_ratio=0.83, negative_equity=False).passed is False               # a positive equity: the rule applies
    assert red_flag(debt_to_equity=33.6, current_ratio=0.83, negative_equity=None).passed is False                # unknown: the rule applies (an unknown is not a deficit)


def test_the_other_red_flags_still_stop_a_company_with_negative_equity_and_the_note_keeps_both():
    gate = red_flag(debt_to_equity=33.6, current_ratio=0.83, negative_equity=True, cfo_negative_with_profit=True)
    assert not gate.passed and gate.value == ["cfo_negative_with_profit"] and gate.note == "cfo_negative_with_profit; " + gate_module.WAIVED_NOTE


# ---------------------------------------------------------------- the families

def test_reversion_with_negative_equity_does_not_require_or_test_debt_to_equity():
    base = run("reversion", {**REV, "debt_to_equity": 30.0})
    assert base.computable and not base.passed and any("not sound" in n and "debt_to_equity 30.0" in n for n in base.notes) and base.waived == []
    waived = run("reversion", {**REV, "debt_to_equity": 30.0, "negative_equity": True})
    assert waived.passed and waived.waived == ["debt_to_equity"] and waived.notes[0] == gate_module.WAIVED_NOTE
    missing = run("reversion", {**{k: v for k, v in REV.items() if k != "debt_to_equity"}, "negative_equity": True})        # Domino's: Finnhub has no value at all
    assert missing.computable and missing.passed and missing.missing == []
    assert not run("reversion", {**{k: v for k, v in REV.items() if k != "debt_to_equity"}}).computable                          # without the deficit it is still missing


def test_the_waiver_does_not_waive_the_other_soundness_tests():
    weak = run("reversion", {**REV, "debt_to_equity": 30.0, "negative_equity": True, "current_ratio": 0.9})
    assert weak.computable and not weak.passed and any("not sound" in n and "debt_to_equity waived" in n and "current_ratio 0.9" in n for n in weak.notes)


def test_capital_return_with_negative_equity_does_not_require_or_test_debt_to_equity():
    assert any("debt_to_equity" in n for n in run("capital_return", {**CR, "debt_to_equity": 30.0}).notes)
    waived = run("capital_return", {**CR, "debt_to_equity": 30.0, "negative_equity": True})
    assert waived.passed and waived.direction == "bullish" and waived.waived == ["debt_to_equity"] and gate_module.WAIVED_NOTE in waived.notes
    missing = run("capital_return", {**{k: v for k, v in CR.items() if k != "debt_to_equity"}, "negative_equity": True})
    assert missing.computable and missing.passed
    assert not run("capital_return", {k: v for k, v in CR.items() if k != "debt_to_equity"}).computable
    still = run("capital_return", {**CR, "debt_to_equity": 30.0, "negative_equity": True, "net_payout_yield_ttm": 0.01, "share_count_change": 0.001})
    assert not still.passed and not any("debt_to_equity" in n for n in still.notes if n != gate_module.WAIVED_NOTE)         # its other rules still fail it


def test_a_family_that_does_not_use_debt_to_equity_records_no_waiver():
    assert run("trend", dict(mom_12_1=0.30, pct_of_52w_high=0.95, above_sma_200=True, ret_21d=0.05, rsi_14=60.0, atr_pct=0.02, negative_equity=True)).waived == []


# ---------------------------------------------------------------- the quality haircut

def test_the_debt_to_equity_haircut_is_skipped_with_the_reason_for_negative_equity():
    plain = quality_multiplier(healthy(debt_to_equity=33.6), TODAY, CONFIG.quality)
    assert [h.check for h in plain.haircuts] == ["debt_to_equity"] and plain.multiplier == pytest.approx(0.7)
    waived = quality_multiplier(healthy(debt_to_equity=33.6, negative_equity=True), TODAY, CONFIG.quality)
    assert waived.haircuts == [] and waived.multiplier == 1.0 and f"debt_to_equity: {gate_module.WAIVED_NOTE}" in waived.haircuts_skipped
    other = quality_multiplier(healthy(debt_to_equity=33.6, negative_equity=True, roe=5.0), TODAY, CONFIG.quality)
    assert [h.check for h in other.haircuts] == ["roe"]                                                                      # the other checks still apply


# ---------------------------------------------------------------- the ranking: the least negative first

def test_companies_are_ranked_by_blend_then_edge_so_a_bearish_one_is_last_and_a_waived_one_is_ranked_like_any_other():
    a = scored(symbol="AAA", families=[fr_bearish(-0.002)])
    b = scored(symbol="BBB", families=[fr_bearish(-0.008)])
    c = scored(symbol="CCC", families=[])
    order = sorted([b, c, a], key=rk.sort_key)
    assert [s.symbol for s in order] == ["CCC", "AAA", "BBB"] and all(s.score == 50 for s in order)       # with no blend all tie on the blend: no edge beats a bearish edge, and a small bearish edge beats a larger one


def fr_bearish(edge):
    from advisor_helpers import fr
    return fr("event", "bearish", edge=edge)
