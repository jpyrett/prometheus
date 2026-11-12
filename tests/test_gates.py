"""Tests for prometheus.agents.advisor.gates (Advisor v1 spec section 9.1, phase 3): each gate at its threshold and one step either side."""
from datetime import date

import pytest
from advisor_helpers import CONFIG, TODAY, healthy, metric, metrics_of

from prometheus.agents.advisor.gates import run_gates, sessions_between

G = CONFIG.gates


def gates(metrics, events=(), held=False, today=TODAY, release=None):
    return run_gates(metrics, list(events), held, today, G, release)


def by_name(result):
    return {g.gate: g for g in result.gates}


def test_a_healthy_symbol_passes_every_gate():
    result = gates(healthy())
    assert result.failed == [] and [g.gate for g in result.gates] == ["no_metrics", "liquidity", "freshness", "red_flag", "event_risk"] and not result.blocks("trend")


def test_no_metrics_fails_and_nothing_else_is_run():
    for empty in (None, {}):
        result = gates(empty)
        assert result.failed == ["no_metrics"] and [g.gate for g in result.gates] == ["no_metrics"] and result.blocks("event") and result.first_failed().gate == "no_metrics"


@pytest.mark.parametrize("adv, last, spread, passes", [
    (5_000_000, 5.0, None, True),                # exactly the limits pass
    (4_999_999, 100.0, None, False), (50e6, 4.99, None, False),
    (50e6, 100.0, 20, True), (50e6, 100.0, 20.5, False),
])
def test_liquidity_at_the_thresholds(adv, last, spread, passes):
    metrics = healthy(adv_20d_dollars=adv, last_close=last)
    if spread is not None:
        metrics["median_spread_bps_1d"] = metric(spread)
    gate = by_name(gates(metrics))["liquidity"]
    assert gate.passed is passes and (gate.note is None) == passes


def test_liquidity_that_cannot_be_computed_fails_and_names_the_missing_input():
    gate = by_name(gates(healthy(adv_20d_dollars=None)))["liquidity"]
    assert gate.passed is False and gate.note == "not computable: needs bars for adv_20d_dollars" and gate.value is None
    assert by_name(gates(healthy(last_close=None)))["liquidity"].note.startswith("not computable")


def test_a_holding_is_not_tested_for_liquidity():
    gate = by_name(gates(healthy(adv_20d_dollars=1.0), held=True))["liquidity"]
    assert gate.passed is True and gate.note == "not applied to a holding"


def test_freshness_counts_weekdays_between_the_newest_bar_and_today():
    assert sessions_between(date(2026, 10, 23), date(2026, 10, 23)) == 0
    assert sessions_between(date(2026, 10, 23), date(2026, 10, 26)) == 1       # Friday's bar, Monday: the weekend is not a session
    assert sessions_between(date(2026, 10, 23), date(2026, 11, 5)) == 9
    assert by_name(gates(healthy(), today=date(2026, 10, 26)))["freshness"].passed is True       # one session behind: allowed
    stale = by_name(gates(healthy(), today=date(2026, 10, 27)))["freshness"]                     # two behind: not
    assert stale.passed is False and "2 sessions before 2026-10-27" in stale.note and stale.value["sessions_behind"] == 2


def test_freshness_also_checks_the_age_of_the_finnhub_figures():
    metrics = healthy()
    metrics["market_cap"] = metric(5e11, as_of="2026-07-10")
    gate = by_name(gates(metrics))["freshness"]
    assert gate.passed is False and "Finnhub figures are from 2026-07-10" in gate.note
    metrics["market_cap"] = metric(5e11, as_of="2026-07-15")        # 100 days exactly: allowed
    assert by_name(gates(metrics))["freshness"].passed is True


def test_freshness_without_bars_fails():
    metrics = healthy(last_close=None)
    assert by_name(gates(metrics))["freshness"].passed is False and by_name(gates(metrics))["freshness"].note.startswith("no bars")


@pytest.mark.parametrize("override, flag", [
    (dict(cfo_negative_with_profit=True), "cfo_negative_with_profit"),
    (dict(debt_to_equity=4.01, current_ratio=0.99), "debt_to_equity 4.01 with current_ratio 0.99"),
])
def test_red_flags_from_the_metrics(override, flag):
    gate = by_name(gates(healthy(**override)))["red_flag"]
    assert gate.passed is False and gate.value == [flag]


@pytest.mark.parametrize("override", [dict(debt_to_equity=4.0, current_ratio=0.99), dict(debt_to_equity=4.5, current_ratio=1.0), dict(debt_to_equity=None), dict(cfo_negative_with_profit=None)])
def test_a_red_flag_needs_its_whole_condition_and_a_missing_value_is_no_flag(override):
    assert by_name(gates(healthy(**override)))["red_flag"].passed is True


def test_red_flag_events_count_for_60_bars_and_no_longer():
    recent = [{"class": "going_concern", "age_bars": 60}, {"class": "restatement", "age_bars": 61}, {"class": "guidance_cut", "age_bars": 1}, {"class": "material_weakness", "age_bars": None}]
    gate = by_name(gates(healthy(), events=recent))["red_flag"]
    assert gate.value == ["going_concern 60 bars ago"]


def test_a_held_symbol_that_fails_a_gate_is_still_reported_for_the_holdings_pass():
    result = gates(healthy(cfo_negative_with_profit=True), held=True)
    assert result.failed == ["red_flag"] and result.first_failed().value == ["cfo_negative_with_profit"]


def test_event_risk_from_the_calendar_and_from_the_last_results_release():
    near = by_name(gates(healthy(days_to_next_earnings=3)))["event_risk"]
    assert near.passed is False and by_name(gates(healthy(days_to_next_earnings=4)))["event_risk"].passed is True
    for release, days, passes in (("2026-07-31", 84, True), ("2026-07-30", 85, False), ("2026-07-23", 92, False), ("2026-07-20", 95, False), ("2026-07-19", 96, True)):
        gate = by_name(gates(healthy(), release=release))["event_risk"]
        assert gate.passed is passes and gate.value == days, (release, days)
    estimated = by_name(gates(healthy(), release="2026-07-30"))["event_risk"]
    assert "estimated from the last earnings release (2026-07-30, 85 days ago)" in estimated.note      # the note says it is an estimate
    unknown = by_name(gates(healthy()))["event_risk"]
    assert unknown.passed is True and "no earnings date known" in unknown.note


def test_event_risk_blocks_every_family_but_event():
    result = gates(healthy(days_to_next_earnings=2))
    assert result.failed == ["event_risk"] and result.blocks("trend") and result.blocks("capital_return") and not result.blocks("event")
    both = gates(healthy(days_to_next_earnings=2, cfo_negative_with_profit=True))
    assert both.blocks("event")                                                                          # another gate failed too: nothing may open
