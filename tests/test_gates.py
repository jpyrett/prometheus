"""Tests for prometheus.agents.advisor.gates (Advisor v1 spec section 9.1, phase 3): each gate at its threshold and one step either side."""
from datetime import date

import pytest
from advisor_helpers import CONFIG, TODAY, healthy, metric, metrics_of

from prometheus.agents.advisor.gates import run_gates, sessions_between

G = CONFIG.gates


def gates(metrics, events=(), today=TODAY, release=None):
    return run_gates(metrics, list(events), today, G, release)


def by_name(result):
    return {g.gate: g for g in result.gates}


def test_no_metrics_fails_and_nothing_else_is_run():
    for empty in (None, {}):
        result = gates(empty)
        assert result.failed == ["no_metrics"] and [g.gate for g in result.gates] == ["no_metrics"] and result.blocks("event") and result.first_failed().gate == "no_metrics"


def test_a_healthy_symbol_passes_the_four_gates_there_are():
    """Design choices entries 93 and 94 (Jacob, 2026-11-15): no_metrics, freshness, red_flag and event_risk are the Advisor's gates, and there is no other."""
    result = gates(healthy())
    assert [g.gate for g in result.gates] == ["no_metrics", "freshness", "red_flag", "event_risk"] and result.failed == []


def test_freshness_counts_weekdays_between_the_newest_bar_and_today():
    assert sessions_between(date(2026, 10, 23), date(2026, 10, 23)) == 0
    assert sessions_between(date(2026, 10, 23), date(2026, 10, 26)) == 1       # Friday's bar, Monday: the weekend is not a session
    assert sessions_between(date(2026, 10, 23), date(2026, 11, 5)) == 9
    assert by_name(gates(healthy(), today=date(2026, 10, 26)))["freshness"].passed is True       # one session behind: allowed
    stale = by_name(gates(healthy(), today=date(2026, 10, 27)))["freshness"]                     # two behind: not
    assert stale.passed is False and "2 sessions before 2026-10-27" in stale.note and stale.value["sessions_behind"] == 2


@pytest.mark.parametrize("override, flag", [
    (dict(cfo_negative_with_profit=True), "cfo_negative_with_profit"),
    (dict(debt_to_equity=4.01, current_ratio=0.99), "debt_to_equity 4.01 with current_ratio 0.99"),
])
def test_red_flags_from_the_metrics(override, flag):
    gate = by_name(gates(healthy(**override)))["red_flag"]
    assert gate.passed is False and gate.value == [flag]


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
