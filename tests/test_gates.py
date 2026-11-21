"""Tests for prometheus.agents.advisor.gates (Advisor v1 spec section 9.1, phase 3): each gate at its threshold and one step either side."""
from datetime import date

from advisor_helpers import CONFIG, TODAY, healthy

from prometheus.agents.advisor.gates import run_gates

G = CONFIG.gates


def gates(metrics, events=(), today=TODAY, release=None):
    return run_gates(metrics, list(events), today, G, release)


def by_name(result):
    return {g.gate: g for g in result.gates}


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


