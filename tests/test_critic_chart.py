"""Chart patterns (Critic v1 Amendment 1, design choices entry 62): fifteen simple patterns over numbers the Data Clerk calculated, each tested at its bounds.

The numbers in these tests are INPUTS for the rules (a 50-day mean of 105 over a 200-day mean of 100), not market data. `BASE` is a chart in which no pattern holds, so a test changes the one number it is about.
"""
from critic_helpers import CONFIG, captured, rec_of, review

from prometheus.agents.critic import chart as ch

C = CONFIG.chart
BASE = dict(last_close=100.0, sma_50=100.0, sma_200=100.0, pct_of_52w_high=0.5, sma_50_vs_200=0.0, cross_50_200_20d=0, structure_63d="mixed", breakout_20d=False, volume_ratio_1d=1.0,
            macd_hist=0.0, rsi_14=40.0, obv_20d_change=0.0, decline_from_peak_63d=0.0, rebound_from_trough_63d=0.0, sessions_since_trough_63d=0)


def metrics(**changes):
    return {name: {"value": value} for name, value in {**BASE, **changes}.items() if value != "missing"}


def run(**changes):
    return ch.assess(rec_of(captured("PG_open")), metrics(**changes), C)


def test_every_bullish_pattern_at_once_is_capped_at_100_and_every_bearish_one_at_0():
    bull = run(cross_50_200_20d=1, last_close=120.0, sma_50=110.0, sma_200=100.0, structure_63d="up", breakout_20d=True, volume_ratio_1d=2.0, pct_of_52w_high=0.97, macd_hist=1.0, rsi_14=60.0, obv_20d_change=0.5)
    assert len(bull.detail.patterns) == 8 and bull.detail.points == 70 and bull.score == 100.0
    bear = run(cross_50_200_20d=-1, last_close=60.0, sma_50=80.0, sma_200=100.0, structure_63d="down", macd_hist=-1.0, rsi_14=85.0, obv_20d_change=-0.5,
               decline_from_peak_63d=-0.40, rebound_from_trough_63d=0.10, sessions_since_trough_63d=5)
    assert len(bear.detail.patterns) == 7 and bear.detail.points == -75 and bear.score == 0.0


# ---------------------------------------------------------------- inside a review


def test_poor_performance_is_a_63_session_return_at_or_under_minus_15_percent_with_a_falling_21_session_return():
    """Critic v3 (spec V4): the Critic's own re-asked returns, never the record's. A WARNING (a sign of the group performance): at the bound it fires, just above it or with a 21-session return that is not negative it does not, and a
    return that could not be calculated is not judged. Inside a whole review of the real PG_open record it is the one bad group: approved, with the reason named."""
    rec = rec_of(captured("PG_open"))
    P = CONFIG.performance
    assert P.poor_ret_63d == -0.15
    [f] = ch.poor_performance(rec, -0.15, -0.0001, P)
    assert (f.code, f.severity, f.dimension) == ("poor_performance", "warning", "chart") and "down 15.0% over 63 sessions" in f.message and f.value == [-0.15, -0.0001] and f.threshold == -0.15
    assert ch.poor_performance(rec, -0.1499, -0.05, P) == [] and ch.poor_performance(rec, -0.30, 0.0, P) == [] and ch.poor_performance(rec, -0.30, 0.02, P) == []
    assert ch.poor_performance(rec, None, -0.05, P) == [] and ch.poor_performance(rec, -0.30, None, P) == []
    r = review(captured("PG_open"), own={}, changes={"PG": {"ret_21d": -0.02, "ret_63d": -0.20}})["PG"]
    assert r["approved"] and r["groups"]["performance"] == ["poor_performance"] and "Note: one reason against it: performance (poor_performance: PG is down 20.0% over 63 sessions" in r["summary"]
    assert "poor_performance" in [w["code"] for w in r["warnings"]] and r["blockers"] == []
    plain = review(captured("PG_open"), own={}, changes={"PG": {"ret_21d": -0.02, "ret_63d": -0.10}})["PG"]
    assert plain["groups"]["performance"] == []
