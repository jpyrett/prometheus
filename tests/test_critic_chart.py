"""Chart patterns (Critic v1 Amendment 1, design choices entry 62): fifteen simple patterns over numbers the Data Clerk calculated, each tested at its bounds.

The numbers in these tests are INPUTS for the rules (a 50-day mean of 105 over a 200-day mean of 100), not market data. `BASE` is a chart in which no pattern holds, so a test changes the one number it is about.
"""
import pytest
from critic_helpers import CONFIG, captured, rec_of, review, tamper

from prometheus.agents.critic import chart as ch
from prometheus.agents.critic import scoring as sc
from prometheus.agents.critic.config import PATTERNS
from prometheus.agents.critic.record import BLOCKER_ORDER

C = CONFIG.chart
BASE = dict(last_close=100.0, sma_50=100.0, sma_200=100.0, pct_of_52w_high=0.5, sma_50_vs_200=0.0, cross_50_200_20d=0, structure_63d="mixed", breakout_20d=False, volume_ratio_1d=1.0,
            macd_hist=0.0, rsi_14=40.0, obv_20d_change=0.0, decline_from_peak_63d=0.0, rebound_from_trough_63d=0.0, sessions_since_trough_63d=0)
POINTS = {p.name: p.points for p in C.patterns}


def metrics(**changes):
    return {name: {"value": value} for name, value in {**BASE, **changes}.items() if value != "missing"}


def run(**changes):
    return ch.assess(rec_of(captured("PG_open")), metrics(**changes), C)


def names(result):
    return [h.name for h in result.detail.patterns]


def codes(result, severity=None):
    return [f.code for f in result.findings if severity in (None, f.severity)]


@pytest.mark.parametrize("pattern, holds, fails", [
    ("golden_cross", dict(cross_50_200_20d=3), [dict(cross_50_200_20d=0), dict(cross_50_200_20d=-3)]),
    ("death_cross", dict(cross_50_200_20d=-1), [dict(cross_50_200_20d=0), dict(cross_50_200_20d=20)]),
    ("bull_alignment", dict(last_close=110.0, sma_50=105.0, sma_200=100.0), [dict(last_close=105.0, sma_50=105.0, sma_200=100.0), dict(last_close=110.0, sma_50=100.0, sma_200=100.0)]),
    ("bear_alignment", dict(last_close=90.0, sma_50=95.0, sma_200=100.0), [dict(last_close=95.0, sma_50=95.0, sma_200=100.0), dict(last_close=90.0, sma_50=95.0, sma_200=95.0)]),
    ("uptrend_structure", dict(structure_63d="up"), [dict(structure_63d="down"), dict(structure_63d="mixed")]),
    ("downtrend_structure", dict(structure_63d="down"), [dict(structure_63d="up"), dict(structure_63d="mixed")]),
    ("breakout_on_volume", dict(breakout_20d=True, volume_ratio_1d=1.5), [dict(breakout_20d=True, volume_ratio_1d=1.49), dict(breakout_20d=False, volume_ratio_1d=3.0)]),
    ("near_high", dict(pct_of_52w_high=0.90), [dict(pct_of_52w_high=0.899)]),
    ("macd_positive", dict(macd_hist=0.01), [dict(macd_hist=0.0), dict(macd_hist=-0.01)]),
    ("macd_negative", dict(macd_hist=-0.01), [dict(macd_hist=0.0), dict(macd_hist=0.01)]),
    ("healthy_rsi", dict(rsi_14=50.0), [dict(rsi_14=49.9), dict(rsi_14=70.1)]),
    ("overbought", dict(rsi_14=80.0), [dict(rsi_14=79.9)]),
    ("obv_rising", dict(obv_20d_change=0.01), [dict(obv_20d_change=0.0), dict(obv_20d_change=-0.01)]),
    ("obv_falling", dict(obv_20d_change=-0.01), [dict(obv_20d_change=0.0), dict(obv_20d_change=0.01)]),
])
def test_each_pattern_holds_at_its_bound_and_not_just_under_it(pattern, holds, fails):
    r = run(**holds)
    assert pattern in names(r)
    hit = next(h for h in r.detail.patterns if h.name == pattern)
    assert hit.points == POINTS[pattern] and hit.direction == PATTERNS[pattern] and hit.evidence
    for change in fails:
        assert pattern not in names(run(**change)), change


def test_every_bullish_pattern_at_once_is_capped_at_100_and_every_bearish_one_at_0():
    bull = run(cross_50_200_20d=1, last_close=120.0, sma_50=110.0, sma_200=100.0, structure_63d="up", breakout_20d=True, volume_ratio_1d=2.0, pct_of_52w_high=0.97, macd_hist=1.0, rsi_14=60.0, obv_20d_change=0.5)
    assert len(bull.detail.patterns) == 8 and bull.detail.points == 70 and bull.score == 100.0
    bear = run(cross_50_200_20d=-1, last_close=60.0, sma_50=80.0, sma_200=100.0, structure_63d="down", macd_hist=-1.0, rsi_14=85.0, obv_20d_change=-0.5,
               decline_from_peak_63d=-0.40, rebound_from_trough_63d=0.10, sessions_since_trough_63d=5)
    assert len(bear.detail.patterns) == 7 and bear.detail.points == -75 and bear.score == 0.0


# ---------------------------------------------------------------- the dead cat bounce

DEAD_CAT = dict(decline_from_peak_63d=-0.25, rebound_from_trough_63d=0.05, sessions_since_trough_63d=15, last_close=90.0, sma_50=95.0)


def test_a_dead_cat_bounce_cannot_earn_the_great_deal_bonus():
    r = run(**DEAD_CAT)
    assert not sc.great_deal_bonus_allowed(r.findings, 1.0, True, 0.7)
    assert sc.great_deal_bonus_allowed([], 1.0, True, 0.7)


# ---------------------------------------------------------------- warnings, notes and what cannot be judged


def test_a_weak_chart_is_a_warning_under_the_mark():
    r = run(cross_50_200_20d=-2, last_close=80.0, sma_50=90.0, sma_200=100.0, structure_63d="down")     # -15 - 10 - 10 = -35: 15
    assert r.score == 15.0 and sorted(codes(r, "warning")) == ["death_cross", "weak_chart"]
    assert "weak_chart" in codes(run(structure_63d="down", macd_hist=-1.0, last_close=80.0, sma_50=90.0, sma_200=100.0))          # -10 -10 -5: 25 is under 35
    assert "weak_chart" not in codes(run(cross_50_200_20d=-2))                                                                      # -15: exactly 35 is not under the mark
    assert "weak_chart" not in codes(run(structure_63d="down"))                                                                     # -10: 40


def test_a_reduction_is_not_judged():
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    r = ch.assess(rec_of(order), metrics(**DEAD_CAT), C)
    assert not r.applicable and r.findings == [] and r.measures == []


# ---------------------------------------------------------------- inside a review


def test_a_dead_cat_bounce_denies_the_recommendation_whatever_the_rest_says():
    new = {k: v for k, v in DEAD_CAT.items() if k not in ("last_close", "sma_50")}     # SNDK's own close (806) is already under its 50-day mean (1517); only the new numbers are added
    out = review(captured("SNDK"), changes={"SNDK": new})["SNDK"]
    assert not out["approved"] and "dead_cat_bounce" in out["denial_reasons"] and out["score"] <= 49 and out["bonus"] == 0 and "approved_shares" not in out
    assert any(p["name"] == "dead_cat_bounce" for p in out["chart"]["patterns"]) and any(b["code"] == "dead_cat_bounce" and b["dimension"] == "chart" for b in out["blockers"])
    listed = [b["code"] for b in out["blockers"]]
    assert listed == sorted(listed, key=BLOCKER_ORDER.index) and listed.index("falling_knife") < listed.index("dead_cat_bounce")      # the closed order: value, then chart


def test_the_scorecard_shows_the_chart_line():
    from prometheus.agents.critic.report import scorecard
    out = review(captured("PG_open"), changes={"PG": {"cross_50_200_20d": 4}})["PG"]
    text = scorecard(out)["text"]
    chart_line = next(line for line in text.splitlines() if line.strip().startswith("chart"))
    assert "golden cross" in chart_line and f"{out['dimensions']['chart']['points']:.0f}/10" in chart_line


def test_a_reduction_has_no_chart_row_score():
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    out = review(order)["PG"]
    assert out["dimensions"]["chart"]["applicable"] is False
