"""Chart patterns (Critic v1 Amendment 1, design choices entry 62): fifteen simple patterns over numbers the Data Clerk calculated, each tested at its bounds.

The numbers in these tests are INPUTS for the rules (a 50-day mean of 105 over a 200-day mean of 100), not market data. `BASE` is a chart in which no pattern holds, so a test changes the one number it is about.
"""
import pytest
from critic_helpers import CONFIG, captured, rec_of, review, tamper
from pydantic import ValidationError

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
    assert r.score == 15.0 and sorted(codes(r, "warning")) == ["chart_bearish", "death_cross", "weak_chart"]       # Critic v3: a net of -35 is also under `chart.bearish_at` (-20), a sign of the group performance
    assert "weak_chart" in codes(run(structure_63d="down", macd_hist=-1.0, last_close=80.0, sma_50=90.0, sma_200=100.0))          # -10 -10 -5: 25 is under 35
    assert "weak_chart" not in codes(run(cross_50_200_20d=-2))                                                                      # -15: exactly 35 is not under the mark
    assert "weak_chart" not in codes(run(structure_63d="down"))                                                                     # -10: 40


def test_a_reduction_is_not_a_record_so_the_chart_never_judges_one():
    """AMENDED (entry 105): this test made the real PG_open record a `reduce` and the chart was "not judged" (`applicable` False). A record whose action is not `open` cannot be built at all now (the Advisor writes
    purchases only and the Critic judges nothing else), so there is nothing for the chart to skip."""
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    with pytest.raises(ValidationError, match="the action is 'reduce': a recommendation is to buy"):
        rec_of(order)
    assert run().applicable is True


# ---------------------------------------------------------------- inside a review


def test_a_dead_cat_bounce_is_one_sign_of_performance_it_denies_only_with_a_bad_sign_in_another_group():
    """AMENDED (Critic v3, rule V5): a dead cat bounce used to deny alone ("whatever the rest says"). It is a sign of the group `performance` now: on the real PG_open record, with the Critic's own numbers made a dead cat bounce (the 50-day
    mean moved above the close), the review is APPROVED with the one reason against it named and no blocker; with the six-month average moved so that the price is 50% above it (`overpriced`, a sign of the group `price`), two groups
    are bad and the review is DENIED, capped, with the groups first in its reasons and the blockers in the closed order."""
    cat = dict(decline_from_peak_63d=-0.25, rebound_from_trough_63d=0.05, sessions_since_trough_63d=15, sma_50=500.0)
    alone = review(captured("PG_open"), changes={"PG": cat})["PG"]
    assert alone["approved"] and alone["blockers"] == [] and alone["denial_reasons"] == [] and alone["groups"] == {"news": [], "performance": ["dead_cat_bounce"], "price": []}
    assert "dead_cat_bounce" in [w["code"] for w in alone["warnings"]] and any(p["name"] == "dead_cat_bounce" for p in alone["chart"]["patterns"])
    assert "Note: one reason against it: performance (dead_cat_bounce:" in alone["summary"] and alone["score"] == alone["score_before_blockers"] > 49
    last = rec_of(captured("PG_open")).features["last_close"]["value"]
    out = review(captured("PG_open"), changes={"PG": {**cat, "sma_126": last / 1.5}})["PG"]
    assert not out["approved"] and out["denial_reasons"] == ["group:performance", "group:price", "overpriced", "dead_cat_bounce"] and out["score"] <= 49 and out["bonus"] == 0
    assert any(p["name"] == "dead_cat_bounce" for p in out["chart"]["patterns"]) and any(b["code"] == "dead_cat_bounce" and b["dimension"] == "chart" for b in out["blockers"])
    listed = [b["code"] for b in out["blockers"]]
    assert listed == sorted(listed, key=BLOCKER_ORDER.index) and listed.index("overpriced") < listed.index("dead_cat_bounce")      # the closed order: value, then chart
    assert out["summary"].startswith("DENY PG: poor performance (dead_cat_bounce:") and " and a price that is too high (overpriced:" in out["summary"]


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


def test_the_scorecard_shows_the_chart_line():
    from prometheus.agents.critic.report import scorecard
    out = review(captured("PG_open"), changes={"PG": {"cross_50_200_20d": 4}})["PG"]
    text = scorecard(out)["text"]
    chart_line = next(line for line in text.splitlines() if line.strip().startswith("chart"))
    assert "golden cross" in chart_line and f"{out['dimensions']['chart']['points']:.0f}/10" in chart_line


def test_a_reduction_is_denied_as_a_defect_it_has_no_chart_row_because_it_is_not_reviewed_at_all():
    """AMENDED (entry 105): the chart row of a reduction was "not applicable". The Critic does not review a record that is not a recommendation to buy: it is a `record_defect`, denied, with no check made."""
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    out = review(order)["PG"]
    assert not out["approved"] and [b["code"] for b in out["blockers"]] == ["record_defect"] and "the Critic does not judge a holding or a sale" in out["blockers"][0]["message"]
    assert out["chart"]["patterns"] == [] and out["action"] == "reduce"
