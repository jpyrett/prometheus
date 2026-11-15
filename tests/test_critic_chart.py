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


def test_a_chart_with_no_pattern_is_neutral_and_judges_all_fifteen():
    r = run()
    assert names(r) == [] and r.score == 50.0 and r.detail.points == 0 and r.detail.unknown == [] and codes(r) == [] and len(r.measures) == 15 == len(PATTERNS)
    assert r.applicable and all(m.available for m in r.measures)


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


def test_the_score_is_the_neutral_score_plus_the_points_of_the_patterns_that_hold():
    r = run(cross_50_200_20d=2, last_close=110.0, sma_50=105.0, sma_200=100.0, macd_hist=0.2)      # a golden cross two bars ago (15), the stack (10), MACD up (5)
    assert sorted(names(r)) == ["bull_alignment", "golden_cross", "macd_positive"] and r.detail.points == 30 and r.score == 80.0
    mixed = run(rsi_14=85.0, macd_hist=0.2, obv_20d_change=0.3)                                    # overbought (-10), MACD up (5), OBV up (5)
    assert sorted(names(mixed)) == ["macd_positive", "obv_rising", "overbought"] and mixed.score == 50.0


def test_every_bullish_pattern_at_once_is_capped_at_100_and_every_bearish_one_at_0():
    bull = run(cross_50_200_20d=1, last_close=120.0, sma_50=110.0, sma_200=100.0, structure_63d="up", breakout_20d=True, volume_ratio_1d=2.0, pct_of_52w_high=0.97, macd_hist=1.0, rsi_14=60.0, obv_20d_change=0.5)
    assert len(bull.detail.patterns) == 8 and bull.detail.points == 70 and bull.score == 100.0
    bear = run(cross_50_200_20d=-1, last_close=60.0, sma_50=80.0, sma_200=100.0, structure_63d="down", macd_hist=-1.0, rsi_14=85.0, obv_20d_change=-0.5,
               decline_from_peak_63d=-0.40, rebound_from_trough_63d=0.10, sessions_since_trough_63d=5)
    assert len(bear.detail.patterns) == 7 and bear.detail.points == -75 and bear.score == 0.0


# ---------------------------------------------------------------- the dead cat bounce

DEAD_CAT = dict(decline_from_peak_63d=-0.25, rebound_from_trough_63d=0.05, sessions_since_trough_63d=15, last_close=90.0, sma_50=95.0)


def test_a_dead_cat_bounce_is_a_blocker_and_each_of_its_five_conditions_is_needed():
    r = run(**DEAD_CAT)
    assert "dead_cat_bounce" in names(r) and codes(r, "blocker") == ["dead_cat_bounce"]
    assert "fell 25% from its peak" in r.findings[0].message and "bounced 5%" in r.findings[0].message and "under its 50-day mean" in r.findings[0].message
    for change in (dict(decline_from_peak_63d=-0.249),                      # it did not fall far enough
                   dict(rebound_from_trough_63d=0.049),                     # no bounce yet
                   dict(sessions_since_trough_63d=16),                      # the low is too old: it is a recovery, not a bounce
                   dict(decline_from_peak_63d=-0.25, rebound_from_trough_63d=0.25),   # (1 - 0.25) x (1 + 0.25) - 1 = -6.25%: nearly back at the peak
                   dict(last_close=95.0)):                                  # back at its 50-day mean
        assert "dead_cat_bounce" not in names(run(**{**DEAD_CAT, **change})), change


def test_the_distance_to_the_peak_is_worked_out_from_the_fall_and_the_bounce():
    on_the_line = run(**{**DEAD_CAT, "decline_from_peak_63d": -0.30, "rebound_from_trough_63d": 0.5 / 3})  # 0.70 x (1 + 1/6) - 1 = -18.33%
    assert "dead_cat_bounce" in names(on_the_line)
    exact = run(**{**DEAD_CAT, "decline_from_peak_63d": -0.5, "rebound_from_trough_63d": 0.7})            # 0.50 x 1.70 - 1 = -15%: the limit itself
    assert "dead_cat_bounce" in names(exact)


def test_a_dead_cat_bounce_cannot_earn_the_great_deal_bonus():
    r = run(**DEAD_CAT)
    assert not sc.great_deal_bonus_allowed(r.findings, 1.0, True, 0.7)
    assert sc.great_deal_bonus_allowed([], 1.0, True, 0.7)


# ---------------------------------------------------------------- warnings, notes and what cannot be judged

def test_a_death_cross_is_a_warning_and_a_golden_cross_a_note():
    dead = run(cross_50_200_20d=-4)
    assert codes(dead, "warning") == ["death_cross"] and codes(dead, "blocker") == [] and "3 sessions ago" in dead.findings[0].message
    gold = run(cross_50_200_20d=1)
    assert codes(gold, "info") == ["golden_cross"] and "today" in gold.findings[0].message


def test_a_weak_chart_is_a_warning_under_the_mark():
    r = run(cross_50_200_20d=-2, last_close=80.0, sma_50=90.0, sma_200=100.0, structure_63d="down")     # -15 - 10 - 10 = -35: 15
    assert r.score == 15.0 and sorted(codes(r, "warning")) == ["death_cross", "weak_chart"]
    assert "weak_chart" in codes(run(structure_63d="down", macd_hist=-1.0, last_close=80.0, sma_50=90.0, sma_200=100.0))          # -10 -10 -5: 25 is under 35
    assert "weak_chart" not in codes(run(cross_50_200_20d=-2))                                                                      # -15: exactly 35 is not under the mark
    assert "weak_chart" not in codes(run(structure_63d="down"))                                                                     # -10: 40


def test_a_pattern_whose_number_is_missing_is_unknown_and_never_guessed():
    r = run(sma_200="missing", cross_50_200_20d="missing")
    assert set(r.detail.unknown) == {"golden_cross", "death_cross", "bull_alignment", "bear_alignment"} and r.score == 50.0
    assert [m.available for m in r.measures if m.name == "bull_alignment"] == [False]
    no_stack = run(sma_50="missing", **{k: v for k, v in DEAD_CAT.items() if k != "sma_50"})
    assert "dead_cat_bounce" in no_stack.detail.unknown and codes(no_stack, "blocker") == []        # no 50-day mean: it cannot be a blocker
    no_volume = run(breakout_20d=True, volume_ratio_1d="missing")
    assert "breakout_on_volume" in no_volume.detail.unknown
    assert "breakout_on_volume" not in run(breakout_20d=False, volume_ratio_1d="missing").detail.unknown      # no breakout is a no, whatever the volume


def test_with_nothing_to_judge_the_chart_says_so_and_stays_neutral():
    r = ch.assess(rec_of(captured("PG_open")), {}, C)
    assert r.score == 50.0 and codes(r, "warning") == ["chart_not_assessable"] and len(r.detail.unknown) == 15 and r.detail.facts == {}
    assert ch.assess(rec_of(captured("PG_open")), None, C).score == 50.0


def test_a_reduction_is_not_judged():
    order = captured("PG_open")
    order.recommendations[0] = tamper(order.recommendations[0], **{"action.action": "reduce"})
    r = ch.assess(rec_of(order), metrics(**DEAD_CAT), C)
    assert not r.applicable and r.findings == [] and r.measures == []


# ---------------------------------------------------------------- inside a review

def test_a_review_carries_the_chart_dimension_and_the_patterns_behind_it():
    out = review(captured("PG_open"), changes={"PG": {"cross_50_200_20d": 4, "structure_63d": "up"}})["PG"]
    d = out["dimensions"]["chart"]
    assert d["weight"] == 10 and d["applicable"] and set(out["dimensions"]) == {"accuracy", "evidence", "risk_fit", "timing", "value", "chart"}
    names_held = {p["name"] for p in out["chart"]["patterns"]}
    assert {"golden_cross", "uptrend_structure"} <= names_held
    assert d["score"] == pytest.approx(min(100, max(0, 50 + out["chart"]["points"]))) and d["points"] == pytest.approx(d["score"] / 10)
    assert any(n["code"] == "golden_cross" for n in out["notes"])


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
