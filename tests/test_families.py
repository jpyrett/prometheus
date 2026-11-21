"""Tests for prometheus.agents.advisor.families (Advisor v1 spec section 8, phase 4).

Each family: a pass with the score worked by hand, every rule failing on its own, `computable: false` with the missing names when the bars are short (the 17-bar case), and the
spec's named cases (the anti-chase rule, the idiosyncratic-drop rule with and without a sector ETF, the fundamental-cause refusal, the big-company haircut, the funded buyback,
the laggard rule). The numbers are literals that put a rule at its threshold; they are not market data.
"""
import pytest
from advisor_helpers import family_input, metrics_of

from prometheus.agents.advisor import families as fam
from prometheus.agents.advisor.record import Quality

# ---------------------------------------------------------------- trend

TREND = dict(mom_12_1=0.30, pct_of_52w_high=0.95, above_sma_200=True, above_sma_50=True, ret_21d=0.05, rsi_14=60.0, atr_pct=0.02, residual_mom_12_1=0.05, macd_hist=0.4, adx_14=25.0, obv_20d_change=0.2,
             vol_63d_ann=0.30, bollinger_pct_b=0.9, breakout_20d=True)


def run(family, metrics, **kw):
    return fam.evaluate(family, family_input(metrics_of(**metrics), **kw))


def test_the_breakout_path_for_a_strong_runner_above_its_50_and_200_day_means():
    """Amendment 4: Moderna on 2026-11-11 (12-1 momentum 757%, 88.6% of its high, up 15.001% in a month, RSI 58.8, above both means) is a trend candidate; a weaker case is not."""
    moderna = run("trend", {**TREND, "mom_12_1": 7.57, "pct_of_52w_high": 0.886, "ret_21d": 0.15001, "rsi_14": 58.8})
    assert moderna.passed and moderna.direction == "bullish" and any("breakout path" in n for n in moderna.notes) and any("uptrend" in n for n in moderna.notes)
    assert run("trend", {**TREND, "mom_12_1": 0.50, "pct_of_52w_high": 0.80, "ret_21d": 0.25}).passed                   # at the limits
    assert not run("trend", {**TREND, "mom_12_1": 0.50, "pct_of_52w_high": 0.799}).passed                             # too far under the high
    assert not run("trend", {**TREND, "mom_12_1": 0.49, "pct_of_52w_high": 0.85}).passed                              # not strong enough to be a runner
    assert not run("trend", {**TREND, "mom_12_1": 0.80, "pct_of_52w_high": 0.85, "above_sma_50": False}).passed       # under its 50-day mean: no breakout path
    assert not run("trend", {**TREND, "mom_12_1": 0.80, "pct_of_52w_high": 0.85, "above_sma_50": None}).passed        # unknown 50-day mean: the old rules
    assert not run("trend", {**TREND, "mom_12_1": 7.57, "pct_of_52w_high": 0.886, "rsi_14": 75.0}).passed             # a blow-off is still stopped by the RSI
    assert run("trend", TREND).notes == []                                                                          # an ordinary pass has no relaxed rule to note


# ---------------------------------------------------------------- event

EVENT = dict(atr_pct=0.02, ret_1d=0.01, volume_ratio_1d=1.0)


def ev(klass="guidance_raise", direction="bullish", age=2, reaction=0.02, volume=2.5, source=3):
    return {"class": klass, "direction": direction, "age_bars": age, "reaction": reaction, "event_volume_ratio": volume, "source": source, "quote": "q"}


def test_a_bullish_event_whose_move_is_over_3_atrs_is_priced_in_and_does_not_pass():
    r = run("event", EVENT, events=[ev(reaction=0.0601)])
    assert r.computable and not r.passed and "priced in" in r.notes[0]
    assert run("event", EVENT, events=[ev(reaction=0.06)]).passed                       # exactly 3 ATRs is allowed


def test_a_bullish_and_a_bearish_event_the_newer_one_votes_and_the_contradiction_is_recorded():
    r = run("event", EVENT, events=[ev("guidance_raise", "bullish", age=4), ev("earnings_miss", "bearish", age=1, reaction=-0.01)])
    assert r.direction == "bearish" and any(n.startswith("contradiction: earnings_miss (age 1) against guidance_raise (age 4)") for n in r.notes) and r.counter_evidence_ids == ["[3]"]
    tie = run("event", EVENT, events=[ev("guidance_raise", "bullish", age=2), ev("earnings_miss", "bearish", age=2, reaction=-0.01)])
    assert tie.direction == "bearish"                                                    # the same age: the cautious reading


def test_event_needs_its_required_numbers():
    r = run("event", {**EVENT, "volume_ratio_1d": None}, events=[ev()])
    assert not r.computable and r.missing == ["volume_ratio_1d"]


# ---------------------------------------------------------------- reversion

REV = dict(ret_5d=-0.09, zscore_21d=-2.5, rsi_14=28.0, atr_pct=0.02, roe=18.0, debt_to_equity=0.9, current_ratio=1.4, revenue_growth=6.0, residual_5d=-0.06, sector_ret_5d=-0.01, spy_ret_5d=0.0,
           above_sma_200=True, above_sma_50=True, bollinger_pct_b=-0.1, volume_ratio_1d=2.2, candle_reversal_1d=True)


def test_the_idiosyncratic_rule_with_a_sector_etf_and_without_one():
    with_sector = run("reversion", {**REV, "residual_5d": -0.02, "sector_ret_5d": -0.06})            # the whole sector fell: not this company's fall
    assert not with_sector.passed and "not idiosyncratic: residual_5d -0.020" in with_sector.notes[0]
    without = {**REV, "residual_5d": None, "sector_ret_5d": None}
    assert run("reversion", without).passed                                                          # falls back to the stock down 7% with SPY flat
    assert "not idiosyncratic: ret_5d" in run("reversion", {**without, "ret_5d": -0.05, "rsi_14": 25.0}, ).notes[0]
    assert not run("reversion", {**without, "spy_ret_5d": -0.04}).passed
    assert "spy_ret_5d is missing" in run("reversion", {**without, "spy_ret_5d": None}).notes[0]


def test_the_trend_filter_falls_back_to_the_50_day_mean_and_is_skipped_with_a_note_without_either():
    assert not run("reversion", {**REV, "above_sma_200": None, "above_sma_50": False}).passed
    assert run("reversion", {**REV, "above_sma_200": None, "above_sma_50": True}).passed
    skipped = run("reversion", {**REV, "above_sma_200": None, "above_sma_50": None})
    assert skipped.passed and any(n.startswith("trend filter skipped:") for n in skipped.notes)


def test_a_fundamental_cause_is_refused_and_a_neutral_one_is_noted():
    for cause in ("guidance_cut", "dilution_offering", "restatement", "going_concern", "material_weakness", "earnings_miss"):
        r = run("reversion", REV, drop_cause={"class": cause, "source": 2})
        assert not r.passed and f"fundamental cause: {cause}" in r.notes
    neutral = run("reversion", REV, drop_cause={"class": "sector_sympathy", "source": 2}, recent_company_sources=3)
    assert neutral.passed and "cause: sector_sympathy" in neutral.notes and neutral.evidence_ids == ["[2]"]
    recent_miss = run("reversion", REV, events=[ev("earnings_miss", "bearish", age=3)])
    assert not recent_miss.passed and "fundamental cause: earnings_miss" in recent_miss.notes


# ---------------------------------------------------------------- capital_return

CR = dict(earnings_yield=0.04, debt_to_equity=0.9, share_count_change=-0.02, payout_coverage=0.6, revenue_growth=4.0, cash_balance=10e9)


@pytest.mark.parametrize("override, message", [
    (dict(share_count_change=-0.004), "share_count_change"), (dict(share_count_change=0.001), "the share count rose"), (dict(debt_to_equity=2.01), "debt_to_equity"),
    (dict(earnings_yield=0.0), "earnings_yield"), (dict(payout_coverage=1.0), "payout_coverage"), (dict(revenue_growth=-5.1), "revenue_growth"),
])
def test_each_capital_return_rule_fails_on_its_own(override, message):
    r = run("capital_return", {**CR, **override})
    assert r.computable and not r.passed and any(message in n for n in r.notes)


def test_a_buyback_authorization_adds_a_tenth_and_is_halved_when_it_exceeds_the_cash():
    buyback = ev("buyback_authorization", age=5, source=4)
    buyback["quote"] = "announced a $20 billion share repurchase authorization"
    funded = run("capital_return", {**CR, "cash_balance": 25e9}, events=[buyback])
    assert funded.score == pytest.approx(0.40 + 0.15 + 0.10 + 0.10 + 0.10) and funded.evidence_ids == ["[4]"]
    unfunded = run("capital_return", {**CR, "cash_balance": 10e9}, events=[buyback])
    assert unfunded.score == pytest.approx(0.40 + 0.15 + 0.10 + 0.05 + 0.10) and any("authorization exceeds cash: $20,000,000,000 against $10,000,000,000" in n for n in unfunded.notes)
    assert run("capital_return", CR, events=[ev("buyback_authorization", age=61)]).score == funded.score - 0.10     # older than 60 bars: no term


# ---------------------------------------------------------------- theme

THEME = {"articles": 12, "positive": 8, "neutral": 3, "negative": 1, "article_share": 0.15, "basket_ret_21d": {"AAA": 0.02, "BBB": 0.10, "CCC": 0.30}}
VERIFIED = [{"verified": True, "counted": True, "direction": "bullish"}]


def theme_run(metrics=None, data=THEME, **kw):
    return run("theme", {"ret_21d": 0.02, "volume_ratio_1d": 1.5, "roe": 20.0, "operating_margin": 10.0, **(metrics or {})}, theme=data, signals=VERIFIED, **kw)


def test_theme_needs_the_discovery_result_and_a_verified_signal_naming_the_company():
    assert not run("theme", {"ret_21d": 0.02}, signals=VERIFIED).computable
    r = run("theme", {"ret_21d": 0.02}, theme=THEME, signals=[])
    assert not r.computable and "a verified signal naming the company" in r.missing[0]


def test_theme_with_a_poor_quality_company_does_not_pass():
    r = theme_run(quality=Quality(multiplier=0.4))
    assert not r.passed and any("quality multiplier 0.40" in n for n in r.notes)


# ---------------------------------------------------------------- conviction (Advisor v1 Amendment 5)

def blend_of(value=0.40, news=0.2, performance=0.5, historical=0.4, filings=0.3, missing=()):
    """A blend as `compute_blend` writes it: the four parts, each with a value and whether it has one (`missing` names the parts with none). INPUTS for the rule, not market data."""
    values = {"news": news, "performance": performance, "historical": historical, "filings": filings}
    return {"value": value, "score": round(50 + 50 * value), "weights": {},
            "components": {n: {"weight": 0.25, "value": 0.0 if n in missing else v, "available": n not in missing, "n": 1, "contribution": 0.0, "parts": []} for n, v in values.items()}}


def conviction(blend, **metrics):
    return fam.evaluate("conviction", family_input(metrics_of(**{"rsi_14": 60.0, **metrics}), blend=blend))


def test_conviction_needs_a_blend_and_the_regime_scales_it():
    r = fam.evaluate("conviction", family_input(metrics_of(rsi_14=60.0)))
    assert not r.computable and r.missing == ["blend"] and not r.passed
    stress = fam.evaluate("conviction", family_input(metrics_of(rsi_14=60.0), blend=blend_of(), regime="stress"))
    assert stress.passed and stress.regime_multiplier == 0.5 and stress.edge_20d == pytest.approx(0.006 * 0.5 * 0.5)
