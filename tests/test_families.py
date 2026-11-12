"""Tests for prometheus.agents.advisor.families (Advisor v1 spec section 8, phase 4).

Each family: a pass with the score worked by hand, every rule failing on its own, `computable: false` with the missing names when the bars are short (the 17-bar case), and the
spec's named cases (the anti-chase rule, the idiosyncratic-drop rule with and without a sector ETF, the fundamental-cause refusal, the big-company haircut, the funded buyback,
the laggard rule). The numbers are literals that put a rule at its threshold; they are not market data.
"""
import pytest
from advisor_helpers import CONFIG, family_input, metrics_of

from prometheus.agents.advisor import families as fam
from prometheus.agents.advisor.record import Haircut, Quality

# ---------------------------------------------------------------- trend

TREND = dict(mom_12_1=0.30, pct_of_52w_high=0.95, above_sma_200=True, above_sma_50=True, ret_21d=0.05, rsi_14=60.0, atr_pct=0.02, residual_mom_12_1=0.05, macd_hist=0.4, adx_14=25.0, obv_20d_change=0.2,
             vol_63d_ann=0.30, bollinger_pct_b=0.9, breakout_20d=True)


def run(family, metrics, **kw):
    return fam.evaluate(family, family_input(metrics_of(**metrics), **kw))


def test_trend_passes_and_the_score_is_the_formula():
    r = run("trend", TREND)
    # 0.40 + 0.20 x clip(0.30 / 0.50) + 0.10 (macd) + 0.10 (adx >= 20) + 0.10 (obv) + 0.10 (vol < 0.35)
    assert r.computable and r.passed and r.direction == "bullish" and r.score == pytest.approx(0.40 + 0.20 * 0.6 + 0.4)
    assert r.edge_20d == pytest.approx(0.006 * 1.0 * r.score * 1.0) and r.horizon_bars == 63 and r.prior_20d == 0.006 and r.basis == "prior" and r.basis_ref == "jegadeesh_titman_1993"
    assert r.completeness == 1.0 and r.features_used["mom_12_1"] == 0.30


def test_a_missing_optional_input_adds_nothing_and_lowers_completeness():
    r = run("trend", {**TREND, "macd_hist": None, "adx_14": None})
    assert r.passed and r.score == pytest.approx(0.40 + 0.12 + 0.2) and r.completeness == pytest.approx(6 / 8, abs=1e-4)


@pytest.mark.parametrize("override, message", [
    (dict(mom_12_1=0.149), "mom_12_1"), (dict(pct_of_52w_high=0.899), "pct_of_52w_high"), (dict(above_sma_200=False), "200-day mean"),
    (dict(ret_21d=0.151, above_sma_50=False), "anti-chase"), (dict(rsi_14=75.0), "rsi_14"), (dict(residual_mom_12_1=-0.01), "residual_mom_12_1"), (dict(bollinger_pct_b=1.06), "bollinger_pct_b"),
])
def test_each_trend_rule_fails_on_its_own_and_says_which(override, message):
    r = run("trend", {**TREND, **override})
    assert r.computable and not r.passed and r.score is None and r.edge_20d is None and any(message in n for n in r.notes) and len(r.notes) == 1


def test_trend_at_the_thresholds_passes():
    assert run("trend", {**TREND, "mom_12_1": 0.15, "pct_of_52w_high": 0.90, "ret_21d": 0.15, "rsi_14": 74.99, "bollinger_pct_b": 1.05}).passed


def test_the_anti_chase_case_a_stock_that_ran_25_percent_in_a_month_is_not_a_trend_buy():
    r = run("trend", {**TREND, "ret_21d": 0.25, "mom_12_1": 0.80, "above_sma_50": False})
    assert not r.passed and "already ran" in r.notes[0]
    r = run("trend", {**TREND, "ret_21d": 0.26, "mom_12_1": 0.80})
    assert not r.passed and "over 0.25" in r.notes[0]                                    # even in an uptrend there is a limit


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


def test_trend_with_17_bars_is_not_computable_and_names_what_it_needed():
    seventeen = dict(mom_12_1=None, pct_of_52w_high=None, above_sma_200=None, ret_21d=None, rsi_14=None, atr_pct=None)
    r = run("trend", seventeen)
    assert not r.computable and not r.passed and r.score is None and r.missing == list(seventeen) and r.notes[0] == "mom_12_1: needs bars for mom_12_1"


def test_trend_is_off_in_the_recovering_regime_and_halved_when_narrow():
    assert run("trend", TREND, regime="recovering").edge_20d == 0.0
    assert run("trend", TREND, regime="narrow").edge_20d == pytest.approx(run("trend", TREND).edge_20d * 0.5)


def test_the_quality_multiplier_scales_the_edge():
    r = run("trend", TREND, quality=Quality(multiplier=0.5, haircuts=[Haircut(check="roe", value=1, factor=0.5)]))
    assert r.edge_20d == pytest.approx(run("trend", TREND).edge_20d * 0.5) and r.score == run("trend", TREND).score


# ---------------------------------------------------------------- event

EVENT = dict(atr_pct=0.02, ret_1d=0.01, volume_ratio_1d=1.0, market_cap=50e9)


def ev(klass="guidance_raise", direction="bullish", age=2, reaction=0.02, volume=2.5, source=3):
    return {"class": klass, "direction": direction, "age_bars": age, "reaction": reaction, "event_volume_ratio": volume, "source": source, "quote": "q"}


def test_a_young_bullish_event_the_price_has_not_priced_passes():
    r = run("event", EVENT, events=[ev()])
    # reaction 2% over an ATR of 2% = 1.0 ATR: 0.50 + 0.30 x (1 - 1/3) + 0.20 (volume 2.5x)
    assert r.passed and r.direction == "bullish" and r.score == pytest.approx(0.50 + 0.30 * (2 / 3) + 0.20) and r.prior_20d == 0.008 and r.evidence_ids == ["[3]"]
    assert r.edge_20d == pytest.approx(0.008 * r.score) and r.features_used["event_class"] == "guidance_raise" and r.features_used["reaction_atr"] == pytest.approx(1.0)


def test_a_bullish_event_whose_move_is_over_3_atrs_is_priced_in_and_does_not_pass():
    r = run("event", EVENT, events=[ev(reaction=0.0601)])
    assert r.computable and not r.passed and "priced in" in r.notes[0]
    assert run("event", EVENT, events=[ev(reaction=0.06)]).passed                       # exactly 3 ATRs is allowed


def test_a_bullish_event_with_no_computable_reaction_is_not_assumed_unpriced():
    r = run("event", EVENT, events=[ev(reaction=None)])
    assert not r.passed and "cannot be computed" in r.notes[0]


def test_a_bearish_event_is_always_a_bearish_vote_even_after_a_big_move():
    r = run("event", EVENT, events=[ev("guidance_cut", "bearish", reaction=-0.2)])
    assert r.passed and r.direction == "bearish" and r.edge_20d < 0 and r.edge_20d == pytest.approx(-0.008 * r.score)


def test_an_event_older_than_10_bars_does_not_count_and_the_family_says_nothing():
    r = run("event", EVENT, events=[ev(age=11)])
    assert not r.computable and "a verified bullish or bearish event" in r.missing[0] and "guidance_raise age 11" in r.notes[0]
    assert run("event", EVENT, events=[ev(age=10)]).computable                        # ten bars old still counts


def test_neutral_events_and_events_without_timing_are_not_trades():
    assert not run("event", EVENT, events=[ev("macro", "neutral")]).computable
    assert not run("event", EVENT, events=[ev(age=None)]).computable


def test_a_bullish_and_a_bearish_event_the_newer_one_votes_and_the_contradiction_is_recorded():
    r = run("event", EVENT, events=[ev("guidance_raise", "bullish", age=4), ev("earnings_miss", "bearish", age=1, reaction=-0.01)])
    assert r.direction == "bearish" and any(n.startswith("contradiction: earnings_miss (age 1) against guidance_raise (age 4)") for n in r.notes) and r.counter_evidence_ids == ["[3]"]
    tie = run("event", EVENT, events=[ev("guidance_raise", "bullish", age=2), ev("earnings_miss", "bearish", age=2, reaction=-0.01)])
    assert tie.direction == "bearish"                                                    # the same age: the cautious reading


def test_the_score_of_a_big_company_is_halved_and_the_note_says_so():
    r = run("event", {**EVENT, "market_cap": 300e9}, events=[ev()])
    assert r.passed and r.score == pytest.approx((0.50 + 0.30 * (2 / 3) + 0.20) * 0.5) and any("score halved" in n for n in r.notes)


def test_event_needs_its_required_numbers():
    r = run("event", {**EVENT, "volume_ratio_1d": None}, events=[ev()])
    assert not r.computable and r.missing == ["volume_ratio_1d"]


# ---------------------------------------------------------------- reversion

REV = dict(ret_5d=-0.09, zscore_21d=-2.5, rsi_14=28.0, atr_pct=0.02, roe=18.0, debt_to_equity=0.9, current_ratio=1.4, revenue_growth=6.0, residual_5d=-0.06, sector_ret_5d=-0.01, spy_ret_5d=0.0,
           above_sma_200=True, above_sma_50=True, bollinger_pct_b=-0.1, volume_ratio_1d=2.2, candle_reversal_1d=True)


def test_reversion_passes_for_a_sound_company_that_fell_with_no_news_of_its_own():
    r = run("reversion", REV)
    # 0.40 + 0.20 x clip((2.5 - 2) / 2) + 0.10 (rsi <= 30) + 0.10 (%B <= 0) + 0.10 (volume 2.2x) + 0.10 (reversal candle)
    assert r.passed and r.direction == "bullish" and r.score == pytest.approx(0.40 + 0.20 * 0.25 + 0.40) and r.horizon_bars == 10 and r.prior_20d == 0.010


@pytest.mark.parametrize("override, message", [
    (dict(ret_5d=-0.05, zscore_21d=-1.9, rsi_14=31.0), "not oversold"),
    (dict(roe=10.0), "not sound"), (dict(debt_to_equity=1.5), "not sound"), (dict(current_ratio=1.0), "not sound"), (dict(revenue_growth=0.0), "not sound"),
    (dict(above_sma_200=False), "below the 200-day mean"),
    (dict(residual_5d=-0.04), "not idiosyncratic"), (dict(sector_ret_5d=-0.03), "not idiosyncratic"),
])
def test_each_reversion_rule_fails_on_its_own(override, message):
    r = run("reversion", {**REV, **override})
    assert r.computable and not r.passed and any(message in n for n in r.notes)


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


def test_company_news_in_the_last_5_bars_without_a_neutral_cause_blocks_reversion():
    r = run("reversion", REV, recent_company_sources=2)
    assert not r.passed and "company-specific news in the last 5 bars (2)" in r.notes[0]


def test_an_earnings_quality_check_switches_reversion_off():
    quality = Quality(multiplier=0.5, haircuts=[Haircut(check="accrual_ratio", value=0.3, factor=0.5)], turned_off=["reversion"])
    r = run("reversion", REV, quality=quality)
    assert not r.passed and "switched off by an earnings-quality check: accrual_ratio" in r.notes[0]


def test_reversion_with_17_bars_is_not_computable():
    r = run("reversion", {**REV, "rsi_14": None, "zscore_21d": None, "atr_pct": None})
    assert not r.computable and r.missing == ["zscore_21d", "rsi_14", "atr_pct"]


# ---------------------------------------------------------------- capital_return

CR = dict(earnings_yield=0.04, debt_to_equity=0.9, net_payout_yield_ttm=0.05, share_count_change=-0.02, payout_coverage=0.6, revenue_growth=4.0, market_cap=300e9, cash_balance=10e9)


def test_capital_return_passes_and_scores_by_the_formula():
    r = run("capital_return", CR)
    # 0.40 + 0.30 x clip(0.05 / 0.08) + 0.10 (shares fell more than 1%) + 0 (no event) + 0.10 (revenue growth)
    assert r.passed and r.direction == "bullish" and r.score == pytest.approx(0.40 + 0.30 * 0.625 + 0.10 + 0.10) and r.horizon_bars == 126 and r.prior_20d == 0.003


@pytest.mark.parametrize("override, message", [
    (dict(net_payout_yield_ttm=0.0299), "net_payout_yield_ttm"), (dict(share_count_change=0.001), "the share count rose"), (dict(debt_to_equity=2.01), "debt_to_equity"),
    (dict(earnings_yield=0.0), "earnings_yield"), (dict(payout_coverage=1.0), "payout_coverage"), (dict(revenue_growth=-5.1), "revenue_growth"),
])
def test_each_capital_return_rule_fails_on_its_own(override, message):
    r = run("capital_return", {**CR, **override})
    assert r.computable and not r.passed and any(message in n for n in r.notes)


def test_without_the_payout_yield_a_shrinking_share_count_stands_in_and_completeness_is_lower():
    r = run("capital_return", {**CR, "net_payout_yield_ttm": None, "share_count_change": -0.012})
    assert r.passed and r.score == pytest.approx(0.40 + 0.15 + 0.10 + 0.10) and r.completeness < run("capital_return", CR).completeness and any("not computable" in n for n in r.notes)
    assert not run("capital_return", {**CR, "net_payout_yield_ttm": None, "share_count_change": -0.004}).passed


def test_it_needs_the_payout_yield_or_the_share_count():
    r = run("capital_return", {**CR, "net_payout_yield_ttm": None, "share_count_change": None})
    assert not r.computable and r.missing == ["share_count_change or net_payout_yield_ttm"]


def test_a_dilution_figure_does_not_vote():
    """Issuance is not a strategy rule (entry 70). A bearish dilution item leaves capital_return bullish; a rising share count is still its own failure."""
    item = {"topic": "dilution", "direction": "bearish", "verified": True, "counted": True, "evidence_id": "E2"}
    base = run("capital_return", CR)
    cited = run("capital_return", CR, signals=[item])
    assert cited.passed and cited.direction == "bullish" and cited.score == base.score and cited.edge_20d == base.edge_20d


def test_a_buyback_authorization_adds_a_tenth_and_is_halved_when_it_exceeds_the_cash():
    buyback = ev("buyback_authorization", age=5, source=4)
    buyback["quote"] = "announced a $20 billion share repurchase authorization"
    funded = run("capital_return", {**CR, "cash_balance": 25e9}, events=[buyback])
    assert funded.score == pytest.approx(0.40 + 0.30 * 0.625 + 0.10 + 0.10 + 0.10) and funded.evidence_ids == ["[4]"]
    unfunded = run("capital_return", {**CR, "cash_balance": 10e9}, events=[buyback])
    assert unfunded.score == pytest.approx(0.40 + 0.30 * 0.625 + 0.10 + 0.05 + 0.10) and any("authorization exceeds cash: $20,000,000,000 against $10,000,000,000" in n for n in unfunded.notes)
    assert run("capital_return", CR, events=[ev("buyback_authorization", age=61)]).score == funded.score - 0.10     # older than 60 bars: no term


# ---------------------------------------------------------------- theme

THEME = {"articles": 12, "positive": 8, "neutral": 3, "negative": 1, "article_share": 0.15, "basket_ret_21d": {"AAA": 0.02, "BBB": 0.10, "CCC": 0.30}}
VERIFIED = [{"verified": True, "counted": True, "direction": "bullish"}]


def theme_run(metrics=None, data=THEME, **kw):
    return run("theme", {"ret_21d": 0.02, "volume_ratio_1d": 1.5, "roe": 20.0, "operating_margin": 10.0, **(metrics or {})}, theme=data, signals=VERIFIED, **kw)


def test_a_laggard_of_a_positive_theme_passes_and_the_score_is_the_formula():
    r = theme_run()
    # median of (0.02, 0.10, 0.30) = 0.10; 0.40 + 0.20 x clip(0.15 / 0.30) + 0.20 x (1 - place 0.0) + 0 (8/12 = 67% < 80%) + 0.10 (volume 1.5x)
    assert r.passed and r.direction == "bullish" and r.score == pytest.approx(0.40 + 0.10 + 0.20 + 0.10) and r.features_used["theme_role"] == "laggard" and r.horizon_bars == 42


def test_a_company_that_already_moved_with_the_theme_is_a_leader_not_a_recommendation():
    r = theme_run({"ret_21d": 0.30})
    assert not r.passed and r.features_used["theme_role"] == "leader" and r.notes[-1] == "already moved 30% in 21 bars (the basket median is 10%)"


@pytest.mark.parametrize("data, message", [
    ({**THEME, "articles": 2}, "only 2 articles"), ({**THEME, "positive": 5, "neutral": 4, "negative": 3}, "of the articles on the theme are positive"),
])
def test_theme_needs_enough_positive_coverage(data, message):
    r = theme_run(data=data)
    assert r.computable and not r.passed and any(message in n for n in r.notes)


def test_theme_needs_the_discovery_result_and_a_verified_signal_naming_the_company():
    assert not run("theme", {"ret_21d": 0.02}, signals=VERIFIED).computable
    r = run("theme", {"ret_21d": 0.02}, theme=THEME, signals=[])
    assert not r.computable and "a verified signal naming the company" in r.missing[0]


def test_theme_with_a_poor_quality_company_does_not_pass():
    r = theme_run(quality=Quality(multiplier=0.4))
    assert not r.passed and any("quality multiplier 0.40" in n for n in r.notes)


# ---------------------------------------------------------------- the registry

def test_required_metrics_cover_every_family_and_the_event_timing():
    names = fam.required_metrics()
    assert set(names) >= set(fam.REQUIRED["trend"]) | set(fam.OPTIONAL["reversion"]) | {"share_count_change", "net_payout_yield_ttm", "recent_bars", "cash_balance"}
    assert len(names) == len(set(names))
    from prometheus.agents.data_clerk.metrics_catalog import METRIC_NAMES
    assert set(names) <= set(METRIC_NAMES), set(names) - set(METRIC_NAMES)       # the Advisor can only ask for what the catalogue has


def test_every_family_is_registered_and_the_config_has_a_section_for_each():
    assert set(fam.EVALUATORS) == set(fam.FAMILIES) == {"trend", "event", "reversion", "capital_return", "theme", "conviction"} and set(CONFIG.regime.multipliers) == set(fam.FAMILIES)


# ---------------------------------------------------------------- conviction (Advisor v1 Amendment 5)

def blend_of(value=0.40, news=0.2, performance=0.5, historical=0.4, filings=0.3, missing=()):
    """A blend as `compute_blend` writes it: the four parts, each with a value and whether it has one (`missing` names the parts with none). INPUTS for the rule, not market data."""
    values = {"news": news, "performance": performance, "historical": historical, "filings": filings}
    return {"value": value, "score": round(50 + 50 * value), "weights": {},
            "components": {n: {"weight": 0.25, "value": 0.0 if n in missing else v, "available": n not in missing, "n": 1, "contribution": 0.0, "parts": []} for n, v in values.items()}}


def conviction(blend, **metrics):
    return fam.evaluate("conviction", family_input(metrics_of(**{"rsi_14": 60.0, **metrics}), blend=blend))


def test_conviction_passes_on_a_favourable_blend_with_a_fixed_score_and_the_formula_edge():
    r = conviction(blend_of())
    assert r.computable and r.passed and r.direction == "bullish" and r.score == 0.5 and r.horizon_bars == 40 and r.basis == "prior" and r.basis_ref == "borrowed_from_trend"
    assert r.edge_20d == pytest.approx(0.006 * 1.0 * 0.5 * 1.0) and r.features_used == {"rsi_14": 60.0}                # prior x regime x score x quality
    assert "blend +0.400 (score 70) from news, performance, historical, filings" in r.notes[0]
    assert conviction(blend_of(value=0.99)).score == 0.5                                                                  # a better blend is not counted twice in the score: it scales the edge later


def test_conviction_at_its_bounds():
    assert conviction(blend_of(value=0.30)).passed and not conviction(blend_of(value=0.299)).passed
    three = blend_of(missing=("news",))                                                                                    # news has nothing to say: three parts are enough
    assert conviction(three).passed and "no news counted" in conviction(three).notes[0]
    two = conviction(blend_of(missing=("news", "filings")))
    assert not two.passed and "only 2 of 4 blend parts" in two.notes[0]
    assert conviction(blend_of(news=0.0)).passed and not conviction(blend_of(news=-0.01)).passed                           # the news part must not be negative
    assert conviction(blend_of(), rsi_14=74.99).passed and not conviction(blend_of(), rsi_14=75.0).passed                 # a blow-off is not bought
    assert conviction(blend_of(), rsi_14=None).passed                                                                      # RSI unknown: no rule


def test_conviction_needs_a_blend_and_the_regime_scales_it():
    r = fam.evaluate("conviction", family_input(metrics_of(rsi_14=60.0)))
    assert not r.computable and r.missing == ["blend"] and not r.passed
    stress = fam.evaluate("conviction", family_input(metrics_of(rsi_14=60.0), blend=blend_of(), regime="stress"))
    assert stress.passed and stress.regime_multiplier == 0.5 and stress.edge_20d == pytest.approx(0.006 * 0.5 * 0.5)
