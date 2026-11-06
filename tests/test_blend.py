"""Tests for prometheus.agents.advisor.blend (design choices entry 55): the score's four parts, 40% news and press releases (the newer the heavier), 30% stock performance, 20% historical
metrics, 10% filings. The numbers are literals worked by hand; they are not market data."""
import json
from datetime import date

import pytest
from advisor_helpers import CONFIG, healthy, metrics_of

from prometheus.agents.advisor import blend as bl
from prometheus.agents.advisor.config import Blend, load_config

CFG = CONFIG.blend
TODAY = date(2026, 11, 6)


def source(chunk_id, days_old):
    return {"chunk_id": chunk_id, "published_utc": f"{date.fromordinal(TODAY.toordinal() - days_old).isoformat()}T12:00:00Z"}


def signal(number, direction="bullish", chunk="finnhub:1#0", **kw):
    return {"source": number, "chunk_id": chunk, "direction": direction, "quote": "q", "verified": True, "counted": True, **kw}


def run(signals=(), events=(), sources=None, metrics=None, negative_equity=False):
    return bl.compute_blend(metrics, list(signals), list(events), sources or {}, TODAY, CFG, negative_equity)


# ---------------------------------------------------------------- the weights

def test_the_weights_are_40_30_20_10_and_must_add_up_to_one():
    assert (CFG.weights.news, CFG.weights.performance, CFG.weights.historical, CFG.weights.filings) == (0.40, 0.30, 0.20, 0.10)
    raw = json.loads(open("configs/advisor_v1.json").read())
    raw["blend"]["weights"]["filings"] = 0.2
    with pytest.raises(Exception, match="add up to 1"):
        Blend.model_validate(raw["blend"])
    assert load_config().blend.weights.news == 0.40
    raw["blend"]["half_life_dayz"] = 7
    with pytest.raises(Exception):
        Blend.model_validate(raw["blend"])                                                                                 # an unknown key is an error, not a silent default


def test_the_blend_is_the_weighted_sum_of_its_four_parts_and_the_score_is_50_plus_50_times_it():
    b = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 0)}, metrics=healthy(mom_12_1=0.5, ret_21d=0.15, pct_of_52w_high=1.0, above_sma_200=True, earnings_yield=0.08, roe=30.0, revenue_growth=15.0,
                                                                                           operating_margin=30.0, current_ratio=2.0, debt_to_equity=0.0, net_payout_yield_ttm=0.08, share_count_change=-0.02, payout_coverage=0.4))
    parts = b["components"]
    assert parts["performance"]["value"] == 1.0 and parts["filings"]["value"] == 1.0                                           # every measure at its top
    assert parts["historical"]["value"] == pytest.approx(5.5 / 6, abs=1e-4)                                                    # five at the top and a debt / equity of 0, which is +0.5 (its top needs a negative ratio)
    assert parts["news"]["value"] == pytest.approx(1 / 3, abs=1e-4)                                                                       # one fresh article: 1 / (1 + the prior 2)
    assert b["value"] == pytest.approx(0.4 / 3 + 0.3 + 0.2 * 5.5 / 6 + 0.1, abs=1e-3) and b["score"] == round(50 + 50 * b["value"])
    assert [parts[g]["contribution"] for g in parts] == [pytest.approx(0.4 / 3, abs=1e-4), 0.3, pytest.approx(0.2 * 5.5 / 6, abs=1e-4), 0.1] and b["weights"] == {"news": 0.4, "performance": 0.3, "historical": 0.2, "filings": 0.1}


def test_a_company_with_nothing_to_say_has_a_neutral_blend_and_every_part_is_marked_unavailable():
    b = run()
    assert b["value"] == 0.0 and b["score"] == 50 and all(not c["available"] and c["value"] == 0.0 and c["n"] == 0 for c in b["components"].values())


# ---------------------------------------------------------------- the news: the newer, the heavier

@pytest.mark.parametrize("days, weight", [(0, 1.0), (7, 0.5), (14, 0.25), (21, 0.125)])
def test_news_loses_half_its_weight_every_seven_days(days, weight):
    b = run(signals=[signal(1)], sources={1: source("finnhub:1#0", days)})
    item = b["components"]["news"]["parts"][0]
    assert item["age_days"] == days and item["weight"] == pytest.approx(weight) and b["components"]["news"]["value"] == pytest.approx(weight / (weight + 2), abs=1e-4)


def test_fresh_news_moves_the_blend_more_than_the_same_news_a_month_old():
    fresh = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 1)})
    old = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 30)})
    assert fresh["value"] > old["value"] > 0 and fresh["components"]["news"]["value"] > 3 * old["components"]["news"]["value"]


def test_a_newer_bearish_story_outweighs_an_older_bullish_one():
    b = run(signals=[signal(1, "bullish"), signal(2, "bearish")], sources={1: source("finnhub:1#0", 14), 2: source("finnhub:2#0", 1)})
    assert b["components"]["news"]["value"] < 0
    flipped = run(signals=[signal(1, "bullish"), signal(2, "bearish")], sources={1: source("finnhub:1#0", 1), 2: source("finnhub:2#0", 14)})
    assert flipped["components"]["news"]["value"] > 0


def test_a_press_release_weighs_half_of_an_article_of_the_same_age_and_is_labelled():
    article = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 0)})["components"]["news"]
    release = run(signals=[signal(1, chunk="pr:abc#0")], sources={1: source("pr:abc#0", 0)})["components"]["news"]
    assert release["parts"][0]["kind"] == "press_release" and release["parts"][0]["weight"] == 0.5 and article["parts"][0]["kind"] == "news"
    assert release["value"] == pytest.approx(0.5 / 2.5, abs=1e-4) and release["value"] < article["value"]


def test_several_stories_shrink_less_toward_zero_than_one():
    one = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 0)})["components"]["news"]["value"]
    five = run(signals=[signal(n) for n in range(1, 6)], sources={n: source(f"finnhub:{n}#0", 0) for n in range(1, 6)})["components"]["news"]["value"]
    assert five == pytest.approx(5 / 7, abs=1e-4) and one == pytest.approx(1 / 3, abs=1e-4) and five > one


def test_an_unverified_or_uncounted_signal_does_not_vote_and_one_source_one_direction_is_one_voice():
    sources = {1: source("finnhub:1#0", 0)}
    assert run(signals=[signal(1, verified=False), signal(1, counted=False)], sources=sources)["components"]["news"]["n"] == 0
    event = {"class": "earnings_beat", "direction": "bullish", "source": 1, "origin": "model", "published_utc": sources[1]["published_utc"]}
    assert run(signals=[signal(1)], events=[event], sources=sources)["components"]["news"]["n"] == 1                       # the same story, the same side: one voice
    other = {**event, "source": 2}
    assert run(signals=[signal(1)], events=[other], sources={**sources, 2: source("finnhub:2#0", 0)})["components"]["news"]["n"] == 2
    neutral = {**event, "direction": "neutral", "source": 3}
    assert run(events=[neutral], sources={3: source("finnhub:3#0", 0)})["components"]["news"]["n"] == 0


def test_a_news_event_votes_with_the_age_of_its_story_and_a_filing_index_event_goes_to_the_filings():
    news = {"class": "guidance_cut", "direction": "bearish", "source": 4, "origin": "model", "published_utc": source("finnhub:4#0", 7)["published_utc"]}
    b = run(events=[news], sources={4: source("finnhub:4#0", 7)})["components"]["news"]
    assert b["value"] == pytest.approx(-0.5 / 2.5, abs=1e-4) and b["parts"][0]["age_days"] == 7
    form144 = {"class": "form_144", "direction": "bearish", "source": 9, "origin": "filing_index", "published_utc": source("sec:x#0", 2)["published_utc"]}
    both = run(events=[form144], sources={9: source("sec:x#0", 2)})["components"]
    assert both["news"]["n"] == 0 and both["filings"]["value"] == -1.0 and both["filings"]["parts"] == [{"ref": "form_144", "direction": "bearish", "value": -1.0}]


def test_a_date_that_cannot_be_read_counts_as_one_half_life_old_not_as_new():
    b = run(signals=[signal(1)], sources={1: {"chunk_id": "finnhub:1#0", "published_utc": None}})["components"]["news"]["parts"][0]
    assert b["age_days"] is None and b["weight"] == 0.5
    assert bl.age_in_days("garbage", TODAY) is None and bl.age_in_days("2026-11-10T00:00:00Z", TODAY) == 0                  # tomorrow's date is not negative age


# ---------------------------------------------------------------- performance, historical, filings

def test_each_measure_is_clipped_to_minus_one_and_one_around_its_neutral_point():
    assert bl.clip(5) == 1.0 and bl.clip(-5) == -1.0
    good = run(metrics=metrics_of(mom_12_1=0.25, ret_21d=-0.15, pct_of_52w_high=0.75, above_sma_200=False))["components"]["performance"]
    assert {p["metric"]: p["value"] for p in good["parts"]} == {"mom_12_1": 0.5, "ret_21d": -1.0, "pct_of_52w_high": 0.0, "above_sma_200": -1.0}
    assert good["value"] == pytest.approx((0.5 - 1.0 + 0.0 - 1.0) / 4, abs=1e-4)


def test_a_lower_is_better_measure_has_a_negative_span_and_debt_to_equity_is_skipped_for_negative_equity():
    low = run(metrics=metrics_of(debt_to_equity=0.0))["components"]["historical"]["parts"]
    high = run(metrics=metrics_of(debt_to_equity=3.0))["components"]["historical"]["parts"]
    assert low[0]["value"] == 0.5 and high[0]["value"] == -1.0
    waived = run(metrics=metrics_of(debt_to_equity=33.6), negative_equity=True)["components"]["historical"]
    assert waived["parts"] == [] and not waived["available"]                                                                # entry 54: not scored as a leverage failure


def test_a_missing_metric_is_left_out_not_counted_as_neutral():
    b = run(metrics=metrics_of(mom_12_1=0.5, ret_21d=None, pct_of_52w_high=None))["components"]["performance"]
    assert [p["metric"] for p in b["parts"]] == ["mom_12_1"] and b["value"] == 1.0


def test_filings_count_payouts_the_share_count_dilution_and_cash_flow_and_a_good_state_of_a_warning_is_silence():
    b = run(metrics=metrics_of(net_payout_yield_ttm=0.055, share_count_change=0.02, payout_coverage=1.2, dilution_current=False, cfo_negative_with_profit=False))["components"]["filings"]
    assert {p["metric"]: p["value"] for p in b["parts"]} == {"net_payout_yield_ttm": 0.5, "share_count_change": -1.0, "payout_coverage": -1.0}     # no dilution and no cash-flow warning: no vote
    bad = run(metrics=metrics_of(dilution_current=True, cfo_negative_with_profit=True))["components"]["filings"]
    assert [p["value"] for p in bad["parts"]] == [-1.0, -1.0] and bad["value"] == -1.0


def test_verified_filing_evidence_items_vote_in_the_filings_part():
    items = [signal(None, "bullish", evidence_id="E3"), signal(None, "bearish", evidence_id="E4"), signal(None, "bullish", evidence_id="E5")]
    b = run(signals=items, metrics=metrics_of(net_payout_yield_ttm=0.08))["components"]
    assert b["news"]["n"] == 0 and b["filings"]["n"] == 4 and b["filings"]["value"] == pytest.approx((1.0 + 1 - 1 + 1) / 4, abs=1e-4)
    sec = run(signals=[signal(7, chunk="sec:0001:items#0")], sources={7: source("sec:0001:items#0", 3)})["components"]
    assert sec["news"]["n"] == 0 and sec["filings"]["n"] == 1                                                              # a quote from a filing source is a filing vote


def test_the_blend_is_json_safe_and_every_number_has_a_part_behind_it():
    b = run(signals=[signal(1)], sources={1: source("finnhub:1#0", 2)}, metrics=healthy(mom_12_1=0.2))
    assert json.loads(json.dumps(b)) == b and set(b["components"]) == {"news", "performance", "historical", "filings"}
    assert all(c["n"] == len(c["parts"]) for c in b["components"].values()) and b["half_life_days"] == 7
