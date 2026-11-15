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


# ---------------------------------------------------------------- the news: the newer, the heavier


# ---------------------------------------------------------------- performance, historical, filings


def test_a_lower_is_better_measure_has_a_negative_span_and_debt_to_equity_is_skipped_for_negative_equity():
    low = run(metrics=metrics_of(debt_to_equity=0.0))["components"]["historical"]["parts"]
    high = run(metrics=metrics_of(debt_to_equity=3.0))["components"]["historical"]["parts"]
    assert low[0]["value"] == 0.5 and high[0]["value"] == -1.0
    waived = run(metrics=metrics_of(debt_to_equity=33.6), negative_equity=True)["components"]["historical"]
    assert waived["parts"] == [] and not waived["available"]                                                                # entry 54: not scored as a leverage failure


