"""Tests for prometheus.agents.advisor.blend (design choices entry 55): the score's four parts, 40% news and press releases (the newer the heavier), 30% stock performance, 20% historical
metrics, 10% filings. The numbers are literals worked by hand; they are not market data."""
import json

import pytest
from advisor_helpers import CONFIG

from prometheus.agents.advisor.config import Blend, load_config

CFG = CONFIG.blend


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
