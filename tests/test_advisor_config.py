"""Tests for prometheus.agents.advisor.config (the settings file is read and checked once)."""
import json
import shutil

import pytest

from prometheus.agents.advisor.config import CONFIG_PATH, FAMILIES, REGIMES, load_config


def test_the_shipped_config_loads_and_every_family_has_a_multiplier_per_regime_and_a_stop():
    config = load_config()
    assert config.config_version == "1" and config.limits.max_candidates == 15 and config.gates.min_adv_dollars == 5e6
    for family in FAMILIES:
        assert set(config.regime.multipliers[family]) >= set(REGIMES) and family in config.trade.stop_atr
    assert {c.name for c in config.quality.checks} >= {"operating_margin", "accrual_ratio", "cfo_negative_with_profit"}
    assert config.families.trend.horizon == 63 and config.sleeves["funds"].eligible == ["capital_return", "trend", "reversion", "conviction"]


def edited(tmp_path, change):
    data = json.loads(CONFIG_PATH.read_text())
    change(data)
    path = tmp_path / "advisor.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("change, message", [
    (lambda d: d["gates"].pop("min_adv_dollars"), "min_adv_dollars"),
    (lambda d: d["gates"].update(min_adv_dolars=1), "min_adv_dolars"),                       # a typo is an error, not a silent default
    (lambda d: d["families"]["trend"].update(horizon="soon"), "horizon"),
    (lambda d: d["regime"]["multipliers"]["trend"].pop("stress"), "has no value for stress"),
    (lambda d: d["trade"]["stop_atr"].pop("theme"), "stop_atr has no value for theme"),
    (lambda d: d["quality"]["checks"][0].update(op="~"), "op"),
    (lambda d: d["quality"]["checks"][0].update(factor=0), "factor"),
    (lambda d: d["sleeves"].update(weekly={"sleeve": "x", "top_k": 1, "eligible": [], "veto_only": []}), "not a case"),
    (lambda d: d["sleeves"]["funds"].update(eligible=["astrology"]), "unknown family"),
])
def test_a_bad_settings_file_is_refused_when_it_is_read(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_config(edited(tmp_path, change))


def test_a_missing_or_broken_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_config(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(tmp_path / "bad.json")
