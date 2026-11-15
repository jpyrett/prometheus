"""Tests for prometheus.agents.advisor.config (the settings file is read and checked once)."""
import json
import shutil

import pytest

from prometheus.agents.advisor.config import CONFIG_PATH, FAMILIES, REGIMES, load_config


def edited(tmp_path, change):
    data = json.loads(CONFIG_PATH.read_text())
    change(data)
    path = tmp_path / "advisor.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("change, message", [
    (lambda d: d["gates"].pop("max_bar_age_sessions"), "max_bar_age_sessions"),
    (lambda d: d["gates"].update(max_bar_age_sesions=1), "max_bar_age_sesions"),                       # a typo is an error, not a silent default
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
