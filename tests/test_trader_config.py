"""The Trader's settings file (configs/trader_v1.json): read once, checked completely (Trader spec section 15), and the PAPER_TRADING switch (rule T4)."""
import copy
import json

import pytest

from prometheus.agents.trader import config as tc
from prometheus.rag.config import REPO_ROOT

FILE = REPO_ROOT / "configs" / "trader_v1.json"


def raw() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8"))


def write(tmp_path, data) -> str:
    path = tmp_path / "trader.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def edited(*path_and_value):
    """The shipped settings with one value changed: edited("orders", "max_chase_pct", 0.5)."""
    *path, value = path_and_value
    data = copy.deepcopy(raw())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


@pytest.mark.parametrize("section, key", [(s, k) for s, v in raw().items() if isinstance(v, dict) for k in v])
def test_a_missing_key_in_any_section_is_an_error(tmp_path, section, key):
    data = copy.deepcopy(raw())
    del data[section][key]
    with pytest.raises(ValueError, match=key):
        tc.load_config(write(tmp_path, data))


@pytest.mark.parametrize("where", [(), ("limits",), ("orders",), ("costs",), ("funding",), ("backtest",), ("ledger",)])
def test_an_unknown_key_is_an_error_and_not_ignored(tmp_path, where):
    data = copy.deepcopy(raw())
    node = data
    for key in where:
        node = node[key]
    node["slippage"] = 1
    with pytest.raises(ValueError, match="slippage"):
        tc.load_config(write(tmp_path, data))


@pytest.mark.parametrize("keys, value, message", [
    (("mode",), "live", "live trading is refused"),
    (("mode",), "Paper", "paper or backtest"),
    (("mode",), "", "paper or backtest"),
    (("mode",), None, "mode"),
    (("orders", "cash_policy"), "reject", "cash_policy must be 'reduce'"),
    (("orders", "cash_policy"), "borrow", "cash_policy must be 'reduce'"),
    (("backtest", "fill_at"), "same_close", "fill_at must be 'next_open'"),
    (("backtest", "fill_at"), "open", "fill_at must be 'next_open'"),
    (("backtest", "benchmark"), "../etc", "not a valid ticker symbol"),
    (("backtest", "benchmark"), "", "not a valid ticker symbol"),
    (("backtest", "benchmark"), 5, "benchmark"),
    (("backtest", "max_follow_days"), 0, "max_follow_days"),
    (("backtest", "max_follow_days"), 401, "max_follow_days"),
    (("backtest", "max_follow_days"), 1.5, "max_follow_days"),
    (("limits", "max_trades"), 0, "max_trades"),
    (("limits", "max_trades"), True, "max_trades"),
    (("limits", "base_turns"), 0, "base_turns"),
    (("limits", "max_turns"), 4, "max_turns must be at least limits.base_turns"),
    (("limits", "model_timeout_seconds"), 0, "model_timeout_seconds"),
    (("limits", "model_timeout_seconds"), -1, "model_timeout_seconds"),
    (("limits", "max_note_words"), 0, "max_note_words"),
    (("limits", "max_summary_words"), 0, "max_summary_words"),
    (("orders", "max_chase_pct"), -0.01, "max_chase_pct"),
    (("orders", "max_chase_pct"), 1, "max_chase_pct"),
    (("orders", "limit_band_pct"), -0.01, "limit_band_pct"),
    (("orders", "limit_band_pct"), 1.5, "limit_band_pct"),
    (("orders", "allow_market"), "yes", "allow_market"),
    (("orders", "allow_market"), 1, "allow_market"),
    (("funding", "require_user_approval"), "false", "require_user_approval"),
    (("funding", "require_user_approval"), 0, "require_user_approval"),
    (("costs", "slippage_bps"), -1, "slippage_bps"),
    (("costs", "slippage_bps"), 1001, "slippage_bps"),
    (("costs", "slippage_bps"), "five", "slippage_bps"),
    (("costs", "commission_per_order"), -0.5, "commission_per_order"),
    (("costs", "commission_per_order"), 1.0, "must be 0 in this version"),
    (("costs", "commission_per_order"), 0.01, "no commission, so any other value would count cash the account never pays"),
    (("ledger", "path"), "", "path"),
    (("ledger", "path"), 7, "path"),
    (("config_version",), "", "config_version"),
])
def test_each_value_this_version_refuses_is_an_error_naming_the_setting(tmp_path, keys, value, message):
    with pytest.raises(ValueError, match=message):
        tc.load_config(write(tmp_path, edited(*keys, value)))


def test_a_file_that_is_missing_not_json_or_not_an_object_is_refused_with_its_name(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        tc.load_config(tmp_path / "nowhere.json")
    with pytest.raises(ValueError, match="not valid JSON"):
        tc.load_config(write(tmp_path, "{not json"))
    with pytest.raises(ValueError, match="must be a JSON object"):
        tc.load_config(write(tmp_path, "[1, 2]"))


# ---------------------------------------------------------------- PAPER_TRADING (rule T4)


