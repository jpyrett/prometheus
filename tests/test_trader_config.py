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


def test_the_shipped_file_loads_and_holds_the_specs_numbers():
    c = tc.load_config()
    assert (c.config_version, c.mode) == ("1", "paper")
    assert (c.limits.max_trades, c.limits.base_turns, c.limits.max_turns, c.limits.model_timeout_seconds, c.limits.max_note_words, c.limits.max_summary_words) == (20, 5, 12, 60, 20, 60)
    assert (c.orders.max_chase_pct, c.orders.limit_band_pct, c.orders.allow_market, c.orders.cash_policy) == (0.01, 0.03, True, "reduce")
    assert (c.costs.slippage_bps, c.costs.commission_per_order) == (5, 0.0) and c.funding.require_user_approval is True
    assert (c.backtest.fill_at, c.backtest.max_follow_days, c.backtest.benchmark) == ("next_open", 90, "SPY") and c.ledger.path == "data/ledger/trades.jsonl"


def test_the_loaded_settings_cannot_be_changed():
    c = tc.load_config()
    with pytest.raises(Exception):
        c.mode = "backtest"
    with pytest.raises(Exception):
        c.orders.max_chase_pct = 0.5


def test_a_copy_of_the_shipped_file_in_another_place_loads_the_same_way(tmp_path):
    assert tc.load_config(write(tmp_path, raw())) == tc.load_config()


def test_the_notes_in_the_file_are_not_settings(tmp_path):
    data = {**raw(), "_other_note": "anything"}
    assert tc.load_config(write(tmp_path, data)).config_version == "1"


def test_the_ledger_is_the_trades_ledger_and_not_another_ones():
    assert tc.load_config().ledger.path == "data/ledger/trades.jsonl" and "recommendations" not in tc.load_config().ledger.path and "reviews" not in tc.load_config().ledger.path


@pytest.mark.parametrize("section", ["mode", "limits", "orders", "costs", "funding", "backtest", "ledger", "config_version"])
def test_a_missing_section_is_an_error_that_names_the_file(tmp_path, section):
    data = raw()
    del data[section]
    with pytest.raises(ValueError, match="trader.json") as error:
        tc.load_config(write(tmp_path, data))
    assert section in str(error.value)


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


def test_a_value_that_is_allowed_changes_what_is_read(tmp_path):
    c = tc.load_config(write(tmp_path, edited("mode", "backtest")))
    assert c.mode == "backtest"
    c = tc.load_config(write(tmp_path, edited("orders", "allow_market", False)))
    assert c.orders.allow_market is False
    c = tc.load_config(write(tmp_path, edited("costs", "slippage_bps", 0)))
    assert c.costs.slippage_bps == 0
    assert tc.load_config(write(tmp_path, edited("backtest", "benchmark", "qqq"))).backtest.benchmark == "QQQ"
    assert tc.load_config(write(tmp_path, edited("limits", "max_turns", 5))).limits.max_turns == 5          # equal to base_turns is allowed
    assert tc.load_config(write(tmp_path, edited("backtest", "max_follow_days", 400))).backtest.max_follow_days == 400


def test_a_commission_of_zero_is_the_only_one_this_version_loads(tmp_path):
    """Spec section 15 (decided after phase 2): `apply_fill` and the account book charge no commission, so the loader refuses any other value (the parametrized refusals above); 0 and 0.0 are the shipped one."""
    assert tc.load_config(write(tmp_path, edited("costs", "commission_per_order", 0))).costs.commission_per_order == 0
    assert tc.load_config(write(tmp_path, edited("costs", "commission_per_order", 0.0))).costs.commission_per_order == 0.0
    assert tc.load_config().costs.commission_per_order == 0.0


def test_a_file_that_is_missing_not_json_or_not_an_object_is_refused_with_its_name(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        tc.load_config(tmp_path / "nowhere.json")
    with pytest.raises(ValueError, match="not valid JSON"):
        tc.load_config(write(tmp_path, "{not json"))
    with pytest.raises(ValueError, match="must be a JSON object"):
        tc.load_config(write(tmp_path, "[1, 2]"))


# ---------------------------------------------------------------- PAPER_TRADING (rule T4)

def test_paper_trading_is_on_in_every_test_by_default():
    assert tc.paper_trading_enabled() is True                                                       # the autouse fixture in conftest.py: no test inherits Jacob's .env


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "YES", "on", "On", "1", " true ", "yes\n"])
def test_the_words_that_let_the_trader_run(monkeypatch, value):
    monkeypatch.setenv("PAPER_TRADING", value)
    assert tc.paper_trading_enabled() is True


@pytest.mark.parametrize("value", ["false", "False", "no", "off", "0", "live", "truee", "t", "enabled", "y", "2", "-1", "null"])
def test_anything_else_refuses_every_run_a_typo_included(monkeypatch, value):
    monkeypatch.setenv("PAPER_TRADING", value)
    assert tc.paper_trading_enabled() is False


def test_without_the_setting_in_the_environment_or_env_the_default_is_true(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPER_TRADING", raising=False)
    monkeypatch.setattr("prometheus.rag.config.REPO_ROOT", tmp_path)                                  # no .env here
    assert tc.paper_trading_enabled() is True
    (tmp_path / ".env").write_text("PAPER_TRADING=false\n", encoding="utf-8")
    assert tc.paper_trading_enabled() is False                                                      # .env is read when the environment says nothing
    monkeypatch.setenv("PAPER_TRADING", "true")
    assert tc.paper_trading_enabled() is True                                                       # and the environment wins


def test_env_example_documents_the_setting_with_the_value_that_lets_the_trader_run():
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "\nPAPER_TRADING=true\n" in example and "\nTRADER_MODEL=openai/gpt-oss-120b\n" in example
