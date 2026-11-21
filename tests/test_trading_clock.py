"""USE_LIVE_TRADING: the desk's clock, real or simulated (design choices entry 76).

REAL DATA ONLY: the quote tests read the real stored daily bars of real symbols (PG on 2026-11-13: open 168.97, high 170.73, low 167.20, close 169.80, the day before 168.03), and the Critic test replays the
real capture of the Critic's review of PG. No agent is replaced by a stand-in and no price is invented; a test that needs the stored bars is skipped when they are not on the machine. The desk, the runner and the
decisions are checked with the real agents in `tests/test_end_to_end_live.py` (opt-in: it costs a few cents).

`tests/conftest.py` pins USE_LIVE_TRADING=true for every test (so no test inherits Jacob's .env, which says false); a test that wants the simulation sets it itself. The real clock is frozen to a Saturday,
2026-11-14 01:30 Eastern, where a test needs to know what "now" is: the moment the setting was written for (the market is closed, and the previous business day is Friday 2026-11-13).
"""
import json
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from prometheus.common import market_calendar as mc
from prometheus.common import trading_clock as tc
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError

ET = ZoneInfo("America/New_York")


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setenv("USE_LIVE_TRADING", "false")


# ---------------------------------------------------------------- the clock


@pytest.mark.parametrize("label, real, expected_day", [
    ("Friday 16:12, after the close: Friday itself (Jacob, 2026-11-20)", datetime(2026, 11, 20, 16, 12, tzinfo=ET), date(2026, 11, 20)),
    ("Friday at the close exactly", datetime(2026, 11, 20, 16, 0, tzinfo=ET), date(2026, 11, 20)),
    ("Friday 15:59, not closed yet: Thursday", datetime(2026, 11, 20, 15, 59, tzinfo=ET), date(2026, 11, 19)),
    ("Friday 23:59: Friday", datetime(2026, 11, 20, 23, 59, tzinfo=ET), date(2026, 11, 20)),
    ("Saturday 00:01, past midnight: Friday", datetime(2026, 11, 21, 0, 1, tzinfo=ET), date(2026, 11, 20)),
    ("Sunday: Friday", datetime(2026, 11, 22, 12, 0, tzinfo=ET), date(2026, 11, 20)),
    ("Tuesday 00:30, past midnight: Monday", datetime(2026, 11, 24, 0, 30, tzinfo=ET), date(2026, 11, 23)),
    ("the early-close Friday at 13:05: that Friday", datetime(2026, 11, 27, 13, 5, tzinfo=ET), date(2026, 11, 27)),
    ("Thanksgiving evening (a holiday): Wednesday", datetime(2026, 11, 26, 18, 0, tzinfo=ET), date(2026, 11, 25)),
])
def test_after_the_close_the_simulated_day_is_today_and_past_midnight_or_at_the_weekend_the_last_day_that_closed(simulated, label, real, expected_day):
    moment = tc.trading_now(real)
    assert tc.last_closed_day(real) == expected_day and tc.market_day(real) == expected_day, label
    assert mc.session_at(moment).state == "regular" and mc.session_at(moment).trading_date == expected_day, label


@pytest.mark.parametrize("on", ["true", "false"])
def test_a_moment_without_a_timezone_is_a_mistake_in_either_mode(monkeypatch, on):
    monkeypatch.setenv("USE_LIVE_TRADING", on)
    with pytest.raises(ValueError, match="timezone"):
        tc.trading_now(datetime(2026, 11, 14, 1, 30))
    with pytest.raises(ValueError):
        tc.last_closed_day(datetime(2026, 11, 14, 1, 30))


# ---------------------------------------------------------------- the calendar's new setting

def test_a_day_that_is_not_a_trading_day_has_no_simulated_moment():
    with pytest.raises(ValueError, match="not a trading day"):
        mc.simulated_moment(date(2026, 11, 14))
    with pytest.raises(ValueError, match="holiday: Thanksgiving"):
        mc.simulated_moment(date(2026, 11, 26))


def calendar_with(tmp_path, **changes):
    raw = json.loads(mc.CALENDAR_PATH.read_text(encoding="utf-8"))
    raw.update(changes)
    path = tmp_path / f"calendar_{'_'.join(f'{k}{v}' for k, v in changes.items())}.json"       # `load_calendar` caches by path: one file name for each config
    path.write_text(json.dumps(raw), encoding="utf-8")
    return mc.load_calendar(path)


def test_the_minutes_before_the_close_must_keep_the_moment_inside_every_regular_session(tmp_path):
    assert calendar_with(tmp_path, simulated_minutes_before_close=209).simulated_minutes_before_close == 209     # an early-close session is 210 minutes long
    for bad in (0, -5, 210, 400):
        with pytest.raises(ValueError, match="simulated_minutes_before_close"):
            calendar_with(tmp_path, simulated_minutes_before_close=bad)
    raw = json.loads(mc.CALENDAR_PATH.read_text(encoding="utf-8"))
    del raw["simulated_minutes_before_close"]
    (tmp_path / "missing.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="simulated_minutes_before_close"):
        mc.load_calendar(tmp_path / "missing.json")


# ---------------------------------------------------------------- the quote: the stored bar of the day (the real stored bars)

FRIDAY = date(2026, 11, 13)


@pytest.fixture(scope="module")
def stored():
    """The real stored bars of PG on 2026-11-12 and 2026-11-13, or skip."""
    try:
        bars = DataClerkTools().daily_bars(["PG"], date(2026, 11, 12), FRIDAY)["bars"]["PG"]
    except FileNotFoundError:
        pytest.skip("the stored daily bars are not on this machine")
    if [b["date"] for b in bars] != ["2026-11-12", "2026-11-13"]:
        pytest.skip("the stored bars do not hold PG for 2026-11-12 and 2026-11-13")
    return {b["date"]: b for b in bars}


def test_a_day_with_no_stored_bar_is_not_answered_with_another_days_price(stored):
    with pytest.raises(ToolInputError, match="no stored daily bar for PG on 2026-11-14"):
        DataClerkTools().close_quote("PG", date(2026, 11, 14))                    # a Saturday: Friday's bar is NOT used
    with pytest.raises(ToolInputError, match="no stored daily bar for ZZZZ"):
        DataClerkTools().close_quote("ZZZZ", FRIDAY)                              # a symbol the store does not have
    with pytest.raises(ToolInputError, match="day must be a date"):
        DataClerkTools().close_quote("PG", "2026-11-13")
    with pytest.raises(ToolInputError):
        DataClerkTools().close_quote("not a ticker!", FRIDAY)


def test_a_missing_bar_folder_is_an_error_not_no_price(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataClerkTools(bars_root=tmp_path / "nowhere").close_quote("PG", FRIDAY)


def test_the_close_quote_is_not_a_tool_of_the_model():
    assert "close_quote" not in DATA_CLERK_TOOLS and len(DATA_CLERK_TOOLS) == 13


# ---------------------------------------------------------------- the rule that the Trader does not trade for real

def test_the_setting_never_turns_paper_trading_off():
    """USE_LIVE_TRADING picks the clock and the quote. It is not read by the Trader's rule T4, which is PAPER_TRADING, and there is no live trading to switch on."""
    import inspect
    from prometheus.agents.trader import config as trader_config
    assert "USE_LIVE_TRADING" not in inspect.getsource(trader_config)
    assert "USE_LIVE_TRADING" in inspect.getsource(tc) and "PAPER_TRADING" not in inspect.getsource(tc).replace("`PAPER_TRADING`", "")
