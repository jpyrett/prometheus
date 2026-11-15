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
from test_critic_recheck import cross

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.common import market_calendar as mc
from prometheus.common import trading_clock as tc
from prometheus.common import work_order as wo
from prometheus.rag import config as rag_config
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError

ET = ZoneInfo("America/New_York")
SATURDAY = datetime(2026, 11, 14, 1, 30, tzinfo=ET)
SIMULATED_FRIDAY = datetime(2026, 11, 13, 15, 30, tzinfo=ET)        # the moment USE_LIVE_TRADING=false gives on that Saturday


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setenv("USE_LIVE_TRADING", "false")


@pytest.fixture
def saturday(monkeypatch, simulated):
    """USE_LIVE_TRADING=false AND the real clock frozen to Saturday 01:30 Eastern."""
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return SATURDAY.astimezone(tz) if tz else SATURDAY.replace(tzinfo=None)
    monkeypatch.setattr(tc, "datetime", Frozen)


# ---------------------------------------------------------------- the setting


# ---------------------------------------------------------------- the clock

def test_with_the_real_clock_the_moment_is_the_moment_given_or_now():
    assert tc.trading_now(SATURDAY) == SATURDAY and tc.trading_now(SATURDAY).tzinfo is not None
    assert abs((tc.trading_now() - datetime.now(timezone.utc)).total_seconds()) < 5
    assert tc.clock_note(SATURDAY) is None and tc.market_day(SATURDAY) == date(2026, 11, 14)


@pytest.mark.parametrize("label, real, expected_day", [
    ("Saturday", datetime(2026, 11, 14, 1, 30, tzinfo=ET), date(2026, 11, 13)),
    ("Sunday", datetime(2026, 11, 15, 23, 0, tzinfo=ET), date(2026, 11, 13)),
    ("Monday morning", datetime(2026, 11, 16, 8, 0, tzinfo=ET), date(2026, 11, 13)),
    ("a Tuesday while the market is open", datetime(2026, 11, 17, 11, 0, tzinfo=ET), date(2026, 11, 16)),
    ("the Friday after Thanksgiving (Thursday was a holiday)", datetime(2026, 11, 27, 10, 0, tzinfo=ET), date(2026, 11, 25)),
    ("the Monday after an early-close Friday", datetime(2026, 11, 30, 10, 0, tzinfo=ET), date(2026, 11, 27)),
])
def test_the_simulated_clock_is_the_previous_business_day_inside_its_regular_session(simulated, label, real, expected_day):
    moment = tc.trading_now(real)
    session = mc.session_at(moment)
    assert tc.market_day(real) == expected_day and mc.eastern_date(moment) == expected_day, label
    assert session.state == "regular" and session.trading_date == expected_day, label       # the market is "open", and the day is a trading day
    assert tc.previous_business_day(real) == expected_day


@pytest.mark.parametrize("on", ["true", "false"])
def test_a_moment_without_a_timezone_is_a_mistake_in_either_mode(monkeypatch, on):
    monkeypatch.setenv("USE_LIVE_TRADING", on)
    with pytest.raises(ValueError, match="timezone"):
        tc.trading_now(datetime(2026, 11, 14, 1, 30))
    with pytest.raises(ValueError):
        tc.previous_business_day(datetime(2026, 11, 14, 1, 30))


def test_the_default_moment_is_derived_from_the_real_clock_only_here(saturday):
    assert tc.trading_now().astimezone(ET) == SIMULATED_FRIDAY and tc.market_day() == date(2026, 11, 13)


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


def finnhub_tripwire(monkeypatch):
    """Finnhub is the real `DataToolbox.get_quote`; with the simulated clock it must never be called, so it is made to fail the test if it is (a tripwire: it fabricates no data)."""
    def asked(self, symbol):
        raise AssertionError("Finnhub was asked for a live quote")
    monkeypatch.setattr(DataToolbox, "get_quote", asked)


def quote_order(*symbols):
    return wo.new_work_order("prices", requester="account_manager", action_type="fetch_quote", subject=list(symbols), live=True)


# ---------------------------------------------------------------- the Critic does not compare the bar with itself

def test_a_simulated_quote_is_not_a_second_provider_so_the_price_check_is_skipped_not_passed():
    live = cross()
    assert "price" in {c.name for c in live[0]}
    simulated_quote = cross(quote_mutate=lambda q: q.update(simulated=True))
    assert "price" not in {c.name for c in simulated_quote[0]} and simulated_quote[1] == [] and {c.name for c in simulated_quote[0]} == {"identity", "high_52w"}
    wildly_different = cross(quote_mutate=lambda q: q.update(simulated=True, price=q["price"] * 1.5, previous_close=1.0))
    assert wildly_different[1] == []                                          # no `price_mismatch` from a quote that is the bar itself
    real_but_different = cross(quote_mutate=lambda q: q.update(price=q["price"] * 1.5))
    assert [f.code for f in real_but_different[1]] == ["price_mismatch"]       # a REAL second provider still is caught


# ---------------------------------------------------------------- the rule that the Trader does not trade for real

def test_the_setting_never_turns_paper_trading_off():
    """USE_LIVE_TRADING picks the clock and the quote. It is not read by the Trader's rule T4, which is PAPER_TRADING, and there is no live trading to switch on."""
    import inspect
    from prometheus.agents.trader import config as trader_config
    assert "USE_LIVE_TRADING" not in inspect.getsource(trader_config)
    assert "USE_LIVE_TRADING" in inspect.getsource(tc) and "PAPER_TRADING" not in inspect.getsource(tc).replace("`PAPER_TRADING`", "")
