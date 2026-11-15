"""Tests for DataClerkTools.daily_bars and DataClerk.request_bars (Trader v1 spec section 12): the stored daily bars of a range, for the Account Manager's backtest.

No network and no model: the bars are day files written into a temporary folder in the real format (`bar_helpers.write_day_files`), and the clerk is built with a model that fails the test if it is touched.
"""
from datetime import date, datetime

import pytest

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk import bars as bars_module
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import work_order as wo
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError

TODAY = date(2026, 10, 23)
SYMBOLS = ("AAPL", "SPY", "PG")


class ForbiddenModel:
    """A model that fails the test if the Data Clerk so much as looks at it: `request_bars` is code."""

    def __getattr__(self, name):
        raise AssertionError(f"request_bars used the model ({name})")


@pytest.fixture
def bars_root(tmp_path):
    root = tmp_path / "day_aggs_v1"
    rows = {d: [{"ticker": t, "open": 100 + i + k, "high": 101.5 + i + k, "low": 99 + i + k, "close": 100.5 + i + k, "volume": 1000 + i} for k, t in enumerate(SYMBOLS)] for i, d in enumerate(weekdays(40, TODAY))}
    write_day_files(root, rows)
    return root


def tools(bars_root):
    return DataClerkTools(toolbox=object(), store=object(), bars_root=bars_root)


def clerk(bars_root):
    c = DataClerk(toolbox=object(), model=ForbiddenModel())
    c.tools.bars_root = bars_root
    return c


def order(symbols=("PG",), start="2026-10-12", end="2026-10-16", **changes):
    result = wo.new_work_order("bars", requester="account_manager", action_type="fetch_bars", subject=list(symbols), bars={"start": start, "end": end})
    for key, value in changes.items():
        setattr(result, key, value)
    return result


# ---------------------------------------------------------------- DataClerkTools.daily_bars


def test_one_day_is_a_range_and_a_weekend_inside_the_range_is_not_a_gap(bars_root):
    found = tools(bars_root).daily_bars(["PG"], date(2026, 10, 16), date(2026, 10, 16))
    assert [b["date"] for b in found["bars"]["PG"]] == ["2026-10-16"]
    saturday = tools(bars_root).daily_bars(["PG"], date(2026, 10, 17), date(2026, 10, 18))
    assert saturday["bars"] == {"PG": []}
    over = tools(bars_root).daily_bars(["PG"], date(2026, 10, 15), date(2026, 10, 20))
    assert [b["date"] for b in over["bars"]["PG"]] == ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"] and over["gaps"] == []


def test_a_range_after_the_newest_stored_day_is_empty_lists_and_a_gap_that_says_so(bars_root):
    found = tools(bars_root).daily_bars(["PG", "SPY"], date(2026, 10, 26), date(2026, 11, 10))
    assert found["bars"] == {"PG": [], "SPY": []}
    assert len(found["gaps"]) == 1 and "no day_aggs_v1 files from 2026-10-26 to 2026-11-10" in found["gaps"][0]


def test_a_missing_day_inside_the_range_is_named_and_no_bar_is_made_up(tmp_path):
    root = tmp_path / "day_aggs_v1"
    days = weekdays(10, TODAY)
    write_day_files(root, {d: [{"ticker": "PG", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}] for d in days}, skip=(date(2026, 10, 14),))
    found = tools(root).daily_bars(["PG"], date(2026, 10, 12), date(2026, 10, 16))
    assert [b["date"] for b in found["bars"]["PG"]] == ["2026-10-12", "2026-10-13", "2026-10-15", "2026-10-16"]
    assert found["gaps"] == ["2026-10-14: no file (a market holiday, or not synced)"]


@pytest.mark.parametrize("symbols, start, end, message", [
    ("PG", date(2026, 10, 12), date(2026, 10, 13), "at least one ticker"),
    ([], date(2026, 10, 12), date(2026, 10, 13), "at least one ticker"),
    (["../x"], date(2026, 10, 12), date(2026, 10, 13), "not a valid ticker symbol"),
    ([7], date(2026, 10, 12), date(2026, 10, 13), "must be text"),
    (["PG"], "2026-10-12", date(2026, 10, 13), "must be dates"),
    (["PG"], datetime(2026, 10, 12, 9, 30), date(2026, 10, 13), "must be dates"),
    (["PG"], date(2026, 10, 12), None, "must be dates"),
    (["PG"], date(2026, 10, 14), date(2026, 10, 13), "on or before end"),
])
def test_daily_bars_refuses_what_makes_no_sense(bars_root, symbols, start, end, message):
    with pytest.raises(ToolInputError, match=message):
        tools(bars_root).daily_bars(symbols, start, end)


def test_a_bar_folder_that_does_not_exist_is_an_error_and_not_no_bars(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        tools(tmp_path / "nowhere").daily_bars(["PG"], date(2026, 10, 12), date(2026, 10, 13))


# ---------------------------------------------------------------- DataClerk.request_bars


def test_a_range_in_the_future_is_data_ready_with_empty_lists_and_the_gap_in_the_notes(bars_root):
    result = clerk(bars_root).request_bars(order(("PG", "SPY"), start="2026-11-05", end="2026-11-30"))
    assert result.status == "data_ready" and result.data.bars == {"PG": [], "SPY": []}
    assert result.data.notes[:2] == ["PG: no stored bar from 2026-11-05 to 2026-11-30", "SPY: no stored bar from 2026-11-05 to 2026-11-30"]
    assert "no day_aggs_v1 files from 2026-11-05 to 2026-11-30" in result.data.notes[2] and result.data.notes[2].startswith("bars: ")


class SpyTools:
    def __init__(self):
        self.asked = []

    def daily_bars(self, *args):
        self.asked.append(args)
        raise AssertionError("a refused order must not read the store")


@pytest.mark.parametrize("changes, message", [
    ({"action_type": "fetch_quote"}, "is not 'fetch_bars'"),
    ({"subject": None}, "needs one or more ticker symbols"),
    ({"bars": {"start": "2026-10-16", "end": "2026-10-12"}}, "is after end"),
    ({"bars": None}, "needs `bars`"),
    ({"bars": {"start": "2024-01-01", "end": "2026-10-12"}}, "at most 400 days"),
])
def test_a_refused_order_is_data_failed_with_the_reasons_and_the_store_is_not_read(bars_root, changes, message):
    c = clerk(bars_root)
    c.tools = SpyTools()
    result = c.request_bars(order(**changes))
    assert result.status == "data_failed" and any(message in e for e in result.errors) and result.data is None and c.tools.asked == []


def test_a_symbol_that_is_not_a_ticker_is_refused_before_anything_is_read(bars_root):
    c = clerk(bars_root)
    c.tools = SpyTools()
    result = c.request_bars(order(("../etc",)))
    assert result.status == "data_failed" and "not a valid ticker symbol" in result.errors[0] and c.tools.asked == []


def test_an_order_that_already_has_its_data_comes_back_unchanged(bars_root):
    c = clerk(bars_root)
    done = c.request_bars(order())
    before = done.model_dump_json()
    c.tools = SpyTools()
    assert c.request_bars(done) is done and done.model_dump_json() == before and c.tools.asked == []


def test_a_store_that_cannot_be_read_makes_the_order_data_failed_and_says_what_kind_of_failure(tmp_path):
    result = clerk(tmp_path / "nowhere").request_bars(order())
    assert result.status == "data_failed" and len(result.errors) == 1
    assert result.errors[0].startswith("the stored bars could not be read (FileNotFoundError:") and result.data.bars == {}
    assert result.data.tool_calls == [{"name": "daily_bars", "args": {"symbols": 1, "start": "2026-10-12", "end": "2026-10-16"}, "outcome": "error"}]


def test_request_bars_writes_nothing_under_the_bar_folder(bars_root):
    before = sorted(str(p) for p in bars_root.parent.rglob("*"))
    clerk(bars_root).request_bars(order(("PG", "SPY")))
    clerk(bars_root).request_bars(order(("PG",), start="2026-12-01", end="2026-12-05"))
    assert sorted(str(p) for p in bars_root.parent.rglob("*")) == before


