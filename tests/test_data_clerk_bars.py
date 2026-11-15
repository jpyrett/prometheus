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

def test_the_bars_of_a_range_are_oldest_first_with_both_ends_included(bars_root):
    found = tools(bars_root).daily_bars(["PG", "AAPL"], date(2026, 10, 12), date(2026, 10, 16))
    assert list(found) == ["bars", "gaps"] and list(found["bars"]) == ["PG", "AAPL"] and found["gaps"] == []
    pg = found["bars"]["PG"]
    assert [b["date"] for b in pg] == ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]
    assert list(pg[0]) == ["date", "open", "high", "low", "close", "volume"]
    assert all(isinstance(b["date"], str) for b in pg)
    assert all(b["low"] <= b["open"] <= b["high"] and b["low"] <= b["close"] <= b["high"] for b in pg)
    assert [b["close"] for b in found["bars"]["AAPL"]] == [b["close"] - 2 for b in pg]               # the rows of the other symbol in the same files are not mixed in


def test_a_bar_is_the_stored_number_and_not_another_symbols(bars_root):
    days = weekdays(40, TODAY)
    i = days.index(date(2026, 10, 14))
    bar = next(b for b in tools(bars_root).daily_bars(["SPY"], date(2026, 10, 14), date(2026, 10, 14))["bars"]["SPY"])
    assert bar == {"date": "2026-10-14", "open": 100 + i + 1, "high": 101.5 + i + 1, "low": 99 + i + 1, "close": 100.5 + i + 1, "volume": 1000 + i}


def test_one_day_is_a_range_and_a_weekend_inside_the_range_is_not_a_gap(bars_root):
    found = tools(bars_root).daily_bars(["PG"], date(2026, 10, 16), date(2026, 10, 16))
    assert [b["date"] for b in found["bars"]["PG"]] == ["2026-10-16"]
    saturday = tools(bars_root).daily_bars(["PG"], date(2026, 10, 17), date(2026, 10, 18))
    assert saturday["bars"] == {"PG": []}
    over = tools(bars_root).daily_bars(["PG"], date(2026, 10, 15), date(2026, 10, 20))
    assert [b["date"] for b in over["bars"]["PG"]] == ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"] and over["gaps"] == []


def test_symbols_are_cleaned_once_each_and_a_symbol_that_is_not_stored_is_an_empty_list(bars_root):
    found = tools(bars_root).daily_bars(["$pg", "PG", " spy ", "ZZZZ"], date(2026, 10, 12), date(2026, 10, 13))
    assert list(found["bars"]) == ["PG", "SPY", "ZZZZ"] and len(found["bars"]["PG"]) == 2 and found["bars"]["ZZZZ"] == []


def test_only_the_day_files_the_range_needs_are_read(bars_root, monkeypatch):
    read = []
    real = bars_module.pv_csv.read_csv
    monkeypatch.setattr(bars_module.pv_csv, "read_csv", lambda path, *a, **k: (read.append(str(path)), real(path, *a, **k))[1])
    tools(bars_root).daily_bars(["PG"], date(2026, 10, 12), date(2026, 10, 14))
    assert sorted(p.rsplit("/", 1)[-1] for p in read) == ["2026-10-12.csv.gz", "2026-10-13.csv.gz", "2026-10-14.csv.gz"]


def test_a_range_after_the_newest_stored_day_is_empty_lists_and_a_gap_that_says_so(bars_root):
    found = tools(bars_root).daily_bars(["PG", "SPY"], date(2026, 10, 26), date(2026, 11, 10))
    assert found["bars"] == {"PG": [], "SPY": []}
    assert len(found["gaps"]) == 1 and "no day_aggs_v1 files from 2026-10-26 to 2026-11-10" in found["gaps"][0]


def test_a_range_that_runs_past_the_stored_days_gives_the_days_there_are(bars_root):
    found = tools(bars_root).daily_bars(["PG"], date(2026, 10, 22), date(2026, 11, 30))
    assert [b["date"] for b in found["bars"]["PG"]] == ["2026-10-22", "2026-10-23"] and found["gaps"] == []


def test_a_missing_day_inside_the_range_is_named_and_no_bar_is_made_up(tmp_path):
    root = tmp_path / "day_aggs_v1"
    days = weekdays(10, TODAY)
    write_day_files(root, {d: [{"ticker": "PG", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}] for d in days}, skip=(date(2026, 10, 14),))
    found = tools(root).daily_bars(["PG"], date(2026, 10, 12), date(2026, 10, 16))
    assert [b["date"] for b in found["bars"]["PG"]] == ["2026-10-12", "2026-10-13", "2026-10-15", "2026-10-16"]
    assert found["gaps"] == ["2026-10-14: no file (a market holiday, or not synced)"]


def test_the_rest_files_add_the_days_the_day_files_lack_and_start_still_applies(tmp_path):
    root = tmp_path / "day_aggs_v1"
    write_day_files(root, {d: [{"ticker": "PG", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}] for d in weekdays(3, date(2026, 10, 14))})   # 12, 13, 14
    extra = tmp_path / "ticker_bars_v1"
    extra.mkdir()
    stamp = lambda d: int(datetime(d.year, d.month, d.day, 4, tzinfo=bars_module.timezone.utc).timestamp() * 1e9)
    (extra / "PG.csv").write_text("ticker,volume,open,close,high,low,window_start,transactions\n" + "".join(
        f"PG,5,3,3.5,4,2.5,{stamp(d)},7\n" for d in (date(2026, 10, 9), date(2026, 10, 14), date(2026, 10, 15), date(2026, 10, 16))), encoding="utf-8")
    found = tools(root).daily_bars(["PG"], date(2026, 10, 13), date(2026, 10, 16))
    assert [(b["date"], b["open"]) for b in found["bars"]["PG"]] == [("2026-10-13", 1), ("2026-10-14", 1), ("2026-10-15", 3), ("2026-10-16", 3)]   # the day file wins on the 14th; the 9th is before start


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


def test_daily_bars_is_not_a_tool_of_the_model():
    assert len(DATA_CLERK_TOOLS) == 13 and "daily_bars" not in DATA_CLERK_TOOLS
    assert "daily_bars" not in [t.name for t in DataClerkTools(toolbox=object(), store=object()).langchain_tools()]


def test_the_default_root_is_the_toolboxs_data_historical(tmp_path):
    class Toolbox:
        root_data_dir = tmp_path / "historical"

    root = Toolbox.root_data_dir / "day_aggs_v1"
    write_day_files(root, {date(2026, 10, 12): [{"ticker": "PG", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]})
    found = DataClerkTools(toolbox=Toolbox(), store=object()).daily_bars(["PG"], date(2026, 10, 12), date(2026, 10, 12))
    assert found["bars"]["PG"][0]["date"] == "2026-10-12"


# ---------------------------------------------------------------- DataClerk.request_bars

def test_request_bars_puts_the_bars_on_the_order_without_a_model_call(bars_root):
    o = order(("pg", "$AAPL", "PG"))
    result = clerk(bars_root).request_bars(o)
    assert result is o and result.status == "data_ready" and result.errors == [] and result.data.notes == []
    assert list(result.data.bars) == ["PG", "AAPL"] and [b["date"] for b in result.data.bars["PG"]] == ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]
    assert result.data.tool_calls == [{"name": "daily_bars", "args": {"symbols": 2, "start": "2026-10-12", "end": "2026-10-16"}, "outcome": "ok"}]
    assert result.data.answer == "daily bars of 2 symbols, 2026-10-12 to 2026-10-16" and result.data.completed_utc.endswith("Z")
    assert wo.WorkOrder.model_validate_json(result.model_dump_json()) == result                          # plain JSON
    assert result.action_type == "fetch_bars" and result.recommendations == [] and result.trades == []


def test_a_symbol_with_no_bar_in_the_range_is_an_empty_list_and_a_note_not_a_failure(bars_root):
    result = clerk(bars_root).request_bars(order(("PG", "ZZZZ")))
    assert result.status == "data_ready" and result.errors == [] and result.data.problems == []
    assert result.data.bars["ZZZZ"] == [] and len(result.data.bars["PG"]) == 5
    assert result.data.notes == ["ZZZZ: no stored bar from 2026-10-12 to 2026-10-16"]


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


def test_a_bars_order_carries_no_search_results_and_no_advice(bars_root):
    result = clerk(bars_root).request_bars(order())
    assert result.data.sources == [] and result.data.sweep == {} and result.advisor_runs == [] and result.recommendations == []
