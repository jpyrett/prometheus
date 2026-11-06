"""Tests for DataClerk.request_metrics and DataClerkTools.calculate_metrics (Advisor v1 spec section 4.1, phase 2, design choices entry 51).

No network: a fake toolbox stands in for Finnhub, the bars are day files written into a temporary folder in the real format, and the stored filings are the real Apple ones.
"""
import json
from datetime import date

import pytest
from test_filing_signals import APPLE_TODAY, store

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import work_order as wo
from prometheus.tools.data_clerk import DATA_CLERK_TOOLS, DataClerkTools, ToolInputError

TODAY = date(2026, 10, 23)
FINNHUB = {"peTTM": 25.0, "roeTTM": 30.0, "marketCapitalization": 4_000_000.0}


class Financials:
    def __init__(self, fail=()):
        self.asked, self.fail = [], set(fail)

    def get(self, symbol):
        self.asked.append(symbol)
        if symbol in self.fail:
            raise RuntimeError(f"No financial metrics returned for {symbol}")
        from prometheus.agents.data_clerk.data_toolbox import parse_metric_payload
        return parse_metric_payload(symbol, FINNHUB, "2026-10-23")


class Toolbox:
    def __init__(self, fail=()):
        self.financials = Financials(fail)


@pytest.fixture
def bars_root(tmp_path):
    root = tmp_path / "day_aggs_v1"
    rows = {d: [{"ticker": t, "open": 100 + i, "high": 101.5 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000} for t in ("AAPL", "SPY", "PG")] for i, d in enumerate(weekdays(40, TODAY))}
    write_day_files(root, rows)
    return root


@pytest.fixture
def filings(tmp_path, monkeypatch, capsys):
    store(tmp_path, monkeypatch, "AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006")  # the real Apple filings; FILINGS_OUTPUT_DIR points at them
    capsys.readouterr()


def clerk(bars_root, toolbox=None):
    c = DataClerk(toolbox=toolbox or Toolbox(), model=object())
    c.tools.bars_root = bars_root
    c.tools.today = APPLE_TODAY
    return c


def order(symbols, *names, **kw):
    return wo.new_work_order("metrics", requester="advisor", action_type="calculate_financials", subject=symbols, metrics=[{"name": n} for n in names], **kw)


def test_the_data_clerk_puts_the_metrics_of_every_symbol_on_the_order_without_a_model_call(bars_root, filings):
    toolbox = Toolbox()
    result = clerk(bars_root, toolbox).request_metrics(order(["aapl", "$PG", "AAPL"], "ret_5d", "pe_ratio", "market_cap", "share_count_change", "net_payout_yield_ttm", "rsi_14"), today=TODAY)
    assert result.status == "data_ready" and result.errors == [] and toolbox.financials.asked == ["AAPL", "PG"]      # cleaned, once each
    m = result.data.metrics
    assert list(m) == ["AAPL", "PG"] and list(m["AAPL"]) == ["ret_5d", "pe_ratio", "market_cap", "share_count_change", "net_payout_yield_ttm", "rsi_14"]
    assert m["AAPL"]["ret_5d"]["value"] == pytest.approx(139.5 / 134.5 - 1) and m["AAPL"]["pe_ratio"]["value"] == 25.0
    assert m["AAPL"]["share_count_change"]["value"] < 0 and m["AAPL"]["net_payout_yield_ttm"]["value"] == pytest.approx((24701e6 + 3921e6) * 4 / 4.0e12)
    assert m["AAPL"]["rsi_14"]["value"] == 100.0
    assert m["PG"]["share_count_change"]["value"] is None and m["PG"]["share_count_change"]["note"] == "no stored filings of PG (fetch_filings has not been run for it)"
    assert result.data.tool_calls == [{"name": "calculate_metrics", "args": {"symbols": 2, "metrics": 6}, "outcome": "ok"}]
    assert any(n.startswith("PG: no filings are stored") for n in result.data.notes)                                  # what could not be read for PG is a note, not a failure
    assert wo.WorkOrder.model_validate_json(result.model_dump_json()) == result                                       # plain JSON


def test_a_metric_that_cannot_be_computed_is_a_result_with_a_note_not_a_failure(bars_root):
    result = clerk(bars_root).request_metrics(order(["AAPL"], "sma_200", "rsi_14", "sue_4q"), today=TODAY)
    assert result.status == "data_ready"
    assert result.data.metrics["AAPL"]["sma_200"]["note"] == "needs 200 bars, 40 stored" and result.data.metrics["AAPL"]["sue_4q"]["value"] is None


def test_finnhub_failing_for_one_symbol_is_a_note_and_the_bars_metrics_still_come(bars_root):
    result = clerk(bars_root, Toolbox(fail=["PG"])).request_metrics(order(["AAPL", "PG"], "ret_5d", "pe_ratio"), today=TODAY)
    assert result.status == "data_ready"
    pg = result.data.metrics["PG"]
    assert pg["ret_5d"]["value"] is not None and pg["pe_ratio"]["value"] is None and "no Finnhub metric payload for PG" in pg["pe_ratio"]["note"]
    assert any(n.startswith("PG: Finnhub metrics failed (RuntimeError: No financial metrics returned for PG") for n in result.data.notes)


def test_a_symbol_that_cannot_be_read_at_all_fails_the_order_and_is_named_while_the_others_stay(bars_root):
    result = clerk(bars_root).request_metrics(order(["AAPL", "ZZZZ"], "ret_5d", "last_close"), today=TODAY)
    assert result.status == "data_failed" and result.errors == ["ZZZZ: nothing could be read (no stored bars for it)"]
    assert result.data.metrics["AAPL"]["last_close"]["value"] == 139.5 and result.data.metrics["ZZZZ"]["last_close"]["value"] is None


def test_a_symbol_that_is_not_a_ticker_is_refused_before_anything_is_read(bars_root):
    toolbox = Toolbox()
    result = clerk(bars_root, toolbox).request_metrics(order(["../etc"], "ret_5d"), today=TODAY)
    assert result.status == "data_failed" and "not a valid ticker symbol" in result.errors[0] and toolbox.financials.asked == []


def test_an_order_that_is_not_a_metrics_order_is_refused_with_the_reasons(bars_root):
    result = clerk(bars_root).request_metrics(order(["AAPL"], "nope"), today=TODAY)
    assert result.status == "data_failed" and result.data is None and "unknown metrics: nope" in result.errors[0]


def test_an_order_that_already_has_its_data_comes_back_unchanged(bars_root):
    c = clerk(bars_root)
    first = c.request_metrics(order(["AAPL"], "ret_5d"), today=TODAY)
    toolbox_calls = c.tools.toolbox.financials.asked
    again = c.request_metrics(first, today=TODAY)
    assert again is first and toolbox_calls == []                       # ret_5d needs no Finnhub call at all: the bars metrics never touch the paid API


def test_only_the_inputs_a_requested_metric_needs_are_read(bars_root, filings):
    toolbox = Toolbox()
    c = clerk(bars_root, toolbox)
    c.request_metrics(order(["AAPL"], "ret_5d"), today=TODAY)
    assert toolbox.financials.asked == []                               # bars only: no Finnhub call, no filings read
    c.request_metrics(order(["AAPL"], "pe_ratio"), today=TODAY)
    assert toolbox.financials.asked == ["AAPL"]


def test_the_sector_of_the_order_gives_the_residual_metrics_an_etf(tmp_path):
    root = tmp_path / "day_aggs_v1"
    rows = {d: [{"ticker": t, "open": 10, "high": 12, "low": 9, "close": c, "volume": 1000} for t, c in (("AAPL", 11.0 + i), ("SPY", 50.0), ("XLK", 20.0 + i / 2))] for i, d in enumerate(weekdays(10, TODAY))}
    write_day_files(root, rows)
    # AAPL closes 11..20 (5 sessions ago 15): +33.3%; XLK closes 20, 20.5 ... 24.5 (5 sessions ago 22): +11.4%; the residual is the difference
    for sector, expected in ((None, None), ("XLK", (20 / 15 - 1) - (24.5 / 22 - 1))):
        result = clerk(root).request_metrics(order(["AAPL"], "residual_5d", **({"sector": sector} if sector else {})), today=TODAY)
        value = result.data.metrics["AAPL"]["residual_5d"]["value"]
        assert (value is None) if expected is None else value == pytest.approx(expected)


# ---------------------------------------------------------------- the tool the model and the MCP server see

def test_calculate_metrics_is_on_the_allow_list_and_is_the_code_behind_request_metrics(bars_root, filings):
    assert DATA_CLERK_TOOLS[-1] == "calculate_metrics"
    tools = DataClerkTools(toolbox=Toolbox(), bars_root=bars_root)
    tools.today = TODAY
    out = tools.calculate_metrics(["aapl"], ["ret_5d", "pe_ratio"])
    assert out["as_of"] == "2026-10-23" and out["results"]["AAPL"]["ret_5d"]["value"] == pytest.approx(139.5 / 134.5 - 1) and out["bars"] == {"AAPL": 40}
    json.dumps(out)                                                     # a tool result goes to the model as JSON
    tool = {t.name: t for t in tools.langchain_tools()}["calculate_metrics"]
    assert list(tool.args) == ["symbols", "metrics"] and "rates NOTHING" in tool.description and "needs 30 bars, 17 stored" in tool.description


def test_calculate_metrics_refuses_what_the_model_must_fix(bars_root):
    tools = DataClerkTools(toolbox=Toolbox(), bars_root=bars_root)
    with pytest.raises(ToolInputError, match="unknown metrics: nope; known metrics: ret_1d"):
        tools.calculate_metrics(["AAPL"], ["nope"])
    with pytest.raises(ToolInputError, match="at most 10 tickers"):
        tools.calculate_metrics([f"S{i}" for i in range(11)], ["ret_5d"])
    with pytest.raises(ToolInputError, match="not a valid ticker symbol"):
        tools.calculate_metrics(["../x"], ["ret_5d"])
    with pytest.raises(ToolInputError, match="needs at least one word"):
        tools.calculate_metrics(["AAPL"], [])
    with pytest.raises(ToolInputError, match="at least one ticker"):
        tools.calculate_metrics([], ["ret_5d"])
