"""Tests for DataClerk.request_metrics and DataClerkTools.calculate_metrics (Advisor v1 spec section 4.1, phase 2, design choices entry 51).

No network: a fake toolbox stands in for Finnhub, the bars are day files written into a temporary folder in the real format, and the stored filings are the real Apple ones.
"""
from datetime import date

import pytest
from test_filing_signals import APPLE_TODAY, store

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import work_order as wo
from prometheus.tools.data_clerk import DataClerkTools, ToolInputError

TODAY = date(2026, 10, 23)
FINNHUB = {"peTTM": 25.0, "roeTTM": 30.0}


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
    result = clerk(bars_root, toolbox).request_metrics(order(["aapl", "$PG", "AAPL"], "ret_5d", "pe_ratio", "share_count_change", "rsi_14"), today=TODAY)
    assert result.status == "data_ready" and result.errors == [] and toolbox.financials.asked == ["AAPL", "PG"]      # cleaned, once each
    m = result.data.metrics
    assert list(m) == ["AAPL", "PG"] and list(m["AAPL"]) == ["ret_5d", "pe_ratio", "share_count_change", "rsi_14"]
    assert m["AAPL"]["ret_5d"]["value"] == pytest.approx(139.5 / 134.5 - 1) and m["AAPL"]["pe_ratio"]["value"] == 25.0
    assert m["AAPL"]["share_count_change"]["value"] < 0
    assert m["AAPL"]["rsi_14"]["value"] == 100.0
    assert m["PG"]["share_count_change"]["value"] is None and m["PG"]["share_count_change"]["note"] == "no stored filings of PG (fetch_filings has not been run for it)"
    assert result.data.tool_calls == [{"name": "calculate_metrics", "args": {"symbols": 2, "metrics": 4}, "outcome": "ok"}]
    assert any(n.startswith("PG: no filings are stored") for n in result.data.notes)                                  # what could not be read for PG is a note, not a failure
    assert wo.WorkOrder.model_validate_json(result.model_dump_json()) == result                                       # plain JSON


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


# ---------------------------------------------------------------- the tool the model and the MCP server see


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
