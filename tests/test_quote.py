"""Tests for DataToolbox.get_quote, the Data Clerk's current-price lookup (design choices entry 42).

Run:  pytest tests/test_quote.py -q

`_get_json` (the one function that touches the network) is replaced, so no test calls Finnhub, uses a key or costs anything. The Finnhub answer used is the shape of its /quote
endpoint: c current price, d change, dp change percent, h high, l low, o open, pc previous close, t the time of the last trade (unix seconds).
"""
import pytest

from prometheus.agents.data_clerk import data_toolbox
from prometheus.agents.data_clerk.data_toolbox import DataToolbox

FINNHUB = {"c": 410.25, "d": 2.5, "dp": 0.6131, "h": 412.0, "l": 405.0, "o": 406.0, "pc": 407.75, "t": 1793739600}  # 2026-11-03T21:00:00Z


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    asked = []
    answer = {"value": FINNHUB}

    def fake_get_json(client, limiter, name, url, params, headers=None):
        asked.append((name, url, params, sorted(headers or {})))  # the real code never puts the key in the url: it is a header
        return answer["value"]

    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-not-real")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key-not-real")
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(fake_get_json))
    box = DataToolbox(base_dir=tmp_path)
    box.asked, box.answer = asked, answer
    return box


def test_a_quote_is_one_finnhub_call_with_the_key_in_a_header(toolbox):
    quote = toolbox.get_quote(" mu ")
    name, url, params, headers = toolbox.asked[0]
    assert len(toolbox.asked) == 1 and url == data_toolbox.FINNHUB_QUOTE_URL == "https://finnhub.io/api/v1/quote" and params == {"symbol": "MU"}
    assert headers == ["X-Finnhub-Token"] and "token" not in str(params) and quote["symbol"] == "MU"


def test_the_quote_has_the_price_the_change_the_range_and_when_it_last_traded(toolbox):
    assert toolbox.get_quote("MU") == {"symbol": "MU", "price": 410.25, "change": 2.5, "change_percent": 0.6131, "high": 412.0, "low": 405.0, "open": 406.0,
                                       "previous_close": 407.75, "as_of_utc": "2026-11-03T21:00:00Z", "source": "Finnhub /quote"}


def test_an_unknown_symbol_is_an_error_not_a_price_of_zero(toolbox):
    toolbox.answer["value"] = {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}  # what Finnhub answers for a symbol it does not know
    with pytest.raises(RuntimeError, match="No quote returned for ZZZZ"):
        toolbox.get_quote("ZZZZ")


@pytest.mark.parametrize("answer", [[], None, {}, {"c": 410.0}, {"c": 0, "t": 1793739600}])
def test_an_answer_without_a_price_or_a_time_is_an_error(toolbox, answer):
    toolbox.answer["value"] = answer
    with pytest.raises(RuntimeError, match="No quote returned"):
        toolbox.get_quote("MU")


@pytest.mark.parametrize("symbol", ["", "   ", None])
def test_a_blank_symbol_is_refused_before_any_call(toolbox, symbol):
    with pytest.raises(ValueError, match="symbol is required"):
        toolbox.get_quote(symbol)
    assert toolbox.asked == []


def test_a_missing_key_is_named_and_nothing_is_called(tmp_path, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(lambda *a, **k: pytest.fail("no call without a key")))
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY is not set"):
        DataToolbox(base_dir=tmp_path).get_quote("MU")
