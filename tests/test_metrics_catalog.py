"""Tests for prometheus.agents.data_clerk.metrics_catalog (Advisor v1 spec section 4.2 and 4.3, phase 2).

No network. Bars come from literal series written as day files in the real format (arithmetic inputs, not market data); the financial payload is Finnhub's field names fed
through the real `parse_metric_payload`; the filing evidence is the real Apple one read from the real stored pages.
"""
from datetime import date

import pytest
from test_filing_signals import APPLE_TODAY, store

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk import filing_signals as fs
from prometheus.agents.data_clerk import metrics_catalog as catalog
from prometheus.agents.data_clerk import technicals as tech
from prometheus.agents.data_clerk.bars import BarStore
from prometheus.agents.data_clerk.data_toolbox import FINANCIAL_METRICS, parse_metric_payload

TODAY = date(2026, 10, 23)
KEYS = {"value", "unit", "as_of", "window", "source", "inputs_summary", "note"}
FINNHUB = {"peTTM": 25.0, "epsTTM": 6.0, "roeTTM": 30.0, "operatingMarginTTM": 20.0, "totalDebt/totalEquityQuarterly": 1.1, "currentRatioQuarterly": 1.4, "revenueGrowthTTMYoy": 5.0,
           "pbAnnual": 9.0}   # a few of Finnhub's own field names


def write_series(root, n, symbols=("AAA", "SPY", "XLK")):
    days = weekdays(n, TODAY)
    rows = {}
    for i, day in enumerate(days):
        rows[day] = [{"ticker": t, "open": 100 + i + k, "high": 101.5 + i + k, "low": 99 + i + k, "close": 100.5 + i + k, "volume": 1000 + i} for k, t in enumerate(symbols)]
    write_day_files(root, rows)
    return BarStore(root).load([*symbols], end=TODAY)


@pytest.fixture
def apple_evidence(tmp_path_factory, monkeypatch, capsys):
    tmp = tmp_path_factory.mktemp("apple")
    filings, _ = store(tmp, monkeypatch, "AAPL", "filings_sample.txt", "edgar_aapl_10q_0000320193-26-000006")
    capsys.readouterr()
    return fs.recommendation_evidence("AAPL", filings, today=APPLE_TODAY)


def all_requests():
    return [{"name": n} for n in catalog.METRIC_NAMES]


def test_the_catalogue_is_complete_and_consistent():
    assert len(catalog.METRIC_NAMES) == len(set(catalog.METRIC_NAMES))
    for name, spec in catalog.METRICS.items():
        assert spec.name == name and callable(spec.fn) and spec.unit and spec.window and spec.description, name
        assert set(spec.inputs) <= {"bars", "financials", "evidence", "splits"}, name      # "splits" (Critic v3): `close_history` is put into the share basis of the last day by the stored split table
    for name in tech.MIN_BARS:                                           # every indicator of the technicals module is in the catalogue (the ones that are not names of their own)
        assert name in catalog.METRICS or name in ("atr_14",) or name.startswith(("ret_", "sma_", "above_sma")), name
    assert set(FINANCIAL_METRICS) <= set(catalog.METRIC_NAMES)           # every Finnhub field kept is a metric
    assert set(catalog.UNAVAILABLE) <= set(catalog.METRIC_NAMES)
    assert len(catalog.METRIC_NAMES) <= 100                              # work_order.MAX_METRICS


def test_compute_gives_one_result_object_per_symbol_per_requested_metric(tmp_path, apple_evidence):
    bars = write_series(tmp_path, 300)
    financials = {"AAA": parse_metric_payload("AAA", FINNHUB, "2026-10-23")}
    out = catalog.compute(["AAA", "ZZZ"], all_requests(), bars, financials, {"AAA": apple_evidence}, TODAY, {"AAA": "XLK"})
    assert set(out) == {"AAA", "ZZZ"}
    for symbol in out:
        assert list(out[symbol]) == list(catalog.METRIC_NAMES)
        for name, result in out[symbol].items():
            assert set(result) == KEYS, name
            assert result["value"] is not None or result["note"], f"{symbol} {name}: a missing value must say why"
            assert result["note"] is None or result["value"] is None, "a note is a gap in words: only on a missing value"
            assert not fs.keys_of(result) & set(catalog.FORBIDDEN_KEYS), f"{name} carries a judgement key"
    aaa = out["AAA"]
    assert aaa["rsi_14"]["value"] == 100.0 and aaa["rsi_14"]["as_of"] == "2026-10-23" and aaa["rsi_14"]["source"].startswith("day_aggs:") and aaa["rsi_14"]["source"].endswith("..2026-10-23")
    assert aaa["rsi_14"]["inputs_summary"] == {"bars": 300, "last_close": 100.5 + 299} and aaa["rsi_14"]["window"] == "last 30 bars"
    assert aaa["ret_5d"]["value"] == pytest.approx((100.5 + 299) / (100.5 + 294) - 1)
    assert aaa["mom_12_1"]["value"] == pytest.approx((100.5 + 299 - 20) / (100.5 + 48) - 1)       # close[-21] over close[-252]
    assert aaa["last_close"]["value"] == 399.5 and aaa["sector_etf"]["value"] == "XLK"
    recent = aaa["recent_bars"]["value"]                                                            # the one list-valued metric: [date, close, volume] of the last 35 bars, oldest first
    assert len(recent) == 35 and recent[-1] == ["2026-10-23", 399.5, 1000 + 299] and recent[0][0] < recent[1][0] < recent[-1][0]
    for name in catalog.UNAVAILABLE:                                                                # the gaps this build has are results with a reason, not missing keys
        assert aaa[name]["value"] is None and aaa[name]["note"] == catalog.UNAVAILABLE[name]
    zzz = out["ZZZ"]                                                                                # a symbol with no bars: every bar metric says so
    assert zzz["rsi_14"]["value"] is None and zzz["rsi_14"]["note"] == "needs 30 bars, 0 stored" and zzz["last_close"]["note"] == "needs 1 bars, 0 stored"


def test_sector_and_market_metrics_need_their_etf_and_spy(tmp_path):
    bars = write_series(tmp_path, 300)
    names = ("sector_etf", "sector_ret_5d", "residual_5d", "residual_mom_12_1", "spy_ret_5d", "spy_above_sma_200", "beta_63d", "corr_63d_spy")
    without = catalog.compute(["AAA"], [{"name": n} for n in names], bars, {}, {}, TODAY)["AAA"]
    assert without["sector_etf"]["note"] == "no sector ETF given for this symbol (WorkOrder.sector)" and without["residual_5d"]["value"] is None and without["spy_ret_5d"]["value"] is not None
    with_etf = catalog.compute(["AAA"], [{"name": n} for n in names], bars, {}, {}, TODAY, {"AAA": "XLK"})["AAA"]
    own, etf = (100.5 + 299) / (100.5 + 294) - 1, (100.5 + 2 + 299) / (100.5 + 2 + 294) - 1     # AAA and the third symbol (XLK) in write_series
    assert with_etf["sector_etf"]["value"] == "XLK" and with_etf["residual_5d"]["value"] == pytest.approx(own - etf)
    assert with_etf["sector_ret_5d"]["value"] == pytest.approx(etf) and with_etf["residual_5d"]["inputs_summary"]["etf"] == "XLK"
    assert with_etf["beta_63d"]["value"] is not None and with_etf["corr_63d_spy"]["value"] == pytest.approx(1.0, abs=1e-3)   # two ramps move together
    no_etf_bars = catalog.compute(["AAA"], [{"name": "sector_ret_5d"}], bars, {}, {}, TODAY, {"AAA": "XLE"})["AAA"]      # XLE is not in the files
    assert no_etf_bars["sector_ret_5d"]["note"] == "needs 2 bars of XLE, 0 stored"


def test_unknown_metric_names_are_refused_with_the_known_names(tmp_path):
    bars = write_series(tmp_path, 30)
    with pytest.raises(ValueError, match=r"unknown metrics: nope; known metrics: ret_1d"):
        catalog.compute(["AAA"], [{"name": "nope"}], bars, {}, {}, TODAY)


def test_the_sector_etf_keys_are_the_eleven_spdr_funds():
    assert catalog.sector_etfs() == ("XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")


def test_every_quantity_is_in_base_units_so_companies_compare_like_with_like():
    """Money is in dollars: no unit of the catalogue is a scaled one (millions, thousands, billions)."""
    import re
    for name, spec in catalog.METRICS.items():
        assert not re.search(r"\b(?:M|K|B|millions?|thousands?|billions?)\b", spec.unit), f"{name} has the scaled unit {spec.unit!r}"
