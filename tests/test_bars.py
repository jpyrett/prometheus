"""Tests for prometheus.agents.data_clerk.bars (Advisor v1 spec section 4.3, phase 1).

The store is fed day files in the real `day_aggs_v1` format written into a tmp_path, and, when the real files are on this machine, the real October ones (read only).
"""
import gzip
from datetime import date
from pathlib import Path

import pytest

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk import technicals
from prometheus.agents.data_clerk.bars import Bar, BarStore, date_of

REAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "historical" / "day_aggs_v1"


def rows(day_index, **by_ticker):
    return [{"ticker": t, "open": 10 + day_index, "high": 12 + day_index, "low": 9 + day_index, "close": 11 + day_index, "volume": 1000 + day_index} for t in by_ticker]


def three_days(tmp_path, skip=()):
    days = weekdays(3, date(2026, 10, 7))
    write_day_files(tmp_path, {d: rows(i, AAA=1, BBB=1) for i, d in enumerate(days)}, skip=skip)
    return days


def test_window_start_nanoseconds_become_the_trading_date():
    assert date_of(1792728000000000000) == date(2026, 10, 23)    # the real stamp of 2026-10-23 (04:00 UTC)
    assert date_of(1759291200000000000) == date(2025, 10, 1)


def test_it_reads_the_files_sorted_and_keeps_only_the_symbols_asked_for(tmp_path):
    days = three_days(tmp_path)
    store = BarStore(tmp_path).load(["aaa"])           # case does not matter
    bars = store.bars("AAA")
    assert [b.date for b in bars] == days and store.bars("BBB") == [] and store.gaps == []
    assert bars[1] == Bar(days[1], 11.0, 13.0, 10.0, 12.0, 1001.0, 100)
    assert store.dates == days and store.loaded == {"AAA"}


def test_end_and_max_bars_limit_what_is_returned(tmp_path):
    days = three_days(tmp_path)
    store = BarStore(tmp_path).load(["AAA"], end=days[1])
    assert [b.date for b in store.bars("AAA")] == days[:2]
    full = BarStore(tmp_path).load(["AAA"])
    assert [b.date for b in full.bars("AAA", max_bars=2)] == days[1:]
    assert [b.date for b in full.bars("AAA", end=days[0])] == days[:1]


def test_a_missing_weekday_between_the_first_and_last_file_is_a_gap_and_no_bar_is_invented(tmp_path):
    days = three_days(tmp_path, skip=(weekdays(3, date(2026, 10, 7))[1],))
    store = BarStore(tmp_path).load(["AAA"])
    assert len(store.bars("AAA")) == 2
    assert store.gaps == [f"{weekdays(3, date(2026, 10, 7))[1]}: no file (a market holiday, or not synced)"]
    assert days[1] not in [b.date for b in store.bars("AAA")]


def test_a_corrupt_file_is_reported_with_its_day_and_the_rest_is_still_read(tmp_path):
    days = three_days(tmp_path)
    bad = tmp_path / f"{days[1]:%Y}" / f"{days[1]:%m}" / f"{days[1]}.csv.gz"
    with gzip.open(bad, "wt") as handle:
        handle.write("not,the,right,columns\n1,2,3,4\n")
    store = BarStore(tmp_path).load(["AAA"])
    assert [b.date for b in store.bars("AAA")] == [days[0], days[2]]
    assert len(store.gaps) == 1 and store.gaps[0].startswith(f"{days[1]}: unreadable (")
    assert "not,the" not in store.gaps[0]               # the file's content is never put in a message


def test_no_files_is_one_gap_and_no_bars(tmp_path):
    store = BarStore(tmp_path / "nothing").load(["AAA"])
    assert store.bars("AAA") == [] and len(store.gaps) == 1 and "no day_aggs_v1 files" in store.gaps[0]


def test_it_never_writes_under_the_root(tmp_path):
    three_days(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    BarStore(tmp_path).load(["AAA", "BBB"])
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="the real day_aggs_v1 files are not on this machine")
def test_the_real_files_give_bars_and_only_the_indicators_that_fit():
    """Phase 1 acceptance: with the real stored sessions (17 in October 2026 when this was written) an indicator answers exactly when it has enough bars."""
    store = BarStore(REAL_ROOT).load(["SPY", "PG"])
    bars = store.bars("PG")
    assert len(bars) >= 10 and all(a.date < b.date for a, b in zip(bars, bars[1:]))
    assert (technicals.rsi_14(bars) is None) == (len(bars) < technicals.MIN_BARS["rsi_14"])
    assert technicals.ret(bars, 5) is not None


# ---------------------------------------------------------------- the narrow sync (no network: the aws call is recorded)

def test_sync_day_aggs_syncs_only_the_day_aggregates_of_the_months_asked_for(tmp_path, monkeypatch):
    from prometheus.agents.data_clerk.data_toolbox import DataToolbox
    box = DataToolbox(base_dir=tmp_path)
    calls = []
    monkeypatch.setattr(box, "_run", lambda cmd, cwd=None, env=None: calls.append((cmd, cwd)))
    folders = box.sync_day_aggs(["2025-12", "2026-01"])
    assert calls[0][0] == ["aws", "configure", "set", "default.s3.preferred_transfer_client", "classic"]            # the classic client, in a temporary config
    calls = calls[1:]
    assert folders == [tmp_path / "data" / "historical" / "day_aggs_v1" / "2025" / "12", tmp_path / "data" / "historical" / "day_aggs_v1" / "2026" / "01"] and all(f.is_dir() for f in folders)
    assert [c[0] for c in calls] == [["aws", "s3", "sync", "s3://flatfiles/us_stocks_sip/day_aggs_v1/2025/12", ".", "--endpoint-url", "https://files.massive.com"],
                                     ["aws", "s3", "sync", "s3://flatfiles/us_stocks_sip/day_aggs_v1/2026/01", ".", "--endpoint-url", "https://files.massive.com"]]
    assert [c[1] for c in calls] == folders and not any("minute_aggs" in " ".join(c[0]) or "trades" in " ".join(c[0]) or "quotes" in " ".join(c[0]) or "--delete" in c[0] for c in calls)


@pytest.mark.parametrize("months", [[], ["2026-13"], ["2026-1"], ["26-10"], ["2026/10"], ["../2026-10"], ["1999-01"], ["x"], ["2026-10"] * 37])
def test_sync_day_aggs_refuses_anything_that_is_not_a_month(tmp_path, monkeypatch, months):
    from prometheus.agents.data_clerk.data_toolbox import DataToolbox
    box = DataToolbox(base_dir=tmp_path)
    monkeypatch.setattr(box, "_run", lambda cmd, cwd=None, env=None: pytest.fail("nothing may be synced"))
    with pytest.raises(ValueError):
        box.sync_day_aggs(months)
    assert not (tmp_path / "data").exists()


def test_the_sync_script_prints_its_plan_and_does_nothing_without_yes(capsys, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_day_aggs", Path(__file__).resolve().parent.parent / "scripts" / "sync_day_aggs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.months_between("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    monkeypatch.setattr("prometheus.agents.data_clerk.data_toolbox.DataToolbox.sync_day_aggs", lambda self, months: pytest.fail("not without --yes"))
    assert module.main(["--from", "2025-09", "--to", "2026-11"]) == 0
    assert "15 month(s), 2025-09 to 2026-11" in capsys.readouterr().out


# ---------------------------------------------------------------- the REST refresh and its merge with the day files

def rest_row(day, close, volume=1000.0):
    from datetime import datetime, timezone
    stamp = int(datetime(day.year, day.month, day.day, 4, tzinfo=timezone.utc).timestamp() * 1000)   # Massive's `t` is milliseconds
    return {"t": stamp, "o": close - 1, "h": close + 1, "l": close - 2, "c": close, "v": volume, "n": 77}


def toolbox_with(tmp_path, monkeypatch, results_by_symbol):
    from prometheus.agents.data_clerk.data_toolbox import DataToolbox
    box = DataToolbox(base_dir=tmp_path)
    calls = []
    monkeypatch.setattr(box, "_api_keys", lambda env=None: {"FINNHUB_API_KEY": "x", "POLYGON_API_KEY": "SECRET-KEY"})

    def fake_get(client, limiter, name, url, params, headers=None):
        calls.append((url, dict(params)))
        symbol = url.split("/ticker/")[1].split("/")[0]
        return {"results": results_by_symbol.get(symbol, [])}
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(fake_get))
    return box, calls


def test_fetch_daily_bars_writes_one_file_per_ticker_in_the_day_file_columns(tmp_path, monkeypatch):
    days = weekdays(3, date(2026, 11, 4))
    box, calls = toolbox_with(tmp_path, monkeypatch, {"PG": [rest_row(d, 160.0 + i) for i, d in enumerate(days)]})
    info = box.fetch_daily_bars(["pg", "$PG", "XLP"], days=30, end=date(2026, 11, 4))
    assert info == {"PG": {"fetched": 3, "stored": 3, "first": str(days[0]), "last": str(days[-1])}, "XLP": {"fetched": 0, "stored": 0, "first": None, "last": None}}
    assert [c[0] for c in calls] == ["https://api.massive.com/v2/aggs/ticker/PG/range/1/day/2026-10-05/2026-11-04", "https://api.massive.com/v2/aggs/ticker/XLP/range/1/day/2026-10-05/2026-11-04"]
    assert calls[0][1]["adjusted"] == "false" and calls[0][1]["sort"] == "asc"                    # unadjusted, like the flat files
    text = (tmp_path / "data" / "historical" / "ticker_bars_v1" / "PG.csv").read_text().splitlines()
    assert text[0] == "ticker,volume,open,close,high,low,window_start,transactions" and len(text) == 4 and text[1].startswith("PG,1000.0,159.0,160.0,161.0,158.0,") and "SECRET-KEY" not in "\n".join(text)
    store = BarStore(tmp_path / "data" / "historical" / "day_aggs_v1").load(["PG"])             # BarStore reads it: no day files at all
    assert [b.date for b in store.bars("PG")] == days and store.bars("PG")[-1].close == 162.0 and store.gaps == []


def test_a_second_fetch_merges_a_date_fetched_again_replaces_the_old_row_and_nothing_is_lost(tmp_path, monkeypatch):
    days = weekdays(3, date(2026, 11, 4))
    box, _ = toolbox_with(tmp_path, monkeypatch, {"PG": [rest_row(d, 160.0) for d in days[:2]]})
    box.fetch_daily_bars(["PG"], 30, date(2026, 11, 4))
    box2, _ = toolbox_with(tmp_path, monkeypatch, {"PG": [rest_row(days[1], 170.0), rest_row(days[2], 171.0)]})
    info = box2.fetch_daily_bars(["PG"], 30, date(2026, 11, 4))
    assert info["PG"]["stored"] == 3 and info["PG"]["first"] == str(days[0])
    store = BarStore(tmp_path / "day_aggs_v1", extra_root=tmp_path / "data" / "historical" / "ticker_bars_v1").load(["PG"])
    assert [b.close for b in store.bars("PG")] == [160.0, 170.0, 171.0]


@pytest.mark.parametrize("symbols, days", [([], 30), (["../x"], 30), (["AAPL;rm"], 30), (["PG"], 0), (["PG"], 801), (["PG"], True)])
def test_fetch_daily_bars_refuses_bad_input_before_any_call(tmp_path, monkeypatch, symbols, days):
    box, calls = toolbox_with(tmp_path, monkeypatch, {})
    with pytest.raises(ValueError):
        box.fetch_daily_bars(symbols, days)
    assert calls == [] and not (tmp_path / "data").exists()


def test_the_rest_bars_add_the_dates_the_day_files_lack_and_the_day_file_wins_where_both_have_one(tmp_path):
    root, extra = tmp_path / "day_aggs_v1", tmp_path / "ticker_bars_v1"
    days = weekdays(5, date(2026, 10, 30))
    write_day_files(root, {d: rows(i, AAA=1) for i, d in enumerate(days[:3])})                    # the day files stop on the third day
    extra.mkdir()
    import csv
    from datetime import datetime, timezone
    with open(extra / "AAA.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"])
        for i, d in enumerate(days):                                                                # the REST file has all five, with a different close for the days the day files have
            writer.writerow(["AAA", 5.0, 5.0, 999.0 + i, 5.0, 5.0, int(datetime(d.year, d.month, d.day, 4, tzinfo=timezone.utc).timestamp() * 1e9), 1])
    store = BarStore(root, extra_root=extra).load(["AAA"])
    bars = store.bars("AAA")
    assert [b.date for b in bars] == days and bars[0].close == 11.0 and bars[2].close == 13.0 and bars[3].close == 1002.0 and bars[4].close == 1003.0    # the day file's closes for the first three
    assert store.gaps == [] and store.dates == days
    assert [b.date for b in BarStore(root, extra_root=extra).load(["AAA"], end=days[3]).bars("AAA")] == days[:4]                # `end` applies to the REST bars too


def test_a_missing_weekday_is_still_a_gap_when_no_source_has_it_and_a_damaged_rest_file_is_named(tmp_path):
    root, extra = tmp_path / "day_aggs_v1", tmp_path / "ticker_bars_v1"
    days = weekdays(4, date(2026, 10, 30))
    write_day_files(root, {d: rows(i, AAA=1) for i, d in enumerate(days)}, skip=(days[1],))
    extra.mkdir()
    (extra / "AAA.csv").write_text("not,the,columns\n1,2,3\n")
    store = BarStore(root, extra_root=extra).load(["AAA"])
    assert len(store.bars("AAA")) == 3 and f"{days[1]}: no file (a market holiday, or not synced)" in store.gaps and any(g.startswith("AAA: the REST bar file is unreadable (") for g in store.gaps)


def test_the_fetch_script_prints_its_plan_with_spy_and_the_etfs_and_does_nothing_without_yes(capsys, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_bars", Path(__file__).resolve().parent.parent / "scripts" / "fetch_bars.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("prometheus.agents.data_clerk.data_toolbox.DataToolbox.fetch_daily_bars", lambda self, symbols, days: pytest.fail("not without --yes"))
    assert module.main(["--symbols", "sndk", "$PG"]) == 0
    out = capsys.readouterr().out
    assert "14 tickers" in out and "SNDK, PG, SPY, XLK" in out and "not run" in out
