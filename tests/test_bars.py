"""Tests for prometheus.agents.data_clerk.bars (Advisor v1 spec section 4.3, phase 1).

The store is fed day files in the real `day_aggs_v1` format written into a tmp_path, and, when the real files are on this machine, the real October ones (read only).
"""
import gzip
from datetime import date
from pathlib import Path

import pytest

from bar_helpers import weekdays, write_day_files
from prometheus.agents.data_clerk.bars import BarStore


def rows(day_index, **by_ticker):
    return [{"ticker": t, "open": 10 + day_index, "high": 12 + day_index, "low": 9 + day_index, "close": 11 + day_index, "volume": 1000 + day_index} for t in by_ticker]


def three_days(tmp_path, skip=()):
    days = weekdays(3, date(2026, 10, 7))
    write_day_files(tmp_path, {d: rows(i, AAA=1, BBB=1) for i, d in enumerate(days)}, skip=skip)
    return days


def test_a_corrupt_file_is_reported_with_its_day_and_the_rest_is_still_read(tmp_path):
    days = three_days(tmp_path)
    bad = tmp_path / f"{days[1]:%Y}" / f"{days[1]:%m}" / f"{days[1]}.csv.gz"
    with gzip.open(bad, "wt") as handle:
        handle.write("not,the,right,columns\n1,2,3,4\n")
    store = BarStore(tmp_path).load(["AAA"])
    assert [b.date for b in store.bars("AAA")] == [days[0], days[2]]
    assert len(store.gaps) == 1 and store.gaps[0].startswith(f"{days[1]}: unreadable (")
    assert "not,the" not in store.gaps[0]               # the file's content is never put in a message


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
