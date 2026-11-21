"""The bar index (prometheus/agents/data_clerk/bar_index.py, design/proving_ground_spec.md 7.6): the same bars as the files, from one small file per symbol, and stale the moment a source changes. The proof is EQUALITY on the REAL stored bars:
the real bar folder for the symbols of the oil requests, and a copy of real day files (a folder of the test's own) for the invalidation, so nothing under data/historical is touched."""
import gzip
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

from prometheus.agents.data_clerk import bar_index
from prometheus.agents.data_clerk.bars import DEFAULT_ROOT, BarStore, _day_files


def read_files(root, symbols, **options) -> BarStore:
    """What `BarStore.load` gives from the FILES: the index is switched off for the call."""
    os.environ["PROMETHEUS_BAR_INDEX"] = "off"
    try:
        return BarStore(root).load(symbols, **options)
    finally:
        os.environ.pop("PROMETHEUS_BAR_INDEX", None)


def same(a: BarStore, b: BarStore, symbols) -> None:
    assert a.dates == b.dates and a.gaps == b.gaps and a.loaded == b.loaded
    for symbol in symbols:
        assert a.bars(symbol) == b.bars(symbol), symbol


def copy_of_real_days(tmp_path: Path, count: int = 12) -> Path:
    """A folder of the test's own with the last `count` REAL day files (the same layout) and a `ticker_bars_v1` beside it with the real REST file of XOM."""
    root = tmp_path / "historical" / "day_aggs_v1"
    for day, path in sorted(_day_files(DEFAULT_ROOT).items())[-count:]:
        target = root / path.relative_to(DEFAULT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (tmp_path / "historical" / "ticker_bars_v1").mkdir()
    shutil.copy2(DEFAULT_ROOT.parent / "ticker_bars_v1" / "XOM.csv", tmp_path / "historical" / "ticker_bars_v1" / "XOM.csv")
    return root


def indexed(root: Path, index_root: Path, symbols, **options) -> BarStore:
    store = BarStore(root)
    store.loaded |= set(symbols)
    files = _day_files(root)
    bar_index.BarIndex(root, store.extra_root, index_root).fill(store, set(symbols), files, options.get("end"), options.get("max_bars", 300), options.get("start"))
    return store


def test_a_changed_day_file_makes_the_index_stale_and_a_new_day_is_appended(tmp_path):
    """A day file whose content changes (same name, new size) invalidates the index: the new numbers are read. A new day file (the daily sync) is added to every symbol without a rebuild. A REST file that changes is read again."""
    root, index_root = copy_of_real_days(tmp_path), tmp_path / "index"
    symbols = ["XOM", "CVX", "SPY"]
    days = sorted(_day_files(root).items())
    # (1) a new day file appears (the daily sync): the index of the days before it is brought up to date by reading that one file
    last_day, last_path = days[-1]
    held = last_path.read_bytes()
    stamp = (last_path.stat().st_atime_ns, last_path.stat().st_mtime_ns)
    last_path.unlink()
    first = indexed(root, index_root, symbols)
    same(read_files(root, symbols), first, symbols)
    assert first.bars("CVX")[-1].date < last_day
    last_path.write_bytes(held)
    os.utime(last_path, ns=stamp)
    with_it = indexed(root, index_root, symbols)
    same(read_files(root, symbols), with_it, symbols)
    assert with_it.bars("CVX")[-1].date == last_day
    # (2) a day file whose content changed: the same name, other bytes. XOM's close of that day is made 1.0 higher
    text = gzip.decompress((root / days[3][1].relative_to(root)).read_bytes()).decode()
    changed = days[3][1]
    rows = text.splitlines()
    fields = rows[0].split(",")
    at = {name: i for i, name in enumerate(fields)}
    for n, row in enumerate(rows[1:], 1):
        parts = row.split(",")
        if parts[at["ticker"]] == "XOM":
            parts[at["close"]] = str(float(parts[at["close"]]) + 1.0)
            rows[n] = ",".join(parts)
    changed.write_bytes(gzip.compress("\n".join(rows).encode() + b"\n"))
    after = indexed(root, index_root, symbols)
    same(read_files(root, symbols), after, symbols)
    assert after.bars("XOM") != with_it.bars("XOM")
    # (3) a REST file that changed
    rest = tmp_path / "historical" / "ticker_bars_v1" / "XOM.csv"
    rest.write_text(rest.read_text() + "")
    os.utime(rest, ns=(rest.stat().st_atime_ns, rest.stat().st_mtime_ns + 5_000_000_000))
    same(read_files(root, symbols), indexed(root, index_root, symbols), symbols)


def test_a_symbol_added_later_is_built_on_first_use_and_the_index_is_for_the_real_folder_only(tmp_path):
    root, index_root = copy_of_real_days(tmp_path), tmp_path / "index"
    indexed(root, index_root, ["XOM"])
    assert {p.stem for p in index_root.glob("*.parquet")} == {"XOM"}
    both = indexed(root, index_root, ["XOM", "CVX"])
    assert {p.stem for p in index_root.glob("*.parquet")} == {"XOM", "CVX"}
    same(read_files(root, ["XOM", "CVX"]), both, ["XOM", "CVX"])
    assert bar_index.index_for(root, root.parent / "ticker_bars_v1") is None and bar_index.index_for(DEFAULT_ROOT, DEFAULT_ROOT.parent / "ticker_bars_v1") is not None
    with pytest.raises(ValueError):
        bar_index.BarIndex(root, root.parent / "ticker_bars_v1", index_root)._symbol_path("../etc")


