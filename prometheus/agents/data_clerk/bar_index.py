"""The bar index: the stored daily bars of each symbol in ONE small Parquet file (design/proving_ground_spec.md section 7.6, P14; design choices entry 126).

    BarStore.load(...) uses it by itself: for the real bar folder the answer comes from `data/index/bars_v1/<SYMBOL>.parquet` and the files of 300 days are not read at all.
    python -m prometheus.proving index          # builds it for every symbol with stored news, SPY and the sector ETFs, ahead of time

WHY. `BarStore.load` reads up to 300 gzip day files of every US ticker for each call, and the Advisor, the Critic and the Trader each call it for each decision day: most of the non-model time of a test. The index holds, for each symbol, EXACTLY the
bars `BarStore` would give from the day files and the per-ticker REST files (`ticker_bars_v1/<SYMBOL>.csv`), and `BarStore.load` uses it when it can prove it is current. The numbers are the same, the files are read once.

WHAT IS IN IT. Per symbol, every bar of every day file (`src` "d") and every row of its REST file (`src` "r"), with the date, open, high, low, close, volume and transactions as the files hold them, sorted by date; `load` then does what it always did with
them: the last `max_bars` day files on or before `end` (and from `start`), the REST bars of the dates the day files of that range lack (THE DAY FILE WINS), the same sort, the same `dates` and the same `gaps`. A manifest (`manifest.json`) names every
day file that was read (its size and its modification time in nanoseconds), the symbols built, each REST file's size and time, and the day files that could not be read; so the index is current only when the files are as the manifest says.
 - A NEW day file (the daily sync) is read on its own and appended to every symbol of the index (incremental).
 - A day file that CHANGED or is gone (its size or its time differs) makes the whole index stale: it is deleted and rebuilt, symbol by symbol, as they are asked for. A REST file that changed is read again for its symbol alone.
 - A symbol that is not in the index is built on first use: ONE pass over the day files for all the symbols asked for in that call (the days are read by a few threads: the reader releases the GIL).
 - It is used for the real bar folder only (`data/historical/day_aggs_v1`): a `BarStore` over any other folder (a test's own) reads its files as always. `PROMETHEUS_BAR_INDEX=off` turns it off. Any failure of the index falls back to the files: the files are the truth.
It is derived data, under `data/index/` (gitignored), never under `data/historical/`: nothing the desk keeps is written. A write is atomic (a temp file, then a rename) and one lock covers a build, so the parallel research of the Time Machine builds each file once.
"""
import contextvars
import json
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq

from prometheus.rag.config import REPO_ROOT, setting

VERSION = 2      # 2: a bar is read only once its day has settled (bars.SETTLE_MINUTES, entry 129); an index built before that rule is rebuilt
INDEX_ROOT = REPO_ROOT / "data" / "index" / "bars_v1"
REAL_ROOT = REPO_ROOT / "data" / "historical" / "day_aggs_v1"
THREADS = 8
SCHEMA = pa.schema([("date", pa.date32()), ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()), ("volume", pa.float64()), ("transactions", pa.int64()), ("src", pa.string())])
_LOCK = threading.RLock()
SAFE_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,11}")


def _stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


class BarIndex:
    def __init__(self, root: Path, extra_root: Path, index_root: Path = INDEX_ROOT):
        self.root, self.extra_root, self.index_root = Path(root), Path(extra_root), Path(index_root)
        self.manifest_path = self.index_root / "manifest.json"

    # ---------------------------------------------------------------- the manifest

    def _read_manifest(self) -> dict | None:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = isinstance(manifest, dict) and manifest.get("version") == VERSION and manifest.get("source") == str(self.root.resolve()) and manifest.get("extra") == str(self.extra_root.resolve())
        return manifest if ok and isinstance(manifest.get("days"), dict) and isinstance(manifest.get("symbols"), dict) and isinstance(manifest.get("unreadable"), dict) else None

    def _fresh_manifest(self) -> dict:
        return {"version": VERSION, "source": str(self.root.resolve()), "extra": str(self.extra_root.resolve()), "days": {}, "unreadable": {}, "symbols": {}}

    def _write(self, path: Path, data: bytes | str | pa.Table) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(handle)
        try:
            if isinstance(data, pa.Table):
                pq.write_table(data, temp)
            else:
                Path(temp).write_text(data, encoding="utf-8")
            os.replace(temp, path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

    def _save(self, manifest: dict) -> None:
        self._write(self.manifest_path, json.dumps(manifest, indent=1, sort_keys=True))

    def _symbol_path(self, symbol: str) -> Path:
        if SAFE_SYMBOL.fullmatch(symbol) is None:
            raise ValueError(f"{symbol!r} is not a symbol the index keeps")
        return self.index_root / f"{symbol}.parquet"

    def reset(self) -> None:
        """Remove the index (derived data): after a failure the files are read, and the index is built again the next time."""
        with _LOCK:
            self._wipe()

    def _wipe(self) -> None:
        """The index is stale: remove what it holds (derived data, rebuilt on demand). Only files this module wrote."""
        for path in self.index_root.glob("*.parquet"):
            path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)

    # ---------------------------------------------------------------- reading the sources

    @staticmethod
    def _read_day(path: Path, symbols: set[str]) -> dict[str, list[tuple]]:
        """{symbol: [(date, open, high, low, close, volume, transactions), ...]} of the wanted symbols in one day file: the rows `BarStore` reads."""
        from prometheus.agents.data_clerk.bars import COLUMNS, date_of
        table = pv_csv.read_csv(path, convert_options=pv_csv.ConvertOptions(include_columns=list(COLUMNS)))
        table = table.filter(pc.is_in(table["ticker"], value_set=pa.array(sorted(symbols), type=pa.string())))
        out: dict[str, list[tuple]] = {}
        for row in table.to_pylist():
            out.setdefault(row["ticker"], []).append((date_of(row["window_start"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), int(row["transactions"])))
        return out

    def _read_days(self, files: dict[date, Path], days: list[date], symbols: set[str]) -> tuple[dict[date, dict[str, list[tuple]]], dict[str, str]]:
        found: dict[date, dict[str, list[tuple]]] = {}
        broken: dict[str, str] = {}

        def one(day: date):
            try:
                return day, self._read_day(files[day], symbols), None
            except Exception as error:      # a corrupt file or a missing column: named as `BarStore` names it
                return day, None, type(error).__name__
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            for day, rows, failure in pool.map(lambda d: contextvars.copy_context().run(one, d), days):      # each read in a copy of the context, as every thread of the package is (the as-of clock follows it)
                if rows is None:
                    broken[day.isoformat()] = failure
                else:
                    found[day] = rows
        return found, broken

    def _read_rest(self, symbol: str) -> tuple[list[tuple], str | None, list[int] | None]:
        """(the REST rows of the symbol, the kind of failure, the file's [size, mtime_ns]): what `BarStore._extra_bars` reads."""
        from prometheus.agents.data_clerk.bars import COLUMNS, date_of, is_part_of_a_day
        path = self.extra_root / f"{symbol}.csv"
        if not path.exists():
            return [], None, None
        stamp = _stat(path)
        try:
            table = pv_csv.read_csv(path, convert_options=pv_csv.ConvertOptions(include_columns=list(COLUMNS)))
            rows = [(date_of(r["window_start"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]), int(r["transactions"])) for r in table.to_pylist() if r["ticker"] == symbol]
            return [row for row in rows if not is_part_of_a_day(row[0], stamp[1])], None, stamp       # as `BarStore._extra_bars`: a bar fetched before its day closed is never read (entry 129)
        except Exception as error:
            return [], type(error).__name__, stamp

    @staticmethod
    def _table(day_rows: list[tuple], rest_rows: list[tuple]) -> pa.Table:
        rows = [(*r, "d") for r in day_rows] + [(*r, "r") for r in rest_rows]
        columns = list(zip(*rows)) if rows else [[] for _ in SCHEMA]
        return pa.table({field.name: pa.array(list(columns[i]), type=field.type) for i, field in enumerate(SCHEMA)}, schema=SCHEMA)

    # ---------------------------------------------------------------- keeping it current

    def _current(self, files: dict[date, Path], wanted: set[str]) -> dict:
        """The manifest brought up to date for `wanted` (builds, appends a new day, rereads a changed REST file) and the symbols' parquet files written; returns the manifest. Holds the lock."""
        manifest = self._read_manifest()
        if manifest is not None:
            for name, (size, mtime) in manifest["days"].items():       # an indexed day file that is gone or changed: everything built from it is stale
                path = files.get(date.fromisoformat(name))
                if path is None or _stat(path) != [size, mtime]:
                    manifest = None
                    break
        if manifest is None:
            self._wipe()
            manifest = self._fresh_manifest()
        indexed = set(manifest["symbols"])
        new_days = sorted(d for d in files if d.isoformat() not in manifest["days"] and d.isoformat() not in manifest["unreadable"])
        if new_days and indexed:       # the daily sync: the new day files are read on their own and appended to every symbol of the index
            found, broken = self._read_days(files, new_days, indexed)
            for symbol in indexed:
                table = pq.read_table(self._symbol_path(symbol))
                extra = [(*row, "d") for day in new_days for row in found.get(day, {}).get(symbol, [])]
                if extra:
                    addition = pa.table({field.name: pa.array([r[i] for r in extra], type=field.type) for i, field in enumerate(SCHEMA)}, schema=SCHEMA)
                    day_part = pa.concat_tables([table.filter(pc.equal(table["src"], "d")), addition])
                    table = pa.concat_tables([day_part.sort_by("date"), table.filter(pc.equal(table["src"], "r"))])
                    self._write(self._symbol_path(symbol), table)
            for day in new_days:
                if day.isoformat() in broken:
                    manifest["unreadable"][day.isoformat()] = broken[day.isoformat()]
                else:
                    manifest["days"][day.isoformat()] = _stat(files[day])
            for name, failure in broken.items():
                manifest["unreadable"][name] = failure
            self._save(manifest)
        elif new_days:       # nothing is indexed yet: the days are recorded by the build below
            pass
        missing = sorted(wanted - set(manifest["symbols"]))
        if missing:       # ONE pass over the day files for every symbol that is missing
            every = sorted(files)
            found, broken = self._read_days(files, every, set(missing))
            for day in every:
                if day.isoformat() in broken:
                    manifest["unreadable"][day.isoformat()] = broken[day.isoformat()]
                elif day.isoformat() not in manifest["days"] and day.isoformat() not in manifest["unreadable"]:
                    manifest["days"][day.isoformat()] = _stat(files[day])
            for symbol in missing:
                day_rows = [row for day in every for row in found.get(day, {}).get(symbol, [])]
                rest_rows, rest_failure, rest_stamp = self._read_rest(symbol)
                self._write(self._symbol_path(symbol), self._table(day_rows, rest_rows))
                manifest["symbols"][symbol] = {"rest": rest_stamp, "rest_gap": rest_failure}
            self._save(manifest)
        for symbol in sorted(wanted & indexed):       # a REST file that changed is read again for its symbol alone
            path = self.extra_root / f"{symbol}.csv"
            stamp = _stat(path) if path.exists() else None
            if manifest["symbols"][symbol]["rest"] != stamp:
                rest_rows, rest_failure, rest_stamp = self._read_rest(symbol)
                table = pq.read_table(self._symbol_path(symbol))
                day_part = table.filter(pc.equal(table["src"], "d"))
                self._write(self._symbol_path(symbol), pa.concat_tables([day_part, self._table([], rest_rows)]))
                manifest["symbols"][symbol] = {"rest": rest_stamp, "rest_gap": rest_failure}
                self._save(manifest)
        return manifest

    # ---------------------------------------------------------------- the answer

    def fill(self, store, wanted: set[str], files: dict[date, Path], end: date | None, max_bars: int, start: date | None) -> None:
        """Set `store._bars`, `.dates` and `.gaps` as `BarStore.load` would from the files (see the module text). Raises when it cannot: the caller then reads the files."""
        from prometheus.agents.data_clerk.bars import Bar
        days = sorted(d for d in files if (end is None or d <= end) and (start is None or d >= start))[-max_bars:]
        with _LOCK:
            manifest = self._current(files, wanted)
        gaps: list[str] = [f"{day}: unreadable ({manifest['unreadable'][day.isoformat()]})" for day in days if day.isoformat() in manifest["unreadable"]]
        chosen = set(days)
        series: dict[str, list] = {}
        seen = set(days)
        for symbol in sorted(wanted):
            table = pq.read_table(self._symbol_path(symbol))
            columns = table.to_pydict()
            day_rows = [Bar(columns["date"][i], columns["open"][i], columns["high"][i], columns["low"][i], columns["close"][i], columns["volume"][i], columns["transactions"][i])
                        for i in range(table.num_rows) if columns["src"][i] == "d" and columns["date"][i] in chosen]
            have = {b.date for b in day_rows}
            if manifest["symbols"][symbol]["rest_gap"]:
                gaps.append(f"{symbol}: the REST bar file is unreadable ({manifest['symbols'][symbol]['rest_gap']})")
            for i in range(table.num_rows):
                if columns["src"][i] != "r":
                    continue
                bar = Bar(columns["date"][i], columns["open"][i], columns["high"][i], columns["low"][i], columns["close"][i], columns["volume"][i], columns["transactions"][i])
                if (end is None or bar.date <= end) and (start is None or bar.date >= start) and bar.date not in have:
                    day_rows.append(bar)
                    seen.add(bar.date)
            if day_rows:
                series[symbol] = day_rows
        dates = sorted(seen)[-max_bars:]
        if not dates:
            gaps.append(f"no day_aggs_v1 files on or before {end or 'today'} under {store.root}" if start is None else f"no day_aggs_v1 files from {start} to {end or 'the newest'} under {store.root}")
        else:
            day = dates[0]
            while day <= dates[-1]:
                if day.weekday() < 5 and day not in seen:
                    gaps.append(f"{day}: no file (a market holiday, or not synced)")
                day += timedelta(days=1)
            for rows in series.values():
                rows.sort(key=lambda b: b.date)
        store._bars, store.dates = {**store._bars, **{s: [*store._bars.get(s, []), *rows] for s, rows in series.items()}}, dates
        store.gaps = [*store.gaps, *gaps]


def enabled() -> bool:
    return setting("PROMETHEUS_BAR_INDEX", "on").strip().lower() not in ("off", "false", "0", "no")


def index_for(root: Path, extra_root: Path) -> BarIndex | None:
    """The index of a bar folder, or None: only the real stored bars are indexed (a test's own folder is read as always), and `PROMETHEUS_BAR_INDEX=off` turns it off."""
    if not enabled():
        return None
    try:
        return BarIndex(root, extra_root) if Path(root).resolve() == REAL_ROOT.resolve() else None
    except OSError:
        return None


def symbols_with_news() -> list[str]:
    """Every symbol with stored news (`news/<SYMBOL>.txt`), SPY and the sector ETFs: what `index` builds ahead of time."""
    from prometheus.agents.data_clerk import metrics_catalog as catalog
    names = {p.stem for p in (REPO_ROOT / "news").glob("*.txt")}
    return sorted(names | {catalog.SPY, *catalog.sector_etfs()})


def main() -> int:
    """`python -m prometheus.proving index`: build the index for every symbol with stored news (free; reads the day files once, writes under data/index/)."""
    import time
    from prometheus.agents.data_clerk.bars import BarStore
    began = time.perf_counter()
    symbols = symbols_with_news()
    store = BarStore(REAL_ROOT).load(symbols)
    ready = sum(1 for s in symbols if store.bars(s))
    print(f"bar index: {len(symbols)} symbols ({ready} with bars), {len(store.dates)} days, {INDEX_ROOT} ({time.perf_counter() - began:.1f} s)")
    return 0
