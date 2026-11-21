"""Daily bars: the stored `day_aggs_v1` files read into memory, once per run (Advisor v1 spec section 4.3, design choices entry 51).

    store = BarStore(Path("data/historical/day_aggs_v1")).load(["SNDK", "PG", "SPY"], end=date(2026, 10, 23))
    store.bars("SNDK")            # [Bar(date, open, high, low, close, volume, transactions), ...] oldest first, at most `max_bars`
    store.gaps                    # what could not be read, in words

One file per trading day, `YYYY/MM/YYYY-MM-DD.csv.gz`, every US ticker in it, columns `ticker,volume,open,close,high,low,window_start,transactions`
(`window_start` is nanoseconds since the epoch, UTC: the trading date). Plain Python, no model, no network. It READS the files and never writes under `data/historical`.

A second, optional source fills in what the day files do not have (`extra_root`, default `<root>/../ticker_bars_v1`): one `<SYMBOL>.csv` per ticker in the same columns, written by
`DataToolbox.fetch_daily_bars` from Massive's REST aggregates (the same numbers). The day files arrive in batches (a month may stop on the 23rd), and a candidate that is not in them at all is
fetched on demand. For a date that both have, the DAY FILE wins; the REST file only adds the dates the day files lack.

Honest about what is stored: a file that is missing between the first and the last stored day is named in `gaps` (a weekday without a file is a market holiday or a
day not synced yet: the store cannot tell which, so it says both); an unreadable file is named with the reason. No bar is ever invented to fill a gap.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv_csv

from prometheus.agents.data_clerk import bar_index
from prometheus.common import trading_clock

COLUMNS = ("ticker", "volume", "open", "close", "high", "low", "window_start", "transactions")
MAX_BARS = 300          # about 14 months of sessions: enough for the 252-bar metrics with room
DEFAULT_ROOT = Path("data/historical/day_aggs_v1")


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    transactions: int


def date_of(window_start_ns: int) -> date:
    """`window_start` (nanoseconds since the epoch, UTC) -> the trading date."""
    return datetime.fromtimestamp(window_start_ns / 1e9, tz=timezone.utc).date()


SETTLE_MINUTES = 90     # a day's bar is final only some time after the close: SPY's bar of 2026-11-20 fetched 2 minutes after it said 690.87 and, fetched again that evening, 689.81 with more volume


def is_part_of_a_day(day: date, fetched_ns: int) -> bool:
    """True when a REST bar dated `day` sits in a file last written BEFORE that day's regular close plus `SETTLE_MINUTES` (design choices entry 129): it was fetched while the session was still open, or before
    the provider had settled the day, so its "close" is the price at the moment of the fetch and its volume is a part of the day's. Such a bar is never read: not as a close, not for an indicator. A fetch after the close rewrites the file and the bar is then the day's own."""
    from prometheus.common.market_calendar import regular_close_utc
    return fetched_ns < (regular_close_utc(day).timestamp() + SETTLE_MINUTES * 60) * 1e9


def _day_files(root: Path) -> dict[date, Path]:
    """{date: path} of the `YYYY/MM/YYYY-MM-DD.csv.gz` files under `root` (a name that is not a date is ignored)."""
    files = {}
    for path in root.glob("*/*/*.csv.gz"):
        try:
            files[date.fromisoformat(path.name[: -len(".csv.gz")])] = path
        except ValueError:
            continue
    return files


class BarStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT, extra_root: Path | str | None = None):
        self.root = Path(root)
        self.extra_root = Path(extra_root) if extra_root else self.root.parent / "ticker_bars_v1"
        self._bars: dict[str, list[Bar]] = {}
        self.dates: list[date] = []     # the trading dates read, oldest first
        self.gaps: list[str] = []
        self.loaded: set[str] = set()

    def load(self, symbols, end: date | None = None, max_bars: int = MAX_BARS, start: date | None = None) -> "BarStore":
        """Read the newest `max_bars` day files on or before `end` (default: the newest stored), keep only `symbols`. Returns self. Call it ONCE per run with every symbol needed.
        `start` (Trader v1: `DataClerkTools.daily_bars`) leaves out every day before it, files and REST bars alike, so a range is read and nothing older than it (left out: as before).
        With no `end` while a past day is bound (`trading_clock.as_of`, the Proving Ground) the newest day read is that day, not the newest stored: an EXPLICIT `end` is the caller's own range and is kept
        (the follower's and the backtest's door to the days after the decision, `DataClerkTools.daily_bars`)."""
        if end is None:
            end = trading_clock.bound_day()
        wanted = {s.strip().upper() for s in symbols}
        self.loaded |= wanted
        files = _day_files(self.root)
        index = bar_index.index_for(self.root, self.extra_root)       # the Proving Ground's bar index (design/proving_ground_spec.md 7.6): the same bars from one small file per symbol, when it can prove it is current
        if index is not None:
            try:
                index.fill(self, wanted, files, end, max_bars, start)
                return self
            except Exception:       # the files are the truth: this call reads them, and the index is built again next time
                index.reset()
        days = sorted(d for d in files if (end is None or d <= end) and (start is None or d >= start))[-max_bars:]
        for day in days:
            try:
                table = pv_csv.read_csv(files[day], convert_options=pv_csv.ConvertOptions(include_columns=list(COLUMNS)))
                table = table.filter(pc.is_in(table["ticker"], value_set=pa.array(sorted(wanted), type=pa.string())))
                for row in table.to_pylist():
                    self._bars.setdefault(row["ticker"], []).append(
                        Bar(date_of(row["window_start"]), row["open"], row["high"], row["low"], row["close"], row["volume"], int(row["transactions"])))
            except Exception as error:  # a corrupt file or a missing column: say which day and what kind, never the file's content
                self.gaps.append(f"{day}: unreadable ({type(error).__name__})")
        seen = set(days)
        for symbol in sorted(wanted):  # the per-ticker REST files add the dates the day files lack
            have = {b.date for b in self._bars.get(symbol, [])}
            for bar in self._extra_bars(symbol):
                if (end is None or bar.date <= end) and (start is None or bar.date >= start) and bar.date not in have:
                    self._bars.setdefault(symbol, []).append(bar)
                    seen.add(bar.date)
        self.dates = sorted(seen)[-max_bars:]
        if not self.dates:
            self.gaps.append(f"no day_aggs_v1 files on or before {end or 'today'} under {self.root}" if start is None else
                             f"no day_aggs_v1 files from {start} to {end or 'the newest'} under {self.root}")
            return self
        day = self.dates[0]
        while day <= self.dates[-1]:  # a weekday with no bar of any symbol between the first and last day read
            if day.weekday() < 5 and day not in seen:
                self.gaps.append(f"{day}: no file (a market holiday, or not synced)")
            day += timedelta(days=1)
        for series in self._bars.values():
            series.sort(key=lambda b: b.date)
        return self

    def _extra_bars(self, symbol: str) -> list[Bar]:
        path = self.extra_root / f"{symbol}.csv"
        if not path.exists():
            return []
        try:
            table = pv_csv.read_csv(path, convert_options=pv_csv.ConvertOptions(include_columns=list(COLUMNS)))
            fetched = path.stat().st_mtime_ns
            bars = [Bar(date_of(r["window_start"]), r["open"], r["high"], r["low"], r["close"], r["volume"], int(r["transactions"])) for r in table.to_pylist() if r["ticker"] == symbol]
            return [bar for bar in bars if not is_part_of_a_day(bar.date, fetched)]   # a bar fetched before its day closed is a part of a day: never read (entry 129)
        except Exception as error:  # a damaged file: said, and the day files are still used
            self.gaps.append(f"{symbol}: the REST bar file is unreadable ({type(error).__name__})")
            return []

    def bars(self, symbol: str, end: date | None = None, max_bars: int = MAX_BARS) -> list[Bar]:
        """The symbol's bars on or before `end`, oldest first, at most `max_bars`. Empty when the symbol is not in the files."""
        series = self._bars.get(symbol.strip().upper(), [])
        if end is not None:
            series = [b for b in series if b.date <= end]
        return series[-max_bars:]
