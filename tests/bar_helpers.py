"""Small helpers the indicator tests share: bars built from literal numbers, and day files in the real `day_aggs_v1` format.

These are inputs for testing ARITHMETIC (a ramp whose RSI is 100, a series whose beta is exactly 2). They are not market data and no test of a recommendation uses them:
the SNDK and PG tests read the real stored bars.
"""
import csv
import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from prometheus.agents.data_clerk.bars import Bar


def weekdays(n: int, end: date = date(2026, 10, 23)) -> list[date]:
    """The n weekdays ending on `end` (a weekday), oldest first. No holidays: a test that needs a gap makes it."""
    days, day = [], end
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days[::-1]


def make_bars(closes, highs=None, lows=None, opens=None, volumes=None, end: date = date(2026, 10, 23)) -> list[Bar]:
    """Bars from closes. By default high = close + 0.5, low = close - 0.5, open = the previous close (the first: its own close), volume 1000."""
    days = weekdays(len(closes), end)
    bars = []
    for i, close in enumerate(closes):
        bars.append(Bar(days[i], opens[i] if opens else (closes[i - 1] if i else close), highs[i] if highs else close + 0.5, lows[i] if lows else close - 0.5, close,
                        volumes[i] if volumes else 1000.0, 10))
    return bars


def write_day_files(root: Path, rows_by_day: dict[date, list[dict]], skip: tuple[date, ...] = ()) -> None:
    """Write `root/YYYY/MM/YYYY-MM-DD.csv.gz` in the real format. `rows_by_day[day]` is a list of {ticker, open, high, low, close, volume}."""
    for day, rows in rows_by_day.items():
        if day in skip:
            continue
        folder = root / f"{day:%Y}" / f"{day:%m}"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = int(datetime(day.year, day.month, day.day, 4, tzinfo=timezone.utc).timestamp() * 1e9)  # the real files stamp the day at 04:00 UTC
        with gzip.open(folder / f"{day}.csv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"])
            for r in rows:
                writer.writerow([r["ticker"], r["volume"], r["open"], r["close"], r["high"], r["low"], stamp, 100])
