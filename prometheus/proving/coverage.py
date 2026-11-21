"""What is STORED, per month, and the range a Time Machine run may use (design/proving_ground_spec.md section 4.3 and 7.1; `GET /api/proving/coverage`, `python -m prometheus.proving coverage`).

    report = compute()                 # {first_bar_day, last_bar_day, usable_last_day, usable: {first_day, last_day}, months: {"2026-10": {bars, news, press_releases, filings, market_news, news_chunks, ...}}, symbols_with_splits, notes}
    print(text(report))

FREE: it reads the stored day files (names only), the stored news, press release and filing files (one date per article) and the vector store's metadata (one `published_ts` per chunk); no model, no network, nothing written but its own
cache (`data/index/coverage.json`, derived: safe to delete). Each count is per calendar month of the article's own publication (or the file's day):
 - `bars`: the day files of the month (a trading day with a stored bar of every symbol); `news`, `press_releases`, `filings`: the articles, releases and filings in the files by their dates; `market_news`: the daily market-news files of the month;
 - `news_chunks`, `press_release_chunks`, `filing_chunks`: the chunks in the vector store (what the Data Clerk really searches: a decision day needs text IN THE STORE, not only in a file).
THE USABLE RANGE of a run: its first day needs `min_history_sessions` stored bar days before it (the 252-session indicators) and news in the store `min_news_days` before it; its last day is the last day whose stored bar may be read as a close
(`bars.usable_last_day`: never today). The page never offers a decision day outside it. The cache is keyed by the files' sizes and modification times, so it is recomputed when something is stored.
"""
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from prometheus.common.market_calendar import is_trading_day, load_calendar, next_trading_day
from prometheus.proving import bars as bars_door
from prometheus.proving.config import load_config
from prometheus.rag.config import REPO_ROOT

CACHE = REPO_ROOT / "data" / "index" / "coverage.json"
DAY_AGGS = REPO_ROOT / "data" / "historical" / "day_aggs_v1"
SPLITS = REPO_ROOT / "data" / "historical" / "splits_v1"
STAMP = re.compile(r"^(published_utc|filed): (\d{4}-\d{2})")


def _month(value: str) -> str | None:
    return value[:7] if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}.*", value) else None


def _stamp(paths: list[Path]) -> list:
    return [[str(p.relative_to(REPO_ROOT)), p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(paths)]


def _count_files(paths: list[Path]) -> Counter:
    """One count per article, release or filing, by the month of its `published_utc` (or `filed`) line."""
    counts: Counter = Counter()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = STAMP.match(line)
                if match:
                    counts[match.group(2)] += 1
    return counts


def _chunks_by_month(kind: str) -> tuple[Counter | None, str | None]:
    """(chunks per month of `published_ts`, None) from the vector store's metadata, or (None, why) when it cannot be read (an ingest may be writing it: try again)."""
    try:
        from prometheus.rag.store import get_vector_store
        collection = get_vector_store(kind)._collection
        counts: Counter = Counter()
        offset, page = 0, 20000
        while True:
            got = collection.get(include=["metadatas"], limit=page, offset=offset)
            rows = got.get("metadatas") or []
            for meta in rows:
                ts = (meta or {}).get("published_ts")
                if isinstance(ts, (int, float)):
                    counts[datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")] += 1
            if len(rows) < page:
                return counts, None
            offset += page
    except Exception as error:
        return None, f"the vector store of {kind} could not be read ({type(error).__name__})"


def _store_stamp() -> list:
    base = REPO_ROOT / "data" / "processed" / "vectorstore" / "chroma.sqlite3"
    return [[base.name, base.stat().st_size, base.stat().st_mtime_ns]] if base.is_file() else []


def compute(force: bool = False) -> dict:
    """The coverage report (see the module text), cached by the stored files' sizes and times."""
    config = load_config()
    day_files = sorted(DAY_AGGS.glob("*/*/*.csv.gz"))
    news_files = sorted((REPO_ROOT / "news").glob("*.txt"))
    market_files = sorted((REPO_ROOT / "news" / "market").glob("*.jsonl"))
    pr_files = sorted((REPO_ROOT / "data" / "press_releases").glob("*.txt"))
    filing_files = sorted((REPO_ROOT / "data" / "filings").glob("*.txt"))
    stamp = {"files": [len(day_files), [_stamp(news_files), _stamp(pr_files), _stamp(filing_files)], [p.name for p in market_files]], "store": _store_stamp(), "split_tables": sorted(p.name for p in SPLITS.glob("*.json")) if SPLITS.is_dir() else [],
             "usable_last_day": bars_door.usable_last_day().isoformat()}
    try:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if not force and cached.get("stamp") == stamp:
            return cached["report"]
    except (OSError, ValueError):
        pass
    began = time.perf_counter()
    months: dict[str, dict] = {}

    def put(month: str | None, key: str, value) -> None:
        if month:
            months.setdefault(month, {}).setdefault(key, 0)
            months[month][key] = value

    days = []
    for path in day_files:
        try:
            days.append(date.fromisoformat(path.name[: -len(".csv.gz")]))
        except ValueError:
            continue
    days.sort()
    for month, n in Counter(d.strftime("%Y-%m") for d in days).items():
        put(month, "bars", n)
    for key, files in (("news", news_files), ("press_releases", pr_files), ("filings", filing_files)):
        for month, n in _count_files(files).items():
            put(month, key, n)
    for month, n in Counter(p.name[:7] for p in market_files).items():
        put(month, "market_news", n)
    notes: list[str] = []
    vector_first: dict[str, str | None] = {}
    for kind, key in (("news", "news_chunks"), ("press_releases", "press_release_chunks"), ("filings", "filing_chunks")):
        counts, why = _chunks_by_month(kind)
        if counts is None:
            notes.append(why)
            vector_first[kind] = None
            continue
        vector_first[kind] = min(counts) if counts else None
        for month, n in counts.items():
            put(month, key, n)
    last_bar = days[-1] if days else None
    usable_last = bars_door.usable_last_day()
    first_usable = None
    if days:
        after_history = days[config.time_machine.min_history_sessions] if len(days) > config.time_machine.min_history_sessions else None      # the 252nd session has 252 stored days before it
        if after_history is not None:
            first_usable = after_history
    news_oldest = _oldest_day_in_store("news")
    if first_usable is not None and news_oldest is not None:
        need = news_oldest + timedelta(days=config.time_machine.min_news_days)
        first_usable = max(first_usable, need)
    if first_usable is not None:
        calendar = load_calendar()
        while not is_trading_day(first_usable, calendar):
            first_usable = next_trading_day(first_usable, calendar)
    report = {"first_bar_day": days[0].isoformat() if days else None, "last_bar_day": last_bar.isoformat() if last_bar else None, "usable_last_day": usable_last.isoformat(),
              "usable": {"first_day": first_usable.isoformat() if first_usable else None, "last_day": usable_last.isoformat(), "min_history_sessions": config.time_machine.min_history_sessions, "min_news_days": config.time_machine.min_news_days,
                         "oldest_news_day_in_store": news_oldest.isoformat() if news_oldest else None},
              "months": {m: months[m] for m in sorted(months)}, "symbols_with_splits": sorted(p.stem for p in SPLITS.glob("*.json")) if SPLITS.is_dir() else [], "notes": notes, "seconds": round(time.perf_counter() - began, 2),
              "backfill": _backfill()}
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"stamp": stamp, "report": report}), encoding="utf-8")
    except OSError:
        pass
    return report


def _oldest_day_in_store(kind: str) -> date | None:
    """The date of the oldest chunk in the vector store of `kind` (None when it cannot be read or is empty)."""
    try:
        from prometheus.rag.store import get_vector_store
        collection = get_vector_store(kind)._collection
        oldest, offset, page = None, 0, 20000
        while True:
            got = collection.get(include=["metadatas"], limit=page, offset=offset)
            rows = got.get("metadatas") or []
            stamps = [m.get("published_ts") for m in rows if isinstance(m, dict) and isinstance(m.get("published_ts"), (int, float))]
            if stamps:
                low = min(stamps)
                oldest = low if oldest is None else min(oldest, low)
            if len(rows) < page:
                break
            offset += page
        return datetime.fromtimestamp(oldest, tz=timezone.utc).date() if oldest is not None else None
    except Exception:
        return None


def _backfill() -> dict | None:
    """What the history backfill has done (`data/historical/backfill_v1/progress.jsonl`): the oldest window start per kind and the symbols, or None when there is none."""
    path = REPO_ROOT / "data" / "historical" / "backfill_v1" / "progress.jsonl"
    if not path.is_file():
        return None
    oldest: dict[str, str] = {}
    symbols: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        kind, start = row.get("kind"), row.get("start")
        if isinstance(kind, str) and isinstance(start, str):
            oldest[kind] = min(oldest.get(kind, start), start)
            if row.get("symbol"):
                symbols.add(row["symbol"])
    return {"oldest_window_start": oldest, "symbols": len(symbols)}


def text(report: dict) -> str:
    """The report as the command prints it."""
    lines = [f"COVERAGE  bars {report['first_bar_day']} to {report['last_bar_day']} (usable up to {report['usable_last_day']}: nothing dated today is read as a close)",
             f"  usable decision days: {report['usable']['first_day']} to {report['usable']['last_day']}  (needs {report['usable']['min_history_sessions']} stored sessions before the start and news in the store {report['usable']['min_news_days']} days before it; the oldest news in the store is {report['usable']['oldest_news_day_in_store']})",
             f"  {'month':<8} {'bars':>5} {'news':>7} {'press':>6} {'filings':>8} {'market':>7} {'news chunks':>12} {'PR chunks':>10} {'filing chunks':>14}"]
    for month, row in report["months"].items():
        lines.append(f"  {month:<8} {row.get('bars', 0):>5} {row.get('news', 0):>7} {row.get('press_releases', 0):>6} {row.get('filings', 0):>8} {row.get('market_news', 0):>7} {row.get('news_chunks', 0):>12} {row.get('press_release_chunks', 0):>10} {row.get('filing_chunks', 0):>14}")
    if report.get("backfill"):
        lines.append(f"  backfill: oldest window start per kind {report['backfill']['oldest_window_start']} ({report['backfill']['symbols']} symbols)")
    lines.append(f"  split tables: {len(report['symbols_with_splits'])} symbols")
    lines.extend(f"  note: {n}" for n in report["notes"])
    return "\n".join(lines)


def report(value: dict) -> str:
    return text(value)
