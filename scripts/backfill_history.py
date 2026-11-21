"""Backfill the stored HISTORY of news, press releases, filings, market-wide news and splits, in windows, for the Proving Ground (design/proving_ground_spec.md section 4.3, design choices entry 127).

    python scripts/backfill_history.py --from 2026-05-20                       # says what it would do and stops
    python scripts/backfill_history.py --from 2026-05-20 --yes                 # does it (PAID: Finnhub and Massive, and it writes the stored text)
    python scripts/backfill_history.py --from 2026-05-20 --to 2026-10-16 --symbols XOM PG --kinds news,press_releases --yes

THE TIME MACHINE replays the desk at past days, so the stored news, press releases, filings and market news must reach back as far as the test does. The providers cut a long range short (the filings endpoints raise
at their limit of one call, a day of market-wide news is thousands of articles), so a range is fetched in WINDOWS of `--window-days` (30) days, OLDEST FIRST, and the windows of one range are the same for every symbol, so an
interrupted run has a contiguous history from `--from` up to where it stopped. The windowed forms of the fetchers (`start` and `end` on `DataToolbox.fetch_news`, `fetch_press_releases`, `fetch_filings`, `sync_market_news`)
are used; the `from_date` form is untouched. Everything that already holds stays: a broken article is discarded whole (entry 83), duplicates are dropped by id or URL, the security guard screens at ingest (entry 116).

KINDS (`--kinds`, default all five):
    news            each symbol's company news (Finnhub in 7-day windows and Massive), appended to news/<SYMBOL>.txt
    press_releases  each symbol's press releases (Finnhub), appended to data/press_releases/<SYMBOL>_PRs.txt
    filings         each symbol's SEC filings (Massive), appended to data/filings/<SYMBOL>.txt. A symbol whose stored filings already reach back to `--from` is skipped (they were fetched for a year)
    market_news     the market-wide news (Massive; Finnhub's general feeds only for a window that reaches the last two days), into news/market/<DAY>.jsonl
    splits          each symbol's split table, ONCE (not windowed), to data/historical/splits_v1/<SYMBOL>.json; a symbol that already has a file is skipped. Needed because the stored bars are unadjusted

THE SYMBOLS are `--symbols`, else every symbol that has stored news (the files of the news folder).

RESUMABLE. Every window that comes back is written to `data/historical/backfill_v1/progress.jsonl` (kind, symbol, start, end and what it fetched), and a window that is there is skipped the next time: run the same command again after
an interruption or a failure. A window that FAILED (an API error) is printed, is not recorded, and is tried again on the next run; after 8 failures in a row the run stops (a key or the network is wrong, and asking again would only add
calls). Without `--yes` nothing is fetched and nothing is written: it prints the plan (the windows, the symbols, what is already done, the MINIMUM number of API calls: paging adds more) and stops.

THEN INGEST. This script does NOT ingest: it prints the commands that make the new text searchable (`python -m prometheus.rag.ingest --all`, and `--press-releases --all`, `--filings --all`), which re-embed what changed.
"""
import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.agents.data_clerk.data_toolbox import FILING_FORMS, FINNHUB_WINDOW_DAYS, DataToolbox  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402
from prometheus.rag.loaders import check_symbol  # noqa: E402

KINDS = ("news", "press_releases", "filings", "market_news", "splits")
WINDOWED = ("news", "press_releases", "filings", "market_news")
WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 30                 # the providers cut a longer range short
MAX_FAILURES_IN_A_ROW = 8
PROGRESS_PATH = REPO_ROOT / "data" / "historical" / "backfill_v1" / "progress.jsonl"
MARKET = "*"                         # the "symbol" of a market-wide window in the progress file
FILING_CALLS = len(FILING_FORMS) + 2   # one index call per form, then the 10-K sections and the 8-K text: the calls of one window of one symbol
INGEST = {"news": "python -m prometheus.rag.ingest --all", "press_releases": "python -m prometheus.rag.ingest --press-releases --all", "filings": "python -m prometheus.rag.ingest --filings --all"}


def windows(first: date, last: date, days: int = WINDOW_DAYS) -> list[tuple[date, date]]:
    """The windows of `days` days from `first` to `last`, both included, OLDEST FIRST, with no gap and no overlap; the last one is shorter when the range does not divide. Pure."""
    if first > last:
        raise ValueError("the range starts after it ends")
    if not 1 <= days <= MAX_WINDOW_DAYS:
        raise ValueError(f"a window is 1 to {MAX_WINDOW_DAYS} days")
    out, start = [], first
    while start <= last:
        end = min(start + timedelta(days=days - 1), last)
        out.append((start, end))
        start = end + timedelta(days=1)
    return out


def minimum_calls(kind: str, start: date, end: date, today: date) -> int:
    """The fewest API calls one window of one symbol makes (paging and a retry add more). News: Finnhub's 7-day windows and one page of Massive's. Press releases: one. Filings: `FILING_CALLS`. Market news: one page of Massive's
    (and the two general Finnhub feeds when the window reaches the last two days). Splits: one. Pure."""
    days = (end - start).days + 1
    if kind == "news":
        return math.ceil(days / FINNHUB_WINDOW_DAYS) + 1
    if kind == "press_releases" or kind == "splits":
        return 1
    if kind == "filings":
        return FILING_CALLS
    if kind == "market_news":
        return 1 + (2 if end >= today - timedelta(days=2) else 0)
    raise ValueError(f"unknown kind {kind!r}")


class Progress:
    """The windows already fetched: one JSON line each in `path` (append only). `done(kind, symbol, start, end)` says whether a window is there; `mark` adds it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._done: set[tuple[str, str, str, str]] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._done.add((row["kind"], row["symbol"], row["start"], row["end"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue    # a line that is not a window is ignored: that window is fetched again, which costs calls and nothing else

    def done(self, kind: str, symbol: str, start: date, end: date) -> bool:
        return (kind, symbol, start.isoformat(), end.isoformat()) in self._done

    def mark(self, kind: str, symbol: str, start: date, end: date, stats: dict) -> None:
        row = {"kind": kind, "symbol": symbol, "start": start.isoformat(), "end": end.isoformat(), "stats": {k: v for k, v in stats.items() if isinstance(v, (int, str)) and k != "path"},
               "done_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._done.add((kind, symbol, row["start"], row["end"]))


def stored_symbols(toolbox: DataToolbox) -> list[str]:
    """Every symbol that has stored news: the files of the news folder (the `market` folder and anything that is not a symbol file are not), sorted."""
    return sorted(p.stem for p in toolbox._news_dir().glob("*.txt") if p.is_file())


def filings_reach(toolbox: DataToolbox, symbol: str) -> date | None:
    """The date of the oldest filing stored for `symbol`, or None when there is no file (or no filing in it)."""
    filed = [head["filed"] for head, _ in toolbox._read_records(toolbox._filings_dir() / f"{symbol}.txt", "<<<FILING>>>") if head.get("filed")]
    try:
        return date.fromisoformat(min(filed)) if filed else None
    except ValueError:
        return None


def has_splits(toolbox: DataToolbox, symbol: str) -> bool:
    return (toolbox.root_data_dir / "splits_v1" / f"{symbol}.json").exists()


def work_list(toolbox: DataToolbox, symbols: list[str], kinds: list[str], spans: list[tuple[date, date]], first: date, progress: Progress) -> list[dict]:
    """What is left to fetch, in the ORDER it will be fetched: window by window, oldest first, and inside a window the market news, then each symbol's kinds. Each item is {kind, symbol, start, end, skipped}: `skipped` is
    None for a window to fetch, else the reason it is not (it is done, or the stored filings already reach back far enough). Splits come first, once per symbol, with start and end None."""
    items = []
    if "splits" in kinds:
        items += [{"kind": "splits", "symbol": s, "start": None, "end": None, "skipped": "already stored" if has_splits(toolbox, s) else None} for s in symbols]
    reach = {s: filings_reach(toolbox, s) for s in symbols} if "filings" in kinds else {}
    for start, end in spans:
        for symbol, kind in [(MARKET, "market_news"), *((s, k) for s in symbols for k in WINDOWED if k != "market_news")]:
            if kind not in kinds:
                continue
            why = ("already fetched (progress file)" if progress.done(kind, symbol, start, end) else
                   f"stored filings already reach back to {reach[symbol]}" if kind == "filings" and reach.get(symbol) is not None and reach[symbol] <= first else None)
            items.append({"kind": kind, "symbol": symbol, "start": start, "end": end, "skipped": why})
    return items


def fetch(toolbox: DataToolbox, item: dict) -> dict:
    """One window of one kind, through the toolbox's windowed fetchers. Returns the toolbox's own stats."""
    kind, symbol, start, end = item["kind"], item["symbol"], item["start"], item["end"]
    if kind == "news":
        return toolbox.fetch_news(symbol=symbol, start=start, end=end)
    if kind == "press_releases":
        return toolbox.fetch_press_releases(symbol, start=start, end=end)
    if kind == "filings":
        return toolbox.fetch_filings(symbol, start=start, end=end)
    if kind == "market_news":
        return toolbox.sync_market_news(start=start, end=end)
    return toolbox.fetch_splits([symbol])[symbol]


def print_plan(out, first: date, last: date, spans, symbols, kinds, items, today: date, progress: Progress) -> int:
    """The plan: the range, the windows, the symbols, what is already done and the minimum number of API calls. Returns that minimum."""
    todo = [i for i in items if i["skipped"] is None]
    calls = sum(minimum_calls(i["kind"], i["start"] or first, i["end"] or last, today) for i in todo)
    print(f"BACKFILL PLAN   {first} to {last} ({(last - first).days + 1} days) in {len(spans)} window(s) of up to {(spans[0][1] - spans[0][0]).days + 1} days, oldest first", file=out)
    for start, end in spans:
        print(f"  window {start} .. {end}", file=out)
    print(f"  kinds   : {', '.join(kinds)}", file=out)
    print(f"  symbols : {len(symbols)}  {' '.join(symbols)}", file=out)
    print(f"  progress file: {progress.path}", file=out)
    for kind in kinds:
        mine = [i for i in items if i["kind"] == kind]
        left = [i for i in mine if i["skipped"] is None]
        reasons = ", ".join(sorted({i["skipped"].split(" to ")[0] if kind == "filings" and i["skipped"].startswith("stored filings") else i["skipped"] for i in mine if i["skipped"]})) or "none"
        needed = sum(minimum_calls(i["kind"], i["start"] or first, i["end"] or last, today) for i in left)
        print(f"  {kind:<15} {len(left):>4} to fetch, {len(mine) - len(left):>4} skipped ({reasons}), at least {needed} API calls", file=out)
    print(f"  TOTAL: at least {calls} API calls (paging and retries add more), PAID: Finnhub and Massive; it appends to news/, data/press_releases/, data/filings/ and data/historical/splits_v1/", file=out)
    return calls


def print_ingest(out, kinds) -> None:
    print("\nNEXT, make the new text searchable (this script does not ingest):", file=out)
    for kind in ("news", "press_releases", "filings"):
        if kind in kinds:
            print(f"    {INGEST[kind]}", file=out)
    if "market_news" in kinds:
        print("    (the market-wide news is read from its day files by the theme search: nothing to ingest)", file=out)


def main(argv: list[str] | None = None, toolbox: DataToolbox | None = None, today: date | None = None, out=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first", required=True, help="the first day to fetch, YYYY-MM-DD")
    parser.add_argument("--to", dest="last", help="the last day to fetch, YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="the symbols (default: every symbol with stored news)")
    parser.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated, from {', '.join(KINDS)} (default: all)")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS, help=f"days in one window, 1 to {MAX_WINDOW_DAYS} (default {WINDOW_DAYS})")
    parser.add_argument("--progress", type=Path, default=PROGRESS_PATH, help="the file that records the windows already fetched")
    parser.add_argument("--yes", action="store_true", help="really fetch (without it the plan is only printed)")
    args = parser.parse_args(argv)
    out = out or sys.stdout
    today = today or datetime.now(timezone.utc).date()
    try:
        first = date.fromisoformat(args.first)
        last = date.fromisoformat(args.last) if args.last else today
        kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
        unknown = [k for k in kinds if k not in KINDS]
        if unknown or not kinds:
            raise ValueError(f"--kinds: {', '.join(unknown) or 'none given'} (known: {', '.join(KINDS)})")
        if last > today:
            raise ValueError(f"--to {last} is in the future (today is {today})")
        spans = windows(first, last, args.window_days)
        toolbox = toolbox or DataToolbox(base_dir=REPO_ROOT)
        symbols = list(dict.fromkeys(check_symbol(s.strip().lstrip("$")) for s in args.symbols)) if args.symbols else stored_symbols(toolbox)
        if not symbols:
            raise ValueError("no symbols: give --symbols, or fetch some news first")
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    progress = Progress(args.progress)
    items = work_list(toolbox, symbols, kinds, spans, first, progress)
    print_plan(out, first, last, spans, symbols, kinds, items, today, progress)
    if not args.yes:
        print("\nnot run: add --yes to fetch", file=out)
        return 0

    todo = [i for i in items if i["skipped"] is None]
    totals: dict[tuple[str, str], dict[str, int]] = {}
    failures, in_a_row = [], 0
    print(f"\nFETCHING {len(todo)} window(s)...", file=out)
    for number, item in enumerate(todo, 1):
        label = f"[{number}/{len(todo)}] {item['kind']} {item['symbol']}" + (f" {item['start']}..{item['end']}" if item["start"] else "")
        try:
            stats = fetch(toolbox, item)
        except Exception as error:   # an API error (never with a key in its message) or a file that could not be written: said, not recorded, tried again next run
            in_a_row += 1
            failures.append((label, f"{type(error).__name__}: {' '.join(str(error).split())[:160]}"))
            print(f"\n{label}: FAILED ({failures[-1][1]})", file=out)
            if in_a_row >= MAX_FAILURES_IN_A_ROW:
                print(f"\nstopped: {in_a_row} failures in a row (a key or the network is wrong). Fix it and run the same command again: the windows that came back are not fetched twice.", file=out)
                break
            continue
        in_a_row = 0
        progress.mark(item["kind"], item["symbol"], item["start"] or first, item["end"] or last, stats)
        shown = {k: v for k, v in stats.items() if k in ("fetched", "matched", "written", "discarded", "duplicates", "failed") and isinstance(v, int)}
        print(f"\n{label}: " + ", ".join(f"{k} {v}" for k, v in shown.items()), file=out)
        total = totals.setdefault((item["symbol"], item["kind"]), {})
        for key, value in shown.items():
            total[key] = total.get(key, 0) + value

    print("\nFETCHED PER SYMBOL", file=out)
    for (symbol, kind), total in sorted(totals.items()):
        print(f"  {symbol:<9} {kind:<15} " + ", ".join(f"{k} {v}" for k, v in total.items()), file=out)
    if failures:
        print(f"\n{len(failures)} window(s) FAILED and were not recorded (run the same command again to retry them):", file=out)
        for label, why in failures:
            print(f"  {label}: {why}", file=out)
    print_ingest(out, kinds)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
