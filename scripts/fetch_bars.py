"""Refresh the DAILY BARS of some tickers (and SPY and the 11 sector ETFs) from Massive's REST aggregates: what the Advisor's indicators and its freshness gate need.

    python scripts/fetch_bars.py --symbols SNDK PG                    # says what it would do and stops
    python scripts/fetch_bars.py --symbols SNDK PG --days 400 --yes   # does it: one paid-API call per ticker, 14 in all with SPY and the ETFs

The bars are written to data/historical/ticker_bars_v1/<SYMBOL>.csv (merged with what is there; nothing else under data/historical is touched) and `BarStore` merges them with the
day files (the day file wins for a date both have). A real network call to a paid API. It goes through `DataToolbox.fetch_daily_bars`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.agents.data_clerk.data_toolbox import DataToolbox  # noqa: E402
from prometheus.agents.data_clerk.metrics_catalog import SPY, sector_etfs  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", required=True, help="tickers, for example SNDK PG")
    parser.add_argument("--days", type=int, default=400, help="calendar days back (1 to 800; 400 gives the 252 sessions the long indicators need)")
    parser.add_argument("--yes", action="store_true", help="really call the API (without it the plan is only printed)")
    args = parser.parse_args(argv)
    symbols = list(dict.fromkeys([*(s.strip().lstrip("$").upper() for s in args.symbols), SPY, *sector_etfs()]))
    print(f"daily bars of {len(symbols)} tickers for the last {args.days} days: {', '.join(symbols)}")
    if not args.yes:
        print("not run: add --yes to call the API")
        return 0
    for symbol, info in DataToolbox(base_dir=REPO_ROOT).fetch_daily_bars(symbols, args.days).items():
        print(f"{symbol:<6} fetched {info['fetched']:>4}, stored {info['stored']:>4}, {info['first']} to {info['last']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
