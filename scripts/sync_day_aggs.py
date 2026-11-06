"""Sync the stored DAY AGGREGATES (daily bars of every US ticker) for a range of months: what the Advisor's indicators need (Advisor v1 spec section 2.1).

    python scripts/sync_day_aggs.py --from 2025-09 --to 2026-11            # says what it would do and stops
    python scripts/sync_day_aggs.py --from 2025-09 --to 2026-11 --yes      # does it: `aws s3 sync` of day_aggs_v1 for each month, about 0.3 MB per trading day

It is a real network call (the Massive flat-file bucket, the credentials of .aws or ~/.aws) and it adds files under data/historical/day_aggs_v1 (it never deletes). Nothing else is
synced: not the minute bars, trades or quotes (gigabytes). It goes through `DataToolbox.sync_day_aggs`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.agents.data_clerk.data_toolbox import DataToolbox  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402


def months_between(first: str, last: str) -> list[str]:
    (y1, m1), (y2, m2) = (tuple(int(p) for p in text.split("-")) for text in (first, last))
    out = []
    while (y1, m1) <= (y2, m2):
        out.append(f"{y1}-{m1:02d}")
        y1, m1 = (y1 + 1, 1) if m1 == 12 else (y1, m1 + 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first", required=True, help="first month, YYYY-MM")
    parser.add_argument("--to", dest="last", required=True, help="last month, YYYY-MM")
    parser.add_argument("--yes", action="store_true", help="really run the sync (without it the plan is only printed)")
    args = parser.parse_args(argv)
    months = months_between(args.first, args.last)
    print(f"day_aggs_v1: {len(months)} month(s), {months[0]} to {months[-1]}, into {REPO_ROOT / 'data' / 'historical' / 'day_aggs_v1'} (about {len(months) * 21 * 0.3:.0f} MB)")
    if not args.yes:
        print("not run: add --yes to sync")
        return 0
    for folder in DataToolbox(base_dir=REPO_ROOT).sync_day_aggs(months):
        print(f"synced {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
