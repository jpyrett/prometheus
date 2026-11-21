"""Capture REAL answers of Finnhub's analyst recommendation trends (`/stock/recommendation`) for six symbols of the stored universe, so that a test can score the analysts' dimension on what Finnhub really says
(design choices entry 137; design/critic_v3_two_reasons_spec.md sections 3 and 6).

    python scripts/capture_finnhub_recommendations.py                        # says what it would do and stops
    python scripts/capture_finnhub_recommendations.py --yes                  # does it
    python scripts/capture_finnhub_recommendations.py --yes XOM PG           # other symbols (six by default)

WHAT IT CALLS (so it asks for --yes): Finnhub's `/stock/recommendation`, once per symbol (six calls, paced as every Finnhub call of the toolbox is). Only this endpoint is in the licence (probed 2026-11-20: the price-target,
upgrade-downgrade and scan endpoints answer 403, and none of them is called). It fetches nothing else, calls no model and does NOT write `data/analysts/` (that store is the Data Clerk's, filled by a Critic review or a backfill).
It writes ONLY `tests/fixtures/finnhub_recommendation_trends.json`: for each symbol the rows exactly as Finnhub gave them (`period`, `strongBuy`, `buy`, `hold`, `sell`, `strongSell`, `symbol`), the moment of the capture and the endpoint.
The key is read by the toolbox (`DataToolbox._api_keys`, the same handling as every Finnhub call); it is never printed or saved.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from prometheus.agents.data_clerk.data_toolbox import FINNHUB_MIN_INTERVAL, FINNHUB_RECOMMENDATION_URL, DataToolbox, _RateLimiter  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "finnhub_recommendation_trends.json"
SYMBOLS = ("XOM", "PG", "SNDK", "NVDA", "AMD", "KO")      # six companies of the stored universe (their news is in news/): an oil major, a staple, a memory maker, two chip makers, a beverage maker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--yes", action="store_true", help="make the calls (Finnhub, one per symbol)")
    args = parser.parse_args()
    symbols = list(dict.fromkeys(s.strip().upper() for s in args.symbols))
    if not args.yes:
        print(f"Would ask Finnhub's /stock/recommendation for {', '.join(symbols)} ({len(symbols)} calls) and write {OUT.relative_to(REPO_ROOT)}. Run again with --yes to do it.")
        return 0
    toolbox = DataToolbox(base_dir=REPO_ROOT)
    key = toolbox._api_keys()["FINNHUB_API_KEY"]
    answers = {}
    with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
        limiter = _RateLimiter(FINNHUB_MIN_INTERVAL)
        for symbol in symbols:
            rows = toolbox._get_json(client, limiter, "Finnhub recommendation", FINNHUB_RECOMMENDATION_URL, {"symbol": symbol}, headers={"X-Finnhub-Token": key})
            answers[symbol] = rows
            print(f"  {symbol}: {len(rows) if isinstance(rows, list) else '?'} periods" + (f", newest {rows[0].get('period')}" if isinstance(rows, list) and rows else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "endpoint": FINNHUB_RECOMMENDATION_URL, "symbols": answers}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
