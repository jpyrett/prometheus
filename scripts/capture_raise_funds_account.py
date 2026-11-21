"""Capture the REAL account and the REAL quotes of Jacob's request "Buy 2 shares of ANT and raise funds if needed" (design choices entry 142; design/direct_raise_funds_spec.md), so that a test can replay them offline.

    python scripts/capture_raise_funds_account.py                 # says what it would do and stops
    python scripts/capture_raise_funds_account.py --yes           # does it

WHAT IT CALLS (so it asks for --yes): the Data Clerk's `request_quotes` for ANT and every holding of account 6553 (one Finnhub /quote call each, the paid key: 13 calls) and Finnhub's symbol search for ANT and GOOGL (2 calls). It READS account 6553 (`list_portfolio`: the cash and the
holdings, nothing else) and NEVER writes, trades or deletes it. It writes ONLY `tests/fixtures/raise_funds/account_6553.json`: the account as it was (the cash and each position's shares, average cost and the day it was
bought: all a throwaway account needs to be made the same) and the quotes exactly as the Data Clerk returned them. Nothing in it is edited. A file that is already there is not overwritten.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.agents.account_manager.account_manager import AccountManager  # noqa: E402
from prometheus.common.money import jsonable  # noqa: E402
from prometheus.persistence.accounts import open_account_store  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "raise_funds" / "account_6553.json"
ACCOUNT = 6553
TARGET = "ANT"
DIRECTORY_QUERIES = ("ANT", "GOOGL", "MCD")      # the tickets the recorded requests name that the recorded directory of entry 88 does not have (MCD: entry 143, the user's own sale in the same request)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="make the calls (Finnhub quotes) and write the fixture")
    args = parser.parse_args()
    print(f"account {ACCOUNT} (read only) and the quotes of {TARGET} and its holdings -> {OUT.relative_to(REPO_ROOT)}")
    if OUT.exists():
        have = json.loads(OUT.read_text(encoding="utf-8"))
        missing = [query for query in DIRECTORY_QUERIES if query not in have.get("directory", {})]
        if not missing:
            print("the fixture is already there, with the symbol directory: it is not overwritten")
            return
        if not args.yes:
            print(f"(the account and the quotes are recorded; the symbol directory answers for {', '.join(missing)} are not: this calls Finnhub's symbol search {len(missing)} time(s); run with --yes)")
            return
        from prometheus.tools.symbols import search_finnhub

        have.setdefault("directory", {}).update({query: search_finnhub(query) for query in missing})      # Finnhub's REAL symbol search for the tickets a request names, unedited; what is recorded stays as it is
        OUT.write_text(json.dumps(jsonable(have), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"added the symbol directory answers {', '.join(missing)} to {OUT.relative_to(REPO_ROOT)}")
        return
    if not args.yes:
        print("(this calls Finnhub once for each symbol: 13 quotes and 2 symbol searches; run with --yes)")
        return
    from prometheus.tools.symbols import search_finnhub

    store = open_account_store()
    portfolio = store.list_portfolio(ACCOUNT)      # a READ: nothing of the account is written
    manager = AccountManager(model=object(), model_id="capture")
    symbols = [TARGET, *portfolio.symbols]
    quoted = manager.request_prices(symbols)
    if quoted.status != "data_ready":
        raise SystemExit("the quotes could not all be had: " + "; ".join(quoted.errors))
    data = {
        "_note": ("The REAL account 6553 (read only) and the REAL Finnhub quotes of ANT and its holdings, as the Data Clerk returned them, recorded " + datetime.now(timezone.utc).strftime("%Y-%m-%d") + " by scripts/capture_raise_funds_account.py. "
                  "Jacob's request on the page was 'Buy 2 shares of ANT and raise funds if needed': the cash did not cover 2 x $88.91. A test makes a THROWAWAY account the same (the cash, and each position bought on its day at its average cost) and "
                  "gives these quotes back to the real Data Clerk. Nothing here is edited."),
        "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cash": portfolio.cash_avail,
        "positions": [{"symbol": p.symbol, "shares": p.shares, "cost_basis": p.cost_basis, "acquired": p.acquired.isoformat() if p.acquired else None} for p in portfolio.positions],
        "quotes": dict(quoted.data.quotes),
        "directory": {query: search_finnhub(query) for query in DIRECTORY_QUERIES},      # Finnhub's REAL symbol search answers for the tickets a request names, unedited: the Data Clerk's directory in a test
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(data), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {OUT.relative_to(REPO_ROOT)}; ANT {data['quotes'][TARGET]['price']}, cash {data['cash']}, {len(data['positions'])} holdings")


if __name__ == "__main__":
    main()
