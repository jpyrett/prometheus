"""The live monitor (prometheus/proving/live.py, design/proving_ground_spec.md section 11, P11): it watches and never writes. Real data only: the real snapshots of 2026-11-20 on live Finnhub quotes (`tests/fixtures/proving_real/live_snapshot_*.json`:
the paper account that bought three NEUTRAL oil companies, and a throwaway account that held the three REAL fills of 2026-11-13 of `auto_even_split_dollars.json`), the stored bars, and a throwaway Postgres account on the simulated clock (no Finnhub call)."""
import inspect
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.agents.trader import simulator
from prometheus.common import trading_clock
from prometheus.persistence import accounts as accounts_module
from prometheus.persistence.accounts import SECRETS, open_account_store
from prometheus.proving import live
from prometheus.proving.bars import read_bars

REAL = Path(__file__).parent / "fixtures" / "proving_real"
SESSION_REAL = Path(__file__).parent / "fixtures" / "session_real"


def test_the_monitor_never_writes_p11():
    """P11: the source of the monitor calls no door that writes, and the book's `history` it reads is a SELECT."""
    source = inspect.getsource(live)
    for name in ("book_trade", "record_fill", "approve_trade", "preview_trade", "decide_", "perform_trades", "add_funds", "remove_funds", "execute_trades", "save_decisions", "record_trade"):
        assert name not in source, name
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", source)      # the SQL words, in capitals
    book = inspect.getsource(accounts_module.AccountStore.history)
    assert "SELECT" in book and not re.search(r"\b(INSERT|UPDATE|DELETE)\b", book)


@pytest.fixture(scope="module")
def book():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        store = open_account_store()
        store.ping()
    except Exception:
        pytest.skip("the local Postgres is not reachable")
    return store


def test_a_throwaway_account_on_the_simulated_clock_is_read_and_left_untouched(book):
    """The real fills of 2026-11-13 booked into a throwaway account, watched as of 2026-11-19 (stored closes, no Finnhub call): the snapshot reads the book, judges the stops on the stored lows, and the book is the same after it."""
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
    capture = json.loads((SESSION_REAL / "auto_even_split_dollars.json").read_text(encoding="utf-8"))
    account_id = book.create_account("Live Monitor Throwaway", "moderate", trade_mode="auto", age=45)
    try:
        manager = AccountManager(clerk=DataClerk(allow_fetch=False))      # the real Data Clerk, which never fetches; no model call is made (a quote and a bar are lookups)
        manager.connect(book, account_id)
        manager.add_funds(Decimal("20000.00"), date(2026, 11, 12))
        for trade in capture["order"]["trades"]:
            if trade["record_id"] in ("d415bba65829", "64cf72c6b6bc", "c17c56d1ce7d"):
                fill = trade["fill"]
                book.book_trade(account_id, trade["symbol"], "buy", Decimal(str(fill["shares"])), Decimal(str(fill["price"])), date.fromisoformat(fill["date"]), name=trade["symbol"], source="bot", record_id=trade["record_id"])
        before = (manager.list_portfolio().model_dump_json(), json.dumps(manager.history(), default=str))
        with trading_clock.as_of(date(2026, 11, 19)):
            shot = live.snapshot(manager, account_id)
        assert (manager.list_portfolio().model_dump_json(), json.dumps(manager.history(), default=str)) == before
        assert all(p.simulated and p.status == "stop_hit" and p.plan is not None for p in shot.positions) and {a.symbol for a in shot.alerts} == {"CVX", "EOG", "XOM"}
        close = {p.symbol: p.price for p in shot.positions}
        assert close == {s: simulator.bar_price(read_bars([s], date(2026, 11, 19), date(2026, 11, 19))[s][0], "close") for s in close}
        assert shot.equity == shot.cash + shot.market_value and shot.cash == Decimal("2000.00") and shot.history[0].date == date(2026, 11, 12) and shot.history[-1].date == date(2026, 11, 18)
    finally:
        book.delete_account(account_id)
