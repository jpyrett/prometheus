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
from prometheus.common.portfolio import trade_amount
from prometheus.persistence import accounts as accounts_module
from prometheus.persistence.accounts import SECRETS, open_account_store
from prometheus.proving import live, recs
from prometheus.proving.bars import read_bars
from prometheus.proving.config import load_config
from prometheus.proving.record import plan_of

REAL = Path(__file__).parent / "fixtures" / "proving_real"
SESSION_REAL = Path(__file__).parent / "fixtures" / "session_real"


def snapshot_of(name: str) -> dict:
    return json.loads((REAL / name).read_text(encoding="utf-8"), parse_float=Decimal)      # a snapshot is money and prices written by the encoder: read as the digits they are


def test_the_monitor_never_writes_p11():
    """P11: the source of the monitor calls no door that writes, and the book's `history` it reads is a SELECT."""
    source = inspect.getsource(live)
    for name in ("book_trade", "record_fill", "approve_trade", "preview_trade", "decide_", "perform_trades", "add_funds", "remove_funds", "execute_trades", "save_decisions", "record_trade"):
        assert name not in source, name
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", source)      # the SQL words, in capitals
    book = inspect.getsource(accounts_module.AccountStore.history)
    assert "SELECT" in book and not re.search(r"\b(INSERT|UPDATE|DELETE)\b", book)


def test_the_real_snapshot_of_neutral_purchases_adds_up_and_shows_no_plan():
    """Three positions bought by a live Auto run as NEUTRAL companies (the Advisor made no plan for them): each is `no_plan`, and every figure is recomputed from the shares, the cost and the live price."""
    shot = snapshot_of("live_snapshot_oil.json")
    assert shot["account_id"] == 8277 and {p["symbol"] for p in shot["positions"]} == {"APA", "COP", "CVX"}
    cost = value = Decimal(0)
    for p in shot["positions"]:
        assert p["status"] == "no_plan" and p["plan"] is None and not p["simulated"] and p["record_id"].startswith("neutral-")
        assert p["value"] == trade_amount(p["shares"], p["price"]) and p["gain"] == p["value"] - trade_amount(p["shares"], p["cost_basis"])
        cost, value = cost + trade_amount(p["shares"], p["cost_basis"]), value + p["value"]
    assert (shot["invested_at_cost"], shot["market_value"], shot["gain"]) == (cost, value, value - cost) == (Decimal("15000.00"), shot["market_value"], shot["market_value"] - Decimal("15000.00"))
    assert shot["equity"] == shot["cash"] + shot["market_value"] and shot["cash"] == Decimal("10000.00") and shot["alerts"] == []
    assert shot["session"]["open"] is True and shot["history"] == []      # an account of today has the live point alone: a bar stored today is not a close


def test_the_real_plan_of_a_real_purchase_is_judged_on_the_stored_lows():
    """The throwaway account held the real fills of 2026-11-13 with the plans of their recommendations: each plan read back from the captured recommendation, and each stop hit found again on the stored bars, read independently."""
    shot = snapshot_of("live_snapshot_plan.json")
    capture = json.loads((SESSION_REAL / "auto_even_split_dollars.json").read_text(encoding="utf-8"))
    by_record = {r["identity"]["record_id"]: r for r in capture["order"]["recommendations"]}
    bars = read_bars([p["symbol"] for p in shot["positions"]], date(2026, 11, 13), date(2026, 11, 19))
    for p in shot["positions"]:
        plan = plan_of(by_record[p["record_id"]])
        assert (p["plan"]["stop"], p["plan"]["target"], p["plan"]["time_stop_date"]) == (plan.stop, plan.target, plan.time_stop_date.isoformat())
        first_low = next(b for b in bars[p["symbol"]] if b["date"] > "2026-11-13" and simulator.bar_price(b, "low") <= plan.stop)
        assert p["status"] == "stop_hit" and p["evidence"] == f"the low of {first_low['date']}"
        assert all(simulator.bar_price(b, "low") > plan.stop for b in bars[p["symbol"]] if "2026-11-13" < b["date"] < first_low["date"])
    assert {a["symbol"] for a in shot["alerts"]} == {"CVX", "EOG", "XOM"} and all("Nothing was sold" in a["sentence"] for a in shot["alerts"])
    assert [h["date"] for h in shot["history"]][0] == "2026-11-12" and shot["history"][0]["equity"] == Decimal("20000.00")      # the first day: the deposit, before the purchases


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
