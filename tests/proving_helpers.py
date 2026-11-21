"""REAL data for the tests of the Proving Ground's pure core (design/proving_ground_spec.md section 14; design choices entry 126): no invented stock data and no stand-in agent.

    bars = real_bars(["XOM", "SPY"])            # the stored daily bars (`data/historical/day_aggs_v1`, read only) from 2025-05-01 to the end of 2026-11-19, as the Data Clerk gives them to the Account Manager
    found = real_records()                        # every DISTINCT recommendation record of the real captures in `tests/fixtures/session_real/` and of the real ledger (`data/ledger/recommendations.jsonl`, read only), 28 of them
    order = replayed("auto_even_split_dollars")   # a real capture's order put through the REAL Trader in backtest mode (no model) over the stored bars: its fills are at the open of Monday 2026-11-16

The bars end on 2026-11-19 ON PURPOSE: the stored bar of the day the tests were written (2026-11-20) was a session in progress and a later refresh changes it; every day up to the 19th is a finished session. A machine without the stored bars
skips (`stored_bars_available`). The recommendations are real (the Advisor's and the Critic's records of the sessions of 2026-11-06, 2026-11-12 and 2026-11-13); their fills in the captures are paper fills at the quote of 2026-11-13.
"""
import copy
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from handshake_helpers import MOMENT, clerk, manager, trader

from prometheus.agents.advisor.trade_plan import add_weekdays
from prometheus.common import work_order as wo
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio
from prometheus.proving import record
from prometheus.proving.outcomes import EntryFill, entry_fill_of

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "tests" / "fixtures" / "session_real"
LEDGER = ROOT / "data" / "ledger"
FIRST, LAST = date(2025, 5, 1), date(2026, 11, 19)


@lru_cache(maxsize=None)
def _bars(symbols: tuple[str, ...]) -> dict[str, list[dict]]:
    return clerk().tools.daily_bars(list(symbols), FIRST, LAST)["bars"]


def real_bars(symbols) -> dict[str, list[dict]]:
    """The stored daily bars of `symbols` (and the benchmark's) from 2025-05-01 to 2026-11-19, oldest first, as the Data Clerk gives them. A symbol with no stored bar (a London listing) is an empty list."""
    return _bars(tuple(sorted({*symbols, "SPY"})))


def stored_bars_available() -> bool:
    try:
        return bool(real_bars(["XOM"])["XOM"])
    except Exception:
        return False


@dataclass
class Real:
    """One distinct real recommendation: its record, where it came from, the day it was made, the order that holds it (the work order of a capture, or just the ledger's reviews), the desk's REAL fill when the capture traded it."""
    record_id: str
    source: str
    rec: dict
    day: date
    order: object
    fill: EntryFill | None


def real_records() -> list[Real]:
    """Every distinct recommendation (by `record_id`) of the real captures and the ledger, oldest decision first and then by symbol. A capture's recommendation carries its capture's order (the Critic's review, the Account Manager's decisions)
    and the real fill of its trade when the capture bought it (a paper fill at the quote of 2026-11-13); a ledger record carries the ledger's reviews."""
    found: dict[str, Real] = {}
    for path in sorted(CAPTURES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("order"), dict) or not data["order"].get("recommendations"):
            continue
        order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
        fills = {t["record_id"]: entry_fill_of(t) for t in order.trades if t.get("status") == "filled" and t.get("side") == "buy"}
        for rec in order.recommendations:
            rid = rec["identity"]["record_id"]
            if date.fromisoformat(rec["identity"]["as_of_utc"][:10]) >= LAST:      # a recommendation made on or after the last bar these tests read has no next session to enter at yet (the live session of Critic version 2, made on 2026-11-20, is one): it is not one of the 28
                continue
            found.setdefault(rid, Real(rid, path.name, rec, date.fromisoformat(rec["identity"]["as_of_utc"][:10]), order, fills.get(rid)))
    reviews = [json.loads(line) for line in (LEDGER / "reviews.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ledger_order = SimpleNamespace(reviews=reviews, trade_decisions=[])           # what `record.verdict_of` and `identity_of` read of an order: the Critic's reviews and the decisions
    for line in (LEDGER / "recommendations.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rid = rec["identity"]["record_id"]
        found.setdefault(rid, Real(rid, "ledger", rec, date.fromisoformat(rec["identity"]["as_of_utc"][:10]), ledger_order, None))
    return sorted(found.values(), key=lambda r: (r.day, r.rec["symbol"], r.record_id))


def replayed(capture: str, cash: str = "40000.00") -> wo.WorkOrder:
    """A real capture's work order put through the REAL Trader in backtest mode with no model (the free path the Time Machine uses): its decisions of 2026-11-13 are bought at the open of the next session (Monday 2026-11-16), through the
    per-trade handshake with the real Account Manager, the funds being `cash`. Its `trades` hold the fills and the Trader's own `outcome` of each purchase (the plan followed through the stored bars)."""
    data = json.loads((CAPTURES / f"{capture}.json").read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    order.status, order.trades, order.trader_runs = "decided", [], []
    manager().perform_trades(order, Portfolio(user="Bob Jones", cash_avail=Decimal(cash)), trader(mode="backtest"), MOMENT)
    return order


def plans_of(order: wo.WorkOrder, day: date = date(2026, 11, 13)) -> dict:
    """{record_id: LotInfo} for the purchases of an order: the plan of each recommendation that was bought (a neutral company has none: nothing closes it) and the key of its outcome."""
    from prometheus.proving.portfolio import LotInfo
    recs = {r["identity"]["record_id"]: r for r in order.recommendations}
    return {t["record_id"]: LotInfo(plan=record.plan_of(recs.get(t["record_id"])), key=f"{day.isoformat()}:{t['record_id']}", name=(recs.get(t["record_id"]) or {}).get("company"), decision_day=day) for t in order.trades}


def restated(plan, close: Decimal, day: date):
    """A real plan's own distances on an earlier real day: every price as the same fraction of the new entry price as it was of the old one, the horizon as the same number of weekdays (the Advisor's own rule). Nothing else is changed."""
    scale = lambda x: None if x is None else price_door(close * x / plan.entry_mid)
    return plan.model_copy(update=dict(entry_mid=close, entry_low=scale(plan.entry_low), entry_high=scale(plan.entry_high), limit_price=scale(plan.limit_price), stop=scale(plan.stop), target=scale(plan.target),
                                       time_stop_date=add_weekdays(day, plan.horizon_bars)))


def close_on(series, day: str) -> Decimal:
    return price_door(next(b for b in series if b["date"] == day)["close"])
