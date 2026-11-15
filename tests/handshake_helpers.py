"""REAL data for the tests of the per-trade handshake (design choices entry 81): no invented stock data, no stand-in agent and no stand-in model.

    order = real_order()                       # the REAL work order of `tests/fixtures/session_real/bob_jones_oil.json` (a real session of 2026-11-14): the real recommendations and the Critic's real reviews,
                                               # decided again by the Account Manager's own decision step: XOM 8 shares and EOG 8 shares, `yes_auto` (APA was denied for risk)
    am = manager()                             # the REAL Account Manager over the REAL Data Clerk (`DataClerk(model=object())`: its quotes and bars are code, its model is never called)
    Trader(use_model=False, ...)               # the REAL Trader, with no model (the sweep places the orders: the free, repeatable path)

THE PRICES ARE THE STORED ONES. The `friday_close` fixture turns the simulated clock on (`USE_LIVE_TRADING=false`, design choices entry 76) and pins its day to Friday 2026-11-13: the real Data Clerk then answers a
price request with the market close of that day read from the stored daily bars (`data/historical/day_aggs_v1`, read only), marked `simulated`, and Finnhub is never asked. The quotes are the real closes of XOM and
EOG on that day; the order's approvals last to 21:00 UTC that day and the moment of every run is 20:30 UTC, as in the real session. A machine without those stored bars skips these tests.

Doubles are only ever a WRAP of the real thing that records what it was asked and calls through (`Recorder`), or a fault injected into one real call to make a real failure path run.
"""
import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import DEFAULT_PORTFOLIO, AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.trader.trader import NullLedger, Trader
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, load_portfolio
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the moment of the real session
FRIDAY = date(2026, 11, 13)
BOB = Portfolio(user="Bob Jones", cash_avail=15000.0)                  # the account the real session was for
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto", auto_min_score=70)


def real_order() -> wo.WorkOrder:
    """The real session's order, decided again: the real recommendations and reviews, the Account Manager's decision step (the same call as `tests/test_session_replay.py`): XOM 8 and EOG 8 shares, `yes_auto`."""
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return decisions.decide_trades(order, BOB, PROFILE, 80, None, MOMENT, "auto", "even", None)


def jacob_as_bob() -> Portfolio:
    """The REAL sample portfolio (`configs/portfolios/jacob_pyrett.json`: $51.23 of cash, MSFT 50, TSLA 123, GM 60, all bought long ago) under the name of the account of the real session, so that its holdings can be
    sold to fund a purchase. Nothing else about it is changed."""
    return load_portfolio(DEFAULT_PORTFOLIO).model_copy(update={"user": "Bob Jones"})


def clerk() -> DataClerk:
    return DataClerk(model=object())


def manager(**options) -> AccountManager:
    """The real Account Manager over the real Data Clerk, with the real session's profile (or the saved account's own, when `store` and `account_id` are given)."""
    options.setdefault("profile", None if options.get("store") is not None else PROFILE)
    return AccountManager(model=object(), model_id="handshake", clerk=clerk(), **options)


def trader(**options) -> Trader:
    """The real Trader with no model (the sweep places the orders) and a ledger that keeps nothing; `mode="backtest"` is the other setting."""
    return Trader(use_model=False, ledger=NullLedger(), **options)


def stored_close_available() -> bool:
    try:
        clerk().tools.close_quote("XOM", FRIDAY)
        clerk().tools.close_quote("EOG", FRIDAY)
        return True
    except Exception:
        return False


@pytest.fixture
def friday_close(monkeypatch):
    """The real Data Clerk's price is the stored close of Friday 2026-11-13 (see the module text). Skips when those bars are not on this machine."""
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.market_day", lambda *a, **k: FRIDAY)
    if not stored_close_available():
        pytest.skip("the stored daily bars of XOM and EOG for 2026-11-13 are not on this machine (data/historical/day_aggs_v1)")


def close_of(symbol: str) -> float:
    """The real stored close the Data Clerk quotes for `symbol` on the pinned Friday."""
    return float(clerk().tools.close_quote(symbol, FRIDAY)["price"])


class Recorder:
    """Wraps the REAL Account Manager's `approve_trade`, `preview_trade` and `record_fill` and the REAL broker's `_place` so that they record what happened, in order, and call through: nothing is replaced. `events` is a
    list of (kind, trade_id) with kind `place`, `preview`, `approve` or `record`; `answers` the Approval or Booking each call returned, in order."""

    def __init__(self, monkeypatch, *managers):
        from prometheus.agents.trader.broker import PaperBroker
        self.events, self.answers = [], []
        recorder = self
        for am in managers:
            for name, kind in (("approve_trade", "approve"), ("preview_trade", "preview"), ("record_fill", "record")):
                setattr(am, name, self._wrap(getattr(am, name), kind))
        original = PaperBroker._place

        def placed(broker, trade, *args, **kw):
            recorder.events.append(("place", trade.trade_id))
            return original(broker, trade, *args, **kw)

        monkeypatch.setattr(PaperBroker, "_place", placed)

    def _wrap(self, real, kind):
        def watched(trade, *args, **kw):
            self.events.append((kind, trade["trade_id"]))
            answer = real(trade, *args, **kw)
            self.answers.append((kind, trade["trade_id"], answer))
            return answer

        return watched

    def kinds(self) -> list[str]:
        return [f"{kind} {trade_id}" for kind, trade_id in self.events]
