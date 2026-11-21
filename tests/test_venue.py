"""The venue seam (design choices entry 128; design/proving_ground_spec.md section 6): `prometheus/agents/trader/venue.py`.

REAL DATA ONLY. The orders are the tickets of the real session of 2026-11-13 (`tests/fixtures/session_real/bob_jones_oil.json`: XOM 8 and EOG 8 shares, decided again by the Account Manager's own step as `handshake_helpers.real_order` does), the quotes
are the stored closes of that Friday and the bars are the stored daily bars (`data/historical/day_aggs_v1`, read only) that the real Data Clerk gives the real Account Manager. The first session after the decision is Monday 2026-11-16: XOM opened
at 169.53 and EOG at 156.51 (read from the stored bars on 2026-11-20; two sessions later both were stopped out by their plans). A machine without those bars skips these tests. Nothing here calls a model, a quote API or a paid API.

The seam is a BEHAVIOUR-FREE change (every Trader test passed unchanged): what these tests add is that the two venues give EXACTLY what the two branches of `PaperBroker._simulate` and `execute._prices` gave (the simulator's own functions, called here
by hand with the same arguments), that a mode that is not paper or backtest is an error, that a venue that places real orders is refused, and that no `if mode ==` is left in the funnel.
"""
import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, clerk, close_of, friday_close, manager, real_order, stored_close_available, trader  # noqa: F401  (friday_close is a fixture)

from prometheus.agents.trader import execute, simulator, tickets, venue
from prometheus.agents.trader.broker import PaperBroker, Prices
from prometheus.agents.trader.config import load_config
from prometheus.agents.trader.venue import BacktestVenue, PaperVenue, venue_for
from prometheus.common.money import price as price_door
from prometheus.tools import trader as trader_tools

ROOT = Path(__file__).resolve().parents[1]
TRADER = ROOT / "prometheus" / "agents" / "trader"
SYMBOLS = ["XOM", "EOG"]
FIRST_SESSION = date(2026, 11, 16)

pytestmark = pytest.mark.skipif(not stored_close_available(), reason="the stored daily bars of XOM and EOG for 2026-11-13 are not on this machine (data/historical/day_aggs_v1)")


def real_bars(*symbols: str) -> dict[str, list[dict]]:
    """The stored bars from the decision day on, as the Data Clerk gives them to the Account Manager (and the benchmark's)."""
    return clerk().tools.daily_bars([*symbols, "SPY"], FRIDAY, date(2026, 11, 19))["bars"]


def real_tickets(mode: str):
    config = load_config()
    return config, tickets.build_tickets(real_order(), BOB, MOMENT, config, mode)


def test_the_two_venues_give_exactly_what_the_two_branches_gave():
    """The same arguments to the simulator's own `fill_paper` and `fill_backtest` give the same `Execution` as the venues, for every real ticket; a symbol with no stored quote or bar and a backtest that has no entry day yet are what the branches said."""
    config, paper_tickets = real_tickets("paper")
    quotes = {s: close_of(s) for s in SYMBOLS}
    snapshot = Prices("paper", FRIDAY, quotes=quotes)
    slip = config.costs.slippage_bps
    assert [t.symbol for t in paper_tickets] == SYMBOLS
    for t in paper_tickets:
        assert venue_for("paper", config).execute(t, t.shares, snapshot) == simulator.fill_paper(t.side, t.order_type, t.limit_price, quotes[t.symbol], t.limit_low, t.limit_high, slip)
        assert venue_for("paper", config).price_of(snapshot, t.symbol) == quotes[t.symbol]
    nowhere = paper_tickets[0].model_copy(update={"symbol": "SHEL.L"})                           # a London listing: no quote in the snapshot, no bar stored
    assert venue_for("paper", config).execute(nowhere, nowhere.shares, snapshot).reject_code == "price_unavailable" and venue_for("paper", config).missing(snapshot, "SHEL.L") == "there is no usable quote for SHEL.L"

    config, test_tickets = real_tickets("backtest")
    bars = real_bars(*SYMBOLS, "SHEL.L")
    entry = simulator.entry_date(MOMENT, bars)
    assert entry == FIRST_SESSION and bars["SHEL.L"] == []
    snapshot = Prices("backtest", entry, bars=bars, entry_date=entry)
    for t in test_tickets:
        done = venue_for("backtest", config).execute(t, t.shares, snapshot)
        assert done == simulator.fill_backtest(t.side, t.order_type, t.limit_price, simulator.bar_on(bars[t.symbol], entry), t.limit_low, t.limit_high, slip)
        assert (done.status, done.price_source) == ("filled", "bar_open") and venue_for("backtest", config).price_of(snapshot, t.symbol) == simulator.bar_price(simulator.bar_on(bars[t.symbol], entry), "open")
        assert venue_for("backtest", config).missing(snapshot, t.symbol) is None
    assert venue_for("backtest", config).price_of(snapshot, "XOM") == Decimal("169.5300") and venue_for("backtest", config).price_of(snapshot, "EOG") == Decimal("156.5100")
    nowhere = test_tickets[0].model_copy(update={"symbol": "SHEL.L"})
    gone = venue_for("backtest", config).execute(nowhere, nowhere.shares, snapshot)
    assert (gone.status, gone.reject_code, gone.reason) == ("rejected", "price_unavailable", "SHEL.L has no stored bar on 2026-11-16") and venue_for("backtest", config).missing(snapshot, "SHEL.L") == gone.reason
    early = Prices("backtest", date(2026, 11, 30), bars=bars, entry_date=None)                  # the next session of a decision made after the last stored day: not stored yet, the orders wait
    waiting = venue_for("backtest", config).execute(test_tickets[0], test_tickets[0].shares, early)
    assert (waiting.status, waiting.reason) == ("active", "waiting for the next session: no daily bar is stored from the entry day on yet") and venue_for("backtest", config).missing(early, "XOM") is None


def test_the_venues_ask_the_account_manager_for_the_one_snapshot_and_the_backtest_follows_each_purchase_by_the_plan(friday_close):
    """`venue.prices` through the REAL Account Manager over the real Data Clerk (the simulated Friday: its quotes are the stored closes, Finnhub is never asked): the paper snapshot holds the closes, the backtest snapshot the bars and the entry day.
    `venue.follow` is `simulator.follow` on the real plan: XOM's stop of the plan (169.3786, the 50-day mean) was touched on the entry bar itself, EOG's (150.028) two sessions later; a venue with no later bars follows nothing."""
    config, am, errors = load_config(), manager(), []
    paper = venue_for("paper", config).prices(am, SYMBOLS, MOMENT, errors)
    assert (paper.mode, paper.trade_date, paper.quotes, paper.failed, paper.unavailable, errors) == ("paper", FRIDAY, {s: close_of(s) for s in SYMBOLS}, "", {}, []) and paper.order_id
    back = venue_for("backtest", config).prices(am, SYMBOLS, MOMENT, errors)
    assert (back.mode, back.entry_date, back.trade_date, back.failed, back.unavailable, errors) == ("backtest", FIRST_SESSION, FIRST_SESSION, "", {}, []) and set(back.bars) == {"XOM", "EOG", "SPY"} and back.order_id
    assert back.bars["XOM"][0]["date"] == "2026-11-13" and venue_for("paper", config).prices(am, [], MOMENT, errors).quotes == {} and venue_for("backtest", config).prices(am, [], MOMENT, errors).bars == {}      # nothing to judge asks nothing

    order = real_order()
    am.perform_trades(order, BOB, trader(mode="backtest"), MOMENT)
    bars, recs = real_bars(*SYMBOLS), {r["identity"]["record_id"]: r["trade"] for r in order.recommendations}
    assert [(t["symbol"], t["status"], t["fill"]["date"], t["fill"]["price"]) for t in order.trades] == [("XOM", "filled", "2026-11-16", Decimal("169.5300")), ("EOG", "filled", "2026-11-16", Decimal("156.5100"))]
    for t, (reason, day, price) in zip(order.trades, [("stop", "2026-11-16", Decimal("169.3786")), ("stop", "2026-11-18", Decimal("150.0280"))]):
        plan, fill = recs[t["record_id"]], t["fill"]
        by_hand = simulator.follow(fill["price"], fill["shares"], fill["date"], bars[t["symbol"]], price_door(plan["stop_price"]), price_door(plan["target_price"]), plan["time_stop_date"], config.costs.slippage_bps, bars["SPY"])
        assert (by_hand.exit_reason, by_hand.exit_date, by_hand.exit_price) == (reason, day, price)
        assert t["outcome"]["exit_date"] == day and t["outcome"]["exit_price"] == price and t["outcome"]["exit_reason"] == reason          # what the Trader's step 9 asked the venue for
    from prometheus.agents.trader.record import Trade
    filled = Trade.model_validate(order.trades[1])
    assert venue_for("backtest", config).follow(filled, recs[filled.record_id], back).model_dump() == filled.outcome.model_dump() and venue_for("paper", config).follow(filled, recs[filled.record_id], paper) is None


def test_an_unknown_mode_is_an_error_and_a_venue_that_places_real_orders_is_refused(monkeypatch):
    """`venue_for` is the one place a mode becomes a venue: paper and backtest are the two simulated ones; anything else is the Trader's own sentence. A venue whose `simulated` is False is refused while PAPER_TRADING is true, and also when it is not
    registered, and made only when PAPER_TRADING is off AND it is registered (none is: live trading is not built, and the Trader's first check still refuses a run without PAPER_TRADING)."""
    config = load_config()
    assert [(v.name, v.simulated, v.books, v.keeps_ledger) for v in (venue_for("paper", config), venue_for("backtest", config))] == [("paper", True, True, True), ("backtest", True, False, False)]
    assert venue.REGISTERED == {} and isinstance(venue_for("paper", config), PaperVenue) and isinstance(venue_for("backtest", config), BacktestVenue)
    for mode in ("live", "alpaca", "Paper", "", None, 3):
        with pytest.raises(ValueError, match="the Trader's mode must be paper or backtest, not .*live trading is not built, and is refused here"):
            venue_for(mode, config)

    class Broker:
        name, simulated, books, keeps_ledger = "alpaca", False, True, True

        def __init__(self, config):
            self.config = config

    monkeypatch.setitem(venue.REGISTERED, "alpaca", Broker)
    monkeypatch.setenv("PAPER_TRADING", "true")
    with pytest.raises(ValueError, match="places real orders"):
        venue_for("alpaca", config)                                      # registered, but PAPER_TRADING gates everything
    monkeypatch.setenv("PAPER_TRADING", "false")
    assert isinstance(venue_for("alpaca", config), Broker)
    monkeypatch.setattr(venue, "REGISTERED", {})
    monkeypatch.setitem(venue.SIMULATED, "sneaky", Broker)               # a venue that does not simulate, filed with the simulated ones: not registered, so refused whatever PAPER_TRADING says
    with pytest.raises(ValueError, match="places real orders"):
        venue_for("sneaky", config)


def test_the_funnel_asks_its_venue_and_no_mode_is_compared_in_the_broker_or_the_steps():
    """A source read: `broker.py`, `execute.py` and the model's tools hold no comparison of a mode with a word and no "paper" or "backtest" in code (a docstring is not code); the one `PaperBroker` made with a mode has the venue of that mode and
    takes another when it is given one; the old homes of the branches are gone."""
    def code_strings(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(node.body[0].value) for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
        return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings and n.value in ("paper", "backtest", "live")]

    def compares_a_mode(path: Path) -> list[int]:
        found = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Compare):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                if "mode" in names and any(isinstance(c, ast.Constant) and isinstance(c.value, str) for c in [node.left, *node.comparators]):
                    found.append(node.lineno)
        return found

    for path in (TRADER / "broker.py", TRADER / "execute.py", ROOT / "prometheus" / "tools" / "trader.py"):
        assert code_strings(path) == [] and compares_a_mode(path) == [], path.name
    assert "venue_for" in (TRADER / "venue.py").read_text(encoding="utf-8") and "venue_for" in (TRADER / "execute.py").read_text(encoding="utf-8")
    assert not hasattr(trader_tools, "market_price") and not hasattr(execute, "_plan_price") and not hasattr(execute, "_quoted")

    config, found = real_tickets("backtest")
    broker = PaperBroker(config, "backtest", manager(), real_order(), Prices("backtest", FIRST_SESSION), found, lambda t: (None, ""), MOMENT)
    assert isinstance(broker.venue, BacktestVenue) and broker.mode == "backtest"
    own = PaperVenue(config)
    assert PaperBroker(config, "backtest", manager(), real_order(), Prices("backtest", FIRST_SESSION), found, lambda t: (None, ""), MOMENT, venue=own).venue is own
    with pytest.raises(ValueError, match="live trading is not built"):
        PaperBroker(config, "live", manager(), real_order(), Prices("backtest", FIRST_SESSION), found, lambda t: (None, ""), MOMENT)
