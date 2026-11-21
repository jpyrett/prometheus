"""The venue seam (design choices entry 128; design/proving_ground_spec.md section 6): `prometheus/agents/trader/venue.py`.

REAL DATA ONLY. The orders are the tickets of the real session of 2026-11-13 (`tests/fixtures/session_real/bob_jones_oil.json`: XOM 8 and EOG 8 shares, decided again by the Account Manager's own step as `handshake_helpers.real_order` does), the quotes
are the stored closes of that Friday and the bars are the stored daily bars (`data/historical/day_aggs_v1`, read only) that the real Data Clerk gives the real Account Manager. The first session after the decision is Monday 2026-11-16: XOM opened
at 169.53 and EOG at 156.51 (read from the stored bars on 2026-11-20; two sessions later both were stopped out by their plans). A machine without those bars skips these tests. Nothing here calls a model, a quote API or a paid API.

The seam is a BEHAVIOUR-FREE change (every Trader test passed unchanged): what these tests add is that the two venues give EXACTLY what the two branches of `PaperBroker._simulate` and `execute._prices` gave (the simulator's own functions, called here
by hand with the same arguments), that a mode that is not paper or backtest is an error, that a venue that places real orders is refused, and that no `if mode ==` is left in the funnel.
"""
from datetime import date
from decimal import Decimal

import pytest
from handshake_helpers import BOB, FRIDAY, MOMENT, clerk, close_of, friday_close, manager, real_order, stored_close_available, trader  # noqa: F401  (friday_close is a fixture)

from prometheus.agents.trader import execute, simulator, tickets, venue
from prometheus.agents.trader.broker import PaperBroker, Prices
from prometheus.agents.trader.config import load_config
from prometheus.agents.trader.venue import BacktestVenue, PaperVenue, venue_for

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


