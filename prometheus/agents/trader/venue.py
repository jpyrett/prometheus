"""The venue: WHERE an order goes (design choices entry 128; design/proving_ground_spec.md section 6; the amendment at the end of design/trader_v1_final_spec.md).

    venue = venue_for("backtest", config)                    # the ONE place that maps a mode to a venue; an unknown mode, or a venue that places real orders, is a ValueError
    prices = venue.prices(account, ["XOM", "SPY"], now, errors)       # step 4: the ONE snapshot of prices, through the Account Manager (`account.request_prices` / `account.request_bars`)
    execution = venue.execute(trade, trade.shares, prices)   # the price rules for ONE trade against that snapshot: filled, active (it rests) or rejected, by the venue's own rules
    outcome = venue.follow(trade, plan, prices)               # step 9: how a filled purchase went through the later bars (the backtest venue); None where there are no later bars

Before this file `PaperBroker._simulate` branched on a `mode` string, and so did `execute._prices`, `_mark_unavailable`, `_outcomes`, `_snapshot` and `tools.trader.market_price`: five places a real broker would have had to
be cut into. A venue is those five in one object. The Trader's funnel (`broker.py`: permission, the handshake with the Account Manager `approve_trade` -> fill -> `verify_fill` -> `record_fill`, one trade at a time, never
retried) and its ten steps (`execute.py`) stay what they were and ask the venue for the price and the fill; nothing else in the desk learns which venue it is. A BEHAVIOUR-FREE change: the arithmetic of both venues is the
simulator's (`fill_paper`, `fill_backtest`, `follow`), untouched.

    PaperVenue      simulated, the current quote of each symbol (Finnhub, through the Account Manager), `simulator.fill_paper`; the fills are booked in the account book when the run asks for it; every trade becoming final is a line of the trades ledger
    BacktestVenue   simulated, the stored daily bars (`simulator.entry_date`, `fill_backtest`, `follow`): a fill at the next open, the purchase followed by its plan; never booked, no ledger

The attributes say what a caller may rely on: `name` (the mode as a trade and a run record it), `simulated` (True for both venues built now), `books` (the Account Manager may book the fills of this venue in the Postgres account
book: a backtest never does), `keeps_ledger` (a trade that became filled or rejected is a line of the trades ledger). `Prices.mode` is the venue's `name`.

A LIVE VENUE (not built; the list of what it still needs is in the amendment of the Trader's spec): `prices` = the broker's own quote, `execute` = submit the shares and wait for the fill report, `simulated = False`, `books = True`.
`venue_for` refuses it unless PAPER_TRADING is off AND it is registered in `REGISTERED` (nothing is), and the Trader's own first check, `PAPER_TRADING is true`, still refuses every run: nothing here can place a real order.

THE TRADER REACHES THE WORLD ONLY THROUGH THE ACCOUNT MANAGER (rule T5): a venue is given the `account` (the Account Manager) and asks it for quotes and bars (`request_prices`, `request_bars`); it imports nothing of the Data Clerk.
PRICES ARE `Decimal` (entry 112): a quote crosses the price door once, in `quoted`; a stored bar's prices cross it inside the fill rules (`simulator.bar_price`); the Advisor's stop and target cross it in `plan_price`.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Protocol

from prometheus.agents.trader import simulator
from prometheus.agents.trader.config import TraderConfig, paper_trading_enabled
from prometheus.agents.trader.record import MODES, Outcome, Trade
from prometheus.agents.trader.simulator import Execution
from prometheus.common.market_calendar import eastern_date
from prometheus.common.money import price as price_door
from prometheus.common.work_order import MAX_BAR_SYMBOLS, MAX_QUOTE_SYMBOLS


@dataclass
class Prices:
    """ONE snapshot of prices: what step 4 of the Trader asked the Account Manager for, once. `trade_date` is the date of every fill of the run (paper: the Eastern date of the run; backtest: the entry date).
    Paper: `quotes` {symbol: price}, a symbol with no usable quote is simply not there. Backtest: `bars` {symbol: [bar, ...]} and `entry_date`, None when no bar is stored from the calendar's entry day on (the future is
    not stored yet: orders wait). (It lived in `broker.py` until entry 128, which still re-exports it.)"""
    mode: str
    trade_date: date
    quotes: dict[str, Decimal] = field(default_factory=dict)
    bars: dict[str, list[dict]] = field(default_factory=dict)
    entry_date: date | None = None
    order_id: str | None = None                 # the request id of the quote or bars order the Account Manager made
    failed: str = ""                            # why no price could be had at all (the request failed or was refused): every order is then `price_unavailable`, which is not the same as "the future is not stored yet"
    unavailable: dict[str, str] = field(default_factory=dict)   # the request is made in groups of at most 30 symbols; when one group failed and the others answered, each symbol of that group is here with why: `price_unavailable`


class Venue(Protocol):
    """What the Trader's funnel and its ten steps need of the place an order goes."""
    name: str            # "paper" | "backtest" | later a broker's name: the mode as a trade and a run record it (`Prices.mode`)
    simulated: bool      # True for both venues built now
    books: bool          # the Account Manager may book this venue's fills in the Postgres account book (a backtest never does)
    keeps_ledger: bool   # a trade that became filled or rejected in a run is a line of the trades ledger

    def prices(self, account, symbols: list[str], now: datetime, errors: list[str]) -> Prices:
        """Step 4: the ONE snapshot of prices for `symbols`, asked of the Account Manager (`account`) in groups. What went wrong is a sentence appended to `errors` (the run's own list) and in the snapshot's `failed` or `unavailable`."""

    def price_of(self, prices: Prices, symbol: str) -> Decimal | None:
        """The price a trade stands at in this snapshot (what the model is shown, what the funding question is asked at): None when there is none."""

    def missing(self, prices: Prices, symbol: str) -> str | None:
        """Why this snapshot holds no price for `symbol` (a sentence), or None when it does, or when it is only that the future is not stored yet (the orders then wait)."""

    def execute(self, trade: Trade, shares: Decimal, prices: Prices) -> Execution:
        """The price rules for ONE trade against the snapshot: `filled` at a price, `active` (it rests) or `rejected` with a code of the Trader's closed list. `shares` is what a real venue would submit; a simulated venue's rules
        decide the price and the status only, and the shares are the Account Manager's approval (the handshake of entry 81)."""

    def follow(self, trade: Trade, plan: dict, prices: Prices) -> Outcome | None:
        """Step 9: how a filled purchase went through the bars after its entry, by the Advisor's own `plan` (the recommendation's `trade`): an `Outcome`; None for a venue that has no later bars to follow it through."""


def groups(symbols: list[str], size: int) -> list[list[str]]:
    """`symbols` in order, cut into groups of at most `size`: a quote order and a bars order each take at most 30 symbols (`MAX_QUOTE_SYMBOLS`, `MAX_BAR_SYMBOLS`)."""
    return [symbols[i:i + size] for i in range(0, len(symbols), size)]


def quoted(quote: dict) -> Decimal | None:
    """The price of one quote (a dict with the `price` the API client gave) through THE price door: a Decimal at four places above zero, or None."""
    try:
        return price_door(quote.get("price"))
    except ValueError:
        return None


def plan_price(measured) -> Decimal | None:
    """The Advisor's stop or target (its float) through THE price door, once; None when the plan has none (a capital-return plan has no target)."""
    try:
        return price_door(measured)
    except ValueError:
        return None


def snapshot(prices: Prices, why: dict[tuple[str, ...], str], answered: bool, errors: list[str], say_failed: bool) -> Prices:
    """The groups that failed (`why`, by their symbols) turned into the snapshot's verdict. None answered: `failed`, the reason once (the same sentence however many groups asked; `say_failed`: it is also a sentence on `errors`).
    Some answered: each symbol of a group that did not is `unavailable`, and `errors` says which group and why."""
    if not why:
        return prices
    if not answered:
        prices.failed = "; ".join(dict.fromkeys(why.values()))
        if say_failed:
            errors.append(prices.failed)
        return prices
    for group, reason in why.items():
        errors.append(f"{reason} (for {', '.join(group)})")
        prices.unavailable.update({symbol: reason for symbol in group})
    return prices


class PaperVenue:
    """The current quote of each symbol: today's paper branch, unchanged (`simulator.fill_paper`)."""
    name, simulated, books, keeps_ledger = "paper", True, True, True

    def __init__(self, config: TraderConfig):
        self.config = config

    def prices(self, account, symbols: list[str], now: datetime, errors: list[str]) -> Prices:
        today = eastern_date(now)
        if not symbols:
            return Prices(self.name, today)
        quotes, ids, why = {}, [], {}                     # why: the reason of each group that failed, by its symbols
        for group in groups(symbols, MAX_QUOTE_SYMBOLS):
            try:
                answer = account.request_prices(group)
            except Exception as error:
                why[tuple(group)] = f"the quotes could not be fetched ({type(error).__name__})"
                continue
            ids.append(answer.request_id)
            found = {s: quoted(q) for s, q in (answer.data.quotes if answer.data else {}).items() if isinstance(q, dict) and quoted(q) is not None}      # a quote is the Data Clerk's float: it becomes a Decimal price here, once
            quotes.update(found)
            for text in (answer.data.problems if answer.data else answer.errors):
                errors.append(str(text)[:200])
            if not found and answer.status != "data_ready":
                why[tuple(group)] = "the quotes could not be fetched: " + ("; ".join(str(e)[:80] for e in answer.errors[:2]) or "the Data Clerk returned none")
        return snapshot(Prices(self.name, today, quotes=quotes, order_id=", ".join(ids) or None), why, answered=len(why) < len(groups(symbols, MAX_QUOTE_SYMBOLS)), errors=errors, say_failed=False)

    def price_of(self, prices: Prices, symbol: str) -> Decimal | None:
        return prices.quotes.get(symbol)                  # a Decimal: it passed the price door when it was fetched

    def missing(self, prices: Prices, symbol: str) -> str | None:
        return None if symbol in prices.quotes else f"there is no usable quote for {symbol}"

    def execute(self, trade: Trade, shares: Decimal, prices: Prices) -> Execution:
        return simulator.fill_paper(trade.side, trade.order_type, trade.limit_price, prices.quotes.get(trade.symbol), trade.limit_low, trade.limit_high, self.config.costs.slippage_bps)

    def follow(self, trade: Trade, plan: dict, prices: Prices) -> Outcome | None:
        return None


class BacktestVenue:
    """The stored daily bars: today's backtest branch, unchanged (`simulator.entry_date`, `fill_backtest`, `follow`)."""
    name, simulated, books, keeps_ledger = "backtest", True, False, False

    def __init__(self, config: TraderConfig):
        self.config = config

    def prices(self, account, symbols: list[str], now: datetime, errors: list[str]) -> Prices:
        today = eastern_date(now)
        if not symbols:
            return Prices(self.name, today)
        wanted = list(dict.fromkeys([*symbols, self.config.backtest.benchmark]))
        bars, ids, why = {}, [], {}
        for group in groups(wanted, MAX_BAR_SYMBOLS):
            try:
                answer = account.request_bars(group, today, today + timedelta(days=self.config.backtest.max_follow_days))
            except Exception as error:
                why[tuple(group)] = f"the stored bars could not be read ({type(error).__name__})"
                continue
            ids.append(answer.request_id)
            if answer.status != "data_ready" or answer.data is None:
                reasons = "; ".join(str(e)[:100] for e in answer.errors[:2]) or "the Data Clerk returned no bars"
                why[tuple(group)] = f"the stored bars could not be read: {reasons}"
                continue
            bars.update({s: list(series) for s, series in answer.data.bars.items()})
        entry = simulator.entry_date(now, bars)
        return snapshot(Prices(self.name, entry or today, bars=bars, entry_date=entry, order_id=", ".join(ids) or None), why, answered=len(why) < len(groups(wanted, MAX_BAR_SYMBOLS)), errors=errors, say_failed=True)

    def price_of(self, prices: Prices, symbol: str) -> Decimal | None:
        bar = simulator.bar_on(prices.bars.get(symbol), prices.entry_date) if prices.entry_date else None
        return simulator.bar_price(bar, "open") if bar else None      # a stored bar's float, through the price door

    def missing(self, prices: Prices, symbol: str) -> str | None:
        if prices.entry_date is not None and simulator.bar_on(prices.bars.get(symbol), prices.entry_date) is None:
            return f"{symbol} has no stored bar on {prices.entry_date.isoformat()}"
        return None

    def execute(self, trade: Trade, shares: Decimal, prices: Prices) -> Execution:
        if prices.entry_date is None:
            return Execution("active", "waiting for the next session: no daily bar is stored from the entry day on yet")
        bar = simulator.bar_on(prices.bars.get(trade.symbol), prices.entry_date)
        if bar is None:
            return Execution("rejected", f"{trade.symbol} has no stored bar on {prices.entry_date.isoformat()}", reject_code="price_unavailable")
        return simulator.fill_backtest(trade.side, trade.order_type, trade.limit_price, bar, trade.limit_low, trade.limit_high, self.config.costs.slippage_bps)

    def follow(self, trade: Trade, plan: dict, prices: Prices) -> Outcome | None:
        return simulator.follow(trade.fill.price, trade.fill.shares, trade.fill.date, prices.bars.get(trade.symbol, []), plan_price(plan.get("stop_price")), plan_price(plan.get("target_price")), plan.get("time_stop_date"),
                                self.config.costs.slippage_bps, prices.bars.get(self.config.backtest.benchmark))


SIMULATED: dict[str, type] = {"paper": PaperVenue, "backtest": BacktestVenue}      # the two venues of this version: both simulate
REGISTERED: dict[str, type] = {}                                                     # real brokers: NONE is built. A venue whose `simulated` is False is only ever made when it is here AND PAPER_TRADING is off


def venue_for(mode, config: TraderConfig) -> Venue:
    """THE one place that maps a mode to a venue. An unknown mode ("live", "real", "Paper", "") is a ValueError in the Trader's own words (a refused run says them). A venue whose `simulated` is False is a ValueError too, unless
    PAPER_TRADING is off AND the venue is in `REGISTERED` (none is): the paper gate refuses everything that is not a simulation."""
    name = mode if isinstance(mode, str) else None
    kind = SIMULATED.get(name) or REGISTERED.get(name)
    if kind is None:
        raise ValueError(f"the Trader's mode must be {' or '.join(MODES)}, not {str(mode)[:20]!r}: live trading is not built, and is refused here")
    venue = kind(config)
    if not venue.simulated and (paper_trading_enabled() or name not in REGISTERED):
        raise ValueError(f"the venue {name!r} places real orders: it is refused unless PAPER_TRADING is off and the venue is registered (none is: live trading is not built)")
    return venue
