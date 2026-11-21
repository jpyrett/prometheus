"""WAITING LIMIT ORDERS: a direct order whose limit price is not met waits as a DAY order, with its cash or its shares SET ASIDE (design choices entry 140; design/limit_orders_spec.md, Part B, rules W1 to W8).

    plan = orders.plan_waiting(entry, price, portfolio, now, fractions)       # a `Plan` (what to set aside, until when), or the sentence that says why the order cannot wait
    session, expires = orders.session_for(now)                                # the trading day a day order placed now is for, and the moment it expires (that session's close)
    watcher = orders.Watcher(store)                                           # the web server's watcher: `tick()` expires what is over and fills what the quotes reached; `start()` runs it in a thread

    python -m prometheus.agents.account_manager.orders --account N [--list | --cancel ID | --check]

WHICH ORDERS WAIT (W1). Only a DIRECT order, the user's own (entry 87): a BUY with a ceiling (`max_price`: "buy 10 MU at $90 or less") whose quote is ABOVE it, and a SALE with a floor (`min_price`: "sell 10 MSFT at $600 or more") whose quote is BELOW it.
Instead of "not placed" it becomes an OPEN ORDER (`trading.open_orders`). A direct order whose quote already meets its limit fills at once, as ever. On the recommendation path a limit stays a filter: nothing from a recommendation waits. A waiting
order needs an amount to set aside: shares the user typed, or dollars; a purchase with neither is not placed. Never a live broker: the paper broker only.

A DAY ORDER (W2). It expires at the close of the trading session it was placed in (16:00 Eastern, 13:00 on an early-close day: `market_calendar`); placed when that session is over (after the close, a weekend, a holiday) it is for the NEXT trading
session. Never carried to another day. `session_for` is that rule and `until_text` says it to the user ("4:00 pm ET today", "4:00 pm ET Monday 2026-11-23").

SET ASIDE (W3). A waiting purchase sets aside its most possible cost (the shares typed times the limit, or the dollars of an order in dollars, to the cent); a waiting sale sets aside its shares. What is set aside is not available to anything
else: the account book's ONE DOOR is `AccountStore.available_cash` / `available_shares` (`list_portfolio` returns the available cash as `cash_avail`, and the whole holding with the reservations beside it), and every reader of what may be spent or
sold reads it (the pot, `approve_trade`, RaiseFunds' holdings, `remove_funds`, the sizing of a direct order, `show_portfolio`); the book's own `book_trade` and `place_open_order` re-check under the account row lock. A placement that would set aside
more than is available is refused at placement.

THE WATCHER (W5). In the web server, started with the app and stopped with it (`Watcher.start` / `stop`): every `orders.watch_seconds` (`configs/account_manager_v1.json`, 60) it EXPIRES every order whose `expires_at` has passed (also the first thing it
does, so nothing stale survives a restart), and DURING THE REGULAR SESSION ONLY it quotes the symbols of the open orders with ONE `request_quotes` through the Data Clerk (the Account Manager's `request_prices`: Finnhub, in the licence) and FILLS each
order whose quote meets its limit (a buy: quote at or under the ceiling; a sale: at or over the floor) through the EXISTING funnel: a one-ticket work order run by the Trader with no model, the paper venue (a fill at the quote, which is inside the
limit), `approve_trade` (which reads the order's own hold as available to THIS order only), `verify_fill`, `record_fill` and the book's `book_trade` (which marks the order `filled` in the same transaction). There is no second booking path. The
Trader sees exactly the quote the order was judged by (`Snapshot`: the answer to the tick's one quote call). A buy in dollars buys `shares.buys(min(dollars, set aside), fill price)` (a fraction only when the book holds fractions, else whole shares).
A fill that FAILS is `rejected`, shown, never retried and its hold is released (entry 82). A quote that failed is tried again at the next tick: an outage never fills and never expires an order early. ONE watcher for each database: a Postgres advisory lock
(`AccountStore.watcher_lock`), so a second server does not check the same orders. Every fill, expiry, cancel and rejection is an event (`notify`: the page shows it) and a line of the audit log (`AUDIT_PATH`).

Nothing is built here that sizes or limits a trade: no size limit, no risk rule (AGENTS.md); the only checks are the funds and the shares, and the limit price the user wrote.
"""
import argparse
import contextvars
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from prometheus.agents.account_manager import funding
from prometheus.agents.account_manager.config import load_config
from prometheus.common import shares as sh
from prometheus.common.market_calendar import eastern_date, is_trading_day, load_calendar, next_trading_day, regular_close_utc, session_at
from prometheus.common.money import dumps, exact, price as price_door
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.work_order import DataResult, MAX_QUOTE_SYMBOLS, new_work_order, verify_quote_request
from prometheus.rag.config import REPO_ROOT

AUDIT_PATH = REPO_ROOT / "logs" / "open_orders_audit.jsonl"      # one line for each placement, fill, expiry, cancel and rejection (gitignored: `logs/`); a test gives the watcher a path of its own
EVENTS = ("placed", "filled", "expired", "cancelled", "rejected")
REASON_WORDS = {"buy": "or less", "sell": "or more"}      # the limit of a purchase is a ceiling, of a sale a floor


# ---------------------------------------------------------------- the day order: which session, until when

def in_regular_session(now: datetime) -> bool:
    """Whether the regular session of a trading day is open at `now` (09:30 to 16:00 Eastern, 13:00 on an early-close day): the only time the watcher fills an order."""
    return session_at(now).state == "regular"


def session_for(now: datetime) -> tuple[date, datetime]:
    """(the trading day a DAY order placed at `now` is for, the moment it expires). A trading day whose regular close has not come yet is today's session; after the close, on a weekend or on a holiday it is the NEXT trading day's. It expires at that
    session's close (16:00 Eastern, 13:00 on an early-close day), as an aware UTC datetime. Raises ValueError for a `now` with no timezone."""
    day = eastern_date(now)
    session = day if is_trading_day(day) and now < regular_close_utc(day) else next_trading_day(day)
    return session, regular_close_utc(session)


def until_text(expires_at: datetime, now: datetime) -> str:
    """When a waiting order ends, as the user is told: "4:00 pm ET today" when it is today's close, else "4:00 pm ET Monday 2026-11-23"."""
    local = expires_at.astimezone(ZoneInfo(load_calendar().timezone))
    clock = f"{local.hour % 12 or 12}:{local.minute:02d} {'am' if local.hour < 12 else 'pm'} ET"
    return f"{clock} today" if local.date() == eastern_date(now) else f"{clock} {local:%A} {local.date().isoformat()}"


def meets(side: str, price, limit) -> bool:
    """Whether the quote meets the limit of a waiting order: a purchase at or under its ceiling, a sale at or over its floor."""
    return exact(price, "a price") <= exact(limit, "a limit") if side == "buy" else exact(price, "a price") >= exact(limit, "a limit")


def limit_of(entry: dict) -> Decimal | None:
    """The limit price of a user's order that can make it a limit order: the ceiling (`max_price`) of a purchase, the floor (`min_price`) of a sale; None when it has none. A sale is never limited by a ceiling, nor a purchase by a floor here
    (a purchase with a floor is placed only while the quote is at or over it: it does not wait, entry 140, F3)."""
    key = "max_price" if entry.get("side") == "buy" else "min_price"
    return None if entry.get(key) is None else price_door(entry[key])


def waits(entry: dict, price) -> bool:
    """Whether this order, priced at `price`, is one that WAITS (W1): it has a limit and the price has not reached it."""
    limit = limit_of(entry)
    return limit is not None and price is not None and not meets(entry["side"], price, limit)


# ---------------------------------------------------------------- the plan of an order that waits

@dataclass(frozen=True)
class Plan:
    """What a waiting order is, before it is placed: the side, the symbol, the limit, the shares (typed) or the dollars (an order in dollars), what it sets aside, and the session it is for."""
    side: str
    symbol: str
    limit: Decimal
    price: Decimal                      # the quote it was judged at
    shares: Decimal | None
    dollars: Decimal | None
    reserve_cash: Decimal               # a purchase's most possible cost
    reserve_shares: Decimal             # a sale's shares
    estimate: Decimal                   # the shares it would buy at its limit (a purchase in dollars), or the shares it holds
    session: date
    expires_at: datetime
    funded_by: str | None = None        # entry 143: "words" or "setting" when the user permitted selling holdings to pay for the rest of this purchase WHEN ITS PRICE IS REACHED (nothing is sold while it waits)
    shortfall: Decimal = Decimal("0.00")      # what the purchase would cost at its limit less the cash that is free: what the sales would have to raise at the very most

    def as_dict(self) -> dict:
        """The plan as it is written on the decision (plain values: `trade_decisions` is written through the one encoder)."""
        return {"side": self.side, "symbol": self.symbol, "limit": self.limit, "price": self.price, "shares": self.shares, "dollars": self.dollars, "reserve_cash": self.reserve_cash, "reserve_shares": self.reserve_shares,
                "estimate": self.estimate, "session": self.session.isoformat(), "expires_utc": self.expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "funded_by": self.funded_by, "shortfall": self.shortfall}


@dataclass(frozen=True)
class Refusal:
    """Why an order cannot wait: the decision it becomes (`no_funds`: the cash or the shares are not there to set aside; `no_system_denied`: the order is not one that can wait) and the sentence for the user."""
    will_trade: str
    reason: str


def plan_waiting(entry: dict, price, portfolio: Portfolio, now: datetime, fractions: bool, funded_by: str | None = None) -> Plan | Refusal:
    """The plan of a direct order that waits for its limit (W1 and W3), or the `Refusal` that says why it cannot wait. `funded_by` (entry 143, "words" or "setting": the Account Manager decides it, from the user's permission) lets a PURCHASE the free cash does not
    cover wait anyway: it sets aside the cash that is free (none, if none is), carries the permission, and nothing is sold until its price is reached (`AccountManager.fill_open_order`). `entry` is the user's own record on the work order, `price` its quote now, `portfolio` the account as the book reads it
    (the AVAILABLE cash, the whole holding and what is already set aside), `fractions` whether the book holds a fraction of a share. Nothing is changed. The book re-checks everything under its row locks when the order is placed
    (`AccountStore.place_open_order`): this is the fast, readable check."""
    side, symbol = entry["side"], entry["symbol"]
    limit = limit_of(entry)
    if limit is None:
        return Refusal("no_system_denied", f"{symbol} has no limit price to wait for")
    price = exact(price, "a price")
    session, expires = session_for(now)
    shares = sh.quantity(entry["shares"]) if entry.get("shares") is not None and entry.get("how_many") in (None, "typed") else None
    dollars = exact(entry["dollars"], "dollars") if entry.get("dollars") is not None and shares is None else None
    if side == "buy":
        if shares is None and dollars is None:
            return Refusal("no_system_denied", f"{symbol} is at ${price:,.2f}, above your limit of ${limit:,.2f}, and you gave no number of shares or dollars: an order that waits needs an amount to set aside, so nothing is placed. "
                                               f"Say how many shares, or how much money, to buy at ${limit:,.2f} or less")
        if shares is not None and not fractions and not sh.whole(shares):
            return Refusal("no_system_denied", f"you asked to buy {sh.text(shares)} shares of {symbol}, which is a fraction of a share, and this account book holds whole shares only: nothing is placed")
        reserve = trade_amount(shares, limit) if shares is not None else dollars
        free = portfolio.cash_avail
        short = Decimal("0.00")
        if reserve > free and funded_by is not None:      # entry 143: the free cash is set aside and the rest is raised by selling when the price is reached
            short, reserve = reserve - free, max(free, Decimal("0.00"))
        elif reserve > free:
            held = f", ${portfolio.reserved_cash:,.2f} of it already set aside for waiting orders" if portfolio.reserved_cash else ""
            what = f"{sh.text(shares)} share{'s' if shares != 1 else ''} at your limit of ${limit:,.2f} is ${reserve:,.2f}" if shares is not None else f"your ${reserve:,.2f}"
            return Refusal("no_funds", f"to wait, {symbol} sets aside the most it can cost: {what}, and the cash available is ${free:,.2f}{held}: the order cannot wait, and nothing is placed")
        estimate = shares if shares is not None else (sh.buys(dollars, limit) if fractions else Decimal(int(dollars // limit)))
        if estimate <= 0:
            return Refusal("no_funds", f"your ${dollars:,.2f} does not buy even one share of {symbol} at your limit of ${limit:,.2f}: nothing is placed")
        return Plan(side, symbol, limit, price, shares, dollars, reserve, Decimal(0), estimate, session, expires, funded_by if short > 0 else None, short)
    if shares is None:
        return Refusal("no_system_denied", f"you named a sale of {symbol} and no number of shares: say how many to sell; the Account Manager does not choose how much of a holding to sell")
    free = portfolio.free_shares(symbol)
    if shares > free:
        held = portfolio.position(symbol)
        held_text = f"the account holds {sh.text(held.shares)}" if held else f"the account holds no {symbol}"
        aside = f", {sh.text(portfolio.reserved_shares.get(symbol, Decimal(0)))} of them already set aside for waiting orders" if portfolio.reserved_shares.get(symbol) else ""
        return Refusal("no_funds", f"to wait, the sale sets aside its {sh.text(shares)} share{'s' if shares != 1 else ''} of {symbol}, and {held_text}{aside}, so {sh.text(free)} can be set aside: the order cannot wait, and nothing is placed")
    sellable = portfolio.sellable_shares(symbol, session)
    if shares > sellable:
        return Refusal("no_funds", f"you asked to sell {sh.text(shares)} shares of {symbol} and only {sh.text(sellable)} can be sold in the session it waits for, {session.isoformat()} (a share cannot be sold the day it was bought: T+1): "
                                   "the order is placed as you typed it or not at all, so nothing is placed")
    return Plan(side, symbol, limit, price, shares, None, Decimal("0.00"), shares, shares, session, expires)


def waiting_sentence(plan: Plan, now: datetime, company: str | None = None) -> str:
    """What the user is told when the order waits (W6): `Waiting: buy 10 MU at $90.00 or less (now $95.12), $900.00 set aside, until 4:00 pm ET today.`"""
    what = (f"buy ${plan.dollars:,.2f} of {plan.symbol}" if plan.dollars is not None else
            f"{plan.side} {sh.text(plan.shares)} {plan.symbol}")
    held = f"${plan.reserve_cash:,.2f} set aside" if plan.side == "buy" else f"{sh.text(plan.reserve_shares)} share{'s' if plan.reserve_shares != 1 else ''} set aside"
    raising = (f" If the price is reached I will sell holdings to pay for the rest (up to ${plan.shortfall:,.2f} at your limit; {funding.how_words(plan.funded_by)}) and then buy; nothing is sold while it waits." if plan.funded_by else "")
    return f"Waiting: {what} at ${plan.limit:,.2f} {REASON_WORDS[plan.side]} (now ${plan.price:,.2f}), {held}, until {until_text(plan.expires_at, now)}.{raising}"


def order_line(order) -> str:
    """One waiting order as a line: `#5 buy 10 MU at $90.00 or less, $900.00 set aside, until 4:00 pm ET today` (`order` is an `OpenOrder`; the time is its own session's close, with its date)."""
    local = order.expires_at.astimezone(ZoneInfo(load_calendar().timezone))
    what = f"buy ${order.dollars:,.2f} of {order.symbol}" if order.dollars is not None else f"{order.side} {sh.text(order.shares)} {order.symbol}"
    held = f"${order.reserved_cash:,.2f} set aside" if order.side == "buy" else f"{sh.text(order.reserved_shares)} shares set aside"
    return f"#{order.order_id} {what} at ${order.limit_price:,.2f} {REASON_WORDS[order.side]}, {held}, {order.status}, until {local:%H:%M} ET {local:%a %Y-%m-%d}" + (" (sells holdings for the rest if the price is reached)" if getattr(order, "raise_funds_by", None) else "")


# ---------------------------------------------------------------- the watcher's one quote call, handed to the Trader

class Snapshot:
    """The Data Clerk's answer to the ONE quote order a watcher tick made, given to the Account Manager that fills an order as its Data Clerk: the Trader's price step then reads exactly the quote the order was judged by (and no second quote
    call is paid for, nor can the price move between the check and the fill). `request_quotes` answers a quote order the way the real Data Clerk does: the quotes of the symbols asked for, `data_failed` and named when one is not there.
    `fallback` (entry 143) is the real Data Clerk, given only to the Account Manager that fills an order the user permitted to RAISE FUNDS: RaiseFunds needs the quotes of the account's holdings too, which the tick did not ask for, so
    a symbol that is not in the snapshot is asked of it (the order's own symbol is always the snapshot's, so the price the order was judged at is the price it fills at). Without it a symbol that is not in the snapshot is a problem, as ever."""

    def __init__(self, quotes: dict[str, dict], fallback=None):
        self.quotes, self.fallback = quotes, fallback

    def request_quotes(self, order):
        reasons = verify_quote_request(order)
        if reasons:
            order.status, order.errors = "data_failed", [*order.errors, *reasons]
            return order
        wanted = [order.subject] if isinstance(order.subject, str) else list(order.subject)
        found = {symbol: self.quotes[symbol] for symbol in wanted if symbol in self.quotes}
        others = [symbol for symbol in wanted if symbol not in self.quotes]
        problems = [f"{symbol}: no quote in the watcher's snapshot" for symbol in others]
        if others and self.fallback is not None:      # the holdings RaiseFunds prices, for a funded fill (entry 143)
            asked = self.fallback.request_quotes(new_work_order(f"Current prices of {', '.join(others)}", requester="account_manager", action_type="fetch_quote", subject=others, live=True))
            if asked.data is not None:
                found.update({symbol: quote for symbol, quote in asked.data.quotes.items() if isinstance(quote, dict)})
            problems = [f"{symbol}: no quote could be had" for symbol in others if symbol not in found] + [str(p) for p in (asked.data.problems if asked.data is not None else asked.errors)][:3]
        order.data = DataResult(answer=", ".join(f"{s} {q['price']}" for s, q in found.items()), quotes=found, problems=problems, notes=["the quote the watcher judged the waiting order by"])
        order.errors = [*order.errors, *problems]
        order.status = "data_failed" if problems else "data_ready"
        return order


# ---------------------------------------------------------------- the audit log

def audit(record: dict, path=None) -> None:
    """Append one line to the orders' audit log (`AUDIT_PATH`, or `path`): what happened to a waiting order and when. A log that cannot be written is never a reason to stop: it returns quietly."""
    try:
        target = path or AUDIT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(dumps({"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **record}) + "\n")
    except (OSError, ValueError, TypeError):
        pass


# ---------------------------------------------------------------- the watcher

class Watcher:
    def __init__(self, store, *, clerk=None, clock=None, trader_factory=None, notify=None, audit_path=None, seconds: int | None = None, lock=None, accounts=None, model=None):
        """store: the account book (`AccountStore`). clerk: the Data Clerk that is asked for the one quote call of a tick (anything with `request_quotes`; the real one is built when first needed). clock: a callable that gives the aware moment
        (the desk's own clock, `trading_now`, by default; a test passes a fake). trader_factory: makes the Trader of one fill (the real one, no model, paper, by default). notify: called with each event (a dict: type, order_id, account_id, symbol,
        side, text, at). audit_path: the audit log. seconds: the pause between ticks (`orders.watch_seconds`). lock: the advisory lock (`store.watcher_lock()`). accounts: the account numbers whose orders this watcher looks at (None: every account, as the web server's watcher does; the
        command's `--check` and every test give the one account they are about, so that nothing else's order is ever expired or filled by them). model: the Account Manager's model, used ONLY to fill an order the user permitted to raise funds (entry 143:
        RaiseFunds is its tree of thought); None builds the real one when such an order is filled. Every other fill makes no model call."""
        self.model = model
        self.store, self.clerk, self._clock, self._trader_factory, self.notify = store, clerk, clock, trader_factory, notify
        self.accounts = None if accounts is None else frozenset(accounts)
        self.audit_path, self.seconds, self.lock = audit_path, seconds, lock
        self.errors: list[str] = []            # what went wrong at a tick, the kind only: a failed quote is tried again at the next tick
        self.owns_lock = False                  # True while this watcher holds the advisory lock (a second server's watcher waits and does nothing)
        self.loops = 0                          # how many times the thread went round (a tick, or a wait for the lock)
        self._stop, self._thread = threading.Event(), None

    # ---- the clock, the Trader, the Account Manager

    def clock(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        from prometheus.common.trading_clock import trading_now
        return trading_now()

    def _manager(self, account_id: int | None, clerk, with_model: bool = False):
        """An Account Manager over the book with no model (it makes no model call here) and this Data Clerk: for the tick's quote call (`account_id` None, no account needed) or to fill one order. `with_model` (entry 143: an order the
        user permitted to raise funds) gives it the real model (or this watcher's `model`), because RaiseFunds is a tree of thought."""
        from prometheus.agents.account_manager.account_manager import AccountManager
        kwargs = {} if account_id is None else {"store": self.store, "account_id": account_id}
        if with_model:
            return AccountManager(model=self.model, clerk=clerk, **kwargs)
        return AccountManager(model=object(), model_id="open-orders", clerk=clerk, **kwargs)

    def _trader(self):
        if self._trader_factory is not None:
            return self._trader_factory()
        from prometheus.agents.trader.trader import Trader
        return Trader(mode="paper", use_model=False)

    def _clerk(self):
        if self.clerk is None:
            from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
            self.clerk = DataClerk()
        return self.clerk

    # ---- one tick

    def tick(self, now: datetime | None = None) -> list[dict]:
        """One check: expire what is over, then (in the regular session only) quote the symbols of the open orders ONCE and fill each order whose quote meets its limit. Returns the events (each already sent to `notify` and the audit log).
        Never raises for a quote or a fill that failed: it is tried again at the next tick (a rejected fill is not)."""
        now = now or self.clock()
        events: list[dict] = []
        for account_id in self._accounts():
            for order in self.store.expire_open_orders(account_id, now):
                events.append(self._event("expired", order, f"Order #{order.order_id} expired at the close of its session: {order_words(order)} was not reached, and what it set aside is available again.", now))
        if not in_regular_session(now):
            return events
        waiting = [order for account_id in self._accounts() for order in self.store.open_orders(account_id)]
        symbols = sorted({order.symbol for order in waiting})
        if not symbols:
            return events
        quotes = self._quotes(symbols)
        for order in waiting:
            quote = quotes.get(order.symbol)
            price = None if quote is None else _price_of(quote)
            if price is None or not meets(order.side, price, order.limit_price):
                continue
            events.extend(self._fill(order, price, quotes, now))
        return events

    def _accounts(self) -> list[int]:
        """The accounts that have an open order and that this watcher looks at."""
        return [account_id for account_id in self.store.accounts_with_open_orders() if self.accounts is None or account_id in self.accounts]

    def _quotes(self, symbols: list[str]) -> dict[str, dict]:
        """{symbol: the quote} for the symbols that were quoted, ONE quote order for all of them (in groups of the most a quote order takes). A symbol that failed is left out and the tick leaves its orders for the next one."""
        quoter, quotes = self._manager(None, self._clerk()), {}
        for start in range(0, len(symbols), MAX_QUOTE_SYMBOLS):
            try:
                answered = quoter.request_prices(symbols[start:start + MAX_QUOTE_SYMBOLS])
            except Exception as error:
                self.errors.append(f"the quotes could not be fetched ({type(error).__name__})")
                continue
            if answered.data is not None:
                quotes.update({symbol: quote for symbol, quote in answered.data.quotes.items() if isinstance(quote, dict)})
        return quotes

    def _fill(self, order, price: Decimal, quotes: dict[str, dict], now: datetime) -> list[dict]:
        """Fill ONE order whose quote meets its limit, through the Account Manager and the Trader (see the module text). The order becomes `filled` in the book's own transaction; a fill that failed is `rejected`, with the reason, and not retried."""
        try:
            funded = order.side == "buy" and order.raise_funds_by is not None
            manager = self._manager(order.account_id, Snapshot(quotes, fallback=self._clerk() if funded else None), with_model=funded)      # an order permitted to raise funds may sell holdings, and RaiseFunds needs the model and the holdings' quotes
            work = manager.fill_open_order(order, price, self._trader(), now)
        except AssertionError:
            raise
        except Exception as error:      # nothing is known about the fill: the book is read to settle it, never guessed
            work, failure = None, f"the fill could not be made ({type(error).__name__})"
        else:
            failure = None
        try:
            now_order = self.store.get_open_order(order.account_id, order.order_id)
        except Exception as error:
            self.errors.append(f"order {order.order_id} could not be read again ({type(error).__name__})")
            return []
        if now_order is not None and now_order.status == "filled":
            trade = next((t for t in (work.trades if work is not None else []) if t.get("status") == "filled"), None)
            return [self._event("filled", now_order, filled_text(now_order, trade), now, trade=trade)]
        if now_order is None or now_order.status != "open":
            return []
        why = failure or _why_not_filled(work)
        try:
            closed = self.store.reject_open_order(order.account_id, order.order_id, why, now)
        except Exception as error:
            self.errors.append(f"order {order.order_id} could not be closed ({type(error).__name__})")
            return []
        return [] if closed is None else [self._event("rejected", closed, f"Order #{closed.order_id} was not filled and is closed: {why}. It will not be tried again, and what it set aside is available again.", now)]

    def _event(self, kind: str, order, text: str, now: datetime, trade: dict | None = None) -> dict:
        event = {"type": kind, "order_id": order.order_id, "account_id": order.account_id, "symbol": order.symbol, "side": order.side, "text": text, "at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        audit({"event": kind, "text": text, "order": order.as_dict(), **({"trade": {k: trade.get(k) for k in ("trade_id", "status", "symbol", "side", "shares", "fill")}} if trade else {})}, self.audit_path)
        if self.notify is not None:
            try:
                self.notify(event)
            except Exception:
                pass
        return event

    def note(self, kind: str, order, text: str, now: datetime | None = None) -> dict:
        """An event that did not come from a tick (a placement, or the user's cancel): sent to `notify` and the audit log like the others."""
        return self._event(kind, order, text, now or self.clock())

    # ---- the thread

    def start(self) -> None:
        """Start the watcher in a daemon thread: the first check at once (so that an order whose session is over expires at server start), then every `orders.watch_seconds`. A second `start` does nothing."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        context = contextvars.copy_context()      # the thread reads the clock and the trace of whoever starts it (the server's own: no day is bound, a quote is the live one), as every thread of the package does (entry 127)
        self._thread = threading.Thread(target=context.run, args=(self._run,), name="open-orders-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and let go of the advisory lock (a join of a few seconds at most)."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        if self.lock is not None:
            self.lock.release()
        self.owns_lock = False

    def _run(self) -> None:
        pause = self.seconds if self.seconds is not None else load_config().orders.watch_seconds
        lock = self.lock if self.lock is not None else self.store.watcher_lock()
        self.lock = lock
        while not self._stop.is_set():
            try:
                self.owns_lock = lock.try_acquire()
                if self.owns_lock:
                    self.tick()
            except Exception as error:      # the watcher never dies of one bad tick: the kind of the failure is kept and the next tick tries again
                self.errors.append(f"{type(error).__name__}")
                self.errors[:] = self.errors[-20:]
            self.loops += 1
            self._stop.wait(pause)


def _price_of(quote: dict) -> Decimal | None:
    """The price of a quote through THE price door (the Data Clerk's float): a Decimal at four places, or None."""
    try:
        return price_door(quote.get("price"))
    except ValueError:
        return None


def order_words(order) -> str:
    """What an order was to do, as a phrase: "buy 10 MU at $90.00 or less"."""
    what = f"buy ${order.dollars:,.2f} of {order.symbol}" if order.dollars is not None else f"{order.side} {sh.text(order.shares)} {order.symbol}"
    return f"{what} at ${order.limit_price:,.2f} {REASON_WORDS[order.side]}"


def filled_text(order, trade: dict | None) -> str:
    fill = (trade or {}).get("fill") or {}
    if not fill:
        return f"Order #{order.order_id} filled: {order_words(order)}."
    return (f"Order #{order.order_id} filled: {'bought' if order.side == 'buy' else 'sold'} {sh.text(fill['shares'])} {order.symbol} at ${price_door(fill['price']):,.2f} "
            f"(${exact(fill['amount'], 'an amount'):,.2f}), the waiting order you left: {order_words(order)}.")


def _why_not_filled(work) -> str:
    """Why a fill did not happen, from what the run recorded: the Trader's own sentence for the trade, else the first message on the work order."""
    if work is None:
        return "the fill was not made"
    trade = next((t for t in work.trades if isinstance(t, dict)), None)
    if trade is not None and trade.get("status") == "rejected":
        return " ".join(f"{trade.get('reject_code')}: {trade.get('reason')}".split())[:300]
    decision = next((d for d in work.trade_decisions if isinstance(d, dict)), None)
    if decision is not None and decision.get("will_trade") not in ("yes_user", "yes_auto"):
        return " ".join(str(decision.get("reason") or decision.get("will_trade")).split())[:300]
    return " ".join("; ".join(work.errors).split())[:300] or "the Trader did not place it"


# ---------------------------------------------------------------- the command

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The waiting limit orders of one saved account (design choices entry 140). --list and --cancel are free (they read and write only the local Postgres); --check quotes the symbols of the open orders (Finnhub) and fills the ones the quotes reached, once.")
    parser.add_argument("--account", required=True, help="the account number (a sample key is read-only here)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="list the account's orders (the default)")
    group.add_argument("--cancel", type=int, metavar="ID", help="cancel one waiting order: what it set aside is available again")
    group.add_argument("--check", action="store_true", help="one check, as the web server's watcher makes it: expire what is over, and (in the regular session) quote and fill what the quotes reached")
    args = parser.parse_args(argv)
    import psycopg

    from prometheus.persistence.accounts import open_account_store
    try:
        store = open_account_store()
        account_id = int(args.account) if args.account.isdigit() else store.account_id_for_seed(args.account)
        if args.check:
            watcher = Watcher(store, notify=lambda event: print(event["text"]), accounts=[account_id])
            events = watcher.tick()
            if not events:
                print("nothing to do: no waiting order was filled or expired" + ("" if in_regular_session(watcher.clock()) else " (the regular session is closed: orders are only filled while it is open)"))
            return 0
        if args.cancel is not None:
            closed = store.cancel_open_order(account_id, args.cancel)
            print(f"order #{closed.order_id} is cancelled: {order_words(closed)}; what it set aside is available again")
            return 0
        waiting = store.open_orders(account_id, ("open",))
        print(f"account {account_id}: {len(waiting)} waiting order{'s' if len(waiting) != 1 else ''}")
        for order in waiting:
            print("  " + order_line(order))
        if waiting:
            print("A waiting order is watched only while the page's server runs (./run_ui.sh)." + ("" if store.watcher_running() else " NO SERVER IS RUNNING a watcher now."))
    except (ValueError, OSError, RuntimeError, psycopg.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
