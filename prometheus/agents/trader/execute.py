"""The ten steps of `Trader.execute_trades` (Trader v1 spec section 4), as `critic/review.py` holds the Critic's nine. Since design choices entry 81 every trade goes through the per-trade HANDSHAKE with the Account
Manager (`broker.py`), and steps 2 and 8 changed (see the amendment at the end of `design/trader_v1_final_spec.md`):

     1 preconditions      PAPER_TRADING is true; the mode names a venue (`venue_for`: paper or backtest); no `book` for a venue that cannot book (`venue.books`: a backtest); the order was decided; it has decisions
     2 account            `account.trading_portfolio` opens the run (the Account Manager takes the funds into its own hands); the portfolio's user is the profile's user; the Account Manager applies the fills of an earlier
                          run that were never booked to its working portfolio (`apply_earlier_fills`); the portfolio the Account Manager then reports is read only to REPORT it (`portfolio_before`) and to build the tickets
     3 tickets            one per decision that says yes_auto or yes_user (tickets.py; a decision whose symbol is not a ticker gets none and is named in the errors); trades already on the order are loaded: filled and rejected are final, active is judged again
     4 prices             ONE snapshot through the Account Manager, made by the VENUE (`venue.py`, entry 128), asked in groups of at most 30 symbols: quotes (paper) or stored bars (backtest); an active trade of an earlier run is judged again, by code
     5 funding consent    the user's yes for the sales that pay for a purchase the cash does not cover (the Account Manager says whether the cash covers it: the Trader never works the funds out)
     6 the model's loop   (trader.py), when the model may and could run and a ticket is `new`
     7 the sweep          code places what is left, every sale before any purchase; what nobody placed is `not_attempted`. EVERY trade of steps 4, 6 and 7 goes through the handshake in `broker.py`: approval, the fill
                          for at most the shares approved, the output check of that one fill (`work_order.verify_fill`), and the report to the Account Manager (`record_fill`, which books it) BEFORE the next trade
     8 the close          the run is closed (nothing more is placed); the whole-run output guard `work_order.verify_trades` REPORTS (what was booked stays booked: it is the book's); the account is read back from the
                          Account Manager for `portfolio_after` and the cash figures. It tells the Account Manager nothing: each fill was told as it happened
     9 outcomes           each filled purchase is followed through the later bars by the venue (`venue.follow`: the backtest venue, by simulator.follow; a venue with no later bars gives none)
    10 records            `order.trades`, one more entry on `order.trader_runs`, the trades ledger (a venue that keeps one: `venue.keeps_ledger`, paper), the status

DECIMAL MONEY (design choices entry 112): the account's figures here (the cash before and after, a snapshot, a ticket's prices, a fill) are `Decimal`. The prices the Account Manager asks the Data Clerk for come back as the floats of the quote
or the bar and cross the price door ONCE: a quote in `_prices` (a `Decimal` price in the snapshot), a stored bar's prices inside the fill rules (`simulator.bar_price`). The trades and the runs are written to the order with `model_dump()` (Python mode: the
Decimals stay Decimals) and to the trades ledger with `model_dump(mode="json")` (numbers, by the one encoder).

A failed precondition is ONE message on `order.errors` and a `trader_runs` entry, and nothing else changes. After that nothing here raises for a data, model or account failure: a trade that cannot be executed is `rejected`
with a code and a sentence, anything else is a message on the order and on the run. A `now` without a timezone raises ValueError before anything is done, as everywhere else.

The Trader reaches the outside world through the Account Manager only (rule T5): the portfolio and the funds, the approval and the booking of each trade, the quotes and the bars. It holds NO cash and NO portfolio of
its own (a test reads the source). It never edits what the Advisor, the Critic and the Account Manager wrote (rule T10: the recommendations, the reviews and the decisions are the same bytes afterwards). It changes
`trades`, `trader_runs`, `status` (to `traded`), `schema_version` (to 4) and `errors`.
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

from prometheus.agents.advisor.calls import parse_object
from prometheus.agents.trader import simulator, tickets
from prometheus.agents.trader.broker import CASH_HINT, PaperBroker, Refused
from prometheus.agents.trader.config import paper_trading_enabled
from prometheus.agents.trader.record import FundingRun, Run, Snapshot, Trade
from prometheus.agents.trader.venue import Prices, Venue, groups, venue_for      # `groups` was here until entry 128 and is still importable from here
from prometheus.common import shares as sh
from prometheus.common.market_calendar import eastern_date
from prometheus.common.money import digits, price_scale
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.trace import traced
from prometheus.common.trading_clock import trading_now
from prometheus.common.work_order import TRADING, WorkOrder, verify_trades

STARTABLE = ("reviewed", "decided", "traded")      # the statuses of an order the Trader may work on


@dataclass
class Context:
    """Everything one run shares between its steps: the order, the account, the settings, the broker and the way back to the Account Manager."""
    trader: object
    order: WorkOrder
    account: object
    config: object
    mode: str
    now: datetime
    run: Run
    book: bool
    venue: Venue | None = None                        # where the orders go (`venue.py`, entry 128): the price, the fill and the exits of this run come from it
    start: Portfolio | None = None                    # the account at the start of this run, as the Account Manager reported it (the fills of an earlier run that were never booked are in it): read to REPORT and to build tickets, never changed
    broker: PaperBroker | None = None
    prices: Prices | None = None
    skipped: list = field(default_factory=list)       # active trades of a run in the other mode: left exactly as they were
    final_before: set = field(default_factory=set)    # trade ids that were final (filled or rejected) before this run
    notifier: "Notifier | None" = None
    after: Portfolio | None = None                    # the account as the Account Manager reported it when the run was closed (None when it could not be read)


# ---------------------------------------------------------------- small helpers

def snapshot(portfolio: Portfolio) -> Snapshot:
    return Snapshot(cash=portfolio.cash_avail, positions=[{"symbol": p.symbol, "shares": p.shares, "cost_basis": p.cost_basis, "acquired": p.acquired.isoformat() if p.acquired else None} for p in portfolio.positions])


def arranged(trades: list[Trade]) -> list[Trade]:
    """The trades in the order they are written on the order: each recommendation trade followed by the funding sales that pay for it. A sale whose purchase is not there goes last (the output guard names it)."""
    legs: dict[str | None, list[Trade]] = {}
    for trade in trades:
        if trade.kind == "funding":
            legs.setdefault(trade.funds_for, []).append(trade)
    out = []
    for trade in trades:
        if trade.kind != "funding":
            out.append(trade)
            out.extend(legs.pop(trade.trade_id, []))
    for rest in legs.values():
        out.extend(rest)
    return out


def all_trades(ctx: Context) -> list[Trade]:
    return arranged([*ctx.broker.trades.values(), *ctx.skipped]) if ctx.broker else arranged(list(ctx.skipped))


def write_trades(ctx: Context) -> None:
    ctx.order.trades = [t.model_dump() for t in all_trades(ctx)]


def message(error: Exception) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())[:160]}"


def money(x: Decimal) -> str:
    return f"${x:,.2f}"


def counts(trades: list[Trade]) -> dict[str, int]:
    own = [t for t in trades if t.kind == "recommendation"]
    return {"filled": sum(t.status == "filled" for t in own), "rejected": sum(t.status == "rejected" for t in own), "active": sum(t.status == "active" for t in own)}


@traced("trader.fill_order")
def fill_order(trades: list[Trade]) -> list[Trade]:
    """Trades in an order the account can take them in: every sale before any purchase (a funding sale before the purchase it pays for), otherwise as they are on the order. Used for the fills of an earlier run and
    for the active trades judged again."""
    return sorted(trades, key=lambda t: t.side != "sell")


# ---------------------------------------------------------------- the close of the run (the model's notify_account_manager and step 8)

class Notifier:
    """Step 8, as a callable that does its work ONCE: it CLOSES the run. It keeps its name because the model's tool is `notify_account_manager`, but since design choices entry 81 it tells the Account Manager NOTHING:
    every fill was reported at once, as it happened (`PaperBroker._fill`). The first call closes the broker (nothing more is placed), moves the broker's notes (the book's refusals, an output check that found something)
    onto the order and the run, REPORTS the whole-run output guard `work_order.verify_trades` (a finding is a message: what was booked stays booked, it is the account book's), reads the account back from the Account
    Manager for `portfolio_after` and the cash figures, and counts what was booked. A second call returns the first answer."""

    def __init__(self, ctx: Context):
        self.ctx, self.done, self.answer = ctx, False, None

    def __call__(self) -> dict:
        if self.done:
            return self.answer
        self.done, self.ctx.broker.closed = True, True
        try:
            self.answer = self._close()
        except AssertionError:   # a programming mistake or a test's guard: not hidden (see `execute`)
            raise
        except Exception as error:   # whatever went wrong, the run goes on and the answer exists
            self._say(f"the Trader could not close the run properly ({message(error)})")
            self.ctx.after = self.ctx.after or self._read_account()
            self.answer = self._answer()
        return self.answer

    def _close(self) -> dict:
        ctx = self.ctx
        ctx.run.booked = any(t.booked for t in ctx.broker.filled_trades())
        write_trades(ctx)
        for text in ctx.broker.notes:
            self._say(text)
        for finding in verify_trades(ctx.order):
            self._say(f"the Trader's output check of the whole run found: {finding}")
        ctx.after = self._read_account()
        return self._answer()

    def _read_account(self) -> Portfolio | None:
        """The account as the Account Manager reports it now; None, and a message, when it cannot be read (nothing is then claimed about it)."""
        try:
            return self.ctx.account.current_portfolio()
        except AssertionError:
            raise
        except Exception as error:
            self._say(f"the Trader could not read the account after the run ({type(error).__name__}): the figures after the run are not known")
            return None

    def _answer(self) -> dict:
        ctx = self.ctx
        return {"cash_before": ctx.start.cash_avail, "cash_after": None if ctx.after is None else ctx.after.cash_avail, **counts(all_trades(ctx)), "booked": ctx.run.booked}

    def _say(self, text: str) -> None:
        if text in self.ctx.run.errors:
            return
        self.ctx.order.errors = [*self.ctx.order.errors, text]
        self.ctx.run.errors.append(text)


# ---------------------------------------------------------------- the summary (rule T8)

NUMBER = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)")
TRADE_REF = re.compile(r"\bT\d+(?:-F\d+)?\b")


def numbers_of(text: str) -> list[str]:
    """The numbers written in a text, as written (a trade id like T1 is not a number; thousands separators and a trailing comma are dropped)."""
    return [m.group(1).rstrip(",").replace(",", "") for m in NUMBER.finditer(TRADE_REF.sub(" ", text))]


def allowed_numbers(ctx: Context, trades: list[Trade]) -> list[Decimal]:
    """Every number a summary may use: a number of the trades (share counts, prices, amounts, limits), the prices the model was shown, the cash before and after, and the counts of trades. All of them Decimals."""
    out: list = []
    for t in trades:
        out += [t.shares, t.shares_approved, t.dollars, t.limit_price, t.ref_price, t.limit_low, t.limit_high]      # `dollars`: an order in dollars (entry 111) may be named in a summary
        if t.limit_low is not None and t.limit_high is not None:
            out += tickets.shown_range(t.limit_low, t.limit_high)
        if t.fill:
            out += [t.fill.price, t.fill.shares, t.fill.amount, t.fill.commission]
    prices = ctx.prices
    if prices:
        out += list(prices.quotes.values()) + [simulator.bar_price(b, "open") for b in (simulator.bar_on(series, prices.entry_date) for series in prices.bars.values()) if b]
    if ctx.start is not None:
        out.append(ctx.start.cash_avail)
    if ctx.after is not None:
        out.append(ctx.after.cash_avail)
    if ctx.broker is not None:
        out += [told["cash_left"] for told in ctx.broker.answers.values() if told.get("cash_left") is not None]      # the cash the model was told after each fill (place_order's `cash_left`): the Account Manager's own figures
    own = counts(trades)
    out += [len(trades), len([t for t in trades if t.kind == "recommendation"]), len([t for t in trades if t.kind == "funding"]), *own.values(), len([t for t in trades if t.placed_by == "model"]), len([t for t in trades if t.placed_by == "code"])]
    return [digits(x) for x in out if isinstance(x, (int, Decimal)) and not isinstance(x, bool)]


def check_summary(text: str, ctx: Context, trades: list[Trade]) -> tuple[str | None, list[str]]:
    """The model's summary is kept only when: its last message is the JSON asked for; every trade it lists exists and its status is the real one; the text is at most `limits.max_summary_words`; and every number in
    the text is a number of the trades, compared at the number of decimals written. Returns (the summary, or None) and what was wrong (each is `dropped` on the run)."""
    if not (text or "").strip():
        return None, ["the model gave no last message, so its summary was not kept"]
    try:
        data = parse_object(text, "trades")
    except ValueError as error:
        return None, [f"the model's last message was not the JSON asked for ({error}), so its summary was not kept"]
    summary, dropped, by_id = data.get("summary"), [], {t.trade_id: t for t in trades}
    if not isinstance(summary, str) or not summary.strip():
        dropped.append("the JSON has no summary text")
        summary = ""
    for entry in data["trades"]:
        trade_id = entry.get("trade_id") if isinstance(entry, dict) else None
        if trade_id not in by_id:
            dropped.append(f"the model listed {str(trade_id)[:20]!r}, which is not a trade of this order")
        elif entry.get("status") != by_id[trade_id].status:
            dropped.append(f"the model said {trade_id} is {str(entry.get('status'))[:20]!r} but it is {by_id[trade_id].status}")
    limit = ctx.config.limits.max_summary_words
    if len(summary.split()) > limit:
        dropped.append(f"the summary has {len(summary.split())} words, over the limit of {limit}")
    allowed = allowed_numbers(ctx, trades)
    for written in dict.fromkeys(numbers_of(summary)):
        places = len(written.split(".")[1]) if "." in written else 0
        if not any(a.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP) == Decimal(written) for a in allowed):      # the number as written: the allowed number rounded to the places the text gives it
            dropped.append(f"the number {written} in the summary is not a number of the trades")
    return (None, dropped) if dropped or not summary else (" ".join(summary.split()), [])


def code_summary(ctx: Context, trades: list[Trade]) -> str:
    """Code's own sentence: what happened to each trade, and the cash."""
    parts = []
    for t in (x for x in trades if x.kind == "recommendation"):
        clause = f"{t.trade_id} {t.side} {sh.text(t.shares_approved if t.status == 'rejected' and not t.shares else t.shares)} {t.symbol} {t.status}"
        if t.status == "filled" and t.fill:
            clause += f" at {t.fill.price:,.4f} ({money(t.fill.amount)})"
        elif t.status == "rejected":
            clause += f" ({t.reject_code})"
        parts.append(clause)
    cash = f"Cash {money(ctx.start.cash_avail)} to {money(ctx.after.cash_avail)}." if ctx.after is not None else f"Cash {money(ctx.start.cash_avail)} at the start; the account could not be read afterwards."
    return f"{ctx.mode.capitalize()} run: " + ("; ".join(parts) if parts else "no trades to place") + f". {cash}"


# ---------------------------------------------------------------- the steps

def refuse(order: WorkOrder, run: Run, text: str) -> WorkOrder:
    """A failed precondition: one message on the order, one entry on `trader_runs`, nothing else changes."""
    run.errors.append(text)
    order.errors = [*order.errors, text]
    order.trader_runs = [*order.trader_runs, run.model_dump()]
    return order


def preconditions(order: WorkOrder, mode: str, book: bool, config) -> tuple[str | None, Venue | None]:
    """Step 1. (the first reason the Trader may not run, or None; the venue of the mode, or None when there is a reason). The venue is `venue_for(mode, config)`: a mode that is not paper or backtest, and a venue that places
    real orders, are refused there (entry 128); `book` is refused for a venue that cannot book (`venue.books`)."""
    if not paper_trading_enabled():
        return "PAPER_TRADING is not true: the Trader only trades on paper and refuses to run (set PAPER_TRADING=true)", None
    try:
        venue = venue_for(mode, config)
    except ValueError as error:
        return str(error), None
    if book and not venue.books:
        return f"a {venue.name} replay never writes the account book: book is refused in {venue.name} mode", None
    if order.status not in STARTABLE:
        return f"the Trader needs an order the Account Manager decided on (status {order.status})", None
    if not order.trade_decisions:
        return "the order has no trade decisions: the Account Manager has not decided on it", None
    return None, venue


@traced("trader.execute")
def execute(trader, order: WorkOrder, account, portfolio, now, funding, approver, book: bool) -> WorkOrder:
    """The ten steps. See the module text."""
    now = now or trading_now()   # the real clock, or the simulated one when USE_LIVE_TRADING=false (entry 76)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    try:
        config = trader.config
    except ValueError as error:   # a settings file that is not valid: the run is refused, with the file's own message
        run = Run(mode=str(trader.mode), now_utc=tickets.utc_text(now), use_model=trader.use_model)
        return refuse(order, run, f"the Trader's settings could not be read: {error}")
    mode = trader.mode or config.mode
    run = Run(model=None, temperature=0.0, mode=str(mode), now_utc=tickets.utc_text(now), use_model=trader.use_model, config_version=config.config_version)
    reason, venue = preconditions(order, mode, book, config)
    if reason:
        return refuse(order, run, reason)
    ctx = Context(trader=trader, order=order, account=account, config=config, mode=mode, now=now, run=run, book=book, venue=venue)
    try:
        return _steps(ctx, portfolio, funding, approver)
    except AssertionError:   # not a failure of data, model or account: an assertion is a programming mistake (or a test's guard, like the one that stops a test from writing the real ledger), and hiding it would hide the bug
        raise
    except Exception as error:   # a bug of ours must not break the desk (rule T7): say so on the order. Trades are booked one at a time, so the ones that filled are written as they stand
        text = f"the Trader stopped unexpectedly ({message(error)})"
        run.errors.append(text)
        order.errors = [*order.errors, text]
        if (ctx.notifier is not None and ctx.notifier.done) or (ctx.broker is not None and ctx.broker.filled):   # a fill is booked the moment it happens (entry 81): once any trade filled, the order must say so, whatever stopped the run
            write_trades(ctx)
        order.trader_runs = [*order.trader_runs, run.model_dump()]
        return order


def _steps(ctx: Context, portfolio, funding, approver) -> WorkOrder:
    order, run, config, mode, now = ctx.order, ctx.run, ctx.config, ctx.mode, ctx.now
    today = eastern_date(now)
    # ---- 2 account
    try:
        base = ctx.account.trading_portfolio(portfolio, book=ctx.book)
    except ValueError as error:
        return refuse(order, run, f"the Trader could not get the portfolio to trade: {error}")
    except Exception as error:   # a database error: the kind only
        return refuse(order, run, f"the Trader could not read the account ({type(error).__name__})")
    try:
        holder = ctx.account.profile.user
    except Exception as error:
        return refuse(order, run, f"the Trader could not read the user's profile ({type(error).__name__})")
    if holder != base.user:
        return refuse(order, run, f"the user profile is for {holder!r} but the portfolio belongs to {base.user!r}: no trade is made")
    try:
        existing = [Trade.model_validate(t) for t in order.trades]
    except Exception as error:
        return refuse(order, run, f"the trades already on the order could not be read ({type(error).__name__})")
    ctx.final_before = {t.trade_id for t in existing if t.status in ("filled", "rejected")}
    unbooked = [t.model_dump() for t in existing if t.status == "filled" and not t.booked]
    problems, ctx.start = [], base                       # the portfolio the Account Manager just gave IS the account at the start: nothing is read again unless an earlier run's fill has to be put into it
    if unbooked:
        try:
            problems = ctx.account.apply_earlier_fills(unbooked)      # the Account Manager puts the fills of an earlier run that were never booked into ITS portfolio
            ctx.start = ctx.account.current_portfolio()
        except Exception as error:
            return refuse(order, run, f"the Trader could not read the account ({type(error).__name__})")
    for text in problems:
        run.errors.append(text)
        order.errors = [*order.errors, text]
    run.portfolio_before = snapshot(ctx.start)
    # ---- 3 tickets
    ctx.skipped = [t for t in existing if t.status == "active" and t.mode != ctx.venue.name]
    for trade in ctx.skipped:
        run.errors.append(f"{trade.trade_id} is an active trade of a {trade.mode} run: it cannot be judged in a {mode} run and was left as it was")
        order.errors = [*order.errors, run.errors[-1]]
    loaded = [t for t in existing if t not in ctx.skipped]
    bad_symbols: list[str] = []
    fresh = tickets.build_tickets(order, ctx.start, now, config, mode, bad_symbols)
    for text in bad_symbols:        # a decision whose symbol is not a ticker got no ticket: say so (spec section 14)
        run.errors.append(text)
        order.errors = [*order.errors, text]
    trades = [*loaded, *fresh]
    target, legs, note = tickets.funding_legs(funding, [t for t in fresh if t.status == "new"], ctx.start, today)
    if funding is not None:
        run.funding = FundingRun(status=str(funding.get("status"))[:30] if isinstance(funding, dict) else None, approved=None, note=note)
    # ---- 4 prices
    judged = [t for t in trades if t.status in ("new", "active")]       # the tickets, the trades to be judged again, and the funding sales that wait for their purchase (an earlier run's)
    symbols = list(dict.fromkeys([t.symbol for t in judged] + [leg["symbol"] for leg in legs]))
    ctx.prices = _prices(ctx, symbols, now)
    permit = lambda t: tickets.permission(order, t.record_id, now)
    ctx.broker = PaperBroker(config, mode, ctx.account, order, ctx.prices, trades, permit, now, cash_hint=CASH_HINT if funding is None else "", venue=ctx.venue)
    ctx.notifier = Notifier(ctx)
    _mark_unavailable(ctx)
    for trade in fill_order([t for t in loaded if t.status == "active" and t.kind == "recommendation"]):
        ctx.broker.rejudge(trade.trade_id)
    # ---- 5 funding consent
    if target is not None and legs:
        _consent(ctx, funding, target, legs, approver, fresh)
    # ---- 6 the model's loop
    trader = ctx.trader
    waiting = ctx.broker.tickets()
    loop = {"ran": False, "text": ""}
    if trader.use_model and waiting:
        loop = trader.run_loop(ctx.broker, ctx.notifier, order.request_id, today, len(waiting), [t.trade_id for t in ctx.broker.trades.values() if t.kind == "recommendation"])
        if loop["ran"]:
            run.model, run.turns, run.tokens_in, run.tokens_out, run.tool_calls = trader.model_name, loop["turns"], loop["tokens_in"], loop["tokens_out"], loop["tool_calls"]
            if loop["error"]:
                run.errors.append(f"the model call failed ({loop['error']}): the program places the rest" + (" (a rate limit)" if loop["rate_limited"] else ""))
            run.errors.extend(loop["stopped"])
        else:
            run.errors.extend(e for e in trader.errors if e not in run.errors)
    # ---- 7 the sweep
    if trader.sweep:
        for side in ("sell", "buy"):
            for trade in ctx.broker.tickets(side):
                try:
                    ctx.broker.place(trade.trade_id, placed_by="code")
                except Refused as error:   # not expected: a ticket that is `new` and open can be placed. The trade is left to close_out
                    run.errors.append(f"{trade.trade_id} was not placed by the program: {error}")
    ctx.broker.close_out()
    # ---- 8 the output guard, then the Account Manager
    ctx.notifier()
    # ---- 9 outcomes
    _outcomes(ctx)
    # ---- 10 records
    return _records(ctx, loop)


def _prices(ctx: Context, symbols: list[str], now: datetime) -> Prices:
    """Step 4: the ONE snapshot for every symbol the trades need, through the Account Manager, made by the venue (`venue.prices`: quotes for paper, stored bars for backtest). The request is made in groups of at most 30
    symbols (the limit of a quote order and of a bars order; spec section 15) and the answers are joined, so a long order is not refused whole. When every group fails the snapshot has `failed` (every ticket is
    `price_unavailable`, with the one reason); when only some do, each symbol of a failed group is `unavailable` with its reason and a message goes on the run, and the others trade. Nothing is asked when there is nothing
    to judge (so an order with no live trade costs no quote)."""
    prices = ctx.venue.prices(ctx.account, symbols, now, ctx.run.errors)
    if prices.order_id:
        ctx.run.price_order = prices.order_id
    return prices


def _mark_unavailable(ctx: Context) -> None:
    """A ticket whose symbol step 4 gave no price for is rejected `price_unavailable` at once, and the model never sees it as placeable (the venue says whether the snapshot holds a price: `venue.missing`). A backtest with no bar from the entry day on is not that: the orders wait."""
    prices, broker = ctx.prices, ctx.broker
    for trade in broker.tickets():
        if prices.failed:
            broker.reject(trade, "price_unavailable", prices.failed)
        elif trade.symbol in prices.unavailable:
            broker.reject(trade, "price_unavailable", prices.unavailable[trade.symbol])
        elif (why := ctx.venue.missing(prices, trade.symbol)) is not None:
            broker.reject(trade, "price_unavailable", why)


def _consent(ctx: Context, funding, target: Trade, legs: list[dict], approver, fresh: list[Trade]) -> None:
    """Step 5: a purchase the cash does not cover, with a valid funding plan, needs the user's yes for the sales (spec section 7). The sales are attached as trades of their own only on a yes. WHETHER THE CASH COVERS
    THE PURCHASE is the Account Manager's answer (`broker.preview`: would it approve all the shares now?), not a sum made here: the Trader holds no cash (design choices entry 81)."""
    broker, run, config = ctx.broker, ctx.run, ctx.config
    if target.status != "new":
        run.funding.note = f"{target.trade_id} was rejected ({target.reject_code}) before any sale: the funding sells are not used"
        return
    price = ctx.venue.price_of(ctx.prices, target.symbol) or target.ref_price
    paid = price_scale(price * (1 + config.costs.slippage_bps / 10000))
    cost = trade_amount(target.shares, paid) + config.costs.commission_per_order
    question = broker.preview(target, paid)
    if question.approved and question.shares >= target.shares:
        run.funding.note = "the cash covers the purchase: the funding sells are not used"
        return
    if not question.approved and question.code != "insufficient_cash":
        run.funding.note = f"the Account Manager would not approve the purchase ({question.code}): the funding sells are not used"
        return
    if len(fresh) + len(legs) > config.limits.max_trades:
        run.funding.note = f"the funding sells would take the run over limits.max_trades ({config.limits.max_trades}): they are not used"
        return
    if config.funding.require_user_approval:
        card = funding_card(ctx, target, legs, cost, question.cash_avail)
        answer = None
        if approver is not None:
            try:
                answer = approver(card)
            except Exception as error:   # a prompt that failed: the user has not said yes
                text = f"the user could not be asked about the funding sells ({type(error).__name__})"
                run.errors.append(text)
                ctx.order.errors = [*ctx.order.errors, text]
        run.funding.approved = answer is True
        run.funding.note = "the user approved the funding sells" if answer is True else "the funding sells were not approved: " + ("the user said no" if answer is False else "nobody answered")
    else:
        run.funding.approved, run.funding.note = True, "the funding sells are used without asking (funding.require_user_approval is false)"
    if run.funding.approved:
        broker.add(tickets.funding_trades(target, legs, config, ctx.mode))


def funding_card(ctx: Context, target: Trade, legs: list[dict], cost: Decimal, cash: Decimal | None) -> dict:
    """What the user is asked (spec section 7): the sells, one per line (symbol, shares, about the proceeds, the gain), and the question. `cash` is what the Account Manager said the account has (None: not known)."""
    held, lines, proceeds = ctx.start, [], Decimal("0.00")
    for leg in legs:
        price = ctx.venue.price_of(ctx.prices, leg["symbol"]) or leg["price"]
        position = held.position(leg["symbol"])
        amount = trade_amount(leg["shares"], price)                                                          # decimal, half up to the cent, the way the book works a sale (entries 111 and 112)
        gain = amount - trade_amount(leg["shares"], position.cost_basis if position else price)
        proceeds += amount
        lines.append(f"  sell {sh.text(leg['shares'])} {leg['symbol']}: about {money(amount)}, a gain of {money(gain) if gain >= 0 else '-' + money(-gain)}")
    text = "\n".join([f"To buy {sh.text(target.shares)} shares of {target.symbol} (about {money(cost)}) with cash of {money(ctx.start.cash_avail if cash is None else cash)}, these holdings would be sold:", *lines, f"  together about {money(proceeds)}"])
    return {"text": text, "symbol": target.symbol, "shares": target.shares, "est_cost": cost,
            "question": f"Sell these to pay for {sh.text(target.shares)} shares of {target.symbol} (about ${cost:,.0f})? [y/N] ", "kind": "funding"}


def _outcomes(ctx: Context) -> None:
    """Step 9: each filled purchase of a recommendation is followed through the later bars by the Advisor's own plan, as the venue does it (`venue.follow`: the backtest venue; a venue with no later bars gives None and the
    trade has no outcome). A simulation: it is never booked."""
    order, run = ctx.order, ctx.run
    for trade in ctx.broker.trades.values():
        if trade.kind != "recommendation" or trade.side != "buy" or trade.status != "filled" or trade.fill is None or trade.outcome is not None:
            continue
        try:
            plan = next((r.get("trade") for r in order.recommendations if isinstance(r, dict) and (r.get("identity") or {}).get("record_id") == trade.record_id), None) or {}
            trade.outcome = ctx.venue.follow(trade, plan, ctx.prices)
        except Exception as error:   # an outcome is a nicety: the fills stand
            run.errors.append(f"{trade.trade_id}: the outcome could not be followed ({message(error)})")


def _records(ctx: Context, loop: dict) -> WorkOrder:
    """Step 10: the trades, the run, the ledger, the status."""
    order, run, config, trader = ctx.order, ctx.run, ctx.config, ctx.trader
    run.placed_by_model = sum(1 for _, by in ctx.broker.placed if by == "model")
    run.placed_by_code = sum(1 for _, by in ctx.broker.placed if by == "code")
    run.portfolio_after = snapshot(ctx.after) if ctx.after is not None else None       # the account as the Account Manager reported it at the close: None, and a message, when it could not be read
    run.price_order = run.price_order or (ctx.prices.order_id if ctx.prices else None)
    write_trades(ctx)
    trades = all_trades(ctx)
    summary, dropped = (check_summary(loop["text"], ctx, trades) if loop.get("ran") and not loop.get("error") else (None, ["the model did not finish, so there is no summary of its own"] if loop.get("ran") else []))
    run.summary, run.summary_by, run.dropped = (summary, "model", dropped) if summary else (code_summary(ctx, trades), "code", dropped)
    if ctx.venue.keeps_ledger:
        _ledger(ctx, trades)
    permitted = [d for d in order.trade_decisions if d.get("will_trade") in TRADING]
    by_record = {t.record_id: t for t in trades if t.kind == "recommendation"}
    done = bool(permitted) and all(d.get("record_id") in by_record and by_record[d["record_id"]].status in ("filled", "rejected") for d in permitted)
    findings = not ctx.notifier.done or bool(verify_trades(order))       # a trade that failed (the book refused it, its booking could not be confirmed) is `rejected` with its reason: the run still went through the whole list and is finished
    if order.trades:
        order.schema_version = 4
    if order.status == "decided" and done and not findings:
        order.status = "traded"
    order.trader_runs = [*order.trader_runs, run.model_dump()]
    return order


def _ledger(ctx: Context, trades: list[Trade]) -> None:
    """The trades ledger, paper mode only: one line for each trade that BECAME filled or rejected in this run (a trade final before the run is not written again, a resting one is not final)."""
    run = ctx.run
    wrote = [t for t in trades if t.status in ("filled", "rejected") and t.trade_id not in ctx.final_before]
    if not wrote:
        return
    try:
        ledger = ctx.trader.ledger
        for trade in wrote:
            ledger.append({"request_id": ctx.order.request_id, "recorded_utc": run.now_utc, **trade.model_dump(mode="json")})
    except (OSError, ValueError, TypeError) as error:   # a ledger that cannot be written (a full disk, a record that is not JSON) is a message: the trades stand
        run.errors.append(f"the trades ledger could not be written ({message(error)})")
