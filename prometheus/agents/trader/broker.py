"""The paper broker: the placing of each trade against ONE snapshot of prices, through the per-trade handshake with the Account Manager (Trader v1 spec sections 6 and 7; design choices entry 81).

    prices = Prices("paper", trade_date, quotes={"PG": 167.10})                      # what step 4 fetched, once: a run is judged against it (`check_trade_status` never fetches again)
    broker = PaperBroker(config, "paper", account, order, prices, trades, permit, now)    # `account` is the Account Manager: the Trader's only door, and the only owner of the funds
    trade = broker.place("T1", "limit", 167.1, note="inside the range")              # a Trade: filled, active (a resting order) or rejected with a code

`simulator.py` has the fill RULES (pure); this applies them to one trade. It is the one place every trade goes through, whoever places it (the model's `place_order` tool, or the code's sweep), so the guards cannot be
skipped by a path that forgets them:

  T1  `permit` (the closure over `work_order.verify_trade`) runs again for every trade as it is placed;
  T2  an order is a type and a price inside the range code computed (`Refused` otherwise): nothing here takes a symbol, a side or a share count from a caller;
  T3  a trade is placed once: only a `new` trade can be placed, and an `active` one is only ever judged again by `rejudge`, by code;
  and  every reduce sale is placed before any purchase; nothing is placed after the run was closed (`closed`, set by the end-of-run `Notifier`). A trade that FAILS (the book refused it, its booking could not be confirmed, its output check found something) is marked failed and never
  retried, and the next trade is placed as usual: nothing stops the run (design choices entry 82).

THE BROKER HOLDS NO CASH AND NO PORTFOLIO (design choices entry 81). Jacob's decision: the Trader places one trade at a time, tells the Account Manager when the order goes through, and only then gets the approval for the next
one, so it cannot overdraw the funds; the Account Manager and its Postgres book are the ONLY owner of the funds and the positions. So the one funnel (`_fill`) is, for every trade that the price rules fill:

    1  APPROVAL     `account.approve_trade(trade, price, ...)`: the Account Manager reads the funds and holdings as they are NOW and says for how many shares (a purchase: the cash pays for them; a sale: they are held and
                    sellable today), or says no with a code (`insufficient_cash`, `not_held`, `not_sellable_today`, `not_permitted`, `approval_expired`, `account_refused`): the trade is `rejected` with that code
    2  THE FILL     the trade fills for at most the shares approved, at the price the rules gave (the approval needed that price, so the fill rules run first and the fill takes effect only after the approval)
    3  OUTPUT CHECK `work_order.verify_fill`: the single-trade part of the output guard. A finding means the fill is NOT reported or booked: the trade is `rejected` (`internal_error`) and the finding goes on the run
    4  STATUS BACK  `account.record_fill(trade)`, AT ONCE: the Account Manager reads the account (SELECT), books the fill in ONE Postgres transaction under row-level locks on the portfolio, and reads the account AGAIN
                    after the commit to confirm the trade is in and the totals are different (`Booking.verified`). A fill the book refuses is NOT a fill: the trade is `rejected` `account_refused` with the book's own
                    sentence, its `Fill` is not kept, and the Account Manager has shown the user the error. A booking that could not be confirmed is the same. A failed trade is NEVER retried
    5  NEXT TRADE   only after that answer, whatever it was: `place` returns when the Account Manager has answered, so no trade can start before the previous one was booked or failed, and the next trade of the list is
                    placed whether the last one was booked or not (its approval reads the account as it is then, so a failure cannot overdraw it)

NO WAITING (Jacob, 2026-11-14: "The handshaking should not wait if there are no further trades to execute"). The funnel asks for an approval only for a trade that is ABOUT TO BE PLACED: the price rules fill it now. A trade
that is rejected before it fills (no price, outside the limits, a limit not reached, not permitted) or that rests (a limit the quote has not reached, a backtest with no bar yet) is never put to the Account Manager and
never reported: it moves nothing and waits for nothing. A refused approval has no `record_fill`. After the LAST trade's `record_fill` has been answered `place` returns and the run goes straight on to its end: nothing here
asks for a "next" trade, polls, sleeps, times out or blocks in any way (every call is a plain call that returns; a test patches `time.sleep` to fail and reads this module's source for a wait). With no ticket to place there is
no handshake at all.

ORDERS IN DOLLARS AND NO SLIPPAGE (design choices entry 111). A trade whose `dollars` is set (copied from the decision) fills the shares the Account Manager APPROVED, not the smaller of the estimate and the approval: the Account Manager turned the dollars into shares at the price
this trade fills at (fewer shares than the estimate when the price is higher, more when it is lower), and what is filled never costs more than the dollars. `costs.slippage_bps` is 0 in the shipped settings (Jacob: "Slippage is not allowed. This is real money"): a paper fill
is at the quote, a backtest fill at the open; the simulator still takes the argument.

CASH is therefore never worked out here: a purchase fills for the shares the Account Manager approved, at most the shares the decision gave (`orders.cash_policy: "reduce"`): fewer than approved is a fill whose `reason` says
so, none is `insufficient_cash`. The one figure the broker reads is `cash_now()`, which asks the Account Manager (the model's `verify_request` shows it). What the Account Manager said about each trade is kept in `answers`
(the word of the booking and the cash left, for the model's `place_order` answer): a report, never read for any arithmetic. The commission of an order (`costs.commission_per_order`, 0 in this version) is passed to the
Account Manager as a setting.

DECIMAL MONEY (design choices entry 112): the snapshot's quotes, a limit price, the proceeds of a sale, a fill's price, shares and amount are `Decimal`. The quotes were turned into Decimals by the price door when the Trader asked for them
(`execute._prices`), a stored bar's prices cross it inside the fill rules (`simulator._bar_price`), and a limit price the model chose arrives through the tool as a Decimal. The trade dicts handed to the Account Manager and to `work_order.verify_fill` are
`model_dump()` (Python mode: the Decimals stay Decimals); the one encoder writes them as numbers when they are saved.

FUNDING (spec section 7). The sells that pay for a purchase are trades of their own (`funds_for`), placed WITH their purchase and never by the model, each through the same five steps. Placing a funded purchase does this in
one go: check the purchase first (permission, and that it would fill now: a resting limit or a rejected price sells nothing); work out what the sells bring (the same fill rules on the same snapshot, so the estimate is
what happens) and ask the Account Manager whether that would pay for even one share (`preview_trade(..., expected_proceeds=...)`: a question, nothing is reserved); if it would not the purchase is `insufficient_cash` and
every sale `funding_failed`, and NOTHING is sold; else fill each sale (one that cannot fill is `funding_failed` and the others go on) and only then ask for the purchase's approval, which reads the cash the BOOK now shows.
Holdings sold and nothing bought is the worst case, and this order is what stops it (the output guard `verify_trades` backs it up).
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable

from prometheus.agents.trader.config import TraderConfig
from prometheus.agents.trader.record import Fill, Trade
from decimal import Decimal

from prometheus.agents.trader.simulator import Execution, bar_on, fill_backtest, fill_paper, usable
from prometheus.agents.trader.tickets import shown_range, utc_text
from prometheus.common import shares as sh
from prometheus.common.money import exact, price_scale
from prometheus.common.portfolio import trade_amount
from prometheus.common.work_order import WorkOrder, verify_fill

CASH_HINT = "run the desk with --raise-funds to plan the sales that would pay for it"


class Refused(ValueError):
    """A placing the broker will not do, with a sentence that says how to fix it (an unknown trade id, a trade already placed, an order outside the range, a purchase while a sale is unplaced). It is the model's to
    correct and try again; it never ends a run."""


@dataclass
class Prices:
    """ONE snapshot of prices: what step 4 of the Trader asked the Account Manager for, once. `trade_date` is the date of every fill of the run (paper: the Eastern date of the run; backtest: the entry date).
    Paper: `quotes` {symbol: price}, a symbol with no usable quote is simply not there. Backtest: `bars` {symbol: [bar, ...]} and `entry_date`, None when no bar is stored from the calendar's entry day on (the future is
    not stored yet: orders wait)."""
    mode: str
    trade_date: date
    quotes: dict[str, Decimal] = field(default_factory=dict)
    bars: dict[str, list[dict]] = field(default_factory=dict)
    entry_date: date | None = None
    order_id: str | None = None                 # the request id of the quote or bars order the Account Manager made
    failed: str = ""                            # why no price could be had at all (the request failed or was refused): every order is then `price_unavailable`, which is not the same as "the future is not stored yet"
    unavailable: dict[str, str] = field(default_factory=dict)   # the request is made in groups of at most 30 symbols; when one group failed and the others answered, each symbol of that group is here with why: `price_unavailable`


def cut_note(note, max_words: int) -> str:
    """The model's `note`, cut by code: white space collapsed, control characters dropped, at most `max_words` words (rule T8: this is one of the two places a model's words reach a record)."""
    if not isinstance(note, str):
        return ""
    clean = "".join(c if c.isprintable() or c.isspace() else " " for c in note)
    return " ".join(clean.split()[:max_words])


@dataclass(frozen=True)
class _Refusal:
    """What `_approval` stands in with when the Account Manager raised instead of answering (it is meant never to): not approved, `account_refused`, a sentence that names the kind of the error."""
    reason: str
    approved: bool = False
    shares: Decimal = Decimal(0)
    code: str = "account_refused"
    cash_avail: Decimal | None = None


@dataclass(frozen=True)
class _Unconfirmed:
    """What `_fill` stands in with when the Account Manager raised instead of answering a `record_fill`: not a fill, nothing known; the trade is failed and the run goes on to the next trade."""
    reason: str
    status: str = "unconfirmed"
    cash_avail: Decimal | None = None
    note: str = ""
    ok: bool = False
    booked: bool = False


class PaperBroker:
    def __init__(self, config: TraderConfig, mode: str, account, order: WorkOrder, prices: Prices, trades: list[Trade], permit: Callable[[Trade], tuple[str | None, str]], now: datetime, cash_hint: str = ""):
        """config: the Trader's settings. account: the Account Manager, with its trading run open (`trading_portfolio`): the only owner of the funds, asked for each trade's approval and told each fill (see the module
        text). order: the work order (the Account Manager checks the decision again with `work_order.verify_trade`, and `verify_fill` reads the decisions). prices: the one snapshot. trades: EVERY trade of the run (the
        tickets, the funding sells, the active ones of an earlier run), in the order they will be shown; the broker changes these objects as it places them. permit: `permit(trade) -> (code, reason)`, (None, "")
        when `work_order.verify_trade` says the trade is allowed now (rule T1): it runs again for every trade placed. cash_hint: a sentence added to an `insufficient_cash` reason when no funding plan is in use."""
        self.config, self.mode, self.prices, self.permit, self.now, self.cash_hint = config, mode, prices, permit, now, cash_hint
        self.account, self.order = account, order
        self.trades: dict[str, Trade] = {t.trade_id: t for t in trades}
        if len(self.trades) != len(trades):
            raise ValueError("two trades have the same trade_id")
        self.filled: list[str] = []            # trade ids in the order they filled in this run (each was reported to the Account Manager the moment it filled)
        self.placed: list[tuple[str, str]] = []  # (trade_id, placed_by) of every trade placed in this run, for the counts of the run
        self.closed = False                    # set when the run is closed (the end-of-run `Notifier`): nothing more is placed
        self.answers: dict[str, dict] = {}     # what the Account Manager said about each trade it was told of: {"account": the booking's word, "cash_left": its figure}. For the model's answer and the summary: a report
        self.notes: list[str] = []             # sentences for the run and the order that this broker cannot write itself (the book's refusals, the output check): the Notifier moves them over

    def add(self, trades: list[Trade]) -> None:
        """Add trades to the run (the funding sales, once the user said yes to them). A trade_id that is already there is a programming mistake."""
        for trade in trades:
            if trade.trade_id in self.trades:
                raise ValueError(f"the trade_id {trade.trade_id} is already on the run")
            self.trades[trade.trade_id] = trade

    # ---------------------------------------------------------------- looking

    def get(self, trade_id) -> Trade:
        trade = self.trades.get(trade_id) if isinstance(trade_id, str) else None
        if trade is None:
            raise Refused(f"there is no trade {str(trade_id)[:30]!r} on this work order: use a trade_id from verify_request")
        return trade

    def tickets(self, side: str | None = None) -> list[Trade]:
        """The recommendation trades still `new` (a funding sale is placed with its purchase, so it is never listed here), optionally one side only, in the order of the tickets."""
        return [t for t in self.trades.values() if t.kind == "recommendation" and t.status == "new" and (side is None or t.side == side)]

    def legs_of(self, buy: Trade) -> list[Trade]:
        """The funding sales that pay for a purchase and have not been judged: `new` (this run) or `active` (waiting for their purchase since an earlier run)."""
        return [t for t in self.trades.values() if t.kind == "funding" and t.funds_for == buy.trade_id and t.status in ("new", "active")]

    def filled_trades(self) -> list[Trade]:
        return [self.trades[i] for i in self.filled]

    # ---------------------------------------------------------------- placing

    def place(self, trade_id, order_type: str = "", limit_price: Decimal = Decimal(0), note: str = "", placed_by: str = "model") -> Trade:
        """Place one ticket (rules T1 to T3, see the module text). `order_type` "" is the default order the ticket carries; "limit" with a `limit_price` of 0 is the default limit price; "market" takes no price. Raises
        `Refused`, with the sentence to fix it, for: an unknown or funding trade id, a trade that is not `new`, a closed run, an order type that is not market or limit, `market` with market orders off, a limit price outside the
        range, and a purchase while a sale is still `new`. Anything the market does to the order is NOT a refusal: it is the trade's status. Returns the trade."""
        trade = self.get(trade_id)
        if self.closed:
            raise Refused("the run was closed (notify_account_manager was called): nothing more can be placed")
        if trade.kind == "funding":
            raise Refused(f"{trade.trade_id} is a funding sale: it is placed together with {trade.funds_for}, never by you")
        if trade.status != "new":
            raise Refused(f"{trade.trade_id} was already placed (it is {trade.status}" + (f", {trade.reject_code}" if trade.reject_code else "") + "): a trade is placed once")
        kind, price = self._order(trade, order_type, limit_price)
        waiting = self.tickets("sell")
        if trade.side == "buy" and waiting:
            raise Refused(f"place the sale {waiting[0].trade_id} first: every sale is placed before any purchase")
        self._place(trade, kind, price, cut_note(note, self.config.limits.max_note_words), placed_by)
        return trade

    def _order(self, trade: Trade, order_type, limit_price) -> tuple[str, Decimal | None]:
        """The order to place: (type, limit price), checked. A caller that gives nothing gets the default order the ticket carries. The limit price is a Decimal (the tool turns the model's number into one at its door; a float here is a TypeError)."""
        try:
            price = exact(limit_price, "limit_price")
        except ValueError:
            raise Refused("limit_price must be a number (0 for the default)") from None
        if order_type in ("", None):
            if price:
                raise Refused('give an order_type of "limit" with a limit_price, or leave both empty to place the default order')
            return trade.order_type, trade.limit_price
        if order_type not in ("market", "limit"):
            raise Refused(f"order_type must be 'market' or 'limit' (or empty for the default order), not {str(order_type)[:20]!r}")
        if order_type == "market":
            if not self.config.orders.allow_market:
                raise Refused("market orders are turned off: use order_type 'limit'")
            if price:
                raise Refused("a market order has no limit_price: leave it 0, or use order_type 'limit'")
            return "market", None
        low, high = trade.limit_low, trade.limit_high
        if low is None or high is None or trade.ref_price is None:
            raise Refused(f"{trade.trade_id} has no price range to place a limit order inside")
        if not price:
            price = trade.limit_price if trade.order_type == "limit" and trade.limit_price else min(max(trade.ref_price, low), high)
        price = price_scale(price) if price > 0 else price
        if not usable(price) or not low <= price <= high:
            shown = shown_range(low, high)
            raise Refused(f"limit_price {price:,.4f} is outside the range allowed for {trade.trade_id}, {shown[0]:,.2f} to {shown[1]:,.2f}")
        return "limit", price

    def rejudge(self, trade_id: str) -> Trade:
        """Judge an `active` trade of an earlier run again, by code, against this run's prices: the permission first (T1: one that ran out is `rejected`), then the fill rules with the order it already has. The model cannot
        place it again (`place` refuses a trade that is not `new`). A trade that is not `active` is left alone. A resting order whose price could not be had in this run (the quote or the bars failed) stays `active` and the
        reason says so: an outage does not make a resting order final."""
        trade = self.get(trade_id)
        if trade.status == "active" and trade.kind == "recommendation" and not self.closed:
            self._run(trade, again=True)
        return trade

    def close_out(self) -> list[str]:
        """What the model (or nobody) left unplaced is rejected `not_attempted`: every recommendation trade still `new`, and the funding sales whose purchase did not happen (`not_attempted` when the purchase was not
        attempted, else `funding_failed`). Returns the trade ids it rejected. Nothing is ever left `new` on a finished order."""
        left = []
        for trade in self.tickets():
            self.reject(trade, "not_attempted", "the Trader did not place it in this run")
            left.append(trade.trade_id)
        for leg in [t for t in self.trades.values() if t.kind == "funding" and t.status == "new"]:
            parent = self.trades.get(leg.funds_for)
            code = "not_attempted" if parent is None or parent.reject_code == "not_attempted" else "funding_failed"
            self.reject(leg, code, f"its purchase {leg.funds_for} was not made, so nothing was sold")
            left.append(leg.trade_id)
        return left

    def cash_now(self) -> Decimal | None:
        """The cash the account has right now, as the Account Manager says it (a read of the book, or of its working portfolio): what the model's `verify_request` shows. None when it could not be read. The broker keeps
        no figure of its own: every call asks."""
        try:
            return self.account.current_portfolio().cash_avail
        except AssertionError:
            raise
        except Exception:   # the account could not be read: the figure is simply not shown
            return None

    # ---------------------------------------------------------------- the core

    def _place(self, trade: Trade, kind: str, price, note: str, placed_by: str) -> None:
        trade.order_type, trade.limit_price, trade.note, trade.placed_by, trade.placed_utc = kind, price, note, placed_by, utc_text(self.now)
        self.placed.append((trade.trade_id, placed_by))
        self._run(trade)

    def _run(self, trade: Trade, again: bool = False) -> None:
        """Judge a trade that has its order: permission (T1), the price, then a funded purchase or a plain one. `again`: an active trade of an earlier run (see `rejudge`)."""
        legs = self.legs_of(trade) if trade.side == "buy" else []
        code, reason = self.permit(trade)
        if code:
            self.reject(trade, code, reason)
            return self._fail_legs(trade, legs)
        execution = self._simulate(trade)
        if again and execution.status == "rejected" and execution.reject_code == "price_unavailable":
            execution = Execution("active", f"could not be judged again in this run: {execution.reason}")
        if execution.status != "filled" or not legs:
            return self._settle(trade, execution, legs)
        self._fund(trade, execution, legs)

    def _simulate(self, trade: Trade) -> Execution:
        """The fill rules for a trade against the snapshot (no cash, no shares: those are applied when it fills)."""
        slip = self.config.costs.slippage_bps
        if self.prices.failed:
            return Execution("rejected", self.prices.failed, reject_code="price_unavailable")
        if trade.symbol in self.prices.unavailable:
            return Execution("rejected", self.prices.unavailable[trade.symbol], reject_code="price_unavailable")
        if self.mode == "paper":
            return fill_paper(trade.side, trade.order_type, trade.limit_price, self.prices.quotes.get(trade.symbol), trade.limit_low, trade.limit_high, slip)
        if self.prices.entry_date is None:
            return Execution("active", "waiting for the next session: no daily bar is stored from the entry day on yet")
        bar = bar_on(self.prices.bars.get(trade.symbol), self.prices.entry_date)
        if bar is None:
            return Execution("rejected", f"{trade.symbol} has no stored bar on {self.prices.entry_date.isoformat()}", reject_code="price_unavailable")
        return fill_backtest(trade.side, trade.order_type, trade.limit_price, bar, trade.limit_low, trade.limit_high, slip)

    def _settle(self, trade: Trade, execution: Execution, legs: list[Trade]) -> None:
        """A trade with no funding sales (or one whose price did not fill): fill it, rest it, or reject it, and say the same of its sales."""
        if execution.status == "rejected":
            self.reject(trade, execution.reject_code, execution.reason)
            self._fail_legs(trade, legs, f"{trade.trade_id} was rejected ({execution.reject_code}): " + execution.reason)
        elif execution.status == "active":
            self._rest(trade, execution.reason)
            for leg in legs:
                self._rest(leg, f"waiting for its purchase {trade.trade_id}", placed=trade)
        else:
            self._fill(trade, execution)

    def _fund(self, buy: Trade, buy_execution: Execution, legs: list[Trade]) -> None:
        """A funded purchase that WOULD fill now (spec section 7, steps 2 to 4): see the module text. The purchase is approved for real only AFTER the sales are filled and booked, so the approval reads the cash the book
        shows then; before that the Account Manager is only asked whether the estimated proceeds would pay for one share."""
        price, plan, proceeds = buy_execution.price, [], Decimal("0.00")
        for leg in legs:
            leg.placed_by, leg.placed_utc = buy.placed_by, buy.placed_utc
            execution = self._simulate(leg)
            if execution.status == "filled":
                proceeds += trade_amount(leg.shares, execution.price)       # the sale's own amount, an estimate of what it brings: the Account Manager adds it to the cash for the question below
                plan.append((leg, execution))
            else:
                self.reject(leg, "funding_failed", self._why_not_sold(leg, execution))
        question = self._approval(buy, price, buy.shares, preview=proceeds)
        if not question.approved:
            why = (f"cash ${question.cash_avail:,.2f} plus about ${proceeds:,.2f} from the sales does not pay for one share at {price:,.4f}" if question.code == "insufficient_cash" and question.cash_avail is not None
                   else question.reason)
            self.reject(buy, question.code, f"{why}: nothing was sold")
            for leg, _ in plan:
                self.reject(leg, "funding_failed", f"nothing was sold: the sales would not pay for even one share of {buy.trade_id}")
            return
        for leg, execution in plan:
            self._fill(leg, execution, shares=leg.shares)     # a sale that fails is failed and not retried; the others go on, and so does the purchase (its approval reads the cash the book shows then)
        self._fill(buy, buy_execution)

    @staticmethod
    def _why_not_sold(leg: Trade, execution: Execution) -> str:
        if execution.status == "active":
            return f"it would not fill now: {execution.reason}"
        return f"it could not be sold: {execution.reason}"

    # ---------------------------------------------------------------- the handshake with the Account Manager

    def _approval(self, trade: Trade, price: Decimal, shares: Decimal, preview: Decimal | None = None):
        """Ask the Account Manager for the approval of this trade (step 1 of the handshake): the Trader's record of it, with the shares asked for, and the price it would fill at. `preview` None is the real approval
        (`approve_trade`); a number is a QUESTION about what it would say with that many more dollars available (`preview_trade`: nothing is approved, placed or reserved). Never raises: an Account Manager that cannot
        answer has not approved (`account_refused`)."""
        asked = trade.model_copy(update={"shares": shares}).model_dump()
        where = dict(order=self.order, now=self.now, on=self.prices.trade_date, commission=self.config.costs.commission_per_order)
        try:
            if preview is None:
                return self.account.approve_trade(asked, price, **where)
            return self.account.preview_trade(asked, price, expected_proceeds=preview, **where)
        except AssertionError:
            raise
        except Exception as error:   # the Account Manager is meant never to raise: if it does, nothing is approved
            return _Refusal(f"the Account Manager could not be asked ({type(error).__name__}): the trade is not approved")

    def preview(self, trade: Trade, price: Decimal, expected_proceeds: Decimal = Decimal(0)):
        """The Account Manager's answer to "would you approve this purchase, all of it, if `expected_proceeds` more were available?": a question for the funding pre-check and the user's consent (see `_fund` and execute's
        `_consent`). Nothing is placed or reserved. The Trader never works the funds out itself."""
        return self._approval(trade, price, trade.shares, preview=expected_proceeds)

    def _fill(self, trade: Trade, execution: Execution, shares: Decimal | None = None) -> None:
        """The handshake around a fill the price rules gave (see the module text): approval, the fill for at most the shares approved, the output check of that one fill, and its report to the Account Manager. Returns when the
        Account Manager has answered, so the next trade only starts after it. A trade the Account Manager does not approve is `rejected` with its code (a funding sale: `funding_failed`); a fill the book refuses, or
        could not confirm, is `rejected` `account_refused` and its `Fill` is not kept: it is not retried and the run goes on with the next trade (design choices entry 82)."""
        wanted = shares or trade.shares
        approval = self._approval(trade, execution.price, wanted)
        if not approval.approved:
            if trade.kind == "funding":
                return self.reject(trade, "funding_failed", f"the Account Manager did not approve the sale ({approval.code}): {approval.reason}")
            return self.reject(trade, approval.code, approval.reason + (f" ({self.cash_hint})" if approval.code == "insufficient_cash" and self.cash_hint else ""))
        count = approval.shares if trade.dollars is not None and trade.side == "buy" else min(wanted, approval.shares)      # never more than approved (the Account Manager never approves more than was asked). An order in DOLLARS (entry 111, M2) fills the shares its dollars pay for at this price: fewer than the estimate when the price is higher, more when it is lower
        for_what = "the money of this request (its pot)" if "the money of this request" in approval.reason else "the cash"      # entry 111, M1: what the Account Manager's approval was held by
        cut = (f"; an order of ${trade.dollars:,.2f}: {sh.text(count)} shares at {execution.price:,.4f} (the estimate was {sh.text(trade.shares_approved)})" + (f", held by {for_what}" if ", held by " in approval.reason else "") if trade.dollars is not None and trade.side == "buy" else
               f"; the size was cut from {sh.text(trade.shares_approved)} to {sh.text(count)} shares for {for_what}" if trade.side == "buy" and count < trade.shares_approved else
               f"; the size was cut from {sh.text(trade.shares_approved)} to {sh.text(count)} shares: only that many were sellable" if trade.side == "sell" and count < trade.shares_approved else "")
        commission = self.config.costs.commission_per_order
        fill = Fill(price=execution.price, shares=count, amount=trade_amount(count, execution.price), commission=commission, date=self.prices.trade_date.isoformat(), price_source=execution.price_source,
                    slippage_bps=execution.slippage_bps)
        done = trade.model_copy(update={"status": "filled", "reject_code": None, "shares": count, "fill": fill, "reason": execution.reason + cut})
        findings = verify_fill(self.order, done.model_dump(), [t.model_dump() for t in self.trades.values()])
        if findings:   # the fill disagrees with the decisions: it is not reported or booked (the trade is failed, and not retried); the next trade is placed as usual
            text = f"{trade.trade_id}: the output check found a fill that does not agree with the decisions, so it was not reported or booked: " + "; ".join(findings)
            self.notes.append(text)
            return self.reject(trade, "internal_error", text[:300])
        try:
            booking = self.account.record_fill(done.model_dump())
        except AssertionError:
            raise
        except Exception as error:   # the Account Manager is meant never to raise: if it does, nobody knows whether the trade was booked, and it is never assumed that it was
            booking = _Unconfirmed(f"the booking could not be confirmed ({type(error).__name__}): the account book may or may not hold this trade")
        self.answers[trade.trade_id] = {"account": booking.status, "cash_left": booking.cash_avail if booking.ok else None}
        if booking.note and booking.note not in self.notes:
            self.notes.append(booking.note)
        if booking.ok:
            trade.status, trade.reject_code, trade.shares, trade.fill, trade.reason, trade.booked = "filled", None, count, fill, done.reason, booking.booked
            self.filled.append(trade.trade_id)
        elif booking.status == "refused":
            self.notes.append(f"{trade.trade_id}: {trade.side} {sh.text(count)} {trade.symbol} was not booked: {booking.reason}")
            self.reject(trade, "account_refused", booking.reason)
        else:   # unconfirmed: nobody knows whether it was booked, so it is not counted as a fill; the trade is failed and not retried, and the next trade is placed (its approval reads the account as the book has it)
            self.notes.append(f"{trade.trade_id}: {trade.side} {sh.text(count)} {trade.symbol}: {booking.reason}")
            self.reject(trade, "account_refused", booking.reason)

    def _rest(self, trade: Trade, reason: str, placed: Trade | None = None) -> None:
        trade.status, trade.reject_code, trade.reason = "active", None, reason
        if placed is not None:
            trade.placed_by, trade.placed_utc = placed.placed_by, placed.placed_utc

    def reject(self, trade: Trade, code: str, reason: str) -> None:
        """Make a trade `rejected` with a code of the closed list and a sentence, and drop its fill (`placed_by` and the order are kept as they were)."""
        trade.status, trade.reject_code, trade.reason, trade.fill = "rejected", code, reason, None

    def _fail_legs(self, buy: Trade, legs: list[Trade], why: str | None = None) -> None:
        """The sales of a purchase that did not happen are `funding_failed`: nothing was sold."""
        for leg in legs:
            leg.placed_by, leg.placed_utc = buy.placed_by, buy.placed_utc
            self.reject(leg, "funding_failed", why or f"its purchase {buy.trade_id} was rejected ({buy.reject_code}), so nothing was sold")
