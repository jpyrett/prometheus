"""The Account Manager's side of the per-trade handshake with the Trader (design choices entry 81; design/trader_handshake_spec.md).

    approval = manager.approve_trade(trade, price, order=order, now=now, on=trade_date)     # an `Approval`: may this trade be made now, and for how many shares?
    booking = manager.record_fill(trade)                                                     # a `Booking`: the fill is applied (and, with a saved account, booked in ONE Postgres transaction)

The Account Manager and its Postgres book are the ONLY owner of the funds and the positions. The Trader holds no cash and no portfolio, not a copy and not a running total: for every trade it asks
`approve_trade` (the Account Manager reads the funds and the holdings AS THEY ARE NOW), places the trade for at most the shares approved, and reports the fill AT ONCE with `record_fill`, which moves the cash and the
position together (`AccountStore.book_trade`: it locks the account row and EVERY position row of the account, checks that the account is active, that the cash covers a purchase and that the shares are held, and writes both or neither). Only
after that answer does the Trader ask for the next trade, so a purchase always meets the funds as the book has them after the earlier trades, and the Trader cannot overdraw them. Nothing is reserved for a later
purchase: any available funds can be used by the trade that asks first.

THE HANDSHAKE NEVER WAITS FOR A NEXT TRADE (Jacob, 2026-11-14: "The handshaking should not wait if there are no further trades to execute"). An approval is asked only for a trade that is ABOUT TO BE PLACED (one the
price rules fill now), and a booking is asked only for a fill the Account Manager approved: a trade that is refused at its approval, or rejected before it fills (no price, a price outside the limits, a resting order, an
account that refused), never causes a `record_fill`, moves nothing and waits for nothing. After the LAST trade's `record_fill` has been answered the run goes straight on to its end (the whole-run report, the account read back,
the records): there is no further `approve_trade`, no unanswered approval, no poll, no sleep, no timeout and no wait of any kind for a "next" trade. With no ticket to place (nothing permitted, everything already final from
an earlier run, or every ticket refused before placement) there is NO handshake at all: no `approve_trade`, no `record_fill`, and no read of the book beyond the two the run needs to report `portfolio_before` and
`portfolio_after`. Every call here is a plain function call that returns; nothing in this module or in the Trader's funnel blocks (tests record the calls and read the source for it).

THE POT AND THE DOLLARS (design choices entry 111; `design/dollar_orders_spec.md`). A run never spends more than its money (M1): a purchase whose decision carries a POT (`size.pot`: the budget the user gave, else the cash, never more than the cash) is
approved for no more than what is left of it (`spendable`, worked out by the Account Manager from the filled purchases on the work order and in this run: `AccountManager.spent_of_pot`), so what a trade fills at cannot push the run past its money; the
reason says "the money of this request" when that, and not the cash, was what held it. A purchase whose decision is an ORDER IN DOLLARS (`dollars`, M2) is approved for the shares those dollars (or what is left of the pot, or the cash) pay for at
the price it would fill at, `shares.buys`: fewer than the estimate when the price is higher, more when it is lower. A direct order whose shares the user typed has no pot and is never cut.

DECIMAL ALL THE WAY (design choices entry 112): the cash, the price, the pot, the dollars and every share count here are `Decimal` (`NonNegativeMoney`, `Price`, `Shares`); the arithmetic is exact (the cash less a cost, `min` of the pools) and the only roundings are the named ones
(`trade_amount`, `shares.buys`). A trade dict (the Trader's record) may carry a quantity or a price as the int or float `json` gave: they go through the doors (`shares.quantity`, `money.price`) before any arithmetic, and a float given to `decide_approval` itself
is a TypeError (`money.exact`), not a cent that quietly differs.

WHAT THE ANSWERS ARE. Both are small frozen records (`extra="forbid"`) and neither call ever raises for a data or database failure: the answer says what happened.
    Approval  `approved`, the `shares` approved (a purchase: the decision's shares cut to what the available cash pays for at the price, `portfolio.affordable_shares`, the account book's rounding: whole shares for a whole
              order, a fractional quantity for an order with a fraction (design choices entry 110); a sale: the shares held and sellable today (T+1), up to the decision's), the `cash_avail` the account has now, and when not approved a `code` of `APPROVAL_CODES` (the Trader's `reject_code`) and a `reason`.
    Booking   `status`: `booked` (in the Postgres book, in one transaction), `applied` (to the Account Manager's WORKING portfolio only: nothing was booked, which is what `book=False` and a run with no saved account mean),
              `refused` (the book or the portfolio said no: nothing changed; `reason` is its own sentence) or `unconfirmed` (a database failure: nobody knows whether the trade was booked, and it is NEVER assumed to have
              been). `cash_avail` is the cash after it, read back from the book, which is the truth.

THREE READS AROUND ONE LOCKED COMMIT, AND A TRADE THAT FAILS (design choices entry 82; Jacob, 2026-11-15). For a run that books, `record_fill` SELECTs the account, commits the trade in ONE transaction (row-level locks on
the whole portfolio, a re-read inside the transaction that rolls it back on any difference), and SELECTs the account AGAIN on a new connection: the trade row must be there and the cash and the shares of the symbol must
be different from the first read (`check_booked`). Only then is the answer `booked` and `verified`, and the Trader may ask for its next trade. A trade that fails is NOT done: nothing is committed (`refused`) or nobody can
settle it from here (`unconfirmed`); the Account Manager, the only agent the user talks to, shows the user the error and the account as it stands (`tell_user`, `show_portfolio`), the Trader marks the trade `rejected`
`account_refused` and never retries it, and the next trade goes on: nothing halts the run (the next approval reads the account as the book has it, so a failure cannot overdraw it).

WHERE THE FUNDS ARE READ FROM. With a saved account and `book` true (`trading_portfolio(portfolio, book=True)`): from the book, every time (`list_portfolio`). Otherwise from the Account Manager's own WORKING portfolio:
the portfolio `trading_portfolio` stored (a desk JSON file, a test, or a snapshot of the saved account when `book` is false), which `record_fill` changes with `apply_fill`. It is the Account Manager's, never the Trader's.
`apply_earlier_fills` puts into it the fills of an EARLIER run that were never booked (they live only on `order.trades`), so that a second run of the same order sees the cash the first one spent.
"""
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from prometheus.common import shares as sh
from prometheus.common.money import NonNegativeMoney, exact, price as price_door
from prometheus.common.portfolio import Portfolio, affordable_shares
from prometheus.common.trace import traced
from prometheus.common.work_order import EXPIRY_PREFIX, iso_date

# The reasons the Account Manager does not approve a trade. They are words of the Trader's closed list of reject codes (`agents/trader/record.py: REJECT_CODES`; a test asserts it): the Trader makes the trade
# `rejected` with the code it is given. `account_refused` is the account's own no: it is not active, it could not be read, the book refused the fill.
APPROVAL_CODES = ("not_permitted", "approval_expired", "insufficient_cash", "not_held", "not_sellable_today", "account_refused")
BOOKING_STATUSES = ("booked", "applied", "refused", "unconfirmed")
SENTENCE = 300                      # the most characters of a sentence that comes from the book or from an error

NO_STORE_NOTE = "book: the fills were not booked: this Account Manager has no saved account (connect one with store= and account_id=)"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Approval(_Record):
    """The Account Manager's answer to "may this trade be made now?". `shares` is what it approves (a quantity above zero when approved, 0 when not: a whole number is an int, a fraction a float of at most 6 decimals); `cash_avail` is the cash the account has now (None when it could not be read);
    `code` is one of `APPROVAL_CODES` when not approved; `reason` is the Account Manager's own sentence ("" when the trade is approved as asked)."""
    approved: bool
    shares: sh.Shares = Decimal(0)
    cash_avail: NonNegativeMoney | None = None
    code: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.approved and (self.shares <= 0 or self.code is not None):
            raise ValueError("an approved trade has shares above zero and no code")
        if not self.approved and (self.shares != 0 or self.code not in APPROVAL_CODES):
            raise ValueError(f"a trade that is not approved has no shares and a code of {', '.join(APPROVAL_CODES)}, not {self.code!r}")
        return self


class Booking(_Record):
    """The Account Manager's answer to "this trade was filled". `status` is one of `BOOKING_STATUSES`; `cash_avail` the cash after it (None when it is not known); `reason` the sentence of a `refused` or `unconfirmed`
    answer (the book's own, or the kind of the failure: never connection details); `note` something worth saying once that is not a failure ("book: the fills were not booked: ..." when booking was asked for and
    there is no saved account). `cash_before` and `verified` are the two reads around a booking (design choices entry 82): the cash the SELECT before it showed, and whether the SELECT after the commit confirmed it.
    `refused` is certain (nothing was committed, and for a failure of the connection the account was read again and is unchanged); `unconfirmed` is the case nobody can settle from here: the Trader marks the trade failed
    either way, shows the sentence, and goes on to the next trade."""
    status: Literal["booked", "applied", "refused", "unconfirmed"]
    cash_avail: NonNegativeMoney | None = None
    reason: str = ""
    note: str = ""
    cash_before: NonNegativeMoney | None = None     # the cash the account showed in the SELECT before the booking (a run that books only)
    verified: bool = False               # a run that books: the account was read AGAIN after the commit and showed the trade (the totals changed and the trade row was there)

    @property
    def ok(self) -> bool:
        """The fill stands: it is booked, or applied to the working portfolio (which is what was asked for)."""
        return self.status in ("booked", "applied")

    @property
    def booked(self) -> bool:
        """The fill is in the Postgres account book."""
        return self.status == "booked"


def held(portfolio: Portfolio, symbol: str) -> Decimal:
    """The shares of `symbol` the portfolio holds (0 when it holds none)."""
    position = portfolio.position(symbol)
    return position.shares if position is not None else Decimal(0)


def unchanged(before: Portfolio, after: Portfolio, symbol: str) -> bool:
    """True when the cash and the shares of `symbol` are the same in both reads: nothing happened between them."""
    return before.cash_avail == after.cash_avail and sh.same(held(before, symbol), held(after, symbol))


def check_booked(before: Portfolio, after: Portfolio, row: dict | None, symbol: str, side: str, shares: Decimal) -> list[str]:
    """What the read AFTER the commit shows against the read BEFORE it (design choices entry 82), as problems, one sentence each; none means the booking is confirmed: the trade is in the book's list of trades as it
    was placed, and the totals are different from before (the cash, and the shares of the symbol). `row` is the trade row read back from the book (None when there is none). The shares are compared as quantities, at six
    decimals (`shares.same`): the book returns a Decimal and the fill holds a float, which compared as they are would make every fractional booking look unconfirmed (entry 110)."""
    problems = []
    if row is None:
        problems.append("the trade is not in the book's list of trades")
    elif (row["symbol"], row["side"]) != (symbol, side) or not sh.same(row["shares"], shares):
        problems.append("the trade in the book's list is not the one that was placed")
    if before.cash_avail == after.cash_avail:
        problems.append("the cash is the same as before the trade")
    if sh.same(held(before, symbol), held(after, symbol)):
        problems.append(f"the shares of {symbol} are the same as before the trade")
    return problems


def refusal(code: str, reason: str, cash_avail: Decimal | None = None) -> Approval:
    """An Approval that says no, with the sentence cut to a sane length."""
    return Approval(approved=False, shares=0, cash_avail=cash_avail, code=code, reason=" ".join(reason.split())[:SENTENCE])


def permission_code(reasons: list[str]) -> str:
    """The code for the reasons `work_order.verify_trade` gave: `approval_expired` when the approval running out is the ONLY one, else `not_permitted`."""
    return "approval_expired" if len(reasons) == 1 and reasons[0].startswith(EXPIRY_PREFIX) else "not_permitted"


@traced("account_manager.decide_approval")
def decide_approval(portfolio: Portfolio, trade: dict, price: Decimal, on: date, commission: Decimal = Decimal(0), expected_proceeds: Decimal = Decimal(0), spendable: Decimal | None = None, pot: Decimal | None = None,
                    dollars: Decimal | None = None) -> Approval:
    """What the account can take of this trade, from the portfolio as it is NOW (pure: no database, no clock). `trade` is the Trader's record of it as a dict (`symbol`, `side`, `shares`: what is asked, and
    `shares_approved`: what the decision allowed, which the approval never exceeds); `price` is the price it would fill at; `on` the trade date (the T+1 lock of a sale); `commission` the Trader's
    setting (0 in this version); `expected_proceeds` the estimated proceeds of sales that are to be filled BEFORE this purchase (the funding pre-check: it counts as cash for this answer only, and nothing is
    reserved or changed). `spendable` is what the RUN may still spend (design choices entry 111, M1): the pot of the request less what its purchases have already cost; None when the purchase has no pot (a direct order whose
    shares the user typed, a RaiseFunds purchase): its limit is the cash, as ever. A purchase is approved for what `min(cash now + expected proceeds, spendable)` pays for, so that a run never spends more than its money
    whatever price a trade fills at; `pot` is only for the sentence. Nothing left to spend is `insufficient_cash`, and the reason says when it is the money of the request that is used up and not the cash.
    `dollars` is the decision's ORDER IN DOLLARS (design choices entry 111, M2; None: an order in shares, as ever): the most the purchase may cost. The Account Manager turns the dollars into shares HERE, at the price it would fill
    at: it approves `shares.buys(min(dollars, spendable, cash now + expected proceeds), price)`, which can be FEWER shares than the estimate the decision was sized with (the price is higher) or MORE (it is lower), and
    never costs more than the dollars. The asked shares and the shares the decision allowed are only the estimate then."""
    side, symbol, asked, allowed = trade.get("side"), trade.get("symbol"), trade.get("shares"), trade.get("shares_approved")
    usable = lambda x: _quantity(x) > 0
    if side not in ("buy", "sell") or not isinstance(symbol, str) or not usable(asked):
        return refusal("account_refused", "the trade is not a purchase or a sale of a number of shares above zero of a symbol", portfolio.cash_avail)
    asked = sh.quantity(asked)
    try:
        price = exact(price, "the price")                      # a float is a TypeError: a price comes in through a door, once
    except ValueError:
        return refusal("account_refused", "the trade has no price above zero", portfolio.cash_avail)
    if not price > 0:
        return refusal("account_refused", "the trade has no price above zero", portfolio.cash_avail)
    wanted = min(asked, sh.quantity(allowed)) if usable(allowed) else asked
    if side == "sell":
        if portfolio.position(symbol) is None:
            return refusal("not_held", f"the portfolio does not hold {symbol}, and there is no short sale", portfolio.cash_avail)
        sellable = portfolio.sellable_shares(symbol, on)
        if sellable <= 0:
            return refusal("not_sellable_today", f"every share of {symbol} was bought today, and a share cannot be sold the day it was bought (T+1)", portfolio.cash_avail)
        shares = min(wanted, sellable)
        why = "" if shares == asked else f"cut from {sh.text(asked)} to {sh.text(shares)} shares: only that many are sellable"
        return Approval(approved=True, shares=shares, cash_avail=portfolio.cash_avail, reason=why)
    available = portfolio.cash_avail + expected_proceeds
    if dollars is not None:
        return _approve_dollars(portfolio, price, commission, available, spendable, pot, dollars)
    held_by_pot = spendable is not None and spendable < available                                # the money of the request is what limits it, not the cash
    shares = affordable_shares(min(available, spendable) if held_by_pot else available, price, wanted, commission)
    what = "one share" if sh.whole(wanted) else "any of the shares"
    if shares <= 0:
        if held_by_pot:
            left = f"${max(spendable, Decimal(0)):,.2f} of its ${pot:,.2f} is left" if pot is not None else f"${max(spendable, Decimal(0)):,.2f} of it is left"
            return refusal("insufficient_cash", f"the money of this request is used up: {left}, and that does not pay for {what} at {price:,.4f} (the cash, ${portfolio.cash_avail:,.2f}, is not the limit)", portfolio.cash_avail)
        return refusal("insufficient_cash", f"cash ${portfolio.cash_avail:,.2f} does not pay for {what} at {price:,.4f}", portfolio.cash_avail)
    why = "" if shares == asked else f"cut from {sh.text(asked)} to {sh.text(shares)} shares for " + (f"the money of this request (${spendable:,.2f} is left of it)" if held_by_pot else "the cash")
    return Approval(approved=True, shares=shares, cash_avail=portfolio.cash_avail, reason=why)


def _approve_dollars(portfolio: Portfolio, price: Decimal, commission: Decimal, available: Decimal, spendable: Decimal | None, pot: Decimal | None, dollars: Decimal) -> Approval:
    """The approval of an ORDER IN DOLLARS: `shares.buys` of the least of the dollars, what is left of the pot and the cash (less the commission), at the price it would fill at. What held it, when it was not the dollars, is
    in the reason; nothing that buys is `insufficient_cash`."""
    limit = min(dollars, available, *([] if spendable is None else [spendable]))
    by = "the money of this request" if spendable is not None and spendable < min(dollars, available) else "the cash" if available < dollars else None
    shares = sh.buys(limit - commission, price)
    if shares <= 0:
        if by == "the money of this request":
            left = f"${max(spendable, Decimal(0)):,.2f} of its ${pot:,.2f} is left" if pot is not None else f"${max(spendable, Decimal(0)):,.2f} of it is left"
            return refusal("insufficient_cash", f"the money of this request is used up: {left}, and that does not pay for any of an order of ${dollars:,.2f} at {price:,.4f} (the cash, ${portfolio.cash_avail:,.2f}, is not the limit)", portfolio.cash_avail)
        return refusal("insufficient_cash", f"${limit:,.2f} does not pay for any of an order of ${dollars:,.2f} at {price:,.4f} (cash ${portfolio.cash_avail:,.2f})", portfolio.cash_avail)
    why = f"an order of ${dollars:,.2f} buys {sh.text(shares)} shares at {price:,.4f}" + (f", held by {by} to ${limit:,.2f}" if by else "")
    return Approval(approved=True, shares=shares, cash_avail=portfolio.cash_avail, reason=why)


def _quantity(x) -> Decimal:
    """`x` as a share quantity, or 0 when it is not one (a bool, a string, a negative number, more than six decimals): what no trade can be made of."""
    try:
        return sh.quantity(x)
    except ValueError:
        return Decimal(0)


def fill_parts(trade: dict) -> tuple[str, str, Decimal, Decimal, date, str | None]:
    """(symbol, side, shares, price, the day, record_id) of a trade the Trader reports as filled: the shares and the price go through their doors here (the fill's numbers are Decimals from the Trader, or the int and float that `json`
    gave for a trade read back from a saved order). Raises ValueError, with the reason, when it is not a filled trade with a fill that can be applied."""
    if not isinstance(trade, dict):
        raise ValueError("the trade is not a record")
    if trade.get("status") != "filled":
        raise ValueError(f"the trade is {trade.get('status')!r}, not filled: only a fill is recorded")
    fill = trade.get("fill")
    if not isinstance(fill, dict):
        raise ValueError("the trade is filled but has no fill")
    when = iso_date(fill.get("date"))
    if when is None:
        raise ValueError(f"the fill's date {str(fill.get('date'))[:20]!r} is not a date")
    record_id = trade.get("record_id")
    try:
        shares = sh.quantity(fill.get("shares"))
    except ValueError:
        raise ValueError("shares must be a number above zero (a whole number, or one with at most 6 decimals)") from None
    try:
        at = price_door(fill.get("price"))
    except ValueError:
        raise ValueError("price must be a finite number above zero") from None
    return trade.get("symbol"), trade.get("side"), shares, at, when, record_id if isinstance(record_id, str) else None


def unbooked_fills(trades: list) -> list[dict]:
    """The fills of an earlier run that live only on the order: the trades that are `filled`, with a fill, not booked, a recommendation or a funding sale. Sales first (a purchase is applied against the cash the
    sales before it left), otherwise in the order they are on the order."""
    own = [t for t in trades if isinstance(t, dict) and t.get("status") == "filled" and not t.get("booked") and isinstance(t.get("fill"), dict) and t.get("kind") in ("recommendation", "funding")]
    return sorted(own, key=lambda t: t.get("side") != "sell")
