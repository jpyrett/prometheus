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

WHAT THE ANSWERS ARE. Both are small frozen records (`extra="forbid"`) and neither call ever raises for a data or database failure: the answer says what happened.
    Approval  `approved`, the `shares` approved (a purchase: the decision's shares cut to the whole shares the available cash pays for at the price, `portfolio.affordable_shares`, the account book's rounding; a sale:
              the shares held and sellable today (T+1), up to the decision's), the `cash_avail` the account has now, and when not approved a `code` of `APPROVAL_CODES` (the Trader's `reject_code`) and a `reason`.
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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prometheus.common.portfolio import Portfolio, affordable_shares
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
    """The Account Manager's answer to "may this trade be made now?". `shares` is what it approves (at least 1 when approved, 0 when not); `cash_avail` is the cash the account has now (None when it could not be read);
    `code` is one of `APPROVAL_CODES` when not approved; `reason` is the Account Manager's own sentence ("" when the trade is approved as asked)."""
    approved: bool
    shares: int = Field(default=0, ge=0)
    cash_avail: float | None = None
    code: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.approved and (self.shares < 1 or self.code is not None):
            raise ValueError("an approved trade has at least one share and no code")
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
    cash_avail: float | None = None
    reason: str = ""
    note: str = ""
    cash_before: float | None = None     # the cash the account showed in the SELECT before the booking (a run that books only)
    verified: bool = False               # a run that books: the account was read AGAIN after the commit and showed the trade (the totals changed and the trade row was there)

    @property
    def ok(self) -> bool:
        """The fill stands: it is booked, or applied to the working portfolio (which is what was asked for)."""
        return self.status in ("booked", "applied")

    @property
    def booked(self) -> bool:
        """The fill is in the Postgres account book."""
        return self.status == "booked"


def held(portfolio: Portfolio, symbol: str) -> int:
    """The shares of `symbol` the portfolio holds (0 when it holds none)."""
    position = portfolio.position(symbol)
    return position.shares if position is not None else 0


def unchanged(before: Portfolio, after: Portfolio, symbol: str) -> bool:
    """True when the cash and the shares of `symbol` are the same in both reads: nothing happened between them."""
    return round(before.cash_avail, 2) == round(after.cash_avail, 2) and held(before, symbol) == held(after, symbol)


def check_booked(before: Portfolio, after: Portfolio, row: dict | None, symbol: str, side: str, shares: int) -> list[str]:
    """What the read AFTER the commit shows against the read BEFORE it (design choices entry 82), as problems, one sentence each; none means the booking is confirmed: the trade is in the book's list of trades as it
    was placed, and the totals are different from before (the cash, and the shares of the symbol). `row` is the trade row read back from the book (None when there is none)."""
    problems = []
    if row is None:
        problems.append("the trade is not in the book's list of trades")
    elif (row["symbol"], row["side"], row["shares"]) != (symbol, side, shares):
        problems.append("the trade in the book's list is not the one that was placed")
    if round(before.cash_avail, 2) == round(after.cash_avail, 2):
        problems.append("the cash is the same as before the trade")
    if held(before, symbol) == held(after, symbol):
        problems.append(f"the shares of {symbol} are the same as before the trade")
    return problems


def refusal(code: str, reason: str, cash_avail: float | None = None) -> Approval:
    """An Approval that says no, with the sentence cut to a sane length."""
    return Approval(approved=False, shares=0, cash_avail=cash_avail, code=code, reason=" ".join(reason.split())[:SENTENCE])


def permission_code(reasons: list[str]) -> str:
    """The code for the reasons `work_order.verify_trade` gave: `approval_expired` when the approval running out is the ONLY one, else `not_permitted`."""
    return "approval_expired" if len(reasons) == 1 and reasons[0].startswith(EXPIRY_PREFIX) else "not_permitted"


def decide_approval(portfolio: Portfolio, trade: dict, price, on: date, commission: float = 0.0, expected_proceeds: float = 0.0) -> Approval:
    """What the account can take of this trade, from the portfolio as it is NOW (pure: no database, no clock). `trade` is the Trader's record of it as a dict (`symbol`, `side`, `shares`: what is asked, and
    `shares_approved`: what the decision allowed, which the approval never exceeds); `price` is the price it would fill at; `on` the trade date (the T+1 lock of a sale); `commission` the Trader's
    setting (0 in this version); `expected_proceeds` the estimated proceeds of sales that are to be filled BEFORE this purchase (the funding pre-check: it counts as cash for this answer only, and nothing is
    reserved or changed)."""
    side, symbol, asked, allowed = trade.get("side"), trade.get("symbol"), trade.get("shares"), trade.get("shares_approved")
    whole = lambda x: isinstance(x, int) and not isinstance(x, bool) and x >= 1
    if side not in ("buy", "sell") or not isinstance(symbol, str) or not whole(asked):
        return refusal("account_refused", "the trade is not a purchase or a sale of a whole number of shares of a symbol", portfolio.cash_avail)
    if not isinstance(price, (int, float)) or isinstance(price, bool) or not price > 0 or price == float("inf"):
        return refusal("account_refused", "the trade has no price above zero", portfolio.cash_avail)
    wanted = min(asked, allowed) if whole(allowed) else asked
    if side == "sell":
        if portfolio.position(symbol) is None:
            return refusal("not_held", f"the portfolio does not hold {symbol}, and there is no short sale", portfolio.cash_avail)
        sellable = portfolio.sellable_shares(symbol, on)
        if sellable < 1:
            return refusal("not_sellable_today", f"every share of {symbol} was bought today, and a share cannot be sold the day it was bought (T+1)", portfolio.cash_avail)
        shares = min(wanted, sellable)
        why = "" if shares == asked else f"cut from {asked} to {shares} shares: only that many are sellable"
        return Approval(approved=True, shares=shares, cash_avail=portfolio.cash_avail, reason=why)
    shares = affordable_shares(portfolio.cash_avail + expected_proceeds, price, wanted, commission)
    if shares < 1:
        return refusal("insufficient_cash", f"cash ${portfolio.cash_avail:,.2f} does not pay for one share at {price:,.4f}", portfolio.cash_avail)
    why = "" if shares == asked else f"cut from {asked} to {shares} shares for the cash"
    return Approval(approved=True, shares=shares, cash_avail=portfolio.cash_avail, reason=why)


def fill_parts(trade: dict) -> tuple[str, str, int, float, date, str | None]:
    """(symbol, side, shares, price, the day, record_id) of a trade the Trader reports as filled. Raises ValueError, with the reason, when it is not a filled trade with a fill that can be applied."""
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
    return trade.get("symbol"), trade.get("side"), fill.get("shares"), fill.get("price"), when, record_id if isinstance(record_id, str) else None


def unbooked_fills(trades: list) -> list[dict]:
    """The fills of an earlier run that live only on the order: the trades that are `filled`, with a fill, not booked, a recommendation or a funding sale. Sales first (a purchase is applied against the cash the
    sales before it left), otherwise in the order they are on the order."""
    own = [t for t in trades if isinstance(t, dict) and t.get("status") == "filled" and not t.get("booked") and isinstance(t.get("fill"), dict) and t.get("kind") in ("recommendation", "funding")]
    return sorted(own, key=lambda t: t.get("side") != "sell")
