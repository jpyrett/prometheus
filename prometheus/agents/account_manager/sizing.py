"""How many shares to buy with the user's own money: the Account Manager's sizing (design choices entry 91; it began as design/sizing_in_account_manager_spec.md, entry 80). Plain Python: no model, no network, no file but the settings.

    sized = size_purchases(purchases, cash, config, allocation=None, budget=None, wanted=None)      # a list of `Sized`, one per purchase, in the order given
    equity, basis = equity_of(portfolio, prices)                                                    # cash plus the holdings at their prices (for "20% of my account", and the report)
    feasible, note = funds_check(cost, portfolio, prices, today)                                    # could the money exist, if holdings in profit were sold?

THE ONLY CHECK IS THE FUNDS (Jacob, 2026-11-15: *"the only sizing that we need is to make sure the trade order has enough funds to cover it"*, after *"profile limits never apply to any orders"* and *"I do not want any of these
limits ... remove them all from all tests and code"*). There is no limit of the profile any more (a share of the account in one stock, the loss a stop would cost, the exposure of one review, a cap on one order) and no
liquidity cap: those caps were this module's before entry 91, and they are gone, with the profile fields that fed them. A purchase is bounded by:

    money       all the cash available, for EVERY purchase: the cash is not reserved, so no purchase's cost is taken off what another sees (Jacob's rule of 2026-11-14); the Trader's handshake meets the cash as it is at each trade
    budget      the budget the user gave, less what the earlier purchases take                      None when no budget was given
    asked       the shares the user's own words came to (below)                                       None when the user did not say how much

A cap is whole shares (`math.floor`, with the settings' `float_guard` added first, so that 2.9999999999999996 is 3). A purchase with no usable price is not sized: 0 shares, `binding` "no_room", and the note says why. A size is
never computed from a guess.

THE ALLOCATION says how the money is spread over the purchases. Three ways (`ALLOCATIONS`; left out, None, is the last):

    round_robin          one share at a time, in rank order, until the funds are used up (Auto mode's way: Jacob, 2026-11-14)
    even                 the same DOLLARS for every purchase (the user said "split it evenly")
    as_much_as_possible  each purchase takes as much as the money, the budget and the amount asked allow, in rank order, the budget shared in that order (Ask mode's way when nothing was asked: the user sees the largest order)

ROUND ROBIN ("buy a share of the first stock, a share of the next, and keep buying shares until all the funds requested are used up"). It is arithmetic done here, so that each stock ends with a FINAL number of shares: the
Trader places ONE trade per stock, never one trade per share. The POT is the funds requested: the `budget` when one was given, else the cash, and never more than the cash. It is drawn down as shares are bought; the cash
itself is not reserved. The purchases are visited in RANK order (`Purchase.rank`, lowest first, ties in the order given); each ROUND buys ONE share of each purchase that is still open, if the share's cost is within what is left
of the pot and the purchase has fewer shares than the user asked for. A purchase that cannot take another share is CLOSED for the rest of the call and remembers WHY (`asked`, `budget` or `money`). The call stops when no purchase is
open. A share costs `trade_amount(n + 1, price) - trade_amount(n, price)`, in cents the way the account book rounds them, so a purchase's cost is exactly `trade_amount(shares, price)` and the total is the sum of those.

EVEN: every purchase gets the same DOLLARS: the pot divided by the number of purchases (and never more than the smallest amount any purchase was asked for), whole shares at each price. A purchase whose even share buys less than one
share is dropped (`no_room`, with the reason) and the others are balanced again; each purchase then carries the equal amount (`Sized.target_dollars`) and how many shared it (`Sized.of`).

THE AMOUNT THE USER ASKED FOR (`wanted`, design choices entry 85: the Account Manager's sizing questions in Ask mode). `wanted` is {record_id: whole shares}: how many shares the user's own words came to for that purchase (a
percentage of the account, a dollar amount or a number of shares, worked out by `clarify.plan_choice`, never by a model). The purchase never gets more than the user asked for; `binding` is "asked" when that was what stopped it.

`binding` says what decided the number of shares, one of: `money`, `budget` (that pool was the smallest: round robin: it closed the purchase; even: it set the equal amount), `asked` (the user asked for fewer shares than the money
allows), `even_split` (the equal amount was set by another purchase's smaller ask, so this one takes less than it could), `no_room` (not even one share fits, or the purchase could not be sized: 0 shares; the notes give the
reason), or None (never produced here: kept in the vocabulary for a decision that carries no size, a sale).

This module decides nothing about permission (that is `decisions.decide_trades`) and plans no sale (that is RaiseFunds): `funds_check` only says whether a sale of holdings in profit could pay for a purchase.
"""
import math
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prometheus.agents.account_manager.config import AccountManagerConfig, load_config
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.tools import trading_math as tm

CAP_NAMES = ("money", "budget")                                    # the pools: the order is the order of a tie
TIE_ORDER = (*CAP_NAMES, "asked")                                  # the order of a tie: a real pool before the amount the user asked for
BINDINGS = (*CAP_NAMES, "asked", "even_split", "no_room")          # what `Sized.binding` may be (None as well)
ALLOCATIONS = ("even", "round_robin", "as_much_as_possible")      # left out (None) is "as_much_as_possible"
Binding = Literal["money", "budget", "asked", "even_split", "no_room"]


class Purchase(BaseModel):
    """One purchase to size, as plain numbers (the Account Manager makes it from the recommendation and the Critic's review, or from a company the Advisor scored neutral: it holds no company text and no score)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action: Literal["open", "add"]
    price: float | None                              # the price to size at (the Critic's deal price, else the plan's entry mid); None: not known, and nothing is sized
    rank: int = Field(default=0, ge=0)               # the order the recommendations were ranked in, lowest first


class Sized(BaseModel):
    """How one purchase was sized (see the module text). `est_cost` is `shares` at the purchase's price in cents; `caps` has every name of `CAP_NAMES` (the most whole shares each pool allows, None for a pool that is not
    there); `notes` says, in plain English, anything that shaped the size. `allocation` is the way the money was spread ("even", "round_robin" or "as_much_as_possible"); `target_dollars` and `of` are the equal amount of an even
    split and how many purchases shared it (None otherwise)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    symbol: str
    shares: int = Field(ge=0)
    est_cost: float = Field(ge=0)
    caps: dict[str, int | None]
    binding: Binding | None
    notes: list[str] = Field(default_factory=list)
    allocation: Literal["even", "round_robin", "as_much_as_possible"] = "as_much_as_possible"
    target_dollars: float | None = None
    of: int | None = None
    asked: int | None = None                      # the shares the user asked for (`wanted`), when they said how much; None otherwise


def _amount(x, allow_zero: bool = False) -> float | None:
    """`x` as a float when it is a finite number above zero (at or above it with `allow_zero`), else None: a price or an equity that is not usable is not known."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        return None
    return float(x)


def _whole(x: float, guard: float) -> int:
    """Whole shares: rounded down, never below 0; `guard` is added first for the float error of the arithmetic."""
    return max(0, math.floor(x + guard))


def _dollars(x: float) -> float:
    """A dollar amount in cents, the way the account book rounds money (half up)."""
    return trade_amount(1, x)


def _tightest(known: dict[str, int]) -> str:
    return min(known, key=lambda name: (known[name], TIE_ORDER.index(name)))


def _pools(guard: float, cash: float, budget: float | None, price: float) -> dict[str, int | None]:
    """The pools as shares at `price`: ALL the cash (it is not reserved: no purchase's cost is taken off it) and what is left of the budget (None when no budget was given)."""
    return {"money": _whole(cash / price, guard), "budget": None if budget is None else _whole(budget / price, guard)}


def size_purchases(purchases: list[Purchase], cash: float, config: AccountManagerConfig | None = None, allocation: str | None = None, budget: float | None = None, wanted: dict[str, int] | None = None) -> list[Sized]:
    """The shares for each purchase (see the module text), one `Sized` for each, in the order given. `cash` is what may be spent in all (the free cash, or more when the caller has the money raised), `config` `load_config()`
    when left out. `allocation` is "even", "round_robin" or "as_much_as_possible" (None is the last); `budget` is the funds requested (the pot of the round robin and of the even split). Raises ValueError for what only a program
    gets wrong: an `allocation` that is not one of those, a `budget` that is not a number above zero, cash that is not a number from 0, two purchases with one `record_id`, or `wanted` with a record that is not a purchase or a
    count that is not a whole number from 0."""
    config = config or load_config()
    if allocation is not None and allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {', '.join(ALLOCATIONS)} (or left out)")
    if budget is not None and _amount(budget) is None:
        raise ValueError("budget must be a number of dollars above zero (or left out)")
    if _amount(cash, allow_zero=True) is None:
        raise ValueError("cash must be a number of dollars, zero or more")
    if len({p.record_id for p in purchases}) != len(purchases):
        raise ValueError("two purchases have the same record_id")
    wanted = dict(wanted or {})
    if set(wanted) - {p.record_id for p in purchases}:
        raise ValueError(f"wanted names a record that is not a purchase: {', '.join(sorted(set(wanted) - {p.record_id for p in purchases}))}")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in wanted.values()):
        raise ValueError("wanted must be a whole number of shares, zero or more, for each purchase")
    guard = config.sizing.float_guard
    order = sorted(range(len(purchases)), key=lambda i: (purchases[i].rank, i))
    done: dict[int, Sized] = {}
    way = allocation or "as_much_as_possible"

    def finish(i: int, shares: int, caps: dict, binding, more: list[str], **kept) -> None:
        p, price = purchases[i], _amount(purchases[i].price)
        done[i] = Sized(record_id=p.record_id, symbol=p.symbol, shares=shares, est_cost=trade_amount(shares, price) if shares and price else 0.0,
                        caps={name: caps.get(name) for name in CAP_NAMES}, binding=binding, notes=list(more), allocation=way, asked=wanted.get(p.record_id), **kept)

    sizable = []                                                                    # in rank order
    for i in order:
        if _amount(purchases[i].price) is None:
            finish(i, 0, {}, "no_room", ["not sized: no price (a size is never computed from a guess)"])
        else:
            sizable.append(i)
    if allocation == "even":
        _split_evenly(purchases, sizable, wanted, cash, budget, guard, finish)
    elif allocation == "round_robin":
        _round_robin(purchases, sizable, wanted, cash, budget, guard, finish)
    else:
        left_budget = None if budget is None else float(budget)                    # the budget is shared in rank order; the cash is not reserved: every purchase sees all of it
        for i in sizable:
            caps = _pools(guard, float(cash), left_budget, float(purchases[i].price))
            known = {name: n for name, n in caps.items() if n is not None}
            shares, binding = min(known.values()), _tightest(known)
            asked = wanted.get(purchases[i].record_id)
            if asked is not None and asked < shares:
                shares, binding = asked, "asked"
            if shares < 1:
                finish(i, 0, caps, "no_room", [f"not even one share fits (shares allowed: {', '.join(f'{name} {n}' for name, n in known.items())}" + (f", the amount asked for {asked}" if asked is not None else "") + ")"])
                continue
            finish(i, shares, caps, binding, [])
            left_budget = None if left_budget is None else left_budget - done[i].est_cost
    return [done[i] for i in range(len(purchases))]


def _split_evenly(purchases: list[Purchase], sizable: list[int], wanted: dict, cash: float, budget: float | None, guard: float, finish) -> None:
    """The even allocation (`allocation="even"`, see the module text): the target is the smaller of the pot's equal share and the smallest amount any purchase still in the split was asked for; a purchase that cannot buy one share
    at the target is dropped and the rest are balanced again."""
    price = {i: float(purchases[i].price) for i in sizable}
    ask = {i: wanted.get(purchases[i].record_id) for i in sizable}
    own = {i: float("inf") if ask[i] is None else float(ask[i]) for i in sizable}                 # the most shares each purchase was asked for (no ask: unbounded, the pot bounds it)
    pot = float(cash) if budget is None else min(float(cash), float(budget))
    pool_name = "budget" if budget is not None and float(budget) < float(cash) else "money"
    whole = {i: _pools(guard, float(cash), None if budget is None else float(budget), price[i]) for i in sizable}
    active, target, share, smallest = list(sizable), 0.0, 0.0, 0.0
    while active:
        share = pot / len(active)
        smallest = min(own[i] * price[i] for i in active)
        target = min(share, smallest)
        short = [i for i in active if _whole(target / price[i], guard) < 1]
        if not short:
            break
        for i in short:
            finish(i, 0, whole[i], "no_room", [f"an even share of ${target:,.2f} does not buy one share at ${price[i]:,.2f}"], target_dollars=_dollars(target), of=len(active))
        active = [i for i in active if i not in short]
    for i in active:
        shares = min(own[i], _whole(target / price[i], guard))
        binding = "asked" if ask[i] is not None and shares == own[i] else "even_split" if smallest < share else pool_name
        finish(i, int(shares), whole[i], binding, [f"even split: about ${target:,.2f} for each of {len(active)} purchases"], target_dollars=_dollars(target), of=len(active))


def _round_robin(purchases: list[Purchase], sizable: list[int], wanted: dict, cash: float, budget: float | None, guard: float, finish) -> None:
    """The round-robin allocation (`allocation="round_robin"`, see the module text). `sizable` is in rank order. Money is counted in `Decimal` cents, so that the total is exactly the sum of `trade_amount` of the shares."""
    price = {i: float(purchases[i].price) for i in sizable}
    ask = {i: wanted.get(purchases[i].record_id) for i in sizable}
    pot = float(cash) if budget is None else min(float(cash), float(budget))
    pot_name = "budget" if budget is not None and float(budget) < float(cash) else "money"                     # what the pot is called when it closes a purchase
    whole = {i: _pools(guard, float(cash), None if budget is None else float(budget), price[i]) for i in sizable}
    shares, cost, closed = dict.fromkeys(sizable, 0), {i: Decimal(0) for i in sizable}, {}   # shares taken, what they cost, and {purchase: what closed it}
    spent, rounds, open_ = Decimal(0), 0, list(sizable)
    while open_:
        took = False
        for i in list(open_):                                                                                  # one share of each purchase still open, in rank order
            n = shares[i]
            after = Decimal(str(trade_amount(n + 1, price[i])))
            step = max(after - cost[i], Decimal("0.01"))                                                       # this share's cost: the cost of n + 1 shares less the cost of n
            if ask[i] is not None and n >= ask[i]:
                why = "asked"
            elif _whole((pot - float(spent)) / float(step), guard) < 1:
                why = pot_name
            else:
                shares[i], spent, cost[i], took = n + 1, spent + step, after, True
                continue
            closed[i] = why
            open_.remove(i)
        rounds += took
    for i in sizable:
        text = {"budget": "the budget has no money for another share", "money": "the cash has no money for another share", "asked": "it has the shares the user asked for"}[closed[i]]
        header = f"round robin: {rounds} round{'s' if rounds != 1 else ''}, one share of each purchase in rank order, ${pot:,.2f} to spend ({pot_name}), ${pot - float(spent):,.2f} left"
        if shares[i]:
            finish(i, shares[i], whole[i], closed[i], [header, f"closed at {shares[i]} share{'s' if shares[i] != 1 else ''}: {text}"])
        else:
            finish(i, 0, whole[i], "no_room", [header, f"not even one share fits: {text}"])


def equity_of(portfolio: Portfolio, prices: dict[str, float]) -> tuple[float | None, str]:
    """(equity, what it was worked out from): the free cash plus every holding at its price in `prices`. A holding with no usable price is counted at its COST BASIS and the note says so. None only when the account holds nothing
    (no cash and no holdings): there is no equity to take a share of (the Account Manager's "20% of my account" needs one)."""
    value, unpriced = portfolio.cash_avail, []
    for position in portfolio.positions:
        price = _amount(prices.get(position.symbol))
        if price is None:
            unpriced.append(position.symbol)
            price = position.cost_basis
        value += position.shares * price
    if not portfolio.positions:
        basis = "cash only (no holdings)"
    else:
        basis = "cash and holdings at the prices given" + (f"; {', '.join(unpriced)} at cost basis (no price)" if unpriced else "")
    return (value if value > 0 else None), (basis if value > 0 else "the account holds no cash and no holdings")


def funds_check(cost: float, portfolio: Portfolio, prices: dict[str, float], today: date) -> tuple[bool, str]:
    """Whether the money for `cost` exists: free cash, or free cash plus what could be sold from the holdings that are in profit and sellable today (RaiseFunds' default rules: only holdings in the green, and one share of
    every holding is kept). A FLAG for the Account Manager: it does not run the tree of thoughts and plans no sale. A holding with no price in `prices` cannot be known to be in profit and is not counted (the note names it).
    Ported unchanged from `advisor/trade_plan.funds_check`, apart from that last sentence of the note."""
    if portfolio.cash_avail >= cost:
        return True, f"cash ${portfolio.cash_avail:,.2f} covers ${cost:,.2f}"
    holdings = [{"symbol": p.symbol, "shares": max(0, portfolio.sellable_shares(p.symbol, today) - 1), "price": prices[p.symbol], "basis": p.cost_basis} for p in portfolio.positions if p.symbol in prices]
    raisable = sum(h["shares"] * h["price"] for h in tm.rank_green(holdings, only_green=True))
    shortfall = cost - portfolio.cash_avail
    unpriced = [p.symbol for p in portfolio.positions if p.symbol not in prices]
    return raisable >= shortfall, (f"short by ${shortfall:,.2f}; ${raisable:,.2f} could be raised from holdings in profit that are sellable today (one share of each kept)"
                                   + (f"; no price for {', '.join(unpriced)}, not counted" if unpriced else ""))
