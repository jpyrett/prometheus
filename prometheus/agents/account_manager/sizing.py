"""How many shares to buy: the Account Manager's sizing (design/sizing_in_account_manager_spec.md). Plain Python: no model, no network, no file but the settings.

    sized = size_purchases(purchases, equity, cash, profile.sizing_limits(), config, allocation=None, budget=None, limit_overrides=None)       # a list of `Sized`, one per purchase, in the order given
    equity, basis = equity_of(portfolio, prices)                                                                                            # cash plus the holdings at their prices
    feasible, note = funds_check(cost, portfolio, prices, today)                                                                            # could the money exist, if holdings in profit were sold?

The Advisor and the Critic decide WHICH companies; HOW MUCH is the Account Manager's, from the user's money, the user's limits and what the user said. This is the arithmetic that was the Critic's
(`critic/risk.size_caps`, `exposure_cap`) and the Advisor's (`advisor/trade_plan.build`, `funds_check`), moved here and reshaped: the inputs are plain numbers, and this module imports nothing of the Critic or the Advisor.

FOR EACH PURCHASE (an `open` or an `add` that the Critic approved) the CAPS are the most whole shares each limit allows. `None` is a cap that could not be worked out, and the notes say which and why:

    position    max_position_pct x equity - what is already held (at the price), over the price      needs the equity
    loss        max_loss_per_trade_pct x equity / the plan's stop distance                            needs the equity and a stop distance
    order       max_order_dollars / price                                                             None when the user set no order limit
    liquidity   max_adv_pct x average daily dollar volume (20 days) / price                           needs that volume; `max_adv_pct` is in configs/account_manager_v1.json
    exposure    what is left of max_new_exposure_pct x equity after the purchases ranked before it    shared by the whole call (below)
    money       all the cash available, for EVERY purchase: the cash is not reserved, so no purchase's cost is taken off what another sees (Jacob's rule of 2026-11-14)
    budget      the budget the user gave, less what the earlier purchases take                        None when no budget was given

The four first are LIMITS of one purchase; exposure and budget are POOLS the purchases share (money is the whole cash for each one, see above). A cap is whole shares (`math.floor`, with the settings' `float_guard`
added first, so that 2.9999999999999996 is 3). A cap that cannot be worked out never stops the others from applying. But when NOT ONE of the four limits can be worked out (no price, or no equity, no order limit, no stop
distance and no volume), the purchase is not sized: 0 shares, `binding` "no_room", and the note says why. A size is never computed from a guess, and it is never "all the cash".

THE ALLOCATION says how the money is spread over the purchases. Three ways (`ALLOCATIONS`; left out, None, is the last):

    round_robin          one share at a time, in rank order, until the funds are used up or every purchase is at a limit (Auto mode's way: Jacob, 2026-11-14)
    even                 the same DOLLARS for every purchase (the user said "split it evenly")
    as_much_as_possible  each purchase takes as much as its caps allow, in rank order, the exposure limit and the budget shared in that order (Ask mode's way when nothing was asked: the user sees the largest order)

ROUND ROBIN ("buy a share of the first stock, a share of the next, and keep buying shares until all the funds requested are used up"). It is arithmetic done here, so that each stock ends with a FINAL number of
shares: the Trader places ONE trade per stock, never one trade per share. The POT is the funds requested: the `budget` when one was given, else the cash, and never more than the cash. It is drawn down as shares
are bought; the cash itself is not reserved (every purchase's own `money` cap still shows all of it). The purchases are visited in RANK order (`Purchase.rank`, lowest first, ties in the order given); each ROUND buys ONE
share of each purchase that is still open, if ALL of these hold: the purchase is under its own limits (`shares < min(position, loss, order, liquidity)`), one more share fits what is left of the exposure room (the pool
`max_new_exposure_pct x equity` all the purchases share, when the equity is known), and the share's cost is within what is left of the pot. A purchase that cannot take another share is CLOSED for the rest of the call and
remembers WHY: the first of its own limit (by name, the first of position, loss, order, liquidity when two tie), exposure, then budget (when a budget was the smaller pot) or money. The call stops when no purchase is open, so
the pot is spent to less than the cost of the cheapest share that could still have been bought (or every purchase is at its own limit). A share costs `trade_amount(n + 1, price) - trade_amount(n, price)`, in cents the way the
account book rounds them, so a purchase's cost is exactly `trade_amount(shares, price)` and the total is the sum of those. Its `binding` is the reason it closed; 0 shares is `no_room` (not even one share fit), with the
reason in the notes.

THE OTHER ALLOCATIONS. By default each purchase takes as much as its caps allow ("as much as I can"), in RANK order, the exposure limit and the budget shared in that order: the first purchase takes what it can, the
next one what is left of them. The CASH is not reserved: every purchase may use all of it. `allocation="even"` gives every purchase the same DOLLARS instead: the largest equal amount that no purchase's own limits and not the
exposure room, the cash or the budget (each divided by the number of purchases, so that the total is at most the cash or the budget) rule out, whole shares at each price, never above a purchase's own limits. A purchase whose
even share buys less than one share is dropped (`no_room`, with the reason) and the others are balanced again; each purchase then carries the equal amount (`Sized.target_dollars`) and how many shared it (`Sized.of`).
`budget` caps the dollars of all the purchases together. `limit_overrides` ({limit name: value}, any of the four limits of the user's profile) replaces a limit for THIS call: a limit the user raised for this
purchase. It is checked like the profile (a name that is not a limit or a value out of range is a ValueError), and every purchase says in its notes that it was sized under a changed limit.

THE AMOUNT THE USER ASKED FOR (`wanted`, design choices entry 85: the Account Manager's sizing questions in Ask mode). `wanted` is {record_id: whole shares}: how many shares the user's own words came to for that purchase
(a percentage of the account, a dollar amount or a number of shares, worked out by `clarify.plan_choice`, never by a model). It is one more limit of that purchase, `asked`, next to position, loss, order and liquidity: the
purchase never gets more than the user asked for, and never more than any limit allows. It is not one of the seven `caps` (those are the user's limits and the pools); `Sized.asked` carries it, and `binding` is "asked" when it
was the smallest (a tie goes to the real limit, so "asked" is said only when the user asked for LESS than every limit allows).

`binding` says what decided the number of shares, one of:

    position, loss, order, liquidity     that limit was the tightest (round robin: the purchase stopped at it)
    exposure, money, budget              that pool was (round robin: it closed the purchase; even: it set the equal amount)
    asked                                the user asked for fewer shares than every limit allows (`wanted`)
    even_split                           the equal amount was set by another purchase's own limit, so this one takes less than it could
    no_room                              not even one share fits (or the purchase could not be sized): 0 shares; the notes give the reason
    None                                 never produced here: kept in the vocabulary for a decision that carries no size (a sale)

When two caps tie, the first of the list above wins. `Sized.caps` has all seven keys. With the default allocation exposure and budget are the room that was left for the purchase when its turn came; with the even AND the
round-robin allocations they are the WHOLE room, as if this purchase were alone (the purchases share it: the notes and `binding` say how it was divided). Money is all the cash. They are None for a purchase that was ruled
out before the pools were shared (no price, no limit worked out, or a limit that leaves no whole share). Costs are in cents, the way the account book rounds them (`common.portfolio.trade_amount`).

This module decides nothing about permission (that is `decisions.decide_trades`) and plans no sale (that is RaiseFunds): `funds_check` only says whether a sale of holdings in profit could pay for a purchase.
"""
import math
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prometheus.agents.account_manager.config import AccountManagerConfig, load_config
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.common.user_profile import SizingLimits
from prometheus.tools import trading_math as tm

LIMIT_CAPS = ("position", "loss", "order", "liquidity")          # the limits of ONE purchase
CAP_NAMES = (*LIMIT_CAPS, "exposure", "money", "budget")           # every cap; the order is the order of the text above: it breaks a tie
TIE_ORDER = (*CAP_NAMES, "asked")                                  # the order of a tie between two limits: a real limit before the amount the user asked for
BINDINGS = (*CAP_NAMES, "asked", "even_split", "no_room")          # what `Sized.binding` may be (None as well)
ALLOCATIONS = ("even", "round_robin", "as_much_as_possible")      # left out (None) is "as_much_as_possible"
Binding = Literal["position", "loss", "order", "liquidity", "exposure", "money", "budget", "asked", "even_split", "no_room"]


class Purchase(BaseModel):
    """One purchase the Critic approved, as plain numbers (the Account Manager makes it from the recommendation and the Critic's review: it holds no company text and no score)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action: Literal["open", "add"]
    price: float | None                              # the price to size at (the Critic's deal price, else the plan's entry mid); None: not known, and nothing is sized
    stop_distance: float | None = None               # the plan's stop distance in dollars per share; None: the loss cap cannot be worked out
    held_shares: int = Field(default=0, ge=0)        # shares already held: they count against the position cap
    adv_20d_dollars: float | None = None             # average daily dollars traded (20 days); None: the liquidity cap cannot be worked out
    rank: int = Field(default=0, ge=0)               # the order the recommendations were ranked in, lowest first


class Sized(BaseModel):
    """How one purchase was sized (see the module text). `est_cost` is `shares` at the purchase's price in cents; `caps` has every name of `CAP_NAMES`; `notes` says, in plain English, every cap that could not be worked out and why, and anything else that shaped the size.
    `allocation` is the way the money was spread ("even", "round_robin" or "as_much_as_possible"); `target_dollars` and `of` are the equal amount of an even split and how many purchases shared it (None otherwise)."""
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
    """`x` as a float when it is a finite number above zero (at or above it with `allow_zero`), else None: a price, an equity or a distance that is not usable is not known."""
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


def _listing(caps: dict[str, int | None]) -> str:
    return ", ".join(f"{'the amount asked for' if name == 'asked' else name} {n}" for name, n in caps.items() if n is not None)


def limit_caps(purchase: Purchase, equity: float | None, limits: SizingLimits, config: AccountManagerConfig) -> tuple[dict[str, int | None], list[str]]:
    """({position, loss, order, liquidity: the most shares that limit allows, or None}, the notes for every one that could not be worked out). No pool is in it. Ported from `critic/risk.size_caps` (position, loss, order) and
    the Advisor's `trade_plan.build` (liquidity). With no usable price nothing can be worked out."""
    guard, notes = config.sizing.float_guard, []
    caps: dict[str, int | None] = dict.fromkeys(LIMIT_CAPS)
    price = _amount(purchase.price)
    if price is None:
        return caps, [f"{purchase.symbol} has no price, so no cap can be worked out"]
    equity = _amount(equity)
    if equity is None:
        notes.append("position cap not worked out: the equity is not known")
    else:
        caps["position"] = _whole((limits.max_position_pct * equity - purchase.held_shares * price) / price, guard)
    stop = _amount(purchase.stop_distance)
    if equity is None or stop is None:
        why = [reason for reason, missing in (("the equity is not known", equity is None), ("the plan has no stop distance", stop is None)) if missing]
        notes.append(f"loss cap not worked out: {' and '.join(why)}")
    else:
        caps["loss"] = _whole(limits.max_loss_per_trade_pct * equity / stop, guard)
    if limits.max_order_dollars is not None:
        caps["order"] = _whole(limits.max_order_dollars / price, guard)
    adv = _amount(purchase.adv_20d_dollars, allow_zero=True)
    if adv is None:
        notes.append("liquidity cap not worked out: the average daily dollar volume is not known")
    else:
        caps["liquidity"] = _whole(config.sizing.max_adv_pct * adv / price, guard)
    return caps, notes


def _pools(room: float | None, guard: float, cash: float, budget: float | None, used: float, price: float) -> dict[str, int | None]:
    """The pools as shares at `price`: what is left of the exposure room and of the budget (None for a pool that is not there) after `used` dollars of the purchases ranked before, and ALL the cash (it is not reserved: no purchase's cost is taken off it)."""
    return {"exposure": None if room is None else _whole((room - used) / price, guard), "money": _whole(cash / price, guard), "budget": None if budget is None else _whole(budget / price, guard)}


def size_purchases(purchases: list[Purchase], equity: float | None, cash: float, limits: SizingLimits, config: AccountManagerConfig | None = None, allocation: str | None = None,
                   budget: float | None = None, limit_overrides: dict | None = None, wanted: dict[str, int] | None = None) -> list[Sized]:
    """The shares for each purchase (see the module text), one `Sized` for each, in the order given. `equity` is cash plus holdings (`equity_of`; None: not known), `cash` what may be spent in all (the free cash, or more when
    the caller has the money raised), `limits` the user's (`UserProfile.sizing_limits()`), `config` `load_config()` when left out. `allocation` is "even", "round_robin" or "as_much_as_possible" (None is the last); `budget` is the
    funds requested (the pot of the round robin and of the even split). Raises ValueError for what only a program gets wrong: an `allocation` that is not one of those, a `budget` that is not a number above zero, cash that is
    not a number from 0, two purchases with one `record_id`, `limit_overrides` that do not make valid limits, or `wanted` with a record that is not a purchase or a count that is not a whole number from 0."""
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
    guard, shared = config.sizing.float_guard, []
    if limit_overrides:
        changed = limits.changed(limit_overrides)
        text = lambda value: "no limit" if value is None else f"{value:g}"
        shared = [f"{name} is {text(getattr(changed, name))} for this purchase (the profile says {text(getattr(limits, name))}): the user changed it" for name in limit_overrides if getattr(changed, name) != getattr(limits, name)]
        limits = changed
    equity = _amount(equity)
    room = None if equity is None else limits.max_new_exposure_pct * equity       # what all the purchases may add together, in dollars
    order = sorted(range(len(purchases)), key=lambda i: (purchases[i].rank, i))
    plan: dict[int, tuple[dict, list[str]]] = {}                                    # {index: (limit caps, notes about caps that could not be worked out)}
    for i in order:
        caps, why = limit_caps(purchases[i], equity, limits, config)
        if purchases[i].record_id in wanted:
            caps = {**caps, "asked": wanted[purchases[i].record_id]}                  # the user's own amount is one more limit of this purchase (it never reaches `Sized.caps`: see `finish`)
        plan[i] = (caps, [*why, "exposure cap not worked out: the equity is not known"] if room is None and _amount(purchases[i].price) is not None else why)
    done: dict[int, Sized] = {}

    way = allocation or "as_much_as_possible"

    def finish(i: int, shares: int, caps: dict, binding, more: list[str], **kept) -> None:
        p, price = purchases[i], _amount(purchases[i].price)
        done[i] = Sized(record_id=p.record_id, symbol=p.symbol, shares=shares, est_cost=trade_amount(shares, price) if shares and price else 0.0,
                        caps={name: caps.get(name) for name in CAP_NAMES}, binding=binding, notes=[*plan[i][1], *shared, *more], allocation=way, asked=wanted.get(p.record_id), **kept)

    sizable = []                                                                    # in rank order
    for i in order:
        caps = plan[i][0]
        known = {name: n for name, n in caps.items() if n is not None}
        if _amount(purchases[i].price) is None or not known:
            why = "no price" if _amount(purchases[i].price) is None else "no limit of the user's could be worked out"
            finish(i, 0, caps, "no_room", [f"not sized: {why} (a size is never computed from a guess)"])
        elif min(known.values()) < 1:
            finish(i, 0, caps, "no_room", [f"not even one share fits the user's limits (shares allowed: {_listing(known)}; the tightest is {_tightest(known)})"])
        else:
            sizable.append(i)
    if allocation == "even":
        _split_evenly(purchases, sizable, plan, room, cash, budget, guard, finish)
    elif allocation == "round_robin":
        _round_robin(purchases, sizable, plan, room, cash, budget, guard, finish)
    else:
        used, left_budget = 0.0, None if budget is None else float(budget)                  # the exposure and the budget are shared in rank order; the cash is not reserved: every purchase sees all of it
        for i in sizable:
            caps = {**plan[i][0], **_pools(room, guard, float(cash), left_budget, used, float(purchases[i].price))}
            known = {name: n for name, n in caps.items() if n is not None}
            shares = min(known.values())
            if shares < 1:
                finish(i, 0, caps, "no_room", [f"not even one share fits (shares allowed: {_listing(known)}; the tightest is {_tightest(known)})"])
                continue
            finish(i, shares, caps, _tightest(known), [])
            cost = done[i].est_cost
            used, left_budget = used + cost, None if left_budget is None else left_budget - cost
    return [done[i] for i in range(len(purchases))]


def _split_evenly(purchases: list[Purchase], sizable: list[int], plan: dict, room: float | None, cash: float, budget: float | None, guard: float, finish) -> None:
    """The even allocation (`allocation="even"`, see the module text): the target is the smaller of the pools' equal share and the smallest dollar limit of any purchase still in the split; a purchase that cannot
    buy one share at the target is dropped and the rest are balanced again."""
    price = {i: float(purchases[i].price) for i in sizable}
    own = {i: min(n for n in plan[i][0].values() if n is not None) for i in sizable}             # the most shares each purchase's own limits allow
    dollars = {name: amount for name, amount in (("exposure", room), ("money", float(cash)), ("budget", None if budget is None else float(budget))) if amount is not None}
    pool_name = min(dollars, key=lambda name: (dollars[name], CAP_NAMES.index(name)))
    pool = dollars[pool_name]
    whole = {i: {**plan[i][0], **_pools(room, guard, float(cash), None if budget is None else float(budget), 0.0, price[i])} for i in sizable}   # the whole room, as if this purchase were alone
    active, target, share, smallest = list(sizable), 0.0, 0.0, 0.0
    while active:
        share = pool / len(active)
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
        own_known = {name: n for name, n in plan[i][0].items() if n is not None}
        binding = _tightest(own_known) if shares == own[i] else "even_split" if smallest < share else pool_name
        finish(i, shares, whole[i], binding, [f"even split: about ${target:,.2f} for each of {len(active)} purchases"], target_dollars=_dollars(target), of=len(active))


def _round_robin(purchases: list[Purchase], sizable: list[int], plan: dict, room: float | None, cash: float, budget: float | None, guard: float, finish) -> None:
    """The round-robin allocation (`allocation="round_robin"`, see the module text). `sizable` is in rank order. Money is counted in `Decimal` cents, so that the total is exactly the sum of `trade_amount(shares, price)`."""
    price = {i: float(purchases[i].price) for i in sizable}
    limits = {i: {name: n for name, n in plan[i][0].items() if n is not None} for i in sizable}               # each purchase's own limits (position, loss, order, liquidity), as shares
    pot = float(cash) if budget is None else min(float(cash), float(budget))
    pot_name = "budget" if budget is not None and float(budget) < float(cash) else "money"                     # what the pot is called when it closes a purchase
    whole = {i: {**plan[i][0], **_pools(room, guard, float(cash), None if budget is None else float(budget), 0.0, price[i])} for i in sizable}   # the whole room, as if this purchase were alone
    shares, cost, closed = dict.fromkeys(sizable, 0), {i: Decimal(0) for i in sizable}, {}   # shares taken, what they cost, and {purchase: what closed it}
    spent, rounds, open_ = Decimal(0), 0, list(sizable)
    while open_:
        took = False
        for i in list(open_):                                                                                  # one share of each purchase still open, in rank order
            n = shares[i]
            after = Decimal(str(trade_amount(n + 1, price[i])))
            step = max(after - cost[i], Decimal("0.01"))                                                       # this share's cost: the cost of n + 1 shares less the cost of n
            if n >= min(limits[i].values()):
                why = _tightest(limits[i])
            elif room is not None and _whole((room - float(spent)) / float(step), guard) < 1:
                why = "exposure"
            elif _whole((pot - float(spent)) / float(step), guard) < 1:
                why = pot_name
            else:
                shares[i], spent, cost[i], took = n + 1, spent + step, after, True
                continue
            closed[i] = why
            open_.remove(i)
        rounds += took
    for i in sizable:
        text = {"exposure": "the exposure room (max_new_exposure_pct of the equity, shared by the purchases) has no room for another share", "budget": "the budget has no money for another share",
                "money": "the cash has no money for another share"}.get(closed[i], f"it is at its {closed[i]} limit")
        header = f"round robin: {rounds} round{'s' if rounds != 1 else ''}, one share of each purchase in rank order, ${pot:,.2f} to spend ({pot_name}), ${pot - float(spent):,.2f} left"
        if shares[i]:
            finish(i, shares[i], whole[i], closed[i], [header, f"closed at {shares[i]} share{'s' if shares[i] != 1 else ''}: {text}"])
        else:
            finish(i, 0, whole[i], "no_room", [header, f"not even one share fits: {text}"])


def equity_of(portfolio: Portfolio, prices: dict[str, float]) -> tuple[float | None, str]:
    """(equity, what it was worked out from): the free cash plus every holding at its price in `prices`. A holding with no usable price is counted at its COST BASIS and the note says so (the Advisor's version had no equity
    at all then; here the size is cut by a limit on a number that is stated to be an approximation). None only when the account holds nothing (no cash and no holdings): there is no equity to take a share of."""
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
