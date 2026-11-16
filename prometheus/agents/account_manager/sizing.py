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

A cap is whole shares (the exact quotient, rounded down: money and shares are `decimal.Decimal`, design choices entry 112, so there is no float error to guard against and the settings' `float_guard` is gone). A purchase with no usable price is not sized: 0 shares, `binding` "no_room", and the note says why. A size is
never computed from a guess.

THE ALLOCATION says how the money is spread over the purchases. Two ways (`ALLOCATIONS`; left out, None, is the first, IN BOTH MODES: design choices entry 106, Jacob, 2026-11-15: *"No. its suppose to be round-robin. Not all in
one."*):

    round_robin          one share at a time, in rank order, until the funds are used up: the default whatever the mode (Jacob, 2026-11-14 for Auto; entry 106 for Ask)
    even                 the same DOLLARS for every purchase, as close as whole shares allow (the user said "split it evenly": the ONLY thing that chooses it, entry 109; neutral-only runs are round-robin too)

FRACTIONAL EVEN SPLIT (design choices entry 110; `fractional=True`, which the Account Manager passes when its switch `sizing.fractional_even_split` is on and the account book can hold a fraction: `AccountManager.decide_trades`).
The even split above is a split in WHOLE shares (Part G, entry 108); this is the one Jacob asked for (2026-11-15: *"It is possible to buy fractional shares. Example: $30000 split evenly is $10000 for stock x, $10000 for stock b,
$10000 for stock c and the remainder is again divided equally as close as possible"*): each purchase takes the number of shares, at six decimals (`shares.buys`), that its own dollars buy at its price, and the dollars are equal to the cent.
The pot (the budget, else the cash, never more than the cash) is counted in cents and divided by the number of purchases; the cents the division leaves go ONE CENT EACH to the first purchases in rank order ($30,000.02 over three
is $10,000.01, $10,000.01 and $10,000.00). Nobody is dropped for "cannot buy one share"; a purchase is `no_room` only when its amount buys nothing at all. A smaller ask holds the split down as it does for whole shares (the equal amount
of EVERY purchase is then the amount of the purchase asked for the fewest dollars, binding `even_split`, and what is left is not spent). For a share price under $10,000 one step costs under a cent, so each cost is its amount to
the cent; what is still left (a few cents at most, for a price above that) goes once round, to the purchase with the smallest amount so far and then the next, as the most it buys. Round robin is not changed: it is whole shares.

THE POT, ORDERS IN DOLLARS AND THE LEFTOVER OF A DIRECT ORDER (design choices entry 111; Jacob, 2026-11-15: *"It is impossible to spend more than the amount in the portfolio and budget"*, *"Yes, I want #2 built"*, *"When a direct order is
given, buy as many whole shares as possible and a fractional share for the leftover funds available"*; `design/dollar_orders_spec.md`). (M1) `pot_of(cash, budget)` is the POT of the call, the budget else the cash and never more than the cash, to the
cent; every `Sized` carries it (`pot`) and the Account Manager writes it on the decision (`size.pot`): the purchases of a run never cost more than it in all, whatever price they fill at. (M2) A purchase sized from money in a fractional split carries
`dollars`, its cost at the sizing price, the most it may cost at the fill (the decision's `shares` are then the estimate); a purchase held to the shares the user asked for (`binding: "asked"`) has none. (M3) With `fractional_leftover` (a DIRECT
order: `AccountManager._size_direct`) the round robin, which buys whole shares as ever, gives what it leaves over to the stock whose turn is next, the one with the fewest shares (a tie in rank order, not closed by its own ask), as the fraction
`shares.buys(its cost so far + the leftover, its price)`, and every purchase it sizes carries `dollars`; `wanted` may then be a quantity (what the user's dollars for EACH stock buy), and the round robin's last step for such a purchase is what is left of its
ask when that is less than one share. The recommendation path does not use `fractional_leftover`: its round robin stays whole shares and leaves its leftover as cash.

(A third way, `as_much_as_possible`, each purchase taking all it could in rank order, put the whole $10,000 of a real Ask-mode request into the first of four stocks and left three "no funds". It is gone: nothing chooses it, and
one purchase alone is the same under the round robin. A model's reply that still says it is read as "the user did not say how to spread it": `clarify.SizingChoice`.)

ROUND ROBIN ("buy a share of the first stock, a share of the next, and keep buying shares until all the funds requested are used up"). It is arithmetic done here, so that each stock ends with a FINAL number of shares: the
Trader places ONE trade per stock, never one trade per share. The POT is the funds requested: the `budget` when one was given, else the cash, and never more than the cash. It is drawn down as shares are bought; the cash
itself is not reserved. The purchases are visited in RANK order (`Purchase.rank`, lowest first, ties in the order given); each ROUND buys ONE share of each purchase that is still open, if the share's cost is within what is left
of the pot and the purchase has fewer shares than the user asked for. A purchase that cannot take another share is CLOSED for the rest of the call and remembers WHY (`asked`, `budget` or `money`). The call stops when no purchase is
open. A share costs `trade_amount(n + 1, price) - trade_amount(n, price)`, in cents the way the account book rounds them, so a purchase's cost is exactly `trade_amount(shares, price)` and the total is the sum of those.

EVEN, in two steps (design choices entry 108; Jacob, 2026-11-15: *"split evenly will try to split the holdings so the balance of each holding is as close as possible ... the remainder is again divided equally as close as
possible"*; whole shares now, fractional shares are a later change). STEP 1: every purchase gets the same DOLLARS: the pot divided by the number of purchases (and never more than the smallest amount any purchase was asked for),
whole shares at each price. A purchase whose even share buys less than one share is dropped (`no_room`, with the reason) and the others are balanced again; each purchase then carries the equal amount (`Sized.target_dollars`) and
how many shared it (`Sized.of`). STEP 2: what step 1 left of the pot is spent ONE SHARE AT A TIME: each turn, among the purchases still in the split that are under the shares the user asked for and whose next share costs no more
than the leftover, the one with the SMALLEST amount so far takes it (a tie: rank order), until none can. (Dividing the leftover by the number of purchases again and repeating would feed the cheapest purchase and make the split
LESS even, so it is not done.) Step 2 does not run when a smaller ask held the equal amount down (`even_split`: the rest is not the split's to spend), and a purchase dropped in step 1 stays dropped. Money is counted in `Decimal`
cents exactly as in the round robin (`_next_share`), so the total never passes the pot; the notes say how many shares came from the leftover and what is left.

THE AMOUNT THE USER ASKED FOR (`wanted`, design choices entry 85: the Account Manager's sizing questions in Ask mode). `wanted` is {record_id: whole shares}: how many shares the user's own words came to for that purchase (a
percentage of the account, a dollar amount or a number of shares, worked out by `clarify.plan_choice`, never by a model). The purchase never gets more than the user asked for; `binding` is "asked" when that was what stopped it.

`binding` says what decided the number of shares, one of: `money`, `budget` (that pool was the smallest: round robin: it closed the purchase; even: it set the equal amount), `asked` (the user asked for fewer shares than the money
allows), `even_split` (the equal amount was set by another purchase's smaller ask, so this one takes less than it could), `no_room` (not even one share fits, or the purchase could not be sized: 0 shares; the notes give the
reason), or None (never produced here: kept in the vocabulary for a decision that carries no size, a sale).

This module decides nothing about permission (that is `decisions.decide_trades`) and plans no sale (that is RaiseFunds): `funds_check` only says whether a sale of holdings in profit could pay for a purchase.

DECIMAL ALL THE WAY (design choices entry 112; Jacob, 2026-11-16: *"Need to always use decimal.Decimal type when dealing with money and share counts"*). The cash, the budget, the pot, every price and every amount here are `Decimal` (`Money`, `Price`),
every share count is a `Decimal` quantity (`shares.Shares`), a cost is `trade_amount` (the exact product, half up to the cent), the pot is rounded DOWN to the cent and a share count bought with money DOWN to six places (`shares.buys`): nothing here
is a float, and a float given to this module is a TypeError (`money.exact`), because it should have come in through a door. The sizing's own arithmetic has no tolerance: the cash after a run is the cash before less the sum of the fills, exactly.
"""
from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prometheus.agents.account_manager.config import AccountManagerConfig, load_config
from prometheus.common import shares as sh
from prometheus.common.money import CENT, NonNegativeMoney, Price, exact, Money
from prometheus.common.portfolio import Portfolio, trade_amount
from prometheus.tools import trading_math as tm

CAP_NAMES = ("money", "budget")                                    # the pools: the order is the order of a tie
BINDINGS = (*CAP_NAMES, "asked", "even_split", "no_room")          # what `Sized.binding` may be (None as well)
ALLOCATIONS = ("even", "round_robin")                              # left out (None) is "round_robin", in both modes (entry 106)
Binding = Literal["money", "budget", "asked", "even_split", "no_room"]


class Purchase(BaseModel):
    """One purchase to size, as plain numbers (the Account Manager makes it from the recommendation and the Critic's review, or from a company the Advisor scored neutral: it holds no company text and no score)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action: Literal["open", "add"]
    price: Price | None                              # the price to size at (the Critic's deal price, else the plan's entry mid), a Decimal at four places; None: not known, and nothing is sized
    rank: int = Field(default=0, ge=0)               # the order the recommendations were ranked in, lowest first


class Sized(BaseModel):
    """How one purchase was sized (see the module text). `est_cost` is `shares` at the purchase's price in cents; `caps` has every name of `CAP_NAMES` (the most whole shares each pool allows, None for a pool that is not
    there); `notes` says, in plain English, anything that shaped the size. `allocation` is the way the money was spread ("even" or "round_robin"); `target_dollars` and `of` are the equal amount of an even
    split (the larger one when a cent was shared out) and how many purchases shared it (None otherwise)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    symbol: str
    shares: sh.Shares                                 # a Decimal quantity: whole (round robin, a whole-share split) or with up to 6 decimals (a fractional even split)
    est_cost: NonNegativeMoney
    caps: dict[str, sh.Shares | None]
    binding: Binding | None
    notes: list[str] = Field(default_factory=list)
    allocation: Literal["even", "round_robin"] = "round_robin"
    target_dollars: Money | None = None
    of: int | None = None
    asked: sh.Shares | None = None                # the shares the user asked for (`wanted`), when they said how much; None otherwise. A fraction when the user asked for DOLLARS of each stock of a direct order (design choices entry 111, M3)
    dollars: Money | None = None                  # an ORDER IN DOLLARS (design choices entry 111, M2): the most this purchase may cost, to the cent = its cost at the sizing price (`est_cost`); `shares` is then the estimate. Only a purchase sized from money in a fractional split has one; None: an order in shares
    pot: NonNegativeMoney | None = None           # the POT of the call, in dollars to the cent: the budget when one was given, else the cash, never more than the cash. Every purchase sized in one call carries the same one; the run never spends more than it (design choices entry 111, M1)


def _amount(x, allow_zero: bool = False) -> Decimal | None:
    """`x` as a Decimal when it is a finite number above zero (at or above it with `allow_zero`), else None: a price or an equity that is not usable is not known. A float is a TypeError (`money.exact`): it comes in through a door."""
    if x is None:
        return None
    try:
        d = exact(x)
    except ValueError:
        return None
    return None if d < 0 or (d == 0 and not allow_zero) else d


def _whole(x: Decimal) -> Decimal:
    """Whole shares: rounded down, never below 0."""
    return max(Decimal(0), x.to_integral_value(rounding=ROUND_DOWN))


def _pools(cash: Decimal, budget: Decimal | None, price: Decimal, fractional: bool = False) -> dict[str, Decimal | None]:
    """The pools as shares at `price`: ALL the cash (it is not reserved: no purchase's cost is taken off it) and what is left of the budget (None when no budget was given). Whole shares, or with `fractional` what the
    dollars buy at six decimals (`shares.buys`)."""
    if fractional:
        return {"money": sh.buys(cash, price), "budget": None if budget is None else sh.buys(budget, price)}
    return {"money": _whole(cash // price), "budget": None if budget is None else _whole(budget // price)}


def pot_of(cash, budget=None) -> Decimal:
    """THE POT of a run (design choices entry 111, M1): the budget the user gave, else the cash, and never more than the cash, in dollars to the cent (rounded down: a fraction of a cent cannot be spent). The purchases of a run
    never cost more than this in all, whatever price they fill at."""
    cash = exact(cash, "the cash")
    pot = cash if budget is None else min(cash, exact(budget, "the budget"))
    return pot.quantize(CENT, rounding=ROUND_DOWN)


def size_purchases(purchases: list[Purchase], cash: Decimal, config: AccountManagerConfig | None = None, allocation: str | None = None, budget: Decimal | None = None, wanted: dict[str, Decimal] | None = None,
                   fractional: bool = False, fractional_leftover: bool = False) -> list[Sized]:
    """The shares for each purchase (see the module text), one `Sized` for each, in the order given. `cash` is what may be spent in all (the free cash, or more when the caller has the money raised), `config` `load_config()`
    when left out. `allocation` is "even" or "round_robin" (None is round_robin, whatever the mode); `budget` is the funds requested (the pot of the round robin and of the even split). Raises ValueError for what only a program
    gets wrong: an `allocation` that is not one of those, a `budget` that is not a number above zero, cash that is not a number from 0, two purchases with one `record_id`, or `wanted` with a record that is not a purchase or a
    count that is not a number of shares, zero or more (`shares.quantity`: a whole number, or one with at most six decimals: only the Account Manager's dollar answer for a direct order makes a fraction, entry 111, M3); and TypeError for a
    float anywhere (it comes in through a door, entry 112). `fractional` (design choices entry 110) makes the EVEN split a split in fractional shares, equal to the cent (see the module text); it changes nothing else: the round robin
    and `wanted` are whole shares, and with it left False the even split is the whole-share split of entry 108."""
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
    try:
        wanted = {record: sh.quantity(n) for record, n in wanted.items()}
    except ValueError:
        raise ValueError("wanted must be a number of shares (a whole number, or one with at most 6 decimals), zero or more, for each purchase") from None
    order = sorted(range(len(purchases)), key=lambda i: (purchases[i].rank, i))
    done: dict[int, Sized] = {}
    way = allocation or "round_robin"
    pot = pot_of(cash, budget)

    def finish(i: int, shares: Decimal, caps: dict, binding, more: list[str], **kept) -> None:
        p, price = purchases[i], _amount(purchases[i].price)
        done[i] = Sized(record_id=p.record_id, symbol=p.symbol, shares=shares, est_cost=trade_amount(shares, price) if shares and price else Decimal("0.00"),
                        caps={name: caps.get(name) for name in CAP_NAMES}, binding=binding, notes=list(more), allocation=way, asked=wanted.get(p.record_id), pot=pot, **kept)

    sizable = []                                                                    # in rank order
    for i in order:
        if _amount(purchases[i].price) is None:
            finish(i, Decimal(0), {}, "no_room", ["not sized: no price (a size is never computed from a guess)"])
        else:
            sizable.append(i)
    if allocation == "even" and fractional:
        _split_in_fractions(purchases, sizable, wanted, cash, budget, finish)
    elif allocation == "even":
        _split_evenly(purchases, sizable, wanted, cash, budget, finish)
    else:
        _round_robin(purchases, sizable, wanted, cash, budget, finish, fractional_leftover)
    return [done[i] for i in range(len(purchases))]


def _next_share(n: Decimal, price: Decimal, cost: Decimal) -> tuple[Decimal, Decimal]:
    """(the cost of n + 1 shares, the cost of the next share) for a purchase that has `n` shares costing `cost`: whole cents the way the account book rounds them (`trade_amount`), so that the cost of a purchase is exactly
    `trade_amount(shares, price)` and the total never passes the pot. The next share costs at least a cent."""
    after = trade_amount(n + 1, price)
    return after, max(after - cost, CENT)


def _split_evenly(purchases: list[Purchase], sizable: list[int], wanted: dict, cash: Decimal, budget: Decimal | None, finish) -> None:
    """The even allocation (`allocation="even"`, see the module text), in two steps. STEP 1: the target is the smaller of the pot's equal share and the smallest amount any purchase still in the split was asked for; a purchase that
    cannot buy one share at the target is dropped and the rest are balanced again; each takes the whole shares the target buys. STEP 2 (design choices entry 108): what is left of the pot is spent one share at a time, the purchase with the
    SMALLEST amount so far taking it (a tie: rank order), among those that are under the shares the user asked for and whose next share fits the leftover; it does not run when a smaller ask held the target down, and a dropped purchase
    stays dropped. `sizable` is in rank order."""
    price = {i: purchases[i].price for i in sizable}
    ask = {i: wanted.get(purchases[i].record_id) for i in sizable}                                 # the most shares each purchase was asked for (None: no ask, the pot bounds it)
    pot = cash if budget is None else min(cash, budget)
    pool_name = "budget" if budget is not None and budget < cash else "money"
    whole = {i: _pools(cash, budget, price[i]) for i in sizable}
    active, target, share, smallest = list(sizable), Decimal(0), Decimal(0), None
    while active:
        share = pot / len(active)
        asked = [ask[i] * price[i] for i in active if ask[i] is not None]
        smallest = min(asked) if asked else None
        target = share if smallest is None else min(share, smallest)
        short = [i for i in active if _whole(target // price[i]) < 1]
        if not short:
            break
        for i in short:
            finish(i, Decimal(0), whole[i], "no_room", [f"an even share of ${target:,.2f} does not buy one share at ${price[i]:,.2f}"], target_dollars=trade_amount(1, target), of=len(active))
        active = [i for i in active if i not in short]
    shares = {i: _whole(target // price[i]) if ask[i] is None else min(ask[i], _whole(target // price[i])) for i in active}          # step 1: the whole shares the equal amount buys
    first = dict(shares)
    cost = {i: trade_amount(shares[i], price[i]) if shares[i] else Decimal("0.00") for i in active}
    left = max(Decimal(0), pot - sum(cost.values(), Decimal("0.00")))
    held_down = smallest is not None and smallest < share                                         # a smaller ask set the equal amount: the rest is not the split's to spend
    while active and not held_down:                                                                # step 2: one share to the purchase furthest behind that can still take one
        fits = [i for i in active if (ask[i] is None or shares[i] < ask[i]) and _next_share(shares[i], price[i], cost[i])[1] <= left]
        if not fits:
            break
        i = min(fits, key=lambda j: cost[j])                                                       # `fits` is in rank order, and min keeps the first of a tie
        after, step = _next_share(shares[i], price[i], cost[i])
        shares[i], cost[i], left = shares[i] + 1, after, left - step
    for i in active:
        bought = shares[i] - first[i]
        binding = "asked" if ask[i] is not None and shares[i] == ask[i] else "even_split" if held_down else pool_name
        notes = [f"even split: about ${target:,.2f} for each of {len(active)} purchases"]
        if held_down:
            notes.append("another purchase's smaller ask set that amount, so what is left of the money is not spent")
        else:
            if bought:
                notes.append(f"{sh.text(bought)} more share{'s' if bought != 1 else ''} from what the equal amounts left over, one at a time to the smallest amount")
            notes.append(f"${left:,.2f} of the ${pot:,.2f} is left over: " + ("no purchase can take another share" if left else "all of it is spent"))
        finish(i, shares[i], whole[i], binding, notes, target_dollars=trade_amount(1, target), of=len(active))


def _split_in_fractions(purchases: list[Purchase], sizable: list[int], wanted: dict, cash: Decimal, budget: Decimal | None, finish) -> None:
    """The fractional even split (`allocation="even"` with `fractional=True`, see the module text). `sizable` is in rank order. Money is counted in `Decimal` cents; a purchase takes `shares.buys(its amount, its price)`, and the
    cost of what it takes is `trade_amount`, the account book's own rounding, so the total never passes the pot."""
    price = {i: purchases[i].price for i in sizable}
    ask = {i: wanted.get(purchases[i].record_id) for i in sizable}
    pot = cash if budget is None else min(cash, budget)
    pool_name = "budget" if budget is not None and budget < cash else "money"
    caps = {i: _pools(cash, budget, price[i], fractional=True) for i in sizable}
    cents = int(pot.quantize(CENT, rounding=ROUND_DOWN) / CENT)                                                     # the pot in whole cents
    base, extra = divmod(cents, len(sizable))
    amount = {i: Decimal(base + (1 if place < extra else 0)) * CENT for place, i in enumerate(sizable)}       # the cents the division left go one each to the first purchases, in rank order
    cost_of = lambda i, n: trade_amount(n, price[i])
    held_down = [i for i in sizable if ask[i] is not None and ask[i] < sh.buys(amount[i], price[i])]      # asked for fewer shares than its amount buys: it sets the amount of every purchase
    if held_down:
        smallest = min(cost_of(i, ask[i]) for i in held_down)
        amount = dict.fromkeys(sizable, smallest)
    shares = {i: sh.buys(amount[i], price[i]) for i in sizable}
    for i in sizable:
        if ask[i] is not None and ask[i] < shares[i]:
            shares[i] = ask[i]
    cost = {i: cost_of(i, shares[i]) if shares[i] else Decimal("0.00") for i in sizable}
    left = Decimal(cents) * CENT - sum(cost.values(), Decimal("0.00"))
    if not held_down:                                                                                        # what is still left: once round, to the smallest amount so far, then the next
        for i in sorted(sizable, key=lambda j: (cost[j], sizable.index(j))):
            if left <= 0:
                break
            more = sh.buys(cost[i] + left, price[i])
            if ask[i] is not None:
                more = min(more, ask[i])
            if more > shares[i]:
                after = cost_of(i, more)
                shares[i], cost[i], left = more, after, left - (after - cost[i])
    target = max(amount.values())
    for i in sizable:
        if shares[i] == 0:
            finish(i, Decimal(0), caps[i], "no_room", [f"an even share of ${amount[i]:,.2f} buys nothing at ${price[i]:,.2f}"], target_dollars=target, of=len(sizable))
            continue
        binding = "asked" if ask[i] is not None and shares[i] == ask[i] else "even_split" if held_down else pool_name
        order_dollars = None if binding == "asked" else cost[i]      # an order in dollars (M2): what the sizing gave the purchase, the most it may cost at the fill; a purchase held to the shares the user asked for is an order in shares
        notes = [f"even split in fractional shares: ${target:,.2f} for each of {len(sizable)} purchases (the cents shared out one at a time in rank order)"]
        if held_down:
            notes.append("another purchase's smaller ask set that amount, so what is left of the money is not spent")
        else:
            notes.append(f"${left:,.2f} of the ${Decimal(cents) * CENT:,.2f} is left over: " + ("less than one step of any purchase can still be bought" if left else "all of it is spent"))
        finish(i, shares[i], caps[i], binding, notes, target_dollars=target, of=len(sizable), dollars=order_dollars)


def _round_robin(purchases: list[Purchase], sizable: list[int], wanted: dict, cash: Decimal, budget: Decimal | None, finish, fractional_leftover: bool = False) -> None:
    """The round-robin allocation (`allocation="round_robin"`, see the module text). `sizable` is in rank order. Money is counted in `Decimal` cents, so that the total is exactly the sum of `trade_amount` of the shares."""
    price = {i: purchases[i].price for i in sizable}
    ask = {i: wanted.get(purchases[i].record_id) for i in sizable}
    pot = cash if budget is None else min(cash, budget)
    pot_name = "budget" if budget is not None and budget < cash else "money"                                 # what the pot is called when it closes a purchase
    whole = {i: _pools(cash, budget, price[i]) for i in sizable}
    shares, cost, closed = dict.fromkeys(sizable, Decimal(0)), {i: Decimal("0.00") for i in sizable}, {}   # shares taken, what they cost, and {purchase: what closed it}
    spent, rounds, open_ = Decimal("0.00"), 0, list(sizable)
    while open_:
        took = False
        for i in list(open_):                                                                                  # one share of each purchase still open, in rank order
            n = shares[i]
            after, step = _next_share(n, price[i], cost[i])                                                    # this share's cost: the cost of n + 1 shares less the cost of n
            if ask[i] is not None and n >= ask[i]:
                why = "asked"
            elif ask[i] is not None and ask[i] - n < 1:                                                        # the LAST step of an ask in dollars that is not a whole number of shares: what is left of it, a fraction (entry 111, M3)
                whole_ask = trade_amount(ask[i], price[i])
                if whole_ask - cost[i] > pot - spent:
                    why = pot_name
                else:
                    shares[i], spent, cost[i], took = ask[i], spent + (whole_ask - cost[i]), whole_ask, True
                    why = "asked"
            elif _whole((pot - spent) // step) < 1:
                why = pot_name
            else:
                shares[i], spent, cost[i], took = n + 1, spent + step, after, True
                continue
            closed[i] = why
            open_.remove(i)
        rounds += took
    bought_fraction, leftover = None, Decimal("0.00")
    if fractional_leftover:                                                                                    # the leftover buys a FRACTION of a share of the stock whose turn is next (entry 111, M3: a direct order with money)
        left = leftover = pot_of(cash, budget) - spent
        turn = [i for i in sizable if closed[i] != "asked"]                                                    # not closed by its own ask
        if left >= CENT and turn:
            i = min(turn, key=lambda j: (shares[j], sizable.index(j)))                                         # the one with the fewest shares, a tie in rank order
            more = sh.buys(cost[i] + left, price[i])
            after = trade_amount(more, price[i]) if more else cost[i]
            if more > shares[i] and after - cost[i] <= left:
                shares[i], spent, cost[i], bought_fraction = more, spent + (after - cost[i]), after, i
    for i in sizable:
        text = {"budget": "the budget has no money for another share", "money": "the cash has no money for another share", "asked": "it has the shares the user asked for"}[closed[i]]
        header = f"round robin: {rounds} round{'s' if rounds != 1 else ''}, one share of each purchase in rank order, ${pot:,.2f} to spend ({pot_name}), ${pot - spent:,.2f} left"
        order_dollars = cost[i] if fractional_leftover and shares[i] else None                                  # an ORDER IN DOLLARS (M3): what the sizing gave the purchase, the most it may cost at the fill
        if shares[i]:
            more = [f"closed at {sh.text(shares[i])} share{'s' if shares[i] != 1 else ''}: {text}"]
            if bought_fraction == i:
                more.append(f"the ${leftover:,.2f} the whole shares left over bought {sh.text(shares[i])} shares in all, a fraction of a share: ${cost[i]:,.2f}")
            finish(i, shares[i], whole[i], closed[i], [header, *more], dollars=order_dollars)
        else:
            finish(i, Decimal(0), whole[i], "no_room", [header, f"not even one share fits: {text}"])


def equity_of(portfolio: Portfolio, prices: dict[str, Decimal]) -> tuple[Decimal | None, str]:
    """(equity, what it was worked out from): the free cash plus every holding at its price in `prices`. A holding with no usable price is counted at its COST BASIS and the note says so. None only when the account holds nothing
    (no cash and no holdings): there is no equity to take a share of (the Account Manager's "20% of my account" needs one)."""
    value, unpriced = portfolio.cash_avail, []
    for position in portfolio.positions:
        price = _amount(prices.get(position.symbol))
        if price is None:
            unpriced.append(position.symbol)
            price = position.cost_basis
        value += trade_amount(position.shares, price)      # a holding's value is `trade_amount`: decimal, half up to the cent (entries 111 and 112)
    if not portfolio.positions:
        basis = "cash only (no holdings)"
    else:
        basis = "cash and holdings at the prices given" + (f"; {', '.join(unpriced)} at cost basis (no price)" if unpriced else "")
    return (value if value > 0 else None), (basis if value > 0 else "the account holds no cash and no holdings")


def funds_check(cost: Decimal, portfolio: Portfolio, prices: dict[str, Decimal], today: date) -> tuple[bool, str]:
    """Whether the money for `cost` exists: free cash, or free cash plus what could be sold from the holdings that are in profit and sellable today (RaiseFunds' default rules: only holdings in the green, and one share of
    every holding is kept). A FLAG for the Account Manager: it does not run the tree of thoughts and plans no sale. A holding with no price in `prices` cannot be known to be in profit and is not counted (the note names it).
    Ported unchanged from `advisor/trade_plan.funds_check`, apart from that last sentence of the note."""
    if portfolio.cash_avail >= cost:
        return True, f"cash ${portfolio.cash_avail:,.2f} covers ${cost:,.2f}"
    holdings = [{"symbol": p.symbol, "shares": _kept_one(portfolio.sellable_shares(p.symbol, today)), "price": prices[p.symbol], "basis": p.cost_basis} for p in portfolio.positions if p.symbol in prices]
    raisable = sum((trade_amount(h["shares"], h["price"]) for h in tm.rank_green(holdings, only_green=True)), Decimal("0.00"))
    shortfall = cost - portfolio.cash_avail
    unpriced = [p.symbol for p in portfolio.positions if p.symbol not in prices]
    return raisable >= shortfall, (f"short by ${shortfall:,.2f}; ${raisable:,.2f} could be raised from holdings in profit that are sellable today (one share of each kept)"
                                   + (f"; no price for {', '.join(unpriced)}, not counted" if unpriced else ""))


def _kept_one(sellable: Decimal) -> Decimal:
    """The shares that may be sold when one share of the holding is kept: none when it holds one or less."""
    return sh.minus(sellable, 1) if sellable >= 1 else Decimal(0)
