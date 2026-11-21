"""A DIRECT ORDER: the user names the trade, so nothing is recommended or reviewed (design choices entries 87, 88, 101 and 102).

    named   = account_manager.name_orders(order, intent.orders, typed, now, approver, mode)   # the Account Manager ASKS the Data Clerk (`request_symbols`: a `lookup_symbol` order) for each stock's company name and symbol, decides what to
                                                                                      # do with a stock that was not found (Auto: drop it; Ask: ask, `missing.py`) and, through `name_the_stocks` below, APPENDS
                                                                                      # the rest to the work order and writes the user's own orders in `order.direct_orders`
    order   = decide_direct(order, portfolio, profile, mode, prices, now, valid_minutes, approver)       # status "new" -> "decided": one trade decision for each of the user's orders

"Buy 50 shares of MSFT" is not a request for advice. The Advisor would only recommend what the user already chose, and the Critic would review a recommendation nobody made, so the desk skips both, and it skips the Data
Clerk's node too (it has no research to do). The work order goes from the Account Manager's reading of the request to the Account Manager's decision and on to the Trader:

    account_manager (reads the request, asks the Data Clerk, appends) -> account_manager_direct (this module: prices, decides) -> trader

THE COMPANY NAME AND THE SYMBOL ARE ON THE WORK ORDER (Jacob, 2026-11-15). A request for research goes through the Data Clerk's node, which receives the work order and appends what it finds; a direct order skips that node, so
the ACCOUNT MANAGER asks the Data Clerk (`DataClerk.request_symbols`: a ticker is looked up in the symbol directory for its company, a company's name for its ticker, by exact matching, never a guess) and appends the answer to
the work order itself: `order.data.companies` (what the Data Clerk found, one row for each stock the directory answered for) and the user's own record in `order.direct_orders` (the symbol, the company's name, and what the user typed
when it was a name). The user names a company ("Buy 50 shares of Microsoft"): the Account Manager goes on with the symbol the Data Clerk returned, and in ASK mode the user is asked to confirm it first, with the symbol, the
company's name, the shares and the amount; in AUTO mode the returned symbol is used without asking. A ticker the user typed is not asked about again (the symbol is theirs). A BRAND ("Google") that the directory does not list
but whose listed owner the Data Clerk's model named and the directory has (Alphabet Inc., GOOGL: entry 101) is a company the user named, and the confirmation and the report say that the symbol is the listed owner of what they typed.

A STOCK THE DATA CLERK DID NOT FIND (entry 102). After the lookup every stock is found (a row with a symbol), NOT LISTED (a row with no symbol: not found, privately held, or fitting several listings: `sort_names`) or NOT ANSWERED
(no row: the lookup failed, which is never "not found"). A BLOCKING issue stops the run in either mode and its last message asks the user to clarify the request (`CLARIFY`): nothing found, a lookup that failed, two orders for one
stock. When some are found and some are not listed, AUTO mode drops the missing ones without a question and tells the user (`AccountManager.name_orders`), and ASK mode asks about each one (`missing.py`: the model words the
question, code checks it and reads the yes or no); a stock that was dropped is written on the record (`order.data.notes`, and the desk's `plan["dropped"]`) and is no longer on `order.direct_orders` or `order.subject`.
`name_the_stocks` refuses to go on while a missing stock is not in its `dropped`, so nothing is ever dropped silently.

AUTO-APPROVED. The user's own order is the approval: the decision says `yes_user` (the user said yes by naming the trade; the Account Manager does not ask a second time, in Ask mode too, unless the symbol is the Data Clerk's:
above) with `origin: "direct"`. `work_order.verify_trade` accepts such a decision only while `order.direct_orders` holds the user's matching record (the same ticker and side, and no more shares than the user typed), so the
word `direct` alone permits nothing, and a decision that is not direct still needs the Critic's approval. The Trader and the handshake are the same code as for a recommendation: one ticket, `approve_trade`, the fill, `record_fill`.

THERE IS NO PROFILE LIMIT (Jacob, 2026-11-15: "profile limits never apply to any orders", then "I do not want any of these limits ... remove them all", then "the only sizing that we need is to make sure the trade order has enough funds
to cover it": design choices entry 91). The profile has no cap on the share of the account one stock may be, on the loss a stop would cost, on the exposure of one review or on one order, and the sizing has no liquidity cap
either: nothing here sizes anything and nothing here imports `sizing` (a test reads this module's source and the Account Manager's `decide_direct` for it). The size is the user's. What stops a direct order is the account itself:
the cash for a purchase, the shares held for a sale (and the T+1 rule, which the Trader's handshake keeps). A trade is placed AS TYPED OR NOT AT ALL: when the cash does not pay for all the shares the order is `no_funds` and
says how many shares the cash does pay for, it is never made smaller on the user's behalf, and the handshake refuses, and does not cut, an order the account can no longer pay for in full. No holding is sold to fund a direct
purchase UNLESS the user permitted it (entry 142: "raise funds if needed" in their words, the setting ALWAYS_RAISE_FUNDS, or a yes in Ask mode): `funding.py`, and the purchase is then still placed as typed.

A DIRECT ORDER WITH MONEY IS AN ORDER IN DOLLARS (design choices entry 111, M2 and M3; Jacob: *"When a direct order is given, buy as many whole shares as possible and a fractional share for the leftover funds available."*). A stock the user named with dollars
("Buy $5,000 of MSFT"), with no number, or with an answer in dollars or a percentage in Ask mode is sized by `AccountManager._size_direct`: the money is spread round-robin in whole shares, what it leaves buys a fraction of a share of the stock with the fewest shares,
and the decision carries `dollars` (the most it may cost: $5,000.00 for MSFT: 9 whole shares and 0.54417 of a share), the POT of the run (`size.pot`, the budget or the cash) and `shares` as the estimate at the price it was sized at; at the fill the Account Manager turns the dollars
into shares at the fill price, so the purchase costs its dollars to the cent. Such an order is not "as typed": the user typed no number of shares. A direct order whose shares the user typed, an answer in shares, and every sale stay orders in shares, placed as typed
or not at all. With fractions off (the switch, or an account book that cannot hold a fraction) the money is spent in whole shares as before, and no decision carries dollars.

THE DOLLARS THE USER WROTE FOR A STOCK ARE THAT STOCK'S LIMIT (design choices entry 122; Jacob: *"It is impossible to spend more than the amount in the portfolio and budget"*). "Buy $3,000 of XOM and $3,000 of MSFT" is two orders with
`dollars` (the intent form has one for each order, `intent.DirectOrder.dollars`, checked as money the user wrote), the user's record on `order.direct_orders` holds it (`dollars`, and `how_many: "dollars"` once sized), and
`AccountManager._size_direct` sizes each as an amount for EACH stock: what its dollars buy (a fraction with an account book that holds one, else whole shares), as an order in dollars, with the POT of the run the sum of the dollars.
A purchase with no number in a request where another purchase has one ("Buy $2,000 of AAPL and MSFT") is not given the cash: Ask mode asks about it, Auto mode does not place it and says "say how much of MSFT" (`how: "unsaid"`).
A request that wrote an amount of money and whose reading has none is refused before this module is reached (`intent.check_amount_was_read`), never turned into all the cash. `work_order.verify_trade` refuses a decision whose dollars
(or estimated cost) are above the user's own dollars for the stock.

A PRICE LIMIT ("Buy 10 of MU under $50 a share", design choices entry 138, L5; entry 140, W1): the user's record carries `max_price` (a CEILING: "or less") and `min_price` (a FLOOR: "or more"). A purchase with a ceiling is placed at once while the price is at
or under it; ABOVE it the order WAITS as a DAY order with its cash set aside (`orders.py`: `_decide_waiting`; it is `no_system_denied` for now, nothing trades, and the Account Manager places the open order, `AccountManager.decide_direct`). A SALE with a floor waits the
same way while the price is UNDER it, with its shares set aside. A purchase with a floor is placed only while the price is at or over it, else it is not placed and told (F3: a buy that waits for the price to RISE is a stop order, not built). A waiting order needs an
amount to set aside (shares typed, or dollars): with none it is not placed, and a waiting order that the cash or the shares cannot set aside is `no_funds`. It takes no part of the money the sizing spends. The handshake refuses a fill past the limit too
(`decide_approval`: the price moves between the decision and the fill). A sale is never limited by a ceiling, nor a purchase by a floor on the way to waiting.

A DIRECT ORDER IS A MARKET ORDER while its limit is met or it has none (the Trader places it at the current quote: there is no slippage, entry 111) and a LIMIT ORDER that waits when the price has not reached the limit (entry 140, `orders.py`: a day order on the paper
broker, watched by the web server's watcher, filled by the same Trader funnel). TODO (Jacob): a STOP order (buy when the price RISES to a level, sell when it FALLS) is not built.

DECIMAL MONEY (design choices entry 112): the prices (Decimals at four places: the Account Manager turns a quote into one when it asks the Data Clerk), the amounts, the dollars and the shares of a decision are `Decimal`. The user's own record on
`order.direct_orders` and the decision are plain dicts: a Decimal after the code that wrote them, an int or a float after `json` read a saved order, so every reader here takes the shares through `shares.quantity` and a price through
`money.price` before any arithmetic.

Nothing here calls a model or an API: the prices and the answers of the Data Clerk are given.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from prometheus.agents.account_manager import orders
from prometheus.agents.critic.common import utc_text
from prometheus.common import shares as sh
from prometheus.common.money import digits, exact, price as price_door
from prometheus.common.portfolio import Portfolio, affordable_shares, trade_amount
from prometheus.common.trace import traced
from prometheus.common.user_profile import UserProfile
from prometheus.common.work_order import DIRECT, DataResult, WorkOrder, verify_direct_orders

MODES = ("ask", "auto")


def record_id_of(request_id: str, number: int) -> str:
    """The id of the user's n-th order of a request: "direct-<request id>-1". Short, and the same on every record that belongs to it (the order, the decision, the trade and the account book's row)."""
    return f"direct-{request_id}-{number}"


CLARIFY = "Please clarify your request and send it again: name each stock by its ticker or by the listed company's name."   # the last message of a run that stopped because the stocks could not be made (entry 102)


@dataclass
class Sorted:
    """What the Data Clerk's answer says about each stock the user named, in the order they typed them: `listed` and `missing` are [(the user's order, the Data Clerk's row)] (a row with a symbol, and a row with none), `unanswered`
    the user's orders that have no row at all (the lookup failed)."""
    listed: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    unanswered: list = field(default_factory=list)


@dataclass
class Naming:
    """What naming the stocks came to: `problems` (why the run stops, in the Data Clerk's words and the Account Manager's: empty when the work order now holds every stock that stays), `orders` the user's orders that stay and
    `dropped` the ones that were left out ({typed, reason, mode}: "ask" at the user's yes, "auto" in Auto mode)."""
    problems: list[str] = field(default_factory=list)
    orders: list = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)


def sort_names(orders, found: WorkOrder) -> Sorted:
    """Sort the user's orders by what the Data Clerk's `lookup_symbol` order says (see the module text). The row of a stock is the one whose `brand` is what the user typed."""
    rows = {row["brand"]: row for row in (found.data.companies if found.data else [])}
    sort = Sorted()
    for wanted in orders:
        row = rows.get(" ".join(wanted.named.split()))
        (sort.unanswered if row is None else sort.listed if row.get("symbol") else sort.missing).append(wanted if row is None else (wanted, row))
    return sort


def data_clerk_lines(order: WorkOrder, found: str, not_found: str) -> list[str]:
    """One line for each stock the Data Clerk answered for, for the session and the page: `found: you typed 'MSFT': Microsoft Corp (MSFT), appended to the work order`, a brand says that the symbol is its listed owner's, and a
    stock that is not listed says why (`not_found: you typed 'Chick-fil-a': privately held, or not known to be listed`)."""
    lines = []
    for row in (order.data.companies if order.data else []):
        if row.get("symbol"):
            lines.append(f"{found}: you typed {row['brand']!r}: {row['company']} ({row['symbol']})" + (", the listed owner of what you typed" if row.get("how") == "brand" else "") + ", appended to the work order")
        else:
            lines.append(f"{not_found}: you typed {row['brand']!r}: {row.get('note') or 'not listed'}")
    return lines


def dropped_note(dropped: dict) -> str:
    """The line of `order.data.notes` for a stock that was left out of the order."""
    how = "in Auto mode" if dropped["mode"] == "auto" else "at the user's word"
    return f"dropped {how}: {dropped['typed']} ({dropped['reason']})"


def name_the_stocks(order: WorkOrder, orders, typed: str, now: datetime, found: WorkOrder, dropped: list[dict] | None = None) -> list[str]:
    """The Account Manager APPENDS the Data Clerk's answer to the work order (see the module text). `found` is the `lookup_symbol` order the Data Clerk answered, for the stocks of `orders` (the user's, as typed); `typed`
    the user's own words; `dropped` the stocks that were left out ({typed, reason, mode}). On success the work order gets `data` (the Data Clerk's rows in `data.companies`, its calls and its answer, and a note for each stock
    that was dropped), `subject` (the symbols) and `direct_orders` (one record for each of the user's trades that stays, in the order they typed them: the id, the side, the symbol, the company's name, the shares, what was typed
    when it was a company's name, the user's words and the moment), and the list returned is empty. Otherwise nothing of the order is changed and the list says why, in the Data Clerk's own words (a name it did not find, one that
    fits several listings, a lookup that failed): a stock that is not listed stops it unless it is in `dropped`, a lookup that failed always does, and so do two orders for the same stock."""
    if found.data is None:
        return [*found.errors] or ["the Data Clerk could not find the symbols of what you named"]
    sort = sort_names(orders, found)
    left_out = {" ".join(d["typed"].split()) for d in dropped or []}
    if sort.unanswered or not sort.listed or [w for w, _ in sort.missing if " ".join(w.named.split()) not in left_out]:
        return [*found.errors] or ["the Data Clerk found no symbol and company name for " + ", ".join(w.named for w in [*sort.unanswered, *[w for w, _ in sort.missing]])]
    entries, problems = [], []
    for number, (wanted, row) in enumerate(sort.listed, 1):
        entries.append({"record_id": record_id_of(order.request_id, number), "side": wanted.side, "symbol": row["symbol"], "company": row["company"], "shares": wanted.shares, "request": typed,
                        "requested_utc": utc_text(now), "typed_company": wanted.company, "how_many": "typed" if wanted.shares is not None else None, "found_by": row.get("how"),
                        **({"dollars": wanted.dollars} if getattr(wanted, "dollars", None) is not None else {}),      # the user's own dollars for the stock (entry 122)
                        **({"max_price": wanted.max_price} if getattr(wanted, "max_price", None) is not None else {}),      # the user's price limit (entry 138)
                        **({"min_price": wanted.min_price} if getattr(wanted, "min_price", None) is not None else {})})      # and the user's price floor (entry 140)
    if len({e["symbol"] for e in entries}) != len(entries):
        problems.append("two of your orders are for the same stock (a ticker and the company's name, for example): name each stock once")
    problems += verify_direct_orders(entries) if not problems else []
    if problems:
        return problems
    data = order.data or DataResult()
    data.companies = [*data.companies, *found.data.companies]
    data.tool_calls, data.problems = [*data.tool_calls, *found.data.tool_calls], [*data.problems, *found.data.problems]
    data.notes = [*data.notes, *[dropped_note(d) for d in dropped or []]]
    data.tokens_in, data.tokens_out = data.tokens_in + found.data.tokens_in, data.tokens_out + found.data.tokens_out
    data.answer, data.completed_utc = found.data.answer, found.data.completed_utc
    order.data, order.subject, order.direct_orders = data, list(dict.fromkeys(e["symbol"] for e in entries)), entries
    return []


def over_limit(entry: dict, price) -> Decimal | None:
    """The user's price limit of this order (`entry["max_price"]`, entry 138) when `price` is ABOVE it and the order is a purchase, else None: no limit, a sale (a sale is never limited by it), no usable price, or a price at or under the limit."""
    if entry.get("side") != "buy" or entry.get("max_price") is None or not _usable(price):
        return None
    limit = price_door(entry["max_price"])      # a Decimal, or the int or float that `json` gave for a saved order
    return limit if exact(price, "a price") > limit else None


def under_floor(entry: dict, price) -> Decimal | None:
    """The user's price floor of this order (`entry["min_price"]`, entry 140) when `price` is UNDER it and the order is a PURCHASE, else None: a purchase with a floor is placed only while the price is at or over it (F3). A sale with a floor is not
    refused by it: it waits (`orders.waits`)."""
    if entry.get("side") != "buy" or entry.get("min_price") is None or not _usable(price):
        return None
    floor = price_door(entry["min_price"])
    return floor if exact(price, "a price") < floor else None


def not_now(entry: dict, price) -> bool:
    """Whether this order, at this price, is not placed now: a purchase above its ceiling or under its floor, and a sale under its floor (the first and the last WAIT, `orders.waits`; the middle one is not placed). The sizing of the user's money leaves such an order out."""
    return over_limit(entry, price) is not None or under_floor(entry, price) is not None or orders.waits(entry, price)


def _decision(entry: dict, mode: str, price, now: datetime, valid_minutes: int) -> dict:
    """The decision of one order before it is decided: every key `decisions.decide_trades` writes, and the four of a direct order: `origin`, the `company` the Data Clerk found, the `price` it was priced at and its own
    `valid_until_utc`; and the user's price limit when they gave one (`max_price`, entry 138; `min_price`, entry 140)."""
    return {"record_id": entry["record_id"], "review_id": None, "symbol": entry["symbol"], "action": "reduce" if entry["side"] == "sell" else "open", "shares": sh.quantity(entry["shares"]) if entry["shares"] is not None else Decimal(0), "est_cost": Decimal("0.00"), "mode": mode,
            "will_trade": None, "funds_avail": None, "funds_note": None, "asked_user": False, "reason": "", "decided_utc": utc_text(now), "size": None,
            "origin": DIRECT, "company": entry["company"], "price": price, "valid_until_utc": utc_text(now + timedelta(minutes=valid_minutes)),
            **({"max_price": price_door(entry["max_price"])} if entry.get("max_price") is not None and entry.get("side") == "buy" else {}),
            **({"min_price": price_door(entry["min_price"])} if entry.get("min_price") is not None else {})}      # the user's price floor (entry 140): a purchase or a sale is only filled at or over it


def _decide_under_floor(entry: dict, mode: str, price, floor: Decimal, now: datetime, valid_minutes: int) -> dict:
    """A purchase priced under the user's floor (entry 140, F3: "buy MU if it is over $20 a share"): not placed, with the price and the floor in the reason. A purchase that waits for the price to RISE is a stop order, which is not built."""
    decision = _decision(entry, mode, price, now, valid_minutes)
    decision.update(will_trade="no_system_denied", funds_avail=False, shares=Decimal(0), est_cost=Decimal("0.00"),
                    reason=f"{entry['symbol']} is at ${exact(price, 'a price'):,.2f} a share, under your floor of ${floor:,.2f}: not placed. A purchase that waits for the price to rise is a stop order, which is not built, so ask again when the price is at or over your floor")
    return decision


def _decide_waiting(entry: dict, portfolio: Portfolio, mode: str, price, now: datetime, valid_minutes: int, fractions: bool, funded_by: str | None = None, why_not: str | None = None) -> dict:
    """An order whose limit the price has not reached (entry 140, W1): a purchase above its ceiling, a sale under its floor. It WAITS as a day order with what it needs SET ASIDE (`orders.plan_waiting`), or it is refused with the reason
    (`no_funds`: the cash or the shares are not there to set aside; `no_system_denied`: no amount to set aside). The decision still says `no_system_denied` while it waits (nothing trades now) and carries `waiting`, the plan: the Account
    Manager places the open order in the account book (`AccountManager.decide_direct`) and writes its number, `open_order_id`, on the decision. `decide_direct` is a pure function: it places nothing itself. `funded_by` (entry 143: the Account
    Manager's decision, from the user's permission) lets a purchase the free cash does not cover wait anyway, to be paid for by selling holdings when its price is reached; `why_not` ends the reason of one that is refused for want of cash
    (what the user can say to let it wait, or why raising funds could not pay for it)."""
    decision = _decision(entry, mode, price, now, valid_minutes)
    plan = orders.plan_waiting(entry, price, portfolio, now, fractions, funded_by)
    if isinstance(plan, orders.Refusal):
        decision.update(will_trade=plan.will_trade, funds_avail=False, shares=Decimal(0), est_cost=Decimal("0.00"), reason=plan.reason + (f"; {why_not}" if why_not and plan.will_trade == "no_funds" else ""))
        return decision
    decision.update(will_trade="no_system_denied", funds_avail=True, shares=plan.estimate, est_cost=plan.reserve_cash, reason=orders.waiting_sentence(plan, now), waiting=plan.as_dict())
    return decision


HOW_FUNDED = {"words": "you said to raise funds", "setting": "the setting ALWAYS_RAISE_FUNDS is on", "asked": "you said yes when asked"}      # how the user permitted RaiseFunds (entry 142; `funding.decision_funding`)
HOW_WORDS = {"asked": "you gave when the Account Manager asked how many", "cash": "the Account Manager worked out: as much as your cash allows", "budget": "the Account Manager worked out from the dollars you gave",
             "dollars": "the Account Manager worked out from the dollars you wrote for this stock"}
SIZED = ("asked", "cash", "budget", "dollars")      # where the number of shares of a purchase can come from when the user typed none: the Account Manager sized it from the user's money (entries 98, 111 and 122)


def _decide_unsized(entry: dict, portfolio: Portfolio, mode: str, price, now: datetime, valid_minutes: int, unsaid: bool = False) -> dict:
    """An order of the user's that still has no number of shares when it is decided (design choices entry 98): a SALE (the Account Manager does not guess how much of a holding to sell: say how many), a purchase
    with no price, one that the cash (or the user's own dollars for it) cannot buy a single share of, or a purchase with no number in a request where another purchase has one (`unsaid`, entry 122: say how much of it; the cash is
    never its default). Never traded."""
    decision = _decision(entry, mode, price if _usable(price) else None, now, valid_minutes)
    symbol = entry["symbol"]
    if entry["side"] == "sell":
        held = portfolio.position(symbol)
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=("you named a sale and no number of shares: say how many to sell" + (f" (the account holds {sh.text(held.shares)} share{'s' if held.shares != 1 else ''} of {symbol})" if held else f" (the account holds no {symbol})")
                                                                                   + "; the Account Manager does not choose how much of a holding to sell" + (f" (${entry['dollars']:,.2f} is not a number of shares to sell)" if entry.get("dollars") is not None else "")))
    elif not _usable(price):
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {symbol}, so the order cannot be sized or priced")
    elif unsaid:
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"you gave an amount for another stock and none for {symbol}: say how much of {symbol} you want (the cash is not spent on it by default); nothing is placed")
    elif entry.get("dollars") is not None:
        decision.update(will_trade="no_funds", funds_avail=False, reason=f"your ${digits(entry['dollars']):,.2f} for {symbol} does not buy even one share at ${price:,.2f} (the cash is ${portfolio.cash_avail:,.2f}): nothing is placed")
    else:
        decision.update(will_trade="no_funds", funds_avail=False, reason=f"the cash ${portfolio.cash_avail:,.2f} does not buy even one share of {symbol} at ${price:,.2f}: nothing is placed")
    return decision


def _usable(price) -> bool:
    """Whether `price` is a price that can be traded on: a number above zero. A float is a TypeError (`money.exact`): the prices come in through a door."""
    if price is None:
        return False
    try:
        return exact(price, "a price") > 0
    except ValueError:
        return False


def _decide_one(entry: dict, portfolio: Portfolio, mode: str, price, now: datetime, valid_minutes: int, funding: dict | None = None, why_not: str | None = None, sale_proceeds: Decimal = Decimal("0.00")) -> dict:
    """One order: priced, then permitted (`yes_user`) or refused with the reason. The first row that fits:

        no price                                          no_system_denied   the order cannot be priced, so it cannot be judged against the account
        a sale of a stock not held, or of more than held  no_funds           (the shares are what a sale needs)
        a purchase the cash does not pay for in full      no_funds           with the shares the cash does pay for (the cash and what the user's own sale of the same request brings: `sale_proceeds`, entry 143)
        a purchase the cash does not pay for in full, and the user permitted selling holdings to pay for it (`funding`, entry 142)   yes_user   with `funding` on the decision: the desk raises the money first
        anything else                                     yes_user           the user's own order is the approval

    `funding` is what the Account Manager put on this purchase when the user permitted RaiseFunds (`funding.decision_funding`: the shortfall of the run, the plan's id, how it was permitted); the decision carries it and a purchase the cash does
    not cover is permitted as typed (never cut), because the money is to be raised before it is placed. `why_not` is the Account Manager's sentence for a purchase the cash does not cover that is NOT funded (the user said no, Auto mode
    without the words, the holdings could not pay for it): it ends the reason. `sale_proceeds` (entry 143, S1) is what the user's OWN SALES of the same request bring (an estimate at their quotes; they are placed before any purchase): it counts
    as cash for a purchase, and the handshake still reads the cash the book shows when the purchase is approved, so a sale that fails or fills lower leaves the purchase refused, never half. Nothing of the profile is read: no limit of it applies (see the module text)."""
    decision = _decision(entry, mode, price if _usable(price) else None, now, valid_minutes)
    symbol, shares, side = entry["symbol"], sh.quantity(entry["shares"]), entry["side"]
    own = f"you named this trade yourself ({side} {sh.text(shares)} share{'s' if shares != 1 else ''} of {symbol}{'' if entry.get('how_many') in (None, 'typed') else ', the number ' + HOW_WORDS[entry['how_many']]}): no recommendation was made and the Critic was not asked"
    if not _usable(price):
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {symbol}, so the order cannot be priced or checked against the account ({own})")
        return decision
    if side == "sell":
        held = portfolio.position(symbol)
        free = portfolio.free_shares(symbol)      # the shares a waiting sale has set aside are not for sale again (entry 140)
        if held is None or free < shares:
            decision.update(will_trade="no_funds", funds_avail=False, reason=(f"the account holds no {symbol}: there is nothing to sell" if held is None else
                                                                                f"the account holds {sh.text(held.shares)} share{'s' if held.shares != 1 else ''} of {symbol}"
                                                                                + (f", {sh.text(held.shares - free)} of them set aside for waiting orders," if free < held.shares else "") + f" and you asked to sell {sh.text(shares)}"))
            return decision
        decision.update(will_trade="yes_user", funds_avail=True, est_cost=Decimal("0.00"), reason=f"the user approved: {own}")      # a sale needs no money: its estimate is 0, as a recommended `reduce`'s
        return decision
    cost = trade_amount(shares, price)
    decision.update(action="add" if portfolio.position(symbol) else "open", est_cost=cost)
    can_pay = affordable_shares(portfolio.cash_avail + sale_proceeds, price, shares)
    paid_by = f"the cash ${portfolio.cash_avail:,.2f} and about ${sale_proceeds:,.2f} from your own sale (placed first)" if sale_proceeds else f"the cash ${portfolio.cash_avail:,.2f}"
    if funding is not None:      # the user permitted raising funds (entry 142): the purchase carries it, whether or not the cash alone would pay for it (the first purchase placed uses the cash)
        decision["funding"] = dict(funding)
    if can_pay < shares and funding is not None:
        decision.update(will_trade="yes_user", funds_avail=True, funds_note=f"needs RaiseFunds: the cash does not cover it, and ${funding['shortfall']:,.2f} is to be raised by selling holdings first",
                        reason=(f"the user approved: {own}; {paid_by} does not pay for {sh.text(shares)} shares of {symbol} at ${price:,.2f} (${cost:,.2f}), so the Account Manager raises the money by selling holdings first "
                                f"(RaiseFunds: ${funding['shortfall']:,.2f} to raise for the request, {HOW_FUNDED.get(funding['by'], 'you permitted it')}) and then buys as you typed it"))
        return decision
    if can_pay < shares:
        decision.update(will_trade="no_funds", funds_avail=False, reason=(f"{paid_by} does not pay for {sh.text(shares)} shares of {symbol} at ${price:,.2f} (${cost:,.2f}); "
                                                                          + (f"the most it pays for is {sh.text(can_pay)} share{'s' if can_pay != 1 else ''}" if can_pay else "it does not pay for even one share" if sh.whole(shares) else "it does not pay for any of it")
                                                                          + ": the order is placed as you typed it or not at all" + (f"; {why_not}" if why_not else "")))
        return decision
    decision.update(will_trade="yes_user", funds_avail=True, reason=f"the user approved: {own}")
    if sale_proceeds and affordable_shares(portfolio.cash_avail, price, shares) < shares:      # the cash alone does not pay for it: the user's own sale does, and it is placed first (entry 143, S1)
        decision.update(funds_note=f"paid for with {paid_by}", reason=f"the user approved: {own}; {paid_by} pays for it")
    return decision


def _waiting_card(entry: dict, decision: dict, cash: Decimal) -> dict:
    """The confirmation card of an order that WAITS for its limit (entry 140): what it is, the limit, the price now, what is set aside and until when."""
    plan = decision["waiting"]
    what = f"{plan['side']} ${plan['dollars']:,.2f} of {entry['symbol']}" if plan.get("dollars") is not None else f"{plan['side']} {sh.text(plan['shares'])} share{'s' if sh.quantity(plan['shares']) != 1 else ''}"
    way = orders.REASON_WORDS[plan["side"]]
    held = f"${plan['reserve_cash']:,.2f} of the cash" if plan["side"] == "buy" else f"{sh.text(plan['reserve_shares'])} share{'s' if sh.quantity(plan['reserve_shares']) != 1 else ''}"
    head = ([f"      YOU NAMED A BRAND, NOT A LISTED COMPANY: the Data Clerk found its listed owner" if entry.get("found_by") == "brand" else f"      YOU NAMED A COMPANY, NOT A TICKER: the Data Clerk found its symbol", f"        you typed   {entry['typed_company']}"]
            if entry.get("typed_company") is not None else [])
    text = "\n".join([*head, f"        company     {entry['company']}", f"        symbol      {entry['symbol']}", f"        order       {what} at ${plan['limit']:,.2f} {way}: a DAY order that WAITS (the price is ${plan['price']:,.2f} now)",
                      f"        it sets aside {held} until it fills or the session ends ({plan['expires_utc']})"])
    return {"text": text, "symbol": entry["symbol"], "company": entry["company"], "shares": plan.get("shares"), "est_cost": decision["est_cost"], "dollars": plan.get("dollars"), "price": decision.get("price"), "action": decision["action"],
            "question": f"Is this the order you meant: {what} of {entry['company']} ({entry['symbol']}) at ${plan['limit']:,.2f} {way}, waiting for the price? [y/N] "}


def confirmation_card(entry: dict, decision: dict, cash: Decimal) -> dict:
    """What the user is shown when they NAMED A COMPANY (Ask mode): the symbol the Data Clerk found, the company's name, the shares and the amount, and the question. A card of the kind `terminal_approver` prints (its
    `text`, then its own `question`)."""
    if decision.get("waiting"):
        return _waiting_card(entry, decision, cash)
    side, amount = entry["side"], trade_amount(entry["shares"], price_door(decision["price"]))
    what = f"{side} {sh.text(entry['shares'])} share{'s' if sh.quantity(entry['shares']) != 1 else ''}"
    dollars = decision.get("dollars")                                                      # an ORDER IN DOLLARS (entry 111, M2) says its dollars first, and the shares as an estimate at the price it was sized at
    brand = entry.get("found_by") == "brand"         # entry 101: what the user typed is not itself listed, and the symbol is its LISTED OWNER's
    head = ([f"      YOU NAMED A BRAND, NOT A LISTED COMPANY: the Data Clerk found its listed owner" if brand else f"      YOU NAMED A COMPANY, NOT A TICKER: the Data Clerk found its symbol", f"        you typed   {entry['typed_company']}"]
            if entry.get("typed_company") is not None else [f"      YOU DID NOT SAY HOW MANY SHARES: {HOW_WORDS.get(entry.get('how_many'), 'the number was worked out')}"])
    text = "\n".join([*head,
                      f"        company     {entry['company']}" + (f"   (the listed owner of {entry['typed_company']!r})" if brand else ""),
                      f"        symbol      {entry['symbol']}",
                      (f"        order       {side} ${dollars:,.2f}: about {sh.text(entry['shares'])} shares at ${decision['price']:,.2f}" if dollars else f"        order       {what} at ${decision['price']:,.2f} = about ${amount:,.2f}")
                      + (f"   (cash ${cash:,.2f})" if side == "buy" else "")])
    return {"text": text, "symbol": entry["symbol"], "company": entry["company"], "shares": entry["shares"], "est_cost": decision["est_cost"], "dollars": dollars, "price": decision.get("price"), "action": decision["action"],
            "question": f"Is this the order you meant: {f'{side} ${dollars:,.2f}' if dollars else what} of {entry['company']} ({entry['symbol']})" + (f", the listed owner of {entry['typed_company']!r}" if brand else "") + ("? [y/N] " if dollars else f", about ${amount:,.2f}? [y/N] ")}


def _confirm_looked_up(order: WorkOrder, entry: dict, decision: dict, mode: str, approver, cash: Decimal) -> None:
    """A company the user NAMED (the symbol is the Data Clerk's): in ASK mode the user confirms the symbol, the company, the shares and the amount before anything goes to the Trader (no answer, or nobody to ask: it stays
    `pending_user` and is not traded; no is `no_user_denied`); in AUTO mode the returned symbol is used without asking, and the reason says so. A decision that is not `yes_user` is not asked about."""
    sized = entry.get("how_many") in SIZED         # the user wrote no number of shares and the Account Manager worked one out (entry 98): the final proposal is shown in Ask mode
    if (entry.get("typed_company") is None and not sized) or decision["will_trade"] != "yes_user":
        return
    named = f"{entry['typed_company']!r} -> {entry['company']} ({entry['symbol']})" if entry.get("typed_company") is not None else f"{entry['company']} ({entry['symbol']}), {sh.text(entry['shares'])} shares"
    kind = "the symbol the Data Clerk found for the listed owner of what you named, " if entry.get("found_by") == "brand" else "the symbol the Data Clerk found for the company you named, "     # entry 101: "Google" is not listed, its owner is
    if mode == "auto":
        if entry.get("typed_company") is not None:
            decision["reason"] += f"; auto mode: {kind}{named}, was used without asking"
        return
    what_named = (kind.replace("you named", "they named") if entry.get("typed_company") is not None else "the order ") + named
    decision["asked_user"] = approver is not None
    answer = None
    if approver is not None:
        try:
            answer = approver(confirmation_card(entry, decision, cash))
        except Exception as error:  # a prompt that failed: the user has not said yes
            order.errors = [*order.errors, f"the user could not be asked about {entry['symbol']} ({type(error).__name__})"]
    decision.update(will_trade="yes_user" if answer is True else "no_user_denied" if answer is False else "pending_user",
                    reason=(f"the user confirmed {what_named}, with the shares and the amount: " if answer is True else
                            f"the user did not confirm {what_named}: " if answer is False else
                            f"{what_named} needs your yes in Ask mode and nobody was there to ask: ") + decision["reason"])


def _decide_whole_only(entry: dict, mode: str, price, now: datetime, valid_minutes: int) -> dict:
    """An order for a FRACTION of a share on an account book whose share columns are still integer (design choices entry 110): refused, with the fix named, before anything is written. Never rounded into a whole number,
    which would be a trade the user did not type."""
    decision = _decision(entry, mode, price if _usable(price) else None, now, valid_minutes)
    decision.update(will_trade="no_system_denied", funds_avail=False, shares=Decimal(0), reason=(f"you asked to {entry['side']} {sh.text(entry['shares'])} shares of {entry['symbol']}, which is a fraction of a share, and this account book holds whole shares only "
                                                                              "(its share columns are still integer): run `bash scripts/setup_postgres.sh` to change that. Nothing is placed, and the order is not rounded to a whole number"))
    return decision


@traced("account_manager.direct_decisions")
def decide_direct(order: WorkOrder, portfolio: Portfolio, profile: UserProfile, mode: str | None, prices: dict[str, Decimal], now: datetime, valid_minutes: int, approver=None,
                  sized: dict[str, dict] | None = None, fractions: bool = True, funding: dict[str, dict] | None = None, why_not: dict[str, str] | None = None, sale_proceeds: Decimal = Decimal("0.00"),
                  waiting_funding: dict[str, str] | None = None) -> WorkOrder:
    """One decision for each of `order.direct_orders` (see the module text); return the SAME order, `decided`, with `trade_decisions`. `prices` is {symbol: the current price} from the Data Clerk. `approver` is the callable
    that asks the user (True, False, None for no answer), used only in Ask mode for a company the user named. `fractions` is whether the account book can hold a fraction of a share (`AccountStore.holds_fractions`; True when
    there is no book): an order for a fraction of a share is refused (`no_system_denied`, the fix named) when it cannot, and never rounded. A refusal of the user's side of things is a message on the order and nothing is decided (an order that is not the
    user's, a profile that is not the portfolio's user's, a mode that is not ask or auto); an order already `decided` or `traded` comes back unchanged. The account is not changed: the Trader's handshake books the fills.
    The profile is read for its user and its mode only. `funding` ({record_id: `funding.decision_funding`}) is what the Account Manager decided when the user permitted selling holdings to pay for a purchase the cash does not cover (entry 142):
    such a purchase is `yes_user` as typed and carries it. `why_not` ({record_id: a sentence}) ends the reason of a purchase the cash does not cover that is not funded. Both are the Account Manager's, made before this pure function.
    `sale_proceeds` (entry 143, S1) is what the user's own sales of the same request bring, counted as cash for a purchase (see `_decide_one`). `waiting_funding` ({record_id: "words" | "setting"}, entry 143, W-F) is the permission to sell holdings,
    when the price is reached, for a purchase that waits and that the free cash does not cover (see `_decide_waiting`); `why_not` also ends the reason of such a purchase that is refused."""
    if order.status in ("decided", "traded"):
        return order
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    if order.status != "new":
        order.errors = [*order.errors, f"the Account Manager decides a direct order on a new work order (status {order.status})"]
        return order
    problems = verify_direct_orders(order.direct_orders)
    if problems:
        order.errors = [*order.errors, *problems]
        return order
    if profile.user != portfolio.user:
        order.errors = [*order.errors, f"the user profile is for {profile.user!r} but the portfolio belongs to {portfolio.user!r}: no trade is decided"]
        return order
    mode = mode or profile.trade_mode
    if mode not in MODES:
        order.errors = [*order.errors, f"mode must be one of {', '.join(MODES)}"]
        return order
    for entry in order.direct_orders:       # an order with no number of shares of the user's: the Account Manager's sizing gives it one (`sized`: {record_id: {"shares", "how"}}), written on the user's record
        if entry["shares"] is None and (sized or {}).get(entry["record_id"]) and sized[entry["record_id"]]["shares"] is not None:
            entry["shares"], entry["how_many"] = sized[entry["record_id"]]["shares"], sized[entry["record_id"]]["how"]
    def decide(entry: dict) -> dict:
        price = prices.get(entry["symbol"])
        if (floor := under_floor(entry, price)) is not None:      # a purchase priced under the user's floor: not placed (entry 140, F3)
            return _decide_under_floor(entry, mode, price, floor, now, valid_minutes)
        if entry["shares"] is not None and not fractions and not sh.whole(entry["shares"]):
            return _decide_whole_only(entry, mode, price, now, valid_minutes)
        if orders.waits(entry, price):      # a purchase above its ceiling, a sale under its floor: it WAITS as a day order (entry 140, W1), or says why it cannot
            return _decide_waiting(entry, portfolio, mode, price, now, valid_minutes, fractions, (waiting_funding or {}).get(entry["record_id"]), (why_not or {}).get(entry["record_id"]))
        if entry["shares"] is None:
            return _decide_unsized(entry, portfolio, mode, prices.get(entry["symbol"]), now, valid_minutes, unsaid=((sized or {}).get(entry["record_id"]) or {}).get("how") == "unsaid")
        return _decide_one(entry, portfolio, mode, prices.get(entry["symbol"]), now, valid_minutes, (funding or {}).get(entry["record_id"]), (why_not or {}).get(entry["record_id"]), sale_proceeds)

    order.trade_decisions = [decide(entry) for entry in order.direct_orders]
    for entry, decision in zip(order.direct_orders, order.trade_decisions):      # what the sizing says of a purchase it sized from the user's money (entry 111): the POT of the run (M1) and, for an ORDER IN DOLLARS, its dollars (M2); the shares and the estimate stay
        given = (sized or {}).get(entry["record_id"])
        if given and given.get("shares") is not None and entry.get("how_many") in SIZED:
            decision["size"] = {"shares": entry["shares"], "pot": given.get("pot"), "allocation": given.get("allocation"), "binding": given.get("binding"), "how": entry["how_many"]}
            if given.get("dollars"):
                decision["dollars"] = given["dollars"]
    for entry, decision in zip(order.direct_orders, order.trade_decisions):
        if decision.get("waiting"):      # an order that waits is confirmed like any other, BEFORE anything is set aside (a company the user named: Ask mode asks)
            decision["will_trade"] = "yes_user"
            _confirm_looked_up(order, entry, decision, mode, approver, portfolio.cash_avail)
            if decision["will_trade"] == "yes_user":
                decision["will_trade"] = "no_system_denied"      # confirmed (or nothing to confirm): it waits
            else:
                decision.pop("waiting", None)      # a no, or nobody to answer: nothing waits and nothing is set aside
        else:
            _confirm_looked_up(order, entry, decision, mode, approver, portfolio.cash_avail)
    order.status = "decided"
    return order


WORDS = {"yes_user": "APPROVED: your own order, sent to the Trader", "no_funds": "NOT PLACED (no funds)", "no_system_denied": "NOT PLACED (no price)",
         "pending_user": "WAITING for your yes (nobody to ask): not placed", "no_user_denied": "NOT PLACED (you did not confirm it)"}
ABOVE_LIMIT_WORD = "NOT PLACED (above your price limit)"      # a `no_system_denied` whose reason is the price limit (entry 138): an order that could not wait
WAITING_WORD = "WAITING in the market (a day order, its cash or shares set aside)"      # a decision with `open_order_id`: the order was placed in the account book (entry 140)
UNDER_FLOOR_WORD = "NOT PLACED (under your price floor)"


def direct_section(order: WorkOrder) -> list[str]:
    """The DIRECT ORDER section of a report (plain text from the records, no model): each trade the user named, the company and symbol the Data Clerk found (and what was typed when it was a name), the price it was judged
    at and the Account Manager's decision. Empty when the order is not a direct one."""
    if not order.direct_orders:
        return []
    lines = ["DIRECT ORDER  you named the trade yourself: no recommendation was made and the Critic was not asked"]
    decisions = {d.get("record_id"): d for d in order.trade_decisions}
    for entry in order.direct_orders:
        d = decisions.get(entry["record_id"])
        who = f"{entry.get('company') or entry['symbol']} ({entry['symbol']})" + (f" [the listed owner of {entry['typed_company']!r}, which you named]" if entry.get("typed_company") and entry.get("found_by") == "brand" else f" [you named the company as {entry['typed_company']!r}]" if entry.get("typed_company") else "")
        what = f"{entry['side']} {sh.text(entry['shares'])} share{'s' if entry['shares'] != 1 else ''} of {who}" if entry["shares"] is not None else f"{entry['side']} {who} (you wrote no number of shares)"
        written = (d or {}).get("dollars") or entry.get("dollars")      # an order in dollars (entries 111 and 122), or one the user gave dollars for: the dollars first, the shares an estimate
        if written and entry["shares"] is not None:
            what = f"{entry['side']} ${digits(written):,.2f} of {who} (about {sh.text(entry['shares'])} shares)"
        elif entry.get("dollars") is not None and entry["shares"] is None:
            what = f"{entry['side']} ${digits(entry['dollars']):,.2f} of {who}"
        if d is None:
            lines.append(f"  {what}: not decided")
            continue
        at = f" at ${d['price']:,.2f}" + (f" (about ${d['est_cost']:,.2f})" if d["est_cost"] else "") if d.get("price") else ""
        word = (WAITING_WORD if d.get("open_order_id") is not None else UNDER_FLOOR_WORD if d["will_trade"] == "no_system_denied" and d.get("min_price") is not None and "under your floor" in d["reason"] else
                ABOVE_LIMIT_WORD if d["will_trade"] == "no_system_denied" and d.get("max_price") is not None and "above your limit" in d["reason"] else WORDS.get(d["will_trade"], d["will_trade"]))
        lines.append(f"  {what}{at}   {word}")
        if d["will_trade"] != "yes_user":
            lines.append(f"      {d['reason']}")
    return lines
