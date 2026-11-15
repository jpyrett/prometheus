"""A DIRECT ORDER: the user names the trade, so nothing is recommended or reviewed (design choices entries 87 and 88).

    found   = account_manager.request_symbols([o.named for o in intent.orders])       # the Account Manager ASKS the Data Clerk for each stock's company name and symbol (a `lookup_symbol` order)
    problems = name_the_stocks(order, intent.orders, typed, now, found)               # the Account Manager APPENDS them to the work order, and writes the user's own orders in `order.direct_orders`
    order   = decide_direct(order, portfolio, profile, mode, prices, now, valid_minutes, approver)       # status "new" -> "decided": one trade decision for each of the user's orders

"Buy 50 shares of MSFT" is not a request for advice. The Advisor would only recommend what the user already chose, and the Critic would review a recommendation nobody made, so the desk skips both, and it skips the Data
Clerk's node too (it has no research to do). The work order goes from the Account Manager's reading of the request to the Account Manager's decision and on to the Trader:

    account_manager (reads the request, asks the Data Clerk, appends) -> account_manager_direct (this module: prices, decides) -> trader

THE COMPANY NAME AND THE SYMBOL ARE ON THE WORK ORDER (Jacob, 2026-11-15). A request for research goes through the Data Clerk's node, which receives the work order and appends what it finds; a direct order skips that node, so
the ACCOUNT MANAGER asks the Data Clerk (`DataClerk.request_symbols`: a ticker is looked up in the symbol directory for its company, a company's name for its ticker, by exact matching, never a guess) and appends the answer to
the work order itself: `order.data.companies` (what the Data Clerk found, one row for each stock) and the user's own record in `order.direct_orders` (the symbol, the company's name, and what the user typed when it was a
name). A name that is not found or fits several listings stops the order with the Data Clerk's own message. The user names a company ("Buy 50 shares of Microsoft"): the Account Manager goes on with the symbol the Data Clerk
returned, and in ASK mode the user is asked to confirm it first, with the symbol, the company's name, the shares and the amount; in AUTO mode the returned symbol is used without asking. A ticker the user typed is not asked
about again (the symbol is theirs).

AUTO-APPROVED. The user's own order is the approval: the decision says `yes_user` (the user said yes by naming the trade; the Account Manager does not ask a second time, in Ask mode too, unless the symbol is the Data Clerk's:
above) with `origin: "direct"`. `work_order.verify_trade` accepts such a decision only while `order.direct_orders` holds the user's matching record (the same ticker and side, and no more shares than the user typed), so the
word `direct` alone permits nothing, and a decision that is not direct still needs the Critic's approval. The Trader and the handshake are the same code as for a recommendation: one ticket, `approve_trade`, the fill, `record_fill`.

THERE IS NO PROFILE LIMIT (Jacob, 2026-11-15: "profile limits never apply to any orders", then "I do not want any of these limits ... remove them all", then "the only sizing that we need is to make sure the trade order has enough funds
to cover it": design choices entry 91). The profile has no cap on the share of the account one stock may be, on the loss a stop would cost, on the exposure of one review or on one order, and the sizing has no liquidity cap
either: nothing here sizes anything and nothing here imports `sizing` (a test reads this module's source and the Account Manager's `decide_direct` for it). The size is the user's. What stops a direct order is the account itself:
the cash for a purchase, the shares held for a sale (and the T+1 rule, which the Trader's handshake keeps). A trade is placed AS TYPED OR NOT AT ALL: when the cash does not pay for all the shares the order is `no_funds` and
says how many shares the cash does pay for, it is never made smaller on the user's behalf, and the handshake refuses, and does not cut, an order the account can no longer pay for in full. No holding is sold to fund a direct
purchase (RaiseFunds is for recommendations).

TODO (Jacob, 2026-11-15): LIMIT ORDERS. A direct order is a MARKET order: the Trader places it at the current quote (the paper broker adds a few basis points of slippage) and it does not look at the market hours. A limit
order ("buy 50 shares of MSFT at $500 or less"; "sell 10 shares of TSLA at $450 or more") is not built: the intent form has no price, the user's record has no limit, the decision does not carry one and the ticket takes its
limit range from the quote. It needs the form (a price the user typed, checked against their words), a limit on the record and the decision, the ticket's range, and an order that may rest until the price is reached.

Nothing here calls a model or an API: the prices and the answers of the Data Clerk are given.
"""
from datetime import datetime, timedelta

from prometheus.agents.critic.common import utc_text
from prometheus.common.portfolio import Portfolio, affordable_shares
from prometheus.common.user_profile import UserProfile
from prometheus.common.work_order import DIRECT, DataResult, WorkOrder, verify_direct_orders

MODES = ("ask", "auto")


def record_id_of(request_id: str, number: int) -> str:
    """The id of the user's n-th order of a request: "direct-<request id>-1". Short, and the same on every record that belongs to it (the order, the decision, the trade and the account book's row)."""
    return f"direct-{request_id}-{number}"


def name_the_stocks(order: WorkOrder, orders, typed: str, now: datetime, found: WorkOrder) -> list[str]:
    """The Account Manager APPENDS the Data Clerk's answer to the work order (see the module text). `found` is the `lookup_symbol` order the Data Clerk answered, for the stocks of `orders` (the user's, as typed); `typed`
    the user's own words. On success the work order gets `data` (the Data Clerk's rows in `data.companies`, its calls and its answer), `subject` (the symbols) and `direct_orders` (one record for each of the user's trades,
    in the order they typed them: the id, the side, the symbol, the company's name, the shares, what was typed when it was a company's name, the user's words and the moment), and the list returned is empty. Otherwise
    nothing of the order is changed and the list says why, in the Data Clerk's own words (a name it did not find, one that fits several listings, a lookup that failed)."""
    if found.status != "data_ready" or found.data is None:
        return [*found.errors] or ["the Data Clerk could not find the symbols of what you named"]
    rows = {row["brand"]: row for row in found.data.companies}
    entries, problems = [], []
    for number, wanted in enumerate(orders, 1):
        row = rows.get(" ".join(wanted.named.split()))
        if row is None:
            problems.append(f"the Data Clerk found no symbol and company name for {wanted.named}")
            continue
        entries.append({"record_id": record_id_of(order.request_id, number), "side": wanted.side, "symbol": row["symbol"], "company": row["company"], "shares": wanted.shares, "request": typed,
                        "requested_utc": utc_text(now), "typed_company": wanted.company, "how_many": "typed" if wanted.shares is not None else None})
    if not problems and len({e["symbol"] for e in entries}) != len(entries):
        problems.append("two of your orders are for the same stock (a ticker and the company's name, for example): name each stock once")
    problems += verify_direct_orders(entries) if not problems else []
    if problems:
        return problems
    data = order.data or DataResult()
    data.companies = [*data.companies, *found.data.companies]
    data.tool_calls, data.problems = [*data.tool_calls, *found.data.tool_calls], [*data.problems, *found.data.problems]
    data.answer, data.completed_utc = found.data.answer, found.data.completed_utc
    order.data, order.subject, order.direct_orders = data, list(dict.fromkeys(e["symbol"] for e in entries)), entries
    return []


def _decision(entry: dict, mode: str, price, now: datetime, valid_minutes: int) -> dict:
    """The decision of one order before it is decided: every key `decisions.decide_trades` writes, and the four of a direct order: `origin`, the `company` the Data Clerk found, the `price` it was priced at and its own
    `valid_until_utc`."""
    return {"record_id": entry["record_id"], "review_id": None, "symbol": entry["symbol"], "action": "reduce" if entry["side"] == "sell" else "open", "shares": entry["shares"] or 0, "est_cost": 0.0, "mode": mode,
            "will_trade": None, "funds_avail": None, "funds_note": None, "asked_user": False, "escalated": False, "reason": "", "decided_utc": utc_text(now), "size": None,
            "origin": DIRECT, "company": entry["company"], "price": price, "valid_until_utc": utc_text(now + timedelta(minutes=valid_minutes))}


HOW_WORDS = {"asked": "you gave when the Account Manager asked how many", "cash": "the Account Manager worked out: as much as your cash allows", "budget": "the Account Manager worked out from the dollars you gave"}


def _decide_unsized(entry: dict, portfolio: Portfolio, mode: str, price, now: datetime, valid_minutes: int) -> dict:
    """An order of the user's that still has no number of shares when it is decided (design choices entry 98): a SALE (the Account Manager does not guess how much of a holding to sell: say how many), a purchase
    with no price, or one that the cash cannot buy a single share of. Never traded."""
    decision = _decision(entry, mode, price if _usable(price) else None, now, valid_minutes)
    symbol = entry["symbol"]
    if entry["side"] == "sell":
        held = portfolio.position(symbol)
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=("you named a sale and no number of shares: say how many to sell" + (f" (the account holds {held.shares:,} share{'s' if held.shares != 1 else ''} of {symbol})" if held else f" (the account holds no {symbol})")
                                                                                   + "; the Account Manager does not choose how much of a holding to sell"))
    elif not _usable(price):
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {symbol}, so the order cannot be sized or priced")
    else:
        decision.update(will_trade="no_funds", funds_avail=False, reason=f"the cash ${portfolio.cash_avail:,.2f} does not buy even one share of {symbol} at ${price:,.2f}: nothing is placed")
    return decision


def _usable(price) -> bool:
    return isinstance(price, (int, float)) and not isinstance(price, bool) and price == price and 0 < price < float("inf")


def _decide_one(entry: dict, portfolio: Portfolio, mode: str, price, now: datetime, valid_minutes: int) -> dict:
    """One order: priced, then permitted (`yes_user`) or refused with the reason. The first row that fits:

        no price                                          no_system_denied   the order cannot be priced, so it cannot be judged against the account
        a sale of a stock not held, or of more than held  no_funds           (the shares are what a sale needs)
        a purchase the cash does not pay for in full      no_funds           with the shares the cash does pay for
        anything else                                     yes_user           the user's own order is the approval

    Nothing of the profile is read: no limit of it applies (see the module text)."""
    decision = _decision(entry, mode, price if _usable(price) else None, now, valid_minutes)
    symbol, shares, side = entry["symbol"], entry["shares"], entry["side"]
    own = f"you named this trade yourself ({side} {shares:,} share{'s' if shares != 1 else ''} of {symbol}{'' if entry.get('how_many') in (None, 'typed') else ', the number ' + HOW_WORDS[entry['how_many']]}): no recommendation was made and the Critic was not asked"
    if not _usable(price):
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {symbol}, so the order cannot be priced or checked against the account ({own})")
        return decision
    if side == "sell":
        held = portfolio.position(symbol)
        if held is None or held.shares < shares:
            decision.update(will_trade="no_funds", funds_avail=False, reason=(f"the account holds no {symbol}: there is nothing to sell" if held is None else
                                                                                f"the account holds {held.shares:,} share{'s' if held.shares != 1 else ''} of {symbol} and you asked to sell {shares:,}"))
            return decision
        decision.update(will_trade="yes_user", funds_avail=True, est_cost=0.0, reason=f"the user approved: {own}")      # a sale needs no money: its estimate is 0, as a recommended `reduce`'s
        return decision
    cost = round(shares * price, 2)
    decision.update(action="add" if portfolio.position(symbol) else "open", est_cost=cost)
    can_pay = affordable_shares(portfolio.cash_avail, price, shares)
    if can_pay < shares:
        decision.update(will_trade="no_funds", funds_avail=False, reason=(f"the cash ${portfolio.cash_avail:,.2f} does not pay for {shares:,} shares of {symbol} at ${price:,.2f} (${cost:,.2f}); "
                                                                          + (f"the most it pays for is {can_pay:,} share{'s' if can_pay != 1 else ''}" if can_pay else "it does not pay for even one share")
                                                                          + ": the order is placed as you typed it or not at all"))
        return decision
    decision.update(will_trade="yes_user", funds_avail=True, reason=f"the user approved: {own}")
    return decision


def confirmation_card(entry: dict, decision: dict, cash: float) -> dict:
    """What the user is shown when they NAMED A COMPANY (Ask mode): the symbol the Data Clerk found, the company's name, the shares and the amount, and the question. A card of the kind `terminal_approver` prints (its
    `text`, then its own `question`)."""
    side, amount = entry["side"], round(entry["shares"] * decision["price"], 2)
    what = f"{side} {entry['shares']:,} share{'s' if entry['shares'] != 1 else ''}"
    head = ([f"      YOU NAMED A COMPANY, NOT A TICKER: the Data Clerk found its symbol", f"        you typed   {entry['typed_company']}"] if entry.get("typed_company") is not None else
            [f"      YOU DID NOT SAY HOW MANY SHARES: {HOW_WORDS.get(entry.get('how_many'), 'the number was worked out')}"])
    text = "\n".join([*head,
                      f"        company     {entry['company']}",
                      f"        symbol      {entry['symbol']}",
                      f"        order       {what} at ${decision['price']:,.2f} = about ${amount:,.2f}" + (f"   (cash ${cash:,.2f})" if side == "buy" else "")])
    return {"text": text, "symbol": entry["symbol"], "company": entry["company"], "shares": entry["shares"], "est_cost": decision["est_cost"], "action": decision["action"],
            "question": f"Is this the order you meant: {what} of {entry['company']} ({entry['symbol']}), about ${amount:,.2f}? [y/N] "}


def _confirm_looked_up(order: WorkOrder, entry: dict, decision: dict, mode: str, approver, cash: float) -> None:
    """A company the user NAMED (the symbol is the Data Clerk's): in ASK mode the user confirms the symbol, the company, the shares and the amount before anything goes to the Trader (no answer, or nobody to ask: it stays
    `pending_user` and is not traded; no is `no_user_denied`); in AUTO mode the returned symbol is used without asking, and the reason says so. A decision that is not `yes_user` is not asked about."""
    sized = entry.get("how_many") in ("asked", "cash", "budget")         # the user wrote no number of shares and the Account Manager worked one out (entry 98): the final proposal is shown in Ask mode
    if (entry.get("typed_company") is None and not sized) or decision["will_trade"] != "yes_user":
        return
    named = f"{entry['typed_company']!r} -> {entry['company']} ({entry['symbol']})" if entry.get("typed_company") is not None else f"{entry['company']} ({entry['symbol']}), {entry['shares']:,} shares"
    if mode == "auto":
        if entry.get("typed_company") is not None:
            decision["reason"] += f"; auto mode: the symbol the Data Clerk found for the company you named, {named}, was used without asking"
        return
    what_named = ("the symbol the Data Clerk found for the company they named, " if entry.get("typed_company") is not None else "the order ") + named
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


def decide_direct(order: WorkOrder, portfolio: Portfolio, profile: UserProfile, mode: str | None, prices: dict[str, float], now: datetime, valid_minutes: int, approver=None,
                  sized: dict[str, dict] | None = None) -> WorkOrder:
    """One decision for each of `order.direct_orders` (see the module text); return the SAME order, `decided`, with `trade_decisions`. `prices` is {symbol: the current price} from the Data Clerk. `approver` is the callable
    that asks the user (True, False, None for no answer), used only in Ask mode for a company the user named. A refusal of the user's side of things is a message on the order and nothing is decided (an order that is not the
    user's, a profile that is not the portfolio's user's, a mode that is not ask or auto); an order already `decided` or `traded` comes back unchanged. The account is not changed: the Trader's handshake books the fills.
    The profile is read for its user and its mode only."""
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
        if entry["shares"] is None and (sized or {}).get(entry["record_id"]):
            entry["shares"], entry["how_many"] = sized[entry["record_id"]]["shares"], sized[entry["record_id"]]["how"]
    order.trade_decisions = [(_decide_one if entry["shares"] is not None else _decide_unsized)(entry, portfolio, mode, prices.get(entry["symbol"]), now, valid_minutes) for entry in order.direct_orders]
    for entry, decision in zip(order.direct_orders, order.trade_decisions):
        _confirm_looked_up(order, entry, decision, mode, approver, portfolio.cash_avail)
    order.status = "decided"
    return order


WORDS = {"yes_user": "APPROVED: your own order, sent to the Trader", "no_funds": "NOT PLACED (no funds)", "no_system_denied": "NOT PLACED (no price)",
         "pending_user": "WAITING for your yes (nobody to ask): not placed", "no_user_denied": "NOT PLACED (you did not confirm it)"}


def direct_section(order: WorkOrder) -> list[str]:
    """The DIRECT ORDER section of a report (plain text from the records, no model): each trade the user named, the company and symbol the Data Clerk found (and what was typed when it was a name), the price it was judged
    at and the Account Manager's decision. Empty when the order is not a direct one."""
    if not order.direct_orders:
        return []
    lines = ["DIRECT ORDER  you named the trade yourself: no recommendation was made and the Critic was not asked"]
    decisions = {d.get("record_id"): d for d in order.trade_decisions}
    for entry in order.direct_orders:
        d = decisions.get(entry["record_id"])
        who = f"{entry.get('company') or entry['symbol']} ({entry['symbol']})" + (f" [you named the company as {entry['typed_company']!r}]" if entry.get("typed_company") else "")
        what = f"{entry['side']} {entry['shares']:,} share{'s' if entry['shares'] != 1 else ''} of {who}" if entry["shares"] is not None else f"{entry['side']} {who} (you wrote no number of shares)"
        if d is None:
            lines.append(f"  {what}: not decided")
            continue
        at = f" at ${d['price']:,.2f}" + (f" (about ${d['est_cost']:,.2f})" if d["est_cost"] else "") if d.get("price") else ""
        lines.append(f"  {what}{at}   {WORDS.get(d['will_trade'], d['will_trade'])}")
        if d["will_trade"] != "yes_user":
            lines.append(f"      {d['reason']}")
    return lines
