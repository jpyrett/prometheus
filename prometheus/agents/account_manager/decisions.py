"""What the desk does with the Critic's verdicts: the Account Manager's trade decisions, and HOW MUCH to buy (Critic v1 spec section 12.2, design choices entries 56, 79, 80 and 104).

    order = decide_trades(order, portfolio, profile, approver, now, mode, balance, budget, prices, wanted)        # status "reviewed" -> "decided" (or stays "reviewed" while a user's answer is pending); a "decided" or "traded" order comes back unchanged

The Advisor recommends companies to BUY and the Critic approves or denies each one; NEITHER decides how much to buy, and neither touches the holdings. The Account Manager reads the verdicts: APPROVED trades, DENIED is
returned to the user (`session.denied_lines`), and it alone knows the money (`sizing.size_purchases`: the cash, the budget and the amount asked for: THE ONLY CHECK IS THAT THE FUNDS COVER THE ORDER, design choices entry 91; no
limit of the profile applies). For each review, in order, the FIRST row that fits (`will_trade` is the week 5 design's WillTrade):

    not approved                                              no_system_denied   final, whatever the reason (a blocker, a score under the mark, a finding that asks for a person to look): the user is never asked to override a denial
    approved, but `now` is after `valid_until_utc`            no_expired         the price check is old
    approved, but not even one share fits (or the money is not there)   no_funds
    approved, Auto mode                                       yes_auto           NOTHING IS ASKED (Jacob, 2026-11-15: "Auto mode will not ask any prompts", "Trade all approved recommendations in auto mode")
    approved, Ask mode                                        the user's answer: yes_user, no_user_denied, or pending_user (nobody to ask)

There is no second mark: the Critic's own approval mark is the only one, and a recommendation the Critic approved trades in Auto mode at any score it approved (entry 104). A review is a recommendation to BUY: one that
is not (an old stored `reduce`, `hold`) has nothing to trade, because a sale never comes from a recommendation (entry 105: the user's own order and RaiseFunds are the only sales). NO CASH IS RESERVED (Jacob, 2026-11-14): each
purchase may use all the funds available in the portfolio, whatever the other purchases cost; the Trader checks the cash again when it places each order.

THE SIZE (entry 80). After the rows above, EVERY approved, unexpired purchase (those that will trade on their own and those that wait for the user) is sized in ONE call of `sizing.size_purchases`,
so that the user is shown the FINAL shares and cost. The price is the Critic's deal price (else the plan's entry mid) and the rank comes from the Advisor's record; the equity (for the record and for "20% of my account") comes from the portfolio and `prices` (the current prices of
the symbols it holds: `sizing.equity_of`). Nothing else bounds a purchase: no limit of the profile, no stop, no volume (entry 91). The ALLOCATION, that is how the money is spread over the purchases:

    `balance="even"`          equal DOLLARS for every purchase, as close as whole shares allow, and what the equal amounts leave over is spent one share at a time to the smallest amount (entry 108); the user said "split it evenly"
                              and nothing else chooses it (entry 109); each decision carries `balance` {mode, target_dollars, of} (the equal amount, and how many shared it). With `fractional` (entry 110: the Account Manager
                              passes it when its switch is on and the account book can hold a fraction) the split is EQUAL TO THE CENT, in fractional shares: $30,000 over three is $10,000.00 of each
    `balance="round_robin"`   one share of each purchase at a time, in rank order, until the funds are used
    no `balance`              round robin, IN BOTH MODES (Jacob, 2026-11-14 for Auto: "take the available funds, buy a share of the first stock, a share of the next, until the requested funds are used up"; 2026-11-15 for Ask,
                              entry 106: *"No. its suppose to be round-robin. Not all in one."* A real Ask-mode request had put the whole $10,000 into the first of four stocks and left the others "no funds"). Never everything into the first.

`budget` is the funds requested (dollars; the cash when left out, and never more than the cash): the pot of the round robin and of the even split. A purchase that sizes to no share becomes `no_funds` with the sizing's reason
and is not asked about. Each sized purchase carries `decision["size"]`: shares, the caps (the cash and the budget), `binding` (what decided the number), the equity and what it was worked out from, the notes, and the allocation.
`sizing.funds_check` then asks whether the cost could exist (cash, or RaiseFunds from holdings in profit).

The Trader (Trader v1, `prometheus/agents/trader/`) reads these decisions: it makes one ticket for each that says `yes_auto` or `yes_user` and nothing for any other, and every trade it places passes
`work_order.verify_trade`, which reads the decision again. So a `pending_user` is never traded; the user's later answer (`record_user_decision`) makes it a `yes_user` that a later call of the Trader carries on from.
The Trader also asks the user, through the same `approver`, whether the sells that pay for a purchase may be made (`terminal_approver` shows the card's own `question`): that is RaiseFunds, the Account Manager's own business, and it is
not on the session's path (design/approved_or_denied_spec.md 3.3). This module itself trades nothing, and calls no API or model: the prices are given.

HOW MUCH, FROM THE USER (entry 85). In Ask mode the Account Manager first asks the user how much to buy when the request did not say (`clarify.py`, a bounded ReAct loop): what comes out is `balance`, `budget`, `wanted` (the
shares the user's words came to for each purchase) and this module sizes under them like any other request: the purchase never gets more than `wanted`.

A PRICE FLOOR AND SHARES TYPED WITH A RECOMMENDATION (entry 140). `min_price` is the user's price floor ("over $20 a share"): the other side of `max_price`, a price and never a size: a purchase priced under it gets no decision (`UNDER_FLOOR`) and every other
decision carries it for the handshake. The shares the user typed with a recommendation ("buy 10 of MU if the recommendation passes", "5 shares of each of the top 3") reach this module as `wanted` (`typed_wanted` maps them to the approved and neutral
purchases; `combine_wanted` keeps the smaller of two limits on one purchase): each such purchase is limited to them, binding `asked`, the funds still the only check, and a denied recommendation buys nothing.

THE POT AND ORDERS IN DOLLARS (design choices entry 111). Every decision sized from money carries the POT of the run in `decision["size"]["pot"]` (the budget, else the cash, never more than the cash: `sizing.pot_of`), which the handshake holds every purchase to and
`work_order.verify_trades` checks. In a FRACTIONAL run (`fractional=True`) a purchase of an even split (approved or neutral) is an ORDER IN DOLLARS: the decision carries `dollars` (its cost at the sizing price; `shares`, `est_cost` and `price` stay as the
ESTIMATE), and the Account Manager turns the dollars into shares at the price the trade fills at. Every place that shows a decision says the dollars first (`order_phrase`). A purchase of the recommendation path's round robin, or one held to the shares the
user asked for, is an order in shares, held by the pot.

DECIMAL MONEY (design choices entry 112): the cash, the budget, the prices, the amounts and the share counts of a decision are `Decimal` (a decision is a plain dict on the work order: a Decimal after this code wrote it, an int or a float after
`json` read it back, and every reader goes through a door). The Critic's deal price and the Advisor's entry price are the Advisor's and the Critic's measurements, and cross into the account here, once, through the price door (`_price_of`, four
places, half up); nothing here is a float.

ASK (entry 79): the user approves the order that WILL be placed, not a draft of it. The sizes are made first, over every purchase that may trade (those that trade on their own and those that wait for the user); only
then is the user asked, one purchase at a time, each card showing the final shares and cost and the whole proposal (`terminal_approver` prints it before the first question). A "no" leaves the other purchases at the
size the user saw: nothing is re-split after an answer, so what was approved is what the Trader gets. A purchase that sized to no share (`no_funds`) is not asked about.
"""
import re
from datetime import datetime
from decimal import Decimal

from prometheus.agents.account_manager import funding as funding_module, neutral, sizing
from prometheus.agents.account_manager.config import load_config as load_account_manager_config
from prometheus.agents.critic import feedback
from prometheus.agents.critic.common import utc_text
from prometheus.agents.critic.report import scorecard
from prometheus.common import shares as sh
from prometheus.common.market_calendar import eastern_date
from prometheus.common.money import digits, exact, price as price_door
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile
from prometheus.common.trading_clock import trading_now
from prometheus.common.trace import traced
from prometheus.common.work_order import WorkOrder, decision_of, parse_utc, review_of

MODES = ("ask", "auto")
BALANCES = ("even", "round_robin")      # what the user may ask for; with none asked it is round_robin, in both modes (entry 106)


def _aware(now: datetime | None) -> datetime:
    now = now or trading_now()   # the real clock, or the simulated one when USE_LIVE_TRADING=false (entry 76)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    return now


def _record(order: WorkOrder, record_id: str) -> dict:
    return next((r for r in order.recommendations if (r.get("identity") or {}).get("record_id") == record_id), {})


def _pending(order: WorkOrder) -> bool:
    return any(d["will_trade"] == "pending_user" for d in order.trade_decisions)


def _expired(review: dict, now: datetime) -> bool:
    until = parse_utc(review.get("valid_until_utc"))
    return until is None or now > until


OVER_LIMIT = "over your price limit: "      # how a note on the work order's data begins when a purchase was priced above the user's limit and got no decision (the Account Manager tells the user)


def over_limit_line(symbol: str, price: Decimal, limit: Decimal) -> str:
    """What the Account Manager tells the user about a purchase priced above their limit (entry 138, L4): it gets NO decision, like a neutral company beyond the number asked for."""
    return f"{symbol} is at ${price:,.2f} a share, above your limit of ${limit:,.2f}: it gets no decision and is not bought"


UNDER_FLOOR = "under your price floor: "      # how a note on the work order's data begins when a purchase was priced under the user's floor and got no decision (entry 140; the Account Manager tells the user)


def under_floor_line(symbol: str, price: Decimal, floor: Decimal) -> str:
    """What the Account Manager tells the user about a purchase priced under their price floor (entry 140, F2): it gets NO decision, like one above their limit."""
    return f"{symbol} is at ${price:,.2f} a share, under your minimum of ${floor:,.2f}: it gets no decision and is not bought"


def _outside(review: dict, record: dict, max_price: Decimal | None, min_price: Decimal | None = None) -> tuple[str, Decimal] | None:
    """("over", price) when the sizing price of this approved purchase (the Critic's deal price, else the plan's entry mid) is ABOVE the user's price limit, ("under", price) when it is UNDER their price floor (entry 140), else None (no limit and
    no floor, no price to compare, or inside them: at a bound is inside)."""
    if max_price is None and min_price is None:
        return None
    at = _purchase(review, record, None, 0).price
    if at is None:
        return None
    if max_price is not None and at > max_price:
        return "over", at
    if min_price is not None and at < min_price:
        return "under", at
    return None


def has_purchase_to_size(order: WorkOrder, now: datetime | None = None, max_price: Decimal | None = None, min_price: Decimal | None = None) -> bool:
    """Whether `decide_trades` would size a purchase on this order: it is waiting to be decided (status `reviewed`) and holds an approved, unexpired review of an `open` (a recommendation to buy) that is not outside the user's price limit
    and floor (`max_price`, entry 138; `min_price`, entry 140). The Account Manager asks for the prices of the holdings only then. Raises ValueError for a `now` with no timezone, as `decide_trades` does."""
    now = _aware(now)
    return order.status == "reviewed" and any(r.get("approved") and r.get("action") == "open" and not _expired(r, now) and _outside(r, _record(order, r["record_id"]), max_price, min_price) is None for r in order.reviews)


def purchases_to_size(order: WorkOrder, portfolio: Portfolio, now: datetime | None = None, max_price: Decimal | None = None, min_price: Decimal | None = None) -> list[sizing.Purchase]:
    """The purchases `decide_trades` would size, as the plain numbers the sizing takes, in the order of the reviews: each approved, unexpired `open` that is not priced outside the user's price limit and floor (`max_price`, entry 138;
    `min_price`, entry 140). The Account Manager's questions about how much (`clarify.py`) are worked out over these, so that they and the decision see the same prices, stops and ranks."""
    now = _aware(now)
    if order.status != "reviewed":
        return []
    return [_purchase(review, _record(order, review["record_id"]), portfolio, len(order.reviews)) for review in order.reviews
            if review.get("approved") and review.get("action") == "open" and not _expired(review, now) and _outside(review, _record(order, review["record_id"]), max_price, min_price) is None]


def company_of(order: WorkOrder, purchase: sizing.Purchase) -> str | None:
    """The company's name of an approved purchase, from the Advisor's record of the recommendation (None when the record has none)."""
    return _record(order, purchase.record_id).get("company")


def typed_wanted(rows: list[tuple[sizing.Purchase, str | None]], shares_of: dict | None, shares_each: Decimal | None = None) -> dict[str, Decimal]:
    """{record_id: shares} the user TYPED with a recommendation (entry 140, C2): `rows` are the purchases to size with their company's name; `shares_of` is {the stock as the user named it: shares} and `shares_each` the shares for every stock.
    A stock named by its ticker matches the purchase of that symbol; one named by its company's name matches the purchase whose company is the same company (`critic.names.same_company`). A purchase the user typed a number for gets exactly
    that as its limit (the sizing's `wanted`, binding `asked`); one they gave `shares_each` for gets that; any other gets nothing here. Only the purchases in `rows` are named, so the sizing never meets a record that is not a purchase."""
    from prometheus.agents.critic.names import same_company

    typed = {name: sh.quantity(n) for name, n in (shares_of or {}).items()}
    every = None if shares_each is None else sh.quantity(shares_each)
    out: dict[str, Decimal] = {}
    for purchase, company in rows:
        named = next((n for name, n in typed.items() if name.casefold() == purchase.symbol.casefold()), None)
        if named is None:
            named = next((n for name, n in typed.items() if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", name) and same_company(purchase.symbol, name, company)), None)      # a ticker names its symbol and nothing else
        if named is not None or every is not None:
            out[purchase.record_id] = named if named is not None else every
    return out


def combine_wanted(first: dict[str, Decimal] | None, second: dict[str, Decimal] | None) -> dict[str, Decimal] | None:
    """Two limits on the shares of the same purchase are both the user's: the smaller wins ("5 shares of MU" and "$3,000 for each": whichever buys fewer). None when neither has any."""
    merged = dict(first or {})
    for record, n in (second or {}).items():
        merged[record] = min(merged[record], n) if record in merged else n
    return merged or None


def balance_note(target: Decimal, count: int) -> str:
    return f"balanced evenly: about ${target:,.2f} for each of {count} purchases"


def _balance_text(decision: dict) -> str:
    if decision.get("balance"):
        return f" ({balance_note(decision['balance']['target_dollars'], decision['balance']['of'])})"
    return " (spent round-robin: one share of each purchase at a time, in rank order)" if (decision.get("size") or {}).get("allocation") == "round_robin" else ""


def order_phrase(decision: dict) -> str:
    """What a decision orders, in words (design choices entry 111, M2). An ORDER IN DOLLARS says so, dollars first and the shares as an ESTIMATE: "$10,000.00, about 63.163213 shares at $158.32". An order in shares says the
    shares and the estimated cost, as ever: "63.163213 shares, about $10,000.00". Every place that shows a decision or a proposal uses this."""
    if decision.get("dollars"):
        return f"${decision['dollars']:,.2f}, about {sh.text(decision['shares'])} shares" + (f" at ${decision['price']:,.2f}" if decision.get("price") else "")
    return f"{sh.text(decision['shares'])} shares" + (f", about ${decision['est_cost']:,.2f}" if decision.get("est_cost") else "")


def _card(review: dict, record: dict, decision: dict, n: int, of: int, proposal: list[dict], cash: Decimal) -> dict:
    """The Critic's scorecard of one review for the user, with what the user is really being asked: the FINAL shares and cost of the decision (the Account Manager's own size), where this question is among the others, the
    whole proposal and the cash. The Critic's card has no size; this shows what would be placed."""
    card = scorecard(review, record)
    buy = f"buy ${decision['dollars']:,.2f} of {decision['symbol']}, about {sh.text(decision['shares'])} shares" + (f" at ${decision['price']:,.2f}" if decision.get("price") else "") if decision.get("dollars") else \
        f"buy {sh.text(decision['shares'])} shares of {decision['symbol']}" + (f", about ${decision['est_cost']:,.2f}" if decision["est_cost"] else "")
    card.update(text=card["text"] + f"\n      WOULD BE PLACED  {buy}{_balance_text(decision)}", shares=decision["shares"], est_cost=decision["est_cost"], dollars=decision.get("dollars"), price=decision.get("price"), action=decision["action"], n=n, of=of, proposal=proposal, cash=cash)
    return card


def _funds(decision: dict, cost: Decimal, portfolio: Portfolio, prices: dict[str, Decimal], today) -> None:
    """Whether the money for a sized purchase is there: yes (the cash covers it), yes by RaiseFunds (holdings in profit could raise it: `sizing.funds_check`), or `no_funds`. The size is already cut by the cash
    (no cash is reserved: every purchase sees all of it), so the second and the third case are what a size that was given more money than the portfolio holds would meet."""
    cash = portfolio.cash_avail
    if cash >= cost:
        decision.update(funds_avail=True)
        return
    feasible, note = sizing.funds_check(cost, portfolio, prices, today)
    if feasible:
        decision.update(funds_avail=True, funds_note=f"needs RaiseFunds: the cash does not cover it, but holdings in profit could ({note})")
    else:
        decision.update(funds_avail=False, will_trade="no_funds", reason=f"cash ${cash:,.2f} does not cover ${cost:,.2f} and no holding in profit could raise the difference ({note})")


def _ask_the_user(order: WorkOrder, to_ask: list, approver, cash: Decimal) -> None:
    """Ask the user about each purchase that waits for them, ONE AT A TIME, showing the final size, an approved recommendation and a neutral one in the same proposal (entry 107). A purchase the even split dropped, or that the
    cash does not cover, is not asked about. No approver, or no answer: it stays `pending_user`. An answer is not followed by a new split: the others keep the size the user saw. Each row is `(decision, card, kept)`: `card(n, of,
    proposal)` makes its card and `kept` is the reason a neutral decision keeps beside the answer ("" for a review)."""
    asking = [row for row in to_ask if row[0]["will_trade"] == "pending_user"]
    proposal = [{"symbol": d["symbol"], "action": d["action"], "shares": d["shares"], "est_cost": d["est_cost"], "dollars": d.get("dollars"), "price": d.get("price")} for d, *_ in asking]
    for n, (decision, card, kept) in enumerate(asking, 1):
        decision["asked_user"] = approver is not None
        if approver is None:
            continue
        answer = None
        try:
            answer = approver(card(n, len(asking), proposal))
        except Exception as error:  # a prompt that failed: the user has not said yes
            order.errors = [*order.errors, f"the user could not be asked about {decision['symbol']} ({type(error).__name__})"]
        decision.update(will_trade="yes_user" if answer is True else "no_user_denied" if answer is False else "pending_user",
                        reason=("the user approved" if answer is True else "the user denied" if answer is False else "waiting for the user") + (f": {kept}" if kept else "") + _balance_text(decision))


def _purchase(review: dict, record: dict, portfolio: Portfolio, rank_after: int) -> sizing.Purchase:
    """One approved purchase as the plain numbers the sizing takes: the price (the Critic's deal price, else the plan's entry mid) and the rank of the recommendation. Nothing else: the funds are the only check (entry 91)."""
    trade, rank = record.get("trade") or {}, (record.get("ranking") or {}).get("rank")
    return sizing.Purchase(record_id=review["record_id"], symbol=review["symbol"], action=review["action"], price=_price_of(review["timing"].get("deal_price")) or _price_of(trade.get("entry_mid")),
                           rank=rank if isinstance(rank, int) and rank >= 0 else rank_after)


def _price_of(measured) -> Decimal | None:
    """The price the sizing works at, from the Critic's deal price or the Advisor's entry price (their measurements, floats once `json` has read them): through the price door, once (four places, half up), or None when it is not a number above zero."""
    try:
        return price_door(measured)
    except ValueError:
        return None


def _size(order: WorkOrder, buying: list[tuple[dict, sizing.Purchase]], portfolio: Portfolio, prices: dict[str, Decimal], allocation: str | None, budget: Decimal | None, now: datetime,
          wanted: dict[str, Decimal] | None = None, fractional: bool = False, funding: tuple[Decimal, str] | None = None) -> None:
    """Size every purchase in `buying` ((decision, purchase) rows; the decisions are changed in place) in ONE call of `sizing.size_purchases` (see the module text): the approved ones and the neutral ones TOGETHER (entry 107),
    in the order given: the shares, the cost, `size`, and for an even split `balance`. A purchase that sizes to NOTHING (no shares at all, not "less than one share": a fraction is a size, entry 110) becomes `no_funds`
    with the sizing's reason; one that is sized goes through the funds check. `funding` (entry 142, R10) is (the money the Account Manager is about to RAISE by selling holdings, how the user permitted it): the purchases are sized as if the account
    held it (the pot is the budget the user gave, else the cash plus that money: never more), each sized decision carries `funding` and the funds check is the Account Manager's own (it checked the holdings before it asked or went on)."""
    equity, basis = sizing.equity_of(portfolio, prices)
    raised = funding[0] if funding else Decimal("0.00")
    if funding and wanted:      # entry 143, P1: some purchases have typed shares and others do not. The cash and the money raised go to the typed ones, in full; a purchase with NO typed shares would be sized by the round robin to whatever money is left, and money RAISED by selling
        # must never reach it (it would take shares away from the ones the user typed), so it gets nothing and the user is told. The cash left after the typed ones is none: the shortfall is what they cost less the cash.
        typed_rows = [(d, p) for d, p in buying if p.record_id in wanted]
        for d, p in [(d, p) for d, p in buying if p.record_id not in wanted]:
            d["size"] = None
            d.update(will_trade="no_funds", funds_avail=False, reason=f"{funding_module.left_out_line([p.symbol])}")
        buying = typed_rows
        if not buying:
            return
    try:
        results = sizing.size_purchases([p for _, p in buying], portfolio.cash_avail + raised, allocation=allocation, budget=budget, wanted=wanted, fractional=fractional)
    except ValueError as error:  # only a program gets these wrong (two records with one id, a price that is not a number): a message on the order, and nothing is bought
        order.errors = [*order.errors, f"the purchases could not be sized: {error}"]
        results = None
    purchase_of = [p for _, p in buying]
    for n, (decision, _) in enumerate(buying):
        sized = results[n] if results else None
        decision["size"] = None if sized is None else {"shares": sized.shares, "caps": sized.caps, "binding": sized.binding, "equity": equity, "equity_basis": basis, "notes": sized.notes, "allocation": sized.allocation,
                                                       **({"asked": sized.asked} if sized.asked is not None else {}), **({"fractional": True} if fractional and sized.allocation == "even" else {}),
                                                       **({"pot": sized.pot} if sized.pot is not None else {})}      # the POT of the run: the handshake never approves a purchase past what is left of it (entry 111, M1)
        if sized is None or sized.shares <= 0:
            decision.update(will_trade="no_funds", funds_avail=False, reason=(sized.notes[-1] if sized and sized.notes else "the purchase could not be sized"))
            continue
        decision.update(shares=sized.shares, est_cost=sized.est_cost)
        if sized.dollars is not None:      # an ORDER IN DOLLARS (entry 111, M2): "at most $X of S"; `shares`, `est_cost` and `price` stay as the estimate at the price it was sized at
            decision.update(dollars=sized.dollars, price=purchase_of[n].price)
        if sized.allocation == "even":
            decision["balance"] = {"mode": "even", "target_dollars": sized.target_dollars, "of": sized.of}
        decision["reason"] += _balance_text(decision)      # " (balanced evenly: ...)" or " (spent round-robin: ...)", and nothing for the default
        if funding:      # to be paid for by selling holdings the user permitted selling (entry 142, R10): RaiseFunds runs before the Trader
            decision.update(funds_avail=True, funds_note=f"needs RaiseFunds: ${funding[0]:,.2f} is to be raised by selling holdings first", funding=funding_module.decision_funding(funding[0], funding[1]))
            continue
        _funds(decision, sized.est_cost, portfolio, prices, eastern_date(now))


@traced("account_manager.trade_decisions")
def decide_trades(order: WorkOrder, portfolio: Portfolio, profile: UserProfile, approver=None, now: datetime | None = None, mode: str | None = None,
                  balance: str | None = None, budget: Decimal | None = None, prices: dict[str, Decimal] | None = None, wanted: dict[str, Decimal] | None = None,
                  neutral_purchases: list[tuple[dict, sizing.Purchase]] | None = None, fractional: bool = False, max_price: Decimal | None = None, min_price: Decimal | None = None,
                  funding: tuple[Decimal, str] | None = None) -> WorkOrder:
    """Turn every review into a trade decision, and size it (see the module text); return the SAME order. In Auto mode an approved purchase is `yes_auto` and NOTHING is asked; in Ask mode `approver` is called with the
    scorecard (a dict with `text`) of each approved purchase and returns True, False, or None (no answer); without one the decision is `pending_user`. `mode` overrides the profile's `trade_mode` for this run. `balance` ("even" or "round_robin") says how the money is spread over the purchases, and `budget` is the funds
    requested (dollars; the cash when left out). With no `balance` the money is spent round-robin, in both modes (entry 106). `prices` is {symbol: the current price} of the symbols the portfolio HOLDS, for the
    equity and the funds check (a holding with no price counts at its cost basis): this function calls no API, the Account Manager fetches them. `wanted` ({record_id: whole shares}) is how many shares the user's own
    words came to for each purchase (entry 85: the Account Manager's questions about how much; the purchase never gets more). `neutral_purchases` are the companies the Advisor scored neutral that count as approved
    ((the Advisor's candidate record, its `sizing.Purchase`) for each, best score first, already limited to the number the user asked for: `neutral.choose`): they are decided here WITH the approved ones, in ONE sizing
    (entry 107), each decision `origin: "neutral"` and after the reviews', and an order with no review at all may carry them. `fractional` (entry 110) makes an even split a split in fractional shares, equal to the cent;
    it is False here unless the caller says so: the Account Manager (`AccountManager.decide_trades`) says it, from its switch `sizing.fractional_even_split` and from whether the account book can hold a fraction.
    `max_price` is the user's PRICE LIMIT ("under $50 a share", entry 138; None: none): an approved purchase whose sizing price is above it gets NO decision (the price moved since the Advisor looked; a note on the order's data says so, `OVER_LIMIT`),
    every other purchase's decision carries it (`decision["max_price"]`) and the handshake refuses a fill above it. A limit is a price, never a size and never a risk rule. `min_price` is the user's PRICE FLOOR ("over $20 a share", entry 140;
    None: none): the same from the other side, a purchase priced under it gets no decision (`UNDER_FLOOR`), every decision carries it (`decision["min_price"]`) and the handshake refuses a fill under it.
    `funding` (entry 142, R10) is (the money to RAISE by selling holdings, how the user permitted it) when the Account Manager decided that the shares the user typed with the recommendation are paid for by selling: the purchases are sized
    as if the account held that money, each sized decision carries `funding` and the desk runs RaiseFunds before the Trader. None: sized to the cash, as ever.
    Never raises for the user's side of things: a refusal is a message on the order."""
    now = _aware(now)
    if order.status in ("decided", "traded"):   # decided already; or decided and then traded by the Trader (Trader v1): neither is decided again
        return order
    if order.status == "recommended" and not order.recommendations and neutral_purchases:
        order.status = "reviewed"      # the Advisor recommended nothing, so the Critic had nothing to review: only the neutral companies are decided here
    if order.status != "reviewed":
        order.errors = [*order.errors, f"the Account Manager decides on an order the Critic reviewed (status {order.status})"]
        return order
    if profile.user != portfolio.user:
        order.errors = [*order.errors, f"the user profile is for {profile.user!r} but the portfolio belongs to {portfolio.user!r}: no trade is decided"]
        return order
    mode = mode or profile.trade_mode
    if mode not in MODES:
        order.errors = [*order.errors, f"mode must be one of {', '.join(MODES)}"]
        return order
    if balance is not None and balance not in BALANCES:
        order.errors = [*order.errors, f"balance must be one of {', '.join(BALANCES)} (or left out)"]
        return order
    if budget is not None:
        try:
            ok = exact(budget, "the budget") > 0          # a float is a TypeError: the budget comes in through a door (the user's own words: `intent.budget`, a Money)
        except ValueError:
            ok = False
        if not ok:
            order.errors = [*order.errors, "budget must be a number of dollars above zero (or left out)"]
            return order
    cash, decisions, to_ask, buying = portfolio.cash_avail, [], [], []      # `cash` is never reduced: a purchase may use any available funds
    for review in order.reviews:
        record = _record(order, review["record_id"])
        action = review["action"]
        if review["approved"] and action == "open" and not _expired(review, now) and (outside := _outside(review, record, max_price, min_price)) is not None:      # priced outside the user's limit or floor: no decision, the user is told (entries 138 and 140)
            if order.data is not None:
                note = OVER_LIMIT + over_limit_line(review["symbol"], outside[1], max_price) if outside[0] == "over" else UNDER_FLOOR + under_floor_line(review["symbol"], outside[1], min_price)
                order.data.notes = [*order.data.notes, note]
            continue
        decision = {"record_id": review["record_id"], "review_id": review["review_id"], "symbol": review["symbol"], "action": action, "shares": Decimal(0), "est_cost": Decimal("0.00"), "mode": mode,
                    "will_trade": None, "funds_avail": None, "funds_note": None, "asked_user": False, "reason": "", "decided_utc": utc_text(now), "size": None}
        if not review["approved"]:
            decision.update(will_trade="no_system_denied", reason=review["summary"])
        elif _expired(review, now):
            decision.update(will_trade="no_expired", reason=f"the approval ran out at {review.get('valid_until_utc')}: the price must be checked again")
        elif action != "open":
            decision.update(will_trade="no_funds", funds_avail=False, reason=f"a {action} recommendation is not a purchase and is never traded: the Advisor recommends purchases only, and a sale comes from the user's own order or RaiseFunds")
        else:
            buying.append((decision, _purchase(review, record, portfolio, len(order.reviews))))
            if max_price is not None:
                decision["max_price"] = max_price      # the handshake refuses a fill above it
            if min_price is not None:
                decision["min_price"] = min_price      # ... and a fill under the floor (entry 140)
            if mode == "auto":      # approved, Auto mode: it trades, and nobody is asked (entry 104)
                decision.update(will_trade="yes_auto", reason=f"approved by the Critic (score {review['score']}): Auto mode trades what the Critic approves")
            else:      # Ask mode: the user decides, but only about the FINAL size: it is asked after the sizing below (entry 79)
                decision.update(will_trade="pending_user", reason="waiting for the user")
                to_ask.append((decision, lambda n, of, proposal, review=review, record=record, decision=decision: _card(review, record, decision, n, of, proposal, cash), ""))
        decisions.append(decision)
    valid_minutes = load_account_manager_config().neutral.valid_minutes
    for candidate, purchase in neutral_purchases or []:      # a neutral recommendation is an approval (entry 107): decided beside the reviews' decisions, and sized with them
        decision = neutral.decision_for(candidate, purchase, mode, now, valid_minutes)
        if max_price is not None:
            decision["max_price"] = max_price
        if min_price is not None:
            decision["min_price"] = min_price
        decisions.append(decision)
        if decision["will_trade"] is not None:      # no price: `no_system_denied`, nothing to size
            continue
        buying.append((decision, purchase))
        if mode == "auto":
            decision.update(will_trade="yes_auto")
        else:
            decision.update(will_trade="pending_user")
            to_ask.append((decision, lambda n, of, proposal, candidate=candidate, decision=decision: {**neutral.confirmation_card(candidate, decision, cash), "n": n, "of": of, "proposal": proposal, "cash": cash}, decision["reason"]))
    if buying:      # over what trades on its own AND what waits for the user, so that the user is shown the size that will be placed
        _size(order, buying, portfolio, prices or {}, balance or "round_robin", budget, now, wanted, fractional, funding)      # ONE rule for every run (entries 106 and 109): round robin unless the user's own words asked for an even split
    if funding:      # what the sized purchases really cost less the cash is the shortfall (the budget or a rounding may have left less than the typed shares need): none left, and nothing is raised (entry 142, R10)
        funded = [d for d, _ in buying if isinstance(d.get("funding"), dict)]
        short = sum((d["est_cost"] for d in funded), Decimal("0.00")) - portfolio.cash_avail
        for d in funded:
            if short > 0:
                d["funding"]["shortfall"] = short
            else:
                d.pop("funding")
                d["funds_note"] = None
    order.trade_decisions = decisions
    _ask_the_user(order, to_ask, approver, cash)
    feedback.attach_decisions(order)
    if not _pending(order):
        order.status = "decided"
    return order


def denied_lines(order: WorkOrder) -> list[str]:
    """The recommendations the Critic DENIED, as the Account Manager tells the user right after it decides (design choices entry 104; Jacob: *"the denied recommendations need to be returned to the user from the
    AccountManager"*): each with the Critic's score, grade and own reason (its summary sentence without the "DENY X:" that begins it: the blockers, the groups, or the person who must look). A denial for bad signs in two (or
    three) groups says so (Critic v3, `denial_reasons` holds `group:news`, `group:performance`: "two reasons against it") before the groups in words, so the user sees why one reason was not enough to deny and two were. Plain text
    from the reviews; nothing is traded for these, and none is asked about. Empty when the Critic denied nothing."""
    rows = [r for r in order.reviews if not r["approved"]]
    if not rows:
        return []
    lines = ["      DENIED by the Critic, not traded:"]
    for r in rows:
        why = r["summary"].split(": ", 1)[1] if r["summary"].startswith("DENY ") and ": " in r["summary"] else r["summary"]
        groups = [reason for reason in r.get("denial_reasons") or [] if str(reason).startswith("group:")]
        said = f"{'two' if len(groups) == 2 else 'three'} reasons against it, " if len(groups) >= 2 else ""
        lines.append(f"        {r['symbol']:<6} score {r['score']} ({r['grade']}): {said}{why}")
    return lines


def record_user_decision(order: WorkOrder, record_id: str, approve: bool, now: datetime | None = None) -> WorkOrder:
    """The user's answer to a decision that was `pending_user`: yes (`yes_user`) or no (`no_user_denied`), or `no_expired` when the approval has run out by now. When no decision is pending any more
    the order is `decided`. A recommendation with no pending decision is a message on the order, not a change."""
    now = _aware(now)
    decision, review = decision_of(order, record_id), review_of(order, record_id)
    if decision is None or review is None or decision["will_trade"] != "pending_user":
        order.errors = [*order.errors, f"there is no pending decision for recommendation {record_id}"]
        return order
    if _expired(review, now):
        decision.update(will_trade="no_expired", reason=f"the approval ran out at {review.get('valid_until_utc')}: the price must be checked again")
    else:
        decision.update(will_trade="yes_user" if approve else "no_user_denied", reason="the user approved" if approve else "the user denied")
    decision.update(asked_user=True, decided_utc=utc_text(now))
    feedback.attach_decisions(order)
    if not _pending(order):
        order.status = "decided"
    return order


def _trade_words(card: dict) -> str:
    """What the question names: an order in dollars says its dollars (entry 111, M2), an order in shares its shares and the cost."""
    if card.get("dollars"):
        return f"${card['dollars']:,.0f} of {card['symbol']}"
    return f"{sh.text(card['shares'])} shares of {card['symbol']}, about ${card['est_cost'] or 0:,.0f}"


def proposal_lines(card: dict) -> list[str]:
    """What the Account Manager proposes, all of it, before the first question (the card's `proposal`): so the user sees the whole order, and the cash, and not one trade at a time."""
    rows = card["proposal"]
    spend = sum((digits(r["est_cost"] or 0) for r in rows), Decimal("0.00"))
    lines = ["", "=" * 118, f"ASK MODE  nothing is traded without your yes. The Account Manager proposes {len(rows)} trade{'s' if len(rows) != 1 else ''}:", "=" * 118]
    for r in rows:
        lines.append(f"  {r['symbol']:<6} buy {order_phrase({**r, 'est_cost': r['est_cost'] or 0})}")
    if spend:
        lines.append(f"  about ${spend:,.2f} of ${card['cash']:,.2f} in cash. You are asked about each one below; a yes places exactly what is shown.")
    return lines


def terminal_approver(ask=input, show=print, interactive: bool | None = None):
    """An approver that asks the person at the terminal (Ask mode): the first card of a proposal prints the whole proposal; each card prints the scorecard and asks "Place this trade? [y/N]" (which says "n of N" when
    there are several), or the card's own `question` when it has one (the Trader's funding card, Trader v1: "Sell these to pay for ...?"). Nobody there (`interactive` False; by default `sys.stdin.isatty()`) means
    it gives no answer (None, so the decision stays `pending_user`): a script that is not being watched never trades, and an answer that is not a clear yes is a no."""
    import sys

    def here() -> bool:
        return sys.stdin.isatty() if interactive is None else interactive

    def approver(card: dict) -> bool | None:
        if not here():
            return None
        if card.get("n") == 1 and card.get("proposal"):
            show("\n".join(proposal_lines(card)))
        show("\n" + card["text"])
        which = f" {card['n']} of {card['of']}" if card.get("of", 1) > 1 else ""
        try:
            answer = ask(card.get("question") or f"Place this trade{which} ({_trade_words(card)})? [y/N] ")
        except EOFError:
            return None
        return answer.strip().lower() in ("y", "yes")

    approver.available = here      # nobody at the keyboard: a caller that would first spend a model call on wording a question (entry 102) can ask first, as it can of the how-much channel
    return approver
