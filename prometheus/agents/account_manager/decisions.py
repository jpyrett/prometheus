"""What the desk does with the Critic's verdicts: the Account Manager's trade decisions, and HOW MUCH to buy or sell (Critic v1 spec section 12.2, design choices entries 56, 79 and 80).

    order = decide_trades(order, portfolio, profile, auto_min_score, approver, now, mode, balance, budget, prices)        # status "reviewed" -> "decided" (or stays "reviewed" while a user's answer is pending); a "decided" or "traded" order comes back unchanged

The Advisor writes recommendations and the Critic approves or denies each one; NEITHER decides how much to buy or sell. That is decided here, from the user's money, the user's limits and what the user said
(`sizing.size_purchases`: the position, loss, order and liquidity limits, the exposure room, the cash and the budget). For each review, in order, the FIRST row that fits (`will_trade` is the week 5 design's WillTrade):

    not approved                                              no_system_denied   final: the user is never asked to override a denial
    approved, but `now` is after `valid_until_utc`            no_expired         the price check is old
    approved, but not even one share fits (or the money is not there)   no_funds           (a sale needs none)
    approved, Auto mode, score >= the mark, nobody needs to look   yes_auto
    anything else (Ask mode, or Auto escalated to the user)   the user's answer: yes_user, no_user_denied, or pending_user (nobody to ask)

In Auto mode a recommendation that approved at 70 to 79, one that carries a finding that asks for a person (`requires_human`), and any SALE, still goes to the user (`escalated: true`): a sale is never automatic, and it
is the WHOLE HOLDING of the symbol (from the portfolio). NO CASH IS RESERVED (Jacob, 2026-11-14): each purchase may use all the funds available in the portfolio, whatever the other purchases cost; the Trader checks the
cash again when it places each order.

THE SIZE (entry 80). After the rows above, EVERY approved, unexpired purchase (an `open` or an `add`: those that will trade on their own and those that wait for the user) is sized in ONE call of `sizing.size_purchases`,
so that the user is shown the FINAL shares and cost. The price is the Critic's deal price (else the plan's entry mid); the stop distance and the rank come from the Advisor's record; the average daily dollars from its
risk numbers; the shares already held from the PORTFOLIO (the Account Manager's own knowledge); the equity from the portfolio and `prices` (the current prices of the symbols it holds: `sizing.equity_of`). The
ALLOCATION, that is how the money is spread over the purchases:

    `balance="even"`          equal DOLLARS for every purchase (the user said "split it evenly"); each decision carries `balance` {mode, target_dollars, of}
    `balance="round_robin"`   one share of each purchase at a time, in rank order, until the funds are used (what Auto mode does when nothing was asked)
    no `balance`, Auto mode   round robin: "take the available funds, buy a share of the first stock, a share of the next, until the requested funds are used up" (Jacob, 2026-11-14)
    no `balance`, Ask mode    as much as the user's limits allow, in rank order (the user sees the largest order and decides)

`budget` is the funds requested (dollars; the cash when left out, and never more than the cash): the pot of the round robin and of the even split. A purchase that sizes to no share becomes `no_funds` with the sizing's reason
and is not asked about. Each sized purchase carries `decision["size"]`: shares, the caps of every limit, `binding` (what decided the number), the equity and what it was worked out from, the notes, and the allocation.
`sizing.funds_check` then asks whether the cost could exist (cash, or RaiseFunds from holdings in profit).

The Trader (Trader v1, `prometheus/agents/trader/`) reads these decisions: it makes one ticket for each that says `yes_auto` or `yes_user` and nothing for any other, and every trade it places passes
`work_order.verify_trade`, which reads the decision again. So a `pending_user` is never traded; the user's later answer (`record_user_decision`) makes it a `yes_user` that a later call of the Trader carries on from.
The Trader also asks the user, through the same `approver`, whether the sells that pay for a purchase may be made (`terminal_approver` shows the card's own `question`): a sale is never automatic, in Auto mode too. This module itself
trades nothing, and calls no API or model: the prices are given.

HOW MUCH, FROM THE USER (entry 85). In Ask mode the Account Manager first asks the user how much to buy when the request did not say (`clarify.py`, a bounded ReAct loop): what comes out is `balance`, `budget`, `wanted` (the
shares the user's words came to for each purchase) and `limit_overrides` (a limit the user raised for this purchase), and this module sizes under them like any other request: the purchase never gets more than `wanted`.

ASK (entry 79): the user approves the order that WILL be placed, not a draft of it. The sizes are made first, over every purchase that may trade (those that trade on their own and those that wait for the user); only
then is the user asked, one purchase at a time, each card showing the final shares and cost and the whole proposal (`terminal_approver` prints it before the first question). A "no" leaves the other purchases at the
size the user saw: nothing is re-split after an answer, so what was approved is what the Trader gets. A purchase that sized to no share (`no_funds`) is not asked about.
"""
import math
from datetime import datetime

from prometheus.agents.account_manager import sizing
from prometheus.agents.critic import feedback
from prometheus.agents.critic.common import utc_text
from prometheus.agents.critic.report import scorecard
from prometheus.common.market_calendar import eastern_date
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile
from prometheus.common.trading_clock import trading_now
from prometheus.common.work_order import WorkOrder, decision_of, parse_utc, review_of

MODES = ("ask", "auto")
BALANCES = ("even", "round_robin")      # what the user may ask for; the third way (as much as the limits allow) is what Ask mode does when nothing is asked


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


def has_purchase_to_size(order: WorkOrder, now: datetime | None = None) -> bool:
    """Whether `decide_trades` would size a purchase on this order: it is waiting to be decided (status `reviewed`) and holds an approved, unexpired review of an `open` or an `add`. The Account Manager asks for the
    prices of the holdings only then. Raises ValueError for a `now` with no timezone, as `decide_trades` does."""
    now = _aware(now)
    return order.status == "reviewed" and any(r.get("approved") and r.get("action") in ("open", "add") and not _expired(r, now) for r in order.reviews)


def purchases_to_size(order: WorkOrder, portfolio: Portfolio, now: datetime | None = None) -> list[sizing.Purchase]:
    """The purchases `decide_trades` would size, as the plain numbers the sizing takes, in the order of the reviews: each approved, unexpired `open` or `add`. The Account Manager's questions about how much (`clarify.py`)
    are worked out over these, so that they and the decision see the same prices, stops and ranks."""
    now = _aware(now)
    if order.status != "reviewed":
        return []
    return [_purchase(review, _record(order, review["record_id"]), portfolio, len(order.reviews)) for review in order.reviews
            if review.get("approved") and review.get("action") in ("open", "add") and not _expired(review, now)]


def _why_user(review: dict, action: str, mark: int) -> str | None:
    """Why an APPROVED recommendation goes to the user even in Auto mode; None when nobody needs to look."""
    if action == "reduce":
        return "a sale is never traded automatically (it is the whole holding, and the user decides)"
    if review["score"] < mark:
        return f"the score {review['score']} is under the automatic-trading mark {mark}"
    if review["requires_human"]:
        return "the review asks for a person to look: " + ", ".join(review["requires_human_reasons"])
    return None


def balance_note(target: float, count: int) -> str:
    return f"balanced evenly: about ${target:,.2f} for each of {count} purchases"


def _balance_text(decision: dict) -> str:
    if decision.get("balance"):
        return f" ({balance_note(decision['balance']['target_dollars'], decision['balance']['of'])})"
    return " (spent round-robin: one share of each purchase at a time, in rank order)" if (decision.get("size") or {}).get("allocation") == "round_robin" else ""


def _card(review: dict, record: dict, decision: dict, n: int, of: int, proposal: list[dict], cash: float) -> dict:
    """The Critic's scorecard of one review for the user, with what the user is really being asked: the FINAL shares and cost of the decision (the Account Manager's own size), where this question is among the others, the
    whole proposal and the cash. The Critic's card has no size; this shows what would be placed."""
    card = scorecard(review, record)
    buy = f"buy {decision['shares']} shares of {decision['symbol']}" + (f", about ${decision['est_cost']:,.2f}" if decision["est_cost"] else "")
    what = f"sell the whole holding of {decision['symbol']} ({decision['shares']} shares)" if decision["action"] == "reduce" else buy
    card.update(text=card["text"] + f"\n      WOULD BE PLACED  {what}{_balance_text(decision)}", shares=decision["shares"], est_cost=decision["est_cost"], action=decision["action"], n=n, of=of, proposal=proposal, cash=cash)
    return card


def _funds(decision: dict, cost: float, portfolio: Portfolio, prices: dict[str, float], today) -> None:
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
        decision.update(funds_avail=False, will_trade="no_funds", escalated=False, reason=f"cash ${cash:,.2f} does not cover ${cost:,.2f} and no holding in profit could raise the difference ({note})")


def _ask_the_user(order: WorkOrder, to_ask: list, approver, cash: float) -> None:
    """Ask the user about each purchase that waits for them, ONE AT A TIME, showing the final size. A purchase the even split dropped, or that the cash does not cover, is not asked about. No approver, or no
    answer: it stays `pending_user`. An answer is not followed by a new split: the others keep the size the user saw. No cash is reserved by an answer either: each purchase was checked against all the cash."""
    asking = [row for row in to_ask if row[0]["will_trade"] == "pending_user"]
    proposal = [{"symbol": d["symbol"], "action": d["action"], "shares": d["shares"], "est_cost": d["est_cost"]} for d, *_ in asking]
    for n, (decision, review, record, why_user) in enumerate(asking, 1):
        decision["asked_user"] = approver is not None
        if approver is None:
            continue
        answer = None
        try:
            answer = approver(_card(review, record, decision, n, len(asking), proposal, cash))
        except Exception as error:  # a prompt that failed: the user has not said yes
            order.errors = [*order.errors, f"the user could not be asked about {review['symbol']} ({type(error).__name__})"]
        decision.update(will_trade="yes_user" if answer is True else "no_user_denied" if answer is False else "pending_user",
                        reason=(f"{why_user}: " if why_user else "") + ("the user approved" if answer is True else "the user denied" if answer is False else "waiting for the user") + _balance_text(decision))


def _purchase(review: dict, record: dict, portfolio: Portfolio, rank_after: int) -> sizing.Purchase:
    """One approved purchase as the plain numbers the sizing takes: the price, the plan's stop distance, the Advisor's average daily dollars and its rank come from the record and the Critic's review; the shares already
    held come from the PORTFOLIO (the Account Manager's own knowledge, never the record's)."""
    trade, rank = record.get("trade") or {}, (record.get("ranking") or {}).get("rank")
    held = portfolio.position(review["symbol"])
    return sizing.Purchase(record_id=review["record_id"], symbol=review["symbol"], action=review["action"], price=review["timing"].get("deal_price") or trade.get("entry_mid"), stop_distance=trade.get("stop_distance"),
                           held_shares=held.shares if held else 0, adv_20d_dollars=(record.get("risk") or {}).get("adv_20d_dollars"), rank=rank if isinstance(rank, int) and rank >= 0 else rank_after)


def _size(order: WorkOrder, buying: list[dict], context: dict, portfolio: Portfolio, profile: UserProfile, prices: dict[str, float], allocation: str | None, budget: float | None, now: datetime,
          wanted: dict[str, int] | None = None, limit_overrides: dict | None = None) -> None:
    """Size every purchase in `buying` (decisions, changed in place) in ONE call of `sizing.size_purchases` (see the module text): the shares, the cost, `size`, and for an even split `balance`. A purchase that sizes to no share
    becomes `no_funds` with the sizing's reason; one that is sized goes through the funds check."""
    equity, basis = sizing.equity_of(portfolio, prices)
    try:
        results = sizing.size_purchases([_purchase(context[d["record_id"]][0], context[d["record_id"]][1], portfolio, len(order.reviews)) for d in buying], equity, portfolio.cash_avail, profile.sizing_limits(),
                                        allocation=allocation, budget=budget, wanted=wanted, limit_overrides=limit_overrides)
    except ValueError as error:  # only a program gets these wrong (two records with one id, a price that is not a number): a message on the order, and nothing is bought
        order.errors = [*order.errors, f"the purchases could not be sized: {error}"]
        results = None
    for n, decision in enumerate(buying):
        sized = results[n] if results else None
        decision["size"] = None if sized is None else {"shares": sized.shares, "caps": sized.caps, "binding": sized.binding, "equity": equity, "equity_basis": basis, "notes": sized.notes, "allocation": sized.allocation,
                                                       **({"asked": sized.asked} if sized.asked is not None else {})}
        if sized is None or sized.shares < 1:
            decision.update(will_trade="no_funds", funds_avail=False, escalated=False, reason=(sized.notes[-1] if sized and sized.notes else "the purchase could not be sized"))
            continue
        decision.update(shares=sized.shares, est_cost=sized.est_cost)
        if sized.allocation == "even":
            decision["balance"] = {"mode": "even", "target_dollars": sized.target_dollars, "of": sized.of}
        decision["reason"] += _balance_text(decision)      # " (balanced evenly: ...)" or " (spent round-robin: ...)", and nothing for the default
        _funds(decision, sized.est_cost, portfolio, prices, eastern_date(now))


def decide_trades(order: WorkOrder, portfolio: Portfolio, profile: UserProfile, auto_min_score: int, approver=None, now: datetime | None = None, mode: str | None = None,
                  balance: str | None = None, budget: float | None = None, prices: dict[str, float] | None = None, wanted: dict[str, int] | None = None,
                  limit_overrides: dict | None = None) -> WorkOrder:
    """Turn every review into a trade decision, and size it (see the module text); return the SAME order. `auto_min_score` is the Critic's own mark for trading without asking; the profile's
    `auto_min_score` replaces it when it has one. `approver` is called with the scorecard (a dict with `text`) of each recommendation that needs the user and returns True, False, or None (no answer);
    without one the decision is `pending_user`. `mode` overrides the profile's `trade_mode` for this run. `balance` ("even" or "round_robin") says how the money is spread over the purchases, and `budget` is the funds
    requested (dollars; the cash when left out). With no `balance` Auto mode spends round-robin and Ask mode gives as much as the limits allow. `prices` is {symbol: the current price} of the symbols the portfolio HOLDS, for the
    equity and the funds check (a holding with no price counts at its cost basis): this function calls no API, the Account Manager fetches them. `wanted` ({record_id: whole shares}) is how many shares the user's own
    words came to for each purchase (entry 85: the Account Manager's questions about how much; the purchase never gets more) and `limit_overrides` the limits the user raised for this call (`sizing.size_purchases`).
    Never raises for the user's side of things: a refusal is a message on the order."""
    now = _aware(now)
    if order.status in ("decided", "traded"):   # decided already; or decided and then traded by the Trader (Trader v1): neither is decided again
        return order
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
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0):
        order.errors = [*order.errors, "budget must be a number of dollars above zero (or left out)"]
        return order
    mark = profile.auto_min_score if profile.auto_min_score is not None else auto_min_score
    cash, decisions, to_ask, context, buying = portfolio.cash_avail, [], [], {}, []      # `cash` is never reduced: a purchase may use any available funds
    for review in order.reviews:
        record = _record(order, review["record_id"])
        action = review["action"]
        decision = {"record_id": review["record_id"], "review_id": review["review_id"], "symbol": review["symbol"], "action": action, "shares": 0, "est_cost": 0.0, "mode": mode,
                    "will_trade": None, "funds_avail": None, "funds_note": None, "asked_user": False, "escalated": False, "reason": "", "decided_utc": utc_text(now), "size": None}
        context[review["record_id"]] = (review, record)
        if not review["approved"]:
            decision.update(will_trade="no_system_denied", reason=review["summary"])
        elif _expired(review, now):
            decision.update(will_trade="no_expired", reason=f"the approval ran out at {review.get('valid_until_utc')}: the price must be checked again")
        elif action not in ("open", "add", "reduce"):
            decision.update(will_trade="no_funds", funds_avail=False, reason=f"a {action} recommendation has nothing to trade")
        elif action == "reduce" and not portfolio.position(review["symbol"]):
            decision.update(will_trade="no_funds", funds_avail=False, reason=f"the portfolio holds no {review['symbol']}: there is nothing to sell")
        else:
            if action == "reduce":      # a sale is the whole holding of the symbol (the Account Manager's own knowledge) and needs no money
                decision.update(shares=portfolio.position(review["symbol"]).shares, funds_avail=True)
            else:
                buying.append(decision)
            why_user = _why_user(review, action, mark)
            if mode == "auto" and why_user is None:
                decision.update(will_trade="yes_auto", reason=f"the score {review['score']} is at or above the automatic-trading mark {mark}")
            else:      # the user decides, but only about the FINAL size: it is asked after the sizing below (entry 79)
                decision.update(will_trade="pending_user", escalated=mode == "auto", reason=(f"{why_user}: " if why_user else "") + "waiting for the user")
                to_ask.append((decision, review, record, why_user))
        decisions.append(decision)
    if buying:      # over what trades on its own AND what waits for the user, so that the user is shown the size that will be placed
        _size(order, buying, context, portfolio, profile, prices or {}, balance or ("round_robin" if mode == "auto" else None), budget, now, wanted, limit_overrides)
    order.trade_decisions = decisions
    _ask_the_user(order, to_ask, approver, cash)
    feedback.attach_decisions(order)
    if not _pending(order):
        order.status = "decided"
    return order


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


def proposal_lines(card: dict) -> list[str]:
    """What the Account Manager proposes, all of it, before the first question (the card's `proposal`): so the user sees the whole order, and the cash, and not one trade at a time."""
    rows = card["proposal"]
    spend = sum(r["est_cost"] or 0 for r in rows if r["action"] != "reduce")
    lines = ["", "=" * 118, f"ASK MODE  nothing is traded without your yes. The Account Manager proposes {len(rows)} trade{'s' if len(rows) != 1 else ''}:", "=" * 118]
    for r in rows:
        lines.append(f"  {r['symbol']:<6} " + (f"sell the whole holding ({r['shares']} shares)" if r["action"] == "reduce" else f"{'add' if r['action'] == 'add' else 'buy'} {r['shares']} shares, about ${r['est_cost'] or 0:,.2f}"))
    if spend:
        lines.append(f"  about ${spend:,.2f} of ${card['cash']:,.2f} in cash. You are asked about each one below; a yes places exactly what is shown.")
    return lines


def terminal_approver(ask=input, show=print, interactive: bool | None = None):
    """An approver that asks the person at the terminal (Ask mode): the first card of a proposal prints the whole proposal; each card prints the scorecard and asks "Place this trade? [y/N]" (which says "n of N" when
    there are several), or the card's own `question` when it has one (the Trader's funding card, Trader v1: "Sell these to pay for ...?"). Nobody there (`interactive` False; by default `sys.stdin.isatty()`) means
    it gives no answer (None, so the decision stays `pending_user`): a script that is not being watched never trades, and an answer that is not a clear yes is a no."""
    import sys

    def approver(card: dict) -> bool | None:
        if not (sys.stdin.isatty() if interactive is None else interactive):
            return None
        if card.get("n") == 1 and card.get("proposal"):
            show("\n".join(proposal_lines(card)))
        show("\n" + card["text"])
        which = f" {card['n']} of {card['of']}" if card.get("of", 1) > 1 else ""
        try:
            answer = ask(card.get("question") or f"Place this trade{which} ({card['shares']} shares of {card['symbol']}, about ${card['est_cost'] or 0:,.0f})? [y/N] ")
        except EOFError:
            return None
        return answer.strip().lower() in ("y", "yes")

    return approver
