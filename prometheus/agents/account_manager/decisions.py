"""What the desk does with the Critic's verdicts: the Account Manager's trade decisions (Critic v1 spec section 12.2, design choices entry 56).

    order = decide_trades(order, portfolio, profile, auto_min_score, approver, now)        # status "reviewed" -> "decided" (or stays "reviewed" while a user's answer is pending)

For each review, in order, the FIRST row that fits (`will_trade` is the week 5 design's WillTrade):

    not approved                                              no_system_denied   final: the user is never asked to override a denial
    approved, but `now` is after `valid_until_utc`            no_expired         the price check is old
    approved, but the money is not there                      no_funds           (a sale needs none)
    approved, Auto mode, score >= the mark, nobody needs to look   yes_auto
    anything else (Ask mode, or Auto escalated to the user)   the user's answer: yes_user, no_user_denied, or pending_user (nobody to ask)

In Auto mode a recommendation that approved at 70 to 79, one that carries a finding that asks for a person (`requires_human`), and any SALE, still goes to the user (`escalated: true`): the Advisor sized
no sale, so a program must not pick one. The cash is counted across the review: two approved purchases cannot both spend the same dollars.

`work_order.verify_trade` is what the Trader must pass before any trade; it reads these decisions. Nothing here trades, and nothing here calls an API or a model.
"""
from datetime import datetime, timezone

from prometheus.agents.critic import feedback
from prometheus.agents.critic.common import utc_text
from prometheus.agents.critic.report import scorecard
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile
from prometheus.common.work_order import WorkOrder, decision_of, parse_utc, review_of

MODES = ("ask", "auto")


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
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


def _why_user(review: dict, action: str, mark: int) -> str | None:
    """Why an APPROVED recommendation goes to the user even in Auto mode; None when nobody needs to look."""
    if action == "reduce":
        return "a sale is never traded automatically (the Advisor sized none)"
    if review["score"] < mark:
        return f"the score {review['score']} is under the automatic-trading mark {mark}"
    if review["requires_human"]:
        return "the review asks for a person to look: " + ", ".join(review["requires_human_reasons"])
    return None


def decide_trades(order: WorkOrder, portfolio: Portfolio, profile: UserProfile, auto_min_score: int, approver=None, now: datetime | None = None, mode: str | None = None) -> WorkOrder:
    """Turn every review into a trade decision (see the module text) and return the SAME order. `auto_min_score` is the Critic's own mark for trading without asking; the profile's
    `auto_min_score` replaces it when it has one. `approver` is called with the scorecard (a dict with `text`) of each recommendation that needs the user and returns True, False, or None (no answer);
    without one the decision is `pending_user`. `mode` overrides the profile's `trade_mode` for this run. Never raises for the user's side of things: a refusal is a message on the order."""
    now = _aware(now)
    if order.status == "decided":
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
    mark = profile.auto_min_score if profile.auto_min_score is not None else auto_min_score
    cash, decisions = portfolio.cash_avail, []
    for review in order.reviews:
        record = _record(order, review["record_id"])
        action, shares = review["action"], review["approved_shares"]
        price = review["timing"].get("deal_price") or (record.get("trade") or {}).get("entry_mid")
        cost = round(shares * price, 2) if price and action != "reduce" else 0.0
        decision = {"record_id": review["record_id"], "review_id": review["review_id"], "symbol": review["symbol"], "action": action, "shares": shares, "est_cost": cost, "mode": mode,
                    "will_trade": None, "funds_avail": None, "funds_note": None, "asked_user": False, "escalated": False, "reason": "", "decided_utc": utc_text(now)}
        if not review["approved"]:
            decision.update(will_trade="no_system_denied", reason=review["summary"])
        elif _expired(review, now):
            decision.update(will_trade="no_expired", reason=f"the approval ran out at {review.get('valid_until_utc')}: the price must be checked again")
        else:
            if action == "reduce":
                decision.update(funds_avail=True)
            elif cash >= cost:
                decision.update(funds_avail=True)
            elif (record.get("trade") or {}).get("funds_feasible"):
                decision.update(funds_avail=True, funds_note="needs RaiseFunds: the cash does not cover it, but holdings in profit could")
            else:
                decision.update(funds_avail=False, will_trade="no_funds", reason=f"cash ${cash:,.2f} does not cover ${cost:,.2f} and no holding in profit could raise the difference")
            if decision["will_trade"] is None:
                why_user = _why_user(review, action, mark)
                if mode == "auto" and why_user is None:
                    decision.update(will_trade="yes_auto", reason=f"the score {review['score']} is at or above the automatic-trading mark {mark}")
                else:
                    decision.update(escalated=mode == "auto", asked_user=approver is not None)
                    answer = None
                    if approver is not None:
                        try:
                            answer = approver(scorecard(review, record))
                        except Exception as error:  # a prompt that failed: the user has not said yes
                            order.errors = [*order.errors, f"the user could not be asked about {review['symbol']} ({type(error).__name__})"]
                    decision.update(will_trade="yes_user" if answer is True else "no_user_denied" if answer is False else "pending_user",
                                    reason=(f"{why_user}: " if why_user else "") + ("the user approved" if answer is True else "the user denied" if answer is False else "waiting for the user"))
            if decision["will_trade"] in ("yes_auto", "yes_user", "pending_user") and cost and cash >= cost:
                cash -= cost       # reserved: a later purchase of the same review cannot spend it again
        decisions.append(decision)
    order.trade_decisions = decisions
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


def terminal_approver(ask=input, show=print, interactive: bool | None = None):
    """An approver that asks the person at the terminal (Ask mode): it shows the scorecard and asks "Place this trade? [y/N]". Nobody there (`interactive` False; by default `sys.stdin.isatty()`) means
    it gives no answer (None, so the decision stays `pending_user`): a script that is not being watched never trades, and an answer that is not a clear yes is a no."""
    import sys

    def approver(card: dict) -> bool | None:
        if not (sys.stdin.isatty() if interactive is None else interactive):
            return None
        show("\n" + card["text"])
        try:
            answer = ask(f"Place this trade ({card['shares']} shares of {card['symbol']}, about ${card['est_cost'] or 0:,.0f})? [y/N] ")
        except EOFError:
            return None
        return answer.strip().lower() in ("y", "yes")

    return approver
