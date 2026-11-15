"""A NEUTRAL RECOMMENDATION IS APPROVED (design choices entry 91; Jacob, 2026-11-15: *"A neutral recommendation is same as Approved"*, with `ALLOW_NEUTRAL_RECOMMENDATIONS=true` in `.env`).

    names = neutral_candidates(order)                                                              # the companies the user named that the Advisor scored NEUTRAL; empty unless the setting is on
    decide_neutral(order, portfolio, names, mode, prices, now, valid_minutes, approver, budget)    # one trade decision for each, appended to `order.trade_decisions`

THE SETTING. `ALLOW_NEUTRAL_RECOMMENDATIONS` (the environment, then `.env`; true, yes, on or 1; unset or anything else is false: nothing changes for anyone who does not set it) is read by the ACCOUNT MANAGER when it works with
the work order's recommendation result (`AccountManager.decide_trades`). When it is true, a company that the user NAMED and that the Advisor scored NEUTRAL (no strategy passed, or no edge: it is on `order.candidates` with `stance`
"neutral", no failed gate and not already held, `work_order.is_neutral_candidate`) counts as approved, and the Account Manager gives the Trader permission to buy it: a trade decision with `origin: "neutral"`, which
`work_order.verify_trade` accepts only while the work order holds that Advisor record. A neutral company has no entry plan, no stop and no Critic review, so the permission is the Advisor's own neutral score, nothing is
forged (no review is made up), and the trade is a MARKET purchase at the quote the Account Manager asked of the Data Clerk, like a direct order.

WHAT STAYS. A BEARISH stance is not a recommendation and trades nothing. A Critic DENIAL is final: a bullish company whose review denied it (risk above the profile, a falling knife) is never turned into an approval by this
setting, because only a company the Advisor did NOT recommend can be neutral. A gate that failed (stale data, a red flag) is not a stance and stops the company. Only companies the user named count (a theme is not bought
blindly: a company the Advisor discovered is never bought as neutral).

HOW MUCH (Jacob, 2026-11-15: no profile limit applies, split evenly). The only check is that the funds cover the order. The neutral companies share the money in EQUAL DOLLARS, in whole shares (`sizing.size_purchases(...,
allocation="even")`, over the cash, or the budget the user gave): one neutral company gets all of it. No limit of the profile exists to cap it, and the Account Manager does no risk management: it asks the Data Clerk for the
price, decides, and hands the order to the Trader, which meets the cash as it is at the time of the trade.

ASK OR AUTO. In AUTO mode the decision is `yes_auto` (the setting is the user's policy) and nobody is asked how much. In ASK mode, when the request did not say how much ("I want to buy MU and AMD") and somebody is at the keyboard,
the Account Manager first ASKS the user for the amount, BEFORE it sizes anything, with the same bounded ReAct questions as for every approved purchase (`agents/account_manager/clarify.py`, entry 85; ONE set of questions covers the
Critic's approved purchases and these): a percentage of the account, dollars, a number of shares or "as much as I can"; the shares the user asks for are what is proposed (`wanted`: exactly those, the cash being the only thing
that can cut them). It then asks about each neutral purchase, as about any purchase that waits for them (no answer: `pending_user`, not traded).
Nothing here calls a model or an API: the prices are given.
"""
from datetime import datetime, timedelta

from prometheus.agents.account_manager import sizing
from prometheus.agents.critic.common import utc_text
from prometheus.common.portfolio import Portfolio
from prometheus.common.work_order import NEUTRAL, WorkOrder, is_neutral_candidate
from prometheus.rag.config import setting

TRUE_WORDS = ("1", "true", "yes", "on")
MODES = ("ask", "auto")


def allow_neutral_recommendations() -> bool:
    """`ALLOW_NEUTRAL_RECOMMENDATIONS` from the environment, then `.env`: true, yes, on or 1 means a neutral recommendation counts as approved; unset, or anything else, means it does not (the behaviour before entry 91)."""
    return setting("ALLOW_NEUTRAL_RECOMMENDATIONS", "false").strip().lower() in TRUE_WORDS


def neutral_candidates(order: WorkOrder) -> list[dict]:
    """The Advisor's candidate records of the companies the user named that it scored neutral (`work_order.is_neutral_candidate`), best score first as the Advisor listed them; EMPTY when the setting is off. Only an order
    the Advisor has answered (`recommended`, or `reviewed` by the Critic) has any."""
    if not allow_neutral_recommendations() or order.status not in ("recommended", "reviewed"):
        return []
    return [c for c in order.candidates if is_neutral_candidate(c)]


def record_id_of(order: WorkOrder, number: int) -> str:
    return f"neutral-{order.request_id}-{number}"


def purchases_for(order: WorkOrder, names: list[dict], prices: dict[str, float]) -> list[sizing.Purchase]:
    """The neutral companies as the plain numbers the sizing takes: one purchase for each of `names`, numbered from 1 in the Advisor's order (`record_id_of`), at the Data Clerk's price (None when it had none). The sizing
    questions of Ask mode (`clarify.py`) and `decide_neutral` work over the SAME purchases, so the shares the user asks for land on the right company."""
    return [sizing.Purchase(record_id=record_id_of(order, n), symbol=c["symbol"], action="open", price=prices.get(c["symbol"]), rank=n) for n, c in enumerate(names, 1)]


def _confirmation_card(candidate: dict, decision: dict, cash: float) -> dict:
    text = "\n".join([f"      A NEUTRAL RECOMMENDATION, TREATED AS APPROVED (ALLOW_NEUTRAL_RECOMMENDATIONS): the Advisor found no strategy for {candidate['symbol']}",
                      f"        company     {candidate.get('company') or candidate['symbol']} ({candidate['symbol']}), Advisor score {candidate.get('score')}, stance neutral",
                      f"        order       buy {decision['shares']:,} shares at ${decision['price']:,.2f} = about ${decision['est_cost']:,.2f}   (cash ${cash:,.2f})"])
    return {"text": text, "symbol": candidate["symbol"], "shares": decision["shares"], "est_cost": decision["est_cost"], "action": decision["action"]}


def decide_neutral(order: WorkOrder, portfolio: Portfolio, names: list[dict], mode: str, prices: dict[str, float], now: datetime, valid_minutes: int, approver=None, budget: float | None = None,
                   wanted: dict[str, int] | None = None, allocation: str | None = None) -> list[dict]:
    """One trade decision for each of `names` (see the module text), APPENDED to `order.trade_decisions`, and returned. `prices` is {symbol: the current price} from the Data Clerk. The neutral companies share the cash (or
    `budget`) in equal dollars (`allocation` "even", or "round_robin" when the user asked for it), and, when the user named the shares they want in Ask mode (`wanted`, {record_id: whole shares}, from the Account
    Manager's questions about how much, entry 85), exactly those shares (the cash is the only thing that can cut them) unless they also asked for a split. A company with no price is `no_system_denied`; one the even share cannot buy a share of is `no_funds`; anything else is `yes_auto` in Auto mode and, in Ask mode, what the user says
    through `approver` (True `yes_user`, False `no_user_denied`, None `pending_user`). The profile is not read here and nothing limits a purchase but the cash."""
    if mode not in MODES:
        order.errors = [*order.errors, f"mode must be one of {', '.join(MODES)}"]
        return []
    purchases = purchases_for(order, names, prices)
    try:
        sized = sizing.size_purchases(purchases, portfolio.cash_avail, allocation=allocation or (None if wanted else "even"), budget=budget, wanted=wanted)
    except ValueError as error:  # only a program gets these wrong: a message on the order, and nothing is bought
        order.errors = [*order.errors, f"the neutral purchases could not be sized: {error}"]
        return []
    made = []
    for candidate, purchase, size in zip(names, purchases, sized):
        why = f"the Advisor scored {candidate['symbol']} NEUTRAL (score {candidate.get('score')}: {candidate.get('reason') or 'no strategy passed'}); ALLOW_NEUTRAL_RECOMMENDATIONS is on, so a neutral recommendation counts as approved"
        decision = {"record_id": purchase.record_id, "review_id": None, "symbol": candidate["symbol"], "action": "open", "shares": 0, "est_cost": 0.0, "mode": mode, "will_trade": None, "funds_avail": None, "funds_note": None,
                    "asked_user": False, "escalated": False, "reason": why, "decided_utc": utc_text(now), "size": None, "origin": NEUTRAL, "company": candidate.get("company"), "price": purchase.price,
                    "valid_until_utc": utc_text(now + timedelta(minutes=valid_minutes))}
        if purchase.price is None:
            decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {candidate['symbol']}, so the order cannot be priced ({why})")
        else:
            decision["size"] = {"shares": size.shares, "caps": size.caps, "binding": size.binding, "notes": size.notes, "allocation": size.allocation, "asked": size.asked}
            if size.shares < 1:
                decision.update(will_trade="no_funds", funds_avail=False, reason=(size.notes[-1] if size.notes else "not even one share fits") + f" ({why})")
            else:
                how = (f"the {size.asked:,} shares the user asked for (at most)" if size.asked is not None else
                       f"equal dollars with the other neutral companies, about ${size.target_dollars:,.2f} each" if size.allocation == "even" and size.target_dollars is not None else
                       "a share of each neutral company in turn" if size.allocation == "round_robin" else "all the cash allows")
                decision.update(shares=size.shares, est_cost=size.est_cost, funds_avail=True, balance={"mode": size.allocation, "target_dollars": size.target_dollars, "of": size.of},
                                reason=why + f"; sized by {how}, no profile limit")
                if mode == "auto":
                    decision.update(will_trade="yes_auto")
                else:
                    decision["asked_user"] = approver is not None
                    answer = None
                    if approver is not None:
                        try:
                            answer = approver(_confirmation_card(candidate, decision, portfolio.cash_avail))
                        except Exception as error:  # a prompt that failed: the user has not said yes
                            order.errors = [*order.errors, f"the user could not be asked about {candidate['symbol']} ({type(error).__name__})"]
                    decision.update(will_trade="yes_user" if answer is True else "no_user_denied" if answer is False else "pending_user",
                                    reason=("the user approved: " if answer is True else "the user denied: " if answer is False else "waiting for the user: ") + decision["reason"])
        made.append(decision)
    order.trade_decisions = [*order.trade_decisions, *made]
    return made


WORDS = {"yes_auto": "BUY (treated as approved, Auto)", "yes_user": "BUY (treated as approved, you said yes)", "pending_user": "WAITING for your yes", "no_user_denied": "NOT BOUGHT (you said no)",
         "no_funds": "NOT BOUGHT (no funds)", "no_system_denied": "NOT BOUGHT (no price)"}


def neutral_lines(order: WorkOrder) -> list[str]:
    """The NEUTRAL section of a report (plain text from the records, no model): each company the Advisor scored neutral that the Account Manager treated as approved, with what it decided. Empty when there is none."""
    rows = [d for d in order.trade_decisions if d.get("origin") == NEUTRAL]
    if not rows:
        return []
    by_symbol = {c["symbol"]: c for c in order.candidates if isinstance(c, dict)}
    lines = ["NEUTRAL RECOMMENDATIONS  treated as approved (ALLOW_NEUTRAL_RECOMMENDATIONS is on): the Advisor found no strategy for them; equal dollars, no profile limit"]
    for d in rows:
        c = by_symbol.get(d["symbol"], {})
        lines.append(f"  {d['symbol']:<6} {(d.get('company') or c.get('company') or ''):<24} Advisor score {c.get('score', '-')} neutral   " + WORDS.get(d["will_trade"], d["will_trade"])
                     + (f": {d['shares']:,} shares at ${d['price']:,.2f} = about ${d['est_cost']:,.2f}" if d["shares"] else ""))
        if d["will_trade"] in ("no_funds", "no_system_denied"):
            lines.append(f"      {d['reason']}")
    return lines
