"""A NEUTRAL RECOMMENDATION IS AN APPROVAL (design choices entries 91 and 107; Jacob, 2026-11-15: *"A neutral recommendation is same as Approved"*, with `ALLOW_NEUTRAL_RECOMMENDATIONS=true`, which the page's "Neutral
recommendations as passes" box sets for a run).

Money is `decimal.Decimal` (design choices entry 112): the prices the sizing takes are Decimals (`purchases_for`), a decision's `shares` and `est_cost` start as Decimal zeros, and the Advisor's score, which only orders the companies, stays a float (a measurement).

    names = neutral_candidates(order)                                            # every company the Advisor scored NEUTRAL, best score first; empty unless the setting is on
    chosen, not_needed = choose(names, approved, count)                          # the number the user asked for is kept
    purchases = purchases_for(order, chosen, prices, after_rank)                 # the plain numbers the sizing takes, ranked after the approved purchases
    decision = decision_for(candidate, purchase, mode, now, valid_minutes)       # its trade decision, not sized yet: `decisions.decide_trades` sizes it WITH the approved purchases

THE SETTING. `ALLOW_NEUTRAL_RECOMMENDATIONS` (the environment, then `.env`; true, yes, on or 1; unset or anything else is false: nothing changes for anyone who does not set it) is read by the ACCOUNT MANAGER when it works with
the work order's recommendation result (`AccountManager.decide_trades`). When it is true, a company that the Advisor scored NEUTRAL (no strategy passed, or no edge: it is on `order.candidates` with `stance` "neutral" and no
failed gate, `work_order.is_neutral_candidate`) counts as an APPROVAL, in both modes, WHOEVER NAMED THE COMPANY (entry 107: it used to be only a company the user typed; a theme's companies, found by the Data Clerk, did not count,
or rather all counted as "named" and were bought in addition to the number asked for). The Account Manager gives the Trader permission to buy it: a trade decision with `origin: "neutral"`, which `work_order.verify_trade`
accepts only while the work order holds that Advisor record and no Critic review beside it. A neutral company has no entry plan, no stop and no Critic review, so the permission is the Advisor's own neutral score, nothing is
forged (no review is made up), and the trade is a MARKET purchase at the quote the Account Manager asked of the Data Clerk, like a direct order.

WHAT STAYS. A BEARISH stance is not a recommendation and trades nothing. A Critic DENIAL is final: a bullish company whose review denied it (risk above the profile, a falling knife) is never turned into an approval by this
setting, because only a company the Advisor did NOT recommend can be neutral. A gate that failed (stale data, a red flag) is not a stance and stops the company.

THE NUMBER THE USER ASKED FOR IS KEPT (entry 107). When the request said how many companies ("the 3 most recommended": the form's `count`), at most that many are bought: the Critic's approved recommendations first, in rank
order, then the neutral companies, the Advisor's best score first, until the number is reached. A denied recommendation does not count. A neutral company beyond the number gets NO decision, and the Account Manager tells the
user it was neutral and not needed (`not_needed_line`). When the request gave no number, nothing is cut.

HOW MUCH (Jacob, 2026-11-15: no profile limit applies). The only check is that the funds cover the order. A neutral purchase is the same thing as an approved one to the Account Manager: ONE call of `sizing.size_purchases`
(in `decisions.decide_trades`) over the approved purchases and then the neutral ones, in that order, so "split it evenly among the 3" is three equal amounts (the even split spends what is left over, entry 108) and the default
is one round-robin over all of them (entry 106), a run of neutral companies alone included (entry 109; entry 91 had given a run of neutral companies alone equal dollars, "Use 1", MU 32 shares and AMD 46 in the real session of
`pam_smith_neutral.json`: Jacob, 2026-11-15: *"use round-robin for neutral-only runs too. Use split evenly if the user specifies it in the prompt"*). There is no default of the neutral kind. No limit of the profile exists to
cap a purchase, and the Account Manager does no risk management: it asks the Data Clerk for the price, decides, and hands the order to the Trader, which meets the cash as it is at the time of the trade.

ASK OR AUTO. In AUTO mode the decision is `yes_auto` (the setting is the user's policy) and nobody is asked how much. In ASK mode, when the request did not say how much ("I want to buy MU and AMD") and somebody is at the
keyboard, the Account Manager first ASKS the user for the amount, BEFORE it sizes anything, with the same bounded ReAct questions as for every approved purchase (`agents/account_manager/clarify.py`, entry 85; ONE set of
questions covers the Critic's approved purchases and these): a percentage of the account, dollars, a number of shares or "as much as I can"; the shares the user asks for are each company's own cap (`wanted`). It then asks
about each purchase, approved or neutral, in one proposal (no answer: `pending_user`, not traded).
Nothing here calls a model or an API: the prices are given.
"""
from datetime import datetime, timedelta
from decimal import Decimal

from prometheus.agents.account_manager import sizing
from prometheus.agents.critic.common import utc_text
from prometheus.common import shares as sh
from prometheus.common.work_order import NEUTRAL, WorkOrder, is_neutral_candidate
from prometheus.common.trace import traced
from prometheus.rag.config import setting

TRUE_WORDS = ("1", "true", "yes", "on")


def allow_neutral_recommendations() -> bool:
    """`ALLOW_NEUTRAL_RECOMMENDATIONS` from the environment, then `.env`: true, yes, on or 1 means a neutral recommendation counts as approved; unset, or anything else, means it does not (the behaviour before entry 91)."""
    return setting("ALLOW_NEUTRAL_RECOMMENDATIONS", "false").strip().lower() in TRUE_WORDS


def neutral_candidates(order: WorkOrder) -> list[dict]:
    """The Advisor's candidate records of EVERY company it scored neutral, whoever named it (`work_order.is_neutral_candidate`), best score first (a tie keeps the Advisor's own order); EMPTY when the setting is off. Only an
    order the Advisor has answered (`recommended`, or `reviewed` by the Critic) has any."""
    if not allow_neutral_recommendations() or order.status not in ("recommended", "reviewed"):
        return []
    rows = [c for c in order.candidates if is_neutral_candidate(c)]
    return sorted(rows, key=lambda c: -(c["score"] if isinstance(c.get("score"), (int, float)) and not isinstance(c.get("score"), bool) else -1))


def choose(names: list[dict], approved: int, count: int | None) -> tuple[list[dict], list[dict]]:
    """(the neutral companies that are bought, the ones that are not needed). `approved` is how many purchases the Critic's approvals already make (a denied recommendation is not one) and `count` the number of companies the
    request asked for (the form's own, None when the request gave none: then nothing is cut). The neutral companies fill what is left of the number, the Advisor's best score first."""
    if count is None:
        return list(names), []
    room = max(0, count - approved)
    return names[:room], names[room:]


def not_needed_line(candidate: dict, count: int) -> str:
    """What the Account Manager tells the user about a neutral company that the number asked for did not need."""
    return (f"{candidate['symbol']} was scored neutral by the Advisor (score {candidate.get('score', '-')}) and counts as approved, but it is not needed: {count} compan{'ies were' if count != 1 else 'y was'} asked for, "
            f"and the approved ones and the better neutral ones make them")


def record_id_of(order: WorkOrder, number: int) -> str:
    return f"neutral-{order.request_id}-{number}"


@traced("account_manager.neutral_purchases")
def purchases_for(order: WorkOrder, names: list[dict], prices: dict[str, Decimal], after_rank: int = 0) -> list[sizing.Purchase]:
    """The neutral companies as the plain numbers the sizing takes: one purchase for each of `names`, numbered from 1 in the order given (`record_id_of`), at the Data Clerk's price (None when it had none), RANKED AFTER the
    approved purchases (`after_rank` is the highest rank of those: the sizing visits the purchases in rank order, so the approved ones come first and then the neutral ones, best score first). The sizing questions of Ask mode
    (`clarify.py`) and `decisions.decide_trades` work over the SAME purchases, so the shares the user asks for land on the right company."""
    return [sizing.Purchase(record_id=record_id_of(order, n), symbol=c["symbol"], action="open", price=prices.get(c["symbol"]), rank=after_rank + n) for n, c in enumerate(names, 1)]


def confirmation_card(candidate: dict, decision: dict, cash: Decimal) -> dict:
    """The card of a neutral purchase for the user (Ask mode): what the Advisor found and what would be placed."""
    text = "\n".join([f"      A NEUTRAL RECOMMENDATION, TREATED AS APPROVED (ALLOW_NEUTRAL_RECOMMENDATIONS): the Advisor found no strategy for {candidate['symbol']}",
                      f"        company     {candidate.get('company') or candidate['symbol']} ({candidate['symbol']}), Advisor score {candidate.get('score')}, stance neutral",
                      (f"        order       buy ${decision['dollars']:,.2f}: about {sh.text(decision['shares'])} shares at ${decision['price']:,.2f}   (cash ${cash:,.2f})" if decision.get("dollars") else
                       f"        order       buy {sh.text(decision['shares'])} shares at ${decision['price']:,.2f} = about ${decision['est_cost']:,.2f}   (cash ${cash:,.2f})")])
    return {"text": text, "symbol": candidate["symbol"], "shares": decision["shares"], "est_cost": decision["est_cost"], "dollars": decision.get("dollars"), "price": decision.get("price"), "action": decision["action"]}


def decision_for(candidate: dict, purchase: sizing.Purchase, mode: str, now: datetime, valid_minutes: int) -> dict:
    """The trade decision of one neutral company, NOT SIZED YET (`decisions.decide_trades` sizes it with the approved purchases, in one call): `origin` "neutral", no review, its own `valid_until_utc`. A company with no price
    is `no_system_denied` (nothing is bought that cannot be priced); any other is left for the decision table (`yes_auto` in Auto mode, the user's answer in Ask mode)."""
    why = (f"the Advisor scored {candidate['symbol']} NEUTRAL (score {candidate.get('score')}: {candidate.get('reason') or 'no strategy passed'}); ALLOW_NEUTRAL_RECOMMENDATIONS is on, so a neutral recommendation counts as approved")
    decision = {"record_id": purchase.record_id, "review_id": None, "symbol": candidate["symbol"], "action": "open", "shares": Decimal(0), "est_cost": Decimal("0.00"), "mode": mode, "will_trade": None, "funds_avail": None, "funds_note": None,
                "asked_user": False, "reason": why + "; no profile limit applies", "decided_utc": utc_text(now), "size": None, "origin": NEUTRAL, "company": candidate.get("company"), "price": purchase.price,
                "valid_until_utc": utc_text(now + timedelta(minutes=valid_minutes))}
    if purchase.price is None:
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {candidate['symbol']}, so the order cannot be priced ({why})")
    return decision


WORDS = {"yes_auto": "BUY (treated as approved, Auto)", "yes_user": "BUY (treated as approved, you said yes)", "pending_user": "WAITING for your yes", "no_user_denied": "NOT BOUGHT (you said no)",
         "no_funds": "NOT BOUGHT (no funds)", "no_system_denied": "NOT BOUGHT (no price)"}


NOT_NEEDED = "neutral, not needed: "      # how a note on the work order's data begins when a neutral company was not needed (`not_needed_line`)


def neutral_lines(order: WorkOrder) -> list[str]:
    """The NEUTRAL section of a report (plain text from the records, no model): each company the Advisor scored neutral that the Account Manager treated as approved, with what it decided, and the ones the number asked for did not
    need (the notes the Account Manager left on the order's data). Empty when there is none."""
    rows = [d for d in order.trade_decisions if d.get("origin") == NEUTRAL]
    unneeded = [n[len(NOT_NEEDED):] for n in (order.data.notes if order.data else []) if n.startswith(NOT_NEEDED)]
    if not rows and not unneeded:
        return []
    by_symbol = {c["symbol"]: c for c in order.candidates if isinstance(c, dict)}
    lines = ["NEUTRAL RECOMMENDATIONS  treated as approved (ALLOW_NEUTRAL_RECOMMENDATIONS is on): the Advisor found no strategy for them; sized with the approved ones, no profile limit"]
    for d in rows:
        c = by_symbol.get(d["symbol"], {})
        lines.append(f"  {d['symbol']:<6} {(d.get('company') or c.get('company') or ''):<24} Advisor score {c.get('score', '-')} neutral   " + WORDS.get(d["will_trade"], d["will_trade"])
                     + ((f": ${d['dollars']:,.2f}, about {sh.text(d['shares'])} shares at ${d['price']:,.2f}" if d.get("dollars") else f": {sh.text(d['shares'])} shares at ${d['price']:,.2f} = about ${d['est_cost']:,.2f}") if d["shares"] else ""))
        if d["will_trade"] in ("no_funds", "no_system_denied"):
            lines.append(f"      {d['reason']}")
    lines += [f"  {text}" for text in unneeded]
    return lines
