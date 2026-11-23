"""IGNORE CRITIC: THE USER SKIPS THE CRITIC FOR ONE RUN (design choices entry 148; Jacob, 2026-11-23: *"The ability to Ignore (Skip) Critic step ... add a new Checkbox: Ignore Critic with defaulted to false"*). The spec is `design/ignore_critic_spec.md`.

With the box ticked the desk goes `advisor -> account_manager_decision` and the Critic's node is never run (`langgraph/desk.py`). Every recommendation the Advisor made then counts as APPROVED, and the Account Manager decides it exactly as it decides a
neutral one (`neutral.py` is the model of this module): a trade decision with `origin: "advisor"`, no review, its own `valid_until_utc`. Money is `decimal.Decimal` (entry 112); the Advisor's score, which only orders the companies, stays a float.

    records, others = recommendations_of(order)                                  # the Advisor's records the Account Manager decides, best rank first (empty unless the order says `ignore_critic`); the ones that are not a purchase
    chosen, beyond = choose(records, count)                                      # the number the request asked for is kept: the Advisor's recommendations come FIRST, then the neutral companies
    purchases = purchases_for(order, chosen, prices, after_rank)                 # the plain numbers the sizing takes, in rank order, at the Account Manager's own quote
    decision = decision_for(record, purchase, mode, now, valid_minutes)          # its trade decision, not sized yet: `decisions.decide_trades` sizes it WITH the neutral purchases, in ONE call

WHAT IT CHANGES, AND WHAT IT DOES NOT. Only a run where the user ticked the box (`--ignore-critic` in the session). A recommendation the Critic would have DENIED may trade, because the Critic does not run: there is no denial to override, and the user chose that. With
the box off nothing here runs and "a denial is final" (entry 104) is as it was. `work_order.verify_trade` accepts an `advisor` decision only while the order's own `ignore_critic` is true, the Advisor's record is on the order, and NO review of it is (a denial that exists
is never relabelled). AUTO trades every one (`yes_auto`, nothing is asked: entry 104); ASK asks about each, in the same proposal as the neutral purchases, with a card that says plainly the Critic did NOT review it.
The Advisor's recommendations are purchases only (entry 105): a record that is not one gets no purchase. Nothing about the size changes (entry 91: the funds are the only check; no size limit, risk rule or liquidity rule is added), the pot, the dollars orders, the
fractions, the typed shares and the price limit and floor all apply as they do to a Critic-approved purchase. THE PRICE: a Critic-approved purchase is priced from the Critic's deal price; with no Critic the Account Manager asks the Data Clerk for the quote, as it does for a
neutral company, and a recommendation with no current price is `no_system_denied`. Nothing here calls a model or an API: the prices are given.
"""
from datetime import datetime, timedelta
from decimal import Decimal

from prometheus.agents.account_manager import sizing
from prometheus.agents.critic.common import utc_text
from prometheus.common import shares as sh
from prometheus.common.trace import traced
from prometheus.common.work_order import ADVISOR, WorkOrder, record_action_of

SKIPPED_LINE = "IGNORE CRITIC: the Critic did not review these recommendations (the user's setting)"      # what the session prints before the decisions
NOT_NEEDED = "ignore critic, not needed: "      # how a note on the work order's data begins when a recommendation was beyond the number asked for (the Account Manager tells the user)
THESIS_CHARS = 320       # the Advisor's thesis in a card is cut to this many characters (it is text for the user to read, not a number)


def _rank_of(record: dict):
    rank = (record.get("ranking") or {}).get("rank")
    return rank if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0 else None


def _record_id(record: dict) -> str | None:
    record_id = (record.get("identity") or {}).get("record_id")
    return record_id if isinstance(record_id, str) and record_id else None


def recommendations_of(order: WorkOrder) -> tuple[list[dict], list[dict]]:
    """(the Advisor's records to buy, the records that are not a purchase), each best rank first (a tie keeps the Advisor's own order). BOTH EMPTY unless the order says the user chose to ignore the Critic (`ignore_critic`) and the
    Advisor has answered it (`recommended`, or `reviewed` once the desk moved it): with the box off nothing changes. A record with no symbol or no record id is not a recommendation and is left out."""
    if not order.ignore_critic or order.status not in ("recommended", "reviewed"):
        return [], []
    rows = [r for r in order.recommendations if isinstance(r, dict) and isinstance(r.get("symbol"), str) and _record_id(r) is not None]
    rows = sorted(rows, key=lambda r: (_rank_of(r) is None, _rank_of(r) or 0))
    return [r for r in rows if record_action_of(r) == "open"], [r for r in rows if record_action_of(r) != "open"]


def choose(records: list[dict], count: int | None) -> tuple[list[dict], list[dict]]:
    """(the recommendations that are bought, the ones that are not needed). `count` is the number of companies the request asked for (the form's own, None when the request gave none: then nothing is cut). The Advisor's recommendations
    come before the neutral companies (`neutral.choose` fills what is left of the number), in the Advisor's rank order."""
    if count is None:
        return list(records), []
    return records[:count], records[count:]


def not_needed_line(record: dict, count: int) -> str:
    """What the Account Manager tells the user about a recommendation the number asked for did not need."""
    return (f"{record['symbol']} was recommended by the Advisor (score {record.get('score', '-')}) and counts as approved, but it is not needed: {count} compan{'ies were' if count != 1 else 'y was'} asked for, "
            f"and better ranked recommendations make them")


def record_id_of(record: dict) -> str:
    """The record id of the purchase: the Advisor's own, so that the Trader's tickets, the Track Record and the page join the decision to the recommendation as for any other."""
    return _record_id(record) or ""


@traced("account_manager.advisor_purchases")
def purchases_for(order: WorkOrder, records: list[dict], prices: dict[str, Decimal], after_rank: int = 0) -> list[sizing.Purchase]:
    """The recommendations as the plain numbers the sizing takes: one purchase for each of `records`, in the order given, at the Data Clerk's price (None when it had none), ranked by the Advisor's own rank (`after_rank + n` when a
    record has none). The sizing questions of Ask mode (`clarify.py`) and `decisions.decide_trades` work over the SAME purchases, so the shares the user asks for land on the right company."""
    return [sizing.Purchase(record_id=record_id_of(r), symbol=r["symbol"], action="open", price=prices.get(r["symbol"]), rank=_rank_of(r) if _rank_of(r) is not None else after_rank + n) for n, r in enumerate(records, 1)]


def _why(record: dict) -> str:
    return (f"the Critic was skipped by the user's 'Ignore Critic' setting, so the Advisor's recommendation of {record['symbol']} (score {record.get('score')}) counts as approved")


def decision_for(record: dict, purchase: sizing.Purchase, mode: str, now: datetime, valid_minutes: int) -> dict:
    """The trade decision of one Advisor recommendation the Critic did not review, NOT SIZED YET (`decisions.decide_trades` sizes it with the neutral purchases, in one call): `origin` "advisor", no review, its own `valid_until_utc`. A company
    with no price is `no_system_denied` (nothing is bought that cannot be priced); any other is left for the decision table (`yes_auto` in Auto mode, the user's answer in Ask mode)."""
    why = _why(record)
    decision = {"record_id": purchase.record_id, "review_id": None, "symbol": record["symbol"], "action": "open", "shares": Decimal(0), "est_cost": Decimal("0.00"), "mode": mode, "will_trade": None, "funds_avail": None, "funds_note": None,
                "asked_user": False, "reason": why + "; no profile limit applies", "decided_utc": utc_text(now), "size": None, "origin": ADVISOR, "company": record.get("company"), "price": purchase.price,
                "valid_until_utc": utc_text(now + timedelta(minutes=valid_minutes))}
    if purchase.price is None:
        decision.update(will_trade="no_system_denied", funds_avail=False, reason=f"there is no current price for {record['symbol']}, so the order cannot be priced ({why})")
    return decision


def not_a_purchase(record: dict, mode: str, now: datetime) -> dict:
    """The decision of a record that is not a purchase (an old stored `reduce` or `hold`; entry 105): `no_funds` with the reason, as the review of one gets. Nothing is sized and nothing trades."""
    action = record_action_of(record)
    return {"record_id": record_id_of(record), "review_id": None, "symbol": record["symbol"], "action": action if action in ("open", "add", "reduce") else "open", "shares": Decimal(0), "est_cost": Decimal("0.00"), "mode": mode, "will_trade": "no_funds",
            "funds_avail": False, "funds_note": None, "asked_user": False, "decided_utc": utc_text(now), "size": None, "origin": ADVISOR, "company": record.get("company"), "price": None, "valid_until_utc": None,
            "reason": f"a {action} recommendation is not a purchase and is never traded: the Advisor recommends purchases only, and a sale comes from the user's own order or RaiseFunds"}


def _thesis(record: dict) -> str:
    text = " ".join(str((record.get("thesis") or {}).get("thesis") or "").split())
    return text if len(text) <= THESIS_CHARS else text[:THESIS_CHARS - 3].rstrip() + "..."


def confirmation_card(record: dict, decision: dict, cash: Decimal) -> dict:
    """The card of an Advisor recommendation for the user (Ask mode): it says plainly that the Critic did NOT review it, shows the Advisor's score and thesis, and what would be placed."""
    stance = record.get("stance") or "bullish"
    lines = [f"      THE CRITIC DID NOT REVIEW THIS RECOMMENDATION (Ignore Critic is on): it counts as approved on the Advisor's word alone",
             f"        company     {record.get('company') or record['symbol']} ({record['symbol']}), Advisor score {record.get('score')}, stance {stance}"]
    if _thesis(record):
        lines.append(f"        thesis      {_thesis(record)}")
    lines.append(f"        order       buy ${decision['dollars']:,.2f}: about {sh.text(decision['shares'])} shares at ${decision['price']:,.2f}   (cash ${cash:,.2f})" if decision.get("dollars") else
                 f"        order       buy {sh.text(decision['shares'])} shares at ${decision['price']:,.2f} = about ${decision['est_cost']:,.2f}   (cash ${cash:,.2f})")
    return {"text": "\n".join(lines), "symbol": record["symbol"], "shares": decision["shares"], "est_cost": decision["est_cost"], "dollars": decision.get("dollars"), "price": decision.get("price"), "action": decision["action"],
            "critic_skipped": True}


WORDS = {"yes_auto": "BUY (Critic skipped, Auto)", "yes_user": "BUY (Critic skipped, you said yes)", "pending_user": "WAITING for your yes", "no_user_denied": "NOT BOUGHT (you said no)",
         "no_funds": "NOT BOUGHT (no funds)", "no_system_denied": "NOT BOUGHT (no price)", "no_expired": "NOT BOUGHT (the price check ran out)"}


def skipped(order: WorkOrder) -> bool:
    """Whether this order was decided without the Critic because the user said so: the order's own flag, and a decision of origin `advisor`."""
    return bool(order.ignore_critic) and any(d.get("origin") == ADVISOR for d in order.trade_decisions if isinstance(d, dict))


def advisor_lines(order: WorkOrder) -> list[str]:
    """The IGNORE CRITIC section of a report (plain text from the records, no model): the line that says the Critic did not review, then each Advisor recommendation decided without it, with what the Account Manager decided. Empty when the
    order was not decided with the Critic skipped."""
    rows = [d for d in order.trade_decisions if isinstance(d, dict) and d.get("origin") == ADVISOR]
    if not order.ignore_critic or not rows:
        return []
    by_id = {(r.get("identity") or {}).get("record_id"): r for r in order.recommendations if isinstance(r, dict)}
    lines = [SKIPPED_LINE]
    for d in rows:
        r = by_id.get(d["record_id"], {})
        lines.append(f"  {d['symbol']:<6} {(d.get('company') or r.get('company') or ''):<24} Advisor score {r.get('score', '-')} {r.get('stance') or ''}   " + WORDS.get(d["will_trade"], d["will_trade"])
                     + ((f": ${d['dollars']:,.2f}, about {sh.text(d['shares'])} shares at ${d['price']:,.2f}" if d.get("dollars") else f": {sh.text(d['shares'])} shares at ${d['price']:,.2f} = about ${d['est_cost']:,.2f}") if d["shares"] else ""))
        if d["will_trade"] in ("no_funds", "no_system_denied"):
            lines.append(f"      {d['reason']}")
    return lines
