"""From the Account Manager's decisions to TICKETS: code only (Trader v1 spec sections 5 and 7).

    tickets = build_tickets(order, portfolio, now, config, mode)               # one Trade (status `new`, or `rejected` with a code) per decision that says yes_auto or yes_user
    target, legs, note = funding_legs(funding, tickets, portfolio, today)      # the sells of a RaiseFunds plan that code accepts, and why when it does not
    sells = funding_trades(target, legs, config, mode)                         # the sells as trades of their own: T1-F1, T1-F2 ...

A ticket is a `Trade` with status `new`. Code makes it from what the other agents wrote, read as plain dicts: the decision (the symbol, the action, the shares the user or the desk allowed), the Critic's review (its
deal price) and the Advisor's trade plan (`recommendation["trade"]`: the entry zone, the limit price). No model is involved, and nothing the model says can change a ticket (rule T2).

    side             `sell` when the decision's action is `reduce`, else `buy`: there is no short sale
    shares_approved  the decision's shares: the Trader never trades more
    shares           a sale: min(shares_approved, the shares sellable today: the T+1 rule). A purchase: shares_approved, cut for cash when it is placed
    ref_price        the review's `timing.deal_price`, else the plan's `entry_mid`
    limits, a buy    high = min(the entry zone's top, ref (1 + max_chase_pct)), low = max(the zone's bottom, ref (1 - limit_band_pct)); without a zone only the percentages apply
    limits, a sale   low = ref (1 - max_chase_pct), high = ref (1 + limit_band_pct)
    default order    the Advisor's own plan: a limit at the plan's `limit_price`, moved inside the limits; a market order when the plan has no limit price (a limit at the reference price when market orders are off)

A ticket that cannot be traded is built `rejected` at once, with a code of the closed list (`record.REJECT_CODES`): `not_permitted` (`verify_trade` gave reasons, which go in `reason`), `approval_expired` (the only reason is
that the approval ran out), `not_held` and `not_sellable_today` (a sale of a stock the portfolio does not hold, or whose shares were all bought today), `price_unavailable` (no reference price), `price_outside_limits` (a
purchase whose entry zone and reference price leave no price to pay: the price has moved past the zone the Critic judged) and `not_attempted` (beyond `limits.max_trades`). The model never sees a rejected ticket as placeable.
A decision whose symbol is not a ticker (`record.is_ticker`) gets NO ticket and uses no number: the symbol is the one piece of the order the model reads as text, so it is never a sentence. `build_tickets` names the
decision in `problems` and the run says so in its errors.

FUNDING (spec section 7). A RaiseFunds plan is used only when ALL of this holds: its status is `plan` with sells, its `target` is the symbol of a `new` purchase of this run, and every sell names a holding, is for at least
one share, at most the shares sellable today, has a price, and is not the target itself. All or nothing: one bad sell and no holding is sold. The sells are trades of their own (`kind: "funding"`, `funds_for` the purchase)
and carry the purchase's `record_id` and `review_id`; the model never places them. Whether the user said yes is the run's business (`execute.py`), not a ticket's.
"""
import re
from datetime import date, datetime, timezone

from prometheus.agents.trader.config import TraderConfig
from prometheus.agents.trader.record import Trade, is_ticker
from prometheus.agents.trader.simulator import r4, usable
from prometheus.common.market_calendar import eastern_date
from prometheus.common.portfolio import Portfolio
from prometheus.common.work_order import EXPIRY_PREFIX, TRADING, WorkOrder, review_of, verify_trade

TRADE_ID = re.compile(r"T(\d+)(?:-F\d+)?")
EXPIRY_REASON = EXPIRY_PREFIX              # the start of the sentence `verify_trade` gives when the approval's time is over: a ticket whose only reason is this is `approval_expired`, not `not_permitted` (the Account Manager's `approve_trade` reads the same constant)


def utc_text(moment: datetime) -> str:
    """A moment as the desk writes them: UTC, seconds, a trailing Z."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def side_of(decision: dict) -> str:
    """The side a decision means: `sell` for a `reduce`, else `buy`. There is no short sale."""
    return "sell" if decision.get("action") == "reduce" else "buy"


def next_number(order: WorkOrder) -> int:
    """The number the next ticket takes: after every `T<n>` (or `T<n>-F<m>`) already on the order, so a second run continues the numbering."""
    numbers = [int(m.group(1)) for t in order.trades if isinstance(t, dict) and (m := TRADE_ID.fullmatch(str(t.get("trade_id", ""))))]
    return max(numbers, default=0) + 1


def permission(order: WorkOrder, record_id: str, now: datetime) -> tuple[str | None, str]:
    """Rule T1: `work_order.verify_trade` is the one check. (None, "") when the trade is permitted; else the code (`approval_expired` when the approval running out is the ONLY reason, else `not_permitted`) and the
    reasons in code's own words. Raises ValueError for a naive `now`, as `verify_trade` does."""
    reasons = verify_trade(order, record_id, now)
    if not reasons:
        return None, ""
    if len(reasons) == 1 and reasons[0].startswith(EXPIRY_REASON):
        return "approval_expired", reasons[0]
    return "not_permitted", "; ".join(reasons)


def limit_range(side: str, ref_price: float, entry_zone, config: TraderConfig) -> tuple[float, float] | None:
    """The price range no order may leave, (low, high), 4 places; None when there is none (a purchase whose reference price sits so far under the entry zone that the zone's bottom is over the ceiling)."""
    chase, band = config.orders.max_chase_pct, config.orders.limit_band_pct
    if side == "buy":
        high, low = ref_price * (1 + chase), ref_price * (1 - band)
        if isinstance(entry_zone, (list, tuple)) and len(entry_zone) == 2 and all(usable(x) for x in entry_zone) and entry_zone[0] <= entry_zone[1]:
            high, low = min(high, float(entry_zone[1])), max(low, float(entry_zone[0]))
    else:
        low, high = ref_price * (1 - chase), ref_price * (1 + band)
    low, high = r4(low), r4(high)
    return (low, high) if low <= high else None


def shown_range(low: float, high: float) -> list[float]:
    """The range the model reads: rounded INWARD to the cent (the bottom up, the top down), so that any price inside what it reads is inside the real range. The real range, to 4 places, when no whole cent is inside it."""
    cents_low, cents_high = -(-round(low * 100, 6) // 1) / 100, (round(high * 100, 6) // 1) / 100
    return [cents_low, cents_high] if cents_low <= cents_high else [low, high]


def default_order(side: str, plan_limit, ref_price: float, low: float, high: float, config: TraderConfig, funding: bool = False) -> tuple[str, float | None]:
    """The order a ticket carries until it is placed otherwise: (`limit`, the plan's limit price moved inside the range); a plan with no limit price gives `market`, or a limit at the reference price moved inside the
    range when `orders.allow_market` is false. A funding sale is a market order, or, with market orders off, a limit at the bottom of its range (it sells at once at any price the Critic's range allows)."""
    clamp = lambda x: r4(min(max(x, low), high))
    if funding:
        return ("market", None) if config.orders.allow_market else ("limit", low)
    if usable(plan_limit):
        return "limit", clamp(float(plan_limit))
    return ("market", None) if config.orders.allow_market else ("limit", clamp(ref_price))


def _record_of(order: WorkOrder, record_id: str) -> dict:
    return next((r for r in order.recommendations if isinstance(r, dict) and (r.get("identity") or {}).get("record_id") == record_id), {})


def _whole(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def build_tickets(order: WorkOrder, portfolio: Portfolio, now: datetime, config: TraderConfig, mode: str, problems: list[str] | None = None) -> list[Trade]:
    """One ticket per decision that says `yes_auto` or `yes_user` and has no trade of its own on the order yet, in the order of the decisions, numbered T1, T2 ... after the trades already there. A ticket is `new`, or
    `rejected` at once (see the module text). Tickets beyond `limits.max_trades` are rejected `not_attempted`, and the reason gives the limit. A decision whose symbol is not a ticker (`record.is_ticker`) gets NO ticket and
    uses no number: a sentence is added to `problems` (when given) so that the run can say so. Changes nothing it is given. Raises ValueError for a naive `now`."""
    today = eastern_date(now)                            # raises ValueError for a naive `now`
    number, tickets = next_number(order), []
    have = {t.get("record_id") for t in order.trades if isinstance(t, dict) and t.get("kind") != "funding"}
    for decision in order.trade_decisions:
        record_id = decision.get("record_id")
        if decision.get("will_trade") not in TRADING or record_id in have:
            continue
        if not is_ticker(decision.get("symbol")):
            if problems is not None:
                problems.append(f"the decision for recommendation {str(record_id)[:24]} has no ticket: its symbol {str(decision.get('symbol'))[:20]!r} is not a ticker")
            continue
        side, symbol, approved = side_of(decision), decision["symbol"], _whole(decision.get("shares"))
        review, plan = review_of(order, record_id) or {}, _record_of(order, record_id).get("trade") or {}
        shares = approved
        base = dict(trade_id=f"T{number}", kind="recommendation", record_id=record_id, review_id=review.get("review_id") or decision.get("review_id"), symbol=symbol, side=side, shares_approved=approved, mode=mode)
        number += 1
        ref = (review.get("timing") or {}).get("deal_price")
        ref = float(ref) if usable(ref) else float(plan["entry_mid"]) if usable(plan.get("entry_mid")) else None
        code, reason = permission(order, record_id, now)
        if side == "sell":
            position = portfolio.position(symbol)
            shares = min(approved, portfolio.sellable_shares(symbol, today)) if position else 0
        if len(tickets) >= config.limits.max_trades:
            tickets.append(Trade(**base, shares=shares, status="rejected", reject_code="not_attempted",
                                 reason=f"the Trader places at most {config.limits.max_trades} trades in one run (limits.max_trades), and this is number {len(tickets) + 1}"))
        elif code:
            tickets.append(Trade(**base, shares=shares, ref_price=ref, status="rejected", reject_code=code, reason=reason))
        elif side == "sell" and portfolio.position(symbol) is None:
            tickets.append(Trade(**base, shares=0, ref_price=ref, status="rejected", reject_code="not_held", reason=f"the portfolio does not hold {symbol}, and there is no short sale"))
        elif side == "sell" and shares == 0:
            tickets.append(Trade(**base, shares=0, ref_price=ref, status="rejected", reject_code="not_sellable_today", reason=f"every share of {symbol} was bought today, and a share cannot be sold the day it was bought (T+1)"))
        elif ref is None:
            tickets.append(Trade(**base, shares=shares, status="rejected", reject_code="price_unavailable", reason="there is no price to judge the trade by: neither the Critic's deal price nor the plan's entry price"))
        elif (limits := limit_range(side, ref, plan.get("entry_zone"), config)) is None:
            tickets.append(Trade(**base, shares=shares, ref_price=ref, status="rejected", reject_code="price_outside_limits",
                                 reason=f"the price the Critic judged, {ref:,.4f}, is too far from the entry zone {plan.get('entry_zone')} for any price to be paid: a new review is needed"))
        else:
            kind, limit_price = default_order(side, plan.get("limit_price"), ref, *limits, config)
            tickets.append(Trade(**base, shares=shares, order_type=kind, limit_price=limit_price, ref_price=ref, limit_low=limits[0], limit_high=limits[1], status="new"))
    return tickets


def funding_legs(funding, tickets: list[Trade], portfolio: Portfolio, today: date) -> tuple[Trade | None, list[dict], str]:
    """The sells of a RaiseFunds plan that code accepts: (the purchase they pay for, the sells [{symbol, shares, price}], a sentence). The sells are empty, and the sentence says why, when the plan cannot be used (see the
    module text); nothing to say (`funding` None) is an empty sentence. `funding["status"] == "funded"` means the cash already covers the purchase: no sells. Does not ask anybody and changes nothing."""
    if funding is None:
        return None, [], ""
    if not isinstance(funding, dict):
        return None, [], "the funding answer is not a record: no holdings are sold"
    status = funding.get("status")
    if status == "funded":
        return None, [], "the funding plan says the cash already covers the purchase: no holdings are sold"
    plan = funding.get("plan") if status == "plan" else None
    sells = plan.get("sells") if isinstance(plan, dict) else None
    if status != "plan" or not isinstance(sells, list) or not sells:
        return None, [], f"the funding answer is {str(status)[:30]!r}, not a plan with sells: no holdings are sold"
    target = next((t for t in tickets if t.kind == "recommendation" and t.side == "buy" and t.status == "new" and t.symbol == funding.get("target")), None)
    if target is None:
        return None, [], f"the funding plan is for {str(funding.get('target'))[:20]}, which is not a purchase this run is placing: no holdings are sold"
    legs, seen = [], set()
    for line in sells:
        symbol, shares, price = (line.get("symbol"), line.get("shares"), line.get("price")) if isinstance(line, dict) else (None, None, None)
        position = portfolio.position(symbol) if isinstance(symbol, str) else None
        if position is None:
            why = "is not a holding"
        elif symbol == target.symbol:
            why = "is the stock being bought"
        elif symbol in seen:
            why = "appears twice in the plan"
        elif not isinstance(shares, int) or isinstance(shares, bool) or shares < 1:
            why = "has no whole number of shares"
        elif shares > portfolio.sellable_shares(symbol, today):
            why = f"is for {shares} shares but only {portfolio.sellable_shares(symbol, today)} are sellable today"
        elif not usable(price):
            why = "has no price"
        else:
            seen.add(symbol)
            legs.append({"symbol": symbol, "shares": shares, "price": float(price)})
            continue
        return None, [], f"the funding plan sells {str(symbol)[:12]}, which {why}: no holdings are sold"
    return target, legs, ""


def funding_trades(buy: Trade, legs: list[dict], config: TraderConfig, mode: str) -> list[Trade]:
    """The sells of `legs` as trades of their own, `T1-F1`, `T1-F2` ... for the purchase `buy`: kind `funding`, `funds_for` the purchase, its `record_id` and `review_id`, status `new` (they are placed with their
    purchase, never by the model). The reference price of a sale is the one the plan was made at; the range is a sale's (`limit_range`)."""
    out = []
    for index, leg in enumerate(legs, 1):
        low, high = limit_range("sell", leg["price"], None, config)
        kind, limit_price = default_order("sell", None, leg["price"], low, high, config, funding=True)
        out.append(Trade(trade_id=f"{buy.trade_id}-F{index}", kind="funding", record_id=buy.record_id, review_id=buy.review_id, funds_for=buy.trade_id, symbol=leg["symbol"], side="sell", shares_approved=leg["shares"],
                         shares=leg["shares"], order_type=kind, limit_price=limit_price, ref_price=leg["price"], limit_low=low, limit_high=high, status="new", mode=mode))
    return out
