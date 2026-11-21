"""RAISING FUNDS FOR A PURCHASE THE CASH DOES NOT COVER (design choices entry 142; design/direct_raise_funds_spec.md, rules R1 to R12).

Jacob, 2026-11-21: *"I did a direct order to buy 2 shares of ANT. There was not enough funds available and the RaiseFunds did not automatically work, even when I asked it in the prompt to raise funds ("Buy 2 shares of ANT and
raise funds if needed"). ... The desk should have prompted me (asked) the User: Do you want to raise funds? and if yes then ran my existing Tree-of-Thought RaiseFunds code ... It would have sold (taken some profits) and then bought the
two shares."* The Account Manager's RaiseFunds (`reasoning/raise_funds.py`, entry 42) was built for a purchase the Advisor recommends and only reached from the CLI flag `--raise-funds`; a direct order the cash did not cover was
"NOT PLACED (no funds)". This module is the small, plain-Python part that decides WHETHER holdings may be sold, and says it; the tree itself, the sizing and the Trader's handshake are unchanged in what they decide.

WHO MAY ASK FOR A SALE (entry 105: a sale comes only from the user's own order or from RaiseFunds; R5). Holdings are sold to pay for a purchase ONLY when the user permitted it, in one of three ways, and never otherwise:
    "words"     the user's own words say so (`intent.raise_funds_in`: "raise funds if needed", "sell some stock to pay for it", "free up cash", "take profits to buy"; a negation such as "don't sell anything" wins);
    "setting"   the setting ALWAYS_RAISE_FUNDS is true (R12: the same as the words being in EVERY request; read through `rag.config.setting`, never from `.env` directly; unset or anything but true is false; "don't sell anything" in the
                words of a request still wins over it for that request). On the page it is the box "Always raise funds", for one run;
    "asked"     the Account Manager ASKED in Ask mode and the user said yes (R3: one question, a card of the kind `raise_funds`; no answer is a no, never a yes).
AUTO MODE ASKS NOTHING (entry 104): with the words or the setting it raises funds with no question, without them nothing is sold and the message says what to add (`NOT_SAID`).

THE SHORTFALL. For the purchases of a run that are placed as the user typed them (a direct order's typed shares, the dollars of a direct order, the typed shares of an approved recommendation: R1 and R10), the shortfall is the cost of
all of them less the AVAILABLE cash (the book's door, entry 140: `cash_avail`), and RaiseFunds raises the WHOLE of it in ONE plan, never selling a stock that is being bought in the run (R10). A purchase with no typed shares (sized to the
cash) never needs it. A waiting order's cash and shares are not available (entry 140), and RaiseFunds sells whole shares (entry 110). No size limit, risk rule or minimum order was added (R9).

THE DECISION CARRIES IT (R7). A purchase that is to be paid for by selling carries `funding: {"shortfall": Decimal, "plan_id": str | None, "by": "words" | "setting" | "asked"}` and is decided as it would have been with the cash in
hand (`yes_user` for a direct order, the sized decision of the recommendation path); the desk then runs RaiseFunds (`AccountManager.fund_decisions`, the `account_manager_funds` node) and the Trader places the sales FIRST, one trade at a time
through the handshake (entry 81: each sale is approved, filled and booked, and only then the purchase, whose approval reads the cash as it is now), so the cash is the only check at every buy (entry 111: it is never overspent, and a
sale that fills lower leaves the purchase to be cut or refused). If RaiseFunds cannot pay for it, the decision becomes `no_funds` with the shortfall and what the holdings could raise (R8).
"""
from decimal import Decimal

from prometheus.agents.account_manager.intent import denies_raise_funds
from prometheus.common import shares as sh
from prometheus.rag.config import setting

ALWAYS_SETTING = "ALWAYS_RAISE_FUNDS"
TRUE_WORDS = ("1", "true", "yes", "on")
NOT_SAID = "say 'and raise funds if needed' to let me sell holdings to pay for it"
BY = ("words", "setting", "asked")


def always_raise_funds() -> bool:
    """`ALWAYS_RAISE_FUNDS` from the environment, then `.env` (through `rag.config.setting`): true, yes, on or 1 is the same as the words "raise funds if needed" being in every request; unset, or anything else, is false."""
    return setting(ALWAYS_SETTING, "false").strip().lower() in TRUE_WORDS


def permission(words: bool, typed: str | None) -> str | None:
    """How the user permitted selling holdings to pay for the purchase without being asked: "words" (their own words said so: `words` is `TradeIntent.raise_funds`), "setting" (ALWAYS_RAISE_FUNDS, unless the request's own words say not to
    sell: `intent.denies_raise_funds`), or None (nothing permits it: Ask mode asks, Auto mode does not sell)."""
    if words:
        return "words"
    if always_raise_funds() and not denies_raise_funds(typed or ""):
        return "setting"
    return None


def how_words(by: str | None) -> str:
    """The permission in words, for a sentence: "you said to raise funds", "ALWAYS_RAISE_FUNDS is on", "you said yes when I asked"."""
    return {"words": "you said to raise funds", "setting": "the setting ALWAYS_RAISE_FUNDS is on", "asked": "you said yes when I asked"}.get(by or "", "you permitted it")


def sales_words(sales: list[dict]) -> str:
    """The user's own sales of the request, for a sentence: "1 MCD" or "1 MCD and 2 F" (S1, entry 143)."""
    return " and ".join(f"{sh.text(s['shares'])} {s['symbol']}" for s in sales)


def cash_words(cash: Decimal, sales: list[dict] | None) -> str:
    """The money a request has, as one phrase for a sentence (S1, entry 143): "cash $60.14", and when the user's own sale of the same request is placed first and brings cash too, "cash $60.14 plus about $348.61 from your own sale of 1 DPZ".
    `cash` is the account's `cash_avail` AFTER those sales (what the Account Manager works with: the sales' estimated proceeds are in it)."""
    if not sales:
        return f"cash ${cash:,.2f}"
    got = sum((s["amount"] for s in sales), Decimal("0.00"))
    return f"cash ${cash - got:,.2f} plus about ${got:,.2f} from your own sale of {sales_words(sales)}"


def left_out_line(symbols: list[str]) -> str:
    """What the user is told when some purchases of a recommendation run have typed shares and others do not (entry 143, P1): the cash and the money raised go to the shares typed, so a purchase with no number of shares gets nothing."""
    names = symbols[0] if len(symbols) == 1 else ", ".join(symbols[:-1]) + " and " + symbols[-1]
    return (f"{names} {'has' if len(symbols) == 1 else 'have'} no number of shares, so {'it gets' if len(symbols) == 1 else 'they get'} nothing: the cash and the money raised go to the shares you typed. "
            f"Say how many shares of {names} you want if you want {'it' if len(symbols) == 1 else 'them'} bought too.")


def ask_card(cash: Decimal, buys: list[dict], shortfall: Decimal, sales: list[dict] | None = None, left_out: list[str] | None = None) -> dict:
    """The question of Ask mode when the cash does not cover the purchases and nothing permits selling (R3): ONE question for all of them. `buys` are [{symbol, shares, price, cost}] (Decimals); `cash` is the account's available cash NOW and
    `sales` the user's own sales of the same request that are placed first ([{symbol, shares, price, amount}]: entry 143, S1; their proceeds are in the shortfall, not in `cash`). The card is of the kind `raise_funds`: `text` is the Account
    Manager's sentence, `question` what the terminal asks, and the rest is what the page shows. No answer is a no. `left_out` (entry 143, P1) are the purchases of a recommendation run that have no typed shares: they get nothing,
    because the cash and the money raised go to the shares typed, and the question says so."""
    what = " and ".join(f"{sh.text(b['shares'])} {b['symbol']} at ${b['price']:,.2f}" for b in buys)
    total = sum((b["cost"] for b in buys), Decimal("0.00"))
    got = sum((s["amount"] for s in sales or []), Decimal("0.00"))
    text = (f"You have ${cash:,.2f} available" + (f" and your own sale of {sales_words(sales)} brings about ${got:,.2f}" if sales else "") + f"; {what} cost ${total:,.2f}, ${shortfall:,.2f} short. Shall I raise the money by selling some of your holdings? "
            "I will show you what I would sell first." + (f" {left_out_line(left_out)}" if left_out else ""))
    return {"kind": "raise_funds", "text": text, "question": "Raise the money by selling some of your holdings? [y/N] ", "cash": cash, "cost": total, "shortfall": shortfall,
            "buys": [{"symbol": b["symbol"], "shares": b["shares"], "price": b["price"], "cost": b["cost"]} for b in buys],
            **({"left_out": list(left_out)} if left_out else {}),
            **({"sales": [{"symbol": s["symbol"], "shares": s["shares"], "price": s["price"], "amount": s["amount"]} for s in sales], "sales_proceeds": got} if sales else {})}


def decision_funding(shortfall: Decimal, by: str) -> dict:
    """What a decision carries when its purchase is to be paid for by selling (R7): the shortfall of the run, the plan's id (written by the funds node when RaiseFunds made a plan) and how the user permitted it."""
    if by not in BY:
        raise ValueError(f"funding is permitted by one of {', '.join(BY)}")
    return {"shortfall": shortfall, "plan_id": None, "by": by}


def needs_funding(order) -> bool:
    """Whether any decision of this work order is a purchase that is to be paid for by selling and that may trade (`yes_auto`, `yes_user`): the desk then runs RaiseFunds before the Trader. A decision that carries `funding` but
    was denied, expired or is waiting for the user's answer needs nothing."""
    return bool(funded_decisions(order))


def funded_decisions(order) -> list[dict]:
    """The decisions that carry `funding` and may trade now, in the order they are on the work order (the order the Trader's tickets follow)."""
    return [d for d in order.trade_decisions if isinstance(d, dict) and isinstance(d.get("funding"), dict) and d.get("will_trade") in ("yes_auto", "yes_user") and d.get("action") != "reduce"]
