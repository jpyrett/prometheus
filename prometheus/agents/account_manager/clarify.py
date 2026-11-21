"""The Account Manager asks the user how much to buy (design choices entry 85; design/sizing_in_account_manager_spec.md, Phase 2).

    plan = AccountManager.clarify_sizing(order, portfolio, prices, channel, now, balance)       # a `SizingPlan`, or None (the default size is shown, and the order says why)
    choice = SizingChoice(unit="percent", amount=20, scope="total", allocation="even")          # what the user's answer comes to
    plan = plan_choice(choice, purchases, equity, cash)                                         # {balance, budget, wanted, text}: what `decisions.decide_trades` sizes under

WHEN. Ask mode only (Auto never asks), a purchase the Critic approved and that has not run out, and the user's request did not already say how much (`TradeIntent.said_how_much`: no percentage, no dollar amount, no number
of shares, no "as much as I can"). Then, BEFORE anything is sized or sent to the Trader, the Account Manager runs a bounded ReAct loop with its model (the `account_manager` role): THOUGHT, then an ACTION that is one of
`ask_user(question)` or `set_sizing(unit, amount, scope, allocation)`, then the OBSERVATION (the user's answer, as data between <answer> tags, or a refusal between <problem> tags). At most `config.clarify.max_questions`
questions (three).

WHAT THE USER MAY ANSWER (Jacob, 2026-11-14: "what percentage, dollar amount to buy or number of shares"): a PERCENTAGE of the account ("20% of my account"), a DOLLAR amount ("$3,000"), a NUMBER OF SHARES ("10 shares"), or
"as much as I can". `scope` is `each` (for each company) or `total` (over the approved companies, spread `even`ly when the user said to, otherwise `round_robin`: one share of each company in turn, in rank order; design choices entry 106,
Jacob: *"No. its suppose to be round-robin. Not all in one."*: nobody puts everything into the first company any more). An amount for EACH company is each company's own cap.

THE MODEL PHRASES THE QUESTION AND READS THE ANSWER INTO THAT FORM. IT DOES NOTHING ELSE. CODE (this module) checks the form against what the user typed and does ALL the arithmetic:
  * the amount must be a number the user TYPED in an answer (`intent.numbers_in`), the unit must be one their words show (design choices entry 138, Jacob's rule for every number the user types: a % sign or "percent" is a percentage;
    a `$` before the number or the word dollars, bucks or usd after it is MONEY, read by `intent.written_amounts`, the one rule; a bare number with neither, "10" or "10 of them", is a number of SHARES, never dollars), a percentage is above 0 and at
    most 100, shares are a whole number, and what is asked must not cost more than the cash: a number nobody typed, a figure above the cash or a percentage outside 0 to 100 is refused, with the reason, and the model asks again
    (at most `max_problems` times), after which the default size is shown, and the order says so;
  * (design choices entry 111, M3) for a DIRECT order whose account can hold a fraction, dollars or a percentage for EACH stock buy `shares.buys(the amount, its price)`, which may have a fraction, and the purchase is an order in dollars; an answer in shares stays whole;
  * (design choices entry 112) money and shares are `Decimal`: the amount is the Decimal of the digits the model's JSON wrote (a bool and a string are refused), compared EXACTLY with the numbers the user typed (`intent.numbers_in`, Decimals too), the cost of a
    request is `trade_amount`, a percentage of the account is the exact quotient and a dollar amount for each stock is turned into shares by `shares.buys` or the exact whole-share quotient: no tolerance and no float guard;
  * the shares of each purchase are WHOLE shares worked out in code: percent x equity / price, dollars / price, or the shares as given (`plan_choice`); the only thing that can cut it is the cash (`sizing.py`: money), and the
    decision says so. There is no limit of the user's profile to raise (design choices entry 91: the only check is that the funds cover the order).

The facts the model is shown (the cash, the equity, each approved purchase with its price and the most shares the cash buys, what the user's request said) are numbers made by code; no news text, no thesis and no score
reach it. The user's answers go to it between tags and are data, never instructions. A model that cannot be reached, that answers nonsense, or a user who does not answer leaves the DEFAULT size (all the cash, a share of each
company in turn, in rank order) and the Account Manager says so. The user then approves each trade as before (entry 79): this module decides nothing about permission and sends nothing to the Trader.
"""
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, ValidationError, field_validator, model_validator

from prometheus.agents.account_manager import sizing
from prometheus.agents.account_manager.config import AccountManagerConfig
from prometheus.agents.account_manager.intent import EVEN_CUE, NUMBER_WORDS, money_written, numbers_in, written_amounts
from prometheus.common import shares as sh
from prometheus.common.money import Money, canonical, cents, digits, json_number
from prometheus.common.portfolio import trade_amount
from prometheus.common.trace import traced
from prometheus.prompts import account_manager_sizing as prompt

UNITS = ("percent", "dollars", "shares", "max")
SCOPES = ("each", "total")
PERCENT_CUE = re.compile(r"%|\bper ?cent\b|\bpercent\b", re.I)
SHARES_CUE = re.compile(r"\bshares?\b", re.I)
# A number written with no `$`, no word dollars, no `%` and no unit (k, thousand, million): "10", "10 shares", "10 of them", "ten" (entry 138: without a $ or the word dollars a number is a QUANTITY, a number of shares).
BARE_NUMBER = re.compile(rf"(?<![\w$.,])(?P<n>\d[\d,]*(?:\.\d+)?|{'|'.join(NUMBER_WORDS)})(?![\w%])(?![.,]\d)(?!\s*(?:%|percent\b|per ?cent\b|k\b|thousand\b|million\b|dollars?\b|bucks?\b|usd\b))", re.I)
MAX_CUE = re.compile(r"\b(?:as (?:much|many)|all (?:of )?(?:it|my|the|our)|everything|max(?:imum)?|the most|most (?:amount|shares?|money)|the largest|the biggest|whatever (?:you|the|is)|go all in|the limit|fully invest\w*)\b", re.I)
ROBIN_CUE = re.compile(r"\bround[- ]?robin\b|\bin turn\b|\bone (?:share )?(?:at a time|of each)\b|\balternat\w+\b|\btake turns\b", re.I)
YES = re.compile(r"^\s*(?:y|yes|yeah|yep|ok|okay|sure|please|do it|raise it)\b", re.I)
PERCENT_ANSWER = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)?", re.I)

# The channel to the user: a callable that takes one card and returns what the user typed, or None when nobody answers (the terminal's `terminal_clarifier`, the session's).
#   {"question": the text to show, "n": which question this is, "of": the most there may be, "facts": lines to show with the first question (empty after), "kind": "how_much"}
Channel = Callable[[dict], "str | None"]


def _amount(value) -> Decimal:
    """The amount of the model's form as the Decimal of the digits it was written with, in its plain form (20, not 20.0): a bool and a string are refused (design choices entry 112)."""
    return canonical(digits(value))


Amount = Annotated[Decimal, BeforeValidator(_amount), PlainSerializer(json_number, return_type=int | float, when_used="json")]


class SizingChoice(BaseModel):
    """What the user's answer comes to, as the model read it (`set_sizing`): the unit, the amount (None for "as much as I can"), whether it is for each company or in total, and how a total is spread."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: Literal["percent", "dollars", "shares", "max"]
    amount: Amount | None = None
    scope: Literal["each", "total"] = "total"
    allocation: Literal["even", "round_robin"] | None = None      # how a total is spread. None: the user did not say, which is round-robin (entry 106); `even` only when they asked for an equal split

    @model_validator(mode="before")
    @classmethod
    def _a_number_of_shares_is_a_number(cls, data):
        """A count of SHARES is a number: a string ("2") or a bool (true) is an error the model is told about, never turned into 2 or 1 (design choices entry 111, M5). The other units are money and a percentage."""
        if isinstance(data, dict) and data.get("unit") == "shares" and isinstance(data.get("amount"), (str, bool)):
            raise ValueError(f"a number of shares is a number, not {data['amount']!r}: write 2, not \"2\"")
        return data

    @field_validator("allocation", mode="before")
    @classmethod
    def _read_the_retired_word(cls, value):
        """The real replies recorded before entry 106 say `as_much_as_possible` (the first company first, as far as the cash allows), which was "the user did not say how to spread it". It is still read, as that: None."""
        return None if value == "as_much_as_possible" else value


class SizingPlan(BaseModel):
    """What `decisions.decide_trades` sizes under: `balance` ("even", "round_robin" or None: nothing was said, so round-robin, entry 106), `budget` (the dollars of a total), `wanted` ({record_id: whole shares} of an amount for each
    company), `text` (one sentence for the user) and `notes`."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    balance: Literal["even", "round_robin"] | None = None
    budget: Money | None = None
    wanted: dict[str, sh.Shares] | None = None
    in_dollars: bool = Field(default=False, exclude=True)         # (not in the dump: the recorded plans have no such key) `wanted` came from DOLLARS or a PERCENTAGE of the account for EACH stock of a direct order (design choices entry 111, M3): each amount may buy a fraction of a share, and each purchase is an order in dollars
    text: str
    notes: list[str] = Field(default_factory=list)


@dataclass
class Outcome:
    """What the loop came to: the `choice` (None when it gave up, and `why` says why), the questions put to the user, their answers, and what it cost."""
    choice: SizingChoice | None = None
    why: str = ""
    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


# ---------------------------------------------------------------- the check and the arithmetic (code)

def _money(x: Decimal) -> str:
    return f"${x:,.2f}"


def _clean(text, limit: int) -> str:
    """One line of plain text, at most `limit` characters: the model's question and the user's answer pass through here. Control characters go, `<` and `>` become brackets (nothing can close a tag), the spaces are tidied."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).replace("<", "(").replace(">", ")")
    return " ".join(text.split())[:limit]


def _in(amount: Decimal, said: str) -> bool:
    """Whether the amount is a number the user typed (exactly: both are Decimals read from digits)."""
    return amount in numbers_in(said)


def bare_numbers(said: str) -> set[Decimal]:
    """The numbers the user's answer writes as a NUMBER OF SHARES (entry 138): written with no `$` and no word dollars (`intent.written_amounts` finds the ones that have them: one rule for the request and for the answer), no `%` and no
    unit of thousands or millions. "10" and "10 of them" are shares; "$10", "10 dollars" and "10 bucks" are money and "10%" a percentage, and none of those is in the set. A program's reading of the user's own words."""
    marked = [(w.start, w.end) for w in written_amounts(said)]
    found = set()
    for match in BARE_NUMBER.finditer(said):
        if any(a <= match.start("n") < b for a, b in marked):
            continue
        word = match["n"].lower()
        found.add(Decimal(NUMBER_WORDS[word]) if word in NUMBER_WORDS else Decimal(word.replace(",", "")))
    return found


def _order(purchases: list[sizing.Purchase]) -> list[sizing.Purchase]:
    return [p for _, p in sorted(enumerate(purchases), key=lambda pair: (pair[1].rank, pair[0]))]


def needed_dollars(choice: SizingChoice, purchases: list[sizing.Purchase], equity: Decimal | None) -> Decimal | None:
    """What the request costs at the prices, in dollars: None when it cannot be worked out (a percentage of an account with no value, a purchase with no price). `max` costs nothing here (the cash decides)."""
    k = len(purchases)
    if choice.unit == "max":
        return Decimal("0.00")
    if choice.unit == "dollars":
        return choice.amount * (k if choice.scope == "each" else 1)
    if choice.unit == "percent":
        return None if equity is None else choice.amount / 100 * equity * (k if choice.scope == "each" else 1)
    if any(p.price is None or p.price <= 0 for p in purchases):
        return None
    if choice.scope == "each":
        return sum((trade_amount(choice.amount, p.price) for p in purchases), Decimal("0.00"))      # money from a count and a price is `trade_amount` (entries 111 and 112)
    return sum((trade_amount(n, p.price) for n, p in zip(_split_shares(choice.amount, purchases), _order(purchases))), Decimal("0.00"))


def check_choice(choice: SizingChoice, said: str, purchases: list[sizing.Purchase], cash: Decimal, equity: Decimal | None, prior_balance: str | None = None) -> None:
    """Raises ValueError, with the reason, unless the user's own answers (`said`) support the choice and the account can pay for it (see the module text). The reason is shown to the model, never believed by it.
    `prior_balance` is a split the user already asked for in the request ("split it evenly")."""
    if not purchases:
        raise ValueError("there is nothing approved to buy")
    spread = len(purchases) > 1 and (choice.scope == "total" or choice.unit == "max")      # only a total over several purchases is spread; the default is round-robin and needs no word from the user (entry 106)
    if spread and choice.allocation == "even" and not (EVEN_CUE.search(said) or prior_balance == "even"):
        raise ValueError("the user did not ask for an even split: leave allocation out unless they said to split evenly")
    if spread and choice.allocation == "round_robin" and prior_balance == "even" and not ROBIN_CUE.search(said):
        raise ValueError("the request already asked for an even split: use allocation even, or leave it out")
    if choice.unit == "max":
        if choice.amount is not None:
            raise ValueError("`max` takes no amount")
        if not MAX_CUE.search(said):
            raise ValueError("the user did not say they want as much as they can: ask them, or ask for an amount")
        return
    amount = choice.amount
    if amount is None or amount <= 0:
        raise ValueError(f"{choice.unit} needs an amount above zero")
    if not said.strip():
        raise ValueError("the user has not answered anything yet: ask them how much first")
    if not _in(amount, said):
        raise ValueError(f"the number {amount:f} is not in anything the user typed: use a number from their own answer")
    if choice.unit == "percent":
        if not PERCENT_CUE.search(said):
            raise ValueError("the user did not give a percentage (no % or 'percent' in their answer)")
        if amount > 100:
            raise ValueError("a percentage of the account is above 0 and at most 100")
    elif choice.unit == "shares":
        if amount not in bare_numbers(said):
            raise ValueError("the user did not give a number of shares: a number of shares is a number written with no $, no % and no word dollars ('10', '10 shares', '10 of them')")
        if amount != amount.to_integral_value():
            raise ValueError("shares are a whole number")
    elif amount not in money_written(said):
        if PERCENT_CUE.search(said) or SHARES_CUE.search(said):
            raise ValueError("the user gave percent or shares, not dollars")
        raise ValueError(f"the number {amount:f} has no $ and no word dollars (or bucks): written that way it is a number of SHARES, never dollars. Use unit shares, or ask whether they mean dollars")
    if choice.unit == "percent" and equity is None:
        raise ValueError("the value of the account is not known, so a percentage of it cannot be worked out")
    cost = needed_dollars(choice, purchases, equity)
    if cost is None:
        raise ValueError("what that costs cannot be worked out (a price is missing)")
    if cents(cost) > cash:      # the cost to the cent, the way the account book rounds money
        raise ValueError(f"that comes to {_money(cost)} and the cash is {_money(cash)}: ask for less")


def _split_shares(total: Decimal, purchases: list[sizing.Purchase]) -> list[Decimal]:
    """`total` whole shares over the purchases in rank order: each gets total // k and the first `total % k` one more. (The cash cuts each afterwards; what it cuts is not moved to another purchase.)"""
    k = len(purchases)
    return [total // k + (1 if n < total % k else 0) for n in range(k)]


def plan_choice(choice: SizingChoice, purchases: list[sizing.Purchase], equity: Decimal | None, cash: Decimal, prior_balance: str | None = None, fractional: bool = False) -> SizingPlan:
    """The choice as what `decisions.decide_trades` sizes under (see the module text). Whole shares are worked out here, in code: percent x equity / price, dollars / price, the shares as given. `prior_balance` is a
    split the user already asked for in the request ("split it evenly"): a total with no allocation of its own keeps it. `fractional` (a direct order with the account able to hold a fraction, design choices entry 111, M3): an
    amount in dollars or a percentage for EACH stock buys `shares.buys(the amount, its price)`, which may have a fraction (and the plan is `in_dollars`); an answer in shares stays a whole number. Raises ValueError for what
    `check_choice` refuses (it is called first by the loop)."""
    ordered, k = _order(purchases), len(purchases)
    if choice.unit == "max":
        balance = choice.allocation or prior_balance      # None: nothing was said, and the sizing's default (round-robin, entry 106) applies
        return SizingPlan(balance=balance, text="as much as your cash allows" + ((", spread evenly" if balance == "even" else ", a share of each company in turn") if k > 1 else ""))
    unit_text = {"percent": f"{choice.amount:f}% of your account", "dollars": _money(choice.amount), "shares": f"{sh.shown(choice.amount)} {sh.noun(choice.amount)}"}[choice.unit]      # the amount is in its plain form (20, 2.5); an answer in shares is a whole number (entry 110: the how-much answer stays whole)
    if choice.scope == "each":
        wanted: dict[str, Decimal] = {}
        for p in ordered:
            if choice.unit == "shares":
                wanted[p.record_id] = choice.amount
            else:
                dollars = choice.amount / 100 * equity if choice.unit == "percent" else choice.amount
                wanted[p.record_id] = (sh.buys(cents(dollars), p.price) if fractional else max(Decimal(0), (dollars // p.price))) if p.price else Decimal(0)
        return SizingPlan(wanted=wanted, in_dollars=fractional and choice.unit != "shares", text=f"{unit_text} in each of the {k} compan{'ies' if k != 1 else 'y'}" if k != 1 else f"{unit_text} in {ordered[0].symbol}")
    if choice.unit == "shares":
        wanted = {p.record_id: n for n, p in zip(_split_shares(choice.amount, ordered), ordered)}
        return SizingPlan(wanted=wanted, text=f"{unit_text} in all, shared out over {k} compan{'ies' if k != 1 else 'y'} in rank order" if k != 1 else unit_text + f" of {ordered[0].symbol}")
    dollars = choice.amount / 100 * equity if choice.unit == "percent" else choice.amount
    budget = min(cash, cents(dollars))
    balance = choice.allocation or prior_balance      # the user's own word, else the split the request already asked for, else None: the sizing's default, round-robin, never the first company first (entry 106)
    how = "spread evenly" if balance == "even" else "a share of each company in turn"
    return SizingPlan(balance=balance, budget=budget, text=(f"{unit_text} ({_money(budget)}) in all" if choice.unit == "percent" else f"{_money(budget)} in all") + (f", {how}" if k > 1 else ""))


def plan_each_dollars(amount: Decimal, purchases: list[sizing.Purchase], cash: Decimal) -> SizingPlan:
    """The plan of a request that SAID "$3,000 each" for the purchases the desk decides (design choices entry 125; no question is asked: the user already said). It is the plan of the answer "an amount for each company"
    (`plan_choice` with scope `each`: each purchase's `wanted` is the whole shares its own dollars buy, which is its own cap), and the POT of the run is the amount times the number of purchases that can be sized (`sizing.pot_of`: never
    above the cash), so a run never spends more than the amount for each stock. Nothing is spread evenly: every purchase has its own cap."""
    plan = plan_choice(SizingChoice(unit="dollars", amount=amount, scope="each"), purchases, None, cash)
    sized = sum(1 for p in purchases if p.price)
    return plan.model_copy(update={"budget": sizing.pot_of(cash, amount * sized) if sized and cash > 0 else None})


# ---------------------------------------------------------------- the facts the model is shown, and the loop

def facts_for(purchases: list[sizing.Purchase], sized: list[sizing.Sized], cash: Decimal, equity: Decimal | None, request: str, prior_balance: str | None, config: AccountManagerConfig) -> dict:
    """The facts for the model: numbers made by code (the cash, the equity, each approved purchase with its price and what the cash buys of it), and the user's own request. Nothing from the news, no thesis, no score."""
    by_id = {s.record_id: s for s in sized}
    return {"cash": cents(cash), "equity": None if equity is None else cents(equity),
            "approved_purchases": [{"symbol": p.symbol, "price": p.price, "most_shares_your_cash_buys": by_id[p.record_id].shares, "most_dollars": by_id[p.record_id].est_cost, "limited_by": by_id[p.record_id].binding}
                                   for p in _order(purchases)],
            "request_said_a_split": prior_balance, "request": _clean(request, 600), "questions_you_may_ask": config.clarify.max_questions}


def facts_lines(facts: dict) -> list[str]:
    """The facts for the USER, to show with the first question: what they can spend and what the cash buys of each approved company."""
    lines = [f"Cash {_money(facts['cash'])}" + (f", the whole account {_money(facts['equity'])}" if facts["equity"] else "") + ". Approved to buy:"]
    for p in facts["approved_purchases"]:
        lines.append(f"  {p['symbol']:<6} at {_money(p['price'])}: your cash buys up to {sh.shown(p['most_shares_your_cash_buys'])} shares (about {_money(p['most_dollars'])}), held back by {p['limited_by'] or 'nothing'}")
    return lines


def _json(reply: str) -> dict:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (reply or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        raw = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if not isinstance(raw, dict):
        raise ValueError("not a JSON object")
    return raw


def _action(raw: dict) -> tuple[str, dict]:
    """(`ask_user` or `set_sizing`, its fields) from a reply; ValueError with the reason when it is neither, or has a field it should not have."""
    action = raw.get("action")
    if action == "ask_user":
        if set(raw) - {"thought", "action", "question"}:
            raise ValueError(f"ask_user takes only a question, not {', '.join(sorted(set(raw) - {'thought', 'action', 'question'}))}")
        if not isinstance(raw.get("question"), str):
            raise ValueError("ask_user needs a question, as text")
        return action, {"question": raw["question"]}
    if action == "set_sizing":
        extra = set(raw) - {"thought", "action", "unit", "amount", "scope", "allocation"}
        if extra:
            raise ValueError(f"set_sizing has no field {', '.join(sorted(extra))}")
        return action, {k: raw[k] for k in ("unit", "amount", "scope", "allocation") if k in raw and raw[k] is not None}
    raise ValueError("action must be ask_user or set_sizing")


@traced("account_manager.clarify_loop")
def run_loop(model, facts: dict, purchases: list[sizing.Purchase], cash: Decimal, equity: Decimal | None, channel: Channel, config: AccountManagerConfig, prior_balance: str | None = None) -> Outcome:
    """The ReAct loop (see the module text). Never raises: a model that fails, a reply that is nonsense three times, or a user who does not answer ends it with `choice` None and `why`."""
    cfg, out = config.clarify, Outcome()
    messages = [SystemMessage(content=prompt.SYSTEM), HumanMessage(content=prompt.facts_message(facts, cfg.max_questions))]
    shown = facts_lines(facts)
    for _ in range(cfg.max_questions + cfg.max_problems + 2):                    # every turn is a question, a refusal or the end: this bounds them all
        try:
            reply = model.invoke(messages)
        except Exception as error:                                                # a model or network failure: the kind only
            out.why = f"the Account Manager's model could not be reached ({type(error).__name__})"
            return out
        meta = getattr(reply, "usage_metadata", None) or {}
        out.calls, out.tokens_in, out.tokens_out = out.calls + 1, out.tokens_in + meta.get("input_tokens", 0), out.tokens_out + meta.get("output_tokens", 0)
        content = reply.content if isinstance(reply.content, str) else ""
        try:
            action, fields = _action(_json(content))
            if action == "ask_user":
                question = _clean(fields["question"], cfg.max_question_chars)
                if not question:
                    raise ValueError("the question is empty")
                if len(out.questions) >= cfg.max_questions:
                    raise ValueError(f"no questions are left ({cfg.max_questions} were asked): call set_sizing with what the user said")
            else:
                choice = SizingChoice(**fields)
                check_choice(choice, " ".join(out.answers), purchases, cash, equity, prior_balance)
                out.choice = choice
                return out
        except (ValueError, ValidationError, TypeError) as error:
            problem = _clean(error.errors()[0]["msg"] if isinstance(error, ValidationError) else str(error), 300)
            out.problems.append(problem)
            if len(out.problems) > cfg.max_problems:
                out.why = f"the answers could not be read into an amount ({problem})"
                return out
            messages = [*messages, AIMessage(content=content), HumanMessage(content=prompt.problem_message(problem))]
            continue
        out.questions.append(question)
        answer = channel({"question": question, "n": len(out.questions), "of": cfg.max_questions, "facts": shown if len(out.questions) == 1 else [], "kind": "how_much"})
        if answer is None or not str(answer).strip():
            out.why = "nobody answered the question"
            return out
        answer = _clean(answer, cfg.max_answer_chars)
        out.answers.append(answer)
        messages = [*messages, AIMessage(content=content), HumanMessage(content=prompt.answer_message(answer, cfg.max_questions - len(out.questions)))]
    out.why = "the loop ran out of turns"
    return out


# ---------------------------------------------------------------- the terminal

def terminal_clarifier(ask=input, show=print, interactive: bool | None = None) -> Channel:
    """The channel to a person at the terminal: shows the facts with the first question (the cash, each approved company and what the cash buys), asks the question and returns what they typed. Nobody there
    (`interactive` False; by default `sys.stdin.isatty()`) means `available()` is False and the Account Manager does not run the loop at all (no model call, no question); a question that gets no answer (end of input)
    returns None. The same shape as `decisions.terminal_approver`."""
    import sys

    def here() -> bool:
        return sys.stdin.isatty() if interactive is None else interactive

    def channel(card: dict) -> str | None:
        if not here():
            return None
        if card.get("facts"):
            show("\n".join(["", "=" * 118, "HOW MUCH?  The Account Manager asks before it proposes the trades. Answer with a percentage of your account (20%), dollars (with a $ or the word dollars: $3,000), a number of shares (a number with no $ is shares: 10 is 10 shares), or \"as much as I can\".", "=" * 118,
                            *card["facts"]]))
        suffix = f"  (question {card['n']} of {card['of']})"
        try:
            return ask(f"\n  {card['question']}{suffix}\n  > ")
        except EOFError:
            return None

    channel.available = here      # the Account Manager asks nobody who is not there
    return channel
