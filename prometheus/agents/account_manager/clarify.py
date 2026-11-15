"""The Account Manager asks the user how much to buy (design choices entry 85; design/sizing_in_account_manager_spec.md, Phase 2).

    plan = AccountManager.clarify_sizing(order, portfolio, prices, channel, now, balance)       # a `SizingPlan`, or None (the default size is shown, and the order says why)
    choice = SizingChoice(unit="percent", amount=20, scope="total", allocation="even")          # what the user's answer comes to
    plan = plan_choice(choice, purchases, equity, cash)                                         # {balance, budget, wanted, text}: what `decisions.decide_trades` sizes under

WHEN. Ask mode only (Auto never asks), a purchase the Critic approved and that has not run out, and the user's request did not already say how much (`TradeIntent.said_how_much`: no percentage, no dollar amount, no number
of shares, no "as much as I can"). Then, BEFORE anything is sized or sent to the Trader, the Account Manager runs a bounded ReAct loop with its model (the `account_manager` role): THOUGHT, then an ACTION that is one of
`ask_user(question)` or `set_sizing(unit, amount, scope, allocation)`, then the OBSERVATION (the user's answer, as data between <answer> tags, or a refusal between <problem> tags). At most `config.clarify.max_questions`
questions (three).

WHAT THE USER MAY ANSWER (Jacob, 2026-11-14: "what percentage, dollar amount to buy or number of shares"): a PERCENTAGE of the account ("20% of my account"), a DOLLAR amount ("$3,000"), a NUMBER OF SHARES ("10 shares"), or
"as much as I can". `scope` is `each` (for each company) or `total` (over the approved companies, spread `even`ly, `round_robin` or as much as the limits allow in rank order).

THE MODEL PHRASES THE QUESTION AND READS THE ANSWER INTO THAT FORM. IT DOES NOTHING ELSE. CODE (this module) checks the form against what the user typed and does ALL the arithmetic:
  * the amount must be a number the user TYPED in an answer (`intent.numbers_in`), the unit must be one their words show (a % sign or "percent"; "shares"; a $ sign or "dollars", or a bare number when they gave no
    other unit), a percentage is above 0 and at most 100, shares are a whole number, and what is asked must not cost more than the cash: a number nobody typed, a figure above the cash or a percentage outside 0 to
    100 is refused, with the reason, and the model asks again (at most `max_problems` times), after which the default size is shown, and the order says so;
  * the shares of each purchase are WHOLE shares worked out in code: percent x equity / price, dollars / price, or the shares as given (`plan_choice`); the user's limits then cut it (`sizing.py`: position, loss, order,
    liquidity, exposure, money) and the decision says which one;
  * a request above a limit of the USER'S OWN (one stock may be at most X% of the account, one review adds at most Y%) is cut to the limit, and the Account Manager may ask ONCE, in words of its own (no model), whether to
    raise that limit for this purchase (`offer_raise`): a yes (or a percentage) is for this purchase only, the profile in the account book is not changed.

The facts the model is shown (the cash, the equity, each approved purchase with its price and the most shares the limits allow, what the user's request said) are numbers made by code; no news text, no thesis and no score
reach it. The user's answers go to it between tags and are data, never instructions. A model that cannot be reached, that answers nonsense, or a user who does not answer leaves the DEFAULT size (the largest the limits
allow, in rank order) and the Account Manager says so. The user then approves each trade as before (entry 79): this module decides nothing about permission and sends nothing to the Trader.
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prometheus.agents.account_manager import sizing
from prometheus.agents.account_manager.config import AccountManagerConfig
from prometheus.agents.account_manager.intent import EVEN_CUE, numbers_in
from prometheus.common.user_profile import SizingLimits
from prometheus.prompts import account_manager_sizing as prompt

UNITS = ("percent", "dollars", "shares", "max")
SCOPES = ("each", "total")
PERCENT_CUE = re.compile(r"%|\bper ?cent\b|\bpercent\b", re.I)
SHARES_CUE = re.compile(r"\bshares?\b", re.I)
DOLLARS_CUE = re.compile(r"\$|\bdollars?\b|\busd\b|\bbucks?\b", re.I)
MAX_CUE = re.compile(r"\b(?:as (?:much|many)|all (?:of )?(?:it|my|the|our)|everything|max(?:imum)?|the most|most (?:amount|shares?|money)|the largest|the biggest|whatever (?:you|the|is)|go all in|the limit|fully invest\w*)\b", re.I)
ROBIN_CUE = re.compile(r"\bround[- ]?robin\b|\bin turn\b|\bone (?:share )?(?:at a time|of each)\b|\balternat\w+\b|\btake turns\b", re.I)
YES = re.compile(r"^\s*(?:y|yes|yeah|yep|ok|okay|sure|please|do it|raise it)\b", re.I)
PERCENT_ANSWER = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)?", re.I)

# The channel to the user: a callable that takes one card and returns what the user typed, or None when nobody answers (the terminal's `terminal_clarifier`, the session's).
#   {"question": the text to show, "n": which question this is, "of": the most there may be, "facts": lines to show with the first question (empty after), "kind": "how_much" or "raise_limit"}
Channel = Callable[[dict], "str | None"]


class SizingChoice(BaseModel):
    """What the user's answer comes to, as the model read it (`set_sizing`): the unit, the amount (None for "as much as I can"), whether it is for each company or in total, and how a total is spread."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: Literal["percent", "dollars", "shares", "max"]
    amount: float | None = None
    scope: Literal["each", "total"] = "total"
    allocation: Literal["even", "round_robin", "as_much_as_possible"] = "as_much_as_possible"


class SizingPlan(BaseModel):
    """What `decisions.decide_trades` sizes under: `balance` ("even", "round_robin" or None: as much as the limits allow), `budget` (the dollars of a total), `wanted` ({record_id: whole shares} of an amount for each
    company), `limit_overrides` (limits the user raised for this purchase), `text` (one sentence for the user) and `notes`."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    balance: Literal["even", "round_robin"] | None = None
    budget: float | None = None
    wanted: dict[str, int] | None = None
    limit_overrides: dict[str, float] | None = None
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

def _money(x: float) -> str:
    return f"${x:,.2f}"


def _clean(text, limit: int) -> str:
    """One line of plain text, at most `limit` characters: the model's question and the user's answer pass through here. Control characters go, `<` and `>` become brackets (nothing can close a tag), the spaces are tidied."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).replace("<", "(").replace(">", ")")
    return " ".join(text.split())[:limit]


def _in(amount: float, said: str) -> bool:
    return any(math.isclose(amount, n, rel_tol=1e-9, abs_tol=0.005) for n in numbers_in(said))


def _order(purchases: list[sizing.Purchase]) -> list[sizing.Purchase]:
    return [p for _, p in sorted(enumerate(purchases), key=lambda pair: (pair[1].rank, pair[0]))]


def needed_dollars(choice: SizingChoice, purchases: list[sizing.Purchase], equity: float | None) -> float | None:
    """What the request costs at the prices, in dollars: None when it cannot be worked out (a percentage of an account with no value, a purchase with no price). `max` costs nothing here (the limits decide)."""
    k = len(purchases)
    if choice.unit == "max":
        return 0.0
    if choice.unit == "dollars":
        return choice.amount * (k if choice.scope == "each" else 1)
    if choice.unit == "percent":
        return None if equity is None else choice.amount / 100.0 * equity * (k if choice.scope == "each" else 1)
    if any(p.price is None or p.price <= 0 for p in purchases):
        return None
    if choice.scope == "each":
        return sum(choice.amount * p.price for p in purchases)
    return sum(n * p.price for n, p in zip(_split_shares(int(choice.amount), purchases), _order(purchases)))


def check_choice(choice: SizingChoice, said: str, purchases: list[sizing.Purchase], cash: float, equity: float | None, prior_balance: str | None = None) -> None:
    """Raises ValueError, with the reason, unless the user's own answers (`said`) support the choice and the account can pay for it (see the module text). The reason is shown to the model, never believed by it.
    `prior_balance` is a split the user already asked for in the request ("split it evenly")."""
    if not purchases:
        raise ValueError("there is nothing approved to buy")
    if choice.allocation != "as_much_as_possible" and len(purchases) > 1 and choice.scope == "total" or (choice.unit == "max" and choice.allocation != "as_much_as_possible" and len(purchases) > 1):
        if choice.allocation == "even" and not (EVEN_CUE.search(said) or prior_balance == "even"):
            raise ValueError("the user did not ask for an even split: use allocation as_much_as_possible unless they said how to spread it")
        if choice.allocation == "round_robin" and not (ROBIN_CUE.search(said) or prior_balance == "round_robin"):
            raise ValueError("the user did not ask for one share of each in turn: use allocation as_much_as_possible unless they said how to spread it")
    if choice.unit == "max":
        if choice.amount is not None:
            raise ValueError("`max` takes no amount")
        if not MAX_CUE.search(said):
            raise ValueError("the user did not say they want as much as they can: ask them, or ask for an amount")
        return
    amount = choice.amount
    if amount is None or isinstance(amount, bool) or not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"{choice.unit} needs an amount above zero")
    if not said.strip():
        raise ValueError("the user has not answered anything yet: ask them how much first")
    if not _in(amount, said):
        raise ValueError(f"the number {amount:g} is not in anything the user typed: use a number from their own answer")
    if choice.unit == "percent":
        if not PERCENT_CUE.search(said):
            raise ValueError("the user did not give a percentage (no % or 'percent' in their answer)")
        if amount > 100:
            raise ValueError("a percentage of the account is above 0 and at most 100")
    elif choice.unit == "shares":
        if not SHARES_CUE.search(said):
            raise ValueError("the user did not give a number of shares (no 'shares' in their answer)")
        if amount != int(amount):
            raise ValueError("shares are a whole number")
    elif not (DOLLARS_CUE.search(said) or not (PERCENT_CUE.search(said) or SHARES_CUE.search(said))):
        raise ValueError("the user gave percent or shares, not dollars")
    if choice.unit == "percent" and equity is None:
        raise ValueError("the value of the account is not known, so a percentage of it cannot be worked out")
    cost = needed_dollars(choice, purchases, equity)
    if cost is None:
        raise ValueError("what that costs cannot be worked out (a price is missing)")
    if cost > cash + 0.005:
        raise ValueError(f"that comes to {_money(cost)} and the cash is {_money(cash)}: ask for less")


def _split_shares(total: int, purchases: list[sizing.Purchase]) -> list[int]:
    """`total` shares over the purchases in rank order: each gets total // k and the first `total % k` one more. (The limits cut each afterwards; what a limit cuts is not moved to another purchase.)"""
    k = len(purchases)
    return [total // k + (1 if n < total % k else 0) for n in range(k)]


def plan_choice(choice: SizingChoice, purchases: list[sizing.Purchase], equity: float | None, cash: float, prior_balance: str | None = None, guard: float = 1e-9) -> SizingPlan:
    """The choice as what `decisions.decide_trades` sizes under (see the module text). Whole shares are worked out here, in code: percent x equity / price, dollars / price, the shares as given. `prior_balance` is a
    split the user already asked for in the request ("split it evenly"): a total with no allocation of its own keeps it. Raises ValueError for what `check_choice` refuses (it is called first by the loop)."""
    ordered, k = _order(purchases), len(purchases)
    if choice.unit == "max":
        balance = {"even": "even", "round_robin": "round_robin"}.get(choice.allocation) or prior_balance
        return SizingPlan(balance=balance, text="as much as your limits allow" + (", spread evenly" if balance == "even" else ", a share of each in turn" if balance == "round_robin" else ", the first company first"))
    unit_text = {"percent": f"{choice.amount:g}% of your account", "dollars": _money(choice.amount), "shares": f"{choice.amount:g} share{'s' if choice.amount != 1 else ''}"}[choice.unit]
    if choice.scope == "each":
        wanted: dict[str, int] = {}
        for p in ordered:
            if choice.unit == "shares":
                wanted[p.record_id] = int(choice.amount)
            else:
                dollars = choice.amount / 100.0 * equity if choice.unit == "percent" else choice.amount
                wanted[p.record_id] = max(0, math.floor(dollars / p.price + guard)) if p.price else 0
        return SizingPlan(wanted=wanted, text=f"{unit_text} in each of the {k} compan{'ies' if k != 1 else 'y'}" if k != 1 else f"{unit_text} in {ordered[0].symbol}")
    if choice.unit == "shares":
        wanted = {p.record_id: n for n, p in zip(_split_shares(int(choice.amount), ordered), ordered)}
        return SizingPlan(wanted=wanted, text=f"{unit_text} in all, shared out over {k} compan{'ies' if k != 1 else 'y'} in rank order" if k != 1 else unit_text + f" of {ordered[0].symbol}")
    dollars = choice.amount / 100.0 * equity if choice.unit == "percent" else choice.amount
    budget = min(float(cash), round(dollars, 2))
    balance = {"even": "even", "round_robin": "round_robin"}.get(choice.allocation) or (prior_balance if k > 1 else None)
    how = "spread evenly" if balance == "even" else "a share of each in turn" if balance == "round_robin" else "the first company first, as far as your limits allow"
    return SizingPlan(balance=balance, budget=budget, text=(f"{unit_text} ({_money(budget)}) in all" if choice.unit == "percent" else f"{_money(budget)} in all") + (f", {how}" if k > 1 else ""))


# ---------------------------------------------------------------- the question about raising a limit (code: no model)

RAISABLE = {"position": "max_position_pct", "exposure": "max_new_exposure_pct"}      # the limits of the user's that are a share of the account: the ones offered for raising


def offer_raise(plan: SizingPlan, purchases: list[sizing.Purchase], equity: float | None, cash: float, limits: SizingLimits, config: AccountManagerConfig, channel: Channel) -> SizingPlan:
    """When the user's own limit (one stock at most X% of the account, one review at most Y%) cut what they asked for, ask ONCE whether to raise it for this purchase, and return the plan with the answer in
    `limit_overrides`. The question is worded here (no model). A yes raises the limit to what the request needs, a percentage raises it to that, and anything else leaves the limits as they are. The profile in the
    account book is never changed. The plan is returned as it was when the request was not cut by those limits, nobody answers, or the answer is no."""
    if equity is None or not purchases or (plan.budget is None and plan.wanted is None):
        return plan                                                                    # nothing was asked for: the default is already the most the limits allow
    sized = sizing.size_purchases(purchases, equity, cash, limits, config, allocation=plan.balance, budget=plan.budget, wanted=plan.wanted)
    prices = [p.price for p in purchases if p.price]
    if plan.wanted is not None:
        cut = [(p, s) for p, s in zip(purchases, sized) if s.binding in RAISABLE and s.shares < plan.wanted.get(p.record_id, 0)]
        asked_dollars = sum(plan.wanted.get(p.record_id, 0) * p.price for p in purchases if p.price)
    else:
        cut = [(p, s) for p, s in zip(purchases, sized) if s.binding in RAISABLE] if prices and plan.budget - sum(s.est_cost for s in sized) >= min(prices) else []
        asked_dollars = plan.budget
    if not cut:
        return plan
    bound = {s.binding for _, s in cut}
    overrides: dict[str, float] = {}
    if "position" in bound:
        def target(p):      # the dollars the request would put in this one stock
            if plan.wanted is not None:
                return plan.wanted.get(p.record_id, 0) * p.price
            return plan.budget / len(purchases) if plan.balance else plan.budget
        need = max((p.held_shares * p.price + target(p)) / equity for p, s in cut if s.binding == "position" and p.price)
        overrides["max_position_pct"] = min(1.0, math.ceil(need * 1000) / 1000)
    if "exposure" in bound:
        overrides["max_new_exposure_pct"] = min(1.0, math.ceil(asked_dollars / equity * 1000) / 1000)
    overrides = {k: v for k, v in overrides.items() if v > getattr(limits, k)}
    if not overrides:
        return plan
    describe = {"max_position_pct": ("one stock may be at most", limits.max_position_pct, "one stock"), "max_new_exposure_pct": ("one review may add at most", limits.max_new_exposure_pct, "one review")}
    cutting = " and ".join(f"{describe[k][0]} {describe[k][1]:.0%} of your account (about {_money(describe[k][1] * equity)})" for k in overrides)
    proposal = " and ".join(f"{describe[k][2]} to {v:.1%}" for k, v in overrides.items())
    bought = ", ".join(f"{s.symbol} {s.shares}" for _, s in cut)
    answer = channel({"question": f"Your limits cut what you asked for: {cutting}. With them the Account Manager would buy {bought}. Raise {proposal} for THIS purchase only (your saved limits stay as they are)? "
                                  "Say yes, a percentage of your account, or no.", "n": 1, "of": 1, "facts": [], "kind": "raise_limit"})
    if answer is None or not str(answer).strip():
        return plan
    answer = _clean(answer, config.clarify.max_answer_chars)
    if YES.match(answer):
        chosen = overrides
    elif (m := PERCENT_ANSWER.fullmatch(answer.strip())) and 0 < float(m.group(1)) <= 100:
        chosen = {k: float(m.group(1)) / 100.0 for k in overrides}
    else:
        return plan
    chosen = {k: v for k, v in chosen.items() if v > getattr(limits, k)}
    if not chosen:
        return plan
    nice = " and ".join(f"{describe[k][2]} up to {v:.1%} of your account" for k, v in chosen.items())
    return plan.model_copy(update={"limit_overrides": chosen, "notes": [*plan.notes, f"you raised your limit for this purchase: {nice}"]})


# ---------------------------------------------------------------- the facts the model is shown, and the loop

def facts_for(purchases: list[sizing.Purchase], sized: list[sizing.Sized], cash: float, equity: float | None, request: str, prior_balance: str | None, config: AccountManagerConfig) -> dict:
    """The facts for the model: numbers made by code (the cash, the equity, each approved purchase with its price and what the user's limits let it be), and the user's own request. Nothing from the news, no thesis, no score."""
    by_id = {s.record_id: s for s in sized}
    return {"cash": round(cash, 2), "equity": None if equity is None else round(equity, 2),
            "approved_purchases": [{"symbol": p.symbol, "price": p.price, "most_shares_your_limits_allow": by_id[p.record_id].shares, "most_dollars": by_id[p.record_id].est_cost, "limited_by": by_id[p.record_id].binding}
                                   for p in _order(purchases)],
            "request_said_a_split": prior_balance, "request": _clean(request, 600), "questions_you_may_ask": config.clarify.max_questions}


def facts_lines(facts: dict) -> list[str]:
    """The facts for the USER, to show with the first question: what they can spend and what the limits allow for each approved company."""
    lines = [f"Cash {_money(facts['cash'])}" + (f", the whole account {_money(facts['equity'])}" if facts["equity"] else "") + ". Approved to buy:"]
    for p in facts["approved_purchases"]:
        lines.append(f"  {p['symbol']:<6} at {_money(p['price'])}: your limits allow up to {p['most_shares_your_limits_allow']} shares (about {_money(p['most_dollars'])}), held back by {p['limited_by'] or 'nothing'}")
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


def run_loop(model, facts: dict, purchases: list[sizing.Purchase], cash: float, equity: float | None, channel: Channel, config: AccountManagerConfig, prior_balance: str | None = None) -> Outcome:
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
    """The channel to a person at the terminal: shows the facts with the first question (the cash, each approved company and what the limits allow), asks the question and returns what they typed. Nobody there
    (`interactive` False; by default `sys.stdin.isatty()`) means `available()` is False and the Account Manager does not run the loop at all (no model call, no question); a question that gets no answer (end of input)
    returns None. The same shape as `decisions.terminal_approver`."""
    import sys

    def here() -> bool:
        return sys.stdin.isatty() if interactive is None else interactive

    def channel(card: dict) -> str | None:
        if not here():
            return None
        if card.get("facts"):
            show("\n".join(["", "=" * 118, "HOW MUCH?  The Account Manager asks before it proposes the trades. Answer with a percentage of your account, dollars, a number of shares, or \"as much as I can\".", "=" * 118,
                            *card["facts"]]))
        suffix = "" if card.get("kind") == "raise_limit" else f"  (question {card['n']} of {card['of']})"
        try:
            return ask(f"\n  {card['question']}{suffix}\n  > ")
        except EOFError:
            return None

    channel.available = here      # the Account Manager asks nobody who is not there
    return channel
