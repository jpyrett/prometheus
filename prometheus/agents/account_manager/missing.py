"""The Account Manager and a stock the Data Clerk did not find (design choices entry 102; the Data Clerk's side is entry 101).

    asked = ask_about_each(model, missing, kept, request, approver, config)       # ASK mode: one question for each missing stock, in the order the user typed them
    asked.dropped, asked.stopped                                                  # what the user said yes to leaving out, or why the order stops

A direct order is a list of stocks, and the Data Clerk may not find one of them ("Chick-fil-a": privately held). Jacob (2026-11-15): *"Missing companies in a list should keep going and ask the user if they want to ignore the
missing companies and continue (ReAct pattern)"*, and: *"Auto mode will not ask any prompts. Stop and ask the user to clarify the initial prompt if there is a blocking issue. Remove missing (not found / unmatched)
companies if in auto mode"*. So, once the Data Clerk has answered, every stock the user named is one of three (`direct.sort_names`): found, not listed (the directory answered and has no such company), or not answered (the
lookup failed). In AUTO mode the not listed ones are dropped without a question and the user is told; in ASK mode (this module) the user is asked about each of them; a blocking issue (nothing found, a lookup that failed,
two orders for one stock) stops the run in either mode and the run's last message asks the user to clarify the request.

THE SHAPE of the questions is that of `clarify.py` (entry 85), a bounded loop: it OBSERVES the Data Clerk's answer for one name, ACTS by telling the user and asking, OBSERVES the reply, and goes on to the next missing name
or stops. One question for each missing name and never more (`config.missing.max_prompts` is checked before the first). THE MODEL WORDS EACH QUESTION AND DOES NOTHING ELSE: it is given facts made by code (the name as
typed, the Data Clerk's reason, the stocks that stay, the user's request) and returns one question as JSON; CODE checks the wording (`check_wording`: it names the missing stock as the user typed it, it is one question, it is
no longer than `config.missing.max_question_chars`) and the user's answer is a yes or a no that CODE reads from the approver channel. A wording that code refuses (it is asked again at most `config.missing.max_problems`
times), a model that fails, or one that takes more than `config.missing.max_seconds`, never blocks the user: the code's own sentence (`own_question`) is shown instead. Nothing here decides what a stock is: the Data
Clerk's reason is shown as it was given.

NO ANSWER IS NEVER A YES. A no stops the order, and so does no answer (no approver, nobody at the keyboard, a question that timed out): in Ask mode nothing is dropped on the user's behalf.
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.agents.account_manager.config import AccountManagerConfig
from prometheus.prompts import account_manager_missing as prompt

KIND = "missing_company"      # the `kind` of the card the approver is given (the Trader's funding card has its own)


@dataclass
class Wording:
    """One question as it will be shown: its text, who wrote it ("model": the Account Manager's model, checked by code; "code": the code's own sentence) and why the code's sentence was used."""
    text: str
    source: str = "code"
    why: str = ""
    problems: list[str] = field(default_factory=list)
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Asked:
    """What the questions came to: the stocks the user said yes to leaving out (`dropped`: {typed, reason, mode}), why the order stops (`stopped`, None when every question got a yes), what was shown, and what the model cost."""
    dropped: list[dict] = field(default_factory=list)
    stopped: str | None = None
    cards: list[dict] = field(default_factory=list)
    wordings: list[Wording] = field(default_factory=list)


def and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def reason_of(row: dict) -> str:
    """The Data Clerk's reason for a stock it did not list, as it gave it, without a closing full stop (the sentences put their own)."""
    return " ".join(str(row.get("note") or "it is not in the directory").split()).rstrip(". ")


def kept_of(listed: list) -> list[dict]:
    """What stays on the order, for the facts and the card: [{typed, symbol, company, how}] in the order the user typed them."""
    return [{"typed": order.named, "symbol": row["symbol"], "company": row["company"], "how": row.get("how")} for order, row in listed]


def kept_line(stock: dict) -> str:
    """One stock that stays on the order, as the Account Manager shows it: what the user typed, the company and the symbol, and that a brand's symbol is its listed owner's (entry 101)."""
    return f"{stock['typed']}: {stock['company']} ({stock['symbol']})" + (", the listed owner of what you typed" if stock.get("how") == "brand" else "")


def own_question(typed: str, reason: str, kept: list[dict]) -> str:
    """The code's own sentence: what is missing, why, and what happens on a yes. It is what the user sees whenever the model's wording is not usable."""
    return f"{typed} was not found: {reason}. OK to continue without it? Your order would go on with {and_list([k['symbol'] for k in kept])}."


def check_wording(text, typed: str, limit: int) -> str:
    """The model's wording, tidied (one line of plain text) and checked by code, or ValueError with the reason: it must name the missing stock as the user typed it (case aside), be ONE question (one question mark), and be no
    longer than `limit` characters. Nothing else about it is trusted: `<`, `>`, `*` and backticks are removed, so that it can close no tag and carry no markdown."""
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).replace("<", "(").replace(">", ")").replace("*", "").replace("`", "").split())
    if not text:
        raise ValueError("the question is empty")
    if len(text) > limit:
        raise ValueError(f"the question is {len(text)} characters: at most {limit}")
    if " ".join(typed.split()).casefold() not in text.casefold():
        raise ValueError(f"the question does not name {typed!r}, the stock as the user typed it")
    if text.count("?") != 1:
        raise ValueError(f"the question must be exactly one question (it has {text.count('?')} question marks)")
    return text


def _question_of(content: str) -> str:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        raw = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("question"), str) or set(raw) - {"question"}:
        raise ValueError('reply with {"question": "..."} and nothing else')
    return raw["question"]


def word_question(model, facts: dict, config: AccountManagerConfig) -> Wording:
    """One question for one missing stock: the model's wording when it passes `check_wording`, else the code's own sentence (`Wording.source` says which, `why` why not). Never raises, never waits more than
    `config.missing.max_seconds` for the model, and asks again at most `config.missing.max_problems` times when code refused the wording."""
    from prometheus.agents.advisor.calls import invoke_with_limit        # the one model call that gives up on a stuck provider (the same helper the Data Clerk's calls use)
    cfg = config.missing
    out = Wording(text=own_question(facts["typed"], facts["reason"], facts["stays_on_the_order"]))
    if model is None:
        out.why = "no model"
        return out
    messages = [SystemMessage(content=prompt.SYSTEM), HumanMessage(content=prompt.facts_message({**facts, "stays_on_the_order": [k["symbol"] for k in facts["stays_on_the_order"]]}))]
    for _ in range(cfg.max_problems + 1):
        try:
            reply = invoke_with_limit(model, messages, cfg.max_seconds)
        except Exception as error:                                       # a model or network failure, or a timeout: the kind only
            out.why = f"the Account Manager's model could not word it ({type(error).__name__})"
            return out
        meta = getattr(reply, "usage_metadata", None) or {}
        out.calls, out.tokens_in, out.tokens_out = out.calls + 1, out.tokens_in + meta.get("input_tokens", 0), out.tokens_out + meta.get("output_tokens", 0)
        content = reply.content if isinstance(reply.content, str) else ""
        try:
            out.text, out.source, out.why = check_wording(_question_of(content), facts["typed"], cfg.max_question_chars), "model", ""
            return out
        except ValueError as problem:
            out.problems.append(str(problem)[:300])
            out.why = f"the model's wording was refused ({out.problems[-1]})"
            messages = [*messages, AIMessage(content=content), HumanMessage(content=prompt.problem_message(str(problem)[:300]))]
    return out


def card_for(typed: str, reason: str, kept: list[dict], n: int, of: int, wording: Wording, request: str = "") -> dict:
    """The card the approver is given for one missing stock (the shape of the Trader's funding card, which also has its own `question`): `text` is what the terminal shows (what is missing, and what stays on the order),
    `question` what it asks, `sentence` the same without "[y/N]" (the page's), and `kept` the stocks that stay."""
    lines = [f"      NOT FOUND   {typed}: {reason}"]
    for index, stock in enumerate(kept):
        lines.append(f"        {'still on your order' if index == 0 else '':<19}   {kept_line(stock)}")
    return {"kind": KIND, "n": n, "of": of, "typed": typed, "reason": reason, "text": "\n".join(lines), "question": wording.text + " [y/N] ", "sentence": wording.text, "kept": kept, "wording": wording.source, "request": request}


def ask_about_each(model, missing: list, kept: list[dict], request: str, approver, config: AccountManagerConfig) -> Asked:
    """ASK mode: one question for each stock the Data Clerk did not find, in the order the user typed them (see the module text). `missing` is [(DirectOrder, row)] and `kept` what stays on the order (`kept_of`).
    `approver` is the yes/no channel of the desk (a card in, True, False or None out; `available()` False, as the terminal's with nobody there, means nobody is asked). A yes drops that stock and the next is asked about; a no, or no
    answer, stops (`Asked.stopped` says why) and nothing is dropped on the user's behalf. Never raises."""
    out = Asked()
    if len(missing) > config.missing.max_prompts:
        out.stopped = f"{len(missing)} of the stocks you named were not found and at most {config.missing.max_prompts} can be asked about one by one"
        return out
    if approver is None or not getattr(approver, "available", lambda: True)():
        out.stopped = "Ask mode needs a person to answer whether to go on without " + and_list([order.named for order, _ in missing]) + ", and nobody was there to ask: nothing was left out on your behalf"
        return out
    for n, (order, row) in enumerate(missing, 1):
        typed, reason = " ".join(order.named.split()), reason_of(row)
        wording = word_question(model, {"typed": typed, "reason": reason, "stays_on_the_order": kept, "request": " ".join(request.split())[:600]}, config)
        out.wordings.append(wording)
        card = card_for(typed, reason, kept, n, len(missing), wording, request)
        out.cards.append(card)
        try:
            answer = approver(card)
        except Exception as error:                                       # a question that could not be asked: the user has not said yes
            out.stopped = f"the question about {typed} could not be asked ({type(error).__name__})"
            return out
        if answer is True:
            out.dropped.append({"typed": typed, "reason": reason, "mode": "ask"})
            continue
        out.stopped = (f"you said no to going on without {typed}" if answer is False else f"nobody answered whether to go on without {typed}: nothing was left out on your behalf")
        return out
    return out
