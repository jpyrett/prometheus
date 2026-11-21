"""The Account Manager asks the user how much to buy, in Ask mode (design choices entry 85; design/sizing_in_account_manager_spec.md, Phase 2): `clarify.py`, `AccountManager.clarify_sizing`, and the
`amount asked for` limit of `sizing.py`.

REAL DATA. The approved purchases are the REAL reviews of `tests/fixtures/session_real/bob_jones_oil_round_robin.json` (Bob Jones, $15,000, conservative: CVX at $219.89 and EOG at $158.32 approved, PSX denied for
risk), sized by the real `decisions` and `sizing` code. The model's part is a REPLAY of its REAL replies, recorded on 2026-11-15 with the real model (`tests/fixtures/session_real/clarify_ask_mode.json`:
openai/gpt-oss-120b on OpenRouter, the `account_manager` role, six kinds of answers): the loop, the validator, the arithmetic and the sizing are the real code, and a model that fails or invents something is made
from those same real replies with ONE field changed (an amount nobody typed, a split nobody asked for, a guess before the first answer). No stand-in agent. The live test (opt-in, paid) is in `test_end_to_end_live.py`.
"""
from decimal import Decimal
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.agents.account_manager import clarify, decisions, sizing
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.account_manager.config import load_config
from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
CAPTURE = json.loads((FIXTURES / "bob_jones_oil_round_robin.json").read_text(encoding="utf-8"))
RECORDED = json.loads((FIXTURES / "clarify_ask_mode.json").read_text(encoding="utf-8"))["scenarios"]
NOW = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)
BOB = Portfolio(user="Bob Jones", cash_avail=Decimal("15000"))
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="ask")
CONFIG = load_config()
CASH = Decimal("15000")
EQUITY = Decimal("15000")
REQUEST = RECORDED["percent_total"]["request"]


def reviewed() -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


PURCHASES = decisions.purchases_to_size(reviewed(), BOB, NOW)
CVX, EOG = PURCHASES


class Replay:
    """The model's REAL recorded replies (see the module text), given back in order. `repeat` gives the last one again when the list runs out (a model that keeps saying the same thing)."""

    def __init__(self, replies, repeat: bool = False):
        self.replies, self.repeat, self.seen, self.model_name = list(replies), repeat, [], "openai/gpt-oss-120b"

    def invoke(self, messages):
        self.seen.append(messages)
        if not self.replies:
            raise AssertionError("the model was asked more times than it has recorded replies")
        content = self.replies[0] if self.repeat and len(self.replies) == 1 else self.replies.pop(0)
        return SimpleNamespace(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 10})


class Broken:
    model_name = "broken"

    def invoke(self, messages):
        raise ConnectionError("connection to server at 127.0.0.1 failed: password authentication failed for user app")


def scenario(name: str):
    s = RECORDED[name]
    return Replay(s["replies"]), list(s["answers"])


def channel_of(answers):
    """A person at the terminal who types `answers` in order; remembers every card it was shown."""
    cards, left = [], list(answers)

    def channel(card):
        cards.append(card)
        return left.pop(0) if left else None

    channel.cards = cards
    return channel


def facts():
    return clarify.facts_for(PURCHASES, sizing.size_purchases(PURCHASES, Decimal("15000.0"), CONFIG), Decimal("15000.0"), EQUITY, REQUEST, None, CONFIG)


def manager(model, screen=None):
    return AccountManager(model=model, model_id="clarify", profile=PROFILE, screen=screen)


def ask_mode(am, **kw):
    return am.decide_trades(reviewed(), BOB, None, NOW, "ask", **kw)


def rows(order):
    return {d["symbol"]: d for d in order.trade_decisions}


# ---------------------------------------------------------------- the amount the user asked for, in the sizing

def sized(**kw):
    return {s.symbol: s for s in sizing.size_purchases(PURCHASES, CASH, CONFIG, **kw)}


# AMENDED (entry 111, M3): 2.5 is a quantity now (what the user's dollars for each stock of a direct order buy: `wanted` may have a fraction); seven decimals and a string are still a programming error


# ---------------------------------------------------------------- the check of the model's form against what the user typed (code)

def check(choice, said, purchases=PURCHASES, cash=CASH, equity=EQUITY, prior=None):
    clarify.check_choice(choice, said, purchases, cash, equity, prior)


@pytest.mark.parametrize("choice, said, why", [
    (clarify.SizingChoice(unit="percent", amount=25), "20%", "25 is not in anything the user typed"),             # a number nobody typed
    (clarify.SizingChoice(unit="dollars", amount=3000), "", "has not answered anything yet"),                       # a guess before the first answer
    (clarify.SizingChoice(unit="max"), "", "did not say they want as much as they can"),
    (clarify.SizingChoice(unit="max"), "maybe 20%", "did not say they want as much as they can"),
    (clarify.SizingChoice(unit="percent", amount=150), "150%", "above 0 and at most 100"),
    (clarify.SizingChoice(unit="percent", amount=20), "20 dollars", "did not give a percentage"),
    (clarify.SizingChoice(unit="shares", amount=10), "$10", "did not give a number of shares"),
    (clarify.SizingChoice(unit="dollars", amount=3000), "3000", "has no $ and no word dollars"),                  # AMENDED (entry 138): it was accepted as dollars; a bare number is shares
    (clarify.SizingChoice(unit="shares", amount=2.5), "2.5 shares", "shares are a whole number"),
    (clarify.SizingChoice(unit="dollars", amount=20), "20%", "gave percent or shares, not dollars"),
    (clarify.SizingChoice(unit="dollars", amount=99999), "$99,999", "the cash is $15,000.00: ask for less"),
    (clarify.SizingChoice(unit="percent", amount=60, scope="each"), "60% each", "ask for less"),                    # 60% of the account in each of two companies
    (clarify.SizingChoice(unit="shares", amount=500, scope="each"), "500 shares each", "ask for less"),
    (clarify.SizingChoice(unit="percent", amount=20, allocation="even"), "20% of my account", "did not ask for an even split"),
    (clarify.SizingChoice(unit="max", amount=5), "as much as I can", "takes no amount"),
])
def test_a_choice_the_users_words_do_not_support_or_the_account_cannot_pay_is_refused_with_the_reason(choice, said, why):
    with pytest.raises(ValueError, match=why.replace("$", r"\$").replace("(", r"\(").replace(")", r"\)")):
        check(choice, said)


def test_round_robin_is_the_default_so_it_needs_no_word_from_the_user_but_it_cannot_overrule_a_split_the_request_asked_for():
    """AMENDED (entry 106). A choice of `round_robin` was refused unless the user said "round robin" or "in turn" ("use as_much_as_possible unless they said how to spread it"). Round-robin is the default now, whatever the mode:
    it is accepted with no cue, and so is a choice that leaves the allocation out (the model's way of saying the user did not say). What stays refused: round-robin over a split the REQUEST already asked for."""
    check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000")
    check(clarify.SizingChoice(unit="dollars", amount=3000), "$3,000")
    check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000 one share of each in turn", prior="even")
    with pytest.raises(ValueError, match="already asked for an even split"):
        check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000", prior="even")


def test_the_percentage_of_an_account_with_no_value_cannot_be_worked_out():
    with pytest.raises(ValueError, match="value of the account is not known"):
        check(clarify.SizingChoice(unit="percent", amount=20), "20%", equity=None)


# ---------------------------------------------------------------- the arithmetic (code), with the real prices

def plan(choice, prior=None):
    return clarify.plan_choice(choice, PURCHASES, EQUITY, Decimal("15000.0"), prior)


# By hand, at the real prices (CVX 219.89, EOG 158.32): one share of each costs 378.21, so 39 rounds cost 14,750.19; $249.81 are left, and the 40th round buys a CVX (219.89, $29.92 left) and cannot buy an EOG (158.32).
ROUND_ROBIN_OF_ALL_THE_CASH = (40, 39)      # AMENDED (entry 106): "as much as I can" over two companies was (68, 94), each taking ALL the $15,000; the default is the round robin, which they share a share at a time


def test_what_is_worked_out_is_what_the_sizing_buys():
    for choice, shares in ((clarify.SizingChoice(unit="percent", amount=5, scope="each"), (3, 4)), (clarify.SizingChoice(unit="shares", amount=10, scope="each"), (10, 10)),
                           (clarify.SizingChoice(unit="shares", amount=7, scope="total"), (4, 3)), (clarify.SizingChoice(unit="max"), ROUND_ROBIN_OF_ALL_THE_CASH)):
        p = plan(choice)
        got = sized(allocation=p.balance, budget=p.budget, wanted=p.wanted)
        assert (got["CVX"].shares, got["EOG"].shares) == shares, choice
    budget = plan(clarify.SizingChoice(unit="dollars", amount=1000, scope="total"))
    got = sized(allocation=budget.balance, budget=budget.budget)
    assert got["CVX"].est_cost + got["EOG"].est_cost <= Decimal("1000")                                                      # a total is never exceeded


# ---------------------------------------------------------------- the loop, with the model's real recorded replies


def reply(name, index=-1):
    return RECORDED[name]["replies"][index]


def test_no_more_than_three_questions_are_put_to_the_user():
    ask = RECORDED["percent_total"]["replies"][0]
    channel = channel_of(["hmm", "not sure", "maybe", "I don't know"])
    outcome = clarify.run_loop(Replay([ask], repeat=True), facts(), PURCHASES, CASH, EQUITY, channel, CONFIG)
    assert len(channel.cards) == 3 == len(outcome.questions) and outcome.choice is None and any("no questions are left" in p for p in outcome.problems)


def test_a_reply_that_is_not_the_form_is_refused_each_time_and_the_loop_gives_up():
    outcome = clarify.run_loop(Replay(["I think about 20 percent would be fine."], repeat=True), facts(), PURCHASES, CASH, EQUITY, channel_of([]), CONFIG)
    assert outcome.choice is None and outcome.problems and all("no JSON object" in p for p in outcome.problems) and outcome.questions == []
    for bad in ('{"action": "buy", "shares": 100}', '{"action": "ask_user"}', '{"action": "set_sizing", "unit": "percent", "amount": 20, "shares": 3}', "[1, 2]", '{"action": "ask_user", "question": "q", "x": 1}'):
        assert clarify.run_loop(Replay([bad], repeat=True), facts(), PURCHASES, CASH, EQUITY, channel_of([]), CONFIG).choice is None


# ---------------------------------------------------------------- the Account Manager: when it asks, and what it sizes


def test_a_model_that_cannot_be_reached_leaves_the_default_size_and_says_so_to_the_user_and_on_the_order():
    shown = []
    order = ask_mode(manager(Broken(), shown.append), clarify=channel_of(["20%"]), request_text=REQUEST)
    got = rows(order)
    assert (got["CVX"]["shares"], got["EOG"]["shares"]) == ROUND_ROBIN_OF_ALL_THE_CASH and got["CVX"]["will_trade"] == "pending_user"      # AMENDED (entry 106): the default size is the round robin
    assert any("how much to buy could not be worked out with you (the Account Manager's model could not be reached (ConnectionError))" in e for e in order.errors)
    said = "\n".join(shown)
    assert "I could not work out how much you want" in said and "The default size is shown" in said and "password" not in said and "127.0.0.1" not in said


# ---------------------------------------------------------------- the terminal, the session and the desk


def test_the_terminal_asks_nobody_who_is_not_there_and_an_end_of_input_is_no_answer():
    assert clarify.terminal_clarifier(interactive=False).available() is False and clarify.terminal_clarifier(interactive=False)({"question": "q", "n": 1, "of": 3, "facts": []}) is None

    def eof(prompt):
        raise EOFError

    assert clarify.terminal_clarifier(ask=eof, interactive=True)({"question": "q", "n": 1, "of": 3, "facts": []}) is None


def test_the_report_says_a_purchase_the_user_sized_is_what_they_asked_for_and_never_that_something_held_it_back():
    from prometheus.application.session import use_of_money_lines
    model, answers = scenario("percent_each_no_raise_needed")
    order = ask_mode(manager(model), clarify=channel_of(answers), request_text=REQUEST)
    after = BOB.model_copy(update={"cash_avail": CASH - 3 * Decimal("219.89") - 4 * Decimal("158.32")})
    for d in order.trade_decisions:
        if d["will_trade"] == "pending_user":
            d["will_trade"] = "yes_user"
    text = "\n".join(use_of_money_lines(order, BOB, after))
    assert "CVX    3 shares" in text and ": the amount you asked for" in text
    assert "held back by the amount" not in text and "held back by" not in text.split("CVX")[1].split("EOG")[0]
