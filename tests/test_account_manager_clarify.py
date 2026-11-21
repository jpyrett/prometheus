"""The Account Manager asks the user how much to buy, in Ask mode (design choices entry 85; design/sizing_in_account_manager_spec.md, Phase 2): `clarify.py`, `AccountManager.clarify_sizing`, and the
`amount asked for` limit of `sizing.py`.

REAL DATA. The approved purchases are the REAL reviews of `tests/fixtures/session_real/bob_jones_oil_round_robin.json` (Bob Jones, $15,000, conservative: CVX at $219.89 and EOG at $158.32 approved, PSX denied for
risk), sized by the real `decisions` and `sizing` code. The model's part is a REPLAY of its REAL replies, recorded on 2026-11-15 with the real model (`tests/fixtures/session_real/clarify_ask_mode.json`:
openai/gpt-oss-120b on OpenRouter, the `account_manager` role, six kinds of answers): the loop, the validator, the arithmetic and the sizing are the real code, and a model that fails or invents something is made
from those same real replies with ONE field changed (an amount nobody typed, a split nobody asked for, a guess before the first answer). No stand-in agent. The live test (opt-in, paid) is in `test_end_to_end_live.py`.
"""
from decimal import Decimal
import copy
import math
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


# ---------------------------------------------------------------- when the Account Manager asks: the user's own words

@pytest.mark.parametrize("request_text, said", [
    (CAPTURE["request"], True),                                                                                   # "Use as much of the money in my account as you can": the real request of the capture
    ("I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies.", True),   # the real request of the first capture
    (REQUEST, False),                                                                                             # the request the questions were recorded for: no amount
    ("I want to buy PG", False),
    ("I want to buy the most recommended oil companies", False),                                                  # "the most recommended" is not an amount
    ("Buy 20% of my account in PG", True), ("buy 10 shares of PG", True), ("buy PG for $3,000", True), ("buy three shares of PG", True),
    ("spend all of my money on PG", True), ("Buy shares of the 3 most recommended oil companies", False), ("split the money evenly among the best 3 oil stocks", False),
])
def test_the_account_manager_asks_how_much_only_when_the_request_did_not_say(request_text, said):
    assert TradeIntent(request=request_text, action="buy", symbols=["PG"]).said_how_much() is said


def test_a_budget_the_model_read_is_a_dollar_amount_the_user_said():
    assert TradeIntent(request="buy PG", action="buy", symbols=["PG"], budget=Decimal("5000")).said_how_much() is True


# ---------------------------------------------------------------- the amount the user asked for, in the sizing

def sized(**kw):
    return {s.symbol: s for s in sizing.size_purchases(PURCHASES, CASH, CONFIG, **kw)}


# AMENDED (entry 111, M3): 2.5 is a quantity now (what the user's dollars for each stock of a direct order buy: `wanted` may have a fraction); seven decimals and a string are still a programming error
@pytest.mark.parametrize("wanted", [{"nobody": 3}, {CVX.record_id: -1}, {CVX.record_id: 0.1234567}, {CVX.record_id: True}, {CVX.record_id: "2"}])
def test_a_wanted_that_is_not_whole_shares_of_a_purchase_is_a_programming_error(wanted):
    with pytest.raises(ValueError):
        sized(wanted=wanted)


# ---------------------------------------------------------------- the check of the model's form against what the user typed (code)

def check(choice, said, purchases=PURCHASES, cash=CASH, equity=EQUITY, prior=None):
    clarify.check_choice(choice, said, purchases, cash, equity, prior)


@pytest.mark.parametrize("choice, said", [
    (clarify.SizingChoice(unit="percent", amount=20), "20% of my account"),
    (clarify.SizingChoice(unit="percent", amount=20), "twenty percent is 20 percent"),
    (clarify.SizingChoice(unit="dollars", amount=3000), "$3,000"),
    # ("3000" with no other unit was accepted here as dollars; since entry 138 a bare number is SHARES: it is refused for dollars below, and accepted for shares in tests/test_request_numbers.py)
    (clarify.SizingChoice(unit="dollars", amount=3000), "3k dollars"),
    (clarify.SizingChoice(unit="shares", amount=10, scope="each"), "10 shares of each"),
    (clarify.SizingChoice(unit="max"), "as much as I can"),
    (clarify.SizingChoice(unit="max"), "everything, the maximum"),
    (clarify.SizingChoice(unit="percent", amount=20, allocation="even"), "20% split evenly"),
    (clarify.SizingChoice(unit="percent", amount=20, allocation="round_robin"), "20%, one share at a time"),
])
def test_a_choice_the_users_own_words_support_is_accepted(choice, said):
    check(choice, said)


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


def test_a_split_the_user_asked_for_in_the_request_is_not_a_split_the_model_invented():
    check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="even"), "$3,000", prior="even")           # the request said "split it evenly"
    with pytest.raises(ValueError):
        check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="even"), "$3,000", prior="round_robin")


def test_round_robin_is_the_default_so_it_needs_no_word_from_the_user_but_it_cannot_overrule_a_split_the_request_asked_for():
    """AMENDED (entry 106). A choice of `round_robin` was refused unless the user said "round robin" or "in turn" ("use as_much_as_possible unless they said how to spread it"). Round-robin is the default now, whatever the mode:
    it is accepted with no cue, and so is a choice that leaves the allocation out (the model's way of saying the user did not say). What stays refused: round-robin over a split the REQUEST already asked for."""
    check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000")
    check(clarify.SizingChoice(unit="dollars", amount=3000), "$3,000")
    check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000 one share of each in turn", prior="even")
    with pytest.raises(ValueError, match="already asked for an even split"):
        check(clarify.SizingChoice(unit="dollars", amount=3000, allocation="round_robin"), "$3,000", prior="even")


def test_the_retired_word_of_the_real_replies_is_read_as_the_user_did_not_say_how_to_spread_it_which_is_round_robin():
    """The REAL replies recorded before entry 106 (`clarify_ask_mode.json`, not edited) say `"allocation": "as_much_as_possible"`. They are still read, as "the user did not say how to spread it" (None), and the plan then
    leaves the split to the sizing's default, round-robin: the same two purchases come out a share of each in turn, and not the first company first."""
    texts = [r for s in RECORDED.values() for r in s["replies"] if "as_much_as_possible" in r]
    assert len(texts) == 3
    for text in texts:
        assert clarify.SizingChoice(**{k: v for k, v in json.loads(text).items() if k in ("unit", "amount", "scope", "allocation")}).allocation is None
    p = plan(clarify.SizingChoice(unit="percent", amount=20, scope="total", allocation="as_much_as_possible"))
    got = sized(allocation=p.balance, budget=p.budget, wanted=p.wanted)
    assert p.balance is None and p.budget == Decimal("3000") and "a share of each company in turn" in p.text
    assert got["CVX"].allocation == got["EOG"].allocation == "round_robin" and got["CVX"].shares >= 1 and got["EOG"].shares >= 1 and got["CVX"].est_cost + got["EOG"].est_cost <= Decimal("3000")
    assert got["CVX"].shares < math.floor(3000 / CVX.price) + 1 and got["EOG"].shares >= math.floor(3000 / (CVX.price + EOG.price))           # neither took the whole $3,000 before the other had its share


def test_the_percentage_of_an_account_with_no_value_cannot_be_worked_out():
    with pytest.raises(ValueError, match="value of the account is not known"):
        check(clarify.SizingChoice(unit="percent", amount=20), "20%", equity=None)


# ---------------------------------------------------------------- the arithmetic (code), with the real prices

def plan(choice, prior=None):
    return clarify.plan_choice(choice, PURCHASES, EQUITY, Decimal("15000.0"), prior)


def test_a_number_of_shares_in_all_is_shared_out_in_rank_order():
    assert plan(clarify.SizingChoice(unit="shares", amount=7, scope="total")).wanted == {CVX.record_id: 4, EOG.record_id: 3}
    assert plan(clarify.SizingChoice(unit="shares", amount=6, scope="total")).wanted == {CVX.record_id: 3, EOG.record_id: 3}


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

@pytest.mark.parametrize("name", list(RECORDED))
def test_the_loop_over_the_real_replies_asks_what_it_asked_and_reads_the_answer_into_the_recorded_plan(name):
    model, answers = scenario(name)
    channel = channel_of(answers)
    outcome = clarify.run_loop(model, facts(), PURCHASES, CASH, EQUITY, channel, CONFIG)
    recorded = RECORDED[name]
    assert outcome.choice is not None and outcome.problems == [] and model.replies == [] and outcome.calls == recorded["calls"]
    asked = [c for c in recorded["cards"] if c["kind"] == "how_much"]
    assert [c["question"] for c in channel.cards] == [c["question"] for c in asked] and outcome.answers == [a for a in recorded["answers"]][:len(asked)]
    again = clarify.plan_choice(outcome.choice, PURCHASES, EQUITY, CASH)
    assert again.model_dump(exclude={"notes", "text"}) == {k: v for k, v in recorded["plan"].items() if k not in ("limit_overrides", "notes", "text")}


def reply(name, index=-1):
    return RECORDED[name]["replies"][index]


def test_a_number_the_user_never_typed_is_refused_and_after_the_allowed_refusals_the_loop_gives_up():
    invented = reply("percent_total").replace('"amount": 20', '"amount": 25').replace('"amount":20', '"amount":25')
    ask = RECORDED["percent_total"]["replies"][0]
    assert invented != RECORDED["percent_total"]["replies"][1]
    model = Replay([ask, invented], repeat=True)
    outcome = clarify.run_loop(model, facts(), PURCHASES, CASH, EQUITY, channel_of(["20% of my account in total"]), CONFIG)
    assert outcome.choice is None and len(outcome.problems) == CONFIG.clarify.max_problems + 1 and all("25 is not in anything the user typed" in p for p in outcome.problems)
    assert "could not be read into an amount" in outcome.why and outcome.calls == 1 + CONFIG.clarify.max_problems + 1


def test_a_split_the_model_chose_and_the_user_never_asked_for_is_refused_then_accepted_when_the_model_drops_it():
    ask, final = RECORDED["percent_total"]["replies"]
    invented = final.replace("as_much_as_possible", "even")
    assert invented != final
    outcome = clarify.run_loop(Replay([ask, invented, final]), facts(), PURCHASES, CASH, EQUITY, channel_of(["20% of my account in total"]), CONFIG)
    assert len(outcome.problems) == 1 and "did not ask for an even split" in outcome.problems[0] and outcome.choice.allocation is None      # AMENDED (entry 106): the real reply's `as_much_as_possible` is read as "did not say how to spread it": None


def test_a_guess_before_the_first_answer_is_refused_and_the_model_then_asks():
    ask, final = RECORDED["as_much_as_i_can"]["replies"]
    outcome = clarify.run_loop(Replay([final, ask, final]), facts(), PURCHASES, CASH, EQUITY, channel_of(["as much as I can"]), CONFIG)      # the real model did this once, when the request already said "as much as I can"
    assert len(outcome.problems) == 1 and "did not say they want as much as they can" in outcome.problems[0] and outcome.choice.unit == "max" and len(outcome.questions) == 1


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


def test_a_model_that_cannot_be_reached_ends_the_loop_with_the_kind_of_the_failure_only():
    outcome = clarify.run_loop(Broken(), facts(), PURCHASES, CASH, EQUITY, channel_of([]), CONFIG)
    assert outcome.choice is None and outcome.why == "the Account Manager's model could not be reached (ConnectionError)" and "127.0.0.1" not in outcome.why and "password" not in outcome.why


def test_a_user_who_does_not_answer_ends_the_loop():
    model, _ = scenario("percent_total")
    outcome = clarify.run_loop(model, facts(), PURCHASES, CASH, EQUITY, channel_of([]), CONFIG)
    assert outcome.choice is None and outcome.why == "nobody answered the question" and len(outcome.questions) == 1


# ---------------------------------------------------------------- the Account Manager: when it asks, and what it sizes

def test_in_ask_mode_a_request_that_did_not_say_how_much_is_asked_before_anything_is_sized_and_the_answer_sizes_the_purchases():
    model, answers = scenario("percent_each_no_raise_needed")
    shown = []
    am, channel = manager(model, shown.append), channel_of(answers)
    order = ask_mode(am, clarify=channel, request_text=REQUEST)
    got = rows(order)
    assert [(d["symbol"], d["shares"], d["will_trade"], d["size"]["binding"], d["size"]["asked"]) for d in (got["CVX"], got["EOG"])] == [("CVX", 3, "pending_user", "asked", 3), ("EOG", 4, "pending_user", "asked", 4)]
    assert got["PSX"]["will_trade"] == "no_system_denied" and len(channel.cards) == 1 and order.status == "reviewed" and order.errors == []     # waiting for the user's yes to each trade, at the size they asked for
    assert "How much: 5% of your account in each of the 2 companies." in "\n".join(shown) and am.last_sizing["questions"] and am.last_sizing["choice"]["unit"] == "percent"


def test_an_answer_in_dollars_spread_evenly_sizes_the_purchases_to_the_equal_dollars():
    model, answers = scenario("dollars_total_even")
    order = ask_mode(manager(model), clarify=channel_of(answers), request_text=REQUEST)
    got = rows(order)
    assert got["CVX"]["balance"]["mode"] == "even" and got["CVX"]["balance"]["of"] == 2 and got["CVX"]["est_cost"] + got["EOG"]["est_cost"] <= Decimal("3000")
    assert abs(got["CVX"]["est_cost"] - got["EOG"]["est_cost"]) < 220


def test_as_much_as_i_can_is_the_default_size_all_the_cash_a_share_of_each_company_in_turn():
    """AMENDED (entry 106). "As much as I can" over two companies was the default size "the most the cash buys" of EACH (CVX 68, EOG 94: the first took what it could and the second saw all the cash again). It is the default
    size, the round robin, now: CVX 40 and EOG 39, together within the $15,000."""
    model, answers = scenario("as_much_as_i_can")
    asked = ask_mode(manager(model), clarify=channel_of(answers), request_text=REQUEST)
    default = ask_mode(manager(Broken()))
    assert [(d["symbol"], d["shares"]) for d in asked.trade_decisions] == [(d["symbol"], d["shares"]) for d in default.trade_decisions] and (rows(asked)["CVX"]["shares"], rows(asked)["EOG"]["shares"]) == ROUND_ROBIN_OF_ALL_THE_CASH
    assert rows(asked)["CVX"]["est_cost"] + rows(asked)["EOG"]["est_cost"] <= CASH and {d["size"]["allocation"] for d in asked.trade_decisions if d["size"]} == {"round_robin"}



def test_a_model_that_cannot_be_reached_leaves_the_default_size_and_says_so_to_the_user_and_on_the_order():
    shown = []
    order = ask_mode(manager(Broken(), shown.append), clarify=channel_of(["20%"]), request_text=REQUEST)
    got = rows(order)
    assert (got["CVX"]["shares"], got["EOG"]["shares"]) == ROUND_ROBIN_OF_ALL_THE_CASH and got["CVX"]["will_trade"] == "pending_user"      # AMENDED (entry 106): the default size is the round robin
    assert any("how much to buy could not be worked out with you (the Account Manager's model could not be reached (ConnectionError))" in e for e in order.errors)
    said = "\n".join(shown)
    assert "I could not work out how much you want" in said and "The default size is shown" in said and "password" not in said and "127.0.0.1" not in said


def test_a_user_who_does_not_answer_leaves_the_default_size_and_says_so():
    model, _ = scenario("percent_total")
    shown = []
    order = ask_mode(manager(model, shown.append), clarify=channel_of([]), request_text=REQUEST)
    assert (rows(order)["CVX"]["shares"], rows(order)["EOG"]["shares"]) == ROUND_ROBIN_OF_ALL_THE_CASH and any("nobody answered the question" in e for e in order.errors)      # AMENDED (entry 106): the default size is the round robin


# ---------------------------------------------------------------- the terminal, the session and the desk

def test_the_terminal_shows_the_facts_with_the_first_question_only_and_returns_what_the_user_types():
    shown, asked = [], []

    def ask(prompt):
        asked.append(prompt)
        return "20% of my account"

    channel = clarify.terminal_clarifier(ask=ask, show=shown.append, interactive=True)
    assert channel.available() is True
    first = {"question": "How much would you like to buy?", "n": 1, "of": 3, "facts": ["Cash $15,000.00", "  CVX at $219.89"], "kind": "how_much"}
    assert channel(first) == "20% of my account" and "HOW MUCH?" in shown[0] and "Cash $15,000.00" in shown[0] and "(question 1 of 3)" in asked[0]
    shown.clear()
    channel({**first, "n": 2, "facts": []})
    assert shown == [] and "(question 2 of 3)" in asked[1]


def test_the_terminal_asks_nobody_who_is_not_there_and_an_end_of_input_is_no_answer():
    assert clarify.terminal_clarifier(interactive=False).available() is False and clarify.terminal_clarifier(interactive=False)({"question": "q", "n": 1, "of": 3, "facts": []}) is None

    def eof(prompt):
        raise EOFError

    assert clarify.terminal_clarifier(ask=eof, interactive=True)({"question": "q", "n": 1, "of": 3, "facts": []}) is None


def test_the_session_and_the_desk_hand_the_question_channel_to_the_account_manager():
    import inspect

    from prometheus.application import session as ss
    from prometheus.langgraph import desk
    assert "clarify=clarifier_for(ask, out)" in inspect.getsource(ss.run_session)
    assert ss.clarifier_for(input).available is not None and ss.clarifier_for(lambda prompt: "y", print).available() is True              # a given ask is a person
    source = inspect.getsource(desk.build_desk)
    assert "clarify=clarify, said_how_much=bool(plan.get(\"said_how_much\")), request_text=plan.get(\"request_text\")" in source and "intent.said_how_much()" in source


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
