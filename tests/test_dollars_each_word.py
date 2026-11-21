"""ONE AMOUNT WRITTEN ONCE IS THE LIMIT OF EVERY STOCK IT NAMES, WHEN THE USER SAID "EACH" (design choices entry 125; design/dollars_each_word_spec.md; Jacob, 2026-11-20: *"add support for 'each' ... each means every
individual stock."*).

Entry 122 made the dollars of an order its stock's limit and refused an amount written once from being the dollars of two orders ("Buy $5,000 of MU and AMD" is ONE total of $5,000, never $5,000 for each: that would spend $10,000
on a request for $5,000), so "Buy $3,000 each of XOM and MSFT" was refused. Now the form has `each`: the model sets it when the user said ONE amount is for every stock (E1), CODE accepts it only when a word that says each is in
their text and every order carries that same amount (E2), the sizing is the one of entry 122 (E3), the recommendation path ("the top 3 oil stocks with $3,000 each") limits every purchase it decides to the amount and the pot is
the amount times the purchases (E4), and the intent says it (E5).

REAL DATA ONLY, as `test_direct_dollars_each.py` (whose helpers these tests use):
  - the real model's replies, recorded on 2026-11-20 with the prompt that has `each` (`tests/fixtures/session_real/direct_dollars_each_word_forms.json`, made by `scripts/capture_direct_dollars_forms.py --each`): they go
    through the real `AccountManager.understand_request` and `parse_intent`; the checks of E2 are made on a real reply edited by ONE field;
  - the stored closes of Friday 2026-11-13 read by the real Data Clerk and the real Trader without a model; the REAL reviews of the real oil sessions (`bob_jones_oil_round_robin.json`: CVX and EOG approved;
    `auto_oil_no_second_mark.json`: EOG and CVX approved, six companies neutral; `pam_smith_neutral.json`: MU and AMD neutral) decided again by the real Account Manager with the amount for each;
  - the live session of 2026-11-20 itself (`tests/fixtures/session_real/auto_direct_dollars_each_word.json`).
"""
import copy
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest
from handshake_helpers import MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_direct_dollars_each import decided, fields
from test_direct_order import BILL, BILL_PROFILE, Replay, account, clerk, never  # noqa: F401
from test_neutral_recommendation import OIL_CASH, OIL_USER, PAM, PAM_PROFILE, allowed, oil_order  # noqa: F401  (allowed is a fixture)

from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import work_order as wo
from prometheus.common.money import money
from prometheus.langgraph import desk
from prometheus.prompts import account_manager_request as prompt

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
FORMS = json.loads((FIXTURES / "direct_dollars_each_word_forms.json").read_text(encoding="utf-8"))["requests"]
FIRST = json.loads((FIXTURES / "direct_dollars_forms.json").read_text(encoding="utf-8"))["requests"]
BOB_CAPTURE = json.loads((FIXTURES / "bob_jones_oil_round_robin.json").read_text(encoding="utf-8"))
EACH = "Buy $3,000 each of XOM and MSFT"
AFTER = "Buy $3,000 of XOM and MSFT each"
APIECE = "Put $2,000 apiece into NVDA and AMD"
SHARES = "Buy 10 shares each of AAPL and MSFT"
TOP3 = "Buy the top 3 oil stocks with $3,000 each"
TOTAL = "Buy $5,000 of MU and AMD"
TWO = "Buy $3,000 of XOM and $3,000 of MSFT"
EV = "Buy the 3 most recommended EV stocks with $5,000"
THREE = Decimal("3000")


def read(typed: str):
    """The real Account Manager reads the real model's recorded replies to `typed`."""
    am = AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    return am.understand_request(typed), am


def edited(typed: str, number: int | None = None, **changes) -> str:
    """The real reply to `typed` with ONE field changed: `changes` for the form, or `number` for the order of that index."""
    form = json.loads(FORMS[typed]["replies"][-1])
    if number is None:
        form.update(changes)
    else:
        form["orders"][number].update(changes)
    return json.dumps(form)


# ---------------------------------------------------------------- E1: the real model reads "each"

def test_the_real_model_reads_each_in_the_five_ways_the_spec_lists():
    got = {typed: read(typed)[0] for typed in (EACH, AFTER, APIECE, SHARES, TOP3)}
    for typed, orders in ((EACH, [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)]), (AFTER, [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)]), (APIECE, [("buy", "NVDA", None, 2000), ("buy", "AMD", None, 2000)]),
                          (SHARES, [("buy", "AAPL", 10, None), ("buy", "MSFT", 10, None)])):
        assert got[typed].action == "order" and got[typed].each is True and fields(got[typed]) == orders and got[typed].budget is None and got[typed].total_budget is None and got[typed].said_how_much()
        assert len(FORMS[typed]["replies"]) == 1      # no retry: the first reply of the real model was the right one
    assert got[EACH].describe() == "buy $3,000.00 of XOM, buy $3,000.00 of MSFT, spending at most $3,000.00 on each stock (your own order: no recommendation is made and the Critic is not asked)"      # E5
    assert got[SHARES].describe() == "buy 10 shares of AAPL, buy 10 shares of MSFT, 10 shares of each stock (your own order: no recommendation is made and the Critic is not asked)"
    top = got[TOP3]      # the recommendation path: the amount is the budget and it means each purchase
    assert (top.action, top.topic, top.count, top.each, top.budget, top.each_dollars, top.total_budget, top.said_how_much()) == ("buy", "oil", 3, True, 3000, Decimal("3000"), None, True)
    assert top.describe() == "buy the 3 most recommended 'oil' companies; spending at most $3,000.00 on each stock"      # E5


def test_every_request_of_the_first_capture_is_read_exactly_as_it_was_with_the_prompt_that_has_each():
    """The prompt changed (`each` is in the form), so the 26 requests of `direct_dollars_forms.json` were asked again: not one reading differs, and `each` is false in every one."""
    assert set(FIRST) <= set(FORMS)
    for typed in FIRST:
        before, after = first_reading(typed, FIRST[typed]["replies"]), first_reading(typed, FORMS[typed]["replies"])
        assert before == after, typed
        assert isinstance(after, str) or after.each is False


def first_reading(typed: str, replies: list):
    for text in replies:
        try:
            return it.parse_intent(text, typed)
        except ValueError as error:
            problem = str(error)
    return problem


# ---------------------------------------------------------------- E2: code checks the words and the numbers, on a real reply edited by one field

@pytest.mark.parametrize("typed, text, why", [
    (TWO, edited(TWO, each=True), "nothing you typed says each"),                                                                          # each, and no word that says it
    ("Buy $3,000 of XOM and MSFT", edited(EACH), "nothing you typed says each"),                                                           # the real reading of the each request, for a text with no cue
    (EACH, edited(EACH, each="true"), "each must be true or false"),                                                                      # a string is not a bool
    (EACH, edited(EACH, each=1), "each must be true or false"),
    ("Buy $3,000 each of XOM and $2,000 of MSFT", edited(EACH, 1, dollars=2000), "each means ONE amount for every stock"),                  # each, and two different amounts
    (EACH, edited(EACH, 1, dollars=None), "each means ONE amount for every stock"),                                                         # each, and an order with no amount
    (EACH, edited(EACH, 0, shares=10, dollars=None), "10 shares of XOM is not what you typed"),                                              # shares nobody wrote, dollars on the other order
    (EACH, edited(EACH, budget=3000), "each: with orders the amount goes on EVERY order"),                                                  # each on a total
    ("Buy $5,000 of MU and AMD each", edited(TOTAL, each=True), "each: with orders the amount goes on EVERY order"),                        # the real reading of the total, with each set and the cue in the words
    (EACH, edited(EACH, 0, dollars=4000), "the amount 4000 is not in what you typed"),                                                      # the amount was not written
    ("Buy 3,000 shares each of XOM and MSFT", edited(EACH), "not as money"),                                                               # written as shares, not as money
    (SHARES, edited(SHARES, 0, shares=12), "12 shares of AAPL is not what you typed"),                                                      # shares nobody wrote
    (TOP3, edited(TOP3, budget=None), "each: the amount for each stock goes in budget"),                                                    # each, and no amount
    (TOP3, edited(TOP3, budget=2500), "the amount 2500 for each stock is not in what you typed as money"),
    ("Look at the best 5G companies", edited("Look at the best 5G companies", each=True), "each: an amount for each stock goes with buying"),     # nothing is bought
    ("Buy the top 3 oil stocks with 3000 each", edited(TOP3), "not in what you typed as money"),                                            # a bare number is not money
])
def test_an_each_the_user_did_not_say_or_amounts_that_differ_are_a_reading_error(typed, text, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        it.parse_intent(text, typed)


def test_the_cue_is_a_small_closed_list_of_whole_words_and_filters_nothing_else():
    for yes in ("each", "Each", "apiece", "a piece", "per stock", "per company", "for every one of them", "for every stock", "per ticker"):
        assert it.EACH_CUE.search(f"buy $1,000 {yes} of A and B"), yes
    for no in ("buy $1,000 of beach and teacher", "buy $1,000 of A and B", "buy $1,000 per share of A", "buy a pieces", "buy $1,000 every day"):
        assert not it.EACH_CUE.search(no), no


# ---------------------------------------------------------------- E4: the recommendation path ("the top 3 with $3,000 each")

def bob_order() -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(BOB_CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def test_each_and_one_total_are_two_ways_to_say_how_much_and_nothing_is_decided(friday_close):
    from test_account_manager_clarify import BOB, PROFILE
    order = bob_order()
    out = AccountManager(model=object(), model_id="each", clerk=clerk(), profile=PROFILE).decide_trades(order, BOB, None, MOMENT, "auto", None, Decimal("5000"), each=THREE)
    assert out.trade_decisions == [] and out.status == "reviewed" and "two ways to say how much" in out.errors[-1]


# ---------------------------------------------------------------- the desk carries it


def test_the_desk_hands_each_to_the_account_manager():
    import inspect
    source = inspect.getsource(desk.build_desk)
    assert 'count=plan.get("count"), each=plan.get("each")' in source and "intent.total_budget" in source


# ---------------------------------------------------------------- the prompt

def test_the_prompt_tells_the_model_about_each_with_made_up_names_only():
    for text in (prompt.SYSTEM, prompt.RETRY):
        assert "each" in text
    assert '"each": false' in prompt.SYSTEM and "ONE amount applies to EVERY stock" in prompt.SYSTEM and "never $5,000 for each" in prompt.SYSTEM
    assert not any(real in prompt.SYSTEM for real in ("XOM", "MSFT", "AAPL", "NVDA", "TSLA", "Microsoft", "Exxon", "oil"))
