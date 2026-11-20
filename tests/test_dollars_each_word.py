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
from test_account_manager_clarify import channel_of
from test_direct_dollars_each import decided, fields
from test_direct_order import BILL, BILL_PROFILE, Replay, account, clerk, never  # noqa: F401
from test_neutral_recommendation import OIL_CASH, OIL_USER, PAM, PAM_PROFILE, allowed, oil_order  # noqa: F401  (allowed is a fixture)
from test_neutral_recommendation import account as neutral_account
from test_neutral_recommendation import advised as pam_advised

from prometheus.agents.account_manager import clarify
from prometheus.agents.account_manager import intent as it
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money, price as price_door
from prometheus.common.portfolio import Portfolio, trade_amount
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


def intent_of(typed: str):
    return it.parse_intent(FORMS[typed]["replies"][-1], typed)


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


def test_the_readings_that_must_not_change_did_not_change_an_amount_written_once_is_one_total_and_two_amounts_are_two():
    total, two, ev = read(TOTAL)[0], read(TWO)[0], read(EV)[0]
    assert (total.each, total.budget, total.total_budget, total.each_dollars, [(o.symbol, o.shares, o.dollars) for o in total.orders]) == (False, 5000, 5000, None, [("MU", None, None), ("AMD", None, None)])
    assert (two.each, two.budget, fields(two)) == (False, None, [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)])
    assert (ev.each, ev.budget, ev.total_budget, ev.each_dollars, ev.count) == (False, 5000, 5000, None, 3) and ev.describe() == "buy the 3 most recommended 'EV' companies; spending at most $5,000.00"


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


def test_without_each_an_amount_written_once_for_two_stocks_is_still_refused_and_the_model_is_told_to_say_each():
    text = edited(EACH, each=False)      # the real reading with `each` false: the model did not see the word
    with pytest.raises(ValueError, match=r"the amount 3000 is the dollars of 2 orders but you wrote it 1 time.*or said each \(each, apiece, per stock: then set each to true\)"):
        it.parse_intent(text, EACH)
    am = AccountManager(model=Replay([text, FORMS[EACH]["replies"][-1]]), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    intent = am.understand_request(EACH)      # the model is asked once more with the reason, and answers with each
    assert intent.each and fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and am.intent_run["retries"] == 1 and "set each to true" in str(am.model.seen[1][-1].content)


def test_a_reading_that_leaves_the_cue_out_of_the_words_is_asked_again_and_then_the_user_is_asked_to_clarify():
    am = AccountManager(model=Replay([edited(TWO, each=True)] * 2), model_id="recorded", clerk=clerk(), profile=BILL_PROFILE)
    with pytest.raises(ValueError, match=r"I could not understand the request \(each is true but nothing you typed says each"):
        am.understand_request(TWO)


def test_the_cue_is_a_small_closed_list_of_whole_words_and_filters_nothing_else():
    for yes in ("each", "Each", "apiece", "a piece", "per stock", "per company", "for every one of them", "for every stock", "per ticker"):
        assert it.EACH_CUE.search(f"buy $1,000 {yes} of A and B"), yes
    for no in ("buy $1,000 of beach and teacher", "buy $1,000 of A and B", "buy $1,000 per share of A", "buy a pieces", "buy $1,000 every day"):
        assert not it.EACH_CUE.search(no), no


def test_one_number_of_shares_for_every_stock_is_checked_as_shares_written_and_the_side_as_ever():
    assert it.shares_written("Buy 10 shares each of AAPL and 2.5 shares of MSFT, three shares of X, 0.1234567 shares of Y, 1,000 of Z, 5%") == {Decimal("10"), Decimal("2.5"), Decimal("3")}
    sell = "Sell 10 shares each of AAPL and MSFT"
    reply = json.loads(FORMS[SHARES]["replies"][-1])
    reply["orders"] = [{**o, "side": "sell"} for o in reply["orders"]]
    assert [(o.side, o.shares) for o in it.parse_intent(json.dumps(reply), sell).orders] == [("sell", 10), ("sell", 10)]
    with pytest.raises(ValueError, match="the last word about buying or selling before the 10 shares is not a buying word"):
        it.parse_intent(FORMS[SHARES]["replies"][-1], sell)      # the real reading said buy, the user said sell


def test_the_model_s_reading_of_buy_with_the_stocks_named_and_the_amount_for_each_is_made_into_the_orders_with_that_amount():
    """The model sometimes reads a request that names the stocks as "buy" (the program makes it a direct order, entry 98): with each, the amount is the dollars of EVERY order and never a total."""
    as_buy = json.dumps({"action": "buy", "topic": None, "symbols": ["XOM", "MSFT"], "companies": [], "count": None, "allocation": "default", "budget": 3000, "each": True, "orders": []})
    intent = it.parse_intent(as_buy, EACH)
    assert intent.action == "order" and intent.each and intent.budget is None and fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and intent.said_how_much()
    with pytest.raises(ValueError, match="nothing you typed says each"):
        it.parse_intent(as_buy, TWO)
    no_amount = json.dumps({"action": "buy", "topic": None, "symbols": ["XOM", "MSFT"], "companies": [], "count": None, "allocation": "default", "budget": 4000, "each": True, "orders": []})
    with pytest.raises(ValueError, match="the amount 4000 for each stock is not in what you typed as money"):
        it.parse_intent(no_amount, EACH)


# ---------------------------------------------------------------- E3: the direct path is sized from each stock's own dollars, as in entry 122

@pytest.fixture
def each_intent():
    return intent_of(EACH)


def test_three_thousand_dollars_each_spends_three_thousand_on_each_and_six_thousand_in_all_to_the_cent(each_intent, friday_close):
    for typed in (EACH, AFTER):
        intent = intent_of(typed)
        order, am = decided(typed, intent)
        prices = {"XOM": close_of("XOM"), "MSFT": close_of("MSFT")}
        for d, entry in zip(order.trade_decisions, order.direct_orders):
            s = d["symbol"]
            assert entry["dollars"] == THREE and entry["how_many"] == "dollars" and entry["shares"] == d["shares"] == sh.buys(THREE, prices[s])
            assert (d["will_trade"], d["dollars"], d["est_cost"], d["size"]["pot"], d["size"]["how"]) == ("yes_user", THREE, Decimal("3000.00"), Decimal("6000.00"), "dollars") and wo.verify_trade(order, d["record_id"], MOMENT) == []
        am.perform_trades(order, Portfolio(user="Bill Gates", cash_avail=Decimal("10000")), trader(), MOMENT)
        assert [(t["symbol"], t["status"], t["fill"]["amount"]) for t in order.trades] == [("XOM", "filled", Decimal("3000.00")), ("MSFT", "filled", Decimal("3000.00"))]
        assert order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("4000.00") and wo.verify_trades(order) == [] and wo.pot_reasons(order) == []


def test_with_less_cash_than_two_amounts_each_is_still_capped_by_its_own_dollars_and_the_run_by_the_cash(each_intent, friday_close):
    order, am = decided(EACH, each_intent, cash="4000")
    assert {d["size"]["pot"] for d in order.trade_decisions} == {Decimal("4000.00")} and all(d["est_cost"] <= THREE for d in order.trade_decisions)
    am.perform_trades(order, Portfolio(user="Bill Gates", cash_avail=Decimal("4000")), trader(), MOMENT)
    assert sum((t["fill"]["amount"] for t in order.trades), Decimal(0)) <= Decimal("4000") and wo.verify_trades(order) == [] and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0


def test_in_ask_mode_each_asks_nothing_about_how_much_and_the_user_only_confirms_each_order(each_intent, friday_close):
    cards, channel = [], channel_of([])
    order, _ = decided(EACH, each_intent, mode="ask", approver=lambda card: cards.append(card) or True, clarify=channel)
    assert channel.cards == []      # no question about how much (the user said: $3,000 for each)
    assert len(cards) == 2 and all("YOU DID NOT SAY HOW MANY SHARES: the Account Manager worked out from the dollars you wrote" in c["text"] for c in cards)      # the final proposal of each order is shown, as for any order in dollars (size first, ask last)
    assert [d["will_trade"] for d in order.trade_decisions] == ["yes_user", "yes_user"] and {d["dollars"] for d in order.trade_decisions} == {THREE}


def test_ten_shares_each_are_ten_shares_of_every_stock_as_typed_and_nothing_else(friday_close):
    """The real reading of "Buy 10 shares each of AAPL and MSFT" with AAPL changed to XOM (the one field of the real reply that is edited, and the typed words with it): the real directory rows of the live session are those of XOM and MSFT."""
    typed = "Buy 10 shares each of XOM and MSFT"
    reply = json.loads(FORMS[SHARES]["replies"][-1])
    reply["symbols"], reply["orders"][0]["symbol"] = ["XOM", "MSFT"], "XOM"
    intent = it.parse_intent(json.dumps(reply), typed)
    assert intent.each and fields(intent) == [("buy", "XOM", 10, None), ("buy", "MSFT", 10, None)] and intent.said_how_much()
    order, am = decided(typed, intent)
    assert [(d["symbol"], d["shares"], d["will_trade"], d["est_cost"]) for d in order.trade_decisions] == [(s, 10, "yes_user", trade_amount(10, close_of(s))) for s in ("XOM", "MSFT")]
    assert [e["how_many"] for e in order.direct_orders] == ["typed", "typed"] and all("dollars" not in d for d in order.trade_decisions)
    am.perform_trades(order, Portfolio(user="Bill Gates", cash_avail=Decimal("20000")), trader(), MOMENT)
    assert [(t["symbol"], t["status"], t["shares"]) for t in order.trades] == [("XOM", "filled", 10), ("MSFT", "filled", 10)] and wo.verify_trades(order) == []


def test_a_decision_may_not_cost_more_than_the_amount_the_user_wrote_for_each(each_intent, friday_close):
    order, _ = decided(EACH, each_intent)
    rid = order.trade_decisions[1]["record_id"]
    order.trade_decisions[1]["dollars"] = Decimal("3000.01")
    assert any("the user wrote $3,000.00 for this stock and the decision's dollars are $3,000.01" in r for r in wo.verify_trade(order, rid, MOMENT))


# ---------------------------------------------------------------- E4: the recommendation path ("the top 3 with $3,000 each")

def bob_order() -> wo.WorkOrder:
    order = wo.WorkOrder.model_validate(copy.deepcopy(BOB_CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def prices_of(order) -> dict[str, Decimal]:
    return {r["symbol"]: price_door(r["timing"]["deal_price"]) for r in order.reviews}


def test_each_limits_every_approved_purchase_to_its_own_amount_and_the_pot_is_the_amount_times_the_purchases(friday_close):
    from test_account_manager_clarify import BOB, PROFILE
    order = bob_order()
    px = prices_of(order)
    told = []
    out = AccountManager(model=object(), model_id="each", clerk=clerk(), profile=PROFILE, screen=told.append).decide_trades(order, BOB, None, MOMENT, "auto", None, None, count=3, each=THREE)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert [(s, got[s]["will_trade"]) for s in ("CVX", "EOG", "PSX")] == [("CVX", "yes_auto"), ("EOG", "yes_auto"), ("PSX", "no_system_denied")]      # the Critic's denial is final
    for s in ("CVX", "EOG"):
        assert got[s]["shares"] == int(THREE // px[s]) and got[s]["est_cost"] == trade_amount(int(THREE // px[s]), px[s]) <= THREE
        assert got[s]["size"]["pot"] == Decimal("6000.00") and got[s]["size"]["asked"] == int(THREE // px[s]) and got[s]["size"]["binding"] == "asked" and "dollars" not in got[s]
        assert trade_amount(got[s]["shares"] + 1, px[s]) > THREE      # one more share would pass the amount for the stock
    assert (got["CVX"]["shares"], got["EOG"]["shares"]) == (13, 18) and (got["CVX"]["est_cost"], got["EOG"]["est_cost"]) == (Decimal("2858.57"), Decimal("2849.76"))
    assert sum(d["est_cost"] for d in out.trade_decisions) == Decimal("5708.33") <= Decimal("6000.00") and out.status == "decided" and out.errors == []
    assert any("How much: $3,000.00 in each of the 2 companies" in line for line in told)


def test_each_asks_nothing_in_ask_mode_the_user_already_said_how_much(friday_close):
    from test_account_manager_clarify import BOB, PROFILE
    cards, channel = [], channel_of([])
    out = AccountManager(model=object(), model_id="each", clerk=clerk(), profile=PROFILE).decide_trades(bob_order(), BOB, lambda card: cards.append(card) or True, MOMENT, "ask", None, None, clarify=channel, said_how_much=True, count=3, each=THREE)
    assert channel.cards == [] and [d["will_trade"] for d in out.trade_decisions] == ["yes_user", "yes_user", "no_system_denied"]      # no question about how much; only the yes or no of each trade
    assert len(cards) == 2 and all("WOULD BE PLACED  buy" in c["text"] for c in cards) and {d["size"]["pot"] for d in out.trade_decisions if d["size"]} == {Decimal("6000.00")}


def test_with_less_cash_than_the_amounts_the_cash_is_the_pot_and_every_purchase_is_still_at_most_its_amount(friday_close):
    from test_account_manager_clarify import PROFILE
    poor = Portfolio(user="Bob Jones", cash_avail=Decimal("4000"))
    out = AccountManager(model=object(), model_id="each", clerk=clerk(), profile=PROFILE).decide_trades(bob_order(), poor, None, MOMENT, "auto", None, None, count=3, each=THREE)
    bought = [d for d in out.trade_decisions if d["will_trade"] == "yes_auto"]
    assert {d["size"]["pot"] for d in bought} == {Decimal("4000.00")} and all(d["est_cost"] <= THREE for d in bought) and sum(d["est_cost"] for d in bought) <= Decimal("4000.00")
    assert len(bought) == 2 and all(d["shares"] >= 1 for d in bought)      # the round robin shares the cash: neither is left with nothing


def test_with_each_the_approved_and_the_neutral_purchases_are_all_limited_and_the_pot_counts_every_one(friday_close, allowed):
    """The real oil session: EOG and CVX approved, COP denied, six companies neutral. Asked for 5 companies with $3,000 each: the two approved and the three best neutral scores, each at most $3,000.00, the pot $15,000.00 (five
    times the amount, never above the $15,000 of cash)."""
    out = neutral_account(profile=OIL_USER).decide_trades(oil_order(), Portfolio(user="Auto Mode Check", cash_avail=Decimal("100000")), None, MOMENT, "auto", None, None, count=5, each=THREE)
    bought = [d for d in out.trade_decisions if d["will_trade"] == "yes_auto"]
    assert [d["symbol"] for d in bought][:2] == ["EOG", "CVX"] and len(bought) == 5 and [d.get("origin") for d in bought].count("neutral") == 3 and [d.get("origin") for d in bought][:2] == [None, None]
    assert all(d["est_cost"] <= THREE for d in bought) and all(d["size"]["asked"] is not None and d["size"]["binding"] == "asked" for d in bought)
    assert {d["size"]["pot"] for d in bought} == {Decimal("15000.00")} and sum(d["est_cost"] for d in bought) <= Decimal("15000.00")
    capped = neutral_account(profile=OIL_USER).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", None, None, count=5, each=Decimal("5000"))      # $15,000 of cash is the pot: five amounts of $5,000 is $25,000
    assert {d["size"]["pot"] for d in capped.trade_decisions if d["size"]} == {Decimal("15000.00")} and sum(d["est_cost"] for d in capped.trade_decisions) <= Decimal("15000.00")


def test_with_each_a_run_of_neutral_companies_alone_is_limited_too(friday_close, allowed):
    out = neutral_account().decide_trades(pam_advised(), PAM, None, MOMENT, "auto", None, None, each=THREE)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert got["SNDK"]["will_trade"] == "no_system_denied"      # the Critic's denial stays final
    assert (got["MU"]["shares"], got["AMD"]["shares"]) == (3, 5) and (got["MU"]["est_cost"], got["AMD"]["est_cost"]) == (Decimal("2334.93"), Decimal("2670.30")) and all(got[s]["will_trade"] == "yes_auto" for s in ("MU", "AMD"))
    assert {got[s]["size"]["pot"] for s in ("MU", "AMD")} == {Decimal("6000.00")}


def test_each_and_one_total_are_two_ways_to_say_how_much_and_nothing_is_decided(friday_close):
    from test_account_manager_clarify import BOB, PROFILE
    order = bob_order()
    out = AccountManager(model=object(), model_id="each", clerk=clerk(), profile=PROFILE).decide_trades(order, BOB, None, MOMENT, "auto", None, Decimal("5000"), each=THREE)
    assert out.trade_decisions == [] and out.status == "reviewed" and "two ways to say how much" in out.errors[-1]


def test_the_plan_of_each_is_the_plan_of_the_answer_an_amount_for_each_company_with_the_pot_of_the_run():
    from test_account_manager_clarify import PURCHASES
    plan = clarify.plan_each_dollars(THREE, PURCHASES, Decimal("15000"))
    answer = clarify.plan_choice(clarify.SizingChoice(unit="dollars", amount=3000, scope="each"), PURCHASES, Decimal("15000"), Decimal("15000"))
    assert plan.wanted == answer.wanted and plan.text == answer.text == "$3,000.00 in each of the 2 companies" and plan.balance is None and answer.budget is None
    assert plan.budget == Decimal("6000.00") and clarify.plan_each_dollars(THREE, PURCHASES, Decimal("4000")).budget == Decimal("4000.00")      # never above the cash


# ---------------------------------------------------------------- the desk carries it

def test_the_desk_plans_the_amount_for_each_and_never_a_total():
    """The REAL desk reads the real reply to "Buy the top 3 oil stocks with $3,000 each" and stops at the Data Clerk's node: the plan holds the amount for each and NO budget."""
    def no_clerk(order):
        raise RuntimeError("the Data Clerk is not part of this test")

    def state_of(typed):
        manager = lambda: AccountManager(model=Replay(FORMS[typed]["replies"]), model_id="recorded", profile=BILL_PROFILE)
        return desk.run_desk({"ask": {"text": typed, "default_count": 3}, "account": {"portfolio": BILL, "profile": BILL_PROFILE}}, clerk_factory=no_clerk, advisor_factory=no_clerk, funds_factory=manager, with_critic=True)

    plan = state_of(TOP3)["plan"]
    assert (plan["each"], plan["budget"], plan["count"], plan["said_how_much"]) == (THREE, None, 3, True)
    total = state_of(EV)["plan"]
    assert (total["each"], total["budget"]) == (None, Decimal("5000"))      # a total stays a total


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


# ---------------------------------------------------------------- the live session of 2026-11-20

def test_the_live_auto_session_that_said_each_spent_at_most_six_thousand_dollars():
    LIVE = json.loads((FIXTURES / "auto_direct_dollars_each_word.json").read_text(encoding="utf-8"))
    assert LIVE["request"] == EACH and LIVE["path"] == ["account_manager", "account_manager_direct", "trader"]
    intent = it.TradeIntent.model_validate(LIVE["intent"])
    assert intent.each is True and fields(intent) == [("buy", "XOM", None, 3000), ("buy", "MSFT", None, 3000)] and intent.budget is None
    order = wo.WorkOrder.model_validate(copy.deepcopy(LIVE["order"]))
    assert order.status == "traded" and order.errors == [] and wo.verify_trades(order) == [] and wo.pot_reasons(order) == [] and wo.verify_direct_orders(order.direct_orders) == []
    for d, t in zip(order.trade_decisions, order.trades):
        assert wo.verify_trade(order, d["record_id"], wo.parse_utc(d["decided_utc"])) == []
        assert (d["will_trade"], d["origin"], money(d["dollars"]), money(d["size"]["pot"]), d["size"]["how"]) == ("yes_user", "direct", Decimal("3000.00"), Decimal("6000.00"), "dollars")
        assert (t["status"], t["booked"], money(t["fill"]["amount"]), t["fill"]["slippage_bps"]) == ("filled", True, Decimal("3000.00"), 0)
    run = order.trader_runs[-1]
    spent = sum((money(t["fill"]["amount"]) for t in order.trades), Decimal(0))
    assert spent == Decimal("6000.00") and money(run["portfolio_before"]["cash"]) - money(run["portfolio_after"]["cash"]) == spent and run["booked"] is True
