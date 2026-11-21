"""RAISING FUNDS FOR THE SHARES TYPED WITH A RECOMMENDATION (design choices entry 142; design/direct_raise_funds_spec.md, rules R10 and R11; Jacob, 2026-11-21: "buy 10 of MU if recommended and raise funds if needed").

The REAL review of a real session (`tests/fixtures/session_real/bob_jones_oil.json`, 2026-11-14: XOM and EOG approved by the Critic, APA denied), the REAL sample portfolio ($51.23 and MSFT 50, TSLA 123, GM 60, `jacob_as_bob`), the real stored closes
of Friday 2026-11-13, the real Account Manager over the real Data Clerk, the REAL replies of its model to RaiseFunds' questions (recorded by the content of the question) and the real Trader with no model. "8 shares of XOM and 8 of EOG"
costs $2,631.28 against $51.23 of cash: $2,580.05 short. The shares typed are each approved purchase's limit (entry 140, C2); with the words (or the setting) the shortfall is raised in ONE plan before the Trader sells and buys.
"""
from decimal import Decimal

import pytest
import raise_funds_helpers as H
from handshake_helpers import MOMENT, friday_close, jacob_as_bob, trader  # noqa: F401  (friday_close is a fixture)
from raise_funds_helpers import Cards, Model, reviewed_order, sample_manager

from prometheus.agents.account_manager import funding
from prometheus.common import work_order as wo

pytestmark = pytest.mark.usefixtures("friday_close")

TYPED = {"XOM": 8, "EOG": 8}
COST = Decimal("2631.28")           # 8 x $170.59 + 8 x $158.32 at the real closes
SHORT = COST - Decimal("51.23")


def decide(mode="auto", raise_funds=True, approver=None, budget=None, text="", typed=TYPED, **options):
    am = sample_manager(Model())
    lines = []
    am.screen = lines.append
    order, portfolio = reviewed_order(), jacob_as_bob()
    am.decide_trades(order, portfolio, approver, MOMENT, mode, budget=budget, shares_of=typed, raise_funds=raise_funds, request_text=text, **options)
    return am, order, portfolio, lines


def trade(am, order, portfolio, approver=None):
    funds = am.fund_decisions(order, portfolio, MOMENT)
    am.perform_trades(order, portfolio, trader(), MOMENT, funding=funds, approver=approver)
    return funds


def test_auto_mode_with_the_words_sizes_the_typed_shares_whole_raises_the_whole_shortfall_in_one_plan_and_sells_first_then_buys_both():
    am, order, portfolio, lines = decide("auto", True)
    xom, apa, eog = order.trade_decisions
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in (xom, apa, eog)] == [("XOM", "yes_auto", 8), ("APA", "no_system_denied", 0), ("EOG", "yes_auto", 8)]       # each gets exactly its typed shares; the denied one buys nothing
    assert xom["funding"] == eog["funding"] == {"shortfall": SHORT, "plan_id": None, "by": "words"} and xom["size"]["pot"] == eog["size"]["pot"] == COST      # the pot is what the typed shares cost: cash plus the money to raise, no more
    assert any("2,631.28" in line and "you said to raise funds" in line for line in lines)
    funds = trade(am, order, portfolio, approver=Cards())                                                          # Auto mode with the words: the approver is never asked
    assert funds["basket"] == ["XOM", "EOG"] and funds["need"] == SHORT and funds["ask_user"] is False and funds["status"] == "plan" and "APA" not in funds["basket"]
    (sale,) = funds["plan"]["sells"]
    assert (sale["symbol"], sale["shares"]) == ("GM", 31) and sale["proceeds"] >= SHORT and sale["gain"] > 0                       # a profit is taken; XOM and EOG are never sold
    assert [(t["trade_id"], t["symbol"], t["side"], t["status"]) for t in order.trades] == [("T1", "XOM", "buy", "filled"), ("T1-F1", "GM", "sell", "filled"), ("T2", "EOG", "buy", "filled")]
    assert order.trader_runs[-1]["portfolio_after"]["cash"] == Decimal("51.23") + sale["proceeds"] - COST and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0 and wo.verify_trades(order) == []


def test_auto_mode_without_the_words_keeps_what_it_always_did_the_cash_buys_what_it_can_and_nothing_is_sold_and_the_user_is_told_what_to_say():
    am, order, portfolio, lines = decide("auto", False)
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in order.trade_decisions if d["symbol"] != "APA"] == [("XOM", "no_funds", 0), ("EOG", "no_funds", 0)] and all("funding" not in d for d in order.trade_decisions)
    assert any(funding.NOT_SAID in line and "The cash buys what it can" in line for line in lines)
    assert am.fund_decisions(order, portfolio, MOMENT) is None and not funding.needs_funding(order)


def test_ask_mode_without_the_words_asks_once_for_both_purchases_and_a_yes_raises_the_money():
    cards = Cards(raise_funds=True, approve=True, funding=True)
    am, order, portfolio, _ = decide("ask", False, approver=cards)
    assert cards.kinds == ["raise_funds", "approve", "approve"]                                                    # one question about the money, then each trade as always (entry 79)
    ask = cards.cards[0]
    assert ask["shortfall"] == SHORT and [(b["symbol"], b["shares"]) for b in ask["buys"]] == [("XOM", 8), ("EOG", 8)] and "XOM at $170.59 and 8 EOG at $158.32 cost $2,631.28" in ask["text"]
    assert [d["funding"]["by"] for d in order.trade_decisions if d["will_trade"] == "yes_user"] == ["asked", "asked"]
    trade(am, order, portfolio, approver=cards)
    assert cards.kinds == ["raise_funds", "approve", "approve", "funding"] and [t["status"] for t in order.trades] == ["filled", "filled", "filled"] and wo.verify_trades(order) == []


@pytest.mark.parametrize("answer", [False, None])
def test_ask_mode_a_no_or_no_answer_sells_nothing_and_the_cash_buys_what_it_can(answer):
    cards = Cards(raise_funds=answer, approve=True)
    am, order, portfolio, lines = decide("ask", False, approver=cards)
    assert cards.kinds == ["raise_funds"] and all("funding" not in d for d in order.trade_decisions) and not funding.needs_funding(order)                  # no yes to approve either: nothing was sized to a trade
    assert any("Nothing is sold" in line for line in lines) and am.fund_decisions(order, portfolio, MOMENT) is None


def test_ask_mode_with_the_words_asks_about_each_trade_and_shows_the_plan_and_no_question_about_raising_funds():
    cards = Cards(approve=True, funding=True)
    am, order, portfolio, _ = decide("ask", True, approver=cards)
    trade(am, order, portfolio, approver=cards)
    assert cards.kinds == ["approve", "approve", "funding"] and cards.cards[-1]["buys"][0]["symbol"] == "XOM" and [b["symbol"] for b in cards.cards[-1]["buys"]] == ["XOM", "EOG"]
    assert [(leg["symbol"], leg["shares"]) for leg in cards.cards[-1]["legs"]] == [("GM", 31)] and [t["status"] for t in order.trades] == ["filled", "filled", "filled"]


def test_a_budget_the_user_gave_is_never_exceeded_so_only_what_the_typed_shares_need_within_it_is_raised():
    am, order, portfolio, _ = decide("auto", True, budget=H.BUDGET)
    xom, _, eog = order.trade_decisions
    spent = xom["est_cost"] + eog["est_cost"]
    assert (xom["shares"], eog["shares"]) == (6, 6) and spent <= H.BUDGET and xom["size"]["pot"] == eog["size"]["pot"] == H.BUDGET and xom["funding"]["shortfall"] == spent - Decimal("51.23")      # fewer than typed: the budget is the pot
    funds = trade(am, order, portfolio)
    assert funds["need"] == spent - Decimal("51.23") and order.trader_runs[-1]["portfolio_after"]["cash"] >= 0 and wo.verify_trades(order) == []
    assert sum(t["fill"]["amount"] for t in order.trades if t["side"] == "buy") <= H.BUDGET                      # never more than the budget (entry 111)


def test_a_purchase_with_no_typed_shares_alone_never_triggers_it():
    am = sample_manager(Model())
    plain, portfolio = reviewed_order(), jacob_as_bob()
    am.decide_trades(plain, portfolio, None, MOMENT, "auto", raise_funds=True)                                     # no shares typed at all: nothing to raise, sized to the cash as ever
    assert all("funding" not in d for d in plain.trade_decisions) and not funding.needs_funding(plain) and am.fund_decisions(plain, portfolio, MOMENT) is None


# ---------------------------------------------------------------- entry 143, P1: only SOME of the purchases have typed shares

ONLY_XOM = {"XOM": 8}
XOM_COST = Decimal("1364.72")           # 8 x $170.59 at the real close
XOM_SHORT = XOM_COST - Decimal("51.23")


def test_when_only_some_purchases_have_typed_shares_the_typed_ones_are_paid_for_in_full_and_the_others_get_nothing_never_money_raised_by_selling():
    am, order, portfolio, lines = decide("auto", True, typed=ONLY_XOM)
    xom, apa, eog = order.trade_decisions
    assert (xom["symbol"], xom["will_trade"], xom["shares"]) == ("XOM", "yes_auto", 8) and xom["funding"] == {"shortfall": XOM_SHORT, "plan_id": None, "by": "words"} and xom["size"]["pot"] == XOM_COST      # the typed shares, in full
    assert (eog["symbol"], eog["will_trade"], eog["shares"], eog["size"], "funding" in eog) == ("EOG", "no_funds", 0, None, False)      # approved by the Critic, no typed shares: it gets nothing
    assert "EOG has no number of shares, so it gets nothing: the cash and the money raised go to the shares you typed" in eog["reason"] and "Say how many shares of EOG you want" in eog["reason"]
    assert any("EOG has no number of shares" in line for line in lines) and (apa["symbol"], apa["will_trade"]) == ("APA", "no_system_denied")
    funds = trade(am, order, portfolio, approver=Cards())                                                          # ONE plan for the shortfall of the typed shares only
    assert funds["basket"] == ["XOM"] and funds["need"] == XOM_SHORT and funds["status"] == "plan"
    (sale,) = funds["plan"]["sells"]
    assert sale["symbol"] not in ("XOM", "EOG") and sale["proceeds"] >= XOM_SHORT
    assert [(t["trade_id"], t["symbol"], t["side"], t["status"]) for t in order.trades] == [("T1", "XOM", "buy", "filled"), ("T1-F1", sale["symbol"], "sell", "filled")]      # EOG was never placed
    after = order.trader_runs[-1]["portfolio_after"]["cash"]
    assert after == Decimal("51.23") + sale["proceeds"] - XOM_COST and after >= 0 and wo.verify_trades(order) == []


def test_the_neutral_companies_with_no_typed_shares_get_nothing_either_when_the_typed_ones_are_funded(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")                                                    # the real session's neutral companies count as approved and have no typed shares
    am, order, portfolio, lines = decide("auto", True)                                                             # XOM and EOG are typed
    neutral = [d for d in order.trade_decisions if d.get("origin") == "neutral"]
    assert neutral and all(d["will_trade"] == "no_funds" and d["shares"] == 0 and "funding" not in d and "has no number of shares" in d["reason"] for d in neutral)
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in order.trade_decisions if d["symbol"] in ("XOM", "EOG")] == [("XOM", "yes_auto", 8), ("EOG", "yes_auto", 8)]      # the typed ones are paid for in full, as before
    assert order.trade_decisions[0]["funding"]["shortfall"] == SHORT and am.fund_decisions(order, portfolio, MOMENT)["basket"] == ["XOM", "EOG"]


def test_ask_mode_without_the_words_says_in_its_one_question_that_the_untyped_purchase_gets_nothing():
    cards = Cards(raise_funds=True, approve=True)
    am, order, portfolio, _ = decide("ask", False, approver=cards, typed=ONLY_XOM)
    ask = cards.cards[0]
    assert cards.kinds == ["raise_funds", "approve"] and ask["left_out"] == ["EOG"] and [(b["symbol"], b["shares"]) for b in ask["buys"]] == [("XOM", 8)] and ask["shortfall"] == XOM_SHORT
    assert "EOG has no number of shares, so it gets nothing" in ask["text"] and [d["will_trade"] for d in order.trade_decisions if d["symbol"] == "EOG"] == ["no_funds"]      # EOG is not asked about: it has no size


def test_with_no_permission_the_untyped_purchase_keeps_the_round_robin_with_the_cash_as_it_always_did():
    am, order, portfolio, lines = decide("auto", False, typed=ONLY_XOM)
    assert all("funding" not in d for d in order.trade_decisions) and not funding.needs_funding(order) and any(funding.NOT_SAID in line for line in lines)
    assert not any("has no number of shares" in d["reason"] for d in order.trade_decisions)                         # nothing was sold, so nothing was left out


def test_the_setting_is_the_same_as_the_words(monkeypatch):
    monkeypatch.setenv("ALWAYS_RAISE_FUNDS", "true")
    am, order, portfolio, _ = decide("auto", False, text="buy 8 of XOM and 8 of EOG if recommended")
    assert {d["funding"]["by"] for d in order.trade_decisions if "funding" in d} == {"setting"}
    am, order, portfolio, lines = decide("auto", False, text="buy 8 of XOM and 8 of EOG if recommended, but don't sell anything")      # the words of the request win over the setting
    assert all("funding" not in d for d in order.trade_decisions) and any(funding.NOT_SAID in line for line in lines)
