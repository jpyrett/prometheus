"""A NEUTRAL RECOMMENDATION IS APPROVED (design choices entry 91; Jacob, 2026-11-15, with `ALLOW_NEUTRAL_RECOMMENDATIONS=true` in `.env`), and no profile limit applies to anything.

REAL DATA ONLY. `tests/fixtures/session_real/pam_smith_neutral.json` is a capture of the real session `./test_prometheus.sh --session --account <Pam Smith> --yes --prompt "Buy MU, SNDK and AMD if the recommendations pass. Go ahead
and trade if the recommendations pass."` (2026-11-15; Pam Smith, 30, aggressive, $50,000, Auto mode; the paid refresh of the data; the real Account Manager, Data Clerk, Advisor, Critic (GLM 5.3) and Trader (gpt-oss-120b), the
real Postgres book, the simulated clock of Friday 2026-11-13): SNDK was bullish and the Critic DENIED it (risk above the profile, a falling knife: final), MU and AMD were scored NEUTRAL by the Advisor (no strategy passed), so
they were treated as approved and bought in equal dollars (entry 91's default for a run of neutral companies alone): MU 32 shares, AMD 46, $502.59 left. Since entry 109 (Jacob, 2026-11-15: *"use round-robin for neutral-only runs too. Use split evenly
if the user specifies it in the prompt"*) the same request, which did not ask for an even split, is spread round-robin (MU 38, AMD 38) and equal dollars come only with the user's words ("split evenly"). The tests re-decide that order with the real Account Manager over the real stored closes (the Data Clerk's quotes) and trade it with
the real Trader, and break the permission one field at a time. No agent and no model is replaced by a stand-in. The tests are skipped when the stored bars are not on the machine.
"""
from decimal import Decimal
import ast
import copy
import inspect
import json
from datetime import timedelta
from pathlib import Path

import pytest
from handshake_helpers import FRIDAY, MOMENT, close_of, friday_close, trader  # noqa: F401  (friday_close is a fixture)
from test_account_manager_clarify import RECORDED, Replay, channel_of

from prometheus.agents.account_manager import neutral
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.common import shares as sh
from prometheus.common import work_order as wo
from prometheus.common.money import money as money_door
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common.user_profile import UserProfile

pytestmark = pytest.mark.usefixtures("friday_close")

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "pam_smith_neutral.json").read_text(encoding="utf-8"))
PAM = Portfolio(user="Pam Smith", cash_avail=Decimal("50000"))
PAM_PROFILE = UserProfile(user="Pam Smith", risk_tolerance="aggressive", age=30, trade_mode="auto")


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")


def advised() -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished and before the Account Manager decided: the Advisor's recommendation (SNDK) and its candidates (MU and AMD, neutral), the Critic's denial of SNDK."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def neutral_only() -> wo.WorkOrder:
    """The same order with nothing for the Critic to review: the Advisor recommended no company, and two were neutral."""
    order = advised()
    order.recommendations, order.reviews, order.status = [], [], "recommended"
    return order


def account(**options) -> AccountManager:
    # AMENDED (entry 110): `fractional_even_split` is False here, so that every test of this file keeps pinning the WHOLE-share even split of entry 108 (the Account Manager splits in fractions unless its switch is off);
    # the fractional split of the same real oil session is in tests/test_fractional_even_split.py
    options.setdefault("fractional_even_split", False)
    return AccountManager(model=options.pop("model", object()), model_id="neutral", clerk=DataClerk(model=object()), profile=options.pop("profile", PAM_PROFILE), **options)


def decided(order, mode="auto", approver=None, portfolio=PAM, **kw):
    return account().decide_trades(order, portfolio, approver, MOMENT, mode, **kw)


# ---------------------------------------------------------------- the setting

@pytest.mark.parametrize("value, on", [("true", True), ("TRUE", True), ("yes", True), ("on", True), ("1", True), (" true ", True), ("false", False), ("no", False), ("off", False), ("0", False), ("maybe", False), ("truee", False)])
def test_the_setting_is_true_yes_on_or_1_and_anything_else_is_false(monkeypatch, value, on):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", value)
    assert neutral.allow_neutral_recommendations() is on


def test_with_the_setting_off_a_neutral_company_is_not_recommended_and_nothing_is_decided_for_it(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "false")
    assert neutral.neutral_candidates(advised()) == []
    out = decided(advised())
    assert [(d["symbol"], d["will_trade"]) for d in out.trade_decisions] == [("SNDK", "no_system_denied")] and out.status == "decided"


def test_with_the_setting_on_every_company_the_advisor_scored_neutral_counts_whoever_named_it(allowed):
    """AMENDED (entry 107; Jacob, 2026-11-15: *"Neutral applies to all recommendations regardless of who typed the company"*). This test said a company the Advisor found for a theme and the user never named was NEVER
    bought as neutral (`origin: "discover"`). The origin does not matter now: a neutral stance and no failed gate is all it takes, and the best Advisor score comes first."""
    names = neutral.neutral_candidates(advised())
    assert [(c["symbol"], c["stance"], c["origin"]) for c in names] == [("MU", "neutral", "subject"), ("AMD", "neutral", "subject")]
    order = advised()
    order.candidates[0]["gate"] = "freshness"                                                    # a gate that failed is not a stance: it stops the company
    assert [c["symbol"] for c in neutral.neutral_candidates(order)] == ["AMD"]                  # AMENDED (entry 105): a company the user already holds no longer stops it (the Advisor knows no holding: see the next test)
    order.candidates[1]["gate"] = "red_flag"
    assert neutral.neutral_candidates(order) == []
    order = advised()
    order.candidates[0]["origin"] = "discover"                                                   # a company the Advisor found, that the user never named: it counts (entry 107)
    order.candidates[1]["origin"] = "sources"
    assert [(c["symbol"], c["origin"]) for c in neutral.neutral_candidates(order)] == [("MU", "discover"), ("AMD", "sources")]
    assert [c["symbol"] for c in neutral.neutral_candidates(order)] == [c["symbol"] for c in sorted(order.candidates, key=lambda c: -c["score"]) if c["stance"] == "neutral"]      # best Advisor score first
    order.candidates[1]["stance"] = "bearish"                                                    # a bearish stance still trades nothing
    assert [c["symbol"] for c in neutral.neutral_candidates(order)] == ["MU"]
    assert neutral.neutral_candidates(wo.WorkOrder.model_validate({**CAPTURE["order"], "status": "new"})) == []     # an order the Advisor has not answered has none


# ---------------------------------------------------------------- the Account Manager approves them, and the Critic's denial stays final

def test_the_real_session_is_reproduced_neutral_companies_are_approved_round_robin_and_the_critics_denial_is_final(allowed):
    """AMENDED (entry 109). The live run spread Pam's $50,000 over MU and AMD in equal dollars (MU 32, AMD 46, $502.59 left): entry 91 gave a run of neutral companies alone that default. Jacob (2026-11-15): *"use round-robin for
    neutral-only runs too. Use split evenly if the user specifies it in the prompt"*. Her request did not say it, so the Account Manager now buys a share of each in turn, by hand from the real closes (MU $778.31, AMD $534.06):
    38 rounds cost $49,870.06 and the $129.94 left buys neither (MU 38, AMD 38). With "split evenly" asked, the equal dollars of the live run come back (the next test)."""
    out = decided(advised())
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert out.status == "decided" and out.errors == [] and [d["symbol"] for d in out.trade_decisions] == ["SNDK", "MU", "AMD"]
    assert got["SNDK"]["will_trade"] == "no_system_denied" and got["SNDK"]["shares"] == 0                                           # bullish, and the Critic denied it: a denial is final, the setting does not touch it
    for symbol in ("MU", "AMD"):
        d = got[symbol]
        assert (d["will_trade"], d["origin"], d["action"], d["mode"], d["review_id"]) == ("yes_auto", "neutral", "open", "auto", None) and d["price"] == close_of(symbol) == {"MU": Decimal("778.31"), "AMD": Decimal("534.06")}[symbol]
        assert d["shares"] == 38 and d["size"]["allocation"] == "round_robin" and "balance" not in d                                    # AMENDED: MU 32 and AMD 46 in equal dollars, the live run
        assert "ALLOW_NEUTRAL_RECOMMENDATIONS is on" in d["reason"] and "no profile limit" in d["reason"]
    spent = got["MU"]["est_cost"] + got["AMD"]["est_cost"]
    assert spent == Decimal("49870.06") and PAM.cash_avail - spent == Decimal("129.94")                                                     # the funds cover the orders: the only check


def test_with_split_evenly_in_the_request_the_neutral_companies_get_the_live_runs_equal_dollars_again(allowed):
    """Entry 109: the user's own words override the round robin. The same real session decided with `balance="even"` (what "split evenly" in the request comes to) is the live run's MU 32 and AMD 46, $25,000 each; the $527.32
    the equal amounts leave over (entry 108) buys neither (MU's next share is $778.31, AMD's $534.06)."""
    live = {d["symbol"]: d for d in CAPTURE["order"]["trade_decisions"]}
    out = decided(advised(), balance="even")
    got = {d["symbol"]: d for d in out.trade_decisions}
    for symbol in ("MU", "AMD"):
        d = got[symbol]
        assert (d["shares"], d["est_cost"], d["price"]) == (sh.quantity(live[symbol]["shares"]), money_door(live[symbol]["est_cost"]), price_door(live[symbol]["price"])) and (d["shares"], d["size"]["allocation"]) == ({"MU": 32, "AMD": 46}[symbol], "even")
        assert d["balance"] == {"mode": "even", "target_dollars": Decimal("25000"), "of": 2} and d["shares"] == int(Decimal("25000") // d["price"]) and Decimal("25000") - d["est_cost"] < d["price"]                  # equal dollars, whole shares
    assert PAM.cash_avail - got["MU"]["est_cost"] - got["AMD"]["est_cost"] == Decimal("527.32")


def test_a_neutral_company_the_user_already_holds_counts_as_approved_like_any_other_named_company(allowed):
    """Entry 105 (design/approved_or_denied_spec.md 4.3 and 8.2). A neutral company that the user already holds used to be left out (the Advisor's holdings pass tagged it `held`, and `is_neutral_candidate` refused a held
    candidate). The Advisor is given no portfolio and has no such tag: the Account Manager, which knows what the user holds, sizes the purchase. Pam holds 10 shares of AMD and both neutral companies are bought (round-robin since entry 109: 38 of each)."""
    holds = Portfolio(user="Pam Smith", cash_avail=Decimal("50000"), positions=[Position(symbol="AMD", shares=10, cost_basis=Decimal("100"), acquired=FRIDAY - timedelta(days=30))])
    out = decided(advised(), portfolio=holds)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert [(got[s]["will_trade"], got[s]["origin"]) for s in ("MU", "AMD")] == [("yes_auto", "neutral"), ("yes_auto", "neutral")] and got["AMD"]["shares"] == 38       # AMENDED (entry 109): 46 in equal dollars, 38 round-robin
    assert [wo.verify_trade(out, got[s]["record_id"], MOMENT) for s in ("MU", "AMD")] == [[], []]


def test_with_nothing_for_the_critic_to_review_only_the_neutral_companies_are_decided(allowed):
    out = decided(neutral_only())
    assert out.status == "decided" and out.reviews == [] and [(d["symbol"], d["will_trade"], d["origin"]) for d in out.trade_decisions] == [("MU", "yes_auto", "neutral"), ("AMD", "yes_auto", "neutral")]


def test_one_neutral_company_gets_all_the_money_and_a_budget_of_the_user_is_the_pot(allowed):
    order = neutral_only()
    order.candidates = [c for c in order.candidates if c["symbol"] == "AMD"]
    (d,) = decided(order).trade_decisions
    assert d["shares"] == int(Decimal("50000") // close_of("AMD")) and d["size"]["allocation"] == "round_robin"                               # nothing limits one stock to a share of the account (AMENDED, entry 109: no "balance" of one purchase: it is not an even split)
    capped = decided(neutral_only(), budget=Decimal("10000"))
    assert [(x["symbol"], x["shares"]) for x in capped.trade_decisions] == [("MU", 8), ("AMD", 7)]      # AMENDED (entry 109), by hand: 7 rounds of $1,312.37 cost $9,186.59 and the $813.41 left buys MU an eighth share ($778.31); $35.10 stays. It was 6 and 9 in equal dollars of $5,000
    even = decided(neutral_only(), budget=Decimal("10000"), balance="even")
    assert [(x["symbol"], x["shares"]) for x in even.trade_decisions] == [("MU", 6), ("AMD", 9)]         # "split evenly" still gives equal dollars: $5,000 each buys MU 6 ($4,669.86) and AMD 9 ($4,806.54), and the $523.60 left is under the next share of either


def test_a_neutral_company_the_cash_cannot_buy_a_share_of_is_no_funds_and_one_with_no_price_is_denied(allowed):
    poor = decided(neutral_only(), portfolio=Portfolio(user="Pam Smith", cash_avail=Decimal("600")))
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in poor.trade_decisions] == [("MU", "no_funds", 0), ("AMD", "yes_auto", 1)]      # AMENDED (entry 109): in equal dollars $300 each bought neither; round-robin, MU ($778.31) does not fit the $600 and AMD's one share ($534.06) does
    even = decided(neutral_only(), portfolio=Portfolio(user="Pam Smith", cash_avail=Decimal("600")), balance="even")
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in even.trade_decisions] == [("MU", "no_funds", 0), ("AMD", "no_funds", 0)]      # "split evenly": $300 each buys no share of MU ($778.31) or AMD ($534.06)
    order = neutral_only()
    order.candidates[0] = {**order.candidates[0], "symbol": "ZZZZ"}
    out = decided(order)
    assert [(d["symbol"], d["will_trade"]) for d in out.trade_decisions] == [("ZZZZ", "no_system_denied"), ("AMD", "yes_auto")] and "no current price for ZZZZ" in out.trade_decisions[0]["reason"]


def test_in_ask_mode_the_user_is_asked_about_each_neutral_purchase(allowed):
    cards = []
    out = decided(neutral_only(), mode="ask", approver=lambda card: cards.append(card) or (card["symbol"] == "MU"))
    assert [(d["symbol"], d["will_trade"], d["asked_user"]) for d in out.trade_decisions] == [("MU", "yes_user", True), ("AMD", "no_user_denied", True)]
    assert [c["symbol"] for c in cards] == ["MU", "AMD"] and "A NEUTRAL RECOMMENDATION, TREATED AS APPROVED" in cards[0]["text"] and "Advisor score 62" in cards[0]["text"]
    waiting = decided(neutral_only(), mode="ask", approver=None)
    assert [d["will_trade"] for d in waiting.trade_decisions] == ["pending_user", "pending_user"] and not any(d["will_trade"] in wo.TRADING for d in waiting.trade_decisions)


# ---------------------------------------------------------------- Ask mode: the user is asked for the number before anything is proposed (Jacob, 2026-11-15)

REQUEST = "I want to buy MU and AMD. Go ahead and trade if the recommendations pass."          # no amount in it: "I want to buy <X>"


def ask_how_much(scenario: str, said_how_much: bool = False, mode: str = "ask"):
    """Pam's neutral companies in a mode, with the model's REAL recorded replies to a user who answers `scenario`'s words (tests/fixtures/session_real/clarify_ask_mode.json)."""
    recorded = RECORDED[scenario]
    model, channel, cards = Replay(recorded["replies"]), channel_of(recorded["answers"]), []
    out = account(model=model).decide_trades(neutral_only(), PAM, lambda card: cards.append(card) or True, MOMENT, mode, clarify=channel, said_how_much=said_how_much, request_text=REQUEST)
    return out, channel, cards, model


def test_in_ask_mode_a_request_with_no_number_is_asked_how_much_of_the_neutral_companies_and_the_answer_is_what_is_proposed(allowed):
    out, channel, cards, model = ask_how_much("percent_each_no_raise_needed")             # the user answers "5% of my account in each company": $2,500 of each
    assert len(channel.cards) == 1 and "How much" in channel.cards[0]["question"] and len(model.seen) == 2                      # asked once, before any card of a trade was shown
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert (got["MU"]["shares"], got["AMD"]["shares"]) == (int(Decimal("2500") // close_of("MU")), int(Decimal("2500") // close_of("AMD"))) == (3, 4)
    assert [(d["will_trade"], d["size"]["binding"], d["size"]["asked"]) for d in out.trade_decisions] == [("yes_user", "asked", 3), ("yes_user", "asked", 4)]      # AMENDED (entry 107): the sizing is the approved ones' (one call), so the decision's reason no longer says "the 3 shares the user asked for": `size.asked` does
    assert [(c["symbol"], c["shares"]) for c in cards] == [("MU", 3), ("AMD", 4)] and out.errors == []                          # the user is then shown exactly these, and says yes to each


def test_a_number_of_shares_the_user_gives_is_the_most_of_each_neutral_company_that_is_bought(allowed):
    out, channel, cards, _ = ask_how_much("shares_each_raise_yes")                                                                 # "10 shares of each"
    assert [(d["symbol"], d["shares"], d["size"]["asked"]) for d in out.trade_decisions] == [("MU", 10, 10), ("AMD", 10, 10)]
    assert sum(d["est_cost"] for d in out.trade_decisions) < Decimal("50000") and [c["shares"] for c in cards] == [10, 10]


def test_one_set_of_questions_covers_what_the_critic_approved_and_the_neutral_companies(allowed):
    order = advised()                                                                                                              # SNDK denied by the Critic, MU and AMD neutral
    recorded = RECORDED["percent_each_no_raise_needed"]
    model, channel = Replay(recorded["replies"]), channel_of(recorded["answers"])
    out = account(model=model).decide_trades(order, PAM, lambda card: True, MOMENT, "ask", clarify=channel, request_text=REQUEST)
    assert len(channel.cards) == 1 and [(d["symbol"], d["will_trade"], d["shares"]) for d in out.trade_decisions] == [("SNDK", "no_system_denied", 0), ("MU", "yes_user", 3), ("AMD", "yes_user", 4)]


def test_nothing_is_asked_in_auto_mode_or_when_the_request_already_said_how_much_or_with_nobody_to_ask(allowed):
    out, channel, _, model = ask_how_much("percent_each_no_raise_needed", mode="auto")
    assert channel.cards == [] and model.seen == [] and [d["will_trade"] for d in out.trade_decisions] == ["yes_auto", "yes_auto"] and out.trade_decisions[0]["shares"] > 20          # Auto never asks: the cash, a share of each in turn (entry 109)
    out, channel, _, model = ask_how_much("percent_each_no_raise_needed", said_how_much=True)
    assert channel.cards == [] and model.seen == [] and [d["shares"] for d in out.trade_decisions] == [38, 38]       # AMENDED (entry 109): a share of each in turn, not the equal dollars (32 and 46) it was
    silent = account().decide_trades(neutral_only(), PAM, None, MOMENT, "ask")                                                       # no channel to the user: the proposal is the most the cash buys, and waits for a yes
    assert [d["will_trade"] for d in silent.trade_decisions] == ["pending_user", "pending_user"]


# ---------------------------------------------------------------- the Trader's permission

def test_the_permission_is_the_advisors_own_neutral_record_and_runs_out_with_the_price(allowed):
    order = decided(advised())
    for d in order.trade_decisions[1:]:
        assert wo.verify_trade(order, d["record_id"], MOMENT) == []
        until = wo.parse_utc(d["valid_until_utc"])
        assert until == MOMENT + timedelta(minutes=30) and wo.verify_trade(order, d["record_id"], until + timedelta(seconds=1))[0].startswith(wo.EXPIRY_PREFIX)
    assert wo.verify_trade(order, order.trade_decisions[0]["record_id"], MOMENT)                                                       # SNDK: the Critic's denial still stops it


def test_the_word_neutral_alone_permits_nothing(allowed):
    base = decided(advised())
    mu = base.trade_decisions[1]
    gone = copy.deepcopy(base)
    gone.candidates = [c for c in gone.candidates if c["symbol"] != "MU"]
    assert "holds no neutral score of the Advisor's for 'MU'" in "; ".join(wo.verify_trade(gone, mu["record_id"], MOMENT))
    bullish = copy.deepcopy(base)
    next(c for c in bullish.candidates if c["symbol"] == "MU")["stance"] = "bullish"
    assert "holds no neutral score" in "; ".join(wo.verify_trade(bullish, mu["record_id"], MOMENT))
    gated = copy.deepcopy(base)
    next(c for c in gated.candidates if c["symbol"] == "MU")["gate"] = "bars_stale"
    assert "holds no neutral score" in "; ".join(wo.verify_trade(gated, mu["record_id"], MOMENT))
    sale = copy.deepcopy(base)
    sale.trade_decisions[1]["action"] = "reduce"
    assert "a neutral recommendation is only a purchase" in "; ".join(wo.verify_trade(sale, mu["record_id"], MOMENT))
    reviewed = copy.deepcopy(base)
    reviewed.reviews = [*reviewed.reviews, {"record_id": mu["record_id"], "approved": True}]
    assert "a neutral recommendation has no Critic review" in "; ".join(wo.verify_trade(reviewed, mu["record_id"], MOMENT))
    nobody = copy.deepcopy(base)
    nobody.trade_decisions[1]["will_trade"] = "pending_user"
    assert "not one of yes_auto, yes_user" in "; ".join(wo.verify_trade(nobody, mu["record_id"], MOMENT))
    del base.trade_decisions[1]["origin"]
    assert wo.verify_trade(base, mu["record_id"], MOMENT) == [f"there is no Critic review of recommendation {mu['record_id']}"]            # not neutral any more: the old rule


def test_a_critic_denial_is_never_turned_into_an_approval_even_if_the_decision_says_neutral(allowed):
    base = decided(advised())
    sndk = base.trade_decisions[0]
    assert base.reviews and base.reviews[0]["symbol"] == "SNDK" and not base.reviews[0]["approved"]
    forged = copy.deepcopy(base)
    forged.trade_decisions[0].update(origin="neutral", will_trade="yes_auto", shares=5, price=Decimal("100"), action="open", valid_until_utc="2026-11-13T21:00:00Z")
    reasons = "; ".join(wo.verify_trade(forged, sndk["record_id"], MOMENT))
    assert "holds no neutral score of the Advisor's for 'SNDK'" in reasons and "a neutral recommendation has no Critic review" in reasons                  # a bullish company with a review is not a neutral candidate


# ---------------------------------------------------------------- the Trader trades it, one trade at a time, through the same handshake

def test_the_trader_buys_the_neutral_companies_through_the_handshake_and_the_portfolio_shows_them(allowed):
    order = decided(advised())
    am = account()
    am.perform_trades(order, PAM, trader(), MOMENT)
    trades = {t["symbol"]: t for t in order.trades}
    assert set(trades) == {"MU", "AMD"} and order.status == "traded" and order.errors == [] and wo.verify_trades(order) == []
    for symbol, t in trades.items():
        assert (t["status"], t["side"], t["review_id"], t["shares"], t["kind"]) == ("filled", "buy", None, 38, "recommendation") and t["order_type"] == "market"
        assert abs(t["fill"]["price"] / close_of(symbol) - 1) < Decimal("0.002")
    after = order.trader_runs[-1]["portfolio_after"]
    assert after["cash"] == Decimal("50000") - sum(t["fill"]["amount"] for t in trades.values()) < 1000 and {p["symbol"]: p["shares"] for p in after["positions"]} == {"MU": 38, "AMD": 38}      # AMENDED (entry 109): round-robin, 38 of each (the live run bought MU 32 and AMD 46 in equal dollars)


def test_a_neutral_purchase_is_cut_to_the_cash_like_any_recommended_one_never_refused_for_a_limit(allowed):
    order = decided(advised())
    am = account()
    am.perform_trades(order, Portfolio(user="Pam Smith", cash_avail=Decimal("26000")), trader(), MOMENT)
    first, second = order.trades
    assert first["status"] == "filled" and first["fill"]["shares"] == first["shares"] and second["status"] in ("filled", "rejected") and wo.verify_trades(order) == []
    assert (second["fill"] or {}).get("shares", 0) < second["shares_approved"] or second["status"] == "rejected"                          # the cash that is left decides it (the handshake), nothing else


# ---------------------------------------------------------------- what the user reads, and the real session's own record

def test_the_report_says_which_companies_were_treated_as_approved_and_why(allowed):
    text = "\n".join(neutral.neutral_lines(decided(advised())))
    assert "NEUTRAL RECOMMENDATIONS  treated as approved (ALLOW_NEUTRAL_RECOMMENDATIONS is on)" in text and "sized with the approved ones, no profile limit" in text
    assert "MU" in text and "BUY (treated as approved, Auto): 38 shares" in text and "AMD" in text and "BUY (treated as approved, Auto): 38 shares at $534.06" in text      # AMENDED (entry 109): MU 38 and AMD 38 round-robin (32 and 46 in equal dollars)
    assert neutral.neutral_lines(advised()) == []


def test_the_real_session_went_advisor_critic_decision_trader_and_booked_both_fills():
    order = wo.WorkOrder.model_validate(CAPTURE["order"])
    assert CAPTURE["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"] and order.status == "traded"
    assert [(d["symbol"], d["will_trade"], d.get("origin")) for d in order.trade_decisions] == [("SNDK", "no_system_denied", None), ("MU", "yes_auto", "neutral"), ("AMD", "yes_auto", "neutral")]
    assert [(t["symbol"], t["status"], t["placed_by"], t["booked"]) for t in order.trades] == [("MU", "filled", "model", True), ("AMD", "filled", "model", True)] and wo.verify_trades(order) == []
    assert money_door(order.trader_runs[0]["portfolio_before"]["cash"]) == Decimal("50000") and money_door(order.trader_runs[0]["portfolio_after"]["cash"]) == Decimal("502.59")
    for d in order.trade_decisions[1:]:
        assert wo.verify_trade(order, d["record_id"], MOMENT) == []


# ---------------------------------------------------------------- the Account Manager does no risk management

def test_the_neutral_module_and_the_account_managers_path_to_it_name_no_limit_and_no_sizing_check_but_the_funds():
    forbidden = {"max_position_pct", "max_loss_per_trade_pct", "max_new_exposure_pct", "max_order_dollars", "SizingLimits", "sizing_limits", "limit_overrides", "stop_distance", "held_shares"}

    def used(source: str) -> set[str]:
        tree = ast.parse(source)
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    import textwrap
    assert used(inspect.getsource(neutral)) & forbidden == set()
    assert used(textwrap.dedent(inspect.getsource(AccountManager.decide_trades))) & forbidden == set()      # AMENDED (entry 107): `_decide_neutral` and its own sizing are gone; the path is `AccountManager.decide_trades` into `decisions.decide_trades`
    from prometheus.agents.account_manager import decisions
    assert used(inspect.getsource(decisions)) & forbidden == set()
    assert not forbidden & set(UserProfile.model_fields)


# ---------------------------------------------------------------- Part E: a neutral recommendation is an approval, in full (entry 107)

OIL_CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "auto_oil_no_second_mark.json").read_text(encoding="utf-8"))
OIL_CASH = Portfolio(user="Auto Mode Check", cash_avail=Decimal("15000"))
OIL_USER = UserProfile(user="Auto Mode Check", risk_tolerance="conservative", age=60, trade_mode="auto")
OIL_LIVE = {d["symbol"]: d for d in OIL_CAPTURE["order"]["trade_decisions"]}


def oil_order() -> wo.WorkOrder:
    """The real oil session (`auto_oil_no_second_mark.json`) as it was when the Critic had finished: EOG 79 and CVX 86 approved, COP denied, six companies the Advisor scored neutral."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(OIL_CAPTURE["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def test_the_real_oil_session_decided_again_is_the_three_companies_asked_for_each_about_a_third_and_the_rest_are_neutral_and_not_needed(allowed):
    """The REAL session whose request was "the 3 most recommended oil companies ... evenly among the 3" (Auto, $15,000): it decided EIGHT purchases (EOG 47 and CVX 34 over all the money, then six neutral companies over the same
    money again) and the Trader gave CNQ one share and rejected five for cash. Decided again: EOG, CVX and CNQ (the best neutral score, 66; OXY ties at 66 and comes after it) in ONE sizing, three equal amounts of about $5,000;
    COP is denied and does not count; OXY, BP, META, VLO and FANG have NO decision and the Account Manager says they were neutral and not needed."""
    told = []
    out = account(profile=OIL_USER, screen=told.append).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert [d["symbol"] for d in out.trade_decisions] == ["EOG", "CVX", "COP", "CNQ"] and out.status == "decided" and out.errors == []
    assert [(s, got[s]["will_trade"], got[s].get("origin")) for s in ("EOG", "CVX", "COP", "CNQ")] == [("EOG", "yes_auto", None), ("CVX", "yes_auto", None), ("COP", "no_system_denied", None), ("CNQ", "yes_auto", "neutral")]
    assert [got[s]["size"]["allocation"] for s in ("EOG", "CVX", "CNQ")] == ["even"] * 3 and {got[s]["balance"]["target_dollars"] for s in ("EOG", "CVX", "CNQ")} == {Decimal("5000")} and {got[s]["balance"]["of"] for s in ("EOG", "CVX", "CNQ")} == {3}
    for symbol in ("EOG", "CVX", "CNQ"):      # AMENDED (entry 108): the equal $5,000 buys EOG 31, CVX 22 and CNQ 100 whole shares ($14,720.50; EOG $4,907.92, CVX $4,837.58, CNQ $4,975.00) and the $279.50 left over is spent one share at a time to the smallest amount: CVX (the smallest, its next share $219.89 fits) and then CNQ ($49.75), $9.86 left
        d = got[symbol]
        assert d["shares"] >= int(Decimal("5000.0") // (d["est_cost"] / d["shares"])) and 4500 < d["est_cost"] < 5100
    assert [(got[s]["shares"], got[s]["est_cost"]) for s in ("EOG", "CVX", "CNQ")] == [(31, Decimal("4907.92")), (23, Decimal("5057.47")), (101, Decimal("5024.75"))] and OIL_CASH.cash_avail - sum(got[s]["est_cost"] for s in ("EOG", "CVX", "CNQ")) == Decimal("9.86")
    said = "\n".join(told)
    for symbol in ("OXY", "BP", "META", "VLO", "FANG"):
        assert symbol not in got and f"{symbol} was scored neutral by the Advisor" in said and "not needed: 3 companies were asked for" in said
    assert "CNQ was scored" not in said and not [d for d in out.trade_decisions if d["will_trade"] == "no_funds"]
    assert wo.verify_trade(out, got["CNQ"]["record_id"], MOMENT) == [] and [wo.verify_trade(out, got[s]["record_id"], MOMENT) for s in ("EOG", "CVX")] == [[], []]
    assert {s: OIL_LIVE[s]["will_trade"] for s in ("OXY", "BP", "META", "VLO", "FANG")} == {s: "yes_auto" for s in ("OXY", "BP", "META", "VLO", "FANG")}      # what the live run did: all six decided


def test_the_three_purchases_are_traded_and_none_is_rejected_for_cash(allowed):
    order = account(profile=OIL_USER).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    account(profile=OIL_USER).perform_trades(order, OIL_CASH, trader(), MOMENT)
    assert [(t["symbol"], t["status"]) for t in order.trades] == [("EOG", "filled"), ("CVX", "filled"), ("CNQ", "filled")] and wo.verify_trades(order) == [] and not [t for t in order.trades if t["status"] == "rejected"]


def test_with_the_setting_off_only_the_two_approved_companies_are_bought_and_nothing_neutral_is_decided(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "false")
    told = []
    out = account(profile=OIL_USER, screen=told.append).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", "even", None, count=3)
    assert [(d["symbol"], d["will_trade"]) for d in out.trade_decisions] == [("EOG", "yes_auto"), ("CVX", "yes_auto"), ("COP", "no_system_denied")] and told == []
    assert {d["balance"]["target_dollars"] for d in out.trade_decisions if d["will_trade"] == "yes_auto"} == {Decimal("7500")}


def test_a_request_with_no_number_cuts_nothing_every_neutral_company_is_decided_with_the_approved_ones_in_one_round_robin(allowed):
    told = []
    out = account(profile=OIL_USER, screen=told.append).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", None, None)
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert [d["symbol"] for d in out.trade_decisions] == ["EOG", "CVX", "COP", "CNQ", "OXY", "BP", "META", "VLO", "FANG"] and told == []
    assert {got[s]["size"]["allocation"] for s in got if got[s]["size"]} == {"round_robin"} and not [d for d in out.trade_decisions if d["will_trade"] == "no_funds"]      # ONE round robin over the eight: nobody is cut for cash
    assert sum(d["est_cost"] for d in out.trade_decisions) <= Decimal("15000") and all(got[s]["shares"] >= 1 for s in ("EOG", "CVX", "CNQ", "OXY", "BP", "META", "VLO", "FANG"))


def test_the_number_is_the_form_s_own_count_a_denied_recommendation_does_not_count_and_the_approved_come_first(allowed):
    told = []
    two = account(profile=OIL_USER, screen=told.append).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", None, None, count=2)
    assert [d["symbol"] for d in two.trade_decisions] == ["EOG", "CVX", "COP"] and len(told) == 6 and all("not needed: 2 companies were asked for" in line for line in told)      # the two approved fill it: all six neutral ones are not needed
    one = account(profile=OIL_USER).decide_trades(oil_order(), OIL_CASH, None, MOMENT, "auto", None, None, count=5)
    assert [d["symbol"] for d in one.trade_decisions] == ["EOG", "CVX", "COP", "CNQ", "OXY", "BP"]      # five companies: the two approved, then the three best neutral scores (66, 66, 64)


def test_the_desk_passes_the_number_the_request_gave_and_never_the_default_of_three():
    """The plan has `find` (3 when the request gave no number: for the Advisor) and `count` (the form's own, None then). Only `count` cuts. The REAL desk reads two real forms (the model's reply is the form of the capture's
    own intent) and stops at the Data Clerk's node."""
    from prometheus.langgraph import desk
    from test_direct_order import Replay as FormReplay

    form = {k: v for k, v in OIL_CAPTURE["intent"].items() if k != "request"}
    for count, expected in ((3, 3), (None, None)):
        reply = json.dumps({**form, "count": count, "allocation": "default"})
        typed = OIL_CAPTURE["request"] if count else "I want to buy oil stocks. Buy the most amount of oil stocks shares from the most recommended companies."
        if count is None:
            reply = json.dumps({**form, "count": None, "allocation": "default", "budget": None})
        manager = lambda reply=reply: AccountManager(model=FormReplay([reply]), model_id="recorded", profile=OIL_USER)

        def no_clerk(order):
            raise RuntimeError("the Data Clerk is not part of this test")

        state = desk.run_desk({"ask": {"text": typed, "default_count": 3}, "account": {"portfolio": OIL_CASH, "profile": OIL_USER}}, clerk_factory=no_clerk, advisor_factory=no_clerk, funds_factory=manager, with_critic=True)
        assert state["plan"]["count"] == expected and state["plan"]["find"] == 3
