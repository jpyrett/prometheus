"""A NEUTRAL RECOMMENDATION IS APPROVED (design choices entry 91; Jacob, 2026-11-15, with `ALLOW_NEUTRAL_RECOMMENDATIONS=true` in `.env`), and no profile limit applies to anything.

REAL DATA ONLY. `tests/fixtures/session_real/pam_smith_neutral.json` is a capture of the real session `./test_prometheus.sh --session --account <Pam Smith> --yes --prompt "Buy MU, SNDK and AMD if the recommendations pass. Go ahead
and trade if the recommendations pass."` (2026-11-15; Pam Smith, 30, aggressive, $50,000, Auto mode; the paid refresh of the data; the real Account Manager, Data Clerk, Advisor, Critic (GLM 5.3) and Trader (gpt-oss-120b), the
real Postgres book, the simulated clock of Friday 2026-11-13): SNDK was bullish and the Critic DENIED it (risk above the profile, a falling knife: final), MU and AMD were scored NEUTRAL by the Advisor (no strategy passed), so
they were treated as approved and bought in equal dollars: MU 32 shares, AMD 46, $502.59 left. The tests re-decide that order with the real Account Manager over the real stored closes (the Data Clerk's quotes) and trade it with
the real Trader, and break the permission one field at a time. No agent and no model is replaced by a stand-in. The tests are skipped when the stored bars are not on the machine.
"""
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
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position
from prometheus.common.user_profile import UserProfile

pytestmark = pytest.mark.usefixtures("friday_close")

CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "session_real" / "pam_smith_neutral.json").read_text(encoding="utf-8"))
PAM = Portfolio(user="Pam Smith", cash_avail=50000.0)
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


def test_with_the_setting_on_only_the_companies_the_user_named_that_the_advisor_scored_neutral_count(allowed):
    names = neutral.neutral_candidates(advised())
    assert [(c["symbol"], c["stance"], c["origin"]) for c in names] == [("MU", "neutral", "subject"), ("AMD", "neutral", "subject")]
    order = advised()
    order.candidates[0]["gate"] = "freshness"                                                    # a gate that failed is not a stance: it stops the company
    order.candidates[1]["held"] = True                                                           # one already held is not bought again
    assert neutral.neutral_candidates(order) == []
    order = advised()
    order.candidates[0]["origin"] = "discover"                                                   # a company the Advisor found, that the user never named, is never bought as neutral
    order.candidates[1]["stance"] = "bearish"                                                    # a bearish stance trades nothing
    assert neutral.neutral_candidates(order) == []
    assert neutral.neutral_candidates(wo.WorkOrder.model_validate({**CAPTURE["order"], "status": "new"})) == []     # an order the Advisor has not answered has none


# ---------------------------------------------------------------- the Account Manager approves them, and the Critic's denial stays final

def test_the_real_session_is_reproduced_neutral_companies_are_approved_in_equal_dollars_and_the_critics_denial_is_final(allowed):
    live = {d["symbol"]: d for d in CAPTURE["order"]["trade_decisions"]}
    out = decided(advised())
    got = {d["symbol"]: d for d in out.trade_decisions}
    assert out.status == "decided" and out.errors == [] and [d["symbol"] for d in out.trade_decisions] == ["SNDK", "MU", "AMD"]
    assert got["SNDK"]["will_trade"] == "no_system_denied" and got["SNDK"]["shares"] == 0                                           # bullish, and the Critic denied it: a denial is final, the setting does not touch it
    for symbol in ("MU", "AMD"):
        d = got[symbol]
        assert (d["will_trade"], d["origin"], d["action"], d["mode"], d["review_id"]) == ("yes_auto", "neutral", "open", "auto", None) and d["price"] == close_of(symbol)
        assert (d["shares"], d["est_cost"], d["price"]) == (live[symbol]["shares"], live[symbol]["est_cost"], live[symbol]["price"])        # what the live run decided
        assert d["balance"] == {"mode": "even", "target_dollars": 25000.0, "of": 2} and "ALLOW_NEUTRAL_RECOMMENDATIONS is on" in d["reason"] and "no profile limit" in d["reason"]
        assert d["shares"] == int(25000.0 // d["price"]) and 25000.0 - d["est_cost"] < d["price"]                                     # equal dollars, whole shares
    assert got["MU"]["est_cost"] + got["AMD"]["est_cost"] <= PAM.cash_avail and PAM.cash_avail - got["MU"]["est_cost"] - got["AMD"]["est_cost"] < 800     # the funds cover the orders: the only check


def test_with_nothing_for_the_critic_to_review_only_the_neutral_companies_are_decided(allowed):
    out = decided(neutral_only())
    assert out.status == "decided" and out.reviews == [] and [(d["symbol"], d["will_trade"], d["origin"]) for d in out.trade_decisions] == [("MU", "yes_auto", "neutral"), ("AMD", "yes_auto", "neutral")]


def test_one_neutral_company_gets_all_the_money_and_a_budget_of_the_user_is_the_pot(allowed):
    order = neutral_only()
    order.candidates = [c for c in order.candidates if c["symbol"] == "AMD"]
    (d,) = decided(order).trade_decisions
    assert d["shares"] == int(50000.0 // close_of("AMD")) and d["balance"]["of"] == 1                                                  # nothing limits one stock to a share of the account
    capped = decided(neutral_only(), budget=10000.0)
    assert [(x["symbol"], x["shares"]) for x in capped.trade_decisions] == [("MU", int(5000.0 // close_of("MU"))), ("AMD", int(5000.0 // close_of("AMD")))]


def test_a_neutral_company_the_cash_cannot_buy_a_share_of_is_no_funds_and_one_with_no_price_is_denied(allowed):
    poor = decided(neutral_only(), portfolio=Portfolio(user="Pam Smith", cash_avail=600.0))                                            # $300 each buys no share of MU ($778) and one of AMD ($534)? no: not even one
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in poor.trade_decisions] == [("MU", "no_funds", 0), ("AMD", "no_funds", 0)]
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
    assert (got["MU"]["shares"], got["AMD"]["shares"]) == (int(2500.0 // close_of("MU")), int(2500.0 // close_of("AMD"))) == (3, 4)
    assert [(d["will_trade"], d["size"]["binding"], d["size"]["asked"]) for d in out.trade_decisions] == [("yes_user", "asked", 3), ("yes_user", "asked", 4)] and "the 3 shares the user asked for" in got["MU"]["reason"]
    assert [(c["symbol"], c["shares"]) for c in cards] == [("MU", 3), ("AMD", 4)] and out.errors == []                          # the user is then shown exactly these, and says yes to each


def test_a_number_of_shares_the_user_gives_is_the_most_of_each_neutral_company_that_is_bought(allowed):
    out, channel, cards, _ = ask_how_much("shares_each_raise_yes")                                                                 # "10 shares of each"
    assert [(d["symbol"], d["shares"], d["size"]["asked"]) for d in out.trade_decisions] == [("MU", 10, 10), ("AMD", 10, 10)]
    assert sum(d["est_cost"] for d in out.trade_decisions) < 50000.0 and [c["shares"] for c in cards] == [10, 10]


def test_one_set_of_questions_covers_what_the_critic_approved_and_the_neutral_companies(allowed):
    order = advised()                                                                                                              # SNDK denied by the Critic, MU and AMD neutral
    recorded = RECORDED["percent_each_no_raise_needed"]
    model, channel = Replay(recorded["replies"]), channel_of(recorded["answers"])
    out = account(model=model).decide_trades(order, PAM, lambda card: True, MOMENT, "ask", clarify=channel, request_text=REQUEST)
    assert len(channel.cards) == 1 and [(d["symbol"], d["will_trade"], d["shares"]) for d in out.trade_decisions] == [("SNDK", "no_system_denied", 0), ("MU", "yes_user", 3), ("AMD", "yes_user", 4)]


def test_nothing_is_asked_in_auto_mode_or_when_the_request_already_said_how_much_or_with_nobody_to_ask(allowed):
    out, channel, _, model = ask_how_much("percent_each_no_raise_needed", mode="auto")
    assert channel.cards == [] and model.seen == [] and [d["will_trade"] for d in out.trade_decisions] == ["yes_auto", "yes_auto"] and out.trade_decisions[0]["shares"] > 20          # Auto never asks: equal dollars of the cash
    out, channel, _, model = ask_how_much("percent_each_no_raise_needed", said_how_much=True)
    assert channel.cards == [] and model.seen == [] and [d["shares"] for d in out.trade_decisions] == [int(25000.0 // close_of("MU")), int(25000.0 // close_of("AMD"))]
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
    forged.trade_decisions[0].update(origin="neutral", will_trade="yes_auto", shares=5, price=100.0, action="open", valid_until_utc="2026-11-13T21:00:00Z")
    reasons = "; ".join(wo.verify_trade(forged, sndk["record_id"], MOMENT))
    assert "holds no neutral score of the Advisor's for 'SNDK'" in reasons and "a neutral recommendation has no Critic review" in reasons                  # a bullish company with a review is not a neutral candidate


# ---------------------------------------------------------------- the Trader trades it, one trade at a time, through the same handshake

def test_the_trader_buys_the_neutral_companies_through_the_handshake_and_the_portfolio_shows_them(allowed):
    order = decided(advised())
    am = account()
    am.perform_trades(order, PAM, trader(), MOMENT)
    live = {t["symbol"]: t for t in CAPTURE["order"]["trades"]}
    trades = {t["symbol"]: t for t in order.trades}
    assert set(trades) == {"MU", "AMD"} and order.status == "traded" and order.errors == [] and wo.verify_trades(order) == []
    for symbol, t in trades.items():
        assert (t["status"], t["side"], t["review_id"], t["shares"], t["kind"]) == ("filled", "buy", None, live[symbol]["shares"], "recommendation") and t["order_type"] == "market"
        assert abs(t["fill"]["price"] / close_of(symbol) - 1) < 0.002
    after = order.trader_runs[-1]["portfolio_after"]
    assert after["cash"] == round(50000.0 - sum(t["fill"]["amount"] for t in trades.values()), 2) < 1000 and {p["symbol"]: p["shares"] for p in after["positions"]} == {"MU": 32, "AMD": 46}


def test_a_neutral_purchase_is_cut_to_the_cash_like_any_recommended_one_never_refused_for_a_limit(allowed):
    order = decided(advised())
    am = account()
    am.perform_trades(order, Portfolio(user="Pam Smith", cash_avail=26000.0), trader(), MOMENT)
    first, second = order.trades
    assert first["status"] == "filled" and first["fill"]["shares"] == first["shares"] and second["status"] in ("filled", "rejected") and wo.verify_trades(order) == []
    assert (second["fill"] or {}).get("shares", 0) < second["shares_approved"] or second["status"] == "rejected"                          # the cash that is left decides it (the handshake), nothing else


# ---------------------------------------------------------------- what the user reads, and the real session's own record

def test_the_report_says_which_companies_were_treated_as_approved_and_why(allowed):
    text = "\n".join(neutral.neutral_lines(decided(advised())))
    assert "NEUTRAL RECOMMENDATIONS  treated as approved (ALLOW_NEUTRAL_RECOMMENDATIONS is on)" in text and "equal dollars, no profile limit" in text
    assert "MU" in text and "BUY (treated as approved, Auto): 32 shares" in text and "AMD" in text and "46 shares" in text
    assert neutral.neutral_lines(advised()) == []


def test_the_real_session_went_advisor_critic_decision_trader_and_booked_both_fills():
    order = wo.WorkOrder.model_validate(CAPTURE["order"])
    assert CAPTURE["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"] and order.status == "traded"
    assert [(d["symbol"], d["will_trade"], d.get("origin")) for d in order.trade_decisions] == [("SNDK", "no_system_denied", None), ("MU", "yes_auto", "neutral"), ("AMD", "yes_auto", "neutral")]
    assert [(t["symbol"], t["status"], t["placed_by"], t["booked"]) for t in order.trades] == [("MU", "filled", "model", True), ("AMD", "filled", "model", True)] and wo.verify_trades(order) == []
    assert order.trader_runs[0]["portfolio_before"]["cash"] == 50000.0 and order.trader_runs[0]["portfolio_after"]["cash"] == 502.59
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
    assert used(textwrap.dedent(inspect.getsource(AccountManager._decide_neutral))) & forbidden == set()
    assert not forbidden & set(UserProfile.model_fields)
