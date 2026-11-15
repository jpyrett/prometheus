"""A real user session, replayed offline (design choices entry 78): the Critic's REAL reviews of a real run through the Account Manager's decision step and the Trader's output guard.

`tests/fixtures/session_real/bob_jones_oil.json` is a capture of `./test_prometheus.sh --session --name "Bob Jones" --age 60 --risk conservative --cash 15000 --mode auto --prompt "I want to buy oil stocks. Buy the
most amount of oil stocks shares from the 3 most recommended companies. Balance the purchased stocks evenly among the 3 recommended companies."` on 2026-11-14 (`--save`): the real Account Manager, Data Clerk, Advisor,
Critic (GLM 5.3) and Trader (gpt-oss-120b) on real stored data and real Finnhub numbers, with the market close of Friday 2026-11-13 as the simulated clock. It holds the request, what the Account Manager understood,
the work order without its data (the Advisor's recommendations, the Critic's reviews, the Account Manager's decisions, the Trader's trades and run) and the nodes of the graph that ran.

The tests re-run the DECISION step over the real reviews and must reproduce what the live run decided; nothing is invented and no agent is replaced by a stand-in. The prices, the scores and the fills are the
market's and the models' of that run. What the real run showed: XOM and EOG were approved and APA was denied for `risk_above_profile` (Bob is conservative); the Critic of that run also recorded a size
(XOM 8 shares, EOG 9), and the even split gave XOM and EOG the same dollars, so EOG was cut from 9 shares to 8.

THE FIXTURE WAS MIGRATED (design choices entry 80). It is the real record of a real run under the OLD Critic, which sized: since then the Critic only approves or denies and the Account Manager decides how much, so the
fields the Critic no longer has were stripped one time (`approved_shares`, `suggested_shares` and the equity, equity basis, caps and `allowed_shares` of each review's `risk`, the `size_reduced` warnings, the
`size` of each recommendation's `critic` block, and the size wording of an approval's summary sentence). Every score, price, dimension, decision and fill is as it was recorded. Its `risk` scores still include the
old position measure and its `accuracy.integrity.checked` still names the old size checks: they ran in that run. The shares of 8 and 8 that the Account Manager decides now are the same as the live run's, worked out
again by the Account Manager's own sizing from the record's own numbers (the deal price, the plan's stop distance, the Advisor's average daily dollars and the account's money).

IT WAS MIGRATED A SECOND TIME (design choices entry 80, step 3: the Advisor stops sizing). Its three recommendations were written under the OLD Advisor, whose trade plan carried a size (XOM 17 shares, APA 61,
EOG 18, each cut by a 20% position cap and a 1% risk budget of the $15,000), so the fields the Advisor's record no longer has were stripped one time from each record: `trade.equity`, `equity_basis`,
`risk_budget`, `suggested_shares`, `position_pct`, `oversized`, `size_note`, `funds_feasible` and `funds_note`, and `portfolio_fit.position_pct_after` and `funds_feasible`. NOTHING else was touched: the
file is the same, byte for byte, apart from those 33 fields (the Account Manager's decisions keep their own `funds_note`: it is not the Advisor's). Every score, price, entry zone, stop, target, dimension,
decision and fill is as it was recorded, and each stripped record validates against the Advisor's record as it is now (`test_the_real_recommendations_are_valid_records_with_prices_and_dates_and_no_size`).
"""
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.common import trading_clock as tc
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position, apply_fill, trade_amount
from prometheus.common.user_profile import UserProfile

CAPTURE = Path(__file__).parent / "fixtures" / "session_real" / "bob_jones_oil.json"
NODES = ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "trader"]
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the simulated moment of the run


@pytest.fixture(scope="module")
def capture():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def reviewed(capture) -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished: the real recommendations and reviews, before the Account Manager decided and before the Trader traded."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


BOB = Portfolio(user="Bob Jones", cash_avail=15000.0)
PROFILE = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto", auto_min_score=70)     # the account's own row, and the mark of the run


def decide(capture, balance="even", budget=None, now=MOMENT, order=None, mode="auto"):
    return decisions.decide_trades(order or reviewed(capture), BOB, PROFILE, 80, None, now, mode, balance, budget)


def kept(decision):
    return {k: decision[k] for k in ("record_id", "symbol", "action", "shares", "est_cost", "mode", "will_trade", "funds_avail", "balance") if k in decision}


# ---------------------------------------------------------------- what the real run was

def test_the_capture_is_one_real_session_through_every_node_of_the_graph(capture):
    assert capture["path"] == NODES and capture["request"].startswith("I want to buy oil stocks.")
    intent = TradeIntent(request=capture["request"], **{k: v for k, v in capture["intent"].items() if k != "request"})
    assert (intent.action, intent.topic, intent.count, intent.allocation, intent.budget) == ("buy", "oil", 3, "even", None)
    order = wo.WorkOrder.model_validate(capture["order"])
    assert order.status == "traded" and order.schema_version == 4 and order.data is None
    assert [r["identity"]["record_id"] for r in order.recommendations] == [r["record_id"] for r in order.reviews] == [d["record_id"] for d in order.trade_decisions]
    run = order.trader_runs[0]
    assert run["model"] == "openai/gpt-oss-120b" and run["temperature"] == 0.0 and run["mode"] == "paper" and run["booked"] is True and run["placed_by_model"] == len(order.trades) and run["errors"] == []


def test_the_real_reviews_are_the_critics_own_arithmetic_a_denial_for_risk_and_two_approvals(capture):
    reviews = {r["symbol"]: r for r in capture["order"]["reviews"]}
    approved = [s for s, r in reviews.items() if r["approved"]]
    assert len(approved) == 2 and sum(1 for r in reviews.values() if not r["approved"]) == 1
    denied = next(r for r in reviews.values() if not r["approved"])
    assert [b["code"] for b in denied["blockers"]] == ["risk_above_profile"] and denied["score"] <= 49
    for symbol in approved:
        r = reviews[symbol]
        assert r["score"] >= 70 and r["valid_until_utc"] == "2026-11-13T21:00:00Z" and r["reviewed_utc"] == "2026-11-13T20:30:00Z"   # an approval lasts to the 16:00 close
    assert not any({"approved_shares", "suggested_shares"} & set(r) or {"caps", "allowed_shares", "equity"} & set(r["risk"]) for r in reviews.values())      # the migrated record carries no size from the Critic


# ---------------------------------------------------------------- the decision step over the real reviews reproduces the real decisions

def test_the_decisions_the_live_run_made_are_reproduced_from_the_real_reviews(capture):
    live = [kept(d) for d in capture["order"]["trade_decisions"]]
    again = decide(capture)
    assert again.status == "decided" and again.errors == [] and [kept(d) for d in again.trade_decisions] == live
    assert [d["will_trade"] for d in again.trade_decisions] == ["yes_auto", "no_system_denied", "yes_auto"] or {d["will_trade"] for d in again.trade_decisions} <= {"yes_auto", "no_system_denied"}


def test_the_even_split_gave_the_two_approved_companies_the_same_dollars_and_cut_the_dearer_one(capture):
    reviews = {r["record_id"]: r for r in capture["order"]["reviews"]}
    decided = [d for d in decide(capture).trade_decisions if d["will_trade"] == "yes_auto"]
    prices = {d["record_id"]: reviews[d["record_id"]]["timing"]["deal_price"] for d in decided}
    own = {d["record_id"]: math.floor(0.10 * 15000.0 / prices[d["record_id"]]) for d in decided}                           # the profile's 10% position limit, in whole shares: what each could be on its own
    target = min(own[r] * prices[r] for r in own)
    assert len(decided) == 2 and {d["balance"]["target_dollars"] for d in decided} == {round(target, 2)} and all(d["balance"]["of"] == 2 for d in decided)
    for d in decided:
        price = prices[d["record_id"]]
        assert d["shares"] == math.floor(target / price + 1e-9) <= own[d["record_id"]] and target - d["est_cost"] < price
    assert [own[d["record_id"]] for d in decided] == [8, 9] and [d["shares"] for d in decided] == [8, 8]                     # the real run cut EOG from 9 shares to 8


def test_the_denied_company_stays_denied_and_is_not_part_of_the_split(capture):
    out = decide(capture).trade_decisions
    denied = next(d for d in out if d["will_trade"] == "no_system_denied")
    assert denied["shares"] == 0 and "balance" not in denied and denied["symbol"] == next(r["symbol"] for r in capture["order"]["reviews"] if not r["approved"])


def test_in_auto_mode_with_no_split_asked_the_funds_are_spent_round_robin_and_each_purchase_is_what_its_own_limits_allow(capture):
    plain = decide(capture, balance=None).trade_decisions
    bought = [d for d in plain if d["will_trade"] == "yes_auto"]
    assert [(d["symbol"], d["shares"], d["size"]["binding"], d["size"]["allocation"]) for d in bought] == [("XOM", 8, "position", "round_robin"), ("EOG", 9, "position", "round_robin")] and not any("balance" in d for d in plain)
    assert all("spent round-robin" in d["reason"] for d in bought)                                                           # both reached their own position limit before the $15,000 ran out


def test_in_auto_mode_a_pot_smaller_than_the_limits_is_spent_a_share_at_a_time_where_the_default_gives_it_all_to_the_first(capture):
    """A budget of $400 against XOM (170.59, rank 1) and EOG (158.32, rank 2), the real deal prices. Round robin: XOM 170.59, EOG 328.91, then 499.50 is over $400: one share of each, closed by the budget. Ask mode's
    default would take XOM's two shares (341.18) and leave $58.82, under the price of EOG: none."""
    robin = [d for d in decide(capture, balance=None, budget=400.0).trade_decisions if d["will_trade"] == "yes_auto"]
    assert [(d["symbol"], d["shares"], d["est_cost"], d["size"]["binding"]) for d in robin] == [("XOM", 1, 170.59, "budget"), ("EOG", 1, 158.32, "budget")]
    greedy = decide(capture, balance=None, budget=400.0, mode="ask").trade_decisions
    assert [(d["symbol"], d["shares"], d["will_trade"]) for d in greedy if d["symbol"] != "APA"] == [("XOM", 2, "pending_user"), ("EOG", 0, "no_funds")]
    assert greedy[2]["size"]["binding"] == "no_room" and {d["size"]["allocation"] for d in greedy if d["size"]} == {"as_much_as_possible"}


def test_an_even_split_asked_for_in_auto_mode_still_gives_equal_dollars_not_round_robin(capture):
    out = [d for d in decide(capture, balance="even", budget=400.0).trade_decisions if d["will_trade"] == "yes_auto"]
    assert {d["size"]["allocation"] for d in out} == {"even"} and {d["balance"]["target_dollars"] for d in out} == {200.0} and [d["shares"] for d in out] == [1, 1]      # $200 each: one share of XOM, one of EOG


def test_round_robin_can_be_asked_for_in_the_users_words_in_ask_mode_too(capture):
    out = decide(capture, balance="round_robin", budget=400.0, mode="ask").trade_decisions
    assert [(d["symbol"], d["shares"], d["will_trade"], d["size"]["allocation"]) for d in out if d["symbol"] != "APA"] == [("XOM", 1, "pending_user", "round_robin"), ("EOG", 1, "pending_user", "round_robin")]


def test_a_budget_in_the_users_words_caps_the_split_by_the_real_prices(capture):
    reviews = {r["record_id"]: r for r in capture["order"]["reviews"]}
    out = [d for d in decide(capture, budget=2000.0).trade_decisions if d["will_trade"] == "yes_auto"]
    assert {d["balance"]["target_dollars"] for d in out} == {1000.0}                                   # $2,000 over the two purchases that trade
    assert [d["shares"] for d in out] == [math.floor(1000.0 / reviews[d["record_id"]]["timing"]["deal_price"] + 1e-9) for d in out] and sum(d["est_cost"] for d in out) <= 2000.0


def test_the_account_manager_passes_the_split_on(capture):
    manager = AccountManager(model=object(), model_id="account-book", profile=PROFILE)
    out = manager.decide_trades(reviewed(capture), BOB, None, MOMENT, "auto", "even", None)
    assert [kept(d) for d in out.trade_decisions] == [kept(d) for d in capture["order"]["trade_decisions"]]


def test_a_review_that_needs_a_person_goes_to_the_user_and_is_part_of_the_split_so_the_user_is_shown_the_size_that_will_be_placed(capture):
    order = reviewed(capture)
    first = next(i for i, r in enumerate(order.reviews) if r["approved"])
    order.reviews[first] = {**order.reviews[first], "requires_human": True, "requires_human_reasons": ["evidence_not_reviewed"]}
    out = decide(capture, order=order)
    pending = out.trade_decisions[first]
    assert pending["will_trade"] == "pending_user" and out.status == "reviewed" and pending["balance"]["of"] == 2        # nobody to ask: it waits, at the size it would be placed
    assert [d["balance"]["of"] for d in out.trade_decisions if d["will_trade"] == "yes_auto"] == [2]


# ---------------------------------------------------------------- Ask mode: the user approves the order that will be placed (design choices entry 79)

ASK = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="ask", auto_min_score=70)


def ask_session(capture, answers, balance="even", budget=None, profile=ASK, mode="ask", order=None):
    """`decide_trades` over the real reviews with a user who gives `answers` in turn; returns the order and the cards the user was shown."""
    cards = []

    def user(card):
        cards.append(card)
        return answers[len(cards) - 1]

    return decisions.decide_trades(order or reviewed(capture), BOB, profile, 80, user, MOMENT, mode, balance, budget), cards


def live_sizes(capture):
    """The shares the real run placed for each symbol (its decisions after the even split)."""
    return {d["symbol"]: d["shares"] for d in capture["order"]["trade_decisions"] if d["will_trade"] == "yes_auto"}


def test_in_ask_mode_the_user_is_shown_the_size_that_will_be_placed_and_that_is_what_is_decided(capture):
    out, cards = ask_session(capture, [True, True])
    final = live_sizes(capture)
    assert [c["symbol"] for c in cards] == ["XOM", "EOG"] and [c["shares"] for c in cards] == [final["XOM"], final["EOG"]]     # the split's size (EOG 8), not the 9 its own limits alone would allow
    assert [(c["n"], c["of"]) for c in cards] == [(1, 2), (2, 2)] and all(c["cash"] == 15000.0 for c in cards)
    assert [(p["symbol"], p["shares"]) for p in cards[0]["proposal"]] == [("XOM", final["XOM"]), ("EOG", final["EOG"])] and cards[0]["proposal"] == cards[1]["proposal"]
    assert f"WOULD BE PLACED  buy {final['EOG']} shares of EOG" in cards[1]["text"] and "balanced evenly" in cards[1]["text"]
    placed = {d["symbol"]: d for d in out.trade_decisions if d["will_trade"] == "yes_user"}
    assert {s: d["shares"] for s, d in placed.items()} == final and all(d["asked_user"] and d["mode"] == "ask" for d in placed.values())
    assert [c["est_cost"] for c in cards] == [placed["XOM"]["est_cost"], placed["EOG"]["est_cost"]] and out.status == "decided"
    assert next(d for d in out.trade_decisions if d["symbol"] == "APA")["will_trade"] == "no_system_denied"                  # a denial is final: the user is not asked to override it


def test_a_no_leaves_the_other_purchase_at_the_size_the_user_saw_nothing_is_split_again(capture):
    out, cards = ask_session(capture, [False, True])                                                                        # no to XOM, yes to EOG
    by = {d["symbol"]: d for d in out.trade_decisions}
    assert by["XOM"]["will_trade"] == "no_user_denied" and by["EOG"]["will_trade"] == "yes_user"
    assert by["EOG"]["shares"] == cards[1]["shares"] == live_sizes(capture)["EOG"]                                          # EOG alone could have been 9: it stays what the user was shown
    assert by["EOG"]["est_cost"] == cards[1]["est_cost"] and out.status == "decided"


def test_with_nobody_to_ask_the_purchases_wait_at_their_final_size_and_nothing_trades(capture):
    out = decisions.decide_trades(reviewed(capture), BOB, ASK, 80, None, MOMENT, "ask", "even", None)
    assert [(d["symbol"], d["will_trade"], d["shares"], d["asked_user"]) for d in out.trade_decisions if d["will_trade"] != "no_system_denied"] == [("XOM", "pending_user", 8, False), ("EOG", "pending_user", 8, False)]
    assert out.status == "reviewed" and not any(d["will_trade"] in wo.TRADING for d in out.trade_decisions)
    out, cards = ask_session(capture, [None, None])                                                                         # the approver gave no answer (EOF): the same
    assert [d["will_trade"] for d in out.trade_decisions] == ["pending_user", "no_system_denied", "pending_user"] and all(d["asked_user"] for d in out.trade_decisions if d["will_trade"] == "pending_user")


def test_no_cash_is_reserved_each_purchase_may_use_all_the_funds_in_the_portfolio(capture):
    """$2,000 in cash and $5,000 of a holding (100 shares of GM at a cost of $50): an equity of $7,000, and a user whose own limits are wide. XOM takes 11 shares ($1,876.49) and EOG 12 ($1,899.84): $3,776.33 in all, each
    one inside the $2,000, none of it reserved."""
    poorer = Portfolio(user="Bob Jones", cash_avail=2000.0, positions=[Position(symbol="GM", shares=100, cost_basis=50.0, acquired=MOMENT.date())])
    wide = ASK.model_copy(update={"max_position_pct": 1.0, "max_new_exposure_pct": 1.0, "max_loss_per_trade_pct": 0.05})
    asked = []
    out = decisions.decide_trades(reviewed(capture), poorer, wide, 80, lambda card: asked.append(card["symbol"]) or True, MOMENT, "ask", None, None)
    assert asked == ["XOM", "EOG"] and [d["will_trade"] for d in out.trade_decisions] == ["yes_user", "no_system_denied", "yes_user"]
    assert [d["shares"] for d in out.trade_decisions if d["will_trade"] == "yes_user"] == [math.floor(2000 / 170.59), math.floor(2000 / 158.32)] == [11, 12]
    assert sum(d["est_cost"] for d in out.trade_decisions) > poorer.cash_avail and all(d["est_cost"] <= poorer.cash_avail for d in out.trade_decisions if d["will_trade"] == "yes_user")


def test_a_purchase_the_even_split_drops_is_never_asked_about(capture):
    out, cards = ask_session(capture, [True, True], budget=200.0)                                                          # $100 each buys no share of XOM ($170.59) or EOG ($158.32)
    assert cards == [] and [d["will_trade"] for d in out.trade_decisions] == ["no_funds", "no_system_denied", "no_funds"] and out.status == "decided"


def test_without_a_split_in_ask_mode_the_user_is_shown_what_the_limits_allow(capture):
    out, cards = ask_session(capture, [True, True], balance=None)
    assert [(c["symbol"], c["shares"]) for c in cards] == [("XOM", 8), ("EOG", 9)] and all("balance" not in d for d in out.trade_decisions)


def test_in_auto_mode_a_review_that_needs_a_person_is_asked_about_at_its_final_size_and_the_rest_trade_on_their_own(capture):
    order = reviewed(capture)
    first = next(i for i, r in enumerate(order.reviews) if r["approved"])
    order.reviews[first] = {**order.reviews[first], "requires_human": True, "requires_human_reasons": ["evidence_not_reviewed"]}
    out, cards = ask_session(capture, [True], profile=PROFILE, mode="auto", order=order)
    assert len(cards) == 1 and cards[0]["of"] == 1 and cards[0]["shares"] == out.trade_decisions[first]["shares"] == 8
    assert [d["will_trade"] for d in out.trade_decisions] == ["yes_user", "no_system_denied", "yes_auto"] and out.trade_decisions[first]["escalated"]


# ---------------------------------------------------------------- the terminal the user types at

class Terminal:
    def __init__(self, *answers):
        self.answers, self.asked, self.shown = list(answers), [], []

    def ask(self, prompt):
        self.asked.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def show(self, text):
        self.shown.append(text)


def test_the_terminal_shows_the_whole_proposal_once_and_then_asks_about_each_trade_with_its_final_size(capture):
    terminal = Terminal("y", "n")
    out = decisions.decide_trades(reviewed(capture), BOB, ASK, 80, decisions.terminal_approver(ask=terminal.ask, show=terminal.show, interactive=True), MOMENT, "ask", "even", None)
    text = "\n".join(terminal.shown)
    assert text.count("ASK MODE  nothing is traded without your yes. The Account Manager proposes 2 trades:") == 1 and text.index("ASK MODE") < text.index("#1  XOM")
    assert "XOM    buy 8 shares, about $1,364.72" in text and "EOG    buy 8 shares, about $1,266.56" in text and "about $2,631.28 of $15,000.00 in cash" in text
    assert terminal.asked == ["Place this trade 1 of 2 (8 shares of XOM, about $1,365)? [y/N] ", "Place this trade 2 of 2 (8 shares of EOG, about $1,267)? [y/N] "]
    assert [d["will_trade"] for d in out.trade_decisions] == ["yes_user", "no_system_denied", "no_user_denied"]
    assert "the user approved (balanced evenly" in out.trade_decisions[0]["reason"] and out.trade_decisions[2]["reason"].startswith("the user denied")


def test_a_terminal_nobody_is_at_asks_nothing_shows_nothing_and_leaves_every_trade_pending(capture):
    terminal = Terminal("y", "y")
    out = decisions.decide_trades(reviewed(capture), BOB, ASK, 80, decisions.terminal_approver(ask=terminal.ask, show=terminal.show, interactive=False), MOMENT, "ask", "even", None)
    assert terminal.asked == [] and terminal.shown == [] and [d["will_trade"] for d in out.trade_decisions] == ["pending_user", "no_system_denied", "pending_user"]


def test_the_session_asks_through_its_own_ask_and_out_and_a_given_ask_is_a_person():
    terminal = Terminal("yes")
    from prometheus.application import session as ss
    approver = ss.approver_for(None, terminal.ask, terminal.show)
    assert approver({"text": "CARD", "symbol": "XOM", "shares": 8, "est_cost": 1364.72}) is True and terminal.shown == ["\nCARD"] and terminal.asked == ["Place this trade (8 shares of XOM, about $1,365)? [y/N] "]
    own = lambda card: False
    assert ss.approver_for(own, terminal.ask, terminal.show) is own
    assert ss.approver_for(None, input, terminal.show)({"text": "CARD", "symbol": "XOM", "shares": 8, "est_cost": 1.0}) is None       # the real input with no terminal (the test run): no answer, never a yes


def test_the_session_says_what_goes_on_to_the_trader_and_who_is_still_to_answer(capture):
    from prometheus.application import session as ss
    yes, _ = ask_session(capture, [True, False])
    lines = "\n".join(ss.decision_lines(yes, "ask"))
    assert "XOM    yes (you approved)  8 shares, about $1,364.72" in lines and "EOG    no (you denied)" in lines and "-> 1 approved trade go on to the Trader." in lines and "still wait" not in lines
    waiting = decisions.decide_trades(reviewed(capture), BOB, ASK, 80, None, MOMENT, "ask", "even", None)
    lines = "\n".join(ss.decision_lines(waiting, "ask"))
    assert "-> nothing was approved to trade: the Trader will not run." in lines and "XOM, EOG still wait for your answer: nobody was at the terminal to ask (Ask mode needs a person at the keyboard" in lines and "--mode auto" in lines
    assert "nobody answered" in "\n".join(ss.decision_lines(waiting, "auto"))


# ---------------------------------------------------------------- the simulated clock dates the real run

def test_with_the_simulated_clock_the_decisions_are_dated_at_the_moment_the_real_run_worked_at(capture, monkeypatch):
    monkeypatch.setenv("USE_LIVE_TRADING", "false")
    saturday = datetime(2026, 11, 14, 1, 30, tzinfo=ZoneInfo("America/New_York"))

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return saturday.astimezone(tz) if tz else saturday.replace(tzinfo=None)
    monkeypatch.setattr(tc, "datetime", Frozen)
    out = decisions.decide_trades(reviewed(capture), BOB, PROFILE, 80, None, None, "auto", "even")        # no moment given: the desk's clock
    assert {d["decided_utc"] for d in out.trade_decisions} == {"2026-11-13T20:30:00Z"} == {d["decided_utc"] for d in capture["order"]["trade_decisions"]}
    assert capture["order"]["trader_runs"][0]["now_utc"] == "2026-11-13T20:30:00Z"                         # the Trader worked at the same moment


# ---------------------------------------------------------------- what the real Trader did

def test_the_real_trades_agree_with_the_real_decisions_and_pass_the_output_guard(capture):
    order = wo.WorkOrder.model_validate(capture["order"])
    assert wo.verify_trades(order) == []
    decisions_by = {d["record_id"]: d for d in order.trade_decisions}
    assert len(order.trades) == sum(1 for d in order.trade_decisions if d["will_trade"] == "yes_auto")
    for trade in order.trades:
        decision, fill = decisions_by[trade["record_id"]], trade["fill"]
        assert trade["status"] == "filled" and trade["side"] == "buy" and fill["shares"] == trade["shares"] == decision["shares"] and trade["symbol"] == decision["symbol"]
        assert trade["limit_low"] <= fill["price"] <= trade["limit_high"] and fill["amount"] == trade_amount(fill["shares"], fill["price"]) and trade["placed_by"] == "model" and trade["booked"] is True
        assert wo.verify_trade(order, trade["record_id"], MOMENT) == []                                    # the permission still holds at the moment of the run


def test_the_real_fills_move_the_account_exactly_as_the_account_books_arithmetic_says(capture):
    """The live run printed `cash $15,000.00 -> $12,367.41` from the real Postgres account book: the same number comes out of `apply_fill` over the real fills."""
    portfolio = BOB
    for trade in capture["order"]["trades"]:
        portfolio = apply_fill(portfolio, trade["symbol"], trade["side"], trade["fill"]["shares"], trade["fill"]["price"], MOMENT.date())
    assert portfolio.cash_avail == 12367.41 and {p.symbol: p.shares for p in portfolio.positions} == {t["symbol"]: t["shares"] for t in capture["order"]["trades"]}
    assert round(15000.0 - sum(t["fill"]["amount"] for t in capture["order"]["trades"]), 2) == portfolio.cash_avail


# ---------------------------------------------------------------- what the real session printed about how much was bought

def test_the_how_much_was_bought_report_says_what_held_each_purchase_back_from_the_accounts_own_decisions(capture):
    from prometheus.application import session as ss
    out = decide(capture)                                                                                           # the Account Manager's decisions over the real reviews: XOM 8, EOG 8, an even split
    after = BOB
    for trade in capture["order"]["trades"]:
        after = apply_fill(after, trade["symbol"], trade["side"], trade["fill"]["shares"], trade["fill"]["price"], MOMENT.date())
    text = "\n".join(ss.use_of_money_lines(out, BOB, after, PROFILE))
    assert "HOW MUCH WAS BOUGHT  spent $2,632.59 of $15,000.00 (18% of the money)" in text                      # the real fills: 8 XOM at 170.6753 and 8 EOG at 158.3992
    assert "XOM    8 shares ($1,364.72) of the $15,000.00: held back by the position limit (10% of equity in one stock)" in text
    assert "EOG    8 shares ($1,266.56) of the $15,000.00: held back by the even split: the same dollars for every company, and another company's limit set the amount" in text
    assert "the exposure limit" not in text.split("To buy more")[0] and "the loss limit" not in text and "the Critic allowed" not in text
    assert "The money was split evenly: the same dollars for every company, in whole shares." in text
    assert "To buy more, raise the limits for the next session (the account book keeps its own): --max-position-pct 0.5" in text
    assert "Advisor" not in text.replace("which recommends", "") and "Critic" not in text                       # no size is credited to the Advisor or the Critic


def test_the_report_says_the_funds_were_spent_round_robin_in_auto_mode_and_what_closed_each_purchase(capture):
    from prometheus.application import session as ss
    out = decide(capture, balance=None, budget=400.0)
    text = "\n".join(ss.use_of_money_lines(out, BOB, BOB, PROFILE))
    assert "XOM    1 share ($170.59) of the $15,000.00: held back by the amount you asked to spend: it had no room for another share" in text and "EOG    1 share ($158.32)" in text
    assert "Auto mode spent the funds round-robin: one share of the first stock, one of the next, and so on in rank order, until the money (or every stock's limits) ran out." in text
    full = "\n".join(ss.use_of_money_lines(decide(capture, balance=None), BOB, BOB, PROFILE))
    assert "XOM    8 shares" in full and "EOG    9 shares" in full and full.count("the position limit (10% of equity in one stock)") == 2 and "round-robin" in full
    nothing = decide(capture, balance=None, budget=100.0)                                                           # $100 buys no share of either
    assert "XOM    no share was bought: " in "\n".join(ss.use_of_money_lines(nothing, BOB, BOB, PROFILE))


# ---------------------------------------------------------------- the Advisor's record holds no size (entry 80)

def test_the_real_recommendations_are_valid_records_with_prices_and_dates_and_no_size(capture):
    from prometheus.agents.advisor.record import PortfolioFit, Recommendation, TradePlan

    gone = {"equity", "equity_basis", "risk_budget", "suggested_shares", "position_pct", "oversized", "size_note", "funds_feasible", "funds_note"}
    for record in capture["order"]["recommendations"]:
        valid = Recommendation.model_validate(record)                                                                  # extra="forbid": a size field left in the capture would fail here
        assert not gone & set(record["trade"]) and not {"position_pct_after", "funds_feasible"} & set(record["portfolio_fit"])
        assert valid.trade.entry_zone and valid.trade.stop_price and valid.trade.stop_distance and valid.trade.time_stop_date and valid.trade.order_style    # the plan: prices and dates
        assert set(record["trade"]) == set(TradePlan.model_fields) and set(record["portfolio_fit"]) == set(PortfolioFit.model_fields)
    assert [r["symbol"] for r in capture["order"]["recommendations"]] == ["XOM", "APA", "EOG"]
