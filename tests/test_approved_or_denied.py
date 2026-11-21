"""Approved or denied (design choices entries 103 to 105; design/approved_or_denied_spec.md): the Critic's verdict is final, Auto mode asks nothing, and only the Account Manager touches the holdings.

REAL data only: the real reviews of the real session `tests/fixtures/session_real/bob_jones_oil.json` (XOM approved at 86, EOG approved at 76, APA denied for its risk) and of `pam_smith_neutral.json` (SNDK denied), and, for
Part C, the real Advisor captures. No stand-in agent and no stand-in model. The captures are not edited: a test that needs another shape of record makes it from the real one and says so.
"""
import copy
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from prometheus.agents.account_manager import decisions
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.application import session
from prometheus.application.web import views
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio
from prometheus.common.user_profile import UserProfile

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
MOMENT = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)          # 15:30 Eastern on Friday 2026-11-13: the simulated moment of the real run
BOB = Portfolio(user="Bob Jones", cash_avail=Decimal("15000"))
AUTO = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="auto")
ASK = UserProfile(user="Bob Jones", risk_tolerance="conservative", age=60, trade_mode="ask")


def capture(name: str) -> dict:
    return json.loads((SESSIONS / f"{name}.json").read_text(encoding="utf-8"))


def reviewed(name: str = "bob_jones_oil") -> wo.WorkOrder:
    """The real work order as it was when the Critic had finished: before the Account Manager decided and before the Trader traded."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture(name)["order"]))
    order.status, order.trade_decisions, order.trades, order.trader_runs = "reviewed", [], [], []
    return order


def never_asked(card):
    pytest.fail(f"Auto mode asked the user about {card.get('symbol')}")


# ---------------------------------------------------------------- Part B: approved trades, nothing is asked in Auto mode

def test_auto_mode_trades_every_approved_review_of_the_real_session_and_asks_nobody():
    """Jacob: *"Auto mode will not ask any prompts"* and *"Trade all approved recommendations in auto mode"*. The real reviews are XOM 86 and EOG 76 (approved) and APA 49 (denied): under the old second mark of 80, EOG
    waited for the user; now both approved purchases are `yes_auto` and the approver is never called, with the Critic's own reason in the decision."""
    out = decisions.decide_trades(reviewed(), BOB, AUTO, never_asked, MOMENT, "auto", None, None)
    by = {d["symbol"]: d for d in out.trade_decisions}
    assert [by[s]["will_trade"] for s in ("XOM", "APA", "EOG")] == ["yes_auto", "no_system_denied", "yes_auto"] and out.status == "decided"
    assert not any(d["asked_user"] for d in out.trade_decisions) and all("escalated" not in d for d in out.trade_decisions)
    assert by["XOM"]["reason"].startswith("approved by the Critic (score 86): Auto mode trades what the Critic approves") and by["EOG"]["reason"].startswith("approved by the Critic (score 76): Auto mode trades what the Critic approves")
    assert by["XOM"]["shares"] >= 1 and by["EOG"]["shares"] >= 1 and by["APA"]["shares"] == 0
    for symbol in ("XOM", "EOG"):                                                       # and the Trader's own check of the decision lets each trade
        assert wo.verify_trade(out, by[symbol]["record_id"], MOMENT) == []
    assert wo.verify_trade(out, by["APA"]["record_id"], MOMENT) != []


def test_the_account_manager_in_auto_mode_asks_nothing_even_with_an_approver_that_would_say_no():
    manager = AccountManager(model=object(), model_id="account-book", profile=AUTO)
    out = manager.decide_trades(reviewed(), BOB, never_asked, MOMENT, "auto")
    assert [d["will_trade"] for d in out.trade_decisions] == ["yes_auto", "no_system_denied", "yes_auto"]


def test_in_ask_mode_the_user_is_still_asked_about_each_approved_purchase_and_never_about_a_denial():
    cards = []
    out = decisions.decide_trades(reviewed(), BOB, ASK, lambda card: cards.append(card["symbol"]) or True, MOMENT, "ask", None, None)
    assert cards == ["XOM", "EOG"] and [d["will_trade"] for d in out.trade_decisions] == ["yes_user", "no_system_denied", "yes_user"]


def test_the_denied_company_is_returned_to_the_user_by_the_account_manager_with_the_critics_reason():
    """Jacob: *"The denied recommendations need to be returned to the user from the AccountManager."* This is what the session prints under the decision node, and what the page's feed shows."""
    out = decisions.decide_trades(reviewed(), BOB, AUTO, never_asked, MOMENT, "auto", None, None)
    lines = session.decision_lines(out, "auto")
    assert "      DENIED by the Critic, not traded:" in lines
    assert ("        APA    score 49 (F): The risk is medium (29.8), above the low level a conservative profile accepts (risk_above_profile)." in lines)
    assert lines.index("      DENIED by the Critic, not traded:") < lines.index("      -> 2 approved trades go on to the Trader.")
    assert not any(line.startswith("        XOM") or line.startswith("        EOG") for line in lines)                  # an approved one is not "denied"
    assert decisions.denied_lines(out) == lines[lines.index("      DENIED by the Critic, not traded:"):lines.index("      DENIED by the Critic, not traded:") + 2]


def test_a_review_that_needed_a_person_is_returned_to_the_user_with_that_as_the_reason():
    """The real XOM review made what the Critic writes when a finding asks for a person (entry 103: denied, `needs_human_review`, the score not capped). The Account Manager returns it like any other denial."""
    order = reviewed()
    xom = order.reviews[0]
    order.reviews[0] = {**xom, "approved": False, "valid_until_utc": None, "denial_reasons": ["needs_human_review"], "requires_human": True, "requires_human_reasons": ["evidence_not_reviewed"],
                        "summary": "DENY XOM: a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: no key)."}
    out = decisions.decide_trades(order, BOB, AUTO, never_asked, MOMENT, "auto", None, None)
    lines = session.decision_lines(out, "auto")
    assert ("        XOM    score 86 (B): a person must look before this can trade: evidence_not_reviewed (The quotes could not be reviewed by the model: no key)." in lines)
    assert [d["will_trade"] for d in out.trade_decisions] == ["no_system_denied", "no_system_denied", "yes_auto"]


def test_a_run_where_everything_was_denied_says_so_in_its_first_reason_for_the_session_and_the_page():
    """The real SNDK review of `pam_smith_neutral.json` is the only review of that run and it was denied. Its trades were neutral purchases, so the order is made what a run with no neutral trade would leave: the Critic's
    review alone."""
    order = reviewed("pam_smith_neutral")
    order.trade_decisions = []
    assert len(order.reviews) == 1 and not order.reviews[0]["approved"]
    reasons = session.why_not(None, order)
    assert reasons[0] == "the Critic denied the recommendation and approved none: SNDK (risk_above_profile, falling_knife, dead_cat_bounce)"
    account = {"cash": Decimal("15000"), "positions": [], "invested": 0, "total": Decimal("15000")}
    verdict = views.run_verdict(account, account, action="buy", order=order)
    assert verdict["status"] == "no_trade" and verdict["headline"] == "Nothing was traded: " + reasons[0] and verdict["reasons"] == reasons


def test_the_critics_feedback_on_a_recommendation_has_no_second_mark_and_no_escalation():
    block = capture("bob_jones_oil")["order"]["recommendations"][0]["critic"]
    assert "auto_min_score" in block["gate"] and "escalated" in block["decision"]                       # the REAL capture is old and is not edited: it still holds both keys (what the new code no longer writes)
    from prometheus.agents.critic import feedback
    out = decisions.decide_trades(reviewed(), BOB, AUTO, never_asked, MOMENT, "auto", None, None)
    new = next(r for r in out.recommendations if r["symbol"] == "XOM")["critic"]
    assert new["decision"]["will_trade"] == "yes_auto" and "escalated" not in new["decision"] and new["decision"]["reason"].startswith("approved by the Critic (score 86)")
    assert feedback.FORMULA.endswith("or a finding that asks for a person to look")      # Critic v3 (was "no finding that asks for a person to look"): the score has no mark; the REAL capture above is old and still holds `approve_min_score`


# ---------------------------------------------------------------- Part C: the Advisor and the Critic never touch the holdings (entry 105)

def keys_of(value):
    """Every key of every dict inside a record, however deep (a list is walked, a value is not a key)."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from keys_of(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from keys_of(inner)


def model_keys(model, seen=None):
    """Every field name of a pydantic model and of the models inside it."""
    import typing

    from pydantic import BaseModel
    seen = seen if seen is not None else set()
    for name, field in model.model_fields.items():
        seen.add(name)
        for inner in typing.get_args(field.annotation) + (field.annotation,):
            args = typing.get_args(inner)
            for candidate in (inner, *args):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel) and candidate is not model:
                    model_keys(candidate, seen)
    return seen


PORTFOLIO_WORDS = re.compile(r"portfolio|held|holding|cost_basis|unrealized", re.I)


def test_no_key_of_a_version_3_record_or_candidate_says_portfolio_held_holding_cost_basis_or_unrealized():
    """Jacob (2026-11-15): *"never any portfolio related information"*. The record models' fields, and the keys of the REAL new records and candidates the Advisor writes on the real captures, hold none of those words."""
    from test_advisor_sndk_pg import replay

    from prometheus.agents.advisor.record import SCHEMA_VERSION, Candidate, Recommendation
    assert SCHEMA_VERSION == 3 and not [k for k in model_keys(Recommendation) | model_keys(Candidate) if PORTFOLIO_WORDS.search(k)]
    seen = set()
    for name in ("SNDK", "PG", "PG_open"):
        order, _ = replay(name)
        assert order.recommendations
        for record in [*order.recommendations, *order.candidates]:
            seen |= set(keys_of(record))
        assert all(r["identity"]["schema_version"] == 3 and r["action"]["action"] == "open" for r in order.recommendations)
    assert {"concentration", "action", "cooldown_of"} <= seen and not [k for k in seen if PORTFOLIO_WORDS.search(k)]


def test_a_version_2_record_loads_and_the_same_record_made_to_say_it_is_held_or_to_reduce_does_not():
    """The real version 2 recommendations of `bob_jones_oil.json` (every action `open`, nothing held) load, and come out as version 3 shape. The SAME record made to say something is held, or to reduce, add, hold or avoid, is
    refused with the reason; so is a record whose `portfolio_fit` carries a number, and a candidate that says it is held."""
    from pydantic import ValidationError

    from prometheus.agents.advisor.record import Candidate, Recommendation
    records = capture("bob_jones_oil")["order"]["recommendations"]
    assert len(records) == 3 and all(r["identity"]["schema_version"] == 2 and r["action"]["held_shares"] == 0 and "portfolio_fit" in r for r in records)
    for record in records:
        loaded = Recommendation.model_validate(record)
        assert loaded.action.action == "open" and loaded.concentration.note == record["portfolio_fit"]["note"] and "portfolio_fit" not in loaded.model_dump()
    record = copy.deepcopy(records[0])
    for change, message in (({"held_shares": 60}, "the record says something is held (held_shares 60)"), ({"cost_basis": 12.5}, "the record says something is held (cost_basis 12.5)"),
                            ({"unrealized_pct": 0.1}, "the record says something is held (unrealized_pct 0.1)")):
        with pytest.raises(ValidationError, match=re.escape(message)):
            Recommendation.model_validate({**record, "action": {**record["action"], **change}})
    for action in ("reduce", "add", "hold", "avoid"):
        with pytest.raises(ValidationError, match=f"the action is '{action}': a recommendation is to buy"):
            Recommendation.model_validate({**record, "action": {**record["action"], "action": action}})
    with pytest.raises(ValidationError, match="the record carries portfolio information"):
        Recommendation.model_validate({**record, "portfolio_fit": {**record["portfolio_fit"], "sector_weight_after": 0.2}})
    candidate = {"symbol": "ZZZ", "origin": "subject", "reason": "stance neutral", "held": False}
    assert Candidate.model_validate(candidate).symbol == "ZZZ" and "held" not in Candidate.model_validate(candidate).model_dump()
    with pytest.raises(ValidationError, match="the candidate says it is held"):
        Candidate.model_validate({**candidate, "held": True})


def test_a_sale_tied_to_a_recommendation_fails_the_trader_guards_but_the_users_own_order_and_a_funding_sale_do_not():
    """`verify_trades` (the whole run) and `verify_fill` (one fill) accept a sale only for the user's own direct order (a decision with `origin: "direct"`) or a funding sale of RaiseFunds; a sale tied to a recommendation is
    refused. The REAL fills of `bob_jones_oil.json` (two purchases) are made a sale of the same decisions: refused; the real direct order of `bill_gates_direct_order_funded.json` made a sale: allowed."""
    order = wo.WorkOrder.model_validate(capture("bob_jones_oil")["order"])
    assert wo.verify_trades(order) == []
    sale = copy.deepcopy(order)
    for decision in sale.trade_decisions:
        decision["action"] = "reduce"                                                         # the decisions say sell, so a sell would match a decision's side: what is refused is whose decision it is
    for trade in sale.trades:
        trade["side"] = "sell"
    reasons = wo.verify_trades(sale)
    assert any("is a sale tied to recommendation" in r and "never from a recommendation" in r for r in reasons) and len([r for r in reasons if "is a sale tied to recommendation" in r]) == 2
    first = sale.trades[0]
    assert any("is a sale tied to recommendation" in r for r in wo.verify_fill(sale, first, [t for t in sale.trades if t is not first]))
    direct = wo.WorkOrder.model_validate(capture("bill_gates_direct_order_funded")["order"])
    assert wo.verify_trades(direct) == []
    for decision in direct.trade_decisions:
        decision["action"] = "reduce"
    for trade in direct.trades:
        trade["side"] = "sell"
    direct.direct_orders[0]["side"] = "sell"
    assert not [r for r in wo.verify_trades(direct) if "is a sale tied to recommendation" in r] and direct.trade_decisions[0]["origin"] == "direct"


def test_the_advisor_writes_only_recommendations_candidates_runs_and_status_and_the_critic_only_reviews_runs_and_the_status_on_each_recommendation():
    """Jacob: they *"only add recommendations and update the recommendations section with the scoring, the actual recommendation, the status, and other metadata"*. Each is run on a real capture and every other part of the work
    order is compared before and after."""
    from critic_helpers import NOW, make_critic, view
    from test_advisor_sndk_pg import FIXTURES, replay

    def dumped(order):
        return json.loads(order.model_dump_json())

    fx = json.loads((FIXTURES / "PG_open.json").read_text(encoding="utf-8"))
    before = dumped(wo.WorkOrder.model_validate(fx["order"]))
    order, _ = replay("PG_open")
    after = dumped(order)
    changed = {k for k in after if after[k] != before[k]}
    assert changed <= {"recommendations", "candidates", "advisor_runs", "status", "schema_version", "data", "errors"} and {"recommendations", "advisor_runs", "status"} <= changed
    assert {k for k in after["data"] if after["data"][k] != before["data"][k]} == {"metrics"}            # the metrics it asked the Data Clerk for, kept on the data for the next agent
    assert after["reviews"] == after["trade_decisions"] == after["trades"] == after["direct_orders"] == after["critic_runs"] == after["trader_runs"] == []
    assert all(r["action"]["action"] == "open" and r["critic"] is None for r in order.recommendations)
    advised = dumped(order)
    make_critic(order).review_recommendations(order, view(), NOW)
    reviewed_ = dumped(order)
    changed = {k for k in reviewed_ if reviewed_[k] != advised[k]}
    assert changed <= {"reviews", "critic_runs", "recommendations", "status", "schema_version"} and {"reviews", "critic_runs", "status"} <= changed
    strip = lambda records: [{k: v for k, v in r.items() if k != "critic"} for r in records]
    assert strip(reviewed_["recommendations"]) == strip(advised["recommendations"]) and all(r["critic"] for r in reviewed_["recommendations"])      # the Critic never changes what the Advisor wrote: it adds its status
    assert reviewed_["candidates"] == advised["candidates"] and reviewed_["advisor_runs"] == advised["advisor_runs"] and reviewed_["trade_decisions"] == reviewed_["trades"] == reviewed_["direct_orders"] == []
    assert reviewed_["data"] == advised["data"]


def test_the_advisor_is_given_the_sample_portfolios_holdings_nowhere_gm_msft_and_tsla_are_not_in_the_replay():
    """The captures were made with the sample portfolio (MSFT, TSLA, GM); replayed with the Advisor as it is now there is no trace of them in anything it writes."""
    from test_advisor_sndk_pg import replay
    for name in ("SNDK", "PG", "PG_open"):
        order, _ = replay(name)
        text = json.dumps([order.recommendations, order.candidates, order.advisor_runs], default=str)
        assert not re.search(r'"(?:symbol|company)": "(?:MSFT|TSLA|GM)"', text) and not any(r["symbol"] in ("MSFT", "TSLA", "GM") for r in [*order.recommendations, *order.candidates])


# ---------------------------------------------------------------- a real Auto session made AFTER this change (2026-11-15, throwaway account, no fetch, deleted afterwards)

def test_the_real_auto_session_asked_nobody_traded_an_approval_at_79_and_returned_cop_with_both_reasons():
    """`tests/fixtures/session_real/auto_oil_no_second_mark.json` is a real session of the new desk (`session --mode auto --yes --no-fetch --cleanup --save`; the real Account Manager, Data Clerk, Advisor (gpt-oss-120b), Critic
    (GLM 5.3) and Trader, the simulated close of Friday 2026-11-13): a conservative 60-year-old with $15,000 asked for the 3 most recommended oil companies, spread evenly. EOG was approved at 79 (under the old mark of 80 it
    would have waited for the user), CVX at 86, and COP was DENIED for its risk AND because a finding asked for a person (fresh negative news): two reasons, a score of 49 that the blocker capped from 65. Nothing was asked."""
    data = capture("auto_oil_no_second_mark")
    order = wo.WorkOrder.model_validate(copy.deepcopy(data["order"]))
    by = {r["symbol"]: r for r in order.reviews}
    assert (by["EOG"]["approved"], by["EOG"]["score"], by["CVX"]["approved"], by["CVX"]["score"]) == (True, 79, True, 86)
    cop = by["COP"]
    assert (cop["approved"], cop["score"], cop["score_before_blockers"], cop["denial_reasons"], cop["requires_human_reasons"]) == (False, 49, 65, ["risk_above_profile", "needs_human_review"], ["fresh_negative_news"])
    live = {d["symbol"]: d for d in order.trade_decisions}
    assert (live["EOG"]["will_trade"], live["CVX"]["will_trade"], live["COP"]["will_trade"]) == ("yes_auto", "yes_auto", "no_system_denied") and not any(d["asked_user"] or "escalated" in d for d in live.values())
    assert {r["action"]["action"] for r in order.recommendations} == {"open"} and {r["identity"]["schema_version"] for r in order.recommendations} == {3}
    advisors = [{k: v for k, v in r.items() if k != "critic"} for r in order.recommendations]                      # the Advisor's own record: the Critic's block has `patterns.holding` (the chart patterns that HOLD), which is not a holding of the user's
    assert not [k for r in advisors for k in keys_of(r) if PORTFOLIO_WORDS.search(k)] and "auto_min_score" not in json.dumps(order.recommendations)      # a version 3 record, and the Critic's block has no second mark
    again = copy.deepcopy(order)
    again.status, again.trade_decisions, again.trades, again.trader_runs = "reviewed", [], [], []
    now = datetime(2026, 11, 13, 20, 30, tzinfo=timezone.utc)
    out = decisions.decide_trades(again, Portfolio(user="Auto Mode Check", cash_avail=Decimal("15000")), UserProfile(user="Auto Mode Check", risk_tolerance="conservative", age=60, trade_mode="auto"), never_asked, now, "auto", "even", None)
    assert [(d["symbol"], d["will_trade"], d["shares"]) for d in out.trade_decisions] == [(s, live[s]["will_trade"], live[s]["shares"]) for s in ("EOG", "CVX", "COP")]       # the same decisions, nobody asked
    lines = session.decision_lines(out, "auto")
    assert any(line.startswith("        COP    score 49 (F): The risk is medium (20.3), above the low level a conservative profile accepts (risk_above_profile); a person must look before this can trade: fresh_negative_news (") for line in lines)


# ---------------------------------------------------------------- Part D: the money is spread round-robin in Ask mode too (entry 106)

def test_ask_mode_sizes_jacobs_request_as_auto_mode_did_with_the_same_real_prices_and_none_is_no_funds():
    """Jacob (2026-11-15), told that his Ask-mode request had put the whole $10,000 into F (839 shares) and left GM, PG and GOOGL with "no funds": *"No. its suppose to be round-robin. Not all in one."* The REAL desk over the
    same request and the same stored closes, in Ask mode: F 23, GM 19, PG 18 and GOOGL 17, exactly what the real Auto session did (`d_pyrett_names_auto.json`), and each is asked about and bought."""
    from test_direct_order import JacobDesk

    before = {d["symbol"]: (d["shares"], d["will_trade"]) for d in capture("d_pyrett_names_ask")["order"]["trade_decisions"]}
    assert before == {"F": (839, "yes_user"), "GM": (0, "no_funds"), "PG": (0, "no_funds"), "GOOGL": (0, "no_funds")}                       # what the real Ask session did: everything in the first stock
    auto = {d["symbol"]: d["shares"] for d in capture("d_pyrett_names_auto")["order"]["trade_decisions"]}
    assert auto == {"F": 23, "GM": 19, "PG": 18, "GOOGL": 17}
    run = JacobDesk("ask", answers=[True] * 5)
    order = run.state["order"]
    # AMENDED (entry 111, M3): a direct order with money buys as many whole shares as the money pays for and a FRACTION of a share with what is left (Jacob: "buy as many whole shares as possible and a fractional share for the leftover funds
    # available"): the leftover of the whole-share round robin goes to GOOGL, the stock with the fewest shares (the real Auto capture, made before, has 17); the other three are as they were
    assert {d["symbol"]: d["shares"] for d in order.trade_decisions} == {**auto, "GOOGL": Decimal("17.025628")} and not [d for d in order.trade_decisions if d["will_trade"] == "no_funds"]
    assert [(t["symbol"], t["status"], t["fill"]["shares"]) for t in order.trades] == [("F", "filled", 23), ("GM", "filled", 19), ("PG", "filled", 18), ("GOOGL", "filled", Decimal("17.025628"))]
    assert sum(d["est_cost"] for d in order.trade_decisions) <= Decimal("10000")


def test_split_evenly_is_still_equal_dollars_in_ask_mode_and_round_robin_is_the_default_in_both_modes():
    out = decisions.decide_trades(reviewed(), BOB, ASK, lambda card: True, MOMENT, "ask", "even", Decimal("2800"))
    a, b = [d for d in out.trade_decisions if d["will_trade"] == "yes_user"]
    assert a["balance"] == {"mode": "even", "target_dollars": Decimal("1400"), "of": 2} and b["balance"]["target_dollars"] == Decimal("1400") and (a["shares"], b["shares"]) == (8, 9)       # AMENDED (entry 108): $1,400 each bought XOM 8 ($1,364.72) and EOG 8 ($1,266.56); the $168.72 left bought EOG one more ($158.32), the smaller amount
    for mode, profile in (("ask", ASK), ("auto", AUTO)):
        out = decisions.decide_trades(reviewed(), BOB, profile, lambda card: True, MOMENT, mode, None, Decimal("2800"))
        assert [(d["symbol"], d["shares"]) for d in out.trade_decisions if d["shares"]] == [("XOM", 8), ("EOG", 9)] and {d["size"]["allocation"] for d in out.trade_decisions if d["size"]} == {"round_robin"}
