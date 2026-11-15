"""The desk with the Critic (Critic spec 13.1): `build_desk(with_critic=True)` runs account_manager -> data_clerk -> advisor -> critic -> account_manager_decision on ONE work order.

Everything is a fake: the Data Clerk marks the order ready, the Advisor puts the REAL captured PG_open recommendation on it (made for this order's id), the Critic is the real one over a fake Data Clerk and
a scripted model, and the clock is Thursday 2026-11-05 10:30 Eastern.
"""
from datetime import datetime, timedelta, timezone

import pytest
from advisor_helpers import MemoryLedger
from critic_helpers import NOW, FakeClerk as CriticClerk, ReviewerModel, captured, profile, tamper
from test_desk_graph import FakeClerk as DeskClerk, FakeResolver

from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.agents.critic.critic import Critic
from prometheus.common.portfolio import Portfolio
from prometheus.langgraph import desk

RICH = Portfolio(user="Jacob Pyrett", cash_avail=74481.40)      # what the sample portfolio is worth at the capture's prices: the cash buys 445 shares of PG at 167.10 (the sample's own $51.23 buys none)


class CapturedAdvisor:
    """The Advisor of version 1 as the desk calls it: puts the captured recommendations (made for THIS order) on the order."""
    seen = None

    def __init__(self, name="PG_open"):
        self.name = name

    def generate_recommendations(self, order, portfolio_, case=None, today=None, **kw):
        CapturedAdvisor.seen = (order, portfolio_, today)
        source = captured(self.name)
        order.data = source.data
        named = set(order.subject or [])           # the captured SNDK run also holds GM; a request for SNDK is answered with SNDK's recommendation
        order.recommendations = [tamper(r, **{"identity.request_id": order.request_id}) for r in source.recommendations if not named or r["symbol"] in named]
        order.candidates = source.candidates
        order.status = "recommended"
        return order


class Manager(AccountManager):
    """The real Account Manager (profile, decisions), with a RaiseFunds that only records what it was asked."""

    def __init__(self, **kw):
        super().__init__(model=object(), **kw)
        self.raised = []

    def raise_funds(self, portfolio_, target, shares, **kw):
        self.raised.append((target, shares))
        return {"status": "funded", "quotes": {}, "run": {}, "plan": None, "errors": [], "target": target, "shares": shares}


def critic_for(name="PG_open", **options):
    source = captured(name)
    return lambda order: Critic(model=ReviewerModel(), clerk=CriticClerk(source, **options), ledger=MemoryLedger())


def run(request=None, name="PG_open", approver=None, critic=None, manager=None, **options):
    CapturedAdvisor.seen = None
    return desk.run_desk(request or {"symbol": "PG", "from_days_ago": 14, "fetch": False, "account": {"portfolio": RICH}}, clerk_factory=lambda order: DeskClerk(), advisor_factory=lambda order: CapturedAdvisor(name),
                         resolver=FakeResolver(), with_critic=True, critic_factory=critic or critic_for(name), funds_factory=(lambda: manager) if manager else (lambda: Manager(profile=profile())),
                         approver=approver, clock=lambda: NOW, **options)


def test_the_graph_has_the_two_new_nodes_and_stops_early_when_nothing_was_recommended():
    g = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_critic=True).get_graph()
    assert set(g.nodes) == {"__start__", "account_manager", "data_clerk", "advisor", "critic", "account_manager_decision", "account_manager_direct", "__end__"}      # account_manager_direct: a trade the user names themselves (entry 87)
    edges = {(e.source, e.target, bool(e.conditional)) for e in g.edges}
    assert ("advisor", "critic", True) in edges and ("advisor", "__end__", True) in edges and ("critic", "account_manager_decision", False) in edges and ("advisor", "account_manager_decision", True) in edges and ("account_manager_decision", "__end__", False) in edges
    with_funds = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_critic=True, with_funds=True).get_graph()
    assert ("account_manager_decision", "account_manager_funds", True) in {(e.source, e.target, bool(e.conditional)) for e in with_funds.edges}


def test_without_the_critic_the_desk_is_exactly_what_it_was():
    g = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None).get_graph()
    assert set(g.nodes) == {"__start__", "account_manager", "data_clerk", "advisor", "__end__"}


def test_one_work_order_travels_through_the_critic_and_the_decision_and_comes_out_decided():
    state = run(approver=lambda card: True)
    order = state["order"]
    assert state["path"] == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision"] and state["errors"] == []
    assert order.status == "decided" and order.schema_version == 3 and len(order.reviews) == 1 and order.reviews[0]["approved"]
    assert [(d["symbol"], d["will_trade"], d["shares"], d["mode"]) for d in order.trade_decisions] == [("PG", "yes_user", 445, "ask")]
    assert set(state["seconds"]) == {"account_manager", "data_clerk", "advisor", "critic", "account_manager_decision"}


def test_the_advisor_runs_version_1_with_the_portfolio_and_the_easterns_date_of_the_clock():
    late = datetime(2026, 11, 5, 19, 30, tzinfo=timezone(timedelta(hours=-5)))        # 19:30 Eastern: the UTC date is already the 6th
    run_late = lambda: desk.run_desk({"symbol": "PG", "fetch": False, "account": {"portfolio": RICH}}, clerk_factory=lambda o: DeskClerk(), advisor_factory=lambda o: CapturedAdvisor(), resolver=FakeResolver(), with_critic=True,
                                     critic_factory=critic_for(), funds_factory=lambda: Manager(profile=profile()), clock=lambda: late)
    run_late()
    order, portfolio_, today = CapturedAdvisor.seen
    assert str(today) == "2026-11-05" and portfolio_.user == "Jacob Pyrett" and order.reviews[0]["reviewed_utc"] == "2026-11-06T00:30:00Z"      # reviewed at the clock's moment, dated by the Eastern day


def test_without_an_answer_the_decision_waits_and_the_order_stays_reviewed():
    order = run()["order"]
    assert order.status == "reviewed" and order.trade_decisions[0]["will_trade"] == "pending_user"


def test_in_auto_mode_the_critics_approval_trades_without_asking():
    asked = []
    state = run({"symbol": "PG", "fetch": False, "account": {"mode": "auto", "portfolio": RICH}}, approver=lambda card: asked.append(card))
    assert state["order"].trade_decisions[0]["will_trade"] == "yes_auto" and asked == [] and state["order"].status == "decided"


def test_the_request_can_give_the_portfolio_and_the_profile_as_objects_or_files():
    """The Account Manager sizes the purchase from the portfolio's money alone (the Critic sees neither the money nor any limit, and there is no limit: design choices entry 91): $74,481.40 of cash buys as many whole shares of PG
    as the money pays for. A portfolio given as a FILE is read too: the sample's $51.23 buys no share of PG."""
    request = {"symbol": "PG", "fetch": False, "account": {"portfolio": RICH, "profile": profile()}}
    d = run(request)["order"].trade_decisions[0]
    assert d["size"]["binding"] == "money" and d["shares"] == d["size"]["caps"]["money"] >= 100
    request = {"symbol": "PG", "fetch": False, "account": {"portfolio": "configs/portfolios/jacob_pyrett.json".replace("configs", str(desk.DEFAULT_PORTFOLIO.parents[1])), "profile": str(desk.DEFAULT_PROFILE)}}
    d = run(request)["order"].trade_decisions[0]
    assert (d["will_trade"], d["shares"], d["size"]["caps"]["money"]) == ("no_funds", 0, 0) and d["size"]["equity_basis"].startswith("cash and holdings")      # the file's own portfolio: $51.23 and three holdings


def test_a_denied_recommendation_is_final_and_with_nothing_approved_the_funds_node_refuses():
    state = run({"symbol": "SNDK", "fetch": False, "funds": {"shares": 1}}, name="SNDK", with_funds=True)
    order = state["order"]
    assert order.trade_decisions[0]["will_trade"] == "no_system_denied" and state["path"][-1] == "account_manager_funds"
    assert state["funds"]["status"] == "refused" and "Critic approved no purchase" in state["funds"]["errors"][0]


def test_raise_funds_buys_what_the_critic_approved_and_the_user_allowed_with_the_account_managers_size():
    manager = Manager(profile=profile())
    state = run({"symbol": "PG", "fetch": False, "funds": {}, "account": {"portfolio": RICH}}, approver=lambda card: True, with_funds=True, manager=manager)
    assert manager.raised == [("PG", 445)] and state["funds"]["status"] == "funded"
    manager = Manager(profile=profile())
    run({"symbol": "PG", "fetch": False, "funds": {"shares": 3}, "account": {"portfolio": RICH}}, approver=lambda card: True, with_funds=True, manager=manager)
    assert manager.raised == [("PG", 3)]
    manager = Manager(profile=profile())
    run({"symbol": "PG", "fetch": False, "funds": {"buy": "MU", "shares": 2}, "account": {"portfolio": RICH}}, with_funds=True, manager=manager)      # an explicit symbol is the user's own request
    assert manager.raised == [("MU", 2)]


def test_nothing_recommended_ends_the_desk_after_the_advisor():
    class Nothing(CapturedAdvisor):
        def generate_recommendations(self, order, portfolio_, case=None, today=None, **kw):
            order.status = "recommended"
            return order
    state = desk.run_desk({"symbol": "PG", "fetch": False}, clerk_factory=lambda o: DeskClerk(), advisor_factory=lambda o: Nothing(), resolver=FakeResolver(), with_critic=True, critic_factory=critic_for(),
                          funds_factory=lambda: Manager(profile=profile()), clock=lambda: NOW)
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and state["order"].reviews == []


def test_a_critic_that_cannot_start_is_a_message_on_the_order_not_an_exception():
    def broken(order):
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")
    state = run(critic=broken)
    assert "the Critic could not start: OPENROUTER_API_KEY is not set" in " ".join(state["order"].errors) and state["order"].reviews == [] and state["order"].status == "recommended"
    assert any("decides on an order the Critic reviewed" in e for e in state["order"].errors)


def test_a_bad_profile_or_portfolio_file_is_a_message_on_the_order():
    state = run({"symbol": "PG", "fetch": False, "account": {"profile": "/nonexistent/profile.json"}})
    assert any("the profile file does not exist" in e for e in state["order"].errors) and state["order"].status == "data_ready"


def test_the_report_has_the_critic_section_with_the_scorecard_and_the_decision():
    state = run(approver=lambda card: True)
    text = desk.report(state, 1.0)
    assert "CRITIC REVIEW  (model z-ai/glm-5.3" in text and "APPROVE" in text and "open a position      valid until 11:30 ET" in text and "TRADE DECISIONS  PG: yes (you approved)" in text
    assert "44 of 75 shares" not in text and "size cut" not in text                                    # the Critic's scorecard names no size
    assert "critic" in text.split("TIMING")[1] and "account_manager_decision" in text.split("TIMING")[1]


def test_a_critic_run_without_the_trader_ends_at_the_decision_and_that_path_is_finished():
    """It printed "(stopped here)" for every good run, because only `recommended` counted as an end and a Critic run ends `decided` (fixed 2026-11-14)."""
    text = desk.report(run(approver=lambda card: True), 1.0)
    path_line = next(l for l in text.splitlines() if l.startswith("PATH"))
    assert path_line == "PATH     account_manager -> data_clerk -> advisor -> critic -> account_manager_decision" and "(stopped here)" not in text


def test_a_decision_that_waits_for_the_user_is_the_end_of_the_critic_path_not_a_stop():
    state = run()                                                                  # Ask mode, nobody to answer
    assert state["order"].status == "reviewed" and desk.reached_its_end(state) and "(stopped here)" not in desk.report(state, 1.0)


def test_an_order_the_critic_never_reviewed_is_a_path_that_stopped():
    def broken(order):
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")
    state = run(critic=broken)                                                     # the Critic ran as a node, could not start, and left the order `recommended`
    assert state["order"].status == "recommended" and "critic" in state["path"] and not desk.reached_its_end(state)
    assert "(stopped here)" in desk.report(state, 1.0)


def test_when_nothing_was_recommended_the_critic_is_never_asked_and_the_path_is_finished():
    class Nothing(CapturedAdvisor):
        def generate_recommendations(self, order, portfolio_, case=None, today=None, **kw):
            order.status = "recommended"
            return order
    state = desk.run_desk({"symbol": "PG", "fetch": False}, clerk_factory=lambda o: DeskClerk(), advisor_factory=lambda o: Nothing(), resolver=FakeResolver(), with_critic=True, critic_factory=critic_for(),
                          funds_factory=lambda: Manager(profile=profile()), clock=lambda: NOW)
    assert state["path"] == ["account_manager", "data_clerk", "advisor"] and desk.reached_its_end(state) and "(stopped here)" not in desk.report(state, 1.0)


def test_the_command_line_knows_the_new_flags(monkeypatch, capsys):
    assert desk.main(["--show-graph", "--critic"]) == 0
    assert "critic" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        desk.main(["--symbol", "PG", "--critic", "--mode", "manual"])
