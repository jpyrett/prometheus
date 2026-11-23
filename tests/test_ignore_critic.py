"""IGNORE CRITIC (design choices entry 148; Jacob, 2026-11-23: *"The ability to Ignore (Skip) Critic step ... a new Checkbox: Ignore Critic with defaulted to false"*; the spec is `design/ignore_critic_spec.md`).

REAL DATA ONLY. The recommendations are the Advisor's REAL ones of `tests/fixtures/session_real/bob_jones_oil.json` (XOM, APA, EOG: the real session of 2026-11-14, in which the Critic DENIED APA) and of `auto_oil_no_second_mark.json`, with the Critic's
reviews and blocks taken off, as the order was when the Advisor had finished and no Critic had run. The prices are the stored closes of Friday 2026-11-13 that the REAL Data Clerk quotes (`friday_close`), the Account Manager and the Trader are the real ones
(the Trader without a model), and the web tests run on throwaway Postgres accounts. The graph tests put the Advisor's recorded recommendations on the order where the Advisor would, and the Critic they are given is a tripwire that fails the test if the desk builds it.
"""
import copy
import inspect
import json
import threading
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from handshake_helpers import BOB, MOMENT, PROFILE, close_of, friday_close, manager, trader  # noqa: F401  (friday_close is a fixture)
from test_web_accounts import Starts, http, new_account, over, store  # noqa: F401  (fixtures and the hook of the web tests)

from prometheus.agents.account_manager import decisions, ignore_critic, neutral
from prometheus.application import session
from prometheus.application.web import desk_run
from prometheus.application.web.accounts import Accounts
from prometheus.application.web.app import create_app
from prometheus.application.web.runs import RunManager
from prometheus.application.web.settings import run_settings
from prometheus.common import work_order as wo
from prometheus.common.money import digits
from prometheus.langgraph import desk

pytestmark = pytest.mark.usefixtures("friday_close")

FIXTURES = Path(__file__).parent / "fixtures" / "session_real"
OIL = json.loads((FIXTURES / "bob_jones_oil.json").read_text(encoding="utf-8"))
OIL_FIVE = json.loads((FIXTURES / "auto_oil_no_second_mark.json").read_text(encoding="utf-8"))
DENIED = next(r for r in OIL["order"]["reviews"] if not r["approved"])      # the real Critic's denial of APA: risk above the profile


def advised(capture=OIL) -> wo.WorkOrder:
    """The real work order as it was when the Advisor had finished and the user had ticked Ignore Critic: the Advisor's recommendations and candidates, NO review (the Critic did not run), no `critic` block on a record, nothing decided."""
    order = wo.WorkOrder.model_validate(copy.deepcopy(capture["order"]))
    order.status, order.reviews, order.critic_runs, order.trade_decisions, order.trades, order.trader_runs, order.ignore_critic = "recommended", [], [], [], [], [], True
    for record in order.recommendations:
        record.pop("critic", None)
    return order


def decided(order=None, mode="auto", approver=None, **kw) -> wo.WorkOrder:
    return manager(profile=PROFILE).decide_trades(order or advised(), BOB, approver, MOMENT, mode, **kw)


def told_and_decided(order, mode="auto", **kw):
    """The same decision with a screen for the Account Manager's words and a work order that has a `data` to keep its notes: (the order, the lines it told the user)."""
    am, told = manager(profile=PROFILE), []
    am.screen = told.append
    order.data = wo.DataResult()
    return am.decide_trades(order, BOB, None, MOMENT, mode, **kw), told


def ids(order) -> dict[str, str]:
    return {r["symbol"]: r["identity"]["record_id"] for r in order.recommendations}


# ---------------------------------------------------------------- the graph: the Critic's node is skipped, the decision is not

class RecordedClerk:
    def request_data(self, order, *args, **kw):
        order.status = "data_ready"
        return order


class RecordedAdvisor:
    """Puts the Advisor's REAL recommendations of the recorded session on the order, as `Advisor.generate_recommendations` does (the Critic's blocks are not on them: no Critic has run)."""

    def generate_recommendations(self, order, case=None, today=None, **kw):
        recorded = advised()
        order.recommendations, order.candidates, order.status = recorded.recommendations, recorded.candidates, "recommended"
        return order

    def advise(self, order):      # the desk without the Critic calls the Advisor's version 0
        return self.generate_recommendations(order)


def tripwire(order):
    raise AssertionError("the Critic was built: with Ignore Critic on, the desk must not run its node")


def broken_critic(order):
    raise RuntimeError("the test builds no Critic")      # the critic node reports it and the desk goes on: it shows the node RAN


def stream_of(ignore: bool, **options):
    request = {"symbol": "XOM", "fetch": False, "account": {"portfolio": BOB, "profile": PROFILE}, **({"ignore_critic": True} if ignore else {})}
    graph = desk.build_desk(clerk_factory=lambda order: RecordedClerk(), advisor_factory=lambda order: RecordedAdvisor(), with_critic=True, critic_factory=options.pop("critic_factory"), funds_factory=lambda: manager(profile=PROFILE),
                            clock=lambda: MOMENT, **options)
    state0 = {"request": request, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": False, "plan": None}
    started, final = [], None
    for mode, chunk in graph.stream(state0, stream_mode=["tasks", "values"]):
        if mode == "tasks" and "result" not in chunk and "error" not in chunk:
            started.append(chunk["name"])
        elif mode == "values":
            final = chunk
    return started, final


def test_with_the_box_on_the_critic_node_never_runs_and_the_decision_does_and_off_the_route_is_the_old_one():
    started, final = stream_of(True, critic_factory=tripwire)      # the tripwire raises if the Critic is built at all
    assert started == ["account_manager", "data_clerk", "advisor", "account_manager_decision"] and final["path"] == started and final["errors"] == []
    order = final["order"]
    assert order.ignore_critic is True and order.reviews == [] and order.status == "decided" and desk.reached_its_end(final)
    assert [(d["symbol"], d["will_trade"], d["origin"]) for d in order.trade_decisions] == [("XOM", "yes_auto", "advisor"), ("APA", "yes_auto", "advisor"), ("EOG", "yes_auto", "advisor")]
    off_started, off_final = stream_of(False, critic_factory=broken_critic)      # the same request with the box off goes to the Critic's node first, as it always did
    assert off_started == ["account_manager", "data_clerk", "advisor", "critic", "account_manager_decision"]
    assert off_final["order"].ignore_critic is False and any("the Critic could not start" in e for e in off_final["order"].errors) and off_final["order"].trade_decisions == []


def test_the_critic_node_stays_in_the_graph_and_a_desk_without_the_critic_ignores_the_flag():
    graph = desk.build_desk(clerk_factory=lambda order: None, advisor_factory=lambda order: None, with_critic=True).get_graph()
    edges = {(e.source, e.target, bool(e.conditional)) for e in graph.edges}
    assert "critic" in graph.nodes and ("advisor", "critic", True) in edges and ("advisor", "account_manager_decision", True) in edges and ("critic", "account_manager_decision", False) in edges
    plain = desk.build_desk(clerk_factory=lambda order: RecordedClerk(), advisor_factory=lambda order: RecordedAdvisor(), resolver=None)
    final = plain.invoke({"request": {"symbol": "XOM", "fetch": False, "ignore_critic": True}, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": False, "plan": None})
    assert final["path"] == ["account_manager", "data_clerk", "advisor"] and final["order"].ignore_critic is False and final["order"].trade_decisions == []


def test_one_place_reads_the_flag_and_only_two_nodes_ask_it_so_a_direct_order_and_a_research_only_request_do_not_change():
    source = inspect.getsource(desk.build_desk)
    assert 'return with_critic and bool(state["request"].get("ignore_critic"))' in source      # one place reads the flag, and only on the desk that has the Critic
    assert source.count("ignores_the_critic(state)") == 2      # `after_advisor` (the route) and `account_manager_decision` (the order's flag): a direct order passes through neither of them
    after = source[source.index("def after_advisor"):source.index("def permitted")]
    assert after.index('== "research"') < after.index("ignores_the_critic(state)")      # "research only" still ends after the Advisor, before the flag is looked at


# ---------------------------------------------------------------- the decisions: Auto trades, Ask asks

def test_auto_trades_every_recommendation_including_the_one_the_critic_denied_round_robin_in_decimal_money():
    order = decided()
    assert order.status == "decided" and order.errors == []
    rows = [(d["symbol"], d["will_trade"], d["origin"], d["review_id"], d["shares"], d["est_cost"], d["price"], d["valid_until_utc"]) for d in order.trade_decisions]
    assert rows == [("XOM", "yes_auto", "advisor", None, Decimal(40), Decimal("6823.60"), close_of("XOM"), "2026-11-13T21:00:00Z"),
                    ("APA", "yes_auto", "advisor", None, Decimal(41), Decimal("1986.04"), close_of("APA"), "2026-11-13T21:00:00Z"),      # the Critic denied APA in the real session: no Critic ran here, so it counts as approved
                    ("EOG", "yes_auto", "advisor", None, Decimal(39), Decimal("6174.48"), close_of("EOG"), "2026-11-13T21:00:00Z")]
    assert all(d["size"]["allocation"] == "round_robin" and d["size"]["pot"] == Decimal("15000.00") and d["record_id"] == ids(order)[d["symbol"]] for d in order.trade_decisions)
    assert sum(d["est_cost"] for d in order.trade_decisions) == Decimal("14984.12") and "Ignore Critic" in order.trade_decisions[0]["reason"] and "score 73" in order.trade_decisions[0]["reason"]
    assert all(wo.verify_trade(order, d["record_id"], MOMENT) == [] for d in order.trade_decisions)


def test_the_split_evenly_words_are_kept_and_the_number_asked_for_cuts_the_purchases():
    even = decided(balance="even")
    assert {d["size"]["allocation"] for d in even.trade_decisions} == {"even"} and all(d["balance"]["mode"] == "even" for d in even.trade_decisions)
    two, told = told_and_decided(advised(), count=2)
    assert [d["symbol"] for d in two.trade_decisions] == ["XOM", "APA"]      # the best ranked first: the third gets no decision, and the user is told
    assert len(told) == 1 and told[0].lstrip().startswith("EOG was recommended by the Advisor (score 72) and counts as approved, but it is not needed: 2 companies were asked for")
    assert two.data.notes == [ignore_critic.NOT_NEEDED + told[0].strip()]


def test_ask_asks_about_each_and_the_card_says_the_critic_did_not_review_it():
    cards = []
    order = decided(mode="ask", approver=lambda card: cards.append(card) or True)
    assert [(d["symbol"], d["will_trade"], d["origin"], d["asked_user"]) for d in order.trade_decisions] == [("XOM", "yes_user", "advisor", True), ("APA", "yes_user", "advisor", True), ("EOG", "yes_user", "advisor", True)]
    assert [(c["symbol"], c["n"], c["of"], c["shares"]) for c in cards] == [("XOM", 1, 3, 40), ("APA", 2, 3, 41), ("EOG", 3, 3, 39)] and len(cards[0]["proposal"]) == 3 and cards[0]["cash"] == Decimal("15000")
    text = cards[0]["text"]
    assert "THE CRITIC DID NOT REVIEW THIS RECOMMENDATION (Ignore Critic is on)" in text and "Advisor score 73" in text and "Buy XOM using a trend strategy" in text and "buy 40 shares at $170.59 = about $6,823.60" in text and cards[0]["critic_skipped"] is True


def test_ask_with_an_approver_that_says_no_is_no_user_denied_and_with_nobody_to_ask_it_stays_pending():
    no = decided(mode="ask", approver=lambda card: False)
    assert [d["will_trade"] for d in no.trade_decisions] == ["no_user_denied"] * 3 and no.status == "decided"
    nobody = decided(mode="ask", approver=None)
    assert [(d["will_trade"], d["asked_user"]) for d in nobody.trade_decisions] == [("pending_user", False)] * 3 and nobody.status == "reviewed"      # no answer is never a yes
    assert all(wo.verify_trade(nobody, d["record_id"], MOMENT) for d in nobody.trade_decisions)


def test_a_recommendation_with_no_current_price_is_not_bought_and_a_record_that_is_not_a_purchase_gets_no_purchase(monkeypatch):
    order = advised()
    order.recommendations[1]["action"]["action"] = "reduce"      # an old record about a holding (entry 105): never a purchase
    out = decided(order)
    by = {d["symbol"]: d for d in out.trade_decisions}
    assert (by["XOM"]["will_trade"], by["EOG"]["will_trade"]) == ("yes_auto", "yes_auto") and by["APA"]["will_trade"] == "no_funds" and by["APA"]["shares"] == 0 and "not a purchase" in by["APA"]["reason"]
    assert wo.verify_trade(out, by["APA"]["record_id"], MOMENT)
    blind = manager(profile=PROFILE)
    monkeypatch.setattr(blind, "request_prices", lambda symbols: (_ for _ in ()).throw(RuntimeError("no quotes")))
    unpriced = blind.decide_trades(advised(), BOB, None, MOMENT, "auto")
    assert [d["will_trade"] for d in unpriced.trade_decisions] == ["no_system_denied"] * 3 and all("no current price" in d["reason"] for d in unpriced.trade_decisions) and any("could not be had" in e for e in unpriced.errors)


def test_the_price_limit_drops_a_purchase_outside_it_and_the_pot_is_the_money():
    out, told = told_and_decided(advised(), max_price=Decimal("100"))
    assert [d["symbol"] for d in out.trade_decisions] == ["APA"] and out.trade_decisions[0]["max_price"] == Decimal("100") and out.trade_decisions[0]["shares"] == Decimal(309)      # XOM and EOG cost more than $100: no decision, the same notes
    assert [line.strip() for line in told] == ["XOM is at $170.59 a share, above your limit of $100.00: it gets no decision and is not bought", "EOG is at $158.32 a share, above your limit of $100.00: it gets no decision and is not bought"]
    assert [n.startswith(decisions.OVER_LIMIT) for n in out.data.notes] == [True, True]
    floor = decided(min_price=Decimal("100"))
    assert [d["symbol"] for d in floor.trade_decisions] == ["XOM", "EOG"] and all(d["min_price"] == Decimal("100") for d in floor.trade_decisions)
    budget = decided(budget=Decimal("3000"))
    assert sum(d["est_cost"] for d in budget.trade_decisions) <= Decimal("3000") and {d["size"]["pot"] for d in budget.trade_decisions} == {Decimal("3000.00")}


# ---------------------------------------------------------------- the Advisor's first, then the neutral ones: the number asked for is kept

def test_count_three_with_two_recommendations_and_two_neutral_companies_buys_the_two_first_and_one_neutral(monkeypatch):
    monkeypatch.setenv("ALLOW_NEUTRAL_RECOMMENDATIONS", "true")
    order = advised(OIL_FIVE)
    order.recommendations = [r for r in order.recommendations if r["symbol"] in ("EOG", "CVX")]      # two of the three the real Advisor made (COP is left out)
    scored_neutral = [c for c in order.candidates if c.get("stance") == "neutral" and c.get("gate") is None]
    order.candidates = [c for c in order.candidates if c in scored_neutral[:2]]
    assert [c["symbol"] for c in order.candidates] == ["CNQ", "OXY"]
    out, told = told_and_decided(order, count=3)
    assert [(d["symbol"], d["origin"]) for d in out.trade_decisions] == [("EOG", "advisor"), ("CVX", "advisor"), ("CNQ", "neutral")]      # OXY is beyond the number: no decision
    assert [d["size"]["allocation"] for d in out.trade_decisions] == ["round_robin"] * 3 and all(d["will_trade"] == "yes_auto" for d in out.trade_decisions)
    assert len(told) == 1 and told[0].lstrip().startswith("OXY was scored neutral by the Advisor") and out.data.notes == [neutral.NOT_NEEDED + told[0].strip()]
    cut, _ = told_and_decided(advised(OIL_FIVE), count=2)
    assert [d["symbol"] for d in cut.trade_decisions] == ["EOG", "CVX"]      # three recommended, two asked for: the best ranked two, COP gets no decision


# ---------------------------------------------------------------- what the Trader may do: the word alone permits nothing (IC8)

def test_verify_trade_accepts_an_advisor_decision_only_under_the_conditions_of_ic8():
    order = decided()
    apa = ids(order)["APA"]
    assert wo.verify_trade(order, apa, MOMENT) == []
    off = copy.deepcopy(order)
    off.ignore_critic = False
    assert "does not say the user chose to ignore the Critic" in "; ".join(wo.verify_trade(off, apa, MOMENT))
    reviewed = copy.deepcopy(order)
    reviewed.reviews = [DENIED]      # the Critic's own denial of APA: a denial is never relabelled
    assert "the Critic's verdict stands" in "; ".join(wo.verify_trade(reviewed, apa, MOMENT))
    approved = copy.deepcopy(order)
    approved.reviews = [{**DENIED, "approved": True}]
    assert "has no Critic review" in "; ".join(wo.verify_trade(approved, apa, MOMENT))
    gone = copy.deepcopy(order)
    gone.recommendations = [r for r in gone.recommendations if r["symbol"] != "APA"]
    assert "holds no recommendation of the Advisor's for 'APA'" in "; ".join(wo.verify_trade(gone, apa, MOMENT))
    sold = copy.deepcopy(order)
    next(r for r in sold.recommendations if r["symbol"] == "APA")["action"]["action"] = "reduce"
    assert "not a purchase to open" in "; ".join(wo.verify_trade(sold, apa, MOMENT))
    sale = copy.deepcopy(order)
    next(d for d in sale.trade_decisions if d["symbol"] == "APA")["action"] = "reduce"
    assert "is only a purchase" in "; ".join(wo.verify_trade(sale, apa, MOMENT))
    late = wo.verify_trade(order, apa, MOMENT + timedelta(minutes=31))
    assert len(late) == 1 and late[0].startswith(wo.EXPIRY_PREFIX)      # the approval of the decision itself ran out: the only reason
    pending = copy.deepcopy(order)
    next(d for d in pending.trade_decisions if d["symbol"] == "APA")["will_trade"] = "pending_user"
    assert "not one of yes_auto, yes_user" in "; ".join(wo.verify_trade(pending, apa, MOMENT))
    other = copy.deepcopy(order)
    next(d for d in other.trade_decisions if d["symbol"] == "APA")["symbol"] = "XOM"
    assert "holds no recommendation of the Advisor's for 'XOM'" in "; ".join(wo.verify_trade(other, apa, MOMENT))


def test_the_trader_places_every_advisor_purchase_through_the_handshake_and_the_run_never_spends_more_than_its_money():
    am = manager(profile=PROFILE)
    order = am.decide_trades(advised(), BOB, None, MOMENT, "auto")
    am.perform_trades(order, BOB, trader(), MOMENT)
    filled = {t["symbol"]: t for t in order.trades if t["status"] == "filled"}
    assert {s: t["fill"]["shares"] for s, t in filled.items()} == {"XOM": Decimal(40), "APA": Decimal(41), "EOG": Decimal(39)}      # each at the quote, nothing past the decision
    spent = sum((t["fill"]["amount"] for t in filled.values()), Decimal("0.00"))
    assert spent == Decimal("14984.12") <= Decimal("15000.00") and wo.verify_trades(order) == [] and order.status == "traded"
    other = manager(profile=PROFILE)
    refused = other.decide_trades(advised(), BOB, None, MOMENT, "auto")
    refused.ignore_critic = False      # the order no longer says the user chose it: the Trader gets no permission for any of them
    other.perform_trades(refused, BOB, trader(), MOMENT)
    assert [t["status"] for t in refused.trades] == ["rejected"] * 3 and {t["reject_code"] for t in refused.trades} == {"not_permitted"}


def test_with_the_box_off_nothing_changes_an_order_with_no_review_is_still_not_decided():
    order = advised()
    order.ignore_critic = False
    out = decided(order)
    assert out.trade_decisions == [] and any("decides on an order the Critic reviewed (status recommended)" in e for e in out.errors)
    old = wo.WorkOrder.model_validate(copy.deepcopy(OIL["order"]))      # a recorded order made before this feature loads with the flag off
    assert old.ignore_critic is False


# ---------------------------------------------------------------- the settings, the page's bridge, the report

def test_strict_cannot_go_with_ignore_critic_and_the_neutral_box_keeps_its_meaning():
    on = run_settings(strict=True, neutral=True, raise_funds=False, ignore_critic=True)
    assert (on.strict, on.ignore_critic, on.allow_neutral) == (False, True, True)
    assert run_settings(strict=True, neutral=True).strict is True and run_settings(strict=True, neutral=True).allow_neutral is False and run_settings(strict=False, neutral=True).ignore_critic is False
    assert run_settings(strict=False, neutral=False, ignore_critic=True).allow_neutral is False


def test_the_script_and_the_session_take_the_flag_and_without_session_it_is_refused():
    import os
    import shlex
    import subprocess

    repo = Path(__file__).resolve().parents[1]

    def script(*args):
        return subprocess.run(["bash", str(repo / "test_prometheus.sh"), "--skip-inspect", *args], input="", capture_output=True, text=True, cwd=repo, env={**os.environ, "PROMETHEUS_DRY_RUN": "1"}, timeout=30)

    person = ("--name", "Bob Jones", "--age", "60", "--risk", "conservative", "--cash", "15000", "--mode", "auto")
    on, off = script("--session", *person, "--prompt", "buy the 3 most recommended oil stocks", "--no-fetch", "--ignore-critic"), script("--session", *person, "--prompt", "buy the 3 most recommended oil stocks", "--no-fetch")
    line = lambda result: shlex.split(next(x for x in result.stdout.splitlines() if x.startswith("DRY RUN:")).removeprefix("DRY RUN:"))
    assert on.returncode == 0 and "--ignore-critic" in line(on) and off.returncode == 0 and "--ignore-critic" not in line(off)
    assert script("--symbol", "MSFT", "--no-fetch", "--ignore-critic").returncode == 2
    args = session.build_parser().parse_args(["--account", "1", "--prompt", "x", "--ignore-critic"])
    assert args.ignore_critic is True and session.build_parser().parse_args(["--account", "1", "--prompt", "x"]).ignore_critic is False


def test_the_report_says_the_critic_did_not_review_before_the_decisions():
    order = decided()
    lines = session.decision_lines(order, "auto")
    assert lines[0].strip() == "IGNORE CRITIC: the Critic did not review these recommendations (the user's setting)" and lines[1].lstrip().startswith("XOM") and "3 approved trades go on to the Trader" in lines[-1]
    assert ignore_critic.advisor_lines(order)[0] == ignore_critic.SKIPPED_LINE and "BUY (Critic skipped, Auto): 40 shares at $170.59" in ignore_critic.advisor_lines(order)[1]
    assert [s for s, _ in session.sized_purchases(order)] == ["XOM", "APA", "EOG"]
    reviewed = wo.WorkOrder.model_validate(copy.deepcopy(OIL["order"]))
    assert not any("IGNORE CRITIC" in line for line in session.decision_lines(reviewed, "auto")) and ignore_critic.advisor_lines(reviewed) == []


def test_the_work_order_view_shows_the_decisions_as_critic_skipped():
    from prometheus.application.web import views

    out = decided()
    view = views.work_order_view(out)
    xom = next(row for row in view["rows"] if row["symbol"] == "XOM")
    assert xom["critic"] is None and xom["decision"]["origin"] == "advisor" and xom["decision"]["words"] == "yes (AUTO) · Critic skipped" and "Critic skipped" in xom["detail"]["message"]
    reviewed = views.work_order_view(wo.WorkOrder.model_validate(copy.deepcopy(OIL["order"])))
    assert [row["critic"] is not None for row in reviewed["rows"] if row["group"] == "recommended"] == [True] * 3      # a run with the Critic is shown as it always was
    assert all("message" not in row["detail"] and "Critic skipped" not in row["decision"]["words"] for row in reviewed["rows"] if row["group"] == "recommended")


def test_a_run_started_event_and_the_run_request_carry_the_box_and_a_missing_value_is_false(http, store):
    seen = Starts(store)
    web = TestClient(create_app(live_target=seen))
    account_id = web.post("/api/accounts", json={"name": "Ignore Critic Holder", "age": 40, "risk": "moderate", "mode": "ask", "cash": "1000.00"}).json()["account"]["id"]
    try:
        run = {"account_id": account_id, "prompt": "Buy 2 shares of MSFT"}
        assert web.post("/api/runs", json=run).status_code == 202
        over(web)
        assert seen.seen[-1]["params"]["ignore_critic"] is False and web.get("/api/health").json()["defaults"]["ignore_critic"] is False
        assert web.post("/api/runs", json={**run, "ignore_critic": True, "strict": True}).status_code == 202
        over(web)
        assert (seen.seen[-1]["params"]["ignore_critic"], seen.seen[-1]["params"]["strict"]) == (True, False)      # Strict cannot go with it (IC9): the server enforces it too
        assert web.post("/api/runs", json={**run, "strict": True}).status_code == 202
        over(web)
        assert (seen.seen[-1]["params"]["ignore_critic"], seen.seen[-1]["params"]["strict"]) == (False, True)
    finally:
        store.delete_account(account_id)


def started_options(store, **box) -> dict:
    """The `options` of the `run_started` event of the real `run_live` on a throwaway account, stopped by the agents that cannot be built (nothing is called)."""
    holder = Accounts()
    account = holder.create({"name": "Ignore Critic Options", "age": 40, "risk": "moderate", "mode": "ask", "cash": "1000.00"})["account"]["id"]
    try:
        def no_agents(args):
            raise RuntimeError("the test builds no agents")

        runs = RunManager()
        params = {"account_id": account, "prompt": "Buy 2 shares of MSFT", "strict": False, "neutral": True, "fetch": False, **box}
        run = runs.start(lambda r: desk_run.run_live(r, params, holder, parts_builder=no_agents), kind="live")
        for _ in range(300):
            if run.status == "finished":
                break
            threading.Event().wait(0.05)
        return next(e["data"] for e in run.events if e["type"] == "run_started")["options"]
    finally:
        store.delete_account(account)


def test_the_options_of_a_live_run_say_whether_the_critic_was_skipped(store):
    assert started_options(store) == {"strict": False, "neutral": True, "fetch": False, "raise_funds": False, "ignore_critic": False}
    assert started_options(store, ignore_critic=True, strict=True) == {"strict": False, "neutral": True, "fetch": False, "raise_funds": False, "ignore_critic": True}


# ---------------------------------------------------------------- the real session: Auto mode, Ignore Critic on, a throwaway account (2026-11-23)

SESSION = json.loads((FIXTURES / "auto_ignore_critic.json").read_text(encoding="utf-8"))


def test_the_real_auto_session_with_the_critic_skipped_has_no_review_and_its_trades_are_within_the_pot():
    """`python -m prometheus.application.session --mode auto --ignore-critic --cleanup --yes --prompt "I want to buy oil stocks. Buy the most amount of oil stocks shares from the 3 most recommended companies."` on a throwaway account with $15,000.00 (the real
    Account Manager, Data Clerk, Advisor and Trader, the real Postgres book, the quotes of Monday 2026-11-23 00:20 ET; no fetch). The Critic's node never ran: the path has no `critic`, the work order has no review and no Critic run, and says it was told to ignore the Critic.
    The Advisor recommended EOG and CVX; the Account Manager's third place of the 3 asked for went to the best neutral company (ALLOW_NEUTRAL_RECOMMENDATIONS was on in the server's environment), COP, as entry 107 has it; all three were bought round-robin."""
    order = SESSION["order"]
    assert SESSION["path"] == ["account_manager", "data_clerk", "advisor", "account_manager_decision", "trader"] and "critic" not in SESSION["path"]
    assert order["ignore_critic"] is True and order["reviews"] == [] and order["critic_runs"] == [] and order["status"] == "traded"
    assert all(r["critic"] is None and r["action"]["action"] == "open" for r in order["recommendations"]) and [r["symbol"] for r in order["recommendations"]] == ["EOG", "CVX"]
    assert [(d["symbol"], d["origin"], d["will_trade"], d["review_id"], d["shares"]) for d in order["trade_decisions"]] == [("EOG", "advisor", "yes_auto", None, 31), ("CVX", "advisor", "yes_auto", None, 30), ("COP", "neutral", "yes_auto", None, 30)]
    assert all(d["record_id"] == r["identity"]["record_id"] for d, r in zip(order["trade_decisions"], order["recommendations"])) and "Ignore Critic" in order["trade_decisions"][0]["reason"]
    pot = digits(order["trade_decisions"][0]["size"]["pot"])
    spent = sum((digits(t["fill"]["amount"]) for t in order["trades"] if t["status"] == "filled"), Decimal("0.00"))
    assert pot == Decimal("15000") and spent == Decimal("14941.30") <= pot and digits(order["trader_runs"][-1]["portfolio_after"]["cash"]) == Decimal("58.70")      # a run never spends more than its money
    assert [(t["symbol"], t["status"], t["fill"]["shares"]) for t in order["trades"]] == [("EOG", "filled", 31), ("CVX", "filled", 30), ("COP", "filled", 30)]
    loaded = wo.WorkOrder.model_validate(copy.deepcopy(order))
    assert wo.verify_trades(loaded) == []      # every fill agrees with its decision, and the pot holds
    inside = wo.parse_utc("2026-11-23T05:30:00Z")
    assert [wo.verify_trade(loaded, d["record_id"], inside) for d in loaded.trade_decisions] == [[], [], []]      # the permission each decision was traded under still holds at the time it was made
    loaded.ignore_critic = False
    assert all(wo.verify_trade(loaded, d["record_id"], inside) for d in loaded.trade_decisions[:2])      # without the order's own flag the two Advisor purchases are not permitted (the neutral one is the neutral rule's)


def test_the_replay_of_the_real_session_says_the_critic_was_skipped_in_its_options_its_stage_and_its_rows():
    from test_web_ui import client, play

    events = play(client("auto_ignore_critic"))
    started = next(data for _, kind, data in events if kind == "run_started")
    assert started["options"] == {"strict": False, "neutral": True, "fetch": False, "ignore_critic": True}
    last = {}
    for _, kind, data in events:
        if kind == "stage":
            last[data["key"]] = data["state"]
    assert last["critique"] == "skipped" and last["prepare"] == "done" and last["trade"] == "done"
    rows = [(r["symbol"], r["group"], r["critic"], r["decision"]["words"]) for r in [d for _, kind, d in events if kind == "work_order"][-1]["rows"][:3]]
    assert rows == [("EOG", "recommended", None, "yes (AUTO) · Critic skipped"), ("CVX", "recommended", None, "yes (AUTO) · Critic skipped"), ("COP", "neutral", None, "yes (AUTO)")]
