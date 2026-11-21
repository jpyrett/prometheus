"""The live run's own pieces (UI v1 spec 4.3), the parts that cost nothing: the three callables the desk is given (`confirm`, `approver`, `clarifier`), and a run on an account that is not there.

The live run itself is PAID (the real agents, OpenRouter and Finnhub) and was driven by hand on a throwaway account: it is not a test (design choices entry 99 says what it showed). Here the real `Run`, the real `Emitter` and the real
`TradeIntent` of a real capture are used; no agent is built, no model is called and no stand-in agent exists.
"""
import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from prometheus.agents.account_manager import missing
from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.application.web import desk_run
from prometheus.application.web.app import create_app
from prometheus.application.web.accounts import Accounts
from prometheus.application.web.emitter import Emitter
from prometheus.application.web.runs import Run, RunManager

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"
INTENT = TradeIntent.model_validate(json.loads((SESSIONS / "bob_jones_oil.json").read_text(encoding="utf-8"))["intent"])
NAMES = json.loads((SESSIONS / "direct_order_names.json").read_text(encoding="utf-8"))               # the real replies of the two models for Jacob's own request (entries 101 and 102)
LIVE_ASK = json.loads((SESSIONS / "d_pyrett_names_ask.json").read_text(encoding="utf-8"))            # the live Ask-mode session of that request: the Data Clerk's rows are on its work order
LOST = "privately held, or not known to be listed"


def missing_card() -> dict:
    """The card the Account Manager gives the approver for Chick-fil-a (`missing.card_for`): the stocks that stayed on the real session's work order, the Data Clerk's real reason, the model's real wording (checked by code)."""
    rows = LIVE_ASK["order"]["data"]["companies"]
    kept = [{"typed": r["brand"], "symbol": r["symbol"], "company": r["company"], "how": r["how"]} for r in rows if r["symbol"]]
    asked = missing.check_wording(missing._question_of(NAMES["wording"]["Chick-fil-a"]["replies"][0]), "Chick-fil-a", 400)
    return missing.card_for("Chick-fil-a", missing.reason_of(next(r for r in rows if r["symbol"] is None)), kept, 1, 1, missing.Wording(text=asked, source="model"), NAMES["request"])


def answering(run: Run, answers: list, seen: list):
    """A thread that answers each question the run asks, in order, with the next of `answers`."""
    def work():
        for answer in answers:
            for _ in range(200):
                waiting = run.open_questions()
                if waiting:
                    seen.append(next(e["data"] for e in run.events if e["type"] == "question" and e["data"]["question_id"] == waiting[0]))
                    run.answer(waiting[0], answer)
                    break
                time.sleep(0.02)
    thread = threading.Thread(target=work)
    thread.start()
    return thread


def test_confirm_asks_in_ask_mode_and_says_yes_in_auto_mode_and_the_stage_waits_while_a_question_is_open():
    auto = Run("auto", "live")
    confirm, _, _ = desk_run.callables(auto, desk_run.Live(Emitter(auto)), 5, auto=True)
    assert confirm(INTENT) is True and [e["type"] for e in auto.events] == ["log"] and auto.events[0]["data"]["text"].startswith("understood: buy the 3 most recommended 'oil' companies")
    ask = Run("ask", "live")
    live = desk_run.Live(Emitter(ask))
    live.running = "account_manager"
    confirm, _, _ = desk_run.callables(ask, live, 5, auto=False)
    seen = []
    thread = answering(ask, [False], seen)
    assert confirm(INTENT) is False
    thread.join(3)
    assert seen[0]["kind"] == "confirm" and seen[0]["text"].startswith("I understood: buy the 3 most recommended") and seen[0]["text"].endswith("Go ahead?")
    states = [e["data"]["state"] for e in ask.events if e["type"] == "stage" and e["data"]["key"] == "work_order"]
    assert states == ["waiting", "busy"]                                         # waiting for you while the question is open, busy again when it closes
    assert [e["data"]["text"] for e in ask.events if e["type"] == "log"][-1] == "no, stop"
    quiet = Run("quiet", "live")
    assert desk_run.callables(quiet, desk_run.Live(Emitter(quiet)), 0.2, auto=False)[0](INTENT) is None            # no answer is never a yes


def test_the_approver_turns_the_desks_card_into_an_approve_question_and_the_clarifier_is_always_available():
    run = Run("cards", "live")
    live = desk_run.Live(Emitter(run))
    live.running = "account_manager_decision"
    _, approver, clarifier = desk_run.callables(run, live, 5, auto=False)
    card = {"text": "  #1  XOM  APPROVE score 86/100 (B)", "symbol": "XOM", "shares": 8, "est_cost": 1364.72, "action": "open", "n": 1, "of": 2, "cash": 15000.0, "proposal": [{"symbol": "XOM", "action": "open", "shares": 8, "est_cost": 1364.72}]}
    seen = []
    thread = answering(run, [True, "20% of my account"], seen)
    assert approver(card) is True
    assert clarifier({"question": "How much would you like to buy?", "n": 1, "of": 3, "facts": ["Cash $15,000.00."]}) == "20% of my account"
    thread.join(3)
    approve, clarify = seen
    assert (approve["kind"], approve["title"], approve["symbol"], approve["shares"], approve["est_cost"], approve["n"], approve["of"], approve["cash"]) == ("approve", "Approve trade 1 of 2", "XOM", 8, 1364.72, 1, 2, 15000.0) and approve["scorecard"] == card["text"]
    assert (clarify["kind"], clarify["text"], clarify["n"], clarify["of"], clarify["facts"]) == ("clarify", "How much would you like to buy?", 1, 3, ["Cash $15,000.00."]) and clarifier.available() is True
    assert [e["data"]["text"] for e in run.events if e["type"] == "log"] == ["approved XOM", "20% of my account"]


def test_a_live_run_on_an_account_that_is_not_there_ends_failed_and_says_which_account():
    manager = RunManager()
    run = manager.start(lambda r: desk_run.run_live(r, {"account_id": 999999999, "prompt": "Buy 2 shares of MSFT", "strict": False, "neutral": True, "fetch": False}, Accounts()), kind="live")
    for _ in range(200):
        if run.status == "finished":
            break
        time.sleep(0.05)
    said, finished = run.events
    assert (said["type"], finished["type"]) == ("log", "run_finished") and finished["data"]["status"] == "failed" and "no account 999999999" in finished["data"]["headline"]
    assert said["data"]["text"] == finished["data"]["headline"] and said["data"]["level"] == "error"      # the feed's last line says how the run ended: it never just stops


def test_a_direct_order_with_a_stock_the_data_clerk_cannot_find_stops_at_the_first_stage_and_the_feed_says_why():
    """Jacob's own run of 2026-11-15, the first live request typed into the page: "Buy as much stock in: F, GM, Procter and gamble and Google. And Chick-fil-a." The Data Clerk found F, GM and PG and answered, in these words,
    "Google: Company Name not found." and "Chick-fil-a: Company Name not found.", so `AccountManager.name_the_stocks` gave those two reasons and the desk ended at its first node WITH a work order. The page then showed
    nothing about it: the feed stopped after "read the request and made the work order" and the stage read done. The intent is the form as the Account Manager understood it (its own sentence in the feed is asserted below);
    no agent is built and no model is called.

    AMENDED for entries 101 and 102: Google is now found (as its listed owner, Alphabet Inc.), and a missing stock is asked about in Ask mode, so the same screen is reached when the user says no to going on without
    Chick-fil-a. The reasons below are the ones the real code gives for that run (`tests/test_direct_order.py`), and the hint is the request to clarify, the same words the terminal session ends with."""
    from prometheus.application.web.emitter import NOT_PLACED, NOT_PLACED_HINT
    from prometheus.common.work_order import new_work_order

    typed = "I have 10000 to spend. Create a new user profile for me: name is D Pyrett. I am 40 with a moderate risk profile. Buy as much stock in: F, GM, Procter and gamble and Google. And Chick-fil-a."
    intent = TradeIntent.model_validate({"request": typed, "action": "order", "budget": 10000.0, "orders": [{"side": "buy", "symbol": "F"}, {"side": "buy", "symbol": "GM"}, {"side": "buy", "company": "Procter and gamble"},
                                                                                                       {"side": "buy", "company": "Google"}, {"side": "buy", "company": "Chick-fil-a"}]})
    assert intent.describe() == ("buy F (you did not say how many shares), buy GM (you did not say how many shares), buy Procter and gamble (you did not say how many shares), buy Google (you did not say how many shares), "
                                 "buy Chick-fil-a (you did not say how many shares), spending $10,000.00 (your own order: no recommendation is made and the Critic is not asked)")      # the line the page showed him
    reasons = [f"Chick-fil-a: {LOST}", "you said no to going on without Chick-fil-a"]
    run = Run("refused", "live")
    out = Emitter(run)
    live = desk_run.Live(out)
    live.path, live.seconds = ["account_manager"], {"account_manager": 21.9}
    final = {"order": new_work_order(typed, requester="desk", action_type="trade"), "plan": {"intent": intent, "direct": True, "action": "order", "request_text": typed}, "errors": reasons, "path": ["account_manager"]}
    desk_run._finish(run, {"account_id": 1}, None, out, live, final, time.perf_counter())       # no account is read: the desk stopped before anything was priced
    stages = {e["data"]["key"]: e["data"]["state"] for e in run.events if e["type"] == "stage"}
    said, finished = run.events[-2:]
    assert stages == {"work_order": "failed"}                                                  # "stopped here" on Create Work Order; the five stages after it were never reached
    assert (finished["type"], finished["data"]["status"], finished["data"]["headline"]) == ("run_finished", "no_trade", NOT_PLACED) and finished["data"]["reasons"] == [*reasons, NOT_PLACED_HINT]
    assert said["type"] == "log" and said["data"]["agent"] == "account_manager" and said["data"]["level"] == "warn" and said["data"]["block"] is True
    assert said["data"]["text"].split("\n") == [NOT_PLACED, f"  - Chick-fil-a: {LOST}", "  - you said no to going on without Chick-fil-a", f"  - {NOT_PLACED_HINT}"]
    assert NOT_PLACED_HINT == "Please clarify your request and send it again: name each stock by its ticker or by the listed company's name."
    assert not [e for e in run.events if e["type"] == "log" and e["data"]["agent"] == "trader"]   # no Trader report for a run the Trader never saw


def test_a_request_that_could_not_be_read_ends_the_run_with_the_reason_and_the_request_to_clarify():
    """The first node's other refusal (entry 102): the Account Manager could not read the request. No work order exists, and the last line of the feed gives the reason and asks the user to say it again."""
    from prometheus.application.web.emitter import NOT_PLACED_HINT

    run = Run("unread", "live")
    out = Emitter(run)
    reason = "I could not understand the request (the model's reply was not usable). Say what you want as a theme, as tickers or as company names, and how many you want to buy from."
    desk_run._finish(run, {"account_id": 1}, None, out, desk_run.Live(out), {"order": None, "plan": None, "errors": [reason], "clarify": True, "path": []}, time.perf_counter())
    said, finished = run.events[-2:]
    assert finished["data"]["status"] == "failed" and finished["data"]["reasons"] == [reason, NOT_PLACED_HINT] and said["data"]["text"].endswith(f"  - {NOT_PLACED_HINT}")
    run = Run("declined", "live")
    out = Emitter(run)
    desk_run._finish(run, {"account_id": 1}, None, out, desk_run.Live(out), {"order": None, "plan": None, "errors": ["stopped: what was understood was not confirmed, so nothing was done"], "path": []}, time.perf_counter())
    assert run.events[-1]["data"]["reasons"] == ["stopped: what was understood was not confirmed, so nothing was done"]         # a no to "is this what you meant" is not a request to clarify


def test_the_answers_route_carries_the_yes_of_a_missing_company_question_back_to_the_desk():
    manager = RunManager()
    http = TestClient(create_app(manager=manager))
    got = []

    def target(run):
        _, approver, _ = desk_run.callables(run, desk_run.Live(Emitter(run)), 5, auto=False)
        got.append(approver(missing_card()))

    run = manager.start(target, kind="live")
    for _ in range(200):
        if run.open_questions():
            break
        time.sleep(0.02)
    question_id = run.open_questions()[0]
    assert http.post(f"/api/runs/{run.id}/answers", json={"question_id": question_id, "answer": "yes"}).status_code == 422             # a yes or a no, never text
    assert http.post(f"/api/runs/{run.id}/answers", json={"question_id": question_id, "answer": True}).status_code == 204
    for _ in range(200):
        if run.status == "finished":
            break
        time.sleep(0.02)
    closed = next(e["data"] for e in run.events if e["type"] == "question_closed")
    assert got == [True] and (closed["answer"], closed["by"]) == (True, "you") and next(e["data"] for e in run.events if e["type"] == "question")["kind"] == "missing_company"


def test_the_live_run_takes_the_box_always_raise_funds_as_the_setting_for_the_run_and_a_missing_value_is_false():
    """Entry 142, R12: `run_live` maps the box with `run_settings` (the one place) and applies it around the whole run (`applied`: set for the run, put back after), exactly as the Neutral box; the run_started event says what it was."""
    import inspect

    source = inspect.getsource(desk_run.run_live)
    assert 'run_settings_module.run_settings(params["strict"], params["neutral"], params.get("raise_funds", False))' in source and "with run_settings_module.applied(chosen):" in source and '"raise_funds": chosen.always_raise_funds' in source


def test_the_approver_turns_the_two_funding_cards_into_their_own_questions_and_their_answers_are_a_yes_a_no_or_nothing():
    """The cash does not cover a purchase (entry 142, R3 and R4): the card of kind `raise_funds` is ONE question in the Account Manager's sentence, the card of kind `funding` is the plan RaiseFunds made (the holdings, their prices and gains, the cash after). No answer is None, never a yes."""
    from decimal import Decimal

    run = Run("funding-cards", "live")
    live = desk_run.Live(Emitter(run))
    live.running = "account_manager_direct"
    _, approver, _ = desk_run.callables(run, live, 5, auto=False)
    ask = {"kind": "raise_funds", "text": "You have $60.14 available; 2 ANT at $88.91 cost $177.82, $117.68 short. Shall I raise the money by selling some of your holdings? I will show you what I would sell first.", "question": "Raise the money by selling some of your holdings? [y/N] ",
           "cash": Decimal("60.14"), "cost": Decimal("177.82"), "shortfall": Decimal("117.68"), "buys": [{"symbol": "ANT", "shares": Decimal(2), "price": Decimal("88.91"), "cost": Decimal("177.82")}]}
    plan = {"kind": "funding", "text": "To buy 2 shares of ANT (about $177.82) with cash of $60.14, these holdings would be sold first:", "symbol": "ANT", "shares": Decimal(2), "est_cost": Decimal("177.82"), "question": "Sell these to pay for 2 shares of ANT (about $178)? [y/N] ",
            "legs": [{"symbol": "DPZ", "shares": Decimal(1), "price": Decimal("348.61"), "amount": Decimal("348.61"), "gain": Decimal("6.89")}], "buys": [{"symbol": "ANT", "shares": Decimal(2), "price": Decimal("88.91"), "cost": Decimal("177.82")}], "cash": Decimal("60.14"), "cash_after": Decimal("230.93")}
    seen = []
    thread = answering(run, [True, False], seen)
    assert approver(ask) is True and approver(plan) is False
    thread.join(3)
    first, second = seen
    assert (first["kind"], first["title"], first["text"], first["shortfall"], first["buys"][0]["symbol"]) == ("raise_funds", "Raise funds?", ask["text"], Decimal("117.68"), "ANT")
    assert (second["kind"], second["title"], second["cash_after"], second["legs"][0]["symbol"], second["legs"][0]["gain"]) == ("funding", "Approve the sales", Decimal("230.93"), "DPZ", Decimal("6.89"))      # the stream writes a Decimal as a JSON number (entry 112)
    assert [e["data"]["text"] for e in run.events if e["type"] == "log"] == ["yes: raise the money by selling holdings", "no: sell nothing"]
    quiet = Run("quiet-funding", "live")
    assert desk_run.callables(quiet, desk_run.Live(Emitter(quiet)), 0.2, auto=False)[1](ask) is None            # no answer is never a yes
