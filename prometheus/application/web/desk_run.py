"""The live run (UI v1 spec 4.3): what `session.run_session` does, with emitters where it prints.

    run_live(run, params, accounts, question_timeout)         # params: {account_id, prompt, strict, neutral, fetch, raise_funds, ignore_critic}; runs in the run's worker thread

THIS IS A PAID RUN (OpenRouter and Finnhub; with `fetch` also the paid data APIs, which write the data files) that writes the UI's own account in Postgres. It builds the REAL parts (`session.build_parts`) fresh for every
run, the real LangGraph desk (`build_desk`, as the session does, with the Critic and the Trader) and streams it with `stream_mode=["tasks", "values"]`: `tasks` says a node started and ended, `values` is the state after each
node (the work order, the path, the seconds). `desk.py` and `session.py` are not changed: this module imports their pieces and adds no rule about what may trade.

The three callables the desk is given turn its questions into cards on the page: `confirm` (what was understood: asked in Ask mode, a yes in Auto mode, as the CLI's `--yes`), `approver` (each trade in Ask mode, and, in Ask mode, whether to go on without a stock the Data Clerk did not find: a card of kind `missing_company`, entry 102; whether to raise funds, a card of kind `raise_funds`, and the plan RaiseFunds made, a card of kind `funding`, entry 142) and `clarifier` (how much, in Ask mode). No answer is never a yes: it is None, exactly what the terminal versions give when nobody is there.
"""
import threading
import time

from prometheus.agents.account_manager.direct import data_clerk_lines, direct_section
from prometheus.agents.account_manager.missing import kept_line
from prometheus.agents.account_manager.neutral import neutral_lines
from prometheus.application import session
from prometheus.application.web import settings as run_settings_module
from prometheus.application.web import tracing, views
from prometheus.application.web.emitter import AGENT_OF_NODE, NOT_PLACED, NOT_PLACED_HINT, Emitter, node_line, utc_now
from prometheus.application.web.runs import Run
from prometheus.common.trading_clock import clock_note
from prometheus.langgraph.desk import build_desk
from prometheus.proving import sessions as proving_sessions

POLL_SECONDS = 1.0
COUNT_MODEL_CALLS = False      # the callbacks of `graph.stream` reached the Account Manager's, the Data Clerk's and the Trader's model calls in the first live run, and not the Advisor's or the Critic's: a total would be wrong, so the page shows none (spec 7)
NO_RUN_STATE = {"request": None, "order": None, "errors": [], "resolution": None, "funds": None, "path": [], "seconds": {}, "with_trader": True, "plan": None}


class Live:
    """Where the run is, for the six stages: the nodes that finished and their seconds, the node running now, whether a question is open, the nodes that raised, and the order so far."""

    def __init__(self, out: Emitter):
        self.out = out
        self.path, self.seconds, self.running, self.waiting, self.failed = [], {}, None, False, set()
        self.order, self.plan, self.request_errors = None, None, []
        self.started = {}

    def refresh(self) -> None:
        self.out.stages(views.stage_states(path=self.path, seconds=self.seconds, running=self.running, waiting=self.waiting, failed=self.failed, order=self.order, request_errors=self.request_errors))


def _understood(plan: dict | None) -> str | None:
    return plan["intent"].describe() if plan and plan.get("intent") is not None else None


def _is_news(plan: dict | None) -> bool:
    """Whether the run is a request that only asked for the news (entry 149): the desk made a news plan."""
    return bool(plan) and plan.get("action") == "news"


def _is_quote(plan: dict | None) -> bool:
    """Whether the run is a request that only asked for the current price (entry 150): the desk made a quote plan."""
    return bool(plan) and plan.get("action") == "quote"


def _top(plan: dict | None) -> int | None:
    """The number of companies a request for recommendations asked for (the desk's `find`: the request's count, else the default), so the work order lists that many (entry 144). None for a direct order."""
    return plan.get("find") if plan and not plan.get("direct") else None


def missing_payload(card: dict) -> dict:
    """The question the page is asked for a stock the Data Clerk did not find (the card of `missing.card_for`): the Account Manager's own sentence as `text` and `question`, what the user typed and the Data Clerk's reason, which
    question of how many this is, and, as the monospace block, what stays on the order. The page's buttons are "OK, continue without it" (True) and "Stop the order" (False)."""
    kept = card.get("kept") or []
    sentence = card.get("sentence") or card.get("question") or ""
    return {"title": f"Continue without {card.get('typed')}?", "text": sentence, "question": sentence, "typed": card.get("typed"), "reason": card.get("reason"), "n": card.get("n"), "of": card.get("of"),
            "facts": ["Still on your order:", *[f"  {kept_line(stock)}" for stock in kept]], "kept": kept, "wording": card.get("wording")}


def raise_funds_payload(card: dict) -> dict:
    """The question the page is asked when the cash does not cover what the user typed and nothing permits selling holdings (the card of `funding.ask_card`, entry 142, R3): the Account Manager's own sentence as `text` and `question`, the cash, what the purchases cost, how short it is, and the purchases."""
    return {"title": "Raise funds?", "text": card.get("text", ""), "question": card.get("text", ""), "cash": card.get("cash"), "cost": card.get("cost"), "shortfall": card.get("shortfall"), "buys": card.get("buys")}


def funding_payload(card: dict) -> dict:
    """The plan RaiseFunds made, shown before anything is sold (the Trader's card of kind `funding`, entry 142, R4): which holdings, how many shares, at what price, the gain or loss on each, the purchases they pay for, the cash before and after."""
    return {"title": "Approve the sales", "text": card.get("text", ""), "question": card.get("question"), "symbol": card.get("symbol"), "shares": card.get("shares"), "est_cost": card.get("est_cost"), "cash": card.get("cash"),
            "cash_after": card.get("cash_after"), "legs": card.get("legs"), "buys": card.get("buys")}


def callables(run: Run, live: Live, timeout: float, auto: bool):
    """The three callables of the desk: `confirm`, `approver` and `clarifier`. While a question is open the busy stage is `waiting`; when it closes it is `busy` again."""
    out = live.out

    def ask(kind: str, payload: dict):
        live.waiting = True
        live.refresh()
        try:
            return run.ask(kind, payload, timeout)
        finally:
            live.waiting = False
            live.refresh()

    def confirm(intent) -> bool | None:
        out.log("account_manager", f"understood: {intent.describe()}")
        if auto:
            return True
        answer = ask("confirm", {"title": "Confirm what was understood", "text": f"I understood: {intent.describe()}. Go ahead?"})
        out.log("you", "yes, go ahead" if answer is True else "no, stop" if answer is False else "(no answer: not going ahead)", "info" if answer is True else "warn")
        return answer if isinstance(answer, bool) else None

    def approver(card: dict) -> bool | None:
        n, of = card.get("n"), card.get("of")
        if card.get("kind") == "missing_company":       # a stock the Data Clerk did not find (entry 102): "OK to continue without it?", worded by the Account Manager's model and checked by code
            typed = card.get("typed")
            answer = ask("missing_company", missing_payload(card))
            out.log("you", f"continue without {typed}" if answer is True else f"stop the order: no to going on without {typed}" if answer is False else f"(no answer about {typed}: the order is not placed)", "info" if answer is True else "warn")
            return answer if isinstance(answer, bool) else None
        if card.get("kind") == "raise_funds":       # the cash does not cover what the user typed and nothing permits selling (Ask mode, entry 142, R3): ONE question for all the purchases
            answer = ask("raise_funds", raise_funds_payload(card))
            out.log("you", "yes: raise the money by selling holdings" if answer is True else "no: sell nothing" if answer is False else "(no answer: nothing is sold)", "info" if answer is True else "warn")
            return answer if isinstance(answer, bool) else None
        if card.get("kind") == "funding":       # the plan RaiseFunds made, shown before anything is sold (entry 142, R4): which holdings, how many shares, at what price, the gain or loss, the cash after
            answer = ask("funding", funding_payload(card))
            out.log("you", "approved the sales" if answer is True else "no: sell nothing" if answer is False else "(no answer: nothing is sold)", "info" if answer is True else "warn")
            return answer if isinstance(answer, bool) else None
        title = f"Approve trade {n} of {of}" if n and of and of > 1 else "Approve this trade"
        answer = ask("approve", {"title": title, "text": card.get("text", ""), "scorecard": card.get("text", ""), "symbol": card.get("symbol"), "shares": card.get("shares"), "est_cost": card.get("est_cost"), "dollars": card.get("dollars"), "price": card.get("price"),
                                 "action": card.get("action"), "n": n, "of": of, "cash": card.get("cash"), "proposal": card.get("proposal"), "question": card.get("question")})
        what = f"{card.get('symbol')}"
        out.log("you", f"approved {what}" if answer is True else f"denied {what}" if answer is False else f"(no answer to {what}: not traded)", "info" if answer is True else "warn")
        return answer if isinstance(answer, bool) else None

    def clarifier(card: dict) -> str | None:
        answer = ask("clarify", {"title": "How much?", "text": card["question"], "question": card["question"], "n": card.get("n"), "of": card.get("of"), "facts": card.get("facts") or []})
        out.log("you", answer if isinstance(answer, str) and answer.strip() else "(no answer)", "info" if answer else "warn")
        return answer if isinstance(answer, str) and answer.strip() else None

    clarifier.available = lambda: True          # a person is at the screen
    return confirm, approver, clarifier


class Poller(threading.Thread):
    """While the Trader works: read the account from the book about once a second and emit it when it changed, so each fill lands in the table as it is booked (spec 4.3 step 5). Best effort: it never raises."""

    def __init__(self, live: Live, read_account):
        super().__init__(daemon=True, name="trade-poller")
        self.live, self.read_account, self.stopping = live, read_account, threading.Event()

    def run(self) -> None:
        while not self.stopping.wait(POLL_SECONDS):
            try:
                self.live.out.portfolio(self.read_account())
                if self.live.order is not None:
                    self.live.out.work_order(self.live.order, understood=_understood(self.live.plan), found=views.companies_found(self.live.order, self.live.plan), top=_top(self.live.plan))
            except Exception:
                continue

    def stop(self) -> None:
        self.stopping.set()
        self.join(timeout=3)


def run_live(run: Run, params: dict, accounts, question_timeout: float = 600.0, *, parts_builder=None, graph_builder=build_desk) -> None:
    out, live, started = Emitter(run), None, time.perf_counter()
    live = Live(out)
    chosen = run_settings_module.run_settings(params["strict"], params["neutral"], params.get("raise_funds", False), params.get("ignore_critic", False))      # entry 142, R12: the box "Always raise funds" is ALWAYS_RAISE_FUNDS for this run; entry 148, IC1 and IC9: the box "Ignore Critic" skips the Critic's node and Strict cannot go with it (a missing value of either is false)
    account_id = params["account_id"]
    try:
        view = accounts.view(account_id)
    except Exception as error:
        out.finished("failed", f"the account could not be read: {getattr(error, 'message', None) or type(error).__name__}", [], [], {}, 0)
        return
    mode = params.get("trade_mode") or view["account"]["mode"]
    run.emit("run_started", {"run_id": run.id, "mode": "live", "account_id": account_id, "prompt": params["prompt"], "options": {"strict": chosen.strict, "neutral": chosen.allow_neutral, "fetch": params["fetch"], "raise_funds": chosen.always_raise_funds, "ignore_critic": chosen.ignore_critic},
                             "trade_mode": mode, "started_utc": utc_now(), "stages": views.stage_list()})
    out.log("you", params["prompt"])
    final, poller, failure = None, None, None
    with run_settings_module.applied(chosen):
        try:
            final, poller = _stream(run, params, accounts, question_timeout, out, live, chosen, mode, parts_builder, graph_builder)
        except Exception as error:      # the kind of failure only, unless it is one of ours (a ValueError or RuntimeError: its message is for the user)
            failure = str(error) if isinstance(error, (ValueError, RuntimeError)) and str(error) else f"the run failed ({type(error).__name__})"
            live.failed.add(live.running or "account_manager")
        finally:
            if isinstance(poller, Poller):
                poller.stop()
    if run.stop_flag.is_set() and live.running:
        live.failed.add(live.running)               # "stopped here": the stage that was running when the user stopped the run
    live.running, live.waiting = None, False
    live.refresh()
    if failure:
        out.finished("failed", failure, [failure], live.path, live.seconds, time.perf_counter() - started, agent="system")      # `finished` writes the feed's last line
        return
    _finish(run, params, accounts, out, live, final, started)


def _stream(run, params, accounts, question_timeout, out, live, chosen, mode, parts_builder, graph_builder):
    args = session.build_parser().parse_args(["--account", str(params["account_id"]), "--prompt", params["prompt"], *(["--fetch"] if params["fetch"] else [])])
    began = time.perf_counter()
    out.log("system", "starting the agents")
    try:
        parts = (parts_builder or session.build_parts)(args)
    except FileNotFoundError:
        raise RuntimeError("the account book is not set up: run scripts/setup_postgres.sh") from None
    except (OSError, RuntimeError, ValueError) as error:           # a missing key: the message names the setting, never a value
        raise RuntimeError(str(error) or f"the agents could not be started ({type(error).__name__})") from None
    out.log("system", f"the agents are ready ({time.perf_counter() - began:.1f} s)")
    guard = getattr(parts, "guard", None)                  # the security layer's doors (entry 116): the request is counted against the rate limit at the page (`POST /api/runs`), so the desk's own door does not count it again
    out.guard, out.account_id = guard, params["account_id"]
    manager = parts.new_manager()
    manager.connect(parts.store, params["account_id"])
    holder = manager.users.get()
    profile = manager.profile
    reader = lambda: views.account_view(manager.users.get(), manager.list_portfolio(), as_of=utc_now())
    out.portfolio(reader(), start=True)
    if note := clock_note():
        out.log("system", note)
    confirm, approver, clarifier = callables(run, live, question_timeout, auto=holder.trade_mode == "auto")

    def new_manager():
        fresh = parts.new_manager()
        fresh.screen = lambda line: out.log("account_manager", str(line).rstrip())                                                     # what the Account Manager tells the user while it trades
        return fresh

    graph = graph_builder(clerk_factory=lambda order: parts.clerk, advisor_factory=lambda order: parts.advisor, funds_factory=new_manager, with_critic=True, critic_factory=lambda order: parts.critic,
                          with_trader=True, trader_factory=lambda order: parts.trader, approver=approver, store_factory=lambda: parts.store, confirm=confirm, clarify=clarifier,
                          **({"guard": guard.without_rate_limit()} if guard is not None else {}))
    request = {"ask": {"text": params["prompt"], "default_count": session.DEFAULT_COUNT, "fetch": params["fetch"]}, "account": {"account": params["account_id"], "profile": profile}, "trade": {"mode": "paper", "book": True, "model": True},
               **({"ignore_critic": True} if chosen.ignore_critic else {})}      # the user's box for this run (entry 148): the desk skips the Critic's node; with it off the request is what it always was
    state0 = {**NO_RUN_STATE, "request": request}
    poller, final, shown = None, None, 0
    out.log("system", "THE DESK: Account Manager → Data Clerk → Advisor → Critic → Account Manager's decision → Trader (a trade you name yourself goes from the Account Manager straight to its decision and the Trader)")
    for mode_, chunk in graph.stream(state0, stream_mode=["tasks", "values"], config=tracing.stream_config(run, on_activity=(lambda calls, tin, tout, model: out.activity(calls, tin, tout, model, live.running)) if COUNT_MODEL_CALLS else None, mode=holder.trade_mode, strict=chosen.strict, neutral=chosen.allow_neutral, ignore_critic=chosen.ignore_critic)):
        if mode_ == "tasks":
            name = chunk.get("name")
            if "result" not in chunk and "error" not in chunk:                                    # a node started
                live.running = name
                live.started[name] = time.perf_counter()
                if name == "trader":
                    poller = Poller(live, reader)
                    poller.start()
                live.refresh()
            else:                                                                                # a node ended: the values chunk that follows carries its path and seconds, and only then does the stage change
                if chunk.get("error"):
                    live.failed.add(name)
                if name == "trader" and isinstance(poller, Poller):
                    poller.stop()
                    poller = None
        else:
            final = chunk
            live.order, live.plan = chunk.get("order"), chunk.get("plan")
            live.request_errors = list(chunk.get("errors") or [])        # the desk's own `errors` are the first node's: with any, the graph ends there (`after_account_manager`), whether or not a work order was made (a direct order whose stocks could not all be found HAS one)
            for node in chunk["path"][shown:]:
                live.path.append(node)
                live.seconds[node] = chunk["seconds"].get(node, 0)
                _after_node(out, node, chunk, mode)
            shown = len(chunk["path"])
            live.running = None
            if live.order is not None and not _is_news(live.plan) and not _is_quote(live.plan):      # a request that only asks for the news has no work order table to show (entry 149): its News pane is the view; the price (entry 150) is said in the output
                out.work_order(live.order, understood=_understood(live.plan), found=views.companies_found(live.order, live.plan), top=_top(live.plan))
            live.refresh()
        if run.stop_flag.is_set():
            out.log("system", "stopped: the desk stops before its next stage; a trade already placed stays", "warn")
            break
    if not run.stop_flag.is_set():
        saved = proving_sessions.save_session(params["prompt"], None, final, account_id=params["account_id"], screen=proving_sessions.screen_of(guard, params["account_id"]))      # the Track Record keeps the run (a failure to save is a log line, never a failed run)
        if saved is not None:
            out.log("system", "the run is kept for the Track Record")
    return final, poller


def _after_node(out: Emitter, node: str, state: dict, mode: str) -> None:
    """The line of a node that finished, and what the session prints under it: the decisions, the Data Clerk's finds of a direct order, the direct order's section."""
    order, plan = state.get("order"), state.get("plan") or {}
    out.log(AGENT_OF_NODE.get(node, "system"), node_line(node, state["seconds"].get(node, 0)))
    if node == "account_manager" and plan.get("action") in ("news", "quote"):      # the news run (entry 149, N4) and the price run (entry 150) ask nothing: what was understood is said, not asked
        out.log("account_manager", f"understood: {plan['intent'].describe()}")
    if node == "data_clerk" and plan.get("action") == "quote" and order is not None:      # the Data Clerk's quote is the run's result (entry 150, Q7): the page's output says it, a line for each stock
        out.quote(views.quote_view(order, plan, out.guard, out.account_id))
    if node == "data_clerk" and plan.get("action") == "news" and order is not None:      # the Data Clerk's answer is the run's result (entry 149, N9): the page selects its News pane
        out.news(views.news_view(order, plan, out.guard, out.account_id))
    if node == "account_manager_decision" and order is not None and order.trade_decisions:
        out.block("account_manager", session.decision_lines(order, mode))
    if node == "account_manager" and plan.get("direct") and order is not None and order.data is not None:
        out.block("data_clerk", data_clerk_lines(order, "found", "not found"))
    if node == "account_manager_direct" and order is not None:
        out.block("account_manager", direct_section(order))


def _finish(run: Run, params: dict, accounts, out: Emitter, live: Live, final: dict | None, started: float) -> None:
    """The reports in the session's own order, the final portfolio and the verdict (spec 4.3 step 6)."""
    total = lambda: time.perf_counter() - started
    stopped = run.stop_flag.is_set()
    order, plan = (final or {}).get("order"), (final or {}).get("plan")
    errors = [str(error) for error in ((final or {}).get("errors") or [])]
    if order is None or plan is None:           # the request could not be read, was not confirmed, or the account could not be used: the desk stopped at its first node
        reason = "; ".join(errors) or "the desk did not start"
        live.request_errors = [reason]
        live.refresh()
        out.finished("stopped" if stopped else "failed", reason, [reason, *([NOT_PLACED_HINT] if (final or {}).get("clarify") else [])], live.path, live.seconds, total())      # a request that could not be read: asked to clarify it (entry 102)
        return
    if _is_news(plan):      # a request that only asked for the news (entry 149): the Data Clerk's answer is the result; no decision, no trade and no use of money to report, and the portfolio is as it was
        out.news(views.news_view(order, plan, out.guard, out.account_id))
        if stopped:
            out.finished("stopped", "Stopped.", [], live.path, live.seconds, total())
        elif order.status != "data_ready":
            reasons = [*order.errors, *([NOT_PLACED_HINT] if (final or {}).get("clarify") else [])]
            out.finished("failed", "The news could not be shown: " + (order.errors[0] if order.errors else "the Data Clerk's answer failed its checks"), reasons, live.path, live.seconds, total())
        else:
            verdict = views.run_verdict(None, None, action="news", order=order)
            out.finished(verdict["status"], verdict["headline"], verdict["reasons"], live.path, live.seconds, total())
        return
    if _is_quote(plan):      # a request that only asked for the current price (entry 150): the Data Clerk's quote is the result; no decision, no trade and no use of money to report, and the portfolio is as it was
        out.quote(views.quote_view(order, plan, out.guard, out.account_id))
        if stopped:
            out.finished("stopped", "Stopped.", [], live.path, live.seconds, total())
        elif order.status != "data_ready":
            reasons = [*order.errors, *([NOT_PLACED_HINT] if (final or {}).get("clarify") else [])]
            out.finished("failed", "The price could not be shown: " + (order.errors[0] if order.errors else "the Data Clerk's quote failed its checks"), reasons, live.path, live.seconds, total())
        else:
            verdict = views.run_verdict(None, None, action="quote", order=order)
            out.finished(verdict["status"], verdict["headline"], verdict["reasons"], live.path, live.seconds, total())
        return
    if errors:                                  # a work order was made and the desk still stopped at its first node: a direct order with a stock the Data Clerk could not find (`AccountManager.name_the_stocks`). The
        live.request_errors = errors            # reasons are the Data Clerk's own words; nothing was priced, decided or traded, so the Trader's and the money's reports have nothing to say
        live.refresh()
        out.work_order(order, understood=_understood(plan), found=views.companies_found(order, plan), top=_top(plan))
        out.finished("stopped" if stopped else "no_trade", NOT_PLACED, [*errors, *([NOT_PLACED_HINT] if plan.get("direct") else [])], live.path, live.seconds, total())
        return
    intent, direct = plan["intent"], bool(plan.get("direct"))
    find, result = (None, None) if direct else (plan["find"], plan["result"])
    if result is not None:
        result.order = order
    buying = intent.action in ("buy", "order")
    if not direct:
        out.block("account_manager", session.status_lines(result, intent, find))
    out.block("account_manager", neutral_lines(order))
    if buying:
        out.block("trader", session.trade_lines(order))
    after_view = accounts.view(params["account_id"])
    before_view = out.before_account
    if buying and session.sized_purchases(order) and not direct and before_view is not None:
        out.block("account_manager", session.use_of_money_lines(order, _portfolio_of(before_view), _portfolio_of(after_view)))
    out.portfolio(after_view)
    out.work_order(order, understood=_understood(plan), found=views.companies_found(order, plan), top=_top(plan))
    if stopped:
        out.finished("stopped", "Stopped. A trade already placed stays.", [], live.path, live.seconds, total())
        return
    verdict = views.run_verdict(before_view, after_view, action=intent.action, order=order, result=result)
    out.finished(verdict["status"], verdict["headline"], verdict["reasons"], live.path, live.seconds, total())


def _portfolio_of(view: dict):
    """The Portfolio a report helper reads, from an account view (shares x average cost is what the page shows; the helper only needs the cash)."""
    from prometheus.common.portfolio import Portfolio

    return Portfolio(user=view["account"]["name"], cash_avail=view.get("available", view["cash"]), positions=[{"symbol": p["symbol"], "name": p["name"], "shares": p["shares"], "cost_basis": p["cost_basis"]} for p in view["positions"]])
