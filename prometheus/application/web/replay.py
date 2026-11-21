"""The replay of a real capture (UI v1 spec 4.6, D8): `--replay tests/fixtures/session_real/bob_jones_oil.json`.

A capture is what `session --save` wrote: the request, what the Account Manager understood, the nodes that ran and the work order without its data. This module plays it through the SAME events and the SAME view code as a
live run (`emitter.py`, `views.py`), node by node, each with the work order cut back to what that node had written. No model is called, no quote is read, no database is touched and no key is needed, so the page can be
built, tested, screenshotted and shown for free.

The capture's numbers are the ints and floats that `json` read: the account they build (`Portfolio`) turns them into Decimals through the typed doors, a recorded fill goes through `shares.quantity` and `money.price` before `apply_fill`, and what the page is sent is
written by the one encoder (design choices entry 112).

The account is the capture's: the portfolio before and after are the Trader's own (`trader_runs[-1].portfolio_before` and `portfolio_after`), the risk level a review's `risk.tolerance_stated`, the mode a decision's `mode`.
The name is the one the file name carries (`bob_jones_oil.json` is Bob Jones): a capture does not keep the account. In Ask mode the replay asks what the live desk asks (the confirmation of what was understood, and each
approval the capture records) and then plays the recorded session whatever the answer: the page says so. Writes to the account are refused: it is the capture's.
"""
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from prometheus.agents.account_manager.direct import data_clerk_lines, direct_section
from prometheus.agents.account_manager.intent import TradeIntent
from prometheus.agents.account_manager.neutral import neutral_lines
from prometheus.agents.critic.report import scorecard
from prometheus.common import shares as sh
from prometheus.application.session import decision_lines, sized_purchases, trade_lines, use_of_money_lines
from prometheus.application.web import views
from prometheus.application.web.emitter import AGENT_OF_NODE, Emitter, node_line, utc_now
from prometheus.application.web.runs import Run
from prometheus.common.money import price as price_door
from prometheus.common.portfolio import Portfolio, apply_fill
from prometheus.common.work_order import WorkOrder

PAUSE = {"account_manager": 1.2, "data_clerk": 2.4, "advisor": 2.4, "critic": 2.4, "account_manager_decision": 1.2, "account_manager_direct": 1.2, "account_manager_funds": 2.4, "trader": 1.2}   # seconds of a node at speed 1.0
FILL_PAUSE = 1.0     # seconds between two fills at speed 1.0


@dataclass
class ReplayHolder:
    """What `views.account_view` reads of a holder: the capture's account, never a row of the book."""
    account_id: int
    display_name: str
    age: int | None
    risk_tolerance: str
    trade_mode: str
    seed_key: str | None = None


def load_capture(path: str | Path) -> dict:
    """The capture, checked: it needs a request, the nodes that ran (`path`) and the work order. Raises ValueError with a message for the user."""
    path = Path(path)
    try:
        capture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"the capture {path.name} could not be read ({type(error).__name__})") from None
    if not isinstance(capture, dict) or not isinstance(capture.get("path"), list) or not isinstance(capture.get("order"), dict) or not capture["path"]:
        raise ValueError(f"{path.name} is not a session capture (it needs a request, the nodes that ran and a work order: make one with `session --save`)")
    capture["_file"] = path.name
    return capture


def _portfolio(user: str, raw: dict | None, names: dict[str, str]) -> Portfolio:
    raw = raw or {}
    return Portfolio(user=user, cash_avail=raw.get("cash", 0),      # the capture's cash is the number `json` read: the field turns it into a Decimal by its digits
                     positions=[{"symbol": p["symbol"], "name": names.get(p["symbol"]), "shares": p["shares"], "cost_basis": p["cost_basis"], "acquired": p.get("acquired")} for p in raw.get("positions", [])])


def _cash_from_reasons(order: dict) -> Decimal:
    """A direct order that was not funded has no Trader run, so no portfolio: the decision's own reason says the cash ("the cash $20,000.00 does not pay for ...")."""
    for decision in order.get("trade_decisions") or []:
        found = re.search(r"cash \$([\d,]+\.\d\d)", decision.get("reason") or "")
        if found:
            return Decimal(found.group(1).replace(",", ""))      # the digits the reason is written with
    return Decimal("0.00")


class Replay:
    def __init__(self, capture: dict, speed: float = 1.0, question_timeout: float = 600.0):
        if speed < 0:
            raise ValueError("the replay speed cannot be negative")
        self.capture, self.speed, self.question_timeout = capture, speed, question_timeout
        order = capture["order"]
        self.order, self.path = order, list(capture["path"])
        self.intent = TradeIntent.model_validate(capture["intent"]) if capture.get("intent") else None
        self.names = {r["symbol"]: r["company"] for r in [*(order.get("candidates") or []), *(order.get("recommendations") or []), *(order.get("direct_orders") or [])] if r.get("company")}      # the names the capture has: a company the Advisor did not name stays unnamed
        stem = Path(capture.get("_file", "replay.json")).stem.split("_")
        self.user = " ".join(w.capitalize() for w in stem[:2]) if len(stem) >= 3 else "Replay user"
        reviews, decisions = order.get("reviews") or [], order.get("trade_decisions") or []
        self.risk = next((r["risk"]["tolerance_stated"] for r in reviews if r.get("risk")), "moderate")
        self.mode = next((d["mode"] for d in decisions if d.get("mode") in ("ask", "auto")), "ask")
        runs = order.get("trader_runs") or []
        before = runs[0]["portfolio_before"] if runs else {"cash": _cash_from_reasons(order), "positions": []}
        after = runs[-1]["portfolio_after"] if runs else before
        self.before = _portfolio(self.user, before, self.names)
        self.after = _portfolio(self.user, after, self.names)
        self.holder = ReplayHolder(account_id=0, display_name=self.user, age=None, risk_tolerance=self.risk, trade_mode=self.mode)

    # ---- the account the routes show
    def account(self, final: bool = False) -> dict:
        return views.account_view(self.holder, self.after if final else self.before, as_of=utc_now(), read_only=True)

    # ---- the run
    def play(self, run: Run) -> None:
        started = time.perf_counter()
        out = Emitter(run)
        order, capture = self.order, self.capture
        run.emit("run_started", {"run_id": run.id, "mode": "replay", "account_id": 0, "prompt": capture.get("request", ""), "options": {"strict": False, "neutral": True, "fetch": False}, "trade_mode": self.mode,
                                 "started_utc": utc_now(), "stages": views.stage_list(), "source": capture.get("_file")})
        out.portfolio(self.account(), start=True)
        out.log("system", f"REPLAY of a recorded session ({capture.get('_file')}): no model is called, no quote is read and no database is touched. The seconds shown are the replay's own pacing.")
        understood = self.intent.describe() if self.intent else capture.get("request")
        out.log("you", capture.get("request", ""))
        out.log("account_manager", f"understood: {understood}")
        out.stages(views.stage_states(path=[], running=None, order=None))
        if self.mode == "ask":
            answer = run.ask("confirm", {"title": "Confirm what was understood", "text": f"I understood: {understood}. Go ahead?"}, self.question_timeout)
            out.log("you", "yes, go ahead" if answer is True else "no, stop" if answer is False else "(no answer: not going ahead)", "info" if answer is True else "warn")
            if answer is not True:
                out.log("system", "Stopped. Nothing was bought or sold." if answer is False else "No answer: not going ahead.", "warn")
                out.stages(views.stage_states(path=[], running=None, order=None))
                out.finished("stopped", "Stopped: what was understood was not confirmed. Nothing was bought or sold.", [], [], {}, time.perf_counter() - started)
                return
            out.log("system", "REPLAY: your answers are not applied. The recorded session goes on as it was.")
        done, seconds, calls, tokens_in, tokens_out = [], {}, 0, 0, 0
        for node in self.path:
            if run.stop_flag.is_set():
                break
            before_node = time.perf_counter()
            out.stages(views.stage_states(path=done, seconds=seconds, running=node, order=views.cut_order(order, done)))
            if node == "trader":
                self._ask_approvals(run, out)
            self._pause(PAUSE.get(node, 1.0))
            done.append(node)
            seconds[node] = round(time.perf_counter() - before_node, 1)
            cut = views.cut_order(order, done)
            work = WorkOrder.model_validate(cut)
            if node == "trader":
                self._play_fills(run, out, done, understood)          # the fills first, one by one, as they were booked; the node's own line after them
            out.log(AGENT_OF_NODE[node], node_line(node, seconds[node]))
            if node == "account_manager_decision" and work.trade_decisions:
                out.block("account_manager", decision_lines(work, self.mode))
            if node == "account_manager" and work.direct_orders and work.data is not None:
                out.block("data_clerk", data_clerk_lines(work, "found", "not found"))
                out.block("account_manager", [note for note in work.data.notes if note.startswith("dropped")])           # a stock that was left out of the order (entry 102): the live run says it as it happens
            if node == "account_manager_direct":
                out.block("account_manager", direct_section(work))
            out.work_order(work, understood=understood)
            out.stages(views.stage_states(path=done, seconds=seconds, running=None, order=cut))
            calls, tokens_in, tokens_out = self._activity(out, node, cut, calls, tokens_in, tokens_out)
        stopped = run.stop_flag.is_set() and len(done) < len(self.path)
        final = WorkOrder.model_validate(views.cut_order(order, done))
        self._reports(out, final)
        after = self.account(final=True) if "trader" in done else self.account()
        out.portfolio(after)
        before_view = self.account()
        total = time.perf_counter() - started
        if stopped:
            out.finished("stopped", "Stopped: the recorded session was cut short.", [], done, seconds, total)
            return
        verdict = views.run_verdict(before_view, after, action=self.intent.action if self.intent else "buy", order=final, mode=self.mode)
        out.finished(verdict["status"], verdict["headline"], verdict["reasons"], done, seconds, total)

    # ---- the pieces
    def _pause(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds * self.speed
        while self.speed > 0 and time.monotonic() < deadline:
            time.sleep(0.05)

    def _ask_approvals(self, run: Run, out: Emitter) -> None:
        """The approvals the capture records (a decision the user was asked about): the live desk asks them in Ask mode before the Trader; the replay asks again, and plays on whatever the answer."""
        order = self.order
        asked = [d for d in order.get("trade_decisions") or [] if d.get("asked_user") and d.get("will_trade") in ("yes_user", "no_user_denied") and d.get("origin") not in ("direct",)]
        records = {r["identity"]["record_id"]: r for r in order.get("recommendations") or []}
        reviews = {r["record_id"]: r for r in order.get("reviews") or []}
        cash = self.before.cash_avail
        proposal = [{"symbol": d["symbol"], "action": d["action"], "shares": d["shares"], "est_cost": d["est_cost"], "dollars": d.get("dollars"), "price": d.get("price")} for d in asked]
        for n, d in enumerate(asked, 1):
            review = reviews.get(d["record_id"])
            text = scorecard(review, records.get(d["record_id"]))["text"] if review else d.get("reason", "")
            answer = run.ask("approve", {"title": f"Approve trade {n} of {len(asked)}", "text": text, "symbol": d["symbol"], "shares": d["shares"], "est_cost": d["est_cost"], "dollars": d.get("dollars"), "price": d.get("price"), "action": d["action"], "n": n, "of": len(asked),
                                         "cash": cash, "proposal": proposal, "scorecard": text}, self.question_timeout)
            out.log("you", f"{'approved' if answer is True else 'denied' if answer is False else 'no answer to'} {d['symbol']}", "info")

    def _play_fills(self, run: Run, out: Emitter, done: list[str], understood: str | None) -> None:
        """The fills one by one, as the live run books them: the table and the portfolio change after each (the capture's own order of trades)."""
        trades = [t for t in self.order.get("trades") or [] if t.get("status") == "filled" and t.get("fill")]
        held = self.before
        for k, trade in enumerate(trades, 1):
            self._pause(FILL_PAUSE)
            fill = trade["fill"]
            held = apply_fill(held, trade["symbol"], trade["side"], sh.quantity(fill["shares"]), price_door(fill["price"]), date.fromisoformat(fill["date"]), name=self.names.get(trade["symbol"]))      # the capture's numbers through their doors
            partial = {**views.cut_order(self.order, done), "trades": [t for t in (self.order.get("trades") or [])[:k]], "status": "decided"}
            out.work_order(partial, understood=understood)
            out.portfolio(views.account_view(self.holder, held, as_of=utc_now(), read_only=True))
            out.log("trader", f"{trade['trade_id']} {trade['symbol']} {trade['side']} {sh.text(fill['shares'])}: {trade.get('reason') or trade['status']}")
            out.log("account_manager", f"booked {trade['trade_id']} in the account: cash ${held.cash_avail:,.2f}")

    def _activity(self, out: Emitter, node: str, cut: dict, calls: int, tin: int, tout: int) -> tuple[int, int, int]:
        runs = {"advisor": cut.get("advisor_runs"), "critic": cut.get("critic_runs"), "trader": cut.get("trader_runs")}.get(node) or []
        if not runs:
            return calls, tin, tout
        calls += sum(r.get("calls", r.get("turns", 0)) or 0 for r in runs)
        tin += sum(r.get("tokens_in", 0) or 0 for r in runs)
        tout += sum(r.get("tokens_out", 0) or 0 for r in runs)
        out.activity(calls, tin, tout, runs[-1].get("model"), node)
        return calls, tin, tout

    def _reports(self, out: Emitter, order: WorkOrder) -> None:
        out.block("account_manager", neutral_lines(order))
        if self.intent is None or self.intent.action in ("buy", "order"):
            out.block("trader", trade_lines(order))
            if sized_purchases(order) and not order.direct_orders:
                out.block("account_manager", use_of_money_lines(order, self.before, self.after))
