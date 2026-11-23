"""What a run says to the page (UI v1 spec 4.2): the same calls for the live run (`desk_run.py`) and the replay of a capture (`replay.py`), so the page cannot tell them apart.

    out = Emitter(run)
    out.log("critic", "reviewed each recommendation ... (45.3 s)")
    out.stages(states)               # the six stages: an event only for a stage whose state (or seconds) changed
    out.work_order(order, understood=..., found=...)
    out.portfolio(account_view)      # plus `changes` and `cash_change` against the portfolio at the start of the run
    out.news(views.news_view(order, plan))      # a request that only asked for the news (entry 149): the page's News pane
    out.quote(views.quote_view(order, plan))    # a request that only asked for the current price (entry 150): a line for each stock in the page's output
"""
import threading
from datetime import datetime, timezone

from prometheus.agents.account_manager.direct import CLARIFY
from prometheus.application.session import NODE_WORDS
from prometheus.application.web import views
from prometheus.application.web.runs import Run

AGENT_OF_NODE = {"account_manager": "account_manager", "data_clerk": "data_clerk", "advisor": "advisor", "critic": "critic", "account_manager_decision": "account_manager", "account_manager_direct": "account_manager", "account_manager_funds": "account_manager", "trader": "trader"}
AGENT_PREFIXES = ("the Account Manager ", "the Data Clerk ", "the Advisor ", "the Critic ", "the Trader ")      # the page shows the agent in its own column
LEVEL_OF_STATUS = {"news": "good", "quote": "good", "traded": "good", "waiting": "good", "looked": "info", "no_trade": "warn", "stopped": "warn", "failed": "error"}   # the colour of the feed's last line (`Emitter.finished`)
NOT_PLACED = "Nothing was placed: the order could not be made as you typed it."
NOT_PLACED_HINT = CLARIFY       # entry 102: the run's last message asks the user to clarify the request and send it again, in the words the terminal session uses too (`direct.CLARIFY`)


def node_line(node: str, seconds: float | None = None) -> str:
    """The session's own sentence for a node that finished, without the agent's name (the page shows it beside) and with its seconds."""
    text = NODE_WORDS.get(node, node)
    for prefix in AGENT_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text + (f" ({seconds:.1f} s)" if seconds is not None else "")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Emitter:
    def __init__(self, run: Run, guard=None, account_id=None):
        self.run = run
        self.guard, self.account_id = guard, account_id         # the security layer's door for the run's last message (entry 116): None (a replay) shows the text as it is
        self.last_states: dict[str, tuple] = {}
        self.before_account: dict | None = None
        self.last_account: dict | None = None
        self.last_rows = None
        self.last_news = None
        self.last_quote = None
        self._lock = threading.Lock()

    def log(self, agent: str, text: str, level: str = "info", block: bool = False) -> None:
        self.run.emit("log", {"agent": agent, "level": level, "text": text, "block": block})

    def block(self, agent: str, lines: list[str], level: str = "info") -> None:
        """A report block: the session's own lines, kept with their line breaks and without the ruler lines of the terminal."""
        body = [line for line in lines if line and set(line) != {"="}]
        if body:
            self.log(agent, "\n".join(body), level, block=True)

    def stages(self, states: list[dict]) -> None:
        for state in states:
            mark = (state["state"], state["node"], state["seconds"])
            if self.last_states.get(state["key"], ("pending", state["node"], 0.0))[0] == "pending" and state["state"] == "pending":
                continue                    # a stage nobody has touched is pending: the page starts that way
            if self.last_states.get(state["key"]) != mark:
                self.last_states[state["key"]] = mark
                self.run.emit("stage", state)

    def work_order(self, order, understood: str | None = None, found: list[dict] | None = None, top: int | None = None) -> dict:
        """Emit the whole work-order view, and keep the order as a plain dict for the records route. Not emitted again when nothing changed. `top`: the number of companies asked for (`views.work_order_view`)."""
        snapshot = views.order_snapshot(order)
        view = views.work_order_view(snapshot, understood=understood, found=found, top=top)
        with self._lock:
            self.run.snapshot = snapshot
            if view == self.last_rows:
                return view
            self.last_rows = view
        self.run.emit("work_order", view)
        return view

    def news(self, view: dict) -> dict:
        """Emit the news view of a request that only asked for the news (entry 149, N9), once: not again when nothing changed. The page selects its News pane when it arrives."""
        with self._lock:
            if view == self.last_news:
                return view
            self.last_news = view
        self.run.emit("news", view)
        return view

    def quote(self, view: dict) -> dict:
        """Emit the quote view of a request that only asked for the current price (entry 150, Q7), once: not again when nothing changed. The page's right column is NOT switched (Jacob, 2026-11-23: the price goes in the main output window): the price is written
        in the output itself, one line for each stock and one for each name that was not found (`views.quote_lines`; a problem is the reason of the run's last line, so it is not said twice)."""
        with self._lock:
            if view == self.last_quote:
                return view
            self.last_quote = view
        self.run.emit("quote", view)
        self.block("data_clerk", views.quote_lines(view, problems=False))
        return view

    def portfolio(self, account: dict, start: bool = False) -> None:
        """Emit the account as it is now, with what changed since the start of the run. The first one (`start`) is the 'before'."""
        with self._lock:
            if start or self.before_account is None:
                self.before_account = account
            change = views.portfolio_changes(self.before_account, account)
            if not start and self.last_account is not None and {k: v for k, v in self.last_account.items() if k != "as_of_utc"} == {k: v for k, v in account.items() if k != "as_of_utc"}:
                return
            self.last_account = account
        self.run.emit("portfolio", {**account, **change})

    def activity(self, calls: int, tokens_in: int, tokens_out: int, model: str | None, node: str | None) -> None:
        self.run.emit("activity", {"calls": calls, "tokens_in": tokens_in, "tokens_out": tokens_out, "model": model, "node": node})

    def finished(self, status: str, headline: str, reasons: list[str], path: list[str], seconds: dict, total: float, agent: str = "account_manager") -> None:
        """The end of a run. The Activity feed gets its LAST LINE first: how the run ended and why (the headline, and under it every reason the headline does not already hold), so that the feed never just stops and
        the user never has to guess (a refused order once ended with nothing in the feed). The user talks to the Account Manager, so the line is its own unless the caller says otherwise. The headline and the reasons are the run's
        FINAL MESSAGE: they go through the security layer first (`views.screened`), which redacts a secret and another account's number."""
        headline, reasons = views.screened(headline, self.guard, self.account_id), [views.screened(reason, self.guard, self.account_id) for reason in reasons]
        more = [reason for reason in reasons if reason and reason not in headline]
        self.log(agent, "\n".join([headline, *[f"  - {reason}" for reason in more]]), LEVEL_OF_STATUS.get(status, "info"), block=bool(more))
        self.run.emit("run_finished", {"status": status, "headline": headline, "reasons": reasons, "path": path, "seconds": seconds, "total_seconds": round(total, 1)})
