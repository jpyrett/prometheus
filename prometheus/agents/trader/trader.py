"""The Trader: the fifth agent of the desk. It executes the trades the Account Manager decided on, against a simulated broker (Trader v1 spec sections 1, 4, 8 and 9).

    trader = Trader()                                                                    # gpt-oss-120b through OpenRouter (the `trader` role: TRADER_MODEL), temperature 0, paper mode
    order = trader.execute_trades(order, account, portfolio, now, funding=None, approver=None, book=False)       # status "decided" -> "traded"; `order.trades` and one more entry on `order.trader_runs`
    Trader(use_model=False).execute_trades(...)                                          # no model at all: code places the default orders. The free, repeatable way to run a backtest

    python -m prometheus.agents.trader.replay --order saved.json --mode backtest --no-model      # a saved order replayed against the stored bars: free, no model, no paid call

The steps are in `execute.py` (the ten steps of spec section 4, as `critic/review.py` holds the Critic's nine). This file is the class and the model's loop. THE TRADER HOLDS NO CASH AND NO PORTFOLIO (design choices
entry 81): for every trade the Account Manager approves the shares against the funds as they are now and is told the fill at once (`broker.py`), one trade at a time, so the Trader cannot overdraw the account; whoever places
the trade, the model's `place_order` or the sweep, it goes through that one funnel. The pattern is the Data Clerk's tool loop
(`agents/data_clerk/data_clerk_agent.py`): an `agent` node, a tools node, a budget around every tool, and a LAST turn without tools, so a loop never ends in an empty answer.

TWO THINGS DIFFER FROM THE DATA CLERK'S LOOP. The tools node is written here and is NOT the prebuilt `ToolNode`: that one runs the calls of one turn at the same time, and two purchases placed in one turn would race for the
same cash, so the same order could give two different results. This node runs the calls one after the other, in the order the model wrote them. And an argument a tool does not have is refused, never dropped
(`tools.trader.call_tool`).

WHAT THE MODEL IS FOR, AND WHAT STOPS IT. Its job is the part a person on a trading desk does: work through the tickets in order, choose the order type inside the limits it is given, check what happened, report to the Account
Manager, and say in two sentences what was done. Code checks all of it (T1 to T10 of the spec). The TEMPERATURE is 0 so that the same order gives the same tool calls: that makes a run repeatable, it does not make the model
truthful. What stops a made-up trade is that the model has no way to express one: its tools take a `trade_id` that code issued, and code owns the symbol, the side, the shares and the price limits. A model that fails, times
out, answers nonsense, or cannot even start never stops a permitted trade (the sweep places it) and never causes one that is not permitted.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from prometheus.agents.trader.config import TraderConfig, load_config
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.models.chat import model_name as chat_model_name
from prometheus.prompts.trader import LAST_TURN, system_prompt, user_message
from prometheus.tools.trader import Budget, TraderTools, budgeted, call_tool, loggable

TEMPERATURE = 0.0      # repeatable, not truthful: see the module text. A test asserts it reaches `get_chat_model`
EFFORTS = ("low", "medium", "high")
PREFERRED_PROVIDERS = ("cerebras", "groq")   # the fast hosts of gpt-oss-120b on OpenRouter (the Account Manager's entry 50)


class NullLedger:
    """A trades ledger that keeps nothing: what `--no-ledger` hands the Trader (the replay's flag, and the runner's) so that a paper run leaves no line in data/ledger/trades.jsonl."""

    def append(self, record) -> None:
        return None


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    forced: bool      # the last turn was asked without tools


def message_text(reply) -> str:
    """The text of a model's reply: its content, or the text parts of it when the content is a list of blocks."""
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else str(part.get("text", "")) if isinstance(part, dict) else "" for part in content)
    return ""


class Trader:
    def __init__(self, model=None, model_id: str | None = None, config_path=None, mode: str | None = None, effort: str | None = "auto", providers: tuple[str, ...] | str | None = "auto",
                 use_model: bool = True, sweep: bool = True, ledger=None):
        """model: a chat model (anything with `bind_tools` and `invoke`; a scripted one in tests); otherwise OpenRouter through `get_chat_model` with the model of the `trader` role (TRADER_MODEL, gpt-oss-120b by default), or
        `model_id`, at temperature 0, built when first needed: a model that cannot start (a missing key) is a note on the run, never an exception, and the sweep still trades. config_path: the Trader's settings
        (configs/trader_v1.json). mode: `paper` or `backtest` (the settings' own when left out); anything else is refused when a run starts (rule T4), not here. effort and providers follow the Account Manager's "auto"
        rule: for gpt-oss, reasoning effort `low` and the providers Cerebras then Groq; `require_parameters` is always on, so OpenRouter only uses a host that supports tool calls. use_model False: no model at all, the sweep
        places the default orders. sweep False: what the model left is rejected `not_attempted` instead of being placed by code. ledger: where the trades go (data/ledger/trades.jsonl in paper mode; a test passes one in a
        temporary folder, or a memory ledger: anything with `append`)."""
        if effort not in (None, "auto", *EFFORTS):
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, 'auto' or None")
        self._model, self._model_id, self._config_path, self._ledger = model, model_id, config_path, ledger
        self._config: TraderConfig | None = None
        self.mode, self.effort, self.providers, self.use_model, self.sweep = mode, effort, providers, use_model, sweep
        self.errors: list[str] = []            # why the Trader's model could not start, when it could not: copied onto the run
        self._model_tried = model is not None

    @property
    def config(self) -> TraderConfig:
        """The Trader's thresholds (configs/trader_v1.json), read and checked once. Raises ValueError, naming the file, when it is not valid."""
        if self._config is None:
            self._config = load_config(self._config_path)
        return self._config

    @property
    def model(self):
        """The Trader's chat model, built on first use. None when it could not be built (the reason is in `errors`). Temperature 0, the timeout of the settings, and a provider block that requires tool support."""
        if not self._model_tried:
            self._model_tried = True
            try:
                is_oss = "gpt-oss" in (self._model_id or chat_model_name("trader"))
                effort = ("low" if is_oss else None) if self.effort == "auto" else self.effort
                providers = (PREFERRED_PROVIDERS if is_oss else None) if self.providers == "auto" else self.providers
                provider = {"require_parameters": True, **({"order": list(providers), "allow_fallbacks": True} if providers else {})}
                extra = {"provider": provider, **({"reasoning": {"effort": effort}} if effort else {})}
                self._model = get_chat_model("trader", model=self._model_id, temperature=TEMPERATURE, timeout=self.config.limits.model_timeout_seconds, extra_body=extra)
            except (RuntimeError, ValueError) as error:   # for example a missing key: the message names the setting, never a value
                self.errors.append(f"the Trader's model could not start: {error}")
        return self._model

    @property
    def model_name(self) -> str | None:
        """The OpenRouter id of the model: what the model says it is, else the id asked for, else the `trader` role's setting. None when no model could be built."""
        if self._model is not None and getattr(self._model, "model_name", None):
            return self._model.model_name
        if self._model_id:
            return self._model_id
        if self._model is None and self._model_tried:
            return None
        return chat_model_name("trader")

    @property
    def ledger(self):
        if self._ledger is None:
            from prometheus.memory.ledger import Ledger
            self._ledger = Ledger(self.config.ledger.path)
        return self._ledger

    def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book: bool = False):
        """Execute the trades the Account Manager decided on (see `execute.py` for the ten steps) and return the SAME order with `trades`, one more entry on `trader_runs`, and `status == "traded"` once every permitted
        decision has a trade that is `filled` or `rejected`. `account` is the Account Manager (the Trader's only door: portfolio, prices, and for each trade its approval and its booking, entry 81). `portfolio` is what the desk loaded (the account's own when it is
        connected). `now` an aware datetime (the real clock when left out). `funding` a RaiseFunds result or None. `approver` the callable the decisions use, asked once per funded purchase. `book` has the Account Manager book each fill in the
        Postgres account book, in one transaction as it happens (paper mode only). Never raises for a data, model or account failure: a trade that cannot be executed is `rejected` with a code and a sentence, anything else is a message on the order. A
        `now` without a timezone is a programming mistake and raises ValueError."""
        from prometheus.agents.trader.execute import execute
        return execute(self, order, account, portfolio, now, funding, approver, book)

    # ---------------------------------------------------------------- the model's loop (step 6)

    def run_loop(self, broker, notifier, request_id: str, today, n_new: int, ticket_ids: list[str]) -> dict:
        """The tool loop over a run's broker. Returns {ran, turns, tokens_in, tokens_out, tool_calls, text, error, stopped}: `text` is the model's last message (the JSON the prompt asks for), `error` the kind of a model or
        network failure (the loop stops there and the sweep carries on), `stopped` the calls a limit of the budget stopped. What the tools did is already in the broker. Never raises."""
        out = {"ran": False, "turns": 0, "tokens_in": 0, "tokens_out": 0, "tool_calls": [], "text": "", "error": "", "rate_limited": False, "stopped": []}
        model = self.model
        if model is None:
            return out
        limits, mode = self.config.limits, broker.mode
        max_turns = min(limits.base_turns + n_new, limits.max_turns)
        budget = Budget(ticket_ids)
        tools = {t.name: t for t in (budgeted(x, budget) for x in TraderTools(broker, notifier).langchain_tools())}
        out["ran"] = True
        try:
            final = self._graph(model, tools, system_prompt(mode, today, limits.max_summary_words), max_turns, out).invoke(
                {"messages": [HumanMessage(content=user_message(request_id, mode, n_new))], "turns": 0, "forced": False}, config={"recursion_limit": 4 * max_turns + 10})
            out["text"] = message_text(final["messages"][-1]) if final["messages"] and isinstance(final["messages"][-1], AIMessage) else ""
            out["turns"] = final["turns"]
        except Exception as error:   # a model or network failure: report the kind, never request details. The trades the tools already placed stand, and the sweep carries on
            out["error"], out["rate_limited"] = type(error).__name__, is_rate_limited(error)
        out["stopped"] = list(budget.stopped)
        return out

    def _graph(self, model, tools: dict, prompt: str, max_turns: int, stats: dict):
        bound = model.bind_tools(list(tools.values()))

        def agent(state: State):
            last = state["turns"] >= max_turns - 1
            messages = [SystemMessage(content=prompt)] + state["messages"]
            reply = model.invoke(messages + [HumanMessage(content=LAST_TURN)]) if last else bound.invoke(messages)   # no tools on the last turn: it has to answer
            usage = getattr(reply, "usage_metadata", None) or {}
            stats["tokens_in"] += int(usage.get("input_tokens", 0) or 0)
            stats["tokens_out"] += int(usage.get("output_tokens", 0) or 0)
            stats["turns"] = state["turns"] + 1
            return {"messages": [reply], "turns": state["turns"] + 1, "forced": last or state["forced"]}

        def after_agent(state: State):
            reply = state["messages"][-1]
            return "tools" if getattr(reply, "tool_calls", None) and state["turns"] < max_turns else END

        def run_tools(state: State):
            """The calls of one turn, ONE AFTER THE OTHER in the order the model wrote them (not the prebuilt ToolNode: it runs them at the same time, and two purchases in one turn would race for the same cash)."""
            replies = []
            for call in state["messages"][-1].tool_calls:
                outcome, text = call_tool(tools, call.get("name"), call.get("args"))
                stats["tool_calls"].append({"name": str(call.get("name"))[:40], "args": loggable(call.get("args")), "outcome": outcome})
                replies.append(ToolMessage(content=text, tool_call_id=call.get("id") or "call", name=str(call.get("name") or "")))
            return {"messages": replies}

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", run_tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile()
