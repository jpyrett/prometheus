"""Named LangSmith spans for the agents' own steps (entry 115; design/langsmith_traceable_spec.md).

LangGraph and LangChain already trace every desk node, every chat-model call and the Data Clerk's tools (entry 114). What a trace does not show is the plain-Python work INSIDE a node: which agent
asked which other agent for what, the Advisor's and the Critic's steps, the sizing, and the account handshake (approve, fill, book). This module is the ONE place that knows `langsmith` for that:

    from prometheus.common.trace import traced

    @traced("account_manager.approve_trade")
    def approve_trade(self, trade, price, *, order, now, on, commission=Decimal(0)): ...

A decorated function changes NOTHING about what it does: same value, same exception, no argument touched.

  - NEST ONLY, NEVER A ROOT (R3). When no trace is open (a test, a command's step before the graph starts, `--account`) the wrapper calls the plain function: it makes no run, summarizes nothing and sends
    nothing. Inside a LangGraph node the current run is the node's, and the span is its child. So the online judge, which looks only at the desk's root run, still sees one root, and with tracing off nothing is built.
  - SMALL, SAFE PAYLOADS (R4). Inputs and outputs go through `summarize`: a work order is its id, its request, its symbols and the COUNT of what it holds (never its articles, filings or press releases);
    a plain record of this project (a pydantic model or a dataclass: `Approval`, `Booking`, `Sized`, `Trade`) is its fields; a Decimal stays a Decimal; lists keep their first 20 items and strings are cut at 2,000
    characters. Anything that is not data (a model, a clerk, a store, a connection, an approver, a callable, a ledger, a writer, a configuration, a message) is its type name only, never its `repr`, which could hold a key
    or a connection string.
  - `run_type` is "chain" for an agent's step and "tool" for a Data Clerk request that fetches or looks up. It is NEVER "llm": the model calls are already `llm` runs from LangChain, and a second one would double the token counts.

Everything here is also true of a span that is NOT sent: `summarize` never raises (what it cannot read is its type name), and a failure of the SDK in a span is the SDK's to log, the step still runs.
"""
import dataclasses
import enum
import functools
import inspect
import numbers
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import PurePath

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

MAX_ITEMS = 20       # a list keeps this many items, then says how many more there were
MAX_KEYS = 30        # a dict keeps this many keys
MAX_TEXT = 2000      # a string is cut here, with a marker
MAX_QUERY = 200      # the request on a work order's span
MAX_DEPTH = 6        # a record inside a record inside ...: deeper than this is its type name
MORE = "…"


def _text(value: str, limit: int = MAX_TEXT) -> str:
    return value if len(value) <= limit else f"{value[:limit]}… [{len(value) - limit} more characters]"


def _kind(value) -> str:
    return f"<{value.__name__} (class)>" if isinstance(value, type) else f"<{type(value).__name__}>"


def _ours(value) -> bool:
    """A class of this project: its records are data, and its objects (a store, a clerk, a model, a ledger) are not."""
    return type(value).__module__.startswith("prometheus.")


def _is_work_order(value) -> bool:
    try:
        from prometheus.common.work_order import WorkOrder
    except ImportError:      # never at run time; the module is plain Python of this project
        return False
    return isinstance(value, WorkOrder)


def _count(items) -> int:
    try:
        return len(items or [])
    except TypeError:
        return 0


def _work_order(order) -> dict:
    """A work order as a trace line: who and what, the symbols, and how much it holds. Its data (articles, filings, press releases, metrics, quotes) is never copied: it is in the node's own state."""
    data = order.data
    subject = order.subject
    symbols = [subject] if isinstance(subject, str) else list(subject or [])
    counts = {
        "companies": _count(getattr(data, "companies", None)),
        "sources": _count(getattr(data, "sources", None)),      # the chunks the Data Clerk read: articles, press releases and filings
        "quotes": _count(getattr(data, "quotes", None)),
        "bars": _count(getattr(data, "bars", None)),
        "recommendations": _count(order.recommendations),
        "reviews": _count(order.reviews),
        "trade_decisions": _count(order.trade_decisions),
        "direct_orders": _count(order.direct_orders),
        "trades": _count(order.trades),
        "errors": _count(order.errors),
    }
    return {"type": "WorkOrder", "id": order.request_id, "requester": order.requester, "action": order.action_type, "status": order.status,
            "query": _text(str(order.query), MAX_QUERY), "symbols": _items(symbols, 0), "counts": counts}


def _items(values, depth: int) -> list:
    values = list(values)
    kept = [_summarize(v, depth + 1) for v in values[:MAX_ITEMS]]
    return kept + [{MORE: len(values) - MAX_ITEMS}] if len(values) > MAX_ITEMS else kept


def _summarize(value, depth: int):
    if value is None or isinstance(value, (bool, Decimal)):
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, enum.Enum):
        return _summarize(value.value, depth)
    if isinstance(value, numbers.Real):      # an int or a float (the measurements stay floats)
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if depth > MAX_DEPTH:
        return _kind(value)
    if _is_work_order(value):
        return _work_order(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        values = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return _items(values, depth)
    if isinstance(value, dict):
        out = {(k if isinstance(k, str) else str(k)): _summarize(v, depth + 1) for k, v in list(value.items())[:MAX_KEYS]}
        return {**out, MORE: len(value) - MAX_KEYS} if len(value) > MAX_KEYS else out
    if _ours(value) and not type(value).__name__.endswith("Config"):
        if hasattr(value, "model_dump") and not isinstance(value, type):
            try:
                return _summarize(value.model_dump(), depth + 1)
            except Exception:
                return _kind(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: _summarize(getattr(value, f.name, None), depth + 1) for f in dataclasses.fields(value)}
    return _kind(value)


def summarize(value):
    """`value` as a small, safe trace payload (see the module text). Never raises."""
    try:
        return _summarize(value, 0)
    except Exception:
        return _kind(value)


def _arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """The call as {parameter: summary}, `self` and `cls` left out, a `**kwargs` flattened (as the SDK does)."""
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:      # a call the function itself will refuse: the span says what was given
        return {"args": summarize(args), "kwargs": summarize(kwargs)}
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and isinstance(arguments.get(name), dict):
            arguments.update(arguments.pop(name))
    return {name: summarize(value) for name, value in arguments.items()}


def _outputs(value) -> dict:
    summary = summarize(value)
    return summary if isinstance(summary, dict) else {"output": summary}


def traced(name, run_type: str = "chain"):
    """A decorator: the function runs as a named span of the open trace, and as itself when none is open (rule R3). `name` is the span's name (`agent.step`, snake case), or a function that takes the call's own
    arguments and returns it (the Advisor's model steps are named by their stage). `run_type` is "chain" or "tool"; never "llm" (rule R7). Not for a generator, a coroutine or a graph node."""
    if run_type not in ("chain", "tool"):
        raise ValueError(f'run_type must be "chain" or "tool", not {run_type!r}')

    def decorate(function):
        signature = inspect.signature(function)

        def call(packed):
            args, kwargs = packed      # the call travels as one value: the SDK gives a parameter called `config` or `run_tree` a meaning of its own, and the project's functions have such parameters
            return function(*args, **kwargs)

        def inputs_of(inputs: dict) -> dict:
            args, kwargs = inputs["packed"]
            return _arguments(signature, args, kwargs)

        span = traceable(name=name if isinstance(name, str) else function.__name__, run_type=run_type, process_inputs=inputs_of, process_outputs=_outputs)(call)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if get_current_run_tree() is None:      # no trace is open: this step never makes a root of its own, and nothing is summarized or sent
                return function(*args, **kwargs)
            extra = None
            if not isinstance(name, str):
                try:
                    extra = {"name": name(*args, **kwargs)}
                except Exception:      # a name that cannot be worked out is the function's own
                    extra = None
            return span((args, kwargs), langsmith_extra=extra)

        return wrapper

    return decorate
