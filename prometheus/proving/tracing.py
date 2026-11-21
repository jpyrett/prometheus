"""LangSmith for the Proving Ground: token counts of a decision day and the OUTCOMES as feedback (design/proving_ground_spec.md section 12; entries 114 and 115).

    usage = UsageHandler()                                    # a LangChain callback handler: the calls and tokens of each model, for the cost of a decision day
    post_outcomes(trace_id, outcomes, day_stats)              # feedback on the decision day's run: how the day's recommendations did

The spans are `common.trace` (`root` for the test, `span` for a decision day, the fill, the walk, the outcomes and the stats); this module only adds what a trace needs after the fact. THE FEEDBACK is written ONLY when tracing is on (a
test forces it off: nothing is posted and nothing fails), never raises, and is IDEMPOTENT: each feedback has a deterministic id (a uuid5 of the run, the key and the record), so scoring the same decision day again updates the row and
does not duplicate it. On a decision day's span: `outcome_excess_return` (the mean excess of the day's TRADED recommendations, a score), `outcome_win_rate` (won over closed, a score) and, for every recommendation, `rec_outcome`
(`value` won, lost or open; the symbol in the comment). Only figures and symbols go to LangSmith: never article text (R4). Nothing here reaches an agent or a model.
"""
import contextvars
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from langchain_core.callbacks import BaseCallbackHandler
from langsmith.utils import tracing_is_enabled

NAMESPACE = uuid.UUID("5f0a4a52-6d1b-5b1e-9c3a-0b6f6f6f9d10")      # the fixed namespace of the Proving Ground's feedback ids


class UsageHandler(BaseCallbackHandler):
    """Counts the chat-model calls of a decision day and their tokens, per model, and TIMES each call (the measurement of design/proving_ground_spec.md P14: where a day's seconds go). `calls` holds one row per call:
    {model, stage (the desk node it ran in: data_clerk, advisor, critic ...), start (seconds after the handler was made), seconds, tokens_in, tokens_out, ok}. Never raises into a model call."""
    raise_error = False

    def __init__(self):
        self._lock = threading.Lock()
        self.models: dict[str, dict] = {}
        self.calls: list[dict] = []
        self._open: dict = {}
        self._zero = time.perf_counter()

    def _begin(self, run_id, serialized, metadata, kwargs) -> None:
        try:
            params = kwargs.get("invocation_params") or {}
            model = params.get("model") or params.get("model_name") or ((serialized or {}).get("kwargs") or {}).get("model_name") or ((serialized or {}).get("kwargs") or {}).get("model")
            with self._lock:
                self._open[run_id] = {"model": model, "stage": (metadata or {}).get("langgraph_node"), "began": time.perf_counter()}
        except Exception:
            pass

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._begin(run_id, serialized, metadata, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs) -> None:
        self._begin(run_id, serialized, metadata, kwargs)

    def _finish(self, run_id, model, tokens_in: int, tokens_out: int, ok: bool) -> None:
        with self._lock:
            opened = self._open.pop(run_id, None) or {}
            now = time.perf_counter()
            began = opened.get("began", now)
            name = str(model or opened.get("model") or "unknown")
            self.calls.append({"model": name, "stage": opened.get("stage"), "start": round(began - self._zero, 2), "seconds": round(now - began, 2), "tokens_in": tokens_in, "tokens_out": tokens_out, "ok": ok})
            if ok:
                row = self.models.setdefault(name, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
                row["calls"] += 1
                row["tokens_in"] += tokens_in
                row["tokens_out"] += tokens_out

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        try:
            tokens_in = tokens_out = 0
            output = getattr(response, "llm_output", None) or {}
            model = output.get("model_name")
            usage = output.get("token_usage")
            if usage:
                tokens_in, tokens_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
            else:
                for group in getattr(response, "generations", []) or []:
                    for generation in group:
                        message = getattr(generation, "message", None)
                        meta = getattr(message, "usage_metadata", None) or {}
                        tokens_in += int(meta.get("input_tokens") or 0)
                        tokens_out += int(meta.get("output_tokens") or 0)
                        model = model or (getattr(message, "response_metadata", None) or {}).get("model_name")
            self._finish(run_id, model, tokens_in, tokens_out, True)
        except Exception:
            pass

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        try:
            self._finish(run_id, None, 0, 0, False)
        except Exception:
            pass

    def summary(self) -> dict:
        with self._lock:
            models = {m: dict(row) for m, row in self.models.items()}
        return {"models": models, "calls": sum(r["calls"] for r in models.values()), "tokens_in": sum(r["tokens_in"] for r in models.values()), "tokens_out": sum(r["tokens_out"] for r in models.values())}

    def call_rows(self) -> list[dict]:
        """The timed calls, in the order they started (a copy)."""
        with self._lock:
            return sorted((dict(c) for c in self.calls), key=lambda c: c["start"])


def feedback_id(run_id: str, key: str, record_id: str | None = None) -> str:
    """The deterministic id of one feedback row: the same run, key and record always give the same id, so a second scoring updates and does not duplicate."""
    return str(uuid.uuid5(NAMESPACE, f"{run_id}|{key}|{record_id or ''}"))


def day_feedback(trace_id: str, outcomes) -> list[dict]:
    """The feedback rows of one decision day (pure: what `post_outcomes` would post): the day's mean excess and win rate over its TRADED recommendations, and one `rec_outcome` for each recommendation. A recommendation that is
    not scored yet (no data, not filled) has no row."""
    rows = []
    traded = [o for o in outcomes if o.traded and o.excess_return_pct is not None]
    if traded:
        rows.append({"run_id": trace_id, "key": "outcome_excess_return", "score": round(sum(o.excess_return_pct for o in traded) / len(traded), 6), "value": None, "comment": f"{len(traded)} traded recommendation(s)", "feedback_id": feedback_id(trace_id, "outcome_excess_return")})
    closed = [o for o in outcomes if o.traded and o.result in ("won", "lost")]
    if closed:
        rows.append({"run_id": trace_id, "key": "outcome_win_rate", "score": round(sum(1 for o in closed if o.result == "won") / len(closed), 6), "value": None, "comment": f"{len(closed)} closed", "feedback_id": feedback_id(trace_id, "outcome_win_rate")})
    for o in outcomes:
        if o.result in ("won", "lost", "open"):
            rows.append({"run_id": trace_id, "key": "rec_outcome", "score": None, "value": o.result, "comment": f"{o.symbol} ({'traded' if o.traded else 'hypothetical'})", "feedback_id": feedback_id(trace_id, "rec_outcome", o.record_id or o.key)})
    return rows


def post_outcomes(trace_id: str | None, outcomes, client=None, *, root_trace_id: str | None = None, address=None) -> int:
    """Post the decision day's feedback on its span. Returns how many rows were sent: 0 when tracing is off, there is no trace id or LangSmith could not be reached. Never raises.

    `root_trace_id` and `address` are the trace the span belongs to and where it was sent (the run tree's `trace_id` and `address`): the SDK asks for them (`create_feedback`: a run-level feedback without a `session_id` is deprecated; the
    run's address stands for it, and with the trace id the feedback goes on the SDK's background queue instead of one request each). Without them the call is the one it always was."""
    if not trace_id or not tracing_is_enabled():
        return 0
    sent = 0
    try:
        if client is None:
            from langsmith.run_trees import get_cached_client
            client = get_cached_client()      # the SDK's own shared client (the one the runs were sent with): not a new connection for each day
        extra = {k: v for k, v in (("trace_id", root_trace_id), ("address", address)) if v is not None}
        for row in day_feedback(trace_id, outcomes):
            try:
                client.create_feedback(run_id=row["run_id"], key=row["key"], score=row["score"], value=row["value"], comment=row["comment"], feedback_id=row["feedback_id"], **extra)
                sent += 1
            except Exception:
                continue
    except Exception:
        return sent
    return sent


# ---------------------------------------------------------------- posting AFTER the run is over (it must not delay its result)

_POSTS: list = []
_POSTS_LOCK = threading.Lock()


def post_in_background(jobs: list, *, root_trace_id: str | None = None, address=None, client=None, workers: int = 8) -> int:
    """Post the feedback of every decision day in a pool of threads and return AT ONCE (how many jobs were queued): the run's result is already emitted and saved, so a slow LangSmith delays nothing. `jobs` is
    [(the decision day's span id, its outcomes)]; a job with no span id (tracing off) is skipped. Never raises, never fails a run, idempotent (the feedback ids are deterministic). `wait_for_posts` is how a command waits for them."""
    queued = [(trace_id, outcomes) for trace_id, outcomes in jobs if trace_id]
    if not queued or not tracing_is_enabled():
        return 0
    pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(queued))), thread_name_prefix="proving-feedback")
    futures = [pool.submit(contextvars.copy_context().run, post_outcomes, trace_id, outcomes, client, root_trace_id=root_trace_id, address=address) for trace_id, outcomes in queued]
    pool.shutdown(wait=False)
    with _POSTS_LOCK:
        _POSTS[:] = [f for f in _POSTS if not f.done()]      # a long-lived server never keeps the posts of runs long gone
        _POSTS.extend(futures)
    return len(futures)


def wait_for_posts(timeout: float | None = None) -> dict:
    """Wait for the feedback `post_in_background` queued (a command does before it exits, so its posts are not lost) and, when the SDK has a background queue, for it to send what it holds. Returns {jobs, rows, seconds, flushed}:
    the jobs that were queued, the rows LangSmith was sent, and how long this waited. Never raises."""
    began = time.perf_counter()
    with _POSTS_LOCK:
        futures, _POSTS[:] = list(_POSTS), []
    rows = 0
    for future in futures:
        try:
            rows += int(future.result(timeout=None if timeout is None else max(0.0, timeout - (time.perf_counter() - began))) or 0)
        except Exception:
            continue
    flushed = False
    if futures:
        try:
            from langsmith.run_trees import get_cached_client
            get_cached_client().flush()
            flushed = True
        except Exception:
            pass
    return {"jobs": len(futures), "rows": rows, "seconds": round(time.perf_counter() - began, 2), "flushed": flushed}
