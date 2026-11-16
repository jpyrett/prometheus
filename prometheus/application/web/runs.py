"""One run at a time: its event log, its open questions, its stop flag (UI v1 spec 3 and 4.2). Knows nothing about agents.

    manager = RunManager()
    run = manager.start(target, kind="live")          # `target(run)` runs in a worker thread; raises RunBusy when a run is active
    run.emit("log", {...})                            # appends an event (numbered from 1) and wakes the readers
    answer = run.ask("confirm", {...}, timeout=600)   # blocks the worker thread until the page answers, the run is stopped or the time passes: None is "no answer", never a yes

A reader (`run.read_from(after)`) sees every event from the start, so a page reloaded in the middle of a run redraws everything. The worker's own errors become `run_finished` with `failed`: the kind of failure only
(`type(error).__name__`) unless it is a ValueError or RuntimeError of ours, whose message is for the user.
"""
import threading
import time
import uuid
from collections import OrderedDict
from typing import Callable

KEEP_FINISHED = 5          # the last few finished runs are kept in memory (a page can still read them)
POLL_SECONDS = 0.25        # how often a waiting question looks at the stop flag


class RunBusy(Exception):
    """A run is already active (D9: one run at a time)."""


class QuestionClosed(Exception):
    """The question was already answered, timed out or stopped."""


class Question:
    def __init__(self, question_id: str, kind: str):
        self.id, self.kind = question_id, kind
        self.answer = None
        self.closed = False
        self.by = None
        self.done = threading.Event()


class Run:
    def __init__(self, run_id: str, kind: str):
        self.id, self.kind = run_id, kind
        self.status = "running"                                     # running | finished
        self.events: list[dict] = []                                # {seq, type, data}
        self.cond = threading.Condition()
        self.questions: dict[str, Question] = {}
        self.stop_flag = threading.Event()
        self.snapshot: dict | None = None                           # the latest work order as a plain dict: what `/records/{key}` reads
        self.finished_sent = False
        self.meta: dict = {}

    # ---- the event log
    def emit(self, type_: str, data: dict) -> int:
        with self.cond:
            seq = len(self.events) + 1
            self.events.append({"seq": seq, "type": type_, "data": {**data, "ts": round(time.time(), 3)}})
            if type_ == "run_finished":
                self.finished_sent = True
            self.cond.notify_all()
            return seq

    def read_from(self, after: int = 0, wait: float = 1.0) -> tuple[list[dict], bool]:
        """The events with a number above `after`, waiting up to `wait` seconds for one, and whether the run is over (finished and nothing more to read)."""
        with self.cond:
            if len(self.events) <= after and self.status == "running":
                self.cond.wait(timeout=wait)
            batch = self.events[after:]
            return batch, self.status != "running" and len(self.events) <= after + len(batch)

    def finish(self) -> None:
        with self.cond:
            self.status = "finished"
            self.cond.notify_all()

    # ---- the page's answers
    def ask(self, kind: str, payload: dict, timeout: float = 600.0) -> object | None:
        """Emit a `question` and block until the page answers (the answer), the run is stopped, or `timeout` passes (None). A run that is already stopped does not ask: it answers None at once."""
        if self.stop_flag.is_set():
            return None
        question = Question(uuid.uuid4().hex[:10], kind)
        with self.cond:
            self.questions[question.id] = question
        self.emit("question", {"question_id": question.id, "kind": kind, **payload})
        deadline = time.monotonic() + timeout
        while not question.done.wait(POLL_SECONDS):
            if self.stop_flag.is_set():
                self._close(question, None, "stopped")
            elif time.monotonic() >= deadline:
                self._close(question, None, "timeout")
        return question.answer

    def answer(self, question_id: str, value) -> None:
        """The page's answer. Raises KeyError for a question this run never asked and QuestionClosed when it is no longer open (answered, timed out or stopped)."""
        question = self.questions[question_id]
        self._close(question, value, "you", strict=True)

    def _close(self, question: Question, value, by: str, strict: bool = False) -> None:
        with self.cond:
            if question.closed:
                if strict:
                    raise QuestionClosed(question.id)
                return
            question.closed, question.answer, question.by = True, value, by
        self.emit("question_closed", {"question_id": question.id, "answer": value, "by": by})
        question.done.set()

    def open_questions(self) -> list[str]:
        return [q.id for q in self.questions.values() if not q.closed]

    def stop(self) -> None:
        """Ask the run to stop: open and later questions answer "no answer", and the desk stops before its next stage. A trade already placed stays."""
        self.stop_flag.set()
        for question in list(self.questions.values()):
            self._close(question, None, "stopped")
        with self.cond:
            self.cond.notify_all()


class RunManager:
    def __init__(self, keep: int = KEEP_FINISHED):
        self._lock = threading.Lock()
        self.active: Run | None = None
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.keep = keep

    def start(self, target: Callable[[Run], None], kind: str = "live", **meta) -> Run:
        """Start `target(run)` in a worker thread. Raises RunBusy when a run is active. The lock is released and the run is finished however `target` ends."""
        with self._lock:
            if self.active is not None and self.active.status == "running":
                raise RunBusy("a run is already active")
            run = Run(uuid.uuid4().hex[:12], kind)
            run.meta.update(meta)
            self.active = run
            self.runs[run.id] = run
            while len(self.runs) > self.keep:
                oldest = next(iter(self.runs))
                if self.runs[oldest].status == "running":
                    break
                del self.runs[oldest]
        threading.Thread(target=self._work, args=(run, target), name=f"run-{run.id}", daemon=True).start()
        return run

    def _work(self, run: Run, target: Callable[[Run], None]) -> None:
        try:
            target(run)
        except BaseException as error:      # the worker never dies silently: the page is told
            if not run.finished_sent:
                text = str(error) if isinstance(error, (ValueError, RuntimeError)) and str(error) else f"the run failed ({type(error).__name__})"
                run.emit("run_finished", {"status": "failed", "headline": text, "reasons": [text], "path": [], "seconds": {}, "total_seconds": 0})
        finally:
            with self._lock:
                run.finish()
                if self.active is run:
                    self.active = None

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def active_id(self) -> str | None:
        run = self.active
        return run.id if run is not None and run.status == "running" else None
