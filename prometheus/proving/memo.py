"""The per-run memo: what does not change within a Time Machine run is fetched ONCE (design/proving_ground_spec.md section 7.6, P14, P2).

    memo = Memo()
    reading = memo.get("request_reading", prompt, lambda: manager.understand_request(prompt))     # once per run, however many decision days ask
    toolbox = DataToolbox(memo=memo)                                                              # `Financials.get` asks the memo: Finnhub's snapshot of a symbol is today's whatever the decision day is (P3)
    search = memo.wrap("symbol_search", symbols.search_finnhub)                                   # the symbol directory: today's, once per name (P3)

It lives for ONE run, is shared by the threads of the parallel research and is THREAD-SAFE: when two threads ask for the same key at once the call is made once and both get its answer (a lock for each key). A value is kept as it came
and every caller gets a DEEP COPY, so one day's reading can never change another's.

IT NEVER MEMOISES ANYTHING THAT DEPENDS ON THE DECISION DAY (P2): the names are a closed list (`NAMES`: the request's reading, its security screening, Finnhub's financials of a symbol and the symbol directory's lookup) and a key
that holds a date or a datetime is refused (ValueError): retrieval, filing evidence, bars and quotes are asked for again each day. `asked` records every (name, key) that was asked for, so a test can read what a run memoised and the page can
say how many calls the memo saved (`hits`). A failed call is not remembered: the next asker tries again. No model, no network of its own.
"""
import copy
import threading
from datetime import date, datetime
from typing import Callable

NAMES = ("request_reading", "request_screening", "financials", "symbol_search")      # what is the same whatever the decision day is


class Memo:
    def __init__(self):
        self._values: dict = {}
        self._locks: dict = {}
        self._guard = threading.Lock()
        self.asked: list[tuple[str, object]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _undated(value) -> bool:
        if isinstance(value, (date, datetime)):
            return False
        if isinstance(value, (tuple, list)):
            return all(Memo._undated(v) for v in value)
        if isinstance(value, dict):
            return all(Memo._undated(k) and Memo._undated(v) for k, v in value.items())
        return True

    def get(self, name: str, key, make: Callable[[], object]):
        """The value of `make()` for (`name`, `key`), made once. Raises ValueError for a name that is not on the list or a key that holds a date."""
        if name not in NAMES:
            raise ValueError(f"the memo keeps only {', '.join(NAMES)}, not {name!r}: whatever depends on the decision day is asked for again each day")
        if not self._undated(key):
            raise ValueError(f"the memo never keeps a dated call ({name!r}): a date is the decision day, and the answer changes with it")
        full = (name, key if isinstance(key, (str, int, tuple)) else repr(key))
        with self._guard:
            self.asked.append(full)
            lock = self._locks.setdefault(full, threading.Lock())
        with lock:
            with self._guard:
                if full in self._values:
                    self.hits += 1
                    return copy.deepcopy(self._values[full])
            value = make()
            with self._guard:
                self._values[full] = value
                self.misses += 1
            return copy.deepcopy(value)

    def wrap(self, name: str, function: Callable) -> Callable:
        """`function` with its answers kept by its arguments (each must be a string, a number, a bool or a tuple of them)."""
        def memoised(*args, **options):
            return self.get(name, (args, tuple(sorted(options.items()))), lambda: function(*args, **options))
        return memoised
