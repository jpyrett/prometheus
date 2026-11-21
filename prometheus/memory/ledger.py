"""The recommendation ledger: an append-only record of every recommendation the Advisor wrote (Advisor v1 spec section 14, design choices entry 51).

    ledger = Ledger()                                   # data/ledger/recommendations.jsonl (the path is in configs/advisor_v1.json)
    ledger.append(record)                               # one JSON line, never rewritten
    ledger.recent("PG", family="capital_return", since=date(2026, 9, 1))      # the earlier records for the symbol, oldest first
    ledger.recent("PG", until=date(2026, 11, 9))                              # only those written on or before that day (the Proving Ground: a past day never reads a later line)
    ledger.problems                                     # lines that could not be read, in words

WHAT IT IS FOR. The Advisor reads it for one rule: the COOLDOWN (do not open the same symbol for the same family again while the first recommendation's time stop has not passed and its
invalidation rules have not fired; an older line whose action is `add` still counts: it is about what was recommended, not what is held). (The holdings pass, which also read it, is gone: entry 105.) It is also the first row of the long-term memory
the week 1 design promises: later, outcomes (the return after 5, 20 and 60 days against SPY, whether the stop was hit first) are added by a separate script from the bars, and
`basis: measured` replaces `prior` once a family has 30 outcomes. The generate step only APPENDS.

THE FILE. One JSON object per line, UTF-8. Append only: no line is changed or removed here. The folder is created on the first write. It is not `data/processed` (saved evidence the
docs cite) and not `data/historical`. A line that is not valid JSON is NOT swallowed and NOT fatal: it is named in `problems` (with its line number, never its content) and every other line is
still read, so one bad write cannot silently turn the cooldown off.
"""
import json
from datetime import date
from pathlib import Path

from prometheus.common import trading_clock
from prometheus.rag.config import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "data" / "ledger" / "recommendations.jsonl"


class Ledger:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else DEFAULT_PATH
        self.path = path if path.is_absolute() else REPO_ROOT / path
        self.problems: list[str] = []

    def append(self, record: dict) -> None:
        """Add one record as one line. The record must be JSON-serializable (a date or a pydantic dump is: `default=str`)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self) -> list[dict]:
        """Every record, oldest first. A line that is not a JSON object is added to `problems` and skipped. A file that does not exist is an empty ledger."""
        self.problems = []
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.problems.append(f"{self.path.name} line {number}: not valid JSON")
                    continue
                if not isinstance(record, dict):
                    self.problems.append(f"{self.path.name} line {number}: not a JSON object")
                    continue
                records.append(record)
        return records

    def recent(self, symbol: str, family: str | None = None, since: date | None = None, until: date | None = None) -> list[dict]:
        """The records of `symbol`, oldest first; with `family`, only those whose trade was driven by that family; with `since`, only those written on or after that date; with `until`, only those written on or
        before it. While a past trading day is bound (`trading_clock.as_of`, the Proving Ground) `until` defaults to that day and is never later: a ledger that holds lines from after the day (a real run made later,
        or the same test run before) shows none of them, whoever asks (the Advisor's cooldown, the Critic's history, the Trust Score). Unbound and left out: no upper bound, as before."""
        bound = trading_clock.bound_day()
        if bound is not None:
            until = bound if until is None else min(until, bound)
        out = []
        for record in self.read():
            if record.get("symbol") != symbol:
                continue
            if family is not None and (record.get("trade") or {}).get("family") != family:
                continue
            stamp = str((record.get("identity") or {}).get("as_of_utc", ""))[:10]
            if since is not None and stamp < since.isoformat():
                continue
            if until is not None and (not stamp or stamp > until.isoformat()):   # a line with no date cannot be placed before the day
                continue
            out.append(record)
        return out
