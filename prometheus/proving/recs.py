"""The recommendations of a finished desk run, ready to be scored (design/proving_ground_spec.md sections 8, 10 and 11; P8).

    for capture in captures(store):                               # the saved sessions of the Track Record (source "run") and the real captures of the tests (source "fixture")
        order = order_of(capture)                                 # the capture's work order, as the desk made it
        for rec in recs_of(order, capture.payload.get("intent")): ...        # one `Rec` per recommendation, traded or not
    index = record_index(store)                                   # {record_id: (recommendation record, capture)}: the live monitor looks up the plan of the recommendation that bought a position

WHAT A RUN HAS TO SCORE. (1) Every recommendation the Advisor made (`order.recommendations`, a plan with a stop, a target and a time stop): traded when a decision bought it and the Trader filled it (the REAL fill is the entry),
hypothetical otherwise, with the reason (`denied` by the Critic, `no_cash`, `not_filled`, `failed`, `beyond_count`). (2) Every company the Advisor scored NEUTRAL that the Account Manager decided on (`origin: "neutral"`, a purchase whose
record id is `neutral-<request>-<n>`: the Advisor's neutral record is on `order.candidates` and has NO plan, so such an outcome has none either: it is followed to the last close and never closed by a stop or a target that was never
made). When no neutral purchase was decided at all (the setting was off) the top neutral candidates of the Advisor, as many as the request still wanted, are scored as hypothetical `neutral_off`: this is how the setting is judged.
(SINCE ADVISOR CONFIG VERSION 2 a neutral candidate has an `exit_plan` (a stop 2 ATR under the last close and a time stop 20 trading days after the decision day, no target): the record carries it and the outcome is followed by it, by the same
`simulator.follow`. A candidate of an earlier run has none and is followed to the last close, as before.)
The DECISION DAY of a run is the day of the Advisor's `as_of_utc` (the market day it recommended for: the simulated clock's previous business day included), else the day of the first fill, else the Eastern date the order was made. A
direct order has no recommendation and gives nothing. No model, no network.
"""
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from prometheus.agents.trader.record import Trade
from prometheus.common.market_calendar import eastern_date
from prometheus.common.work_order import WorkOrder, iso_date
from prometheus.proving import record
from prometheus.proving.outcomes import EntryFill, entry_fill_of
from prometheus.proving.store import ProvingStore
from prometheus.rag.config import REPO_ROOT

FIXTURE_FOLDERS = (REPO_ROOT / "tests" / "fixtures" / "session_real", REPO_ROOT / "tests" / "fixtures" / "proving_real")      # the real captures: the Track Record is not empty on day one (labelled source "fixture")
FIXTURE_PREFIXES = {"proving_real": "live_session"}                                                                           # in proving_real only the captures of live runs (the saved real runs of the Time Machine are not sessions)


@dataclass(frozen=True)
class Capture:
    source: str            # "run" (a run of the page or the session, saved by `sessions.save_session`) or "fixture" (a real capture kept with the tests)
    name: str              # the request id (a run) or the file name (a fixture)
    payload: dict
    mtime: int


@dataclass(frozen=True)
class Rec:
    """One recommendation of a run, with what became of it."""
    kind: str                           # "recommendation" (the Advisor's plan) | "neutral" (a company it scored neutral)
    record: dict                        # the Advisor's record, or a neutral candidate completed with the decision's record id
    record_id: str | None
    symbol: str
    traded: bool
    fill: EntryFill | None              # the REAL fill when traded
    trade: dict | None                  # the Trader's trade (a dict, as the work order has it)
    not_traded_reason: str | None
    decision: dict | None


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("order"), dict) and isinstance(value["order"].get("request_id"), str) else None


def captures(store: ProvingStore | None = None, *, fixtures: bool = True) -> list[Capture]:
    """Every capture that could have recommendations: the saved sessions of `store` (source "run", oldest first), then the real fixtures (source "fixture", by name). A file that is not a capture is left out."""
    found: list[Capture] = []
    if store is not None:
        folder = store.root / "sessions"
        for request_id in store.sessions():
            payload = store.load_session(request_id)
            if payload is not None and isinstance(payload.get("order"), dict):
                found.append(Capture("run", request_id, payload, (folder / f"{request_id}.json").stat().st_mtime_ns))
    if fixtures:
        for folder in FIXTURE_FOLDERS:
            prefix = FIXTURE_PREFIXES.get(folder.name, "")
            for path in sorted(folder.glob(f"{prefix}*.json")) if folder.is_dir() else []:
                payload = _read(path)
                if payload is not None:
                    found.append(Capture("fixture", path.name, payload, path.stat().st_mtime_ns))
    return found


def default_store_for_reading() -> ProvingStore:
    """The store the saved sessions are read from (the one `sessions.default_store` gives: a test points it at a temporary folder)."""
    from prometheus.proving import sessions
    return sessions.default_store()


def order_of(capture: Capture) -> WorkOrder:
    return WorkOrder.model_validate(capture.payload["order"])


def decision_day_of(order: WorkOrder) -> date | None:
    """The market day the run recommended for (see the module text); None when nothing in the order says."""
    for rec in order.recommendations:
        day = iso_date(str(((rec or {}).get("identity") or {}).get("as_of_utc", ""))[:10])
        if day is not None:
            return day
    fills = sorted(t["fill"]["date"] for t in order.trades if isinstance(t, dict) and (t.get("fill") or {}).get("date"))
    if fills and iso_date(fills[0]):
        return iso_date(fills[0])
    try:
        return eastern_date(datetime.fromisoformat(order.created_utc.replace("Z", "+00:00")).astimezone(timezone.utc))
    except (ValueError, AttributeError):
        return None


def _trade_of(order: WorkOrder, record_id: str | None) -> dict | None:
    return next((t for t in order.trades if isinstance(t, dict) and t.get("record_id") == record_id and t.get("side") == "buy"), None) if record_id else None


def _why_not(order: WorkOrder, record_id: str | None, decision: dict | None, trade: dict | None, neutral_off: bool = False) -> str:
    review = record.review_of(order, record_id) if record_id else None
    if neutral_off:
        return "neutral_off"
    if review is not None and review.get("approved") is not True:
        return "denied"
    will = (decision or {}).get("will_trade")
    if trade is not None:
        return "not_filled" if trade.get("status") in ("rejected", "unfilled", "expired") else "failed"
    if will == "no_funds":
        return "no_cash"
    if will in ("no_system_denied", "no_user_denied"):
        return "denied"
    if decision is None:
        return "beyond_count"
    return "failed"


def neutral_record_of(candidates: list, symbol: str, record_id: str) -> dict:
    """The Advisor's neutral candidate of `symbol` (a dict of `order.candidates`) shaped like a recommendation record, carrying the decision's record id. It has no trade plan (`trade` is empty); since Advisor config version 2 it has an
    `exit_plan` (a stop and a time stop, no target) and the record carries it, so that `record.plan_of` follows the position by it. A candidate saved before has none: such a position is followed to the last close and never closed."""
    candidate = next((c for c in candidates if isinstance(c, dict) and c.get("symbol") == symbol), {})
    found = {"symbol": symbol, "company": candidate.get("company"), "stance": candidate.get("stance", "neutral"), "score": candidate.get("score"), "ranking": {"rank": candidate.get("rank")},
             "identity": {"record_id": record_id}, "trade": {}}
    if isinstance(candidate.get("exit_plan"), dict):
        found["exit_plan"] = candidate["exit_plan"]
    return found


def _neutral_record(order: WorkOrder, symbol: str, record_id: str) -> dict:
    return neutral_record_of(order.candidates, symbol, record_id)


def recs_of(order: WorkOrder, intent: dict | None = None) -> list[Rec]:
    """The recommendations of a work order to be scored, in the order the Advisor ranked them, then the neutral ones (see the module text)."""
    found: list[Rec] = []
    decisions = {d.get("record_id"): d for d in order.trade_decisions if isinstance(d, dict)}
    for recommendation in order.recommendations:
        record_id = ((recommendation.get("identity") or {}).get("record_id")) or recommendation.get("record_id")
        decision, trade = decisions.get(record_id), _trade_of(order, record_id)
        fill = entry_fill_of(trade) if trade is not None else None
        found.append(Rec("recommendation", recommendation, record_id, recommendation.get("symbol"), fill is not None, fill, trade, None if fill is not None else _why_not(order, record_id, decision, trade), decision))
    neutral_decided = [d for d in order.trade_decisions if isinstance(d, dict) and d.get("origin") == "neutral"]
    for decision in neutral_decided:
        record_id, symbol = decision.get("record_id"), decision.get("symbol")
        trade = _trade_of(order, record_id)
        fill = entry_fill_of(trade) if trade is not None else None
        found.append(Rec("neutral", _neutral_record(order, symbol, record_id), record_id, symbol, fill is not None, fill, trade, None if fill is not None else _why_not(order, record_id, decision, trade), decision))
    if not neutral_decided and isinstance(intent, dict):
        wanted = intent.get("count") if isinstance(intent.get("count"), int) and not isinstance(intent.get("count"), bool) else 0
        approved = sum(1 for r in found if r.kind == "recommendation" and (record.review_of(order, r.record_id or "") or {}).get("approved") is True)
        neutral = sorted((c for c in order.candidates if isinstance(c, dict) and c.get("stance") == "neutral" and c.get("gate") is None and isinstance(c.get("score"), int)), key=lambda c: -c["score"])
        for candidate in neutral[:max(wanted - approved, 0)]:
            record_id = f"neutral-{order.request_id}-{candidate['symbol']}"
            found.append(Rec("neutral", _neutral_record(order, candidate["symbol"], record_id), record_id, candidate["symbol"], False, None, None, "neutral_off", None))
    return found


_INDEX: dict = {"stamp": None, "index": {}}      # the last index, kept while no capture file has changed


def record_index(store: ProvingStore | None = None) -> dict[str, tuple[dict, Capture]]:
    """{record_id: (the Advisor's recommendation record, the capture it is in)} over the saved sessions and the fixtures; the first capture to hold a record id keeps it (a run saved by the page comes before a fixture). A company the desk
    bought as a NEUTRAL one (`neutral-<request>-<n>`) whose candidate has an exit plan is in it too, as its candidate record (`neutral_record_of`). Kept while no capture file has changed (their names and mtimes)."""
    found = captures(store)
    stamp = (str(store.root) if store is not None else None, tuple((c.source, c.name, c.mtime) for c in found))
    if _INDEX["stamp"] != stamp:
        index: dict = {}
        for capture in found:
            order = capture.payload["order"]
            for recommendation in order.get("recommendations") or []:
                record_id = ((recommendation or {}).get("identity") or {}).get("record_id")
                if isinstance(record_id, str):
                    index.setdefault(record_id, (recommendation, capture))
            for decision in order.get("trade_decisions") or []:      # a company the Advisor scored NEUTRAL that the desk bought (`neutral-<request>-<n>`): its record is the candidate, when the candidate has an exit plan (one saved before has none: it is not here, and the Live view says the Advisor made no plan)
                if isinstance(decision, dict) and decision.get("origin") == "neutral" and isinstance(decision.get("record_id"), str) and isinstance(decision.get("symbol"), str):
                    found_record = neutral_record_of(order.get("candidates") or [], decision["symbol"], decision["record_id"])
                    if "exit_plan" in found_record:
                        index.setdefault(decision["record_id"], (found_record, capture))
        _INDEX.update(stamp=stamp, index=index)
    return _INDEX["index"]
