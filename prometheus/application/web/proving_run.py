"""The Proving Ground's side of the bridge (design/proving_ground_spec.md section 9): the worker thread of a Time Machine run, the replay of a saved run, and the service the routes call.

    proving = Proving()                                  # the store `data/proving`; Proving(fixtures=Path, paid=False) is the server's --replay mode
    proving.coverage(); proving.plan(body)               # free
    proving.start(body, runs)                            # a Time Machine run on the RunManager (kind "proving"): PAID. Refused in --replay mode with the replay's own sentence
    proving.replay(run_id, runs)                         # a saved run played through the same events: free
    proving.track()                                      # the Track Record, free
    proving.critic_report()                              # the report card of the Critic's rules, free
    proving.live(account_id, manager_for)                # one live snapshot (Finnhub quotes), never a write

A proving run goes through the SAME `RunManager` as a desk run, so the two exclude each other (one run at a time) and the existing events and stop routes serve it. The stream is `run_started`, then `pv_started`, `pv_research`, `pv_day`, `pv_decision`,
`pv_outcome` and `pv_finished` (the days are researched in parallel, so the desk's own `stage` events are NOT sent: each day reports through `pv_research` and a `log` line), and `run_finished` ends it as ever. `pv_decision` carries the
day's desk as `views.work_order_view` so the page shows the SAME work-order table and recommendation panel for any day. This module adds no rule and no number: the engine computes, `proving_views` shapes, the page draws.
"""
import time
from pathlib import Path

from prometheus.application.web import proving_views, views
from prometheus.application.web.emitter import Emitter, utc_now
from prometheus.application.web.runs import Run, RunManager
from prometheus.common.money import jsonable
from prometheus.proving import coverage as coverage_module
from prometheus.proving import engine, live as live_module, report_card, track
from prometheus.proving.config import load_config
from prometheus.proving.record import ProvingRun
from prometheus.proving.speed import speed_of
from prometheus.proving.store import SAFE_NAME, ProvingStore

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "proving_real"
REPLAY_SENTENCE = "this is a recorded session: a new Time Machine run is a paid run and is not made here"
REPLAY_LIVE = {8277: "live_snapshot_oil.json"}          # the recorded paper account; any other number shows the recorded throwaway account with plans
REPLAY_LIVE_OTHER = "live_snapshot_plan.json"


class ProvingError(Exception):
    """A refusal with the HTTP status it is answered with."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Proving:
    def __init__(self, store: ProvingStore | None = None, fixtures: Path | None = None, paid: bool = True):
        """`store`: where saved runs and the cache are (the default store when left out). `fixtures`: a folder of saved real runs served read-only (the server's --replay mode); `paid` False refuses a new Time Machine run."""
        self._store, self.fixtures, self.paid = store, Path(fixtures) if fixtures else None, paid

    @property
    def store(self) -> ProvingStore:
        if self._store is None:
            self._store = ProvingStore()
        return self._store

    # ---------------------------------------------------------------- free reads
    def coverage(self) -> dict:
        return coverage_module.compute()

    def plan(self, body: dict) -> dict:
        request = engine.TimeMachineRequest.model_validate(body)
        return jsonable(engine.plan(request, self.store, self.coverage()))

    def _fixture_runs(self) -> dict[str, ProvingRun]:
        found: dict[str, ProvingRun] = {}
        for path in sorted(self.fixtures.glob("*.json")) if self.fixtures and self.fixtures.is_dir() else []:
            try:
                run = ProvingRun.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if run.kind == "time_machine":
                found[run.run_id] = run
        return found

    def runs(self) -> list[dict]:
        """The index of saved runs, newest first (the store's, then the recorded fixtures)."""
        rows = [proving_views.run_row(r) for r in self.store.runs() if r.get("kind") == "time_machine"]
        seen = {r["run_id"] for r in rows}
        for run in self._fixture_runs().values():
            if run.run_id not in seen:
                rows.append(proving_views.run_row(self.store._row(run) if hasattr(self.store, "_row") else {}))
        return sorted(rows, key=lambda r: str(r.get("created_utc")), reverse=True)

    def load(self, run_id: str) -> ProvingRun:
        if not isinstance(run_id, str) or SAFE_NAME.fullmatch(run_id) is None:
            raise ProvingError(404, f"no saved run {run_id!r}")
        try:
            return self.store.load_run(run_id)
        except (FileNotFoundError, ValueError):
            fixture = self._fixture_runs().get(run_id)
            if fixture is None:
                raise ProvingError(404, f"no saved run {run_id}") from None
            return fixture

    def view(self, run_id: str) -> dict:
        return proving_views.run_view(self.load(run_id))

    def day(self, run_id: str, day: str) -> dict:
        found = proving_views.day_view(self.load(run_id), day)
        if found is None:
            raise ProvingError(404, f"the run {run_id} has no desk for {day}")
        return jsonable(found)

    def record(self, run_id: str, day: str, key: str) -> dict:
        found = proving_views.record_of(self.load(run_id), day, key)
        if found is None:
            raise ProvingError(404, f"no row {key} on the desk of {day}")
        return jsonable(found)

    def delete(self, run_id: str) -> None:
        if not isinstance(run_id, str) or SAFE_NAME.fullmatch(run_id) is None or not self.store.delete_run(run_id):
            raise ProvingError(404, f"no saved run {run_id}" + (": a recorded run cannot be deleted" if run_id in self._fixture_runs() else ""))

    def track(self) -> dict:
        run = track.build(self.store if self._store is not None else None)
        return proving_views.run_view(run)

    def critic_report(self) -> dict:
        """The report card of the Critic's rules (`proving/report_card.py`, Critic v2 rule C1): every saved Time Machine run (the store's and, in --replay mode, the recorded ones) and the Track Record, each finding code scored against the outcomes. Free; the shape is
        the card's own and the page only draws it (P12)."""
        runs = report_card.saved_runs(self.store, self.fixtures)
        return report_card.build([*runs, track.build(self.store if self._store is not None else None)], store=self.store)

    def live(self, account_id: int, manager_for) -> dict:
        if not self.paid and self.fixtures is not None:      # --replay: the recorded snapshots, no quote, no book
            name = REPLAY_LIVE.get(account_id, REPLAY_LIVE_OTHER)
            import json
            return json.loads((self.fixtures / name).read_text(encoding="utf-8"))
        return jsonable(live_module.snapshot(manager_for(account_id), account_id).model_dump(mode="json"))

    # ---------------------------------------------------------------- the runs
    def start(self, body: dict, runs: RunManager):
        """Start a Time Machine run (PAID). Raises ProvingError 409 in replay mode and ValueError for a request that is not valid or cannot run (a message for the user with the usable range). RunBusy when a run is active."""
        if not self.paid:
            raise ProvingError(409, REPLAY_SENTENCE)
        request = engine.TimeMachineRequest.model_validate(body)
        problems = engine.check(request, self.coverage())
        if problems:
            raise ValueError("; ".join(problems))
        return runs.start(lambda run: run_time_machine(run, request, self.store), kind="proving")

    def replay(self, run_id: str, runs: RunManager, speed: int = 1):
        saved = self.load(run_id)
        return runs.start(lambda run: play_saved(run, saved, speed), kind="proving")


def _started(run: Run, text: str, options: dict) -> None:
    run.emit("run_started", {"run_id": run.id, "mode": "proving", "kind": "time_machine", "prompt": text, "options": options, "started_utc": utc_now(), "stages": [], "trade_mode": "auto"})


def run_time_machine(run: Run, request: engine.TimeMachineRequest, store: ProvingStore) -> None:
    """The worker thread of a Time Machine run (PAID): the engine's events become the run's events, a `log` line per day goes to the drawer, and `run_finished` ends the stream."""
    out, began = Emitter(run), time.perf_counter()
    _started(run, request.prompt, {"strict": request.strict, "neutral": request.neutral and not request.strict, "fetch": False, "cadence": request.cadence, "risk": request.risk, "speed": speed_of(request.speed, load_config())})
    out.log("you", request.prompt)

    def emit(type_: str, data: dict) -> None:
        if type_ == "pv_decision":
            day = data["day"]
            run.snapshot = data["order"]
            run.emit("pv_decision", {"day": day, "view": views.work_order_view(data["order"], understood=data.get("understood")), "understood": data.get("understood")})
            return
        run.emit(type_, jsonable(data))
        if type_ == "pv_research":
            out.log("advisor", f"{data['day']}: {data['status']}, {data['recommended']} recommended, {data['approved']} approved, {data['denied']} denied ({data['seconds']} s)" + (f": {data['reason']}" if data.get("reason") else ""))
    try:
        saved = engine.run_time_machine(request, store=store, emit=emit, should_stop=run.stop_flag.is_set)
    except engine.RefusedRequest as error:
        out.finished("failed", str(error), [str(error)], [], {}, time.perf_counter() - began, agent="system")
        return
    timing = saved.coverage.get("timing") or {}
    out.finished("stopped" if saved.status == "stopped" else "finished", saved.stats.headline.sentence, [], [], {"research": timing.get("research_seconds", 0), "walk": timing.get("walk_seconds", 0)}, time.perf_counter() - began, agent="system")


def play_saved(run: Run, saved: ProvingRun, speed: int = 1) -> None:
    """The replay of a saved run: the same events, paced (`speed` seconds for each walk day; 0 plays at once). Free: no model, no quote, no book."""
    out, began = Emitter(run), time.perf_counter()
    _started(run, (saved.request or {}).get("prompt", ""), {"strict": bool((saved.request or {}).get("strict")), "neutral": bool((saved.request or {}).get("neutral")), "fetch": False, "replay_of": saved.run_id, "speed": saved.speed})
    out.log("system", f"REPLAY of the saved run {saved.run_id}: no model is called and no quote is read; the pacing is the replay's own.")
    for type_, data, pause in proving_views.replay_events(saved):
        if run.stop_flag.is_set():
            out.finished("stopped", "Stopped: the saved run was cut short.", [], [], {}, time.perf_counter() - began, agent="system")
            return
        if type_ == "pv_decision":
            run.snapshot = data["order"]
            data = {"day": data["day"], "view": views.work_order_view(data["order"], understood=data.get("understood")), "understood": data.get("understood")}
        run.emit(type_, jsonable(data))
        if pause and speed > 0:
            time.sleep(speed / 20)
    out.finished("finished", saved.stats.headline.sentence, [], [], {}, time.perf_counter() - began, agent="system")
