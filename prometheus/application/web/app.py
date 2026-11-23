"""The routes (UI v1 spec 4.1 and 4.2): validation, and calling the others. No logic beyond that.

    create_app(replay=Replay(...))        # a recorded session: no key, no network, no database
    create_app()                          # the live desk: the real agents and the Postgres account book

All under `/api`. JSON in and out; a refusal is `{"error": "<a message for the user>"}` with a 4xx status. The run's events are Server-Sent Events (`id`, `event`, `data`); a client that connects late, or reconnects
with `Last-Event-ID` (or `?after=`), gets every event after the one it names. In production the built page (`ui/dist`) is served at `/`.

MONEY (design choices entry 112): the views carry `Decimal`s, and every JSON the bridge sends, the event stream (`money.dumps`) and the account, funds and record routes (`money.jsonable`), writes a Decimal as a NUMBER, a whole value as an integer: the page receives the same
numbers as before and is not changed.
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from prometheus.agents.account_manager.intent import MAX_TEXT
from prometheus.application.web import tracing, views
from prometheus.application.web.accounts import Accounts, ServiceError
from prometheus.application.web.proving_run import FIXTURES as PROVING_FIXTURES, Proving, ProvingError
from prometheus.application.web.replay import Replay
from prometheus.application.web.runs import QuestionClosed, RunBusy, RunManager
from prometheus.application.web.watch import Notices
from prometheus.common.market_calendar import load_calendar
from prometheus.common.money import dumps, jsonable
from prometheus.common.user_account import clean_trade_mode
from prometheus.common.trading_clock import clock_note, trading_now, use_live_trading
from prometheus.security.guard import Guard

REPO_ROOT = Path(__file__).resolve().parents[3]
UI_DIST = REPO_ROOT / "ui" / "dist"
DEFAULT_PROMPT = "Buy an even mixture of 3 most recommended stocks in the pizza restaurants using $3000 of my available money"
KEEP_ALIVE_SECONDS = 15
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def clock_pill() -> str:
    """The short words of the header's clock pill when the clock is simulated (USE_LIVE_TRADING=false), else an empty text."""
    if use_live_trading():
        return ""
    from zoneinfo import ZoneInfo
    local = trading_now().astimezone(ZoneInfo(load_calendar().timezone))
    return f"SIMULATED CLOCK \u00b7 {local:%a %b %d} CLOSE".upper()


def active_kind(runs) -> str | None:
    """What the active run is: "proving" (a Time Machine run or the replay of a saved one), "desk" (a desk run or the replay of a session), or None when nothing runs: the page disables the other tab's Go!/Launch with the reason."""
    active = runs.active
    if runs.active_id() is None or active is None:
        return None
    return "proving" if active.kind == "proving" else "desk"


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(*, replay: Replay | None = None, accounts: Accounts | None = None, manager: RunManager | None = None, dev: bool = False, question_timeout: float = 600.0,
               live_target=None, static_dir: Path | None = UI_DIST, guard: Guard | None = None, proving: Proving | None = None, watch=None) -> FastAPI:
    """`live_target(run, params, accounts, question_timeout)` runs the live desk (`desk_run.run_live` by default, imported when first needed: it builds the real agents). `guard` (entry 116) counts the requests of `POST /api/runs`
    against the account's rate limit (HTTP 429 with the policy's message); `Guard.code_only()` by default: the rate limit needs no model.

    `watch` (design choices entry 140) is the WATCHER of the waiting limit orders (`watch.OrderWatch`: anything with `start()`, `stop()` and `notices`): it is started with the app and stopped with it (the lifespan: the app must be run as a server, or
    entered as a context by a test client), and it is NEVER built here. None, the default, is no watcher at all: a test or a replay builds the app without one, and a test that wants one passes one with a fake clock. The page's notices (a fill, an
    expiry, a cancel) are `watch.notices`, or a log of the app's own when there is no watcher: `GET /api/orders/events`."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if watch is not None:
            watch.start()
        try:
            yield
        finally:
            if watch is not None:
                watch.stop()

    app = FastAPI(title="Prometheus desk", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    runs, accounts = manager or RunManager(), accounts or Accounts()
    guard = guard or Guard.code_only()
    proving = proving or (Proving(fixtures=PROVING_FIXTURES, paid=False) if replay else Proving())      # --replay serves the saved real runs and refuses a paid one
    app.state.runs, app.state.replay, app.state.accounts, app.state.guard, app.state.proving = runs, replay, accounts, guard, proving
    notices = getattr(watch, "notices", None) or Notices()
    app.state.watch, app.state.notices = watch, notices
    if dev:
        app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return error(422, "the request is not valid: " + "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()[:3]))

    @app.exception_handler(ServiceError)
    async def refused(request: Request, exc: ServiceError):
        return error(exc.status, exc.message)

    @app.exception_handler(ProvingError)
    async def proving_refused(request: Request, exc: ProvingError):
        return error(exc.status, exc.message)

    @app.exception_handler(ValueError)
    async def bad_value(request: Request, exc: ValueError):
        return error(422, str(exc))

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "replay" if replay else "live", "book": "ok" if replay else accounts.book_status(), "clock_note": clock_note() or "", "clock_pill": clock_pill(),
                "defaults": {"prompt": DEFAULT_PROMPT, "strict": False, "neutral": True, "fetch": False, "raise_funds": False, "ignore_critic": False}, "limits": {"prompt_chars": MAX_TEXT},
                "active_run": runs.active_id(), "active_kind": active_kind(runs), "tracing": tracing.status(), "replay_file": replay.capture.get("_file") if replay else None}

    # ---------------------------------------------------------------- accounts
    def read_only_replay() -> JSONResponse:
        return error(409, "this is a recorded session: its account is read-only")

    @app.post("/api/accounts", status_code=201)
    def create_account(body: dict = Body(...)):
        return read_only_replay() if replay else jsonable(accounts.create(body))      # `jsonable`: the one encoder, a Decimal is a number (entry 112)

    @app.get("/api/accounts/{account_id}")
    def get_account(account_id: int):
        return jsonable(replay.account() if replay else accounts.view(account_id))

    @app.patch("/api/accounts/{account_id}")
    def patch_account(account_id: int, body: dict = Body(...)):
        return read_only_replay() if replay else jsonable(accounts.patch(account_id, body))

    @app.post("/api/accounts/{account_id}/funds")
    def funds(account_id: int, body: dict = Body(...)):
        return read_only_replay() if replay else jsonable(accounts.funds(account_id, body.get("action"), body.get("amount")))

    # ---------------------------------------------------------------- waiting limit orders (entry 140)
    @app.get("/api/accounts/{account_id}/orders")
    def account_orders(account_id: int):
        return {"orders": [], "reserved_cash": 0, "available": 0, "watching": False} if replay else jsonable(accounts.orders(account_id))

    @app.delete("/api/orders/{order_id}")
    def cancel_order(order_id: int):
        if replay:
            return read_only_replay()
        order = accounts.cancel_order(order_id)
        notices.add({"type": "cancelled", "order_id": order["order_id"], "account_id": order["account_id"], "symbol": order["symbol"], "side": order["side"],
                     "text": f"Order #{order['order_id']} cancelled: what it set aside is available again.", "at": order["closed_utc"]})
        return jsonable(order)

    @app.get("/api/orders/events")
    def order_events(after: int = Query(0, ge=0), once: bool = Query(False)):
        """The notices of the watcher and the cancel route as Server-Sent Events (`order`): every one after `after`, then the new ones as they come; `once=true` sends what there is and ends (the tests' and the recordings' handle)."""
        def stream():
            seen, beat = after, time.monotonic()
            while True:
                batch, _ = notices.read_from(seen, wait=0.0 if once else 1.0)
                for e in batch:
                    seen = e["seq"]
                    yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {dumps(e['data'])}\n\n"
                    beat = time.monotonic()
                if once:
                    return
                if time.monotonic() - beat >= KEEP_ALIVE_SECONDS:
                    yield ": keep-alive\n\n"
                    beat = time.monotonic()
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    # ---------------------------------------------------------------- runs
    @app.post("/api/runs", status_code=202)
    def start_run(body: dict = Body(...)):
        prompt = body.get("prompt")
        if not replay:
            if not isinstance(prompt, str) or not prompt.strip():
                return error(422, "type a request first")
            if len(prompt) > MAX_TEXT:
                return error(422, f"the request is longer than {MAX_TEXT} characters")
            account_id = body.get("account_id")
            if isinstance(account_id, bool) or not isinstance(account_id, int) or account_id < 1:
                return error(422, "open an account first (account_id must be an account number)")
            mode = body.get("mode")
            mode = None if mode is None else clean_trade_mode(mode)         # a bad value is a 422 (the book's own check)
            if (refused := guard.check_rate(account_id)) is not None:        # spam and SLA protection (entry 116): too many requests from one account in a minute or an hour, before the book is touched
                return error(429, refused.message)
            holder = accounts.writable(account_id)
            if runs.active_id():                                              # refused BEFORE anything is saved: a run that is not going to start changes nothing
                return error(409, "a run is already active: wait for it to finish, or stop it")
            trade_mode = holder.trade_mode
            if mode is not None and mode != trade_mode:                       # the Mode switch is the account's trade mode: saved through the Account Manager first, so the run uses the mode on the screen
                trade_mode = accounts.patch(account_id, {"mode": mode})["account"]["mode"]
        try:
            if replay:
                run = runs.start(replay.play, kind="replay")
            else:
                from prometheus.application.web import desk_run
                ignore_critic = bool(body.get("ignore_critic", False))      # entry 148, IC1: the box "Ignore Critic", one run, a missing value is false
                params = {"account_id": account_id, "prompt": " ".join(prompt.split()), "strict": bool(body.get("strict", False)) and not ignore_critic, "neutral": bool(body.get("neutral", True)), "fetch": bool(body.get("fetch", False)), "raise_funds": bool(body.get("raise_funds", False)),
                          "ignore_critic": ignore_critic, "trade_mode": trade_mode}      # Strict cannot go with Ignore Critic (IC9): the Critic does not run, so "only what the Critic approves" has nothing to say; `run_settings` says the same
                target = live_target or desk_run.run_live
                run = runs.start(lambda r: target(r, params, accounts, question_timeout), kind="live")
        except RunBusy:
            return error(409, "a run is already active: wait for it to finish, or stop it")
        return {"run_id": run.id}

    def find(run_id: str):
        run = runs.get(run_id)
        if run is None:
            raise ServiceError(404, f"no run {run_id}")
        return run

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(0, ge=0), last_event_id: str | None = Header(None)):
        run = find(run_id)
        start = after
        if last_event_id and last_event_id.isdigit():
            start = max(start, int(last_event_id))

        def stream():
            seen, beat = start, time.monotonic()
            while True:
                batch, over = run.read_from(seen, wait=1.0)
                for e in batch:
                    seen = e["seq"]
                    yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {dumps(e['data'])}\n\n"
                    beat = time.monotonic()
                if over:
                    return
                if time.monotonic() - beat >= KEEP_ALIVE_SECONDS:
                    yield ": keep-alive\n\n"
                    beat = time.monotonic()
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @app.post("/api/runs/{run_id}/answers", status_code=204)
    def answer(run_id: str, body: dict = Body(...)):
        run = find(run_id)
        question_id = body.get("question_id")
        if not isinstance(question_id, str) or question_id not in run.questions:
            return error(404, "no such question")
        value = body.get("answer")
        kind = run.questions[question_id].kind
        if kind == "clarify" and not isinstance(value, str):
            return error(422, "the answer to a how-much question is text")
        if kind != "clarify" and not isinstance(value, bool):
            return error(422, "the answer to this question is true or false")
        try:
            run.answer(question_id, value)
        except QuestionClosed:
            return error(409, "that question is closed")
        return Response(status_code=204)

    @app.post("/api/runs/{run_id}/stop", status_code=202)
    def stop(run_id: str):
        find(run_id).stop()
        return {"stopping": True}

    @app.get("/api/runs/{run_id}/records/{key}")
    def records(run_id: str, key: str):
        run = find(run_id)
        found = views.records_of(run.snapshot, key) if run.snapshot is not None else None
        if found is None:
            return error(404, f"no row {key} on this work order")
        return jsonable(found)


    # ---------------------------------------------------------------- the Proving Ground (design/proving_ground_spec.md section 9)
    @app.get("/api/proving/coverage")
    def proving_coverage():
        return jsonable(proving.coverage())

    @app.post("/api/proving/plan")
    def proving_plan(body: dict = Body(...)):
        return proving.plan(body)                                         # free: reads nothing with a model

    @app.post("/api/proving/runs", status_code=202)
    def proving_start(body: dict = Body(...)):
        if not proving.paid:
            return error(409, "this is a recorded session: a new Time Machine run is a paid run and is not made here")
        if (refused := guard.check_rate(None)) is not None:
            return error(429, refused.message)
        if runs.active_id():
            return error(409, "a run is already active: wait for it to finish, or stop it")
        try:
            run = proving.start(body, runs)
        except RunBusy:
            return error(409, "a run is already active: wait for it to finish, or stop it")
        return {"run_id": run.id}

    @app.get("/api/proving/runs")
    def proving_runs():
        return jsonable(proving.runs())

    @app.get("/api/proving/runs/{run_id}")
    def proving_run(run_id: str):
        return jsonable(proving.view(run_id))

    @app.get("/api/proving/runs/{run_id}/days/{day}")
    def proving_day(run_id: str, day: str):
        return proving.day(run_id, day)

    @app.get("/api/proving/runs/{run_id}/days/{day}/records/{key}")
    def proving_record(run_id: str, day: str, key: str):
        return proving.record(run_id, day, key)

    @app.post("/api/proving/runs/{run_id}/replay", status_code=202)
    def proving_replay(run_id: str, speed: int = Query(1, ge=0, le=10)):
        if runs.active_id():
            return error(409, "a run is already active: wait for it to finish, or stop it")
        try:
            run = proving.replay(run_id, runs, speed)
        except RunBusy:
            return error(409, "a run is already active: wait for it to finish, or stop it")
        return {"run_id": run.id}

    @app.delete("/api/proving/runs/{run_id}", status_code=204)
    def proving_delete(run_id: str):
        proving.delete(run_id)
        return Response(status_code=204)

    @app.get("/api/proving/track")
    def proving_track():
        return jsonable(proving.track())

    @app.get("/api/proving/critic-report")
    def proving_critic_report():
        return jsonable(proving.critic_report())                          # free: the Critic's rules scored against the saved outcomes (`proving/report_card.py`)

    @app.get("/api/proving/live/{account_id}")
    def proving_live(account_id: int):
        return proving.live(account_id, lambda n: accounts.manager(n))

    @app.get("/api/proving/live/{account_id}/events")
    def proving_live_events(account_id: int, once: bool = Query(False)):
        """One live snapshot every `poll_seconds` as Server-Sent Events (`snapshot`); `once=true` sends one and ends (the tests' and the recordings' handle)."""
        from prometheus.proving.config import load_config as load_proving_config

        def stream():
            while True:
                try:
                    shot = proving.live(account_id, lambda n: accounts.manager(n))
                except (ProvingError, ServiceError) as exc:
                    yield f"event: error\ndata: {dumps({'error': getattr(exc, 'message', str(exc))})}\n\n"
                    return
                yield f"event: snapshot\ndata: {dumps(shot)}\n\n"
                if once or not proving.paid:
                    return
                time.sleep(load_proving_config().live.poll_seconds)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    # ---------------------------------------------------------------- the built page
    if static_dir and (Path(static_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return app
