"""The routes (UI v1 spec 4.1 and 4.2): validation, and calling the others. No logic beyond that.

    create_app(replay=Replay(...))        # a recorded session: no key, no network, no database
    create_app()                          # the live desk: the real agents and the Postgres account book

All under `/api`. JSON in and out; a refusal is `{"error": "<a message for the user>"}` with a 4xx status. The run's events are Server-Sent Events (`id`, `event`, `data`); a client that connects late, or reconnects
with `Last-Event-ID` (or `?after=`), gets every event after the one it names. In production the built page (`ui/dist`) is served at `/`.

MONEY (design choices entry 112): the views carry `Decimal`s, and every JSON the bridge sends, the event stream (`money.dumps`) and the account, funds and record routes (`money.jsonable`), writes a Decimal as a NUMBER, a whole value as an integer: the page receives the same
numbers as before and is not changed.
"""
import time
from pathlib import Path

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from prometheus.agents.account_manager.intent import MAX_TEXT
from prometheus.application.web import tracing, views
from prometheus.application.web.accounts import Accounts, ServiceError
from prometheus.application.web.replay import Replay
from prometheus.application.web.runs import QuestionClosed, RunBusy, RunManager
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


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(*, replay: Replay | None = None, accounts: Accounts | None = None, manager: RunManager | None = None, dev: bool = False, question_timeout: float = 600.0,
               live_target=None, static_dir: Path | None = UI_DIST, guard: Guard | None = None) -> FastAPI:
    """`live_target(run, params, accounts, question_timeout)` runs the live desk (`desk_run.run_live` by default, imported when first needed: it builds the real agents). `guard` (entry 116) counts the requests of `POST /api/runs`
    against the account's rate limit (HTTP 429 with the policy's message); `Guard.code_only()` by default: the rate limit needs no model."""
    app = FastAPI(title="Prometheus desk", docs_url=None, redoc_url=None, openapi_url=None)
    runs, accounts = manager or RunManager(), accounts or Accounts()
    guard = guard or Guard.code_only()
    app.state.runs, app.state.replay, app.state.accounts, app.state.guard = runs, replay, accounts, guard
    if dev:
        app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return error(422, "the request is not valid: " + "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()[:3]))

    @app.exception_handler(ServiceError)
    async def refused(request: Request, exc: ServiceError):
        return error(exc.status, exc.message)

    @app.exception_handler(ValueError)
    async def bad_value(request: Request, exc: ValueError):
        return error(422, str(exc))

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "replay" if replay else "live", "book": "ok" if replay else accounts.book_status(), "clock_note": clock_note() or "", "clock_pill": clock_pill(),
                "defaults": {"prompt": DEFAULT_PROMPT, "strict": False, "neutral": True, "fetch": True}, "limits": {"prompt_chars": MAX_TEXT},
                "active_run": runs.active_id(), "tracing": tracing.status(), "replay_file": replay.capture.get("_file") if replay else None}

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
                params = {"account_id": account_id, "prompt": " ".join(prompt.split()), "strict": bool(body.get("strict", False)), "neutral": bool(body.get("neutral", True)), "fetch": bool(body.get("fetch", False)), "trade_mode": trade_mode}
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

    # ---------------------------------------------------------------- the built page
    if static_dir and (Path(static_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return app
