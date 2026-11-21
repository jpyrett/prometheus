"""The account routes of the web bridge (UI v1 spec 4.1 and 4.5) against the REAL local Postgres, on THROWAWAY accounts the tests create and delete (the three samples are read and never written: `tests/test_account_book.py`
pins their counts). The real Account Manager is built (it makes a model client and calls nothing). Skipped when the book is not reachable.
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from prometheus.application.web.app import create_app
from prometheus.persistence.accounts import SECRETS, open_account_store


@pytest.fixture(scope="module")
def store():
    if not SECRETS.is_file():
        pytest.skip("secrets/postgres_local.txt is not on this machine (run scripts/setup_postgres.sh)")
    try:
        opened = open_account_store()
        opened.ping()
    except Exception:
        pytest.skip("the local Postgres is not reachable")
    return opened


@pytest.fixture
def http(store):
    made = []
    web = TestClient(create_app())
    web.made = made
    yield web
    for account_id in made:
        try:
            store.delete_account(account_id)
        except Exception:
            pass


def new_account(http, **fields) -> dict:
    body = {"name": "Web Test Holder", "age": 52, "risk": "moderate", "mode": "ask", "cash": "10000.00", **fields}
    response = http.post("/api/accounts", json=body)
    assert response.status_code == 201, response.text
    http.made.append(response.json()["account"]["id"])
    return response.json()


def test_an_account_is_created_read_changed_funded_and_defunded_through_the_routes(http):
    view = new_account(http)
    account_id = view["account"]["id"]
    assert view["account"] == {"id": account_id, "name": "Web Test Holder", "age": 52, "risk": "moderate", "mode": "ask", "sample": False, "read_only": False}
    assert (view["cash"], view["invested"], view["total"], view["positions"]) == (10000.0, 0, 10000.0, [])
    assert http.get(f"/api/accounts/{account_id}").json()["cash"] == 10000.0
    changed = http.patch(f"/api/accounts/{account_id}", json={"name": "Renamed Holder", "age": 61, "risk": "aggressive", "mode": "auto"}).json()
    assert changed["account"] | {"id": 0} == {"id": 0, "name": "Renamed Holder", "age": 61, "risk": "aggressive", "mode": "auto", "sample": False, "read_only": False}
    assert http.post(f"/api/accounts/{account_id}/funds", json={"action": "add", "amount": "250.50"}).json()["cash"] == 10250.5
    assert http.post(f"/api/accounts/{account_id}/funds", json={"action": "remove", "amount": "50.50"}).json()["cash"] == 10200.0
    too_much = http.post(f"/api/accounts/{account_id}/funds", json={"action": "remove", "amount": "10200.01"})
    assert too_much.status_code == 422 and too_much.json()["error"] == "the account has $10,200.00 in cash: $10,200.01 cannot be removed"
    assert http.get(f"/api/accounts/{account_id}").json()["cash"] == 10200.0


def test_every_refusal_is_a_message_for_the_user_and_nothing_is_written(http):
    account_id = new_account(http)["account"]["id"]
    for call, text in ((lambda: http.post(f"/api/accounts/{account_id}/funds", json={"action": "add", "amount": 500}), "a string of dollars"),
                       (lambda: http.post(f"/api/accounts/{account_id}/funds", json={"action": "add", "amount": "1.005"}), "two decimal places"),
                       (lambda: http.post(f"/api/accounts/{account_id}/funds", json={"action": "steal", "amount": "5.00"}), "add or remove"),
                       (lambda: http.patch(f"/api/accounts/{account_id}", json={"risk": "low"}), "conservative, moderate, aggressive"),
                       (lambda: http.patch(f"/api/accounts/{account_id}", json={"age": 12}), "18 to 120"),
                       (lambda: http.patch(f"/api/accounts/{account_id}", json={"cash": "5"}), "cannot change cash"),
                       (lambda: http.post("/api/accounts", json={"name": "A", "age": 30, "risk": "moderate", "mode": "ask", "cash": 100}), "a string of dollars"),
                       (lambda: http.post("/api/accounts", json={"name": "A", "age": 30, "risk": "moderate", "mode": "maybe", "cash": "100.00"}), "ask, auto")):
        response = call()
        assert response.status_code == 422 and text in response.json()["error"], response.text
    after = http.get(f"/api/accounts/{account_id}").json()
    assert (after["cash"], after["account"]["risk"], after["account"]["age"]) == (10000.0, "moderate", 52)
    assert http.get("/api/accounts/999999999").status_code == 404


def test_a_sample_account_can_be_read_and_never_written(http, store):
    jacob = store.account_id_for_seed("jacob_sample")
    view = http.get(f"/api/accounts/{jacob}").json()
    assert view["account"]["sample"] is True and view["account"]["read_only"] is True and view["cash"] == 51.23
    for call in (http.patch(f"/api/accounts/{jacob}", json={"name": "Hacked"}), http.post(f"/api/accounts/{jacob}/funds", json={"action": "add", "amount": "5.00"}),
                 http.post("/api/runs", json={"account_id": jacob, "prompt": "Buy 2 shares of MSFT"})):
        assert call.status_code == 409 and call.json()["error"] == "sample accounts are read-only here"
    assert http.get(f"/api/accounts/{jacob}").json()["cash"] == 51.23


def test_a_run_needs_a_prompt_and_an_account_and_the_health_says_the_book_is_ok(http):
    assert http.get("/api/health").json()["book"] == "ok" and http.get("/api/health").json()["mode"] == "live"
    assert http.post("/api/runs", json={"account_id": 1, "prompt": "   "}).status_code == 422
    assert http.post("/api/runs", json={"account_id": 1, "prompt": "x" * 601}).json()["error"] == "the request is longer than 600 characters"
    assert http.post("/api/runs", json={"prompt": "Buy 2 shares of MSFT"}).status_code == 422
    assert http.post("/api/runs", json={"account_id": 999999999, "prompt": "Buy 2 shares of MSFT"}).status_code == 404


# ---------------------------------------------------------------- the Mode switch and the run (the paid run itself is stood in for by the `live_target` hook: no agent is built)

class Starts:
    """The `live_target` of `create_app`: it records what the run was given and the trade mode the Account Manager's book held at that moment, then ends the run. A run that is `hold`-ing waits to be released."""

    def __init__(self, store, hold: threading.Event | None = None):
        self.store, self.hold, self.seen = store, hold, []

    def __call__(self, run, params, accounts, question_timeout):
        self.seen.append({"params": dict(params), "saved_mode": self.store.get_user(params["account_id"]).trade_mode})
        if self.hold is not None:
            self.hold.wait(10)
        run.emit("run_finished", {"status": "looked", "headline": "stood in for the paid run", "reasons": [], "path": [], "seconds": {}, "total_seconds": 0})


def over(web: TestClient, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while web.app.state.runs.active_id() and time.monotonic() < deadline:
        time.sleep(0.02)


def test_a_run_saves_the_mode_on_the_screen_through_the_account_manager_before_it_starts(store):
    target = Starts(store)
    web = TestClient(create_app(live_target=target))
    account_id = web.post("/api/accounts", json={"name": "Mode Test Holder", "age": 40, "risk": "moderate", "mode": "ask", "cash": "1000.00"}).json()["account"]["id"]
    try:
        run = {"account_id": account_id, "prompt": "Buy 2 shares of MSFT"}
        assert web.post("/api/runs", json={**run, "mode": "auto"}).status_code == 202
        over(web)
        assert target.seen[-1]["saved_mode"] == "auto" and target.seen[-1]["params"]["trade_mode"] == "auto"          # saved BEFORE the run target was called
        assert web.get(f"/api/accounts/{account_id}").json()["account"]["mode"] == "auto"
        assert web.post("/api/runs", json=run).status_code == 202                                                       # no mode in the request: the account's own stays
        over(web)
        assert target.seen[-1]["saved_mode"] == "auto" and target.seen[-1]["params"]["trade_mode"] == "auto"
        assert web.post("/api/runs", json={**run, "mode": "ask"}).status_code == 202
        over(web)
        assert target.seen[-1]["saved_mode"] == "ask" and web.get(f"/api/accounts/{account_id}").json()["account"]["mode"] == "ask"
        bad = web.post("/api/runs", json={**run, "mode": "maybe"})
        assert bad.status_code == 422 and "ask, auto" in bad.json()["error"] and len(target.seen) == 3 and web.get("/api/health").json()["active_run"] is None
        assert web.get(f"/api/accounts/{account_id}").json()["account"]["mode"] == "ask"
    finally:
        store.delete_account(account_id)


def test_a_run_that_is_refused_because_another_is_active_saves_no_mode(store):
    release = threading.Event()
    target = Starts(store, release)
    web = TestClient(create_app(live_target=target))
    account_id = web.post("/api/accounts", json={"name": "Busy Test Holder", "age": 40, "risk": "moderate", "mode": "ask", "cash": "1000.00"}).json()["account"]["id"]
    try:
        assert web.post("/api/runs", json={"account_id": account_id, "prompt": "Buy 2 shares of MSFT"}).status_code == 202
        refused = web.post("/api/runs", json={"account_id": account_id, "prompt": "Buy 2 shares of MSFT", "mode": "auto"})
        assert refused.status_code == 409 and web.get(f"/api/accounts/{account_id}").json()["account"]["mode"] == "ask"
    finally:
        release.set()
        over(web)
        store.delete_account(account_id)


