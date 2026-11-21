"""What the page shows, from the REAL captures of real sessions (UI v1 spec 4.3.1, 4.4, 4.5; design choices entry 99). `views.py` is pure: it reads a work order and writes rows, stages and an account.

No stand-in anywhere: the captures are `tests/fixtures/session_real/` (`bob_jones_oil.json`, `pam_smith_neutral.json`, `bill_gates_direct_order_*.json`), made by real runs of the real agents.
"""
from decimal import Decimal
import json
from pathlib import Path


from prometheus.application.web import views
from prometheus.application.web.replay import Replay, load_capture
from prometheus.common.work_order import WorkOrder

SESSIONS = Path(__file__).parent / "fixtures" / "session_real"


def capture(name: str) -> dict:
    return json.loads((SESSIONS / f"{name}.json").read_text(encoding="utf-8"))


def rows_of(name: str) -> dict:
    data = capture(name)
    view = views.work_order_view(views.cut_order(data["order"], data["path"]), understood="x")
    return {r["key"]: r for r in view["rows"]}


def test_bob_jones_oil_has_three_recommendations_two_filled_and_the_scored_rest():
    rows = rows_of("bob_jones_oil")
    xom, apa, eog = rows["XOM"], rows["APA"], rows["EOG"]
    assert [r["group"] for r in (xom, apa, eog)] == ["recommended"] * 3 and (xom["rank"], apa["rank"], eog["rank"]) == (1, 2, 3)
    assert xom["critic"] == {"approved": True, "score": 86, "grade": "B"} and xom["advisor"]["score"] == 73 and xom["advisor"]["stance"] == "bullish"
    assert (xom["decision"]["words"], xom["decision"]["shares"], xom["decision"]["est_cost"]) == ("yes (AUTO)", 8, 1364.72)
    assert xom["trade"] == {"status": "filled", "side": "buy", "shares": 8, "price": 170.6753, "amount": 1365.4}
    assert apa["critic"]["approved"] is False and apa["critic"]["score"] == 49 and apa["decision"]["will_trade"] == "no_system_denied" and apa["trade"] is None
    assert eog["critic"]["score"] == 76 and eog["trade"]["status"] == "filled" and eog["trade"]["price"] == 158.3992
    scored = [r for r in rows.values() if r["group"] == "scored"]
    assert len(rows) == 15 and len(scored) == 12 and scored[0]["key"] == "CVX" and scored[0]["reason"] == "not recommended: rank 4 is beyond the top 3" and scored[0]["critic"] is None and scored[0]["decision"] is None


def test_the_detail_of_a_recommended_row_carries_every_section_the_panel_draws_and_a_scored_row_only_its_own():
    rows = rows_of("bob_jones_oil")
    detail = rows["XOM"]["detail"]
    assert detail["header"] == {"score": 73, "stance": "bullish", "rank": 1, "of": 15, "strategy": "trend"}
    assert detail["thesis"].startswith("Buy XOM using a trend strategy") and len(detail["bull"]) == 3 and len(detail["bear"]) == 2 and detail["bull"][0]["ref"] == "[1]"
    assert [(b["label"], b["weight"]) for b in detail["score_build"]] == [("News", 40), ("Performance", 30), ("Historical", 20), ("Filings", 10)]
    assert detail["plan"]["entry_zone"] == "$168.44 – $172.74" and detail["plan"]["order"] == "Limit $171.67" and detail["plan"]["time_stop"] == "Feb 10, 2027" and detail["plan"]["horizon"] == "63 sessions"
    assert {"label": "Last close", "value": "$170.59"} in detail["key_numbers"] and {"label": "RSI 14", "value": "48.2"} in detail["key_numbers"]
    review = detail["review"]
    assert [d["name"] for d in review["dimensions"]] == ["accuracy", "evidence", "risk_fit", "timing", "value", "chart"] and review["dimensions"][0]["points"] == 30.0 and review["risk"]["level"] == "low"
    assert [w["code"] for w in review["warnings"]] == ["source_not_tagged"] and review["blockers"] == []
    assert detail["order"]["decision"]["words"] == "yes (AUTO)" and detail["order"]["trade"]["status"] == "filled"
    assert rows["APA"]["detail"]["review"]["blockers"][0]["code"] == "risk_above_profile"
    scored = rows["CVX"]["detail"]
    assert "review" not in scored and "order" not in scored and "thesis" not in scored and [s["name"] for s in scored["strategies"]][:2] == ["trend", "event"] and scored["strategies"][0]["passed"] is True


def test_pam_smith_neutral_passes_are_neutral_rows_with_their_origin_and_leave_the_scored_group():
    rows = rows_of("pam_smith_neutral")
    assert [(k, r["group"]) for k, r in rows.items()] == [("SNDK", "recommended"), ("MU", "neutral"), ("AMD", "neutral")]
    mu = rows["MU"]
    assert mu["decision"]["origin"] == "neutral" and mu["decision"]["will_trade"] == "yes_auto" and mu["decision"]["shares"] == 32 and mu["critic"] is None and mu["advisor"]["stance"] == "neutral"
    assert mu["trade"]["status"] == "filled" and "counts as approved" in mu["detail"]["message"] and "review" not in mu["detail"]
    assert rows["SNDK"]["critic"]["approved"] is False and rows["SNDK"]["decision"]["will_trade"] == "no_system_denied"


def test_a_direct_order_is_one_direct_row_with_no_advisor_and_no_critic():
    data = capture("bill_gates_direct_order_funded")
    view = views.work_order_view(views.cut_order(data["order"], data["path"]), understood=None)
    assert view["kind"] == "direct" and len(view["rows"]) == 1
    row = view["rows"][0]
    assert (row["group"], row["symbol"], row["company"], row["advisor"], row["critic"]) == ("direct", "MSFT", "Microsoft Corp", None, None)
    assert row["decision"]["origin"] == "direct" and row["decision"]["will_trade"] == "yes_user" and row["trade"]["status"] == "filled" and row["trade"]["shares"] == 50
    assert row["detail"]["message"].startswith("Your own order: no recommendation is made and the Critic is not asked") and row["detail"]["your_order"] == "buy 50 shares of MSFT"
    assert not {"thesis", "plan", "review", "score_build"} & row["detail"].keys()
    unfunded = capture("bill_gates_direct_order")
    refused = views.work_order_view(views.cut_order(unfunded["order"], unfunded["path"]))["rows"][0]
    assert refused["decision"]["will_trade"] == "no_funds" and refused["trade"] is None


def test_each_node_shows_only_what_it_had_written():
    data = capture("bob_jones_oil")
    after = {node: views.work_order_view(views.cut_order(data["order"], data["path"][:n])) for n, node in enumerate(data["path"], 1)}
    assert after["account_manager"]["rows"] == []
    assert {r["group"] for r in after["data_clerk"]["rows"]} == {"found"} and len(after["data_clerk"]["rows"]) == 15
    assert after["advisor"]["status"] == "recommended" and after["advisor"]["rows"][0]["critic"] is None and after["advisor"]["rows"][0]["decision"] is None
    assert after["critic"]["rows"][0]["critic"]["score"] == 86 and after["critic"]["rows"][0]["decision"] is None
    assert after["account_manager_decision"]["rows"][0]["decision"]["will_trade"] == "yes_auto" and after["account_manager_decision"]["rows"][0]["trade"] is None
    assert after["trader"]["status"] == "traded" and after["trader"]["rows"][0]["trade"]["status"] == "filled"


def test_stage_states_follow_the_path_and_say_skipped_failed_waiting_and_not_reached():
    bob = capture("bob_jones_oil")
    every = lambda states: [s["state"] for s in states]
    done = views.stage_states(path=bob["path"], seconds={"data_clerk": 37.2, "critic": 45.3})
    assert every(done) == ["done"] * 6 and done[1]["seconds"] == 37.2 and [s["key"] for s in done] == ["work_order", "gather", "recommend", "critique", "prepare", "trade"]
    assert every(views.stage_states(path=bob["path"][:3], running="critic")) == ["done", "done", "done", "busy", "pending", "pending"]
    assert every(views.stage_states(path=bob["path"][:3], running="critic", waiting=True))[3] == "waiting"
    direct = capture("bill_gates_direct_order_company_auto")
    assert every(views.stage_states(path=direct["path"])) == ["done", "skipped", "skipped", "skipped", "done", "done"]
    neutral = ["account_manager", "data_clerk", "advisor", "account_manager_decision", "trader"]          # a neutral pass: no Critic
    assert every(views.stage_states(path=neutral))[3] == "skipped"
    stopped = WorkOrder.model_validate({**bob["order"], "status": "data_failed"})
    assert every(views.stage_states(path=["account_manager", "data_clerk"], order=stopped))[1] == "failed"
    could_not = WorkOrder.model_validate({**bob["order"], "errors": ["the Advisor could not start: a missing key"]})
    assert every(views.stage_states(path=["account_manager", "data_clerk", "advisor"], order=could_not))[2] == "failed"
    assert every(views.stage_states(path=[], request_errors=["the request could not be read"]))[0] == "failed"
    assert every(views.stage_states(path=["account_manager", "data_clerk"], failed={"advisor"}, running=None))[2] == "failed"


def test_the_account_view_is_shares_times_average_cost_and_the_cash_and_the_changes_against_the_start():
    replay = Replay(load_capture(SESSIONS / "bob_jones_oil.json"))
    before, after = replay.account(), replay.account(final=True)
    assert (before["cash"], before["invested"], before["total"], before["positions"]) == (Decimal("15000.00"), 0, Decimal("15000.00"), [])
    assert after["cash"] == Decimal("12367.41") and after["invested"] == Decimal("2632.59") and after["total"] == Decimal("15000.0")
    assert {(p["symbol"], p["shares"], p["value"]) for p in after["positions"]} == {("XOM", 8, Decimal("1365.4")), ("EOG", 8, Decimal("1267.19"))} and {p["symbol"]: p["name"] for p in after["positions"]} == {"XOM": "ExxonMobil", "EOG": None}
    assert views.portfolio_changes(before, after) == {"changes": {"EOG": 8, "XOM": 8}, "cash_change": Decimal("-2632.59"), "cash_before": Decimal("15000.0")}
    assert before["account"] == {"id": 0, "name": "Bob Jones", "age": None, "risk": "conservative", "mode": "auto", "sample": False, "read_only": True}


def test_the_records_of_a_row_are_the_raw_records_and_a_row_that_is_not_there_is_none():
    data = capture("bob_jones_oil")
    found = views.records_of(data["order"], "XOM")
    assert set(found) == {"recommendation", "review", "decision", "trade"} and found["review"]["score"] == 86 and found["trade"]["fill"]["price"] == 170.6753
    assert set(views.records_of(data["order"], "CVX")) == {"candidate"}
    assert views.records_of(data["order"], "ZZZZ") is None
    neutral = capture("pam_smith_neutral")
    assert set(views.records_of(neutral["order"], "MU")) == {"candidate", "decision", "trade"}


def test_a_review_of_critic_version_3_shows_its_three_groups_and_the_analysts_in_one_line_and_an_older_review_shows_neither():
    """The page's review part (design/critic_v3_two_reasons_spec.md section 5: "the Critic's card shows the three groups (bad / clean) and the analyst counts in one line each; no new rule on the page"). The REAL PG_open record reviewed
    by the Critic with the REAL Finnhub counts of the capture; the old review is the real one of `bob_jones_oil`, made before the groups existed."""
    from critic_helpers import FakeClerk, captured, review, tampered_order
    counts = json.loads((Path(__file__).parent / "fixtures" / "finnhub_recommendation_trends.json").read_text(encoding="utf-8"))["symbols"]["PG"]
    periods = [{"period": r["period"], **{f: r[f] for f in ("strongBuy", "buy", "hold", "sell", "strongSell")}} for r in counts]
    order = tampered_order("PG_open", **{"trade.family": "trend"})
    new = review(order, clerk=FakeClerk(order, own={}, analysts={"PG": periods}, changes={"PG": {"ret_5d": -0.021}}), own={})["PG"]
    part = views._review_part(new)
    assert [(g["name"], g["label"], g["bad"], g["codes"]) for g in part["groups"]] == [("news", "News", False, []), ("performance", "Performance", True, ["trend_not_confirmed"]), ("price", "Price", False, [])]
    assert part["analysts"] == "6 strong buy, 14 buy, 14 hold, 1 sell, 0 strong sell in 2026-11-01 (35 analysts): mean 3.71 of 5, down 0.04 since 2026-08-01" and part["approved"] is True
    assert [d["name"] for d in part["dimensions"]][-1] == "analysts" and part["dimensions"][-1]["label"] == "Analysts" and part["pattern_bonus"] == new["pattern_bonus"]
    old = views._review_part(capture("bob_jones_oil")["order"]["reviews"][0])
    assert old["groups"] == [] and old["analysts"] is None and old["pattern_bonus"] == 0 and len(old["dimensions"]) == 6
