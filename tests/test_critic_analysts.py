"""Critic v3, section 3 (design/critic_v3_two_reasons_spec.md): the analysts' recommendation counts. The dimension `analysts` (weight 10) and the news sign `analysts_bearish`, on REAL Finnhub answers
(`tests/fixtures/finnhub_recommendation_trends.json`: six symbols, four months each, captured 2026-11-20 by `scripts/capture_finnhub_recommendations.py`), the Data Clerk's door for them (`request_analyst_ratings`, the toolbox's
fetch and its store, point in time) and the whole review.

The counts of the sign's tests (a sell-heavy street) are INPUTS for the rule, not market data: every one of the six real symbols leans buy.
"""
import json
from datetime import date

import pytest
from critic_helpers import CONFIG, FakeClerk, captured, review, tampered_order

from prometheus.agents.critic import analysts as an
from prometheus.agents.data_clerk import data_toolbox as dt
from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.agents.data_clerk.data_toolbox import DataToolbox
from prometheus.common import trading_clock as tc
from prometheus.common import work_order as wo
from prometheus.rag.config import REPO_ROOT

CAPTURE = json.loads((REPO_ROOT / "tests" / "fixtures" / "finnhub_recommendation_trends.json").read_text(encoding="utf-8"))
REAL = CAPTURE["symbols"]
A = CONFIG.analysts
FIELDS = ("strongBuy", "buy", "hold", "sell", "strongSell")


def periods(symbol):
    """The real rows of one symbol as the Data Clerk gives them: newest first, each {period, strongBuy, buy, hold, sell, strongSell}."""
    return sorted(({"period": r["period"], **{f: r[f] for f in FIELDS}} for r in REAL[symbol]), key=lambda r: r["period"], reverse=True)


def mean(row):
    return (5 * row["strongBuy"] + 4 * row["buy"] + 3 * row["hold"] + 2 * row["sell"] + 1 * row["strongSell"]) / sum(row[f] for f in FIELDS)


def row(period="2026-11-01", strong_buy=0, buy=0, hold=0, sell=0, strong_sell=0):
    return {"period": period, "strongBuy": strong_buy, "buy": buy, "hold": hold, "sell": sell, "strongSell": strong_sell}


# ---------------------------------------------------------------- the score, on the real answers

def test_the_capture_is_six_symbols_of_four_real_months_and_every_one_leans_buy():
    assert set(REAL) == {"XOM", "PG", "SNDK", "NVDA", "AMD", "KO"} and CAPTURE["endpoint"] == dt.FINNHUB_RECOMMENDATION_URL and all(len(rows) == 4 for rows in REAL.values())
    assert all(r["symbol"] == s and set(FIELDS) <= set(r) for s, rows in REAL.items() for r in rows)


@pytest.mark.parametrize("symbol", sorted(REAL))
def test_the_dimension_is_the_means_place_between_strong_sell_and_strong_buy(symbol):
    """`mean = (5 strongBuy + 4 buy + 3 hold + 2 sell + 1 strongSell) / n` of the newest month and `score = 100 (mean - 1) / 4`, on each real answer; the change of the mean is against the month three months earlier, shown and not scored."""
    rows = periods(symbol)
    newest, earlier = rows[0], next(r for r in rows if r["period"] == "2026-08-01")
    r = an.assess(symbol, rows, A)
    assert r.applicable and r.detail.period == "2026-11-01" and r.detail.n == sum(newest[f] for f in FIELDS) >= A.min_analysts
    assert r.detail.mean == pytest.approx(mean(newest), abs=1e-4) and r.score == pytest.approx(100 * (mean(newest) - 1) / 4, abs=0.01) and r.detail.score == r.score
    assert r.detail.counts == {f: newest[f] for f in FIELDS} and r.detail.compared_period == "2026-08-01" and r.detail.mean_change == pytest.approx(mean(newest) - mean(earlier), abs=1e-4)
    assert [m.name for m in r.measures] == ["analyst_mean", "mean_change"] and r.measures[0].weight == 100 and r.measures[1].weight == 0 and f"in 2026-11-01 ({r.detail.n} analysts)" in r.measures[0].note
    assert r.findings == [] and 3.5 < r.detail.mean < 4.8                                   # none of the six leans against the stock


def test_the_newest_usable_month_counts_whatever_order_they_come_in_and_a_bad_row_is_not_a_month():
    rows = periods("XOM")
    shuffled = [rows[2], rows[0], {"period": "2026-12-01", "strongBuy": "x", "buy": 1, "hold": 1, "sell": 1, "strongSell": 1}, {"period": "not a date", **{f: 1 for f in FIELDS}}, rows[3], rows[1],
                {"period": "2026-10-15", "strongBuy": True, "buy": 1, "hold": 1, "sell": 1, "strongSell": 1}, {"period": "2026-09-15", "strongBuy": -1, "buy": 1, "hold": 1, "sell": 1, "strongSell": 1}]
    got = an.assess("XOM", shuffled, A)
    assert got.detail.period == "2026-11-01" and got.detail.mean == an.assess("XOM", rows, A).detail.mean


def test_without_a_usable_month_or_with_too_few_analysts_the_dimension_is_not_applicable_and_asks_for_nobody():
    for none in (None, [], [row(strong_buy="many")]):
        r = an.assess("PG", none, A)
        assert not r.applicable and r.findings == [] and r.score == 0.0 and r.detail.applicable is False and r.detail.note == "no analyst ratings the day could know" and not r.measures[0].available
    few = an.assess("PG", [row(strong_buy=1, buy=1)], A)                                    # 2 analysts: under `min_analysts` (3)
    assert not few.applicable and few.detail.n == 2 and few.detail.period == "2026-11-01" and "only 2 analyst(s)" in few.detail.note and few.findings == []
    assert an.assess("PG", [row(strong_buy=1, buy=1, hold=1)], A).applicable


def test_a_change_needs_the_month_three_months_earlier_and_enough_analysts_in_it():
    rows = periods("PG")
    no_aug = an.assess("PG", rows[:3], A)                                                   # 2026-11, 10, 09: no August
    assert no_aug.applicable and no_aug.detail.mean_change is None and no_aug.measures[1].available is False and "no month 3 months earlier" in no_aug.measures[1].note
    thin_aug = [rows[0], rows[1], rows[2], row("2026-08-01", strong_buy=1)]
    assert an.assess("PG", thin_aug, A).detail.mean_change is None
    assert an.months_before("2026-01-01", 3) == (2025, 10) and an.months_before("2026-11-01", 3) == (2026, 8) and an.months_before("garbage", 3) is None


# ---------------------------------------------------------------- the sign

def test_the_sign_fires_for_a_sell_heavy_street_or_a_fall_of_half_a_point_and_it_is_a_warning_of_the_analysts_dimension():
    lean = an.assess("PG", [row(buy=2, hold=6, sell=10, strong_sell=4)], A)                    # INPUTS for the rule: mean 2.27, at or under 2.5
    [f] = lean.findings
    assert (f.code, f.severity, f.dimension, f.requires_human) == ("analysts_bearish", "warning", "analysts", False) and "2.27 of 5" in f.message and "10 sell" in f.message and lean.applicable
    edge = an.assess("PG", [row(hold=2, sell=2)], A)                                        # exactly 2.5 (the mean of 3, 3, 2, 2): the bound is "at or under"
    assert edge.detail.mean == 2.5 and [f.code for f in edge.findings] == ["analysts_bearish"]
    fell = an.assess("PG", [row("2026-11-01", strong_buy=2, buy=10, hold=10), row("2026-08-01", strong_buy=10, buy=10, hold=2)], A)       # 3.82 against 4.36: down 0.54 in three months, though the street still leans buy
    assert fell.detail.mean > A.bearish_mean and fell.detail.mean_change <= A.bearish_drop and "fell" in fell.findings[0].message and "since 2026-08-01" in fell.findings[0].message
    both = an.assess("PG", [row("2026-11-01", buy=2, hold=6, sell=10, strong_sell=4), row("2026-08-01", strong_buy=4, buy=10, hold=8)], A)
    assert "at or under 2.5" in both.findings[0].message and "fell" in both.findings[0].message and len(both.findings) == 1
    assert not an.assess("PG", [row("2026-11-01", strong_buy=2, buy=10, hold=10), row("2026-08-01", strong_buy=4, buy=10, hold=8)], A).findings    # down 0.18: not enough


def test_none_of_the_six_real_answers_is_bearish_and_the_real_months_show_small_changes():
    for symbol in REAL:
        r = an.assess(symbol, periods(symbol), A)
        assert r.findings == [] and abs(r.detail.mean_change) < 0.3, (symbol, r.detail.mean, r.detail.mean_change)


# ---------------------------------------------------------------- the Data Clerk's door: the fetch, the store, point in time

class Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"the analysts' request used the model ({name})")


@pytest.fixture
def toolbox(tmp_path, monkeypatch):
    """A real `DataToolbox` over a temporary base folder; Finnhub answers with the REAL rows of the capture (the shape the API gives) and no key is read."""
    monkeypatch.setattr(DataToolbox, "_api_keys", lambda self, env=None: {"FINNHUB_API_KEY": "none", "POLYGON_API_KEY": "none"})
    calls = []

    def answer(client, limiter, name, url, params, headers=None):
        calls.append((name, url, dict(params)))
        return json.loads(json.dumps(REAL[params["symbol"]]))
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(answer))
    box = DataToolbox(base_dir=tmp_path)
    box.calls = calls
    return box


def request(symbols, store=None):
    return wo.new_work_order("analyst ratings", requester="critic", action_type="fetch_analyst_ratings", subject=list(symbols), ratings={"store": store} if store else None)


def test_the_toolbox_asks_finnhub_once_per_symbol_and_keeps_each_answer_by_month_in_its_own_file(toolbox, tmp_path):
    out = toolbox.fetch_analyst_ratings(["XOM", "PG"])
    assert [c[2] for c in toolbox.calls] == [{"symbol": "XOM"}, {"symbol": "PG"}] and all(c[1] == dt.FINNHUB_RECOMMENDATION_URL for c in toolbox.calls)
    assert out["XOM"]["fetched"] == out["XOM"]["written"] == 4 and out["XOM"]["discarded"] == 0 and out["XOM"]["newest"] == "2026-11-01"
    stored = json.loads((tmp_path / "data" / "analysts" / "XOM.json").read_text(encoding="utf-8"))
    assert sorted(stored) == ["2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01"] and stored["2026-11-01"]["strongBuy"] == 5 and stored["2026-11-01"]["buy"] == 8 and stored["2026-11-01"]["hold"] == 19 and stored["2026-11-01"]["fetched_utc"].endswith("Z")
    assert not list((tmp_path / "data" / "analysts").glob("*.tmp")) and (tmp_path / "data" / "analysts" / "PG.json").exists()          # written atomically: no temporary file is left


def test_the_newest_answer_for_a_month_wins_a_month_it_no_longer_lists_stays_and_a_bad_row_is_discarded_and_counted(toolbox, tmp_path, monkeypatch):
    toolbox.fetch_analyst_ratings(["XOM"])
    revised = [{**REAL["XOM"][0], "hold": 3}, {**REAL["XOM"][1], "buy": "ten"}, {"symbol": "XOM", "period": "bad date", **{f: 1 for f in FIELDS}}, {**REAL["XOM"][2], "symbol": "OTHER"}, {**REAL["XOM"][0], "period": "2026-12-01", "strongBuy": 9}]
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(lambda client, limiter, name, url, params, headers=None: revised))
    out = toolbox.fetch_analyst_ratings(["XOM"])
    assert out["XOM"]["fetched"] == 2 and out["XOM"]["discarded"] == 3
    stored = {r["period"]: r for r in toolbox.read_analyst_ratings("XOM")}
    assert stored["2026-11-01"]["hold"] == 3 and stored["2026-12-01"]["strongBuy"] == 9 and stored["2026-10-01"]["buy"] == 9 and stored["2026-08-01"]["strongBuy"] == 6      # the revised month, a new one, and two the answer left out


def test_the_store_is_read_point_in_time_a_month_is_usable_from_its_own_first_day(toolbox):
    toolbox.fetch_analyst_ratings(["XOM"])
    assert [r["period"] for r in toolbox.read_analyst_ratings("XOM")] == ["2026-11-01", "2026-10-01", "2026-09-01", "2026-08-01"]
    assert [r["period"] for r in toolbox.read_analyst_ratings("XOM", until=date(2026, 11, 1))][0] == "2026-11-01"
    assert [r["period"] for r in toolbox.read_analyst_ratings("XOM", until=date(2026, 10, 31))] == ["2026-10-01", "2026-09-01", "2026-08-01"]
    assert toolbox.read_analyst_ratings("XOM", until=date(2026, 7, 31)) == [] and toolbox.read_analyst_ratings("ZZZZ") == []
    assert set(toolbox.read_analyst_ratings("XOM")[0]) == {"period", *FIELDS}


def test_while_a_day_is_bound_nothing_is_fetched_and_a_period_after_the_day_is_never_read_whatever_is_asked_for(toolbox, monkeypatch):
    toolbox.fetch_analyst_ratings(["XOM"])
    toolbox.calls.clear()
    day = date(2026, 10, 14)                  # a Wednesday: the months of 2026-10-01 and before are known, 2026-11-01 is the future
    with tc.as_of(day):
        with pytest.raises(RuntimeError, match="the Proving Ground never fetches: fetch_analyst_ratings was refused"):
            toolbox.fetch_analyst_ratings(["XOM"])
        assert [r["period"] for r in toolbox.read_analyst_ratings("XOM", until=date(2026, 12, 31))] == ["2026-10-01", "2026-09-01", "2026-08-01"]      # asking for later is cut to the bound day
        clerk = DataClerk(toolbox=toolbox, model=Forbidden())
        order = clerk.request_analyst_ratings(request(["XOM"]), date(2026, 12, 31))
    assert toolbox.calls == [] and order.status == "data_ready" and order.errors == []
    result = order.data.analysts["XOM"]
    assert [p["period"] for p in result["periods"]] == ["2026-10-01", "2026-09-01", "2026-08-01"] and result["fetched"] is False
    with tc.as_of(date(2026, 11, 2)):         # the next trading day: the month of 2026-11-01 is known from its own first day
        assert [r["period"] for r in toolbox.read_analyst_ratings("XOM")][0] == "2026-11-01"


def test_the_data_clerks_request_fetches_stores_and_answers_with_the_months_the_day_could_know(toolbox, tmp_path):
    clerk = DataClerk(toolbox=toolbox, model=Forbidden())
    order = clerk.request_analyst_ratings(request(["XOM", "PG"], "data/analysts"), date(2026, 11, 20))
    assert order.status == "data_ready" and order.errors == [] and [c[2]["symbol"] for c in toolbox.calls] == ["XOM", "PG"]
    assert list(order.data.analysts) == ["XOM", "PG"] and order.data.analysts["PG"]["fetched"] is True and order.data.analysts["PG"]["note"] is None
    assert [p["period"] for p in order.data.analysts["PG"]["periods"]] == ["2026-11-01", "2026-10-01", "2026-09-01", "2026-08-01"] and order.data.tool_calls[0]["outcome"] == "ok"
    assert (tmp_path / "data" / "analysts" / "PG.json").exists() and clerk.request_analyst_ratings(order) is order          # an order that has its data comes back unchanged
    back = clerk.request_analyst_ratings(request(["PG"]), date(2026, 10, 31))
    assert [p["period"] for p in back.data.analysts["PG"]["periods"]][0] == "2026-10-01"                                     # `today` cuts the months the day could not know, unbound too


def test_a_call_that_fails_is_a_problem_in_words_and_the_stored_months_still_come_back(toolbox, monkeypatch):
    toolbox.fetch_analyst_ratings(["XOM"])

    def down(client, limiter, name, url, params, headers=None):
        raise RuntimeError("Finnhub recommendation HTTP 429: slow down")
    monkeypatch.setattr(DataToolbox, "_get_json", staticmethod(down))
    order = DataClerk(toolbox=toolbox, model=Forbidden()).request_analyst_ratings(request(["XOM", "PG"]), date(2026, 11, 20))
    assert order.status == "data_failed" and len(order.errors) == 2 and "XOM: the analyst ratings could not be fetched (RuntimeError: Finnhub recommendation HTTP 429" in order.errors[0]
    assert len(order.data.analysts["XOM"]["periods"]) == 4 and order.data.analysts["XOM"]["fetched"] is False and "stored counts are used" in order.data.analysts["XOM"]["note"]
    assert order.data.analysts["PG"]["periods"] == [] and "none stored" in order.data.analysts["PG"]["note"]
    assert "429" not in str(order.data.tool_calls) and order.data.tool_calls[0]["outcome"] == "error"


def test_a_request_that_is_not_allowed_is_refused_before_anything_is_asked(toolbox):
    clerk = DataClerk(toolbox=toolbox, model=Forbidden())
    for bad, why in ((request(["../etc"]), "ticker"), (request([]), "ticker symbols"), (request(["XOM"], "../secrets"), "store"), (request(["XOM"], "/tmp/x"), "store"), (request(["XOM"], "news/x"), "store")):
        order = clerk.request_analyst_ratings(bad)
        assert order.status == "data_failed" and why in order.errors[0] and toolbox.calls == []
    order = request(["XOM"])
    order.action_type = "fetch_quote"
    assert "is not 'fetch_analyst_ratings'" in clerk.request_analyst_ratings(order).errors[0]
    assert wo.verify_ratings_request(request([f"S{i}" for i in range(31)])) == ["an analyst-ratings request is for at most 30 symbols at a time"]
    assert wo.verify_ratings_request(request(["XOM", "PG"], "data/analysts")) == []


# ---------------------------------------------------------------- inside a review

def test_a_review_asks_for_the_analysts_once_for_every_symbol_as_of_the_record_day_and_scores_them():
    order = captured("PG_open")
    clerk = FakeClerk(order, own={}, analysts={"PG": periods("PG")})
    r = review(order, clerk=clerk, own={})["PG"]
    ask = [a for a in clerk.asked if a[0] == "analysts"]
    assert ask == [("analysts", ["PG"], date(2026, 11, 5))] and r["audit"]["analysts_order"] and order.critic_runs[-1]["analysts_order"] == r["audit"]["analysts_order"]
    d = r["dimensions"]["analysts"]
    assert d["applicable"] and d["weight"] == 10 and d["score"] == pytest.approx(100 * (mean(periods("PG")[0]) - 1) / 4, abs=0.01) and d["points"] == pytest.approx(d["score"] / 10, abs=0.01)
    a = r["analysts"]
    assert a["applicable"] and a["period"] == "2026-11-01" and a["n"] == 35 and a["counts"] == {"strongBuy": 6, "buy": 14, "hold": 14, "sell": 1, "strongSell": 0} and a["compared_period"] == "2026-08-01"
    assert r["approved"] and r["groups"]["news"] == [] and sum(d["weight"] for d in r["dimensions"].values()) == 100
    assert "analysts 7/10" in r["summary"]


def test_a_review_whose_symbols_have_no_counts_is_scored_on_the_other_six_and_a_failed_request_denies_nothing():
    order = captured("PG_open")
    none = review(order, own={})["PG"]
    assert not none["dimensions"]["analysts"]["applicable"] and none["analysts"]["applicable"] is False and none["approved"]
    order = captured("PG_open")
    failed = review(order, clerk=FakeClerk(order, own={}, fail=["analysts"]), own={})["PG"]
    assert failed["approved"] and failed["score"] == none["score"] and not failed["dimensions"]["analysts"]["applicable"]
    assert order.critic_runs[-1]["errors"] == ["analysts: RuntimeError: the analyst ratings are down"]


def test_a_review_of_the_analysts_changes_the_score_by_their_weight_and_by_nothing_else():
    order = captured("PG_open")
    base = review(order, own={})["PG"]
    order2 = captured("PG_open")
    with_analysts = review(order2, clerk=FakeClerk(order2, own={}, analysts={"PG": periods("PG")}), own={})["PG"]
    others = {n: d["points"] for n, d in base["dimensions"].items() if n != "analysts"}
    assert {n: d["points"] for n, d in with_analysts["dimensions"].items() if n != "analysts"} == others
    assert sum(d["weight"] for d in base["dimensions"].values() if d["applicable"]) == 90 and sum(d["weight"] for d in with_analysts["dimensions"].values() if d["applicable"]) == 100
    assert base["pattern_bonus"] == with_analysts["pattern_bonus"] and base["bonus"] == with_analysts["bonus"] and base["groups"] == with_analysts["groups"]


def test_a_bearish_street_is_a_news_sign_it_approves_alone_and_denies_with_a_bad_sign_in_another_group():
    """The real PG_open record with a sell-heavy street (rule inputs): one bad group (news) is an approval that says it; with a trend record whose stock fell over five sessions (performance) the review is denied for the two."""
    lean = [row(buy=2, hold=6, sell=10, strong_sell=4)]
    order = captured("PG_open")
    one = review(order, clerk=FakeClerk(order, own={}, analysts={"PG": lean}), own={})["PG"]
    assert one["approved"] and one["blockers"] == [] and one["groups"] == {"news": ["analysts_bearish"], "performance": [], "price": []}
    assert "analysts_bearish" in [w["code"] for w in one["warnings"]] and "Note: one reason against it: news (analysts_bearish: The analysts lean against PG" in one["summary"]
    order = tampered_order("PG_open", **{"trade.family": "trend"})
    two = review(order, clerk=FakeClerk(order, own={}, analysts={"PG": lean}, changes={"PG": {"ret_5d": -0.021}}), own={})["PG"]
    assert not two["approved"] and two["denial_reasons"] == ["group:news", "group:performance", "trend_not_confirmed", "analysts_bearish"] and two["score"] <= 49
    assert [b["code"] for b in two["blockers"]] == ["trend_not_confirmed"] and "analysts_bearish" in [w["code"] for w in two["warnings"]]
    assert two["summary"].startswith("DENY PG: poor news (analysts_bearish: The analysts lean against PG") and " and poor performance (trend_not_confirmed: " in two["summary"]


def test_the_critic_reaches_the_analysts_only_through_the_data_clerk():
    for path in (REPO_ROOT / "prometheus" / "agents" / "critic").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "httpx" not in text and "finnhub.io" not in text and "read_analyst_ratings" not in text and "fetch_analyst_ratings(" not in text, path
    assert "request_analyst_ratings" in (REPO_ROOT / "prometheus" / "agents" / "critic" / "recheck.py").read_text(encoding="utf-8")


def test_the_scorecard_the_user_reads_has_the_analysts_row_the_groups_the_pattern_bonus_and_says_a_two_group_denial():
    from prometheus.agents.critic.report import scorecard
    order = captured("PG_open")
    r = review(order, clerk=FakeClerk(order, own={}, analysts={"PG": periods("PG")}), own={})["PG"]
    lines = scorecard(r)["text"].splitlines()
    analysts = next(line for line in lines if line.strip().startswith("analysts"))
    assert "7/10" in analysts and "6 strong buy, 14 buy, 14 hold, 1 sell, 0 strong sell in 2026-11-01 (35 analysts): mean 3.71 of 5, down 0.04 since 2026-08-01" in analysts
    assert "      groups:   news clean, performance clean, price clean" in lines and next(line for line in lines if line.strip().startswith("patterns")).split()[1] == "+6.25"
    old = scorecard(json.loads(json.dumps({**r, "dimensions": {k: v for k, v in r["dimensions"].items() if k != "analysts"}, "groups": {}, "pattern_bonus": 0.0, "analysts": {}})))["text"]
    assert "analysts" not in old and "groups:" not in old and "patterns" not in old.split("accuracy")[1]                  # a review made before version 3 reads as it always did
    order = tampered_order("PG_open", **{"trade.family": "trend"})
    lean = [row(buy=2, hold=6, sell=10, strong_sell=4)]
    two = review(order, clerk=FakeClerk(order, own={}, analysts={"PG": lean}, changes={"PG": {"ret_5d": -0.021}}), own={})["PG"]
    text = scorecard(two)["text"]
    assert "groups:   news BAD (analysts_bearish), performance BAD (trend_not_confirmed), price clean" in text and "DENIED: bad signs in poor news (analysts_bearish) and poor performance (trend_not_confirmed)" in text
    assert "BLOCKER trend_not_confirmed:" in text and "warnings: " in text and "analysts_bearish" in next(line for line in text.splitlines() if "warnings:" in line)
