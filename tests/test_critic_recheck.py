"""Accuracy: the Data Clerk is asked again and the old and new values are compared (Critic spec 8.1). The three requests, the tolerances, the sources read again, the cross-checks."""
import copy
from datetime import timedelta

import pytest
from critic_helpers import CONFIG, NOW, FakeClerk, captured, rec_of

from prometheus.agents.critic import recheck as rc
from prometheus.agents.critic.claims import claims_of
from prometheus.common import work_order as wo

ACC = CONFIG.accuracy


def setup(name="PG_open", **options):
    order = captured(name)
    rec = rec_of(order)
    clerk = FakeClerk(order, **options)
    claims = claims_of(rec, order)
    got = rc.gather(clerk, order, [(rec, "2026-11-05", claims)], NOW, CONFIG)
    names, decisive = rc.key_metrics(rec, CONFIG.limits.max_metrics)
    return order, rec, clerk, claims, got, names, decisive


# ---------------------------------------------------------------- what to ask for

def test_the_key_metrics_start_with_the_decisive_ones_and_hold_the_critics_own():
    order = captured("PG_open")
    rec = rec_of(order)
    names, decisive = rc.key_metrics(rec, 60)
    assert names[:2] == ["last_close", "atr_14"] and decisive >= {"last_close", "atr_14"}
    family = next(f for f in rec.families if f.family == rec.trade.family)
    assert all(n in decisive for n in family.features_used if n in rec.features and n in names)
    assert set(rc.ALWAYS) <= set(names) and set(rc.OWN) <= set(names) and len(names) == len(set(names))
    assert all(n in names for g in ("performance", "historical", "filings") for p in rec.blend["components"][g]["parts"] for n in [p.get("metric")] if n)


def test_a_cap_never_cuts_a_decisive_metric():
    rec = rec_of(captured("PG_open"))
    names, decisive = rc.key_metrics(rec, 3)
    assert len(names) == len(decisive) and set(names) == decisive and len(rc.key_metrics(rec, 60)[0]) <= 60


def test_a_reduction_is_checked_on_every_passed_family():
    order = captured("PG_open")
    order.recommendations[0]["action"]["action"], order.recommendations[0]["trade"]["family"] = "reduce", None
    names, decisive = rc.key_metrics(rec_of(order), 60)
    assert {n for f in rec_of(order).families if f.passed for n in f.features_used if n in names and n in rec_of(order).features} <= decisive


# ---------------------------------------------------------------- the three requests

def test_one_order_of_each_kind_is_sent_for_all_the_records():
    order, rec, clerk, claims, got, names, _ = setup()
    kinds = [a[0] for a in clerk.asked]
    assert kinds == ["metrics", "quotes", "verification"]
    assert clerk.asked[0][1] == ["PG"] and set(clerk.asked[0][2]) >= set(names[:5]) and str(clerk.asked[0][3]) == "2026-11-05"
    assert clerk.asked[1][1] == ["PG"]        # only the recommended symbol: the Critic is given no portfolio, so it never asks for the price of a holding (the Account Manager does)
    verify = clerk.asked[2][1]
    assert set(verify) == {"chunks", "evidence", "symbols", "company", "fresh"} and verify["symbols"] == ["PG"] and verify["company"] == {"PG": "Procter & Gamble"}
    assert verify["fresh"] == {"since_days": CONFIG.evidence.fresh_since_days, "k": CONFIG.evidence.fresh_k}
    assert verify["chunks"] and all(not c.startswith("evidence:") for c in verify["chunks"]) and set(verify["evidence"]["PG"]) == {"E15", "E21"}
    assert got.verified and got.errors == [] and set(got.metric_orders) == {"2026-11-05"} and got.quote_order and len(got.verify_orders) == 1


def test_the_critic_is_the_requester_and_never_allows_a_fetch():
    seen = []

    class Spy(FakeClerk):
        def request_metrics(self, order, today=None):
            seen.append((order.requester, order.action_type))
            return super().request_metrics(order, today)

        def request_quotes(self, order):
            seen.append((order.requester, order.action_type, order.live))
            return super().request_quotes(order)

        def request_verification(self, order, now=None):
            seen.append((order.requester, order.action_type))
            return super().request_verification(order, now)

    order = captured("PG_open")
    rec = rec_of(order)
    rc.gather(Spy(order), order, [(rec, "2026-11-05", claims_of(rec, order))], NOW, CONFIG)
    assert seen == [("critic", "calculate_financials"), ("critic", "fetch_quote", True), ("critic", "verify_data")]


def test_records_of_different_days_get_one_metric_order_each():
    order = captured("PG_open")
    rec = rec_of(order)
    clerk = FakeClerk(order)
    got = rc.gather(clerk, order, [(rec, "2026-11-05", []), (rec, "2026-11-04", [])], NOW, CONFIG)
    assert [a[0] for a in clerk.asked].count("metrics") == 2 and set(got.metric_orders) == {"2026-11-04", "2026-11-05"}


def test_more_than_sixty_chunks_are_asked_in_a_second_order():
    order = captured("PG_open")
    rec = rec_of(order)
    many = [rc.Claim(ref=f"[{i}]", kind="signal", direction="bullish", quote="q" * 20, source=i, chunk_id=f"finnhub:{i}#0") for i in range(1, 71)]
    clerk = FakeClerk(order)
    got = rc.gather(clerk, order, [(rec, "2026-11-05", many)], NOW, CONFIG)
    verifications = [a[1] for a in clerk.asked if a[0] == "verification"]
    assert [len(v["chunks"]) for v in verifications] == [60, 10] and "symbols" in verifications[0] and "symbols" not in verifications[1] and len(got.verify_orders) == 2


@pytest.mark.parametrize("fail, label", [("metrics", "metrics"), ("quotes", "quotes"), ("verification", "verification")])
def test_a_request_that_raises_is_named_and_the_others_still_run(fail, label):
    order, rec, clerk, claims, got, names, decisive = setup(fail=[fail])
    assert any(e.startswith(f"{label}: RuntimeError") for e in got.errors) and {a[0] for a in clerk.asked} == {"metrics", "quotes", "verification"}
    assert (got.metrics == {}) == (fail == "metrics") and (got.quotes == {}) == (fail == "quotes") and got.verified == (fail != "verification")


def test_an_order_the_data_clerk_marks_failed_puts_its_errors_in_the_runs_errors():
    class Refusing(FakeClerk):
        def request_quotes(self, order):
            order.status, order.errors = "data_failed", ["PG: no quote (RuntimeError: HTTP 429)"]
            return order

    order = captured("PG_open")
    rec = rec_of(order)
    got = rc.gather(Refusing(order), order, [(rec, "2026-11-05", [])], NOW, CONFIG)
    assert any("quotes: PG: no quote" in e for e in got.errors) and got.quotes == {}


# ---------------------------------------------------------------- old against new

def compared(mutate=None, **options):
    order, rec, clerk, claims, got, names, decisive = setup(**options)
    fetched = copy.deepcopy(got.metrics_for(rec, "2026-11-05"))
    if mutate:
        mutate(fetched)
    return rec, rc.compare(rec, fetched, names, decisive, ACC), decisive


def test_everything_matching_gives_no_finding_and_a_perfect_rate():
    rec, (rechecks, findings, rate), _ = compared()
    assert findings == [] and rate == 1.0 and rechecks and all(r.match for r in rechecks)
    assert {"last_close", "atr_14"} <= {r.name for r in rechecks} and all(r.name in rec.features for r in rechecks)


def test_a_decisive_number_that_differs_is_a_blocker_by_itself():
    def bad(f):
        f["last_close"]["value"] *= 1.10
    _, (rechecks, findings, rate), _ = compared(bad)
    [finding] = findings
    assert (finding.code, finding.severity, finding.dimension) == ("data_mismatch", "blocker", "accuracy") and "last_close was" in finding.message
    assert next(r for r in rechecks if r.name == "last_close").match is False and next(r for r in rechecks if r.name == "last_close").decisive and rate < 1.0


def test_a_decisive_number_that_cannot_be_calculated_again_is_a_mismatch_with_the_clerks_note():
    def gone(f):
        f["atr_14"] = {**f["atr_14"], "value": None, "note": "needs 20 bars, 17 stored"}
    _, (rechecks, findings, _), _ = compared(gone)
    assert findings[0].code == "data_mismatch" and next(r for r in rechecks if r.name == "atr_14").note == "needs 20 bars, 17 stored"


def test_several_small_mismatches_among_the_other_numbers_trip_the_match_rate_floor():
    order, rec, clerk, claims, got, names, decisive = setup()
    fetched = copy.deepcopy(got.metrics_for(rec, "2026-11-05"))
    others = [r.name for r in rc.compare(rec, fetched, names, decisive, ACC)[0] if not r.decisive and isinstance(fetched[r.name]["value"], float)]
    total = len(rc.compare(rec, fetched, names, decisive, ACC)[0])
    wrong = 0
    for name in others:
        fetched[name]["value"] = fetched[name]["value"] * 3 + 1
        wrong += 1
        rechecks, findings, rate = rc.compare(rec, fetched, names, decisive, ACC)
        if rate < ACC.min_match_rate:
            break
    assert findings and findings[0].code == "data_mismatch" and f"{rate:.0%}" in findings[0].message and rate == pytest.approx((total - wrong) / total) and wrong >= 2


def test_nothing_returned_for_the_symbol_is_data_unverified():
    order, rec, clerk, claims, got, names, decisive = setup()
    for empty in (None, {}):
        rechecks, findings, rate = rc.compare(rec, empty, names, decisive, ACC)
        assert [f.code for f in findings] == ["data_unverified"] and rechecks == [] and rate is None
    rechecks, findings, rate = rc.compare(rec, {"unrelated_metric": {"value": 1}}, ["unrelated_metric"], decisive, ACC)
    assert [f.code for f in findings] == ["data_unverified"] and rate is None


@pytest.mark.parametrize("source, name, kind", [("day_aggs:2025-10-02..2026-11-05", "rsi_14", "bars"), ("finnhub_metric:2026-11-06", "roe", "provider"), ("finnhub_metric:2026-11-06", "market_cap", "provider_price_linked"),
                                                ("filing:0000080424-26-000256", "cash_balance", "filing"), ("derived:filing:0000080424-26-000256+finnhub_metric:2026-11-06", "net_payout_yield_ttm", "derived"),
                                                ("n/a", "x", "derived")])
def test_how_a_number_is_checked_comes_from_where_it_came(source, name, kind):
    assert rc.kind_of(source, name, ACC) == kind


@pytest.mark.parametrize("kind, old, new, match", [
    ("bars", 100.0, 100.00005, True), ("bars", 100.0, 100.5, False), ("bars", 0.0, 1e-10, True), ("bars", 0.0, 1e-3, False),
    ("bars_changed", 100.0, 101.9, True), ("bars_changed", 100.0, 103.0, False),
    ("provider", 30.0, 30.5, True), ("provider", 30.0, 31.0, False), ("provider", 0.0, 0.005, True),
    ("provider_price_linked", 400e9, 419e9, True), ("provider_price_linked", 400e9, 430e9, False),
    ("filing", 5e9, 5e9, True), ("filing", 5e9, 5.1e9, False), ("filing_changed", 5e9, 5.4e9, True), ("filing_changed", 5e9, 6e9, False),
    ("derived", 0.04, 0.0415, True), ("derived", 0.04, 0.05, False),
])
def test_each_tolerance_at_its_edge(kind, old, new, match):
    assert rc._same(old, new, kind, ACC)[0] is match


def test_a_bool_and_a_text_must_be_equal():
    assert rc._same(True, True, "bars", ACC) == (True, None) and rc._same(True, False, "bars", ACC)[0] is False and rc._same("XLK", "XLK", "derived", ACC)[0] and not rc._same("XLK", "XLE", "derived", ACC)[0]
    assert not rc._same(1, True, "bars", ACC)[0]     # a bool is not a number


def test_other_bars_loosen_the_tolerance_and_warn_once():
    def shifted(f):
        for n, r in f.items():
            if str(r.get("source")).startswith("day_aggs:"):
                r["source"] = "day_aggs:2025-10-03..2026-11-05"
        f["last_close"]["value"] *= 1.01          # 1%: inside the looser bars tolerance (2%), outside the strict one
    _, (rechecks, findings, _), _ = compared(shifted)
    assert [f.code for f in findings] == ["bars_changed"] and findings[0].severity == "warning"
    assert next(r for r in rechecks if r.name == "last_close").kind == "bars_changed" and next(r for r in rechecks if r.name == "last_close").match


def test_another_filing_loosens_the_tolerance_and_warns_for_a_person():
    order, rec, clerk, claims, got, names, decisive = setup()
    name = next(n for n, f in rec.features.items() if f["source"].startswith("filing:") and isinstance(f["value"], (int, float)) and not isinstance(f["value"], bool))
    fetched = copy.deepcopy(got.metrics_for(rec, "2026-11-05"))
    fetched[name]["source"] = "filing:0000080424-26-999999"
    fetched[name]["value"] = rec.features[name]["value"] * 1.05
    rechecks, findings, _ = rc.compare(rec, fetched, [name], decisive, ACC)
    assert [f.code for f in findings] == ["statements_changed"] and findings[0].requires_human and rechecks[0].kind == "filing_changed" and rechecks[0].match


# ---------------------------------------------------------------- the sources, read again

def sources_of(mutate=None, name="PG_open", **options):
    order, rec, clerk, claims, got, names, decisive = setup(name, **options)
    verification = copy.deepcopy(got.verification)
    if mutate:
        mutate(verification, claims)
    return rc.check_sources(rec, claims, order, verification, got.verified), claims


def test_sources_that_hold_give_no_finding():
    (checks, findings, rate), claims = sources_of()
    assert findings == [] and rate == 1.0 and len(checks) == len(claims) >= 3 and all(c.found and c.quote_in_text and c.same_text for c in checks)


def test_a_source_the_store_cannot_find_is_a_blocker():
    (checks, findings, rate), claims = sources_of(lambda v, c: v["chunks"].update({c[0].chunk_id: {"found": False, "note": "not in the stored news"}}))
    assert [f.code for f in findings] == ["source_missing"] and "not in the stored news" in findings[0].message and checks[0].found is False and rate < 1.0


def test_a_quote_that_is_no_longer_in_its_source_is_a_blocker():
    (checks, findings, _), claims = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(text="The article was rewritten and says nothing like it did."))
    assert [f.code for f in findings] == ["source_missing"] and "no longer in its source" in findings[0].message and checks[0].quote_in_text is False


def test_an_altered_headline_or_date_is_a_blocker():
    (_, findings, _), _ = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(headline="A different headline"))
    assert [f.code for f in findings] == ["source_altered"]
    (_, findings, _), _ = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(published_utc="2026-11-04T00:00:00Z"))
    assert [f.code for f in findings] == ["source_altered"]


def test_the_rest_of_the_text_changing_is_only_a_warning():
    (checks, findings, rate), _ = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(text=v["chunks"][c[0].chunk_id]["text"] + " An editor's note was added."))
    assert [(f.code, f.severity) for f in findings] == [("source_text_changed", "warning")] and checks[0].same_text is False and rate == 1.0


def test_a_source_that_does_not_list_the_symbol_is_a_warning_and_is_marked_untagged():
    (checks, findings, _), _ = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(tickers=["MSFT"]))
    assert [(f.code, f.severity) for f in findings] == [("source_not_tagged", "warning")] and checks[0].tagged is False


def test_a_filing_evidence_item_that_is_gone_or_changed_is_found_by_its_id():
    def gone(v, c):
        ev = next(x for x in c if x.kind == "evidence")
        v["evidence"]["PG"][ev.evidence_id] = {"found": False, "note": "PG has no evidence item"}
    (checks, findings, _), claims = sources_of(gone)
    assert [f.code for f in findings] == ["source_missing"] and next(c for c in checks if c.ref.startswith("E")).found is False


def test_when_the_verification_request_failed_every_source_is_missing_and_says_so():
    (checks, findings, rate), claims = sources_of(fail=["verification"], mutate=lambda v, c: [v.update(chunks={}, evidence={}) for _ in [0]])
    assert len(findings) == len(claims) and all(f.code == "source_missing" and "verification request failed" in f.message for f in findings) and rate == 0.0


def test_a_record_that_cites_nothing_has_a_perfect_source_rate():
    order, rec, *_ = setup()
    assert rc.check_sources(rec, [], order, {}, True) == ([], [], 1.0)


# ---------------------------------------------------------------- the cross-checks

def cross(mutate=None, quote_mutate=None, **options):
    order, rec, clerk, claims, got, names, decisive = setup(**options)
    fetched, quote = copy.deepcopy(got.metrics_for(rec, "2026-11-05")), copy.deepcopy(got.quotes.get("PG"))
    verification = copy.deepcopy(got.verification)
    if mutate:
        mutate(fetched, verification)
    if quote_mutate:
        quote_mutate(quote)
    return rc.cross_checks(rec, fetched, quote, verification, ACC)


def test_when_everything_agrees_all_three_cross_checks_pass():
    checks, findings, identity, rate = cross()
    assert findings == [] and rate == 1.0 and {c.name for c in checks} == {"price", "identity", "high_52w"} and identity.listed is True and identity.listed_company == "PG INC"


def test_the_close_is_compared_with_the_quotes_price_on_the_same_day_and_its_previous_close_the_day_after():
    # the stored bar is dated 2026-11-05 (the quote's own day): the quote's price is the other provider's close
    checks, findings, _, _ = cross(quote_mutate=lambda q: q.update(price=q["price"] * 1.03, previous_close=1.0))
    assert [f.code for f in findings] == ["price_mismatch"] and findings[0].severity == "blocker"
    # the bar is the day BEFORE the quote's: then the quote's previous close is the comparison
    checks, findings, _, _ = cross(mutate=lambda f, v: f["last_close"].update(as_of="2026-11-04"), quote_mutate=lambda q: q.update(price=1.0, previous_close=q["price"]))
    assert findings == [] and next(c for c in checks if c.name == "price").passed
    checks, findings, _, _ = cross(mutate=lambda f, v: f["last_close"].update(as_of="2026-11-04"), quote_mutate=lambda q: q.update(previous_close=q["price"] * 1.5))
    assert [f.code for f in findings] == ["price_mismatch"]


def test_an_after_hours_print_on_the_bars_day_is_not_the_close():
    """The quote's price outside the regular session is the last trade, not the daily close. A move after the close is not two providers disagreeing (entry 71). The print at the close still is."""
    after = cross(quote_mutate=lambda q: q.update(price=q["price"] * 1.03, as_of_utc="2026-11-05T21:30:00Z"))   # 16:30 Eastern, after the 16:00 close
    assert "price" not in {c.name for c in after[0]} and after[1] == []
    before = cross(quote_mutate=lambda q: q.update(price=q["price"] * 1.03, as_of_utc="2026-11-05T13:00:00Z"))  # 08:00 Eastern, pre-market
    assert "price" not in {c.name for c in before[0]}
    closing = cross(quote_mutate=lambda q: q.update(price=q["price"] * 1.03, as_of_utc="2026-11-05T21:00:00Z"))  # 16:00 Eastern, the closing print
    assert [f.code for f in closing[1]] == ["price_mismatch"]


def test_a_bar_from_any_other_day_is_not_compared_at_all():
    checks, findings, _, _ = cross(mutate=lambda f, v: f["last_close"].update(as_of="2026-10-20"))
    assert "price" not in {c.name for c in checks} and findings == []


def test_no_quote_means_no_price_check():
    checks, _, _, _ = cross(quote_mutate=lambda q: q.clear())
    assert "price" not in {c.name for c in checks}


def test_a_symbol_the_directory_does_not_list_is_a_blocker_and_a_failed_lookup_only_a_warning():
    _, findings, identity, rate = cross(listed={"PG": False})
    assert [f.code for f in findings if f.severity == "blocker"] == ["symbol_unknown"] and identity.listed is False
    _, findings, identity, _ = cross(listed={"PG": None})
    assert [(f.code, f.severity, f.requires_human) for f in findings] == [("identity_unverified", "warning", True)] and identity.listed is None and "429" in findings[0].message


def test_a_company_name_the_directory_does_not_recognise_is_a_warning_for_a_person():
    checks, findings, identity, _ = cross(mutate=lambda f, v: v["symbols"]["PG"].update(company="ACME ROCKET SKATES LTD"))
    assert [(f.code, f.severity, f.requires_human) for f in findings] == [("company_mismatch", "warning", True)] and not next(c for c in checks if c.name == "identity").passed


def test_a_52_week_high_that_disagrees_with_finnhubs_is_a_warning():
    checks, findings, _, rate = cross(mutate=lambda f, v: f["high_52w"].update(value=f["high_52w"]["value"] * 1.2))
    assert [(f.code, f.severity) for f in findings] == [("high_52w_differs", "warning")] and rate == pytest.approx(2 / 3)


def test_with_nothing_to_cross_check_the_rate_is_zero():
    rec = rec_of(captured("PG_open"))
    assert rc.cross_checks(rec, None, None, {}, ACC)[3] == 0.0


# ---------------------------------------------------------------- no equity: the Critic does not know the user's money

def test_the_critic_has_no_equity_and_never_compares_one_with_the_advisors():
    """Design choices entry 80: `own_equity` (the portfolio at the quotes of the review) and `equity_check` (the `equity_differs` and `equity_unverified` warnings) were only for the position size, which is the Account
    Manager's now. Nothing of the Critic takes a portfolio, and its settings have no `equity_rel`."""
    assert not hasattr(rc, "own_equity") and not hasattr(rc, "equity_check") and "equity_rel" not in type(ACC).model_fields
    import inspect
    assert "portfolio" not in inspect.signature(rc.gather).parameters


# ---------------------------------------------------------------- the accuracy score

def test_the_accuracy_score_weights_the_three_shares():
    score, measures = rc.accuracy_score(1.0, 1.0, 1.0, ACC)
    assert score == 100 and [m.name for m in measures] == ["metrics_match", "sources_held", "cross_checks_passed"] and [m.weight for m in measures] == [50, 30, 20]
    assert rc.accuracy_score(0.9, 1.0, 0.5, ACC)[0] == pytest.approx(100 * (0.5 * 0.9 + 0.3 + 0.2 * 0.5))
    score, measures = rc.accuracy_score(None, 1.0, 1.0, ACC)
    assert score == 50 and measures[0].available is False


def test_the_record_date_decides_which_days_metrics_are_asked_for():
    order = captured("PG_open")
    rec = rec_of(order)
    clerk = FakeClerk(order)
    rc.gather(clerk, order, [(rec, "2026-11-05", [])], NOW + timedelta(days=3), CONFIG)
    assert str(clerk.asked[0][3]) == "2026-11-05"        # the bars up to the record's own day, not up to the review's
    assert wo.verify_metric_request(wo.new_work_order("m", requester="critic", action_type="calculate_financials", subject=["PG"], metrics=[{"name": n} for n in rc.key_metrics(rec, 60)[0]])) == []
