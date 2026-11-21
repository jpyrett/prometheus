"""Accuracy: the Data Clerk is asked again and the old and new values are compared (Critic spec 8.1). The three requests, the tolerances, the sources read again, the cross-checks."""
import copy

from critic_helpers import CONFIG, NOW, FakeClerk, captured, rec_of

from prometheus.agents.critic import recheck as rc
from prometheus.agents.critic.claims import claims_of

ACC = CONFIG.accuracy


def setup(name="PG_open", **options):
    order = captured(name)
    rec = rec_of(order)
    clerk = FakeClerk(order, **options)
    claims = claims_of(rec, order)
    got = rc.gather(clerk, order, [(rec, "2026-11-05", claims)], NOW, CONFIG)
    names, decisive = rc.key_metrics(rec, CONFIG.limits.max_metrics)
    return order, rec, clerk, claims, got, names, decisive


# ---------------------------------------------------------------- the three requests


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


def test_a_decisive_number_that_cannot_be_calculated_again_is_a_warning_with_the_clerks_note_and_the_price_that_cannot_is_a_fault():
    def gone(f):
        f["atr_14"] = {**f["atr_14"], "value": None, "note": "needs 20 bars, 17 stored"}
    _, (rechecks, findings, _), _ = compared(gone)
    assert [(f.code, f.severity) for f in findings] == [("data_unverified", "warning")] and next(r for r in rechecks if r.name == "atr_14").note == "needs 20 bars, 17 stored"      # AMENDED (V7): it was a `data_mismatch` blocker

    def no_price(f):
        f["last_close"] = {**f["last_close"], "value": None, "note": "needs 1 bars, 0 stored"}
    _, (_, findings, _), _ = compared(no_price)
    assert [(f.code, f.severity) for f in findings][0] == ("data_unreliable", "blocker") and "could not be calculated again" in findings[0].message


def test_nothing_returned_for_the_symbol_is_data_unreliable():
    """AMENDED (V7): nothing to compare the record with was `data_unverified`, a blocker; it is `data_unreliable` now (a fault: the record cannot be checked at all). `data_unverified` is the warning for one metric."""
    order, rec, clerk, claims, got, names, decisive = setup()
    for empty in (None, {}):
        rechecks, findings, rate = rc.compare(rec, empty, names, decisive, ACC)
        assert [(f.code, f.severity) for f in findings] == [("data_unreliable", "blocker")] and rechecks == [] and rate is None
    rechecks, findings, rate = rc.compare(rec, {"unrelated_metric": {"value": 1}}, ["unrelated_metric"], decisive, ACC)
    assert [(f.code, f.severity) for f in findings] == [("data_unreliable", "blocker")] and rate is None


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


def test_a_quote_that_is_no_longer_in_its_source_is_a_blocker():
    (checks, findings, _), claims = sources_of(lambda v, c: v["chunks"][c[0].chunk_id].update(text="The article was rewritten and says nothing like it did."))
    assert [f.code for f in findings] == ["source_missing"] and "no longer in its source" in findings[0].message and checks[0].quote_in_text is False


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


def test_a_company_name_the_directory_does_not_recognise_is_a_warning_for_a_person():
    checks, findings, identity, _ = cross(mutate=lambda f, v: v["symbols"]["PG"].update(company="ACME ROCKET SKATES LTD"))
    assert [(f.code, f.severity, f.requires_human) for f in findings] == [("company_mismatch", "warning", True)] and not next(c for c in checks if c.name == "identity").passed


# ---------------------------------------------------------------- no equity: the Critic does not know the user's money

def test_the_critic_has_no_equity_and_never_compares_one_with_the_advisors():
    """Design choices entry 80: `own_equity` (the portfolio at the quotes of the review) and `equity_check` (the `equity_differs` and `equity_unverified` warnings) were only for the position size, which is the Account
    Manager's now. Nothing of the Critic takes a portfolio, and its settings have no `equity_rel`."""
    assert not hasattr(rc, "own_equity") and not hasattr(rc, "equity_check") and "equity_rel" not in type(ACC).model_fields
    import inspect
    assert "portfolio" not in inspect.signature(rc.gather).parameters
