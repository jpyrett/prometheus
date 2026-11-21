"""Tests for prometheus.application.recommendations (design choices entry 53): classify a request, run the steps, print the report. The Data Clerk and the model are fakes; the Finnhub rows of the
company lookups are REAL (tests/fixtures/finnhub_pizza_rows.json); the Advisor is the real one on fake numbers, so the ranking and the report are the real code's."""
import json
from pathlib import Path

import pytest
from advisor_helpers import EMPTY_PORTFOLIO, TODAY, FakeClerk, MemoryLedger, Scripted, company, full_metrics, plan_reply, ready_order

from prometheus.agents.advisor.advisor import Advisor
from prometheus.application import recommendations as rec
from prometheus.common import work_order as wo

ROWS = json.loads((Path(__file__).parent / "fixtures" / "finnhub_pizza_rows.json").read_text())


def search(query):
    rows = ROWS.get(query, [])
    return rows if isinstance(rows, list) else []


# ---------------------------------------------------------------- classify


def test_the_kind_can_be_forced():
    assert rec.classify("Pizza", "phrase", search).kind == "phrase"
    assert rec.classify("dpz", "symbols", search).symbols == ["DPZ"]                          # forced as symbols, lower case is cleaned
    assert rec.classify("Domino's Pizza", "company", search).symbols == ["DPZ"]
    with pytest.raises(ValueError, match="no listed company is named 'Little Caesars'"):
        rec.classify("Little Caesars", "company", search)


def test_a_lookup_that_fails_makes_a_phrase_with_a_note_unless_a_company_was_demanded():
    def down(query):
        raise rec.symbol_tools.SymbolLookupError("the symbol lookup failed (HTTP 500)")
    request = rec.classify("Pizza", search=down)
    assert request.kind == "phrase" and "the company lookup failed" in request.note
    with pytest.raises(ValueError, match="the company lookup failed"):
        rec.classify("Pizza", "company", down)


@pytest.mark.parametrize("text, kind", [("", None), ("   ", None), ("x" * 121, None), ("../x", "symbols"), ("DPZ " + " ".join(f"S{i}" for i in range(15)), None), ("bad symbol!", "symbols")])
def test_a_request_that_cannot_be_one_is_refused(text, kind):
    with pytest.raises(ValueError):
        rec.classify(text, kind, search)


# ---------------------------------------------------------------- the run

TEXTS = [(1, "DPZ", "Domino's Pizza (DPZ) is the largest pizza chain and lifted its outlook.", "2026-10-22T10:00:00Z"), (2, "PZZA", "Papa John's (PZZA) is a large pizza chain.", "2026-10-22T10:00:00Z"),
         (3, "YUM", "Yum! Brands (YUM) owns Pizza Hut.", "2026-10-22T10:00:00Z")]
FOUND = [{"brand": "Domino's", "parent": "Domino's Pizza", "symbol": "DPZ", "company": "Domino's Pizza, Inc.", "public": True, "how": "company name", "note": None},
         {"brand": "Papa John's", "parent": "Papa John's", "symbol": "PZZA", "company": "Papa John's International Inc", "public": True, "how": "the start of the listed name", "note": None},
         {"brand": "Pizza Hut", "parent": "Yum! Brands", "symbol": "YUM", "company": "Yum! Brands, Inc.", "public": True, "how": "company name", "note": None},
         {"brand": "Little Caesars", "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "privately held, or not known to be listed: not scored"},
         {"brand": "Donatos", "parent": None, "symbol": None, "company": None, "public": False, "how": None, "note": "privately held, or not known to be listed: not scored"}]
METRICS = {"DPZ": full_metrics(), "PZZA": full_metrics(share_count_change=-0.008), "YUM": full_metrics(share_count_change=-0.006)}


class Clerk(FakeClerk):
    """The Data Clerk's calls the run makes, recording each: find_companies, refresh_data, request_data (fills the order with the texts), and request_metrics/discover for the Advisor."""

    def __init__(self, metrics=METRICS, found=FOUND, sources=TEXTS, ready=True, sector="XLY"):
        super().__init__(metrics)
        self.found, self.sources, self.ready, self.sector = found, sources, ready, sector

    def find_companies(self, phrase, count=10, today=None):
        self.asked.append(("find", phrase, count))
        return {"phrase": phrase, "companies": self.found, "sector_etf": self.sector, "notes": [], "run": {}}

    def refresh_data(self, symbols, **kw):
        self.asked.append(("refresh", list(symbols), kw))
        return [{"step": "daily_bars", "symbol": None, "outcome": "ok", "detail": {}}, {"step": "filings", "symbol": symbols[0], "outcome": "error", "detail": "ValueError: no Item heading"}]

    def request_data(self, order, today=None):
        self.asked.append(("data", list(order.subject), order.case, order.sector))
        if not self.ready:
            order.status, order.errors = "data_failed", ["no search of the stored news succeeded"]
            return order
        order.data, order.status = ready_order(subject=tuple(order.subject), sources=self.sources).data, "data_ready"
        return order


def theses(*symbols):
    return json.dumps({"theses": [{"symbol": s, "thesis": "Open it on its capital return: the expected excess return is an unmeasured prior.", "bull_case": [{"text": "Return on equity is 20.", "ref": "roe"}], "bear_case": []} for s in symbols]})


class Relevance:
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, query, tickers=None, k=1, min_score=None):
        score = self.scores.get(tickers[0])
        return [type("Hit", (), {"score": score})()] if score is not None else []


def go(request_text="Pizza", kind=None, clerk=None, portfolio=EMPTY_PORTFOLIO, replies=None, **kw):
    clerk = clerk or Clerk()
    model = Scripted(*(replies or (plan_reply(), theses("DPZ", "PZZA", "YUM"))))
    advisor = Advisor(model=model, clerk=clerk, ledger=MemoryLedger())
    result = rec.run(rec.classify(request_text, kind, search), clerk=clerk, advisor=advisor, portfolio=portfolio, today=TODAY, relevance_searches=(Relevance({"DPZ": 0.71, "PZZA": 0.66, "YUM": 0.31}),), **kw)
    return result, clerk, model


def test_no_company_found_is_a_note_and_no_order():
    result, clerk, model = go("Pizza", clerk=Clerk(found=FOUND[3:]))
    assert result.order is None and result.notes == ["no listed company was found to score"] and model.seen == []
    assert "NO RECOMMENDATIONS: no listed company was found to score" in rec.report(result)


def test_a_data_clerk_order_that_is_not_ready_stops_before_the_advisor_and_says_why():
    result, _, model = go("DPZ", clerk=Clerk(ready=False))
    assert result.order.status == "data_failed" and "the Data Clerk's order is not ready: no search of the stored news succeeded" in result.notes[0] and model.seen == []
    assert "NO RECOMMENDATIONS" in rec.report(result)


# ---------------------------------------------------------------- the report

def test_the_report_shows_the_companies_the_ranking_best_to_worst_and_each_recommendation_with_its_plan():
    text = rec.report(go("Pizza", find=5, fetch=True)[0])
    assert "search phrase 'Pizza'   |   case: industry   |   top 5" in text and "COMPANIES FOUND by the Data Clerk" in text
    assert "Pizza Hut" in text and "(owned by Yum! Brands)" in text and "Little Caesars" in text and "NOT SCORED" in text and "not scored (no listed company): Little Caesars, Donatos" in text
    assert "DATA REFRESH: 1 of 2 steps ok -- failed: filings DPZ: ValueError: no Item heading" in text
    ranking = text[text.index("RANKING, best to worst"):text.index("RECOMMENDATIONS  (")]
    assert ranking.index("DPZ") < ranking.index("PZZA") < ranking.index("YUM") and "RECOMMENDED #1" in ranking and "on topic" in ranking and "weak" in ranking and "0.71" in ranking   # the relevance column
    assert "RECOMMENDATIONS  (3 of the top 5 asked for; the others did not pass: see the ranking above)" in text
    assert "plan    : entry" in text and "stop $" in text and "thesis  : Open it on its capital return" in text and "market regime: trend" in text
    assert "size    :" not in text and "of equity" not in text and "funds feasible" not in text and "share(s)" not in text          # entry 80: the Advisor's plan has no size, so the report shows none
    assert "MODEL CALLS (Advisor): plan: 1 call(s)" in text and "Nothing here is bought or sold" in text


def test_a_model_that_fails_leaves_the_numbers_and_says_the_thesis_is_not_written():
    result, _, _ = go("Pizza", replies=(plan_reply(), RuntimeError("down")))
    text = rec.report(result)
    assert "thesis  : not written (the thesis model call failed (RuntimeError))" in text and "RECOMMENDED #1" in text and "ERROR the thesis model call failed" in text


# ---------------------------------------------------------------- relevance (ChromaDB), on its own


def test_an_index_that_cannot_be_read_is_no_number_never_a_guess():
    def broken(*a, **k):
        raise RuntimeError("index unreadable")
    assert rec.phrase_relevance("Pizza", ["DPZ"], (broken,)) == {"DPZ": None}


# ---------------------------------------------------------------- the command

def fake_main(monkeypatch, capsys, *argv, critic=False):
    argv = argv if critic else (*argv, "--no-critic")      # these tests are about the Advisor's report; tests/test_recommendations_critic.py runs the Critic
    clerk = Clerk()
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.DataClerk", lambda **kw: clerk)
    monkeypatch.setattr("prometheus.agents.advisor.advisor.Advisor", lambda **kw: Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=MemoryLedger()))
    monkeypatch.setattr(rec, "classify", lambda text, kind=None, search=None, _real=rec.classify: _real(text, kind, search or __import__("test_recommendations").search))
    monkeypatch.setattr(rec, "phrase_relevance", lambda phrase, symbols, searches=None: {s: 0.5 for s in symbols})
    code = rec.main(list(argv))
    return code, capsys.readouterr(), clerk


def test_the_command_refuses_bad_options_and_a_missing_key_with_exit_2(monkeypatch, capsys):
    for argv in (("Pizza", "--find", "0"), ("Pizza", "--phrase", "Pizza"), ("Pizza", "--from-days-ago", "99")):
        with pytest.raises(SystemExit) as stop:
            rec.main(list(argv))
        assert stop.value.code == 2
    capsys.readouterr()
    assert rec.main(["x" * 130, "--cash", "10"]) == 2 and "longer than 120" in capsys.readouterr().err
    assert rec.main(["--find", "3"]) == 2 and "give a symbol, a list of symbols, a company name or a search phrase" in capsys.readouterr().err

    def no_key(**kw):
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env; see .env.example)")
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.DataClerk", no_key)
    monkeypatch.setattr(rec, "classify", lambda text, kind=None, search=None: rec.Request("symbols", text, symbols=["DPZ"]))
    assert rec.main(["DPZ", "--cash", "10"]) == 2 and "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


# ---------------------------------------------------------------- a list to compare: tickers, names and brands


def test_a_list_can_be_forced_and_must_have_two_to_fifteen_items():
    assert rec.classify("Burger King, Wendy's", "list", search).items == ["Burger King", "Wendy's"]
    for bad in ("MCD", "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P", "MCD, " + "x" * 61):
        with pytest.raises(ValueError):
            rec.classify(bad, "list", search)


class ListClerk(Clerk):
    """find_companies answers one item at a time, as the real one does for count=1; a brand has a listed owner."""
    OWNERS = {"Burger King": {"brand": "Burger King", "parent": "Restaurant Brands International", "symbol": "QSR", "company": "Restaurant Brands International Inc", "public": True, "how": "company name", "note": None},
              "McDonald's": {"brand": "McDonald's", "parent": "McDonald's", "symbol": "MCD", "company": "McDonald's Corp", "public": True, "how": "company name", "note": None},
              "Donatos": FOUND[4]}

    def find_companies(self, phrase, count=10, today=None):
        self.asked.append(("find", phrase, count))
        extra = [{"brand": "WEN", "parent": None, "symbol": "WEN", "company": None, "public": True, "how": "stored market news", "note": "5 stored articles"}]
        return {"phrase": phrase, "companies": [self.OWNERS[phrase], *extra] if phrase in self.OWNERS else [], "sector_etf": "XLY", "notes": [], "run": {}}


def test_an_item_with_no_listed_company_is_kept_unscored_and_the_same_company_twice_is_once():
    metrics = {s: full_metrics() for s in ("MCD", "YUM")}
    result, _, _ = go("MCD, YUM, Donatos, McDonald's", clerk=ListClerk(metrics), replies=(plan_reply(), theses("MCD", "YUM")))
    assert [c["symbol"] for c in result.companies] == ["MCD", "YUM", None]                                                 # Donatos: private; "McDonald's" is MCD again
    assert result.companies[2]["note"] == "privately held, or not known to be listed: not scored"
    unknown, _, _ = go("MCD, YUM, Blargh Co", clerk=ListClerk(metrics), replies=(plan_reply(), theses("MCD", "YUM")))
    assert unknown.companies[2]["symbol"] is None and "no listed company found for this name" in unknown.companies[2]["note"]


# ---------------------------------------------------------------- the Advisor's model settings and the time limit of a model call

def advisor_model_settings(monkeypatch, **kw):
    seen = {}
    monkeypatch.setattr("prometheus.agents.advisor.advisor.get_chat_model", lambda role, model=None, **options: seen.update(role=role, model=model, **options) or object())
    monkeypatch.setattr("prometheus.models.chat.model_name", lambda role: "nvidia/nemotron-3-super-120b-a12b")        # the ADVISOR_MODEL of .env, whatever it is today
    Advisor(**kw)
    return seen


def test_gpt_oss_gets_low_effort_and_cerebras_then_groq_and_any_other_model_gets_neither(monkeypatch):
    oss = advisor_model_settings(monkeypatch, model_id="openai/gpt-oss-120b")
    assert oss["extra_body"] == {"reasoning": {"effort": "low"}, "provider": {"order": ["cerebras", "groq"], "allow_fallbacks": True}} and oss["model"] == "openai/gpt-oss-120b" and oss["role"] == "advisor"
    assert "extra_body" not in advisor_model_settings(monkeypatch, model_id="deepseek/deepseek-v4.1-flash")
    assert "extra_body" not in advisor_model_settings(monkeypatch)                                                       # the default Advisor (nemotron) is unchanged
    forced = advisor_model_settings(monkeypatch, model_id="x/y", effort="high", providers=("groq",))
    assert forced["extra_body"] == {"reasoning": {"effort": "high"}, "provider": {"order": ["groq"], "allow_fallbacks": True}}
    off = advisor_model_settings(monkeypatch, model_id="openai/gpt-oss-120b", effort=None, providers=None)
    assert "extra_body" not in off
    assert advisor_model_settings(monkeypatch, model_id="openai/gpt-oss-120b", providers="auto", effort="medium")["extra_body"]["reasoning"] == {"effort": "medium"}


# ---------------------------------------------------------------- negative equity is marked everywhere in the results (entry 54)

def deficit_run(**kw):
    metrics = {"DPZ": full_metrics(debt_to_equity=None, negative_equity=True), "PZZA": full_metrics(debt_to_equity=33.6, negative_equity=True),
               "YUM": full_metrics(debt_to_equity=31.0, current_ratio=0.9, negative_equity=False)}
    return go("DPZ PZZA YUM", clerk=Clerk(metrics), replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), find=5, **kw)


def test_the_report_marks_negative_equity_in_the_ranking_the_numbers_and_the_recommendation():
    result, _, _ = deficit_run()
    text = rec.report(result)
    ranking = text[text.index("RANKING"):text.index("KEY NUMBERS")]
    assert "DPZ*" in ranking and "PZZA*" in ranking and "YUM*" not in ranking and "YUM " in ranking
    assert "* NEGATIVE EQUITY (DPZ, PZZA): a stockholders' deficit, so debt/equity has no meaning: the debt/equity rules were WAIVED" in ranking
    assert "ranked by the blended score, the highest first" in ranking
    assert "Excess" not in ranking and "expected 20-day excess" not in text                           # Amendment 6: no profit number in the ranking or the recommendations
    table = text[text.index("KEY NUMBERS"):text.index("RECOMMENDATIONS  (")]
    assert "waived" in [line for line in table.split("\n") if line.strip().startswith("Debt / equity")][0] and "DEFICIT" in table and "positive" in table
    assert "equity  : NEGATIVE (stockholders' deficit): the debt/equity rules were waived for this company" in text
    assert text.count("equity  : NEGATIVE") == 2                                                                              # DPZ and PZZA, not YUM


# ---------------------------------------------------------------- the blend in the report (entry 55)


def test_blend_lines_names_the_newest_items_with_their_age_and_weight():
    blend = {"score": 62, "value": 0.24, "half_life_days": 7, "weights": {}, "components": {
        "news": {"weight": 0.4, "value": 0.4, "available": True, "n": 4, "contribution": 0.16, "parts": [
            {"ref": "[1]", "kind": "news", "direction": "bullish", "age_days": 1, "recency": 0.9, "weight": 0.9}, {"ref": "[2]", "kind": "news", "direction": "bearish", "age_days": 14, "recency": 0.25, "weight": 0.25},
            {"ref": "[3]", "kind": "press_release", "direction": "bullish", "age_days": None, "recency": 0.5, "weight": 0.25}, {"ref": "[4]", "kind": "news", "direction": "bullish", "age_days": 30, "recency": 0.05, "weight": 0.05}]},
        "performance": {"weight": 0.3, "value": 0.1, "available": True, "n": 1, "contribution": 0.03, "parts": []}, "historical": {"weight": 0.2, "value": 0.2, "available": True, "n": 1, "contribution": 0.04, "parts": []},
        "filings": {"weight": 0.1, "value": 0.0, "available": False, "n": 0, "contribution": 0.0, "parts": []}}}
    lines = rec.blend_lines(blend)
    assert lines[0].startswith("      blend   : 62 (0-100; +0.24) = news + press releases +0.40 x 40% stock performance +0.10 x 30%") and lines[0].endswith("filings +0.00 x 10% (n/a: counted as 0)")
    assert "4 item(s)" in lines[1] and "[1] bullish 1d old, weight 0.90" in lines[1] and "[3] bullish ?d old, weight 0.25" in lines[1] and "[4]" not in lines[1]       # the three heaviest
    assert rec.blend_lines(None) == []


# ---------------------------------------------------------------- --no-ledger: explore without the real cooldowns


def test_the_command_with_no_ledger_gives_the_advisor_a_throwaway_one(monkeypatch, capsys):
    seen = {}
    clerk = Clerk()
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.DataClerk", lambda **kw: clerk)
    monkeypatch.setattr("prometheus.agents.advisor.advisor.Advisor", lambda **kw: seen.update(kw) or Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=kw["ledger"] or MemoryLedger()))
    monkeypatch.setattr(rec, "classify", lambda text, kind=None, search=None: rec.Request("symbols", text, symbols=["DPZ", "PZZA", "YUM"]))
    assert rec.main(["DPZ PZZA YUM", "--cash", "1000", "--no-ledger"]) == 0
    assert seen["ledger"] is not None and "prometheus_ledger_" in str(seen["ledger"].path)
    assert rec.main(["DPZ PZZA YUM", "--cash", "1000"]) == 0 and seen["ledger"] is None                                      # the default: the Advisor's own (the real) ledger
