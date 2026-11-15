"""Tests for prometheus.application.recommendations (design choices entry 53): classify a request, run the steps, print the report. The Data Clerk and the model are fakes; the Finnhub rows of the
company lookups are REAL (tests/fixtures/finnhub_pizza_rows.json); the Advisor is the real one on fake numbers, so the ranking and the report are the real code's."""
import json
from pathlib import Path

import pytest
from advisor_helpers import EMPTY_PORTFOLIO, TODAY, FakeClerk, MemoryLedger, Scripted, advisor_with, company, full_metrics, plan_reply, ready_order, thesis_reply

from prometheus.agents.advisor.advisor import Advisor
from prometheus.application import recommendations as rec
from prometheus.common import work_order as wo
from prometheus.common.portfolio import Portfolio, Position

ROWS = json.loads((Path(__file__).parent / "fixtures" / "finnhub_pizza_rows.json").read_text())


def search(query):
    rows = ROWS.get(query, [])
    return rows if isinstance(rows, list) else []


# ---------------------------------------------------------------- classify

@pytest.mark.parametrize("text, symbols", [("DPZ", ["DPZ"]), ("DPZ PZZA YUM", ["DPZ", "PZZA", "YUM"]), ("DPZ,PZZA", ["DPZ", "PZZA"]), ("DPZ, PZZA , YUM", ["DPZ", "PZZA", "YUM"]),
                                           ("$DPZ", ["DPZ"]), ("DPZ DPZ", ["DPZ"]), ("BRK.B", ["BRK.B"])])
def test_a_ticker_or_a_list_of_tickers_in_capitals_is_symbols(text, symbols):
    request = rec.classify(text, search=search)
    assert request.kind == "symbols" and request.symbols == symbols


def test_text_that_is_exactly_a_listed_companys_name_is_a_company():
    request = rec.classify("Domino's Pizza", search=search)
    assert request.kind == "company" and request.symbols == ["DPZ"] and request.company == "Domino's Pizza, Inc."


@pytest.mark.parametrize("text", ["Pizza", "pizza delivery", "weight loss drugs", "AI memory shortage", "Little Caesars"])
def test_anything_else_is_a_phrase(text):
    request = rec.classify(text, search=search)
    assert request.kind == "phrase" and request.phrase == text and request.symbols == []


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
METRICS = {"DPZ": full_metrics(net_payout_yield_ttm=0.07), "PZZA": full_metrics(net_payout_yield_ttm=0.045), "YUM": full_metrics(net_payout_yield_ttm=0.06, share_count_change=-0.004)}


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


def test_a_phrase_finds_the_companies_orders_the_data_ranks_them_and_leaves_the_private_ones_unscored():
    result, clerk, _ = go("Pizza", find=5)
    assert result.case == "industry" and result.sector == "XLY" and [c["symbol"] for c in result.companies] == ["DPZ", "PZZA", "YUM", None, None]
    assert clerk.asked[0] == ("find", "Pizza", 10) and clerk.asked[1][:3] == ("data", ["DPZ", "PZZA", "YUM"], "industry") and clerk.asked[1][3] == "XLY"
    order = result.order
    assert order.status == "recommended" and [r["symbol"] for r in order.recommendations] == ["DPZ", "PZZA", "YUM"]                       # best to worst by the real ranking code
    assert [r["ranking"]["universe_rank"] for r in order.recommendations] == [1, 2, 3] and [r["identity"]["sleeve"] for r in order.recommendations] == ["industry"] * 3
    scores = [r["score"] for r in order.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert result.relevance == {"DPZ": 0.71, "PZZA": 0.66, "YUM": 0.31} and set(result.seconds) >= {"find", "data_clerk", "advisor", "relevance"}


def test_find_is_the_number_of_new_ideas_and_the_company_finder_is_asked_for_more_than_that():
    result, clerk, _ = go("Pizza", find=2, replies=(plan_reply(), theses("DPZ", "PZZA")))
    assert [r["symbol"] for r in result.order.recommendations] == ["DPZ", "PZZA"] and clerk.asked[0] == ("find", "Pizza", 10)
    yum = next(c for c in result.order.candidates if c["symbol"] == "YUM")
    assert yum["rank"] == 3 and yum["reason"] == "rank 3 is beyond the top 2"
    result, clerk, _ = go("Pizza", find=8, replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")))
    assert clerk.asked[0] == ("find", "Pizza", 16)


def test_symbols_company_and_phrase_use_the_right_case_and_only_a_phrase_calls_the_company_finder():
    result, clerk, _ = go("DPZ PZZA YUM", find=3)
    assert result.case == "funds" and not any(a[0] == "find" for a in clerk.asked) and [c["how"] for c in result.companies] == ["symbol as typed"] * 3 and result.relevance == {}
    company_run, _, _ = go("Domino's Pizza", replies=(plan_reply(), theses("DPZ")))
    assert company_run.case == "funds" and company_run.companies[0]["company"] == "Domino's Pizza, Inc." and company_run.companies[0]["how"] == "company name"
    forced, _, _ = go("Pizza", case="funds")
    assert forced.case == "funds"


def test_nothing_is_fetched_without_fetch_and_with_it_the_holdings_get_their_bars():
    quiet, clerk, _ = go("Pizza")
    assert quiet.refresh == [] and not any(a[0] == "refresh" for a in clerk.asked)
    portfolio = Portfolio(user="t", cash_avail=500.0, positions=[Position(symbol="MCD", shares=3, cost_basis=100.0)])
    fetched, clerk, _ = go("Pizza", fetch=True, portfolio=portfolio, news_days=10, filings_days=200, replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=Clerk({**METRICS, "MCD": full_metrics()}))
    call = next(a for a in clerk.asked if a[0] == "refresh")
    assert call[1] == ["DPZ", "PZZA", "YUM"] and call[2] == {"news_days": 10, "filings_days": 200, "bars_for": ["MCD"]} and len(fetched.refresh) == 2


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


def test_a_company_that_is_not_recommended_is_in_the_ranking_with_its_reason_and_a_gate_failure_is_ranked_last():
    metrics = {**METRICS, "YUM": full_metrics(net_payout_yield_ttm=0.08, adv_20d_dollars=1000.0), "PZZA": full_metrics(earnings_yield=0.0, net_payout_yield_ttm=0.035, share_count_change=-0.006)}
    result, _, _ = go("Pizza", clerk=Clerk(metrics), replies=(plan_reply(), theses("DPZ")))
    text = rec.report(result)
    rows = rec.ranking_rows(result)
    assert [(r["rank"], r["symbol"], r["status"]) for r in rows] == [(1, "DPZ", "RECOMMENDED"), (2, "PZZA", "not recommended"), (3, "YUM", "not recommended")]    # YUM has the best numbers but fails liquidity: last
    assert "no: gate liquidity failed" in text and "no: stance neutral (no family passed)" in text
    assert "RECOMMENDATIONS  (1 of the top 5 asked for; the others did not pass" in text


def test_the_holdings_are_reviewed_in_their_own_section_and_are_not_ranked_with_the_new_ideas():
    portfolio = Portfolio(user="t", cash_avail=500.0, positions=[Position(symbol="MCD", name="McDonald's Corp", shares=10, cost_basis=40.0)])
    result, _, _ = go("Pizza", portfolio=portfolio, clerk=Clerk({**METRICS, "MCD": full_metrics(cfo_negative_with_profit=True, last_close=300.0)}), replies=(plan_reply(), theses("MCD", "DPZ", "PZZA", "YUM")))
    text = rec.report(result)
    assert [r["symbol"] for r in rec.ranking_rows(result)] == ["DPZ", "PZZA", "YUM"] and "YOUR HOLDINGS (reviewed before the new ideas)" in text and "MCD     REDUCE" in text


def test_a_model_that_fails_leaves_the_numbers_and_says_the_thesis_is_not_written():
    result, _, _ = go("Pizza", replies=(plan_reply(), RuntimeError("down")))
    text = rec.report(result)
    assert "thesis  : not written (the thesis model call failed (RuntimeError))" in text and "RECOMMENDED #1" in text and "ERROR the thesis model call failed" in text


def test_the_report_is_plain_text_with_no_secrets_and_no_model_call():
    result, _, model = go("Pizza")
    before = len(model.seen)
    text = rec.report(result)
    assert len(model.seen) == before and "api_key" not in text.lower() and "token" not in text.lower() and isinstance(text, str)


# ---------------------------------------------------------------- relevance (ChromaDB), on its own

def test_the_relevance_of_a_phrase_is_the_best_similarity_in_a_companys_stored_news_and_press_releases():
    news, press = Relevance({"DPZ": 0.5, "YUM": 0.2}), Relevance({"DPZ": 0.8, "PZZA": 0.45})
    assert rec.phrase_relevance("Pizza", ["DPZ", "PZZA", "YUM", "ZZZ"], (news, press)) == {"DPZ": 0.8, "PZZA": 0.45, "YUM": 0.2, "ZZZ": None}


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


def test_the_command_prints_the_report_and_saves_the_order(monkeypatch, capsys, tmp_path):
    save = tmp_path / "order.json"
    code, out, clerk = fake_main(monkeypatch, capsys, "Pizza", "--find", "3", "--cash", "1000", "--today", "2026-10-23", "--save", str(save))
    assert code == 0 and "PROMETHEUS RECOMMENDATIONS" in out.out and "RANKING, best to worst" in out.out and "finished in" in out.out and ".. Finding the companies of 'Pizza'" in out.out
    assert wo.WorkOrder.model_validate_json(save.read_text()).status == "recommended" and f"(work order saved to {save})" in out.out
    assert not any(a[0] == "refresh" for a in clerk.asked)                                                                  # no --fetch: nothing is fetched


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

@pytest.mark.parametrize("text", ['MCD, YUM, "Burger King"', "MCD, YUM, Burger King", "MCD, YUM and Burger King, QSR", 'MCD,YUM,"Burger King"', "MCD, Burger King"])
def test_a_comma_list_with_a_ticker_and_a_name_or_brand_is_a_list_in_the_order_typed(text):
    request = rec.classify(text, search=search)
    assert request.kind == "list" and request.items[0] == "MCD" and "Burger King" in request.items and request.symbols[0] == "MCD" and "Burger King" not in request.symbols


def test_the_items_of_a_list_are_split_on_commas_and_and_and_ampersand_and_quotes_are_dropped():
    request = rec.classify('MCD and "Burger King", $YUM & QSR', search=search)
    assert request.items == ["MCD", "Burger King", "YUM", "QSR"] and request.symbols == ["MCD", "YUM", "QSR"]


@pytest.mark.parametrize("text", ["AI and robotics", "Mac and cheese", "weight loss, GLP-1 drugs", "Burger King", "MCD YUM", "DPZ, PZZA"])
def test_text_that_is_not_such_a_list_keeps_its_old_kind(text):
    assert rec.classify(text, search=search).kind in ("phrase", "symbols")       # no ticker with a name and a comma (or quotes): a phrase; every item a ticker: symbols


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


def test_a_list_looks_up_a_brand_for_its_listed_owner_keeps_tickers_as_typed_and_ranks_them_as_an_industry():
    metrics = {s: full_metrics() for s in ("MCD", "YUM", "QSR")}
    result, clerk, _ = go('MCD, YUM, "Burger King"', clerk=ListClerk(metrics, sources=[(1, "MCD", "McDonald's (MCD) is a burger chain.", "2026-10-22T10:00:00Z"), (2, "YUM", "Yum! Brands (YUM) owns Taco Bell.", "2026-10-22T10:00:00Z"),
                                                                                       (3, "QSR", "Restaurant Brands (QSR) owns Burger King.", "2026-10-22T10:00:00Z")]), replies=(plan_reply(), theses("MCD", "YUM", "QSR")), find=3)
    assert [c["symbol"] for c in result.companies] == ["MCD", "YUM", "QSR"] and result.companies[2]["parent"] == "Restaurant Brands International" and result.case == "industry" and result.sector == "XLY"
    assert clerk.asked[0] == ("find", "Burger King", 1) and not any(a[:2] == ("find", "MCD") for a in clerk.asked) and all(c["symbol"] != "WEN" for c in result.companies)   # the news-added ticker is left out
    assert clerk.asked[1][:3] == ("data", ["MCD", "YUM", "QSR"], "industry") and result.relevance == {}
    text = rec.report(result)
    assert "comparison of MCD, YUM, Burger King" in text and "owned by Restaurant Brands International" in text and "tickers as typed" in text and "RANKING, best to worst" in text and "QSR" in text


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


def test_a_scripted_model_is_used_as_is_with_no_settings(monkeypatch):
    seen = advisor_model_settings(monkeypatch, model=Scripted())
    assert seen == {}


def test_a_model_call_that_hangs_is_given_up_after_the_limit_and_is_a_run_error_not_an_exception():
    import time
    from prometheus.agents.advisor import calls

    class Hangs:
        model_name = "slow"

        def invoke(self, messages):
            time.sleep(5)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="more than 0.05 seconds"):
        calls.invoke_with_limit(Hangs(), [], 0.05)
    assert time.monotonic() - started < 2
    calls.MODEL_TIMEOUT, saved = 0.05, calls.MODEL_TIMEOUT
    try:
        run = calls.new_run("plan", Hangs())
        assert calls.ask_json(Hangs(), [], run, "companies", "again") is None and run["error"] == "the plan model call failed (TimeoutError)" and run["calls"] == 0
    finally:
        calls.MODEL_TIMEOUT = saved


def test_a_model_call_that_answers_or_fails_in_time_behaves_as_before():
    from prometheus.agents.advisor import calls

    class Replies:
        def invoke(self, messages):
            return "reply"

    class Fails:
        def invoke(self, messages):
            raise ValueError("bad")

    assert calls.invoke_with_limit(Replies(), []) == "reply"
    with pytest.raises(ValueError, match="bad"):
        calls.invoke_with_limit(Fails(), [])


# ---------------------------------------------------------------- only the companies that were named, and the comparison sections of the report

def test_only_subject_leaves_out_the_symbols_the_sources_name_but_nobody_asked_for():
    from prometheus.agents.advisor.candidates import collect_candidates
    order = ready_order(subject=("DPZ",), sources=[(1, "DPZ", "Domino's Pizza (DPZ) lifted its outlook.", "2026-10-22T10:00:00Z"), (2, "WEN", "Wendy's (WEN) is another burger chain.", "2026-10-22T10:00:00Z")])
    everyone, _ = collect_candidates(order, EMPTY_PORTFOLIO, "industry", None, 15, 3)
    named, _ = collect_candidates(order, EMPTY_PORTFOLIO, "industry", None, 15, 3, only_subject=True)
    assert [c["symbol"] for c in everyone] == ["DPZ", "WEN"] and [c["symbol"] for c in named] == ["DPZ"]


def test_a_run_ranks_only_the_named_companies_even_when_the_news_names_others():
    sources = [*TEXTS, (4, "WEN", "Wendy's (WEN) is another burger chain.", "2026-10-22T10:00:00Z"), (5, "CMG", "Chipotle (CMG) is a burrito chain.", "2026-10-22T10:00:00Z")]
    result, _, _ = go("DPZ PZZA YUM", clerk=Clerk({**METRICS, "WEN": full_metrics(), "CMG": full_metrics()}, sources=sources), replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), find=5)
    assert sorted(r["symbol"] for r in rec.ranking_rows(result)) == ["DPZ", "PZZA", "YUM"]


def test_the_report_has_a_table_of_key_numbers_and_says_why_each_company_was_not_recommended():
    metrics = {**METRICS, "YUM": full_metrics(net_payout_yield_ttm=0.08, adv_20d_dollars=1000.0), "PZZA": full_metrics(earnings_yield=0.0, net_payout_yield_ttm=0.035, share_count_change=-0.006)}
    result, _, _ = go("DPZ PZZA YUM", clerk=Clerk(metrics), replies=(plan_reply(), theses("DPZ")))
    text = rec.report(result)
    table = text[text.index("KEY NUMBERS"):text.index("WHY NOT RECOMMENDED")]
    assert "DPZ" in table and "PZZA" in table and "YUM" in table and "Last close" in table and "Market cap" in table and "Debt / equity" in table and "$100.00" in table
    why = text[text.index("WHY NOT RECOMMENDED"):text.index("RECOMMENDATIONS  (")]
    assert "PZZA" in why and "YUM" in why and "DPZ" not in why.split("\n")[1] and "trend" in why and "capital_return" in why and "gate liquidity failed" in why


def test_key_numbers_formats_each_kind_and_leaves_out_a_row_nobody_has():
    assert [rec._number(v, k) for v, k in ((None, "frac"), (0.1234, "frac"), (278.044, "usd"), (198.3e9, "big"), (225e6, "big"), (31.85, "num1"), (3.514, "num2"))] == ["n/a", "12.3%", "$278.04", "$198.3B", "$225M", "31.9", "3.51"]
    result, _, _ = go("DPZ PZZA YUM", replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")))
    rows = rec.key_numbers(result, ["DPZ", "PZZA", "YUM"])
    assert rows[0].split() == ["DPZ", "PZZA", "YUM"] and rows[1].startswith("  Last close") and rows[1].split()[-3:] == ["$100.00"] * 3
    assert all(len(row) == len(rows[0]) for row in rows)                                                                     # every row is as wide as the header: the columns line up
    assert rec.key_numbers(result, ["NOPE"]) == []


# ---------------------------------------------------------------- negative equity is marked everywhere in the results (entry 54)

def deficit_run(**kw):
    metrics = {"DPZ": full_metrics(net_payout_yield_ttm=0.07, debt_to_equity=None, negative_equity=True), "PZZA": full_metrics(net_payout_yield_ttm=0.045, debt_to_equity=33.6, negative_equity=True),
               "YUM": full_metrics(net_payout_yield_ttm=0.06, debt_to_equity=31.0, current_ratio=0.9, negative_equity=False)}
    return go("DPZ PZZA YUM", clerk=Clerk(metrics), replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), find=5, **kw)


def test_a_company_with_negative_equity_is_computed_and_flagged_on_its_record_and_candidates():
    result, _, _ = deficit_run()
    by_symbol = {r["symbol"]: r for r in result.order.recommendations}
    assert by_symbol["DPZ"]["negative_equity"] and by_symbol["PZZA"]["negative_equity"] and "YUM" not in by_symbol
    yum = next(c for c in result.order.candidates if c["symbol"] == "YUM")                                                    # positive equity and a debt/equity of 31: the rule applies and the gate stops it
    assert not yum["negative_equity"] and yum["gate"] == "red_flag" and "debt_to_equity 31 with current_ratio" in yum["reason"]
    assert all(f["waived"] == ["debt_to_equity"] for r in (by_symbol["DPZ"], by_symbol["PZZA"]) for f in r["families"] if f["family"] in ("reversion", "capital_return"))
    assert [n for r in (by_symbol["DPZ"],) for g in r["gates"] if g["gate"] == "red_flag" for n in [g["note"]]] == ["debt_to_equity rule waived: negative stockholders' equity (the ratio has no meaning)"]
    assert "debt_to_equity: debt_to_equity rule waived: negative stockholders' equity (the ratio has no meaning)" in by_symbol["PZZA"]["quality"]["haircuts_skipped"]


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


def test_a_company_without_a_deficit_is_not_marked():
    result, _, _ = go("DPZ PZZA YUM", find=5)
    text = rec.report(result)
    assert "NEGATIVE EQUITY" not in text and "*" not in text[text.index("RANKING"):text.index("RECOMMENDATIONS  (")]


# ---------------------------------------------------------------- the blend in the report (entry 55)

def test_the_ranking_shows_the_blend_and_its_four_parts_with_a_legend_and_a_recommendation_explains_them():
    result, _, _ = go("DPZ PZZA YUM", replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), find=5)
    text = rec.report(result)
    ranking = text[text.index("RANKING"):text.index("KEY NUMBERS")]
    assert all(h in ranking for h in ("Blend", "News", "Price", "Hist", "File"))
    assert "Blend (0-100) = 40% news and press releases (the newer the heavier) + 30% stock performance + 20% historical metrics + 10% filings" in ranking and "n/a = nothing to" in ranking
    first = result.order.recommendations[0]["blend"]
    row = next(line for line in ranking.split("\n") if line.strip().startswith("1 "))
    assert str(first["score"]) in row and f"{first['components']['performance']['value']:+.2f}" in row
    detail = text[text.index("RECOMMENDATIONS  ("):]
    assert f"blend   : {first['score']} (0-100; {first['value']:+.2f}) = news + press releases" in detail and "x 40%" in detail and "x 30%" in detail and "x 20%" in detail and "x 10%" in detail
    assert "news    :" not in detail                                                                                          # this plan reply names no quote: no news item to list


def test_a_part_with_nothing_to_say_is_n_a_in_the_table_and_counted_as_zero_in_the_blend_line():
    result, _, _ = go("DPZ PZZA YUM", replies=(plan_reply(), theses("DPZ", "PZZA", "YUM")), find=5)
    blend = result.order.recommendations[0]["blend"]
    assert blend["components"]["news"]["available"] is False                                                              # the plan reply of this test names no quote
    text = rec.report(result)
    assert "n/a" in text[text.index("RANKING"):text.index("KEY NUMBERS")] and "news + press releases +0.00 x 40% (n/a: counted as 0)" in text


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

def test_a_throwaway_ledger_is_empty_and_outside_the_repo_and_the_real_ledger_is_not_touched(tmp_path):
    from prometheus.memory.ledger import DEFAULT_PATH, Ledger
    ledger = rec.throwaway_ledger()
    assert ledger.read() == [] and "data/ledger" not in str(ledger.path) and str(ledger.path).startswith(__import__("tempfile").gettempdir())
    ledger.append({"x": 1})
    assert ledger.read() == [{"x": 1}] and rec.throwaway_ledger().path != ledger.path and Ledger().path.name == DEFAULT_PATH.name


def test_the_command_with_no_ledger_gives_the_advisor_a_throwaway_one(monkeypatch, capsys):
    seen = {}
    clerk = Clerk()
    monkeypatch.setattr("prometheus.agents.data_clerk.data_clerk_agent.DataClerk", lambda **kw: clerk)
    monkeypatch.setattr("prometheus.agents.advisor.advisor.Advisor", lambda **kw: seen.update(kw) or Advisor(model=Scripted(plan_reply(), theses("DPZ", "PZZA", "YUM")), clerk=clerk, ledger=kw["ledger"] or MemoryLedger()))
    monkeypatch.setattr(rec, "classify", lambda text, kind=None, search=None: rec.Request("symbols", text, symbols=["DPZ", "PZZA", "YUM"]))
    assert rec.main(["DPZ PZZA YUM", "--cash", "1000", "--no-ledger"]) == 0
    assert seen["ledger"] is not None and "prometheus_ledger_" in str(seen["ledger"].path)
    assert rec.main(["DPZ PZZA YUM", "--cash", "1000"]) == 0 and seen["ledger"] is None                                      # the default: the Advisor's own (the real) ledger
