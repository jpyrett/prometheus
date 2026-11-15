"""Tests for DataClerk.find_companies and DataClerk.refresh_data (design choices entry 53): a phrase ("Pizza") becomes real tickers, and the data for them is fetched.

The Finnhub directory rows are REAL (tests/fixtures/finnhub_pizza_rows.json, fetched on 2026-11-05: Domino's, Papa John's, Yum! Brands, and the nothing that Little Caesars and Donatos
(private) give). The model is scripted: what the tests check is that CODE decides what a name is, whatever the model says.
"""
import json
from pathlib import Path

import pytest
from advisor_helpers import Scripted

from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
from prometheus.tools import symbols

ROWS = json.loads((Path(__file__).parent / "fixtures" / "finnhub_pizza_rows.json").read_text())


class Search:
    """Finnhub's symbol search as recorded: a query it has no recorded answer for finds nothing."""

    def __init__(self, fail_429=0):
        self.asked, self.fail_429 = [], fail_429

    def __call__(self, query):
        self.asked.append(query)
        if self.fail_429:
            self.fail_429 -= 1
            raise symbols.SymbolLookupError("the symbol lookup failed (HTTP 429)")
        rows = ROWS.get(query, [])
        return rows if isinstance(rows, list) else []


class Tools:
    def __init__(self, found=(), fail=False):
        self.found, self.fail = list(found), fail

    def discover_tickers(self, phrase, from_date=7, top=10):
        if self.fail:
            raise RuntimeError("store unreadable")
        return {"tickers": self.found}


def proposal(*companies, sector="XLY"):
    return json.dumps({"companies": list(companies), "sector_etf": sector})


PIZZA = [{"brand": "Domino's", "parent": "Domino's Pizza", "ticker": "DPZ"}, {"brand": "Papa John's", "parent": "Papa John's", "ticker": "PZZA"},
         {"brand": "Pizza Hut", "parent": "Yum! Brands", "ticker": "YUM"}, {"brand": "Little Caesars", "parent": None, "ticker": None}, {"brand": "Donatos", "parent": None, "ticker": None}]


def clerk(reply, search=None, tools=None):
    c = DataClerk(model=Scripted(*reply) if isinstance(reply, tuple) else Scripted(reply), symbol_search=search or Search())
    c.lookup_interval = 0
    c.tools = tools or Tools()
    return c


def test_a_phrase_becomes_real_tickers_by_lookup_and_private_companies_are_listed_as_not_scored():
    out = clerk(proposal(*PIZZA)).find_companies("Pizza", count=8)
    by_brand = {c["brand"]: c for c in out["companies"]}
    assert [c["symbol"] for c in out["companies"]] == ["DPZ", "PZZA", "YUM", None, None] and out["sector_etf"] == "XLY" and out["notes"] == []
    assert by_brand["Domino's"]["company"] == "Domino's Pizza, Inc." and by_brand["Domino's"]["how"] == """company name "Domino's Pizza": the same words as the listing"""   # the US listing, not DMP.AX or DOM.L
    assert by_brand["Papa John's"]["symbol"] == "PZZA" and by_brand["Papa John's"]["how"] == "the start of the listed name"                                           # the start of "Papa John's International Inc"
    assert by_brand["Pizza Hut"]["symbol"] == "YUM" and by_brand["Pizza Hut"]["parent"] == "Yum! Brands" and by_brand["Pizza Hut"]["company"] == "Yum! Brands, Inc."   # the brand is not listed: its owner is
    for private in ("Little Caesars", "Donatos"):
        assert by_brand[private]["public"] is False and by_brand[private]["symbol"] is None and "not scored" in by_brand[private]["note"]
    assert out["run"]["stage"] == "companies" and out["run"]["calls"] == 1 and out["run"]["error"] is None


def test_the_models_ticker_is_never_used_the_lookup_decides_and_a_wrong_guess_is_said():
    wrong = {"brand": "Domino's", "parent": "Domino's Pizza", "ticker": "DMP.AX"}
    row = clerk(proposal(wrong)).find_companies("Pizza")["companies"][0]
    assert row["symbol"] == "DPZ" and row["note"] == "the model guessed DMP.AX; the lookup says DPZ"


def test_a_ticker_hint_is_accepted_only_when_finnhub_has_it_and_its_name_shares_a_word_with_the_company():
    # "Yum Brands" (no "!") finds nothing in Finnhub's directory (it is punctuation sensitive), but the hint YUM exists and its name is "YUM! BRANDS INC"
    ok = clerk(proposal({"brand": "Pizza Hut", "parent": "Yum Brands", "ticker": "YUM"})).find_companies("Pizza")["companies"][0]
    assert ok["symbol"] == "YUM" and ok["how"].startswith("the model's ticker YUM, checked")
    nothing = clerk(proposal({"brand": "Mystery Pizza", "parent": "Zorblax Pizza Holdings", "ticker": "ZZZZ"})).find_companies("Pizza")["companies"][0]
    assert nothing["symbol"] is None and nothing["public"] is False and "no listed company named 'Zorblax Pizza Holdings'" in nothing["note"]
    unrelated = clerk(proposal({"brand": "Mystery Pizza", "parent": "Zorblax Pizza Holdings", "ticker": "DPZ"})).find_companies("Pizza")["companies"][0]
    assert unrelated["symbol"] is None                                                       # DPZ exists and "Domino's Pizza" shares the word "pizza" with it: one shared word is not enough


def test_a_rate_limited_lookup_waits_and_tries_again_it_is_not_not_found(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = clerk(proposal(PIZZA[0]), search=Search(fail_429=1))
    c.lookup_interval = 0.01
    assert c.find_companies("Pizza")["companies"][0]["symbol"] == "DPZ"


def test_a_lookup_that_keeps_failing_is_named_and_the_company_is_not_scored(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = clerk(proposal(PIZZA[0]), search=Search(fail_429=99))
    c.lookup_interval = 0.01
    out = c.find_companies("Pizza")
    assert out["companies"][0]["symbol"] is None and "lookup failed" in out["companies"][0]["note"] and any("the lookup failed" in n for n in out["notes"])


def test_tickers_the_stored_market_news_ties_to_the_phrase_are_added_when_they_have_enough_articles():
    found = [{"symbol": "CAKE", "articles": 5}, {"symbol": "WING", "articles": 2}, {"symbol": "DPZ", "articles": 9}]
    out = clerk(proposal(PIZZA[0]), tools=Tools(found)).find_companies("Pizza")
    assert [(c["symbol"], c["how"]) for c in out["companies"]][1:] == [("CAKE", "stored market news")] and out["companies"][0]["symbol"] == "DPZ"
    assert clerk(proposal(PIZZA[0]), tools=Tools(fail=True)).find_companies("Pizza")["notes"] == ["stored-news discovery failed (RuntimeError)"]


def test_a_model_that_fails_or_gives_junk_leaves_notes_and_never_raises():
    out = DataClerk(model=Scripted(RuntimeError("down")), symbol_search=Search())
    out.lookup_interval, out.tools = 0, Tools()
    result = out.find_companies("Pizza")
    assert result["companies"] == [] and result["notes"] == ["the companies model call failed (RuntimeError)"]
    junk = clerk(("not json", "still not json")).find_companies("Pizza")
    assert junk["companies"] == [] and "not usable" in junk["notes"][0]
    retry = clerk(("sorry", proposal(PIZZA[0]))).find_companies("Pizza")
    assert [c["symbol"] for c in retry["companies"]] == ["DPZ"] and retry["run"]["calls"] == 2


def test_the_proposals_are_capped_and_junk_entries_are_ignored():
    many = [{"brand": f"B{i}", "parent": None} for i in range(30)] + ["junk", 5]
    assert len(clerk(proposal(*many)).find_companies("Pizza", count=5)["companies"]) == 5


@pytest.mark.parametrize("phrase, count", [("", 5), ("   ", 5), ("a" * 61, 5), ("pizza; rm -rf /", 5), ("../x", 5), ("Pizza", 0), ("Pizza", 21), ("Pizza", True)])
def test_a_bad_phrase_or_count_is_refused_before_the_model_is_asked(phrase, count):
    model = Scripted()
    c = DataClerk(model=model, symbol_search=Search())
    with pytest.raises(ValueError):
        c.find_companies(phrase, count)
    assert model.seen == []


# ---------------------------------------------------------------- refresh_data

class FetchTools:
    """The tools refresh_data calls, recording each call; `fail` is {(step, symbol): Exception}."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail or {}
        self.toolbox = self

    def _do(self, step, symbol, args):
        self.calls.append((step, symbol, args))
        if (step, symbol) in self.fail:
            raise self.fail[(step, symbol)]

    def fetch_daily_bars(self, symbols, days, end):
        self._do("daily_bars", None, (list(symbols), days))
        return {s: {"stored": 276} for s in symbols}

    def fetch_news(self, symbol, from_date=7):
        self._do("news", symbol, from_date)
        return {"fetched": 10, "written": 9, "duplicates": 1, "indexed": {}}

    def fetch_press_releases(self, symbol, from_date=365):
        self._do("press_releases", symbol, from_date)
        return {"fetched": 3, "written": 3, "duplicates": 0}

    def fetch_filings(self, symbol, from_date=365):
        self._do("filings", symbol, from_date)
        return {"fetched": 51, "written": 51, "edgar_text": {"candidates": 4, "written": 4}}


def fetcher(allow=True, fail=None):
    c = DataClerk(model=object(), allow_fetch=allow)
    c.tools = FetchTools(fail)
    return c


def test_refresh_fetches_bars_for_the_symbols_spy_and_the_etfs_then_each_symbols_news_press_releases_and_filings():
    c = fetcher()
    report = c.refresh_data(["dpz", "$PZZA", "DPZ"], news_days=10, filings_days=300, bars_days=400)
    assert all(r["outcome"] == "ok" for r in report) and [(r["step"], r["symbol"]) for r in report] == [("daily_bars", None)] + [(s, sym) for sym in ("DPZ", "PZZA") for s in ("news", "press_releases", "filings")]
    bars = c.tools.calls[0]
    assert bars[2][0][:4] == ["DPZ", "PZZA", "SPY", "XLK"] and len(bars[2][0]) == 14 and bars[2][1] == 400
    assert ("news", "DPZ", 10) in c.tools.calls and ("filings", "PZZA", 300) in c.tools.calls and ("press_releases", "DPZ", 300) in c.tools.calls
    assert report[1]["detail"] == {"fetched": 10, "written": 9, "duplicates": 1}


def test_a_step_that_fails_is_reported_with_its_kind_and_the_other_steps_still_run():
    c = fetcher(fail={("filings", "DPZ"): ValueError("no Item heading was found (key=SECRET)"), ("news", "PZZA"): RuntimeError("HTTP 500")})
    report = c.refresh_data(["DPZ", "PZZA"])
    errors = {(r["step"], r["symbol"]): r["detail"] for r in report if r["outcome"] == "error"}
    assert set(errors) == {("filings", "DPZ"), ("news", "PZZA")} and errors[("news", "PZZA")] == "RuntimeError: HTTP 500"
    assert len(report) == 7 and [r["outcome"] for r in report if r["symbol"] == "PZZA" and r["step"] == "filings"] == ["ok"]


def test_without_allow_fetch_nothing_is_fetched_and_it_says_so():
    c = fetcher(allow=False)
    report = c.refresh_data(["DPZ"])
    assert report == [{"step": "all", "symbol": None, "outcome": "refused", "detail": report[0]["detail"]}] and "allow_fetch" in report[0]["detail"] and c.tools.calls == []


@pytest.mark.parametrize("symbols_", [[], ["../x"], [f"S{i}" for i in range(16)]])
def test_refresh_refuses_bad_symbols_before_any_call(symbols_):
    c = fetcher()
    with pytest.raises(ValueError):
        c.refresh_data(symbols_)
    assert c.tools.calls == []
