"""Tests for prometheus.tools.symbols: a company name becomes a ticker by lookup, and an ambiguous name is refused (Part 9 of the RAG guide).

Run:  pytest tests/test_symbols.py -q

No network and no key: the lookup is faked. The rows are copied from Finnhub's real answers (the Indian Procter & Gamble subsidiaries, the Canadian
depositary receipts, the look-alike tickers), so the awkward cases are tested against real data and not against rows made up to pass.
"""
import httpx
import pytest

from prometheus.tools import symbols
from prometheus.tools.symbols import Resolution, SymbolLookupError, is_plain_ticker, resolve_company


def row(symbol, description, type_="Common Stock"):
    return {"symbol": symbol, "displaySymbol": symbol, "type": type_, "description": description}


# real answers from Finnhub's /search
PG_NAME = [row("PG", "Procter & Gamble Co"), row("PGHH.NS", "Procter & Gamble Hygiene and Health Care Ltd"), row("PGHL.NS", "Procter & Gamble Health Ltd"),
           row("GJR", "Strats (SM) Trust for Procter & Gamble Securities Series 2006-1")]
PG_TICKER = [row("PG", "PROCTER & GAMBLE CO/THE"), row("PG.TO", "PROCTER & GAMBLE CO/THE-CDR", "Canadian DR"), row("PG.NE", "PROCTER & GAMBLE CO/THE-CDR", "Canadian DR"),
             row("PG.BK", "PEOPLE'S GARMENT PUB CO LTD"), row("PG.MT", "PG PLC"), row("PG.MX", "PROCTER & GAMBLE CO/THE"), row("PG.VI", "PROCTER & GAMBLE CO/THE"),
             row("PCG", "PG&E Corp")]
META = [row("META", "META PLATFORMS INC-CLASS A"), row("META.TO", "META PLATFORMS INC-CDR", "Canadian DR"), row("META.NE", "META PLATFORMS INC-CDR", "Canadian DR"),
        row("META.ST", "METACON AB"), row("META.L", "WT INDUS METALS ENHANCED", "ETP"), row("META.BK", "META CORP PCL"), row("META.JK", "NUSANTARA INFRASTRUCTURE TBK")]
MSFT = [row("MSFT", "Microsoft Corp")]


class FakeSearch:
    """The lookup, recording what it was asked and answering from a table of canned answers."""

    def __init__(self, **answers):
        self.answers, self.asked = answers, []

    def __call__(self, query):
        self.asked.append(query)
        return {k.lower(): v for k, v in self.answers.items()}.get(query.lower(), [])  # Finnhub's search ignores case, so the fake does too


# ---------------------------------------------------------------- what needs no lookup

@pytest.mark.parametrize("text, plain", [("MSFT", True), ("$AAPL", True), (" PG ", True), ("BRK.B", True), ("BF-B", True), ("A", True),
                                         ("msft", False), ("microsoft", False), ("Procter&Gamble", False), ("PG&E", False), ("ABCDEFGHIJK", False), ("", False)])
def test_only_a_ticker_in_capitals_skips_the_lookup(text, plain):
    assert is_plain_ticker(text) is plain


# ---------------------------------------------------------------- the lookup

def test_the_company_name_from_the_request_resolves_to_pg_and_not_to_its_look_alikes():
    """The name as typed, "Procter&Gamble": Finnhub returns nothing for it, so the "&" is spaced; then PG is the only row whose whole name matches."""
    search = FakeSearch(**{"Procter & Gamble": PG_NAME})
    result = resolve_company("Procter&Gamble", search)
    assert result == Resolution(typed="Procter&Gamble", symbol="PG", company="Procter & Gamble Co", how="name")
    assert search.asked == ["Procter & Gamble"]  # never the unspaced form that returns nothing


@pytest.mark.parametrize("typed", ["Procter & Gamble", "procter  &  gamble", "PROCTER&GAMBLE", "  Procter &Gamble ", "Procter& Gamble"])
def test_the_spacing_and_case_of_the_name_do_not_matter(typed):
    assert resolve_company(typed, FakeSearch(**{"Procter & Gamble": PG_NAME})).symbol == "PG"


def test_the_subsidiaries_and_the_trust_are_not_chosen_even_though_their_names_contain_the_name():
    for wrong in ("PGHH.NS", "PGHL.NS", "GJR"):
        assert resolve_company("Procter & Gamble", FakeSearch(**{"Procter & Gamble": PG_NAME})).symbol != wrong


def test_a_typed_ticker_in_lower_case_is_recognised_as_a_symbol_and_not_looked_up_as_a_name():
    result = resolve_company("pg", FakeSearch(pg=PG_TICKER))
    assert (result.symbol, result.how, result.company) == ("PG", "ticker", "Procter & Gamble Co/The")  # not PG.MX or PCG ("PG&E")
    assert resolve_company("$pg", FakeSearch(pg=PG_TICKER)).symbol == "PG"


def test_meta_is_the_ticker_even_though_a_dozen_other_things_start_with_it():
    result = resolve_company("meta", FakeSearch(meta=META))
    assert (result.symbol, result.how) == ("META", "ticker")


def test_a_name_with_one_company_behind_it_resolves():
    assert resolve_company("microsoft", FakeSearch(microsoft=MSFT)).symbol == "MSFT"
    assert resolve_company("Microsoft Corp", FakeSearch(**{"Microsoft Corp": MSFT})).symbol == "MSFT"


def test_a_name_that_is_only_the_start_of_one_companys_name_resolves_when_it_is_the_only_one():
    rows = [row("BRK.B", "BERKSHIRE HATHAWAY INC-CL B")]
    result = resolve_company("Berkshire Hathaway", FakeSearch(**{"Berkshire Hathaway": rows}))
    assert result.symbol == "BRK.B"  # a dotted symbol is accepted when there is no undotted one


def test_a_us_symbol_is_preferred_to_the_same_company_on_a_foreign_exchange():
    rows = [row("ACME.L", "ACME CORP"), row("ACME", "ACME CORP"), row("ACME.TO", "ACME CORP")]
    assert resolve_company("Acme", FakeSearch(Acme=rows)).symbol == "ACME"


@pytest.mark.parametrize("typed, answers, listed", [
    ("alphabet", {"alphabet": [row("GOOGL", "ALPHABET INC-CL A"), row("GOOG", "ALPHABET INC-CL C")]}, ("GOOGL", "GOOG")),
    ("Acme", {"Acme": [row("ACMA", "ACME CORP"), row("ACMB", "ACME INC")]}, ("ACMA", "ACMB")),               # two companies with the same name: never a guess
    ("meta materials", {"meta materials": [row("MMAT", "META MATERIALS INC"), row("MMTX", "META MATERIALS INC")]}, ("MMAT", "MMTX")),
])
def test_an_ambiguous_name_is_refused_with_the_candidates_listed_and_never_resolved_by_picking_one(typed, answers, listed):
    with pytest.raises(SymbolLookupError) as caught:
        resolve_company(typed, FakeSearch(**answers))
    message = str(caught.value)
    assert "matches several companies" in message and "Type the ticker symbol instead" in message and all(s in message for s in listed)


def test_a_name_that_matches_nothing_is_refused_and_so_is_a_name_that_is_only_foreign_receipts_and_funds():
    with pytest.raises(SymbolLookupError, match="no company found for 'xyzzy corp'; type the ticker symbol instead"):
        resolve_company("xyzzy corp", FakeSearch())
    only_receipts = [row("META.TO", "META PLATFORMS INC-CDR", "Canadian DR"), row("META.L", "WT INDUS METALS ENHANCED", "ETP")]
    with pytest.raises(SymbolLookupError, match="no company found"):
        resolve_company("meta platforms", FakeSearch(**{"meta platforms": only_receipts}))


def test_no_more_than_five_candidates_are_listed():
    rows = [row(f"AB{c}", f"ABC {c} HOLDINGS") for c in "DEFGHIJ"]
    with pytest.raises(SymbolLookupError) as caught:
        resolve_company("abc", FakeSearch(abc=rows))
    assert str(caught.value).count("(Abc ") == 5


def test_a_word_that_is_a_tickers_symbol_is_taken_as_that_ticker_before_it_is_taken_as_a_name():
    """The rule the guide states: what was typed IS a symbol of a row, so "Acme" with a row ACME is ACME, with no ambiguity about the two companies named Acme."""
    result = resolve_company("Acme", FakeSearch(Acme=[row("ACME", "ACME CORP"), row("ACMX", "ACME INC")]))
    assert (result.symbol, result.how) == ("ACME", "ticker")


def test_the_default_lookup_is_found_when_the_function_is_called_not_when_it_is_defined(monkeypatch):
    """A default argument would bind Finnhub's search at import time, and a replacement (or a test) would then still reach the network."""
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch(microsoft=MSFT))
    assert resolve_company("microsoft").symbol == "MSFT"


def test_the_same_symbol_returned_twice_counts_once():
    assert resolve_company("microsoft", FakeSearch(microsoft=MSFT + MSFT)).symbol == "MSFT"


@pytest.mark.parametrize("text, message", [("", "the company name is empty"), ("   ", "the company name is empty"), ("$", "the company name is empty"),
                                           ("x" * 61, "longer than 60 characters")])
def test_an_empty_or_over_long_name_is_refused_before_any_lookup(text, message):
    search = FakeSearch()
    with pytest.raises(SymbolLookupError, match=message):
        resolve_company(text, search)
    assert search.asked == []


@pytest.mark.parametrize("text", ["../etc/passwd", "https://x.test/a", "AAPL; rm -rf", "a|b", "name\nwith newline", "x<y>", "$(whoami)", "{evil}"])
def test_text_that_cannot_be_a_ticker_or_a_name_is_refused_before_any_lookup(text):
    search = FakeSearch()
    with pytest.raises(SymbolLookupError, match="is not a valid ticker symbol or company name"):
        resolve_company(text, search)
    assert search.asked == []


@pytest.mark.parametrize("text", ["Procter & Gamble", "Procter&Gamble", "Dillard's", "O'Reilly Automotive", "Berkshire Hathaway Inc.", "Coca-Cola", "AT&T", "BRK.B", "Dillard\u2019s"])
def test_ordinary_company_names_pass_the_character_check(text):
    search = FakeSearch()
    with pytest.raises(SymbolLookupError, match="no company found"):  # nothing in the fake, but the text was accepted and searched
        resolve_company(text, search)
    assert len(search.asked) == 1


def test_a_lookup_that_returns_an_unusable_symbol_is_refused():
    with pytest.raises(SymbolLookupError, match="not a usable ticker symbol"):
        resolve_company("Acme", FakeSearch(Acme=[row("../ETC", "ACME CORP")]))


def test_odd_rows_from_the_lookup_are_skipped():
    rows = [None, "text", {"type": "Common Stock"}, {"symbol": "", "type": "Common Stock"}, row("ACME", "ACME CORP")]
    assert resolve_company("Acme", FakeSearch(Acme=rows)).symbol == "ACME"


# ---------------------------------------------------------------- the real lookup, with the network faked

def fake_get(monkeypatch, response=None, error=None):
    def get(url, params=None, timeout=None):
        assert url == symbols.SEARCH_URL and params["q"] and timeout
        if error:
            raise error
        return response
    monkeypatch.setattr(symbols.httpx, "get", get)


def ok(payload):
    return httpx.Response(200, json=payload, request=httpx.Request("GET", symbols.SEARCH_URL))


def test_the_lookup_sends_the_key_and_returns_the_rows(monkeypatch):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "secret-key-123")
    seen = {}
    monkeypatch.setattr(symbols.httpx, "get", lambda url, params=None, timeout=None: seen.update(params=params) or ok({"count": 1, "result": MSFT}))
    assert symbols.search_finnhub("microsoft") == MSFT and seen["params"] == {"q": "microsoft", "token": "secret-key-123"}


def test_without_a_key_the_lookup_says_how_to_go_on_and_names_the_setting_not_a_value(monkeypatch):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "")
    with pytest.raises(SymbolLookupError, match="needs FINNHUB_API_KEY in .env .or type the ticker in capitals"):
        symbols.search_finnhub("microsoft")


@pytest.mark.parametrize("status", [401, 429, 500])
def test_an_http_error_is_reported_with_its_code_and_without_the_key(monkeypatch, status):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "secret-key-123")
    fake_get(monkeypatch, response=httpx.Response(status, request=httpx.Request("GET", symbols.SEARCH_URL + "?token=secret-key-123")))
    with pytest.raises(SymbolLookupError) as caught:
        symbols.search_finnhub("microsoft")
    assert f"HTTP {status}" in str(caught.value) and "secret-key-123" not in str(caught.value) and "type the ticker in capitals" in str(caught.value)


def test_a_network_failure_is_reported_by_kind_and_without_the_key_or_the_url(monkeypatch):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "secret-key-123")
    fake_get(monkeypatch, error=httpx.ConnectError("could not reach https://finnhub.io/api/v1/search?token=secret-key-123"))
    with pytest.raises(SymbolLookupError) as caught:
        symbols.search_finnhub("microsoft")
    assert "ConnectError" in str(caught.value) and "secret-key-123" not in str(caught.value) and "finnhub.io" not in str(caught.value)


@pytest.mark.parametrize("payload", [{"count": 0}, {"result": None}, {"result": "oops"}, {"result": {"a": 1}}])
def test_an_odd_answer_is_no_rows_and_not_a_crash(monkeypatch, payload):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "k")
    fake_get(monkeypatch, response=ok(payload))
    assert symbols.search_finnhub("x") == []


def test_an_answer_that_is_not_json_is_reported(monkeypatch):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "k")
    fake_get(monkeypatch, response=httpx.Response(200, text="<html>", request=httpx.Request("GET", symbols.SEARCH_URL)))
    with pytest.raises(SymbolLookupError, match="the symbol lookup failed"):
        symbols.search_finnhub("x")


# ---------------------------------------------------------------- the command line

def test_the_command_line_prints_the_ticker_the_company_and_how_it_matched(monkeypatch, capsys):
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch(**{"Procter & Gamble": PG_NAME}))
    assert symbols.main(["Procter&Gamble"]) == 0
    assert capsys.readouterr().out.strip() == "PG: Procter & Gamble Co (typed 'Procter&Gamble', matched as a name)"


def test_a_plain_ticker_on_the_command_line_makes_no_lookup(monkeypatch, capsys):
    monkeypatch.setattr(symbols, "search_finnhub", lambda query: pytest.fail("no lookup for a ticker in capitals"))
    assert symbols.main(["$MSFT"]) == 0 and "MSFT: a ticker in capitals, used as it is (no lookup)" in capsys.readouterr().out


def test_an_ambiguous_name_on_the_command_line_exits_with_1_and_says_why(monkeypatch, capsys):
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch(alphabet=[row("GOOGL", "ALPHABET INC-CL A"), row("GOOG", "ALPHABET INC-CL C")]))
    assert symbols.main(["alphabet"]) == 1
    assert "matches several companies" in capsys.readouterr().err
