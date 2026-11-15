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


# ---------------------------------------------------------------- the lookup


def test_meta_is_the_ticker_even_though_a_dozen_other_things_start_with_it():
    result = resolve_company("meta", FakeSearch(meta=META))
    assert (result.symbol, result.how) == ("META", "ticker")


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


# ================================================================ the exact lookup for a press release request (design choices entry 34)
#
# The rules: the symbol first; if it is not found, the company name; exact matches only (no fuzzy matching: that is a TODO); the work order's messages are
# "Symbol: X is not found" and "Company Name not found."; a lookup that FAILS is not "not found". The rows are the real ones above.

from prometheus.tools.symbols import COMPANY_NOT_FOUND, SYMBOL_NOT_FOUND, TargetNotFound, find_company, find_symbol, resolve_target  # noqa: E402

PG_CO = "Procter & Gamble Co"  # the name as Finnhub's name search describes PG (the PG_NAME rows)
ABROAD = [r for r in PG_TICKER if r["symbol"] in ("PG.MX", "PG.VI")]  # two real rows of one company's name, both common stock, both on foreign exchanges


# ---------------------------------------------------------------- the symbol

def test_the_spec_messages_are_exactly_these_words():
    assert SYMBOL_NOT_FOUND.format(symbol="ZZZZ") == "Symbol: ZZZZ is not found" and COMPANY_NOT_FOUND == "Company Name not found."


@pytest.mark.parametrize("bad", ["", "   ", "../../.env", "A B", "TOOLONGSYMBOL1", "AAPL;ls", None, 5, ["AAPL"]])
def test_text_that_is_not_a_symbol_is_not_found_and_nothing_is_looked_up(bad):
    search = FakeSearch()
    assert find_symbol(bad, search) is None and search.asked == []


# ---------------------------------------------------------------- the company name


class AnyQuery:
    """A lookup that answers the same rows to any query, and records the queries."""

    def __init__(self, rows):
        self.rows, self.asked = rows, []

    def __call__(self, query):
        self.asked.append(query)
        return self.rows


# The rows Finnhub returned in the first real run (2026-10-29), recorded as they came:
REAL_APPLE_INC = [row("AAPL", "Apple Inc."), row("PNPL", "Pineapple Inc")]  # q="Apple Inc"
REAL_APPLE = [row("AAPL", "Apple Inc."), row("APLE", "Apple Hospitality REIT Inc"), row("603020.SS", "Apple Flavor & Fragrance Group Co Ltd"), row("MLP", "Maui Land & Pineapple Company Inc"),
              row("AAPI", "Apple iSports Group Inc"), row("2788.T", "Apple International Co Ltd"), row("PAPL", "Pineapple Financial Inc"), row("164A.T", "Applepark Co Ltd")]  # q="Apple"
REAL_MSFT = [row("MSFT", "Microsoft Corporation")]  # q="Microsoft Corp" and q="Microsoft Corporation"
REAL_PG = [row("PG", "The Procter & Gamble Company")]  # q="Procter & Gamble Co"


# ---------------------------------------------------------------- how the search is asked (found by the first real run)


def test_a_name_that_still_matches_several_listings_is_never_guessed_and_the_detail_says_which():
    with pytest.raises(TargetNotFound) as caught:
        find_company("Procter & Gamble Co/The", AnyQuery(ABROAD))
    assert str(caught.value) == COMPANY_NOT_FOUND  # the work order's message is the same as for an unknown name
    assert "PG.MX" in caught.value.detail and "PG.VI" in caught.value.detail and "Give the symbol" in caught.value.detail


@pytest.mark.parametrize("bad", ["", "   ", None, 5, "Apple; rm -rf /", "a|b", "$(whoami)", "x" * 61, "tab\x00nul", "`id`"])
def test_text_that_cannot_be_a_company_name_is_not_found_and_nothing_is_looked_up(bad):
    search = FakeSearch()
    assert find_company(bad, search) is None and search.asked == []


# ---------------------------------------------------------------- the rules together


def test_giving_neither_is_a_value_error_and_looks_nothing_up():
    search = FakeSearch()
    for symbol, company in ((None, None), ("", ""), ("  ", " "), ("$", None), ("$ ", "  ")):  # a lone "$" is nothing once the dollar sign is dropped
        with pytest.raises(ValueError, match="give a symbol, a company name, or both"):
            resolve_target(symbol, company, search)
    assert search.asked == []


def test_a_symbol_that_is_text_a_person_could_not_have_meant_is_echoed_safely_and_nothing_is_looked_up():
    search = FakeSearch()
    with pytest.raises(TargetNotFound) as caught:
        resolve_target("../../etc/passwd\x1b[31m", None, search)
    assert str(caught.value) == "Symbol: ..?..?etc?passwd??31m is not found" and search.asked == []  # no slash, no escape character and no bracket in what is printed


# ---------------------------------------------------------------- the reason of a lookup error (no advice that only fits the desk)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_the_real_lookup_error_has_a_reason_without_the_advice_that_only_fits_the_desk(monkeypatch, status):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "secret-key-123")
    fake_get(monkeypatch, response=httpx.Response(status, request=httpx.Request("GET", symbols.SEARCH_URL)))
    with pytest.raises(SymbolLookupError) as caught:
        symbols.search_finnhub("microsoft")
    assert caught.value.reason == f"the symbol lookup failed (HTTP {status})" and "type the ticker" not in caught.value.reason
    assert "type the ticker in capitals" in str(caught.value)  # what the desk's symbol field shows is unchanged


def test_a_missing_key_and_a_network_failure_have_reasons_too(monkeypatch):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "")
    with pytest.raises(SymbolLookupError) as no_key:
        symbols.search_finnhub("x")
    assert no_key.value.reason == "the symbol lookup needs FINNHUB_API_KEY in .env"
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "k")
    fake_get(monkeypatch, error=httpx.ConnectError("boom"))
    with pytest.raises(SymbolLookupError) as network:
        symbols.search_finnhub("x")
    assert network.value.reason == "the symbol lookup failed (ConnectError)"


# ---------------------------------------------------------------- the command line

def test_the_exact_command_line_says_how_it_was_found(monkeypatch, capsys):
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch(META=META, **{PG_CO: PG_NAME}))
    assert symbols.main(["META", "--exact", "--company", PG_CO]) == 0
    assert capsys.readouterr().out.strip() == "META: Meta Platforms Inc-Class A (found by its symbol, typed 'META')"
    assert symbols.main(["ZZZZ", "--exact", "--company", PG_CO]) == 0
    assert "PG: Procter & Gamble Co (found by its company name, typed 'Procter & Gamble Co')" in capsys.readouterr().out


def test_the_exact_command_line_prints_the_spec_message_and_exits_1_when_nothing_is_found(monkeypatch, capsys):
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch())
    assert symbols.main(["ZZZZ", "--exact"]) == 1 and capsys.readouterr().out.strip() == "Symbol: ZZZZ is not found"
    assert symbols.main(["--exact", "--company", "Nope Inc"]) == 1 and capsys.readouterr().out.strip() == "Company Name not found."


def test_the_exact_command_line_exits_2_for_a_failed_lookup_or_no_argument(monkeypatch, capsys):
    def broken(query):
        raise SymbolLookupError("x; advice", reason="the symbol lookup failed (HTTP 429)")

    monkeypatch.setattr(symbols, "search_finnhub", broken)
    assert symbols.main(["META", "--exact"]) == 2 and "ERROR: the symbol lookup failed (HTTP 429)" in capsys.readouterr().err
    assert symbols.main(["--exact"]) == 2 and "give a symbol, a company name, or both" in capsys.readouterr().err


def test_the_plain_command_line_still_needs_a_name():
    with pytest.raises(SystemExit) as caught:
        symbols.main([])
    assert caught.value.code == 2


# ---------------------------------------------------------------- "exact" means a misspelling is NOT matched (Jacob's definition, 2026-10-29)
#
# His examples: "Proctor & Gamble" (spelled wrong) and "Pepsi" instead of "Pepsico". Fuzzy matching is a TODO. The tests give the lookup the best case for a typo:
# the directory hands back the RIGHT row (AnyQuery answers the same rows to any query), and the comparison still has to refuse it.


# ================================================================ the flexible name search: tier 2 (generic company words ignored), still no typos
#
# Tier 1 (the same words) always wins. Tier 2 runs only if tier 1 found nothing, and ignores only the generic company words (Inc, Corp, Corporation, Co,
# Company, Ltd, Limited, plc, Incorporated, The, and). The rows below are the real ones recorded from Finnhub; the logic tests with other rows say so.

from prometheus.tools.symbols import _core  # noqa: E402


from prometheus.tools.symbols import _words  # noqa: E402


def test_tier_one_always_wins_so_nothing_that_matched_before_changes():
    """Logic test with two rows made for the purpose: both have the core 'foo'. 'Foo Inc' has the same words as the first, so it is found, and tier 2 (which would
    find both and refuse) never runs."""
    rows = [row("FOO", "Foo Inc"), row("FOOB", "Foo Corp Ltd")]
    assert find_company("Foo Inc", AnyQuery(rows)).symbol == "FOO"
    with pytest.raises(TargetNotFound) as caught:                        # 'Foo' has tier 2 only: both rows share the core, so it is refused, not guessed
        find_company("Foo", AnyQuery(rows))
    assert str(caught.value) == COMPANY_NOT_FOUND and "FOO" in caught.value.detail and "FOOB" in caught.value.detail


def test_a_name_that_several_listings_share_in_tier_two_is_refused_with_the_real_candidates():
    """Real rows: PG.MX and PG.VI are both 'PROCTER & GAMBLE CO/THE' on foreign exchanges. 'Procter & Gamble' reaches them only through tier 2."""
    with pytest.raises(TargetNotFound) as caught:
        find_company("Procter & Gamble", AnyQuery(ABROAD))
    assert str(caught.value) == COMPANY_NOT_FOUND and "PG.MX" in caught.value.detail and "PG.VI" in caught.value.detail and "Give the symbol" in caught.value.detail


