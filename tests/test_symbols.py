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


# ---------------------------------------------------------------- the lookup


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


@pytest.mark.parametrize("payload", [{"count": 0}, {"result": None}, {"result": "oops"}, {"result": {"a": 1}}])
def test_an_odd_answer_is_no_rows_and_not_a_crash(monkeypatch, payload):
    monkeypatch.setattr(symbols, "setting", lambda name, default="": "k")
    fake_get(monkeypatch, response=ok(payload))
    assert symbols.search_finnhub("x") == []


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


# ---------------------------------------------------------------- the symbol

def test_the_spec_messages_are_exactly_these_words():
    assert SYMBOL_NOT_FOUND.format(symbol="ZZZZ") == "Symbol: ZZZZ is not found" and COMPANY_NOT_FOUND == "Company Name not found."


# ---------------------------------------------------------------- the company name


class AnyQuery:
    """A lookup that answers the same rows to any query, and records the queries."""

    def __init__(self, rows):
        self.rows, self.asked = rows, []

    def __call__(self, query):
        self.asked.append(query)
        return self.rows


# The rows Finnhub returned in the first real run (2026-10-29), recorded as they came:


# ---------------------------------------------------------------- how the search is asked (found by the first real run)


@pytest.mark.parametrize("bad", ["", "   ", None, 5, "Apple; rm -rf /", "a|b", "$(whoami)", "x" * 61, "tab\x00nul", "`id`"])
def test_text_that_cannot_be_a_company_name_is_not_found_and_nothing_is_looked_up(bad):
    search = FakeSearch()
    assert find_company(bad, search) is None and search.asked == []


# ---------------------------------------------------------------- the rules together


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
