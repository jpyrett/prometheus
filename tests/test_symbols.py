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


def test_a_symbol_is_found_when_a_row_has_that_symbol():
    search = FakeSearch(PG=PG_TICKER)
    found = find_symbol("PG", search)
    assert (found.symbol, found.how, found.typed) == ("PG", "ticker", "PG") and found.company == "Procter & Gamble Co/The" and search.asked == ["PG"]


@pytest.mark.parametrize("typed", ["pg", "$PG", " pg ", "$pg", "Pg"])
def test_the_case_a_dollar_sign_and_spaces_of_a_symbol_do_not_matter(typed):
    assert find_symbol(typed, FakeSearch(PG=PG_TICKER)).symbol == "PG"


@pytest.mark.parametrize("typed, rows", [("PGX", PG_TICKER), ("P", PG_TICKER), ("PG.TO", PG_TICKER), ("MET", META)])
def test_a_symbol_that_only_looks_like_one_is_not_found_no_prefix_no_similar_no_depositary_receipt(typed, rows):
    """Exact only: 'PGX' and 'P' are not 'PG', 'MET' is not 'META', and PG.TO is a Canadian depositary receipt (not common stock) in the real answer."""
    assert find_symbol(typed, FakeSearch(**{typed: rows})) is None


def test_a_common_stock_on_a_foreign_exchange_is_a_symbol_when_that_is_what_was_typed():
    assert find_symbol("PG.MX", FakeSearch(**{"PG.MX": PG_TICKER})).symbol == "PG.MX"


@pytest.mark.parametrize("bad", ["", "   ", "../../.env", "A B", "TOOLONGSYMBOL1", "AAPL;ls", None, 5, ["AAPL"]])
def test_text_that_is_not_a_symbol_is_not_found_and_nothing_is_looked_up(bad):
    search = FakeSearch()
    assert find_symbol(bad, search) is None and search.asked == []


# ---------------------------------------------------------------- the company name

def test_a_company_name_is_found_when_a_row_has_exactly_that_name():
    search = FakeSearch(**{PG_CO: PG_NAME})
    found = find_company(PG_CO, search)
    assert (found.symbol, found.company, found.how) == ("PG", PG_CO, "name") and search.asked == [PG_CO]


@pytest.mark.parametrize("typed", [PG_CO.upper(), PG_CO.lower(), "  Procter   &  Gamble   Co  "])
def test_the_case_and_the_extra_spaces_of_a_name_do_not_matter(typed):
    assert find_company(typed, FakeSearch(**{" ".join(typed.split()): PG_NAME})).symbol == "PG"


@pytest.mark.parametrize("typed", ["Procter", "Gamble Co", "P&G", "Gamble & Procter Co", "Procter Gamble Hygiene"])
def test_there_is_no_fuzzy_matching_a_name_must_have_the_same_core_words_in_the_same_order(typed):
    """Not found, although a person might know what was meant: a part of the name, an abbreviation, the words in another order. Tier 2 forgives only the
    generic company words (Inc, Corp, Co, Company, The, and), never a part of the name (the typo-tolerant TODO in the code)."""
    assert find_company(typed, FakeSearch(**{typed: PG_NAME})) is None


@pytest.mark.parametrize("typed", ["Procter&Gamble Co", "Procter & Gamble Co.", "procter, gamble & co", "PROCTER-&-GAMBLE CO"])
def test_only_punctuation_case_and_spacing_may_differ_from_the_data_sources_name(typed):
    """Found by the first real run: Finnhub's names carry full stops ('Apple Inc.'), which nobody types. The words and their order must still be the same."""
    search = AnyQuery(PG_NAME)  # Finnhub's search is tolerant of how a query is spelled; the comparison afterwards is what is strict
    assert find_company(typed, search).symbol == "PG" and "&" in search.asked[0]
    assert find_company("Procter&Gamble Co", AnyQuery(PG_NAME)).symbol == "PG"


def test_the_ampersand_is_spaced_in_the_query_because_finnhub_finds_nothing_for_the_unspaced_form():
    search = AnyQuery(PG_NAME)
    find_company("Procter&Gamble Co", search)
    assert search.asked == ["Procter & Gamble Co"]  # entry 31's finding; a query, not a match rule


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


@pytest.mark.parametrize("typed", ["Apple Inc", "Apple Inc.", "apple inc", "APPLE  INC", " Apple, Inc "])
def test_apple_inc_is_found_by_the_name_without_its_full_stop_and_not_mistaken_for_pineapple(typed):
    assert find_company(typed, AnyQuery(REAL_APPLE_INC)).symbol == "AAPL"


def test_apple_finds_apple_inc_among_eight_real_rows_but_a_part_of_a_longer_name_does_not():
    """Tier 2: 'Apple' and 'Apple Inc.' have the same core words. The other seven real rows ('Apple Hospitality REIT Inc', 'Apple iSports Group Inc', ...) have
    more words, so only AAPL matches and nothing is guessed."""
    assert find_company("Apple", FakeSearch(Apple=REAL_APPLE)).symbol == "AAPL"
    assert find_company("Apple Hospitality", FakeSearch(**{"Apple Hospitality": REAL_APPLE})) is None  # a part of 'Apple Hospitality REIT Inc': not its core
    assert find_company("Pineapple Inc", FakeSearch(**{"Pineapple Inc": REAL_APPLE_INC})).symbol == "PNPL"


def test_the_real_full_names_are_found_and_so_are_the_forms_people_type_without_the_generic_words():
    assert find_company("Microsoft Corporation", AnyQuery(REAL_MSFT)).symbol == "MSFT"
    assert find_company("The Procter & Gamble Company", AnyQuery(REAL_PG)).symbol == "PG"
    for typed in ("Microsoft Corp", "Microsoft", "microsoft inc"):          # tier 2: Corp, Inc, nothing: all the same company words
        assert find_company(typed, AnyQuery(REAL_MSFT)).symbol == "MSFT", typed
    for typed in ("Procter & Gamble Company", "Procter & Gamble", "Procter and Gamble", "Procter&Gamble Co"):   # 'The', 'Co' and '&' vs 'and' do not matter
        assert find_company(typed, AnyQuery(REAL_PG)).symbol == "PG", typed


# ---------------------------------------------------------------- how the search is asked (found by the first real run)

@pytest.mark.parametrize("typed, queries", [
    ("Apple Inc", ["Apple Inc", "Apple"]),
    ("Microsoft Corporation", ["Microsoft"]),  # 21 characters: Finnhub answers HTTP 422 "q too long" above 20
    ("The Procter & Gamble Company", ["The Procter & Gamble", "The Procter &", "The Procter"]),
    ("Procter&Gamble Co", ["Procter & Gamble Co", "Procter & Gamble", "Procter &"]),  # the "&" is spaced: Finnhub finds nothing for the unspaced form
    ("Meta Platforms Inc", ["Meta Platforms Inc", "Meta Platforms", "Meta"]),
    ("International Business Machines Corporation", ["International"]),
    ("Supercalifragilisticexpialidocious Inc", ["Supercalifragilistic"]),  # one word longer than the limit: cut
])
def test_the_search_is_asked_with_the_leading_words_that_fit_in_twenty_characters_longest_first(typed, queries):
    assert symbols._name_queries(typed) == queries
    assert all(len(q) <= 20 for q in queries) and len(queries) <= 3


def test_a_name_whose_first_search_finds_nothing_is_asked_again_with_fewer_words_and_still_matched_exactly():
    """Real: 'Meta Platforms Inc' (no punctuation) finds 0 rows at Finnhub, 'Meta Platforms' finds 'Meta Platforms, Inc.'. The match is still the same words."""
    answers = {"Meta Platforms Inc": [], "Meta Platforms": [row("META", "Meta Platforms, Inc.")]}
    asked = []

    def search(query):
        asked.append(query)
        return answers.get(query, [])

    assert find_company("Meta Platforms Inc", search).symbol == "META" and asked == ["Meta Platforms Inc", "Meta Platforms"]  # stopped as soon as one is found


def test_a_name_that_is_not_there_costs_at_most_three_searches():
    search = FakeSearch()
    assert find_company("Acme Global Widget Works Ltd", search) is None
    assert search.asked == ["Acme Global Widget", "Acme Global", "Acme"]  # five prefixes would fit the idea; three is the cap, longest first


def test_a_longer_search_that_finds_the_row_does_not_ask_the_shorter_ones():
    search = AnyQuery(REAL_APPLE_INC)
    assert find_company("Apple Inc", search).symbol == "AAPL" and search.asked == ["Apple Inc"]


def test_the_subsidiaries_and_the_trust_with_longer_names_are_not_the_company():
    only_the_others = [r for r in PG_NAME if r["symbol"] != "PG"]
    assert find_company(PG_CO, FakeSearch(**{PG_CO: only_the_others})) is None


def test_a_us_listing_is_preferred_to_foreign_listings_of_the_same_name():
    assert find_company("Procter & Gamble Co/The", AnyQuery(PG_TICKER)).symbol == "PG"  # PG, not PG.MX or PG.VI


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

def test_when_both_are_given_and_the_symbol_is_found_the_symbol_is_used_and_the_name_is_never_looked_up():
    search = FakeSearch(META=META, **{PG_CO: PG_NAME})
    found = resolve_target("META", PG_CO, search)  # a company name that is another company's: the symbol wins
    assert (found.symbol, found.how) == ("META", "ticker") and search.asked == ["META"]


def test_when_the_symbol_is_not_found_the_company_name_is_used():
    search = FakeSearch(**{PG_CO: PG_NAME})
    found = resolve_target("ZZZZ", PG_CO, search)
    assert (found.symbol, found.how) == ("PG", "name") and search.asked == ["ZZZZ", PG_CO]  # the symbol first, then the name


def test_the_company_name_alone_is_looked_up_without_a_symbol_lookup():
    search = FakeSearch(**{PG_CO: PG_NAME})
    assert resolve_target(None, PG_CO, search).symbol == "PG" and resolve_target("  ", PG_CO, search).symbol == "PG" and search.asked == [PG_CO, PG_CO]


def test_a_symbol_alone_that_is_not_found_says_so_with_the_symbol_in_the_message():
    with pytest.raises(TargetNotFound) as caught:
        resolve_target("ZZZZ", None, FakeSearch())
    assert str(caught.value) == "Symbol: ZZZZ is not found" and caught.value.detail == ""
    with pytest.raises(TargetNotFound) as again:
        resolve_target("$zzzz", "", FakeSearch())  # a blank company name is no company name
    assert str(again.value) == "Symbol: zzzz is not found"  # as typed, without the dollar sign


def test_a_company_name_that_is_not_found_says_company_name_not_found():
    with pytest.raises(TargetNotFound) as caught:
        resolve_target(None, "Nonexistent Holdings", FakeSearch())
    assert str(caught.value) == "Company Name not found."


def test_with_both_given_and_neither_found_the_last_step_tried_is_the_message():
    """The symbol was tried and failed, then the name was tried and failed: the message is the name's. (The spec: 'If the company name is not found: return Company Name not found.')"""
    search = FakeSearch()
    with pytest.raises(TargetNotFound) as caught:
        resolve_target("ZZZZ", "Nonexistent Holdings", search)
    assert str(caught.value) == "Company Name not found." and search.asked == ["ZZZZ", "Nonexistent Holdings", "Nonexistent"]  # the symbol, then the name (and its shorter search)


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


def test_a_lookup_that_fails_is_a_lookup_error_and_never_not_found():
    def broken(query):
        raise SymbolLookupError("the symbol lookup failed (HTTP 429); type the ticker in capitals to skip it", reason="the symbol lookup failed (HTTP 429)")

    with pytest.raises(SymbolLookupError) as caught:
        resolve_target("META", "Meta Platforms Inc", broken)
    assert not isinstance(caught.value, TargetNotFound) and caught.value.reason == "the symbol lookup failed (HTTP 429)"


def test_the_default_search_is_found_when_the_function_is_called(monkeypatch):
    monkeypatch.setattr(symbols, "search_finnhub", FakeSearch(META=META))
    assert resolve_target("META", None).symbol == "META"


# ---------------------------------------------------------------- the reason of a lookup error (no advice that only fits the desk)

def test_the_reason_is_the_message_unless_a_shorter_one_is_given():
    assert SymbolLookupError("only this").reason == "only this" and SymbolLookupError("long; advice", reason="long").reason == "long"
    assert str(SymbolLookupError("long; advice", reason="long")) == "long; advice"


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

@pytest.mark.parametrize("typed, rows", [
    ("Proctor & Gamble", REAL_PG), ("The Proctor & Gamble Company", REAL_PG), ("The Procter & Gambel Company", REAL_PG),
    ("Microsft Corporation", REAL_MSFT), ("Microsoft Corporaton", REAL_MSFT),
    ("Aple Inc", REAL_APPLE_INC), ("Apple Ink", REAL_APPLE_INC), ("Appel Inc", REAL_APPLE_INC),
])
def test_a_spelling_mistake_is_not_matched_even_when_the_directory_returns_the_right_row(typed, rows):
    assert find_company(typed, AnyQuery(rows)) is None


def test_the_correct_spelling_of_the_same_names_is_matched():
    assert find_company("The Procter & Gamble Company", AnyQuery(REAL_PG)).symbol == "PG"
    assert find_company("Microsoft Corporation", AnyQuery(REAL_MSFT)).symbol == "MSFT"
    assert find_company("Apple Inc", AnyQuery(REAL_APPLE_INC)).symbol == "AAPL"


def test_pepsi_is_not_pepsico_and_a_missing_or_extra_word_is_a_different_name():
    """Jacob's second example, at the level of the comparison (no directory row needed): different words are a different name."""
    assert symbols._words("Pepsi") != symbols._words("Pepsico")
    assert symbols._words("Pepsico") == symbols._words("PEPSICO")        # case is not a difference
    assert symbols._words("PepsiCo") != symbols._words("PepsiCo Inc")    # a missing 'Inc' is: the company-suffix gap is part of the fuzzy-matching TODO
    assert symbols._words("PepsiCo, Inc.") == symbols._words("pepsico inc")  # punctuation is not


def test_a_typo_in_the_company_name_is_company_name_not_found_whatever_the_symbol_did():
    with pytest.raises(TargetNotFound) as caught:
        resolve_target("ZZZZ", "Proctor & Gamble", AnyQuery(REAL_PG))  # the symbol fails, then the misspelt name fails
    assert str(caught.value) == COMPANY_NOT_FOUND
    assert resolve_target("PG", "Proctor & Gamble", AnyQuery(REAL_PG)).symbol == "PG"  # a good symbol wins: the typo is never looked at
    with pytest.raises(TargetNotFound) as only_name:
        resolve_target(None, "Proctor & Gamble", AnyQuery(REAL_PG))
    assert str(only_name.value) == "Company Name not found."


# ================================================================ the flexible name search: tier 2 (generic company words ignored), still no typos
#
# Tier 1 (the same words) always wins. Tier 2 runs only if tier 1 found nothing, and ignores only the generic company words (Inc, Corp, Corporation, Co,
# Company, Ltd, Limited, plc, Incorporated, The, and). The rows below are the real ones recorded from Finnhub; the logic tests with other rows say so.

from prometheus.tools.symbols import _core  # noqa: E402


def test_the_core_of_a_name_leaves_out_only_the_generic_company_words():
    assert _core("PepsiCo Inc") == _core("Pepsico") == "pepsico"          # Jacob's example: Pepsico finds PepsiCo Inc
    assert _core("Microsoft Corp") == _core("Microsoft Corporation") == _core("MICROSOFT") == "microsoft"
    assert _core("Procter and Gamble") == _core("The Procter & Gamble Company") == _core("PROCTER & GAMBLE CO/THE") == "procter gamble"
    assert _core("Apple") == _core("Apple Inc.") == "apple"
    assert _core("Meta Platforms, Inc.") == "meta platforms"               # a word that says what the company is stays
    assert _core("The Company") == "" and _core("") == "" and _core(None) == ""   # nothing left: a name made of generic words matches nothing
    assert _core("The Procter and Gamble Co Ltd Inc") == "procter gamble"  # the generic words are ignored wherever they stand and however many there are; the other words keep their order


@pytest.mark.parametrize("right, wrong", [("Pepsico", "Pepsi"), ("Procter & Gamble", "Proctor & Gamble"), ("Microsoft", "Microsft"), ("Apple Inc", "Aple Inc"),
                                         ("Procter and Gamble", "Procter and Gambel"), ("Meta Platforms", "Meta Platfroms")])
def test_the_core_still_tells_a_typo_from_the_name_so_no_typo_is_forgiven(right, wrong):
    assert _core(right) != _core(wrong) and _words(right) != _words(wrong)


from prometheus.tools.symbols import _words  # noqa: E402


@pytest.mark.parametrize("typed", ["Pepsi", "Proctor & Gamble", "Microsft Corp", "Aple", "Procter Gambl", "Procter & Gamble Cmpany Ltd"])
def test_with_the_flexible_search_a_typo_is_still_not_found_even_when_the_directory_returns_the_right_row(typed):
    for rows in (REAL_PG, REAL_MSFT, REAL_APPLE_INC):
        assert find_company(typed, AnyQuery(rows)) is None


def test_tier_one_always_wins_so_nothing_that_matched_before_changes():
    """Logic test with two rows made for the purpose: both have the core 'foo'. 'Foo Inc' has the same words as the first, so it is found, and tier 2 (which would
    find both and refuse) never runs."""
    rows = [row("FOO", "Foo Inc"), row("FOOB", "Foo Corp Ltd")]
    assert find_company("Foo Inc", AnyQuery(rows)).symbol == "FOO"
    with pytest.raises(TargetNotFound) as caught:                        # 'Foo' has tier 2 only: both rows share the core, so it is refused, not guessed
        find_company("Foo", AnyQuery(rows))
    assert str(caught.value) == COMPANY_NOT_FOUND and "FOO" in caught.value.detail and "FOOB" in caught.value.detail


def test_every_name_found_before_is_found_the_same_way_now():
    for typed, rows, symbol in [("Apple Inc", REAL_APPLE_INC, "AAPL"), ("Apple Inc.", REAL_APPLE_INC, "AAPL"), ("Microsoft Corporation", REAL_MSFT, "MSFT"),
                                ("The Procter & Gamble Company", REAL_PG, "PG"), ("Pineapple Inc", REAL_APPLE_INC, "PNPL")]:
        found = find_company(typed, AnyQuery(rows))
        assert found.symbol == symbol and found.how == "name"


def test_a_name_that_several_listings_share_in_tier_two_is_refused_with_the_real_candidates():
    """Real rows: PG.MX and PG.VI are both 'PROCTER & GAMBLE CO/THE' on foreign exchanges. 'Procter & Gamble' reaches them only through tier 2."""
    with pytest.raises(TargetNotFound) as caught:
        find_company("Procter & Gamble", AnyQuery(ABROAD))
    assert str(caught.value) == COMPANY_NOT_FOUND and "PG.MX" in caught.value.detail and "PG.VI" in caught.value.detail and "Give the symbol" in caught.value.detail


def test_a_us_listing_is_still_preferred_in_tier_two():
    assert find_company("Procter & Gamble", AnyQuery(PG_TICKER)).symbol == "PG"  # PG, PG.MX and PG.VI share the core name: the undotted one wins, as in tier 1


def test_a_name_of_only_generic_words_matches_nothing():
    for typed in ("The Company", "Inc", "Corp and Co"):
        assert find_company(typed, AnyQuery(REAL_PG + REAL_MSFT + REAL_APPLE_INC)) is None


def test_the_symbol_still_wins_and_the_flexible_name_is_only_the_fallback():
    search = AnyQuery(REAL_MSFT)
    assert resolve_target("MSFT", "Apple", search).symbol == "MSFT"                              # the symbol was found: the name is never looked at
    assert resolve_target("ZZZZ", "Microsoft Corp", AnyQuery(REAL_MSFT)).symbol == "MSFT"        # the symbol failed: the flexible name finds it
    with pytest.raises(TargetNotFound) as caught:
        resolve_target("ZZZZ", "Microsft Corp", AnyQuery(REAL_MSFT))                              # the typo fails in both tiers
    assert str(caught.value) == COMPANY_NOT_FOUND


def test_a_name_with_no_core_words_never_matches_a_row_that_also_has_none():
    """Logic test, one row made for the purpose: 'The Company Inc' is all generic words, so its core is empty. Without the guard an empty core would equal
    the empty core of 'The Company' and match a company nobody named."""
    rows = [row("TCI", "The Company Inc")]
    assert _core("The Company Inc") == "" and _core("The Company") == ""
    assert find_company("The Company", AnyQuery(rows)) is None
    assert find_company("The Company Inc", AnyQuery(rows)).symbol == "TCI"  # tier 1 still finds the exact words
