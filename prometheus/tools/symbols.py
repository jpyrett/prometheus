"""Company name -> ticker symbol, by lookup and never by guess (design choices entry 31).

    python -m prometheus.tools.symbols "Procter & Gamble"        # PG  (Procter & Gamble Co)
    python -m prometheus.tools.symbols meta                      # META: a typed ticker
    python -m prometheus.tools.symbols "Procter&Gamble"          # PG: the missing spaces are put back
    python -m prometheus.tools.symbols AAPL --exact --company "Apple Inc"   # the press release rules (entry 34, below): symbol first, exact only

The desk's `symbol` parameter takes a ticker or a company name. A ticker written in capitals (MSFT, BRK.B, $AAPL) is used as it is, with no
lookup. Anything else (a name, or a ticker in lower case) is looked up in Finnhub's symbol search, with the key that is already in .env. It is a
LOOKUP on purpose: a model asked for "Procter & Gamble's ticker" usually says PG, and it was seen to say it, but a model that is wrong once
fetches the wrong company's news with a paid API and writes the file. Here the answer comes from a data source, is shown to the user with the
company's name, and an ambiguous name is refused instead of resolved.

How a lookup decides (the rows are what Finnhub returned; only `Common Stock` rows count, so depositary receipts and ETPs are ignored):
  1. what was typed IS a symbol of a row ("meta" -> META): that row
  2. otherwise the typed name equals the name of exactly one row, ignoring case, punctuation, "&" and company suffixes (Co, Inc, Corp, Ltd, plc,
     The): that row. "Procter & Gamble" -> PG, and not its Indian subsidiaries (PGHH.NS) or the Strats trust (GJR) whose names are longer
  3. otherwise the typed name is the start of the names of exactly one row: that row
  4. otherwise it is ambiguous (or unknown): an error that lists up to five candidates and asks for the ticker. It never picks one.
A symbol without a dot (a US listing) is preferred to one with a dot (a foreign exchange) at every step, except when no row has none (BRK.B).

Finnhub's search returns nothing for "Procter&Gamble" and finds "Procter & Gamble", so the "&" is spaced before the search.
"""
import argparse
import re
import sys
from dataclasses import dataclass

import httpx

from prometheus.rag.config import setting
from prometheus.rag.loaders import check_symbol

SEARCH_URL = "https://finnhub.io/api/v1/search"
TYPES = ("Common Stock",)
SUFFIXES = {"co", "corp", "corporation", "inc", "incorporated", "ltd", "limited", "plc", "the", "company"}
MAX_TEXT = 60
SHOWN = 5


class SymbolLookupError(ValueError):
    """The name could not be turned into one ticker. The message is for the user and holds no key. `reason` is the same message without the
    advice that only fits the desk's symbol field ("type the ticker in capitals to skip it"), for a caller that has no such way out."""

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason or message


@dataclass(frozen=True)
class Resolution:
    typed: str     # what the user typed
    symbol: str    # the ticker
    company: str   # the company's name as the data source gives it
    how: str       # "ticker" (what was typed is a symbol) or "name"


def is_plain_ticker(text: str) -> bool:
    """A ticker written in capitals, with an optional leading $: it needs no lookup. 'msft' and 'microsoft' are not plain."""
    return re.fullmatch(r"\$?[A-Z0-9][A-Z0-9.\-]{0,9}", text.strip()) is not None


def _norm(text: str) -> str:
    """Lower case, punctuation and '&' gone, company suffixes dropped: 'Procter & Gamble Co' and 'PROCTER & GAMBLE CO/THE' both become 'procter gamble'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(w for w in words if w not in SUFFIXES)


def _display(description: str) -> str:
    return description.title() if description.isupper() else description


def search_finnhub(query: str) -> list[dict]:
    """Finnhub's symbol search: a list of {symbol, type, description, ...}. Raises SymbolLookupError, never with the key in the message."""
    key = setting("FINNHUB_API_KEY", "")
    if not key:
        raise SymbolLookupError("the symbol lookup needs FINNHUB_API_KEY in .env (or type the ticker in capitals, which needs no lookup)",
                                reason="the symbol lookup needs FINNHUB_API_KEY in .env")
    try:
        response = httpx.get(SEARCH_URL, params={"q": query, "token": key}, timeout=20)
        response.raise_for_status()
        rows = response.json().get("result", [])
    except httpx.HTTPStatusError as error:
        what = f"the symbol lookup failed (HTTP {error.response.status_code})"
        raise SymbolLookupError(f"{what}; type the ticker in capitals to skip it", reason=what) from None
    except (httpx.HTTPError, ValueError) as error:
        what = f"the symbol lookup failed ({type(error).__name__})"
        raise SymbolLookupError(f"{what}; type the ticker in capitals to skip it", reason=what) from None
    return rows if isinstance(rows, list) else []


def _prefer_us(rows: list[dict]) -> list[dict]:
    undotted = [r for r in rows if "." not in r["symbol"]]
    return undotted or rows


def _ambiguous(text: str, rows: list[dict]) -> SymbolLookupError:
    if not rows:
        return SymbolLookupError(f"no company found for {text!r}; type the ticker symbol instead")
    shown = ", ".join(f"{r['symbol']} ({_display(r.get('description', ''))})" for r in rows[:SHOWN])
    return SymbolLookupError(f"{text!r} matches several companies: {shown}. Type the ticker symbol instead")


def resolve_company(text: str, search=None) -> Resolution:
    """The ticker for a company name or a ticker typed in any case. `search` is the lookup (a fake in tests; Finnhub's when left out, looked up at the
    time of the call so that it can be replaced). Raises SymbolLookupError."""
    search = search or search_finnhub
    typed = text.strip().lstrip("$").strip() if isinstance(text, str) else ""
    if not typed:
        raise SymbolLookupError("the company name is empty")
    if len(typed) > MAX_TEXT:
        raise SymbolLookupError(f"the company name is longer than {MAX_TEXT} characters")
    if not re.fullmatch(r"[A-Za-z0-9 &.,'\u2019\-]+", typed):  # letters, digits, spaces and & . , ' -: nothing that is a path, a url or a command
        raise SymbolLookupError(f"{typed[:20]!r} is not a valid ticker symbol or company name")
    query = re.sub(r"\s+", " ", re.sub(r"\s*&\s*", " & ", typed)).strip()  # Finnhub finds "Procter & Gamble" and nothing for "Procter&Gamble"
    rows = [r for r in search(query) if isinstance(r, dict) and r.get("type") in TYPES and r.get("symbol")]
    rows = list({r["symbol"]: r for r in rows}.values())

    def found(row: dict, how: str) -> Resolution:
        try:
            symbol = check_symbol(row["symbol"])
        except ValueError:
            raise SymbolLookupError(f"the lookup returned {row['symbol']!r}, which is not a usable ticker symbol") from None
        return Resolution(typed=typed, symbol=symbol, company=_display(row.get("description", "")), how=how)

    as_symbol = re.sub(r"[^A-Za-z0-9.\-]", "", typed).upper() if " " not in typed else ""
    for row in rows:
        if as_symbol and row["symbol"].upper() == as_symbol:
            return found(row, "ticker")
    wanted = _norm(typed)
    exact = _prefer_us([r for r in rows if _norm(r.get("description", "")) == wanted]) if wanted else []
    if len(exact) == 1:
        return found(exact[0], "name")
    if len(exact) > 1:
        raise _ambiguous(typed, exact)
    starts = _prefer_us([r for r in rows if wanted and _norm(r.get("description", "")).startswith(wanted + " ")])
    if len(starts) == 1:
        return found(starts[0], "name")
    raise _ambiguous(typed, starts or _prefer_us(rows))


# ---------------------------------------------------------------- exact lookup for a press release request (design choices entry 34)
#
# A press release work order carries two arguments, a symbol and a company name (either, or both). The rules, in code and not in a model:
#   1. a symbol is given: look it up. Found: use it (even if a company name is also given).
#   2. the symbol is not found, and a company name is given: look the name up. Found: use its symbol.
#   3. nothing found: the message below. A symbol with no company name gets SYMBOL_NOT_FOUND; a company name that was tried gets COMPANY_NOT_FOUND.
# "Found" means Finnhub's symbol directory (the same search as above) has a row that matches what was typed. A symbol matches a row's symbol (case ignored).
# A company name matches in TWO TIERS, and the first tier that finds a row wins:
#   tier 1  the SAME WORDS as the row's name (case, spacing and punctuation ignored: 'Apple Inc' is 'Apple Inc.').
#   tier 2  only if tier 1 finds nothing: the same words once the generic company words are left out of BOTH names: Inc, Corp, Corporation, Co, Company, Ltd,
#           Limited, plc, Incorporated, a leading 'The', and 'and' (`_core`). So 'Pepsico' is 'PepsiCo Inc', 'Microsoft Corp' is 'Microsoft Corporation',
#           'Apple' is 'Apple Inc.' and 'Procter and Gamble' is 'The Procter & Gamble Company'. A tier 2 match must be UNIQUE (US listings preferred): if
#           several rows share the core name it is refused with the candidates, never guessed. Nothing that tier 1 finds can change because of tier 2.
# NO spelling tolerance in either tier. Jacob's definition (2026-10-29): a typo is NOT matched, for example "Proctor & Gamble" (spelled wrong) or "Pepsi" for
# "Pepsico". There is no similarity score anywhere, so those stay "not found" even when the directory returns the right row (tests/test_symbols.py).
# A lookup that FAILS (HTTP 429, no key) is a SymbolLookupError, never "not found".
#
# TODO: typo-tolerant (fuzzy) company-name matching, if it is ever wanted: a similarity measure and a rule for when it is unsure. It would go in find_company
# below and nowhere else. (`resolve_company` above has its own looser matching for the desk's news symbol field.)

SYMBOL_NOT_FOUND = "Symbol: {symbol} is not found"
COMPANY_NOT_FOUND = "Company Name not found."
NAME_CHARS = re.compile(r"[A-Za-z0-9 &.,'\u2019\-/()]+")  # what a company name may contain. The "/" and the brackets are in real names ("PROCTER & GAMBLE CO/THE"),
                                                          # so an exact match needs them; ; | $ ` < > and a newline stay out. A name is only ever sent as a query value and compared


class TargetNotFound(SymbolLookupError):
    """The symbol or the company name is not in the data source. str(error) is exactly the message for the work order (SYMBOL_NOT_FOUND or
    COMPANY_NOT_FOUND); `detail` says more when there is more to say (for example that a name matches several listings), else it is empty."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def _clean(text) -> str:
    """Whitespace collapsed and stripped; anything that is not text is empty."""
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


def _echo(text: str) -> str:
    """What the user typed, as it may appear in a message: only the characters a symbol or a name may hold, and not too long."""
    return re.sub(r"[^A-Za-z0-9 &.,'\u2019\-$]", "?", text)[:MAX_TEXT]


def _words(text) -> str:
    """TIER 1. The words of a name, in order, lower case, with the punctuation and the spacing gone: 'Apple Inc.' and 'APPLE  INC' are the same, 'Apple' and
    'Apple Inc.' are not. Found by the first real run: Finnhub's names are 'Apple Inc.', 'Microsoft Corporation', 'The Procter & Gamble Company', and nobody
    types the full stop. Nothing is dropped here and nothing is guessed."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold())) if isinstance(text, str) else ""


GENERIC_WORDS = SUFFIXES | {"and"}   # the words that do not say which company it is (the desk's SUFFIXES, and 'and': 'Procter and Gamble' for 'Procter & Gamble')


def _core(text) -> str:
    """TIER 2. `_words` without the generic company words (GENERIC_WORDS: Inc, Corp, Corporation, Co, Company, Ltd, Limited, plc, Incorporated, The, and):
    'PepsiCo Inc' and 'Pepsico' are both 'pepsico'. It never forgives a spelling mistake ('pepsi' is not 'pepsico', 'proctor gamble' is not 'procter gamble')
    and it is only used when tier 1 found nothing. A name made only of generic words has no core (the empty text), which matches nothing."""
    return " ".join(w for w in _words(text).split() if w not in GENERIC_WORDS)


QUERY_MAX = 20     # Finnhub's search answers HTTP 422 {"error":"q too long"} for a longer query (found by the first real run: 'Microsoft Corporation' is 21)
QUERY_TRIES = 3    # at most this many searches for one name


def _name_queries(typed: str) -> list[str]:
    """The searches to ask Finnhub for a typed name, longest first: the leading words that fit in QUERY_MAX characters, then fewer. This is how the
    search is ASKED and never how a name is MATCHED (that is `_words`, exact). Why not just the name: the search takes at most 20 characters, and it is
    sensitive to punctuation ('Meta Platforms Inc' finds nothing, 'Meta Platforms' finds 'Meta Platforms, Inc.'). The '&' is spaced because Finnhub finds
    'Procter & Gamble' and nothing for 'Procter&Gamble' (entry 31)."""
    words = re.sub(r"\s*&\s*", " & ", typed).split()
    queries = [" ".join(words[:n]) for n in range(len(words), 0, -1) if len(" ".join(words[:n])) <= QUERY_MAX]
    return list(dict.fromkeys(queries or [words[0][:QUERY_MAX]]))[:QUERY_TRIES]


def _rows(rows) -> list[dict]:
    return [r for r in rows if isinstance(r, dict) and r.get("type") in TYPES and r.get("symbol")]


def find_symbol(symbol, search=None) -> Resolution | None:
    """The row whose symbol EQUALS `symbol` (case and a leading '$' ignored), or None. Not a similar symbol, not a prefix: 'AAP' does not find AAPL.
    Raises SymbolLookupError when the lookup itself fails."""
    search = search or search_finnhub
    typed = _clean(symbol).lstrip("$").strip()
    try:
        wanted = check_symbol(typed)
    except ValueError:  # empty, too long, a space or a path: not a symbol, so nothing to look up
        return None
    for row in _rows(search(wanted)):
        if row["symbol"].upper() == wanted:
            return Resolution(typed=typed, symbol=wanted, company=_display(row.get("description", "")), how="ticker")
    return None


def find_company(name, search=None) -> Resolution | None:
    """The row whose name matches `name`, or None: TIER 1, the same words (`_words`); only if no row has them, TIER 2, the same words without the generic
    company words (`_core`). Never a similar spelling: see the comment above. In either tier US listings are preferred to foreign ones, and a name that still
    matches several rows is not guessed: it is a TargetNotFound whose detail lists them. Raises SymbolLookupError when the lookup itself fails."""
    search = search or search_finnhub
    typed = _clean(name)
    if not typed or len(typed) > MAX_TEXT or not NAME_CHARS.fullmatch(typed):
        return None
    wanted, core = _words(typed), _core(typed)
    same: list[dict] = []
    for query in _name_queries(typed):  # usually the first one finds it; a shorter one is asked only when the longer one found no row
        rows = _rows(search(query))
        by_symbol = lambda match: list({r["symbol"]: r for r in rows if match(r.get("description"))}.values())
        same = _prefer_us(by_symbol(lambda d: _words(d) == wanted))                               # tier 1
        if not same and core:                                                                      # tier 2, only when tier 1 found nothing
            same = _prefer_us(by_symbol(lambda d: _core(d) == core))
        if same:
            break
    if len(same) > 1:
        shown = ", ".join(f"{r['symbol']} ({_display(r.get('description', ''))})" for r in same[:SHOWN])
        raise TargetNotFound(COMPANY_NOT_FOUND, detail=f"{_echo(typed)!r} matches several listings: {shown}. Give the symbol")
    if not same:
        return None
    return Resolution(typed=typed, symbol=check_symbol(same[0]["symbol"]), company=_display(same[0].get("description", "")), how="name")


def resolve_target(symbol, company, search=None) -> Resolution:
    """The symbol and company a press release request is about: the symbol if it is found, else the company name's symbol (rules above).
    Raises TargetNotFound with the work order's message, SymbolLookupError if the lookup fails, ValueError if neither argument is given."""
    typed_symbol, typed_company = _clean(symbol).lstrip("$").strip(), _clean(company)
    if not typed_symbol and not typed_company:
        raise ValueError("give a symbol, a company name, or both")
    if typed_symbol:
        found = find_symbol(typed_symbol, search)
        if found:
            return found  # the symbol wins, even if a company name was given too
        if not typed_company:
            raise TargetNotFound(SYMBOL_NOT_FOUND.format(symbol=_echo(typed_symbol)))
    found = find_company(typed_company, search)
    if found:
        return found
    raise TargetNotFound(COMPANY_NOT_FOUND)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn a company name into a ticker symbol, by lookup.")
    parser.add_argument("name", nargs="?", default="", help='a company name such as "Procter & Gamble", or a ticker')
    parser.add_argument("--exact", action="store_true",
                        help="the press release rules instead: NAME is the SYMBOL argument and --company the company name; the symbol first, "
                             "matched on the words (never a typo), and the messages the work order carries")
    parser.add_argument("--company", default="", help="with --exact: the company name, used when the symbol is not found")
    args = parser.parse_args(argv)
    if args.exact:
        try:
            r = resolve_target(args.name, args.company)
        except TargetNotFound as error:
            print(f"{error}" + (f"  ({error.detail})" if error.detail else ""))
            return 1
        except (SymbolLookupError, ValueError) as error:  # the reason, without the desk's "type the ticker in capitals": that is no way out here
            print(f"ERROR: {getattr(error, 'reason', error)}", file=sys.stderr)
            return 2
        print(f"{r.symbol}: {r.company} (found by its {'symbol' if r.how == 'ticker' else 'company name'}, typed {r.typed!r})")
        return 0
    if not args.name:
        parser.error("give a name or a ticker (or use --exact with a symbol and/or --company)")
    try:
        if is_plain_ticker(args.name):
            print(f"{args.name.strip().lstrip('$')}: a ticker in capitals, used as it is (no lookup)")
            return 0
        r = resolve_company(args.name)
    except SymbolLookupError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"{r.symbol}: {r.company} (typed {r.typed!r}, matched as a {r.how})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
