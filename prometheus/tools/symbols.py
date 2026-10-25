"""Company name -> ticker symbol, by lookup and never by guess (design choices entry 31).

    python -m prometheus.tools.symbols "Procter & Gamble"        # PG  (Procter & Gamble Co)
    python -m prometheus.tools.symbols meta                      # META: a typed ticker
    python -m prometheus.tools.symbols "Procter&Gamble"          # PG: the missing spaces are put back

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
    """The name could not be turned into one ticker. The message is for the user and holds no key."""


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
        raise SymbolLookupError("the symbol lookup needs FINNHUB_API_KEY in .env (or type the ticker in capitals, which needs no lookup)")
    try:
        response = httpx.get(SEARCH_URL, params={"q": query, "token": key}, timeout=20)
        response.raise_for_status()
        rows = response.json().get("result", [])
    except httpx.HTTPStatusError as error:
        raise SymbolLookupError(f"the symbol lookup failed (HTTP {error.response.status_code}); type the ticker in capitals to skip it") from None
    except (httpx.HTTPError, ValueError) as error:
        raise SymbolLookupError(f"the symbol lookup failed ({type(error).__name__}); type the ticker in capitals to skip it") from None
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn a company name into a ticker symbol, by lookup.")
    parser.add_argument("name", help='a company name such as "Procter & Gamble", or a ticker')
    args = parser.parse_args(argv)
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
