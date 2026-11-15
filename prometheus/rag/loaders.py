"""Load the DataToolbox's news files into LangChain Documents (the "Documents" box of the slide).

The DataToolbox writes news/<SYMBOL>.txt as a series of records (format: docs/DataToolbox.md):

    <<<ARTICLE>>>
    id: finnhub:143136906                one "name: value" line per header field
    published_utc: 2026-10-23T19:03:00Z
    ...
    headline: The iPhone Duo Launch Kicks Off a Crucial Few Weeks for Apple
    <<<SUMMARY>>>
    the summary text
    <<<INSIGHTS>>>
    AAPL | positive | the reasoning      (Polygon articles only; may be empty)
    <<<END>>>

This module turns each record into one Document. It never imports the DataToolbox: the file format is
the contract between them. The one change it makes to the text is to decode HTML entities (`Procter &amp; Gamble` becomes
`Procter & Gamble`, `Nvidia&#39;s` becomes `Nvidia's`): see decode_entities and design choices entry 18.

A NEWS ARTICLE (or press release) THAT FAILS IS DISCARDED (Jacob, 2026-11-15, design choices entry 83): a record the DataToolbox would not have written (a header line with no colon, an unknown marker, no
<<<END>>>, bad UTF-8, a field that is missing, a time that is not a time, no ticker) is SKIPPED, completely: it becomes no Document, so it is never chunked, never stored in the RAG, never found by a search, and never
reaches the Advisor's scoring or any other result. The records before and after it still load. `load_news(path, skipped=[...])` lists each skipped record (`Skipped`: the file and line, and why) for a person to look
at; nothing else is told. SEC filings are not news and keep the strict rule: a filing the toolbox would not have written stops the load with a ValueError that names the file and the line.

Look at the result:   python -m prometheus.rag.loaders AAPL

The DataToolbox's press release files (<SYMBOL>_PRs.txt, a second record format) are loaded by `load_press_releases` in this module, into Documents
with the same metadata keys as the news ones (design choices entry 34):   python -m prometheus.rag.loaders AAPL --press-releases
The SEC filing files (<SYMBOL>.txt in the filings folder, a third record format) are loaded by `load_filings`: one Document per section of a filing that has
text, one short Document for a filing that has none (design choices entry 35):   python -m prometheus.rag.loaders AAPL --filings
"""
import argparse
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from langchain_core.documents import Document

from prometheus.rag.config import filings_dir, news_dir, press_releases_dir

ARTICLE_START, SUMMARY_START, INSIGHTS_START, ARTICLE_END = (
    "<<<ARTICLE>>>", "<<<SUMMARY>>>", "<<<INSIGHTS>>>", "<<<END>>>")
REQUIRED_FIELDS = ("id", "headline", "published_utc", "url")
# Press releases are a second, independent format (the DataToolbox's fetch_press_releases; design choices entries 4 and 34):
# <<<PRESS_RELEASE>>>, "name: value" header lines, <<<TEXT>>>, the text, <<<END>>>.
PRESS_RELEASE_START, PRESS_TEXT_START, PRESS_RELEASE_END = "<<<PRESS_RELEASE>>>", "<<<TEXT>>>", "<<<END>>>"
PRESS_REQUIRED_FIELDS = ("id", "symbol", "headline", "published_utc")  # no url: the toolbox writes an empty one when Finnhub gives none
PRESS_RELEASE_PUBLISHER = "Press release"  # the file has no publisher; this is what a citation shows ("[2] Press release, 2025-10-30, AAPL, ...")
PRESS_RELEASE_ID_PREFIX = "pr:"            # so a press release and a news article can never share a chunk id on one work order
PRESS_RELEASES_SUFFIX = "_PRs.txt"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")  # AAPL, BRK.B, BF-B
INSIGHT_LINE = re.compile(r"(?m)^[A-Z][A-Z0-9.\-]* \| (positive|neutral|negative) \| ")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")
ESCAPED_MARKUP = re.compile(r"&(lt|gt|#0*(60|62)|#x0*(3c|3e));", re.I)  # < and > in every spelling: what the loader never decodes


def _decode_one(match: re.Match) -> str:
    entity = match.group(0)
    decoded = html.unescape(entity)
    if decoded in ("<", ">"):
        return entity  # decoding these would write `<script>` into text that has none
    if decoded == entity or not decoded or any(unicodedata.category(c) in ("Cc", "Cs") or c == "\ufffd" for c in decoded):
        return entity  # not a known entity (&D;), or one that decodes to a control character or to nothing usable: leave it as written
    return " " if decoded == "\xa0" else decoded


def decode_entities(text: str) -> str:
    """Decode HTML entities that are fully formed (`&amp;`, `&#39;`, `&quot;`, `&nbsp;`, `&eacute;`, `&#x1F680;`), one match at a time, and nothing else.

    A bare `&copy` or `AT&T` or `R&D;` is left alone, and so is an entity that decodes to a control character. `<` and `>` are never written, in any
    spelling (`&lt;`, `&#60;`, `&#x3c;`), so that decoding never puts markup into text that has none (`&lt;script&gt;` stays as it is). The text is not scanned again after a decode:
    `&amp;lt;` becomes `&lt;`, the characters, and not `<`. Design choices entry 18."""
    return ENTITY.sub(_decode_one, text)


def check_symbol(symbol: str) -> str:
    """The symbol, stripped and upper-cased, if it is a safe ticker symbol; otherwise a ValueError.

    Letters, digits, '.' and '-' only, at most 10 characters. A symbol will come from an LLM's tool call and
    can end up in a file name or a search filter, so something like '../../.env' is refused."""
    clean = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(clean):
        raise ValueError(f"not a valid ticker symbol: {clean!r}")
    return clean


def news_file(symbol: str) -> Path:
    """The news file for one symbol: AAPL -> <news dir>/AAPL.txt (the symbol is checked first)."""
    return news_dir() / f"{check_symbol(symbol)}.txt"


def press_release_file(symbol: str) -> Path:
    """The press release file for one symbol: AAPL -> <press releases dir>/AAPL_PRs.txt (the symbol is checked first)."""
    return press_releases_dir() / f"{check_symbol(symbol)}{PRESS_RELEASES_SUFFIX}"


class Skipped(NamedTuple):
    """A record that failed and was discarded (design choices entry 83): `where` is "<file>:<first line>", `reason` says why in one sentence."""
    where: str
    reason: str


def load_news(path: Path | str, skipped: list | None = None) -> list[Document]:
    """Read one news file. Returns one Document per article, in file order. A record that fails is SKIPPED and ignored completely (see the module text): the others still load. `skipped`, when given, receives one
    `Skipped` for each record that was discarded."""
    path = Path(path)
    docs = []
    for record in _records(path):
        if isinstance(record, Skipped):
            if skipped is not None:
                skipped.append(record)
            continue
        where = f"{path}:{record[0]}"
        try:
            docs.append(_to_document(path, *record))
        except (ValueError, OverflowError) as e:   # a field that is missing, a time that is not a time, no ticker: the article failed
            if skipped is not None:
                skipped.append(Skipped(where, str(e).removeprefix(f"{where}: ")))
    return docs


def _records(path: Path):
    """Yield, for each record, (first line number, header fields, summary text, insights text), or a `Skipped` when the record is broken: a header line without a colon, an unknown marker, bad UTF-8, a record that
    starts before the last one ended or has no <<<END>>>. A broken record is read to its end (or to the next <<<ARTICLE>>>) and discarded, so the next record is read as it should be. Text between records is
    reported once per run of lines and ignored. A record that ended and is followed by text that is not a record (an unescaped <<<END>>> inside its text, say) was cut short, so it is discarded too: a record is
    given only when the next thing in the file is another record, a blank line or the end of the file."""
    state = start = None  # state: None between records, else "head", "summary" or "insights"
    head: dict[str, str] = {}
    summary: list[str] = []
    insights: list[str] = []
    broken = ""           # why the record in progress failed; "" while it is fine
    junk = False          # a run of text between records was already reported
    pending = None        # the record that just ended: given when the next thing is a record, a blank line or the end of the file
    with open(path, "rb") as f:  # read bytes and decode line by line, so a failure can name its line
        for number, raw in enumerate(f, 1):
            try:  # line 1 may start with a byte order mark (BOM); "utf-8-sig" removes it
                line = raw.decode("utf-8-sig" if number == 1 else "utf-8").rstrip("\r\n")
            except UnicodeDecodeError as e:
                if state is None:
                    if pending is not None:
                        yield Skipped(f"{path}:{pending[0]}", f"text follows its {ARTICLE_END} (line {number}: not valid UTF-8), so the record was cut short")
                        pending, junk = None, True
                    elif not junk:
                        yield Skipped(f"{path}:{number}", f"text outside a record: not valid UTF-8 ({e.reason})")
                        junk = True
                else:
                    broken = broken or f"line {number}: not valid UTF-8 ({e.reason}, at byte {e.start} of the line)"
                continue
            if line == ARTICLE_START:
                if pending is not None:
                    yield pending
                    pending = None
                if state is not None:
                    yield Skipped(f"{path}:{start}", broken or f"the record has no {ARTICLE_END} (a new record starts at line {number})")
                state, start, head, summary, insights, broken, junk = "head", number, {}, [], [], "", False
            elif state is None:
                if line.strip():
                    if pending is not None:
                        yield Skipped(f"{path}:{pending[0]}", f"text follows its {ARTICLE_END} (line {number}: {line[:40]!r}), so the record was cut short")
                        pending, junk = None, True
                    elif not junk:
                        yield Skipped(f"{path}:{number}", f"text outside a record: {line[:40]!r}")
                        junk = True
            elif line == SUMMARY_START:
                state = "summary"
            elif line == INSIGHTS_START:
                state = "insights"
            elif line == ARTICLE_END:
                if broken:
                    yield Skipped(f"{path}:{start}", broken)
                else:
                    pending = (start, head, "\n".join(summary).strip(), "\n".join(insights).strip())
                state, junk = None, False
            elif line.startswith("<<<"):
                broken = broken or f"line {number}: unknown marker {line!r}"
            elif state == "head":
                name, colon, value = line.partition(":")  # split on the FIRST colon: headlines contain colons
                if not colon:
                    broken = broken or f"line {number}: header line without a colon: {line[:40]!r}"
                else:
                    head[name.strip()] = value.strip()
            elif state == "summary":
                summary.append(line)
            else:
                insights.append(line)
    if pending is not None:
        yield pending
    if state is not None:
        yield Skipped(f"{path}:{start}", broken or f"the record has no {ARTICLE_END}")


def _to_document(path: Path, start: int, head: dict[str, str], summary: str, insights: str) -> Document:
    """One record -> one Document. The text is the headline, summary and insight lines, with the HTML entities decoded (decode_entities)."""
    where = f"{path}:{start}"
    missing = [name for name in REQUIRED_FIELDS if not head.get(name)]
    if missing:
        raise ValueError(f"{where}: the record is missing {', '.join(missing)}")
    try:
        published = datetime.strptime(head["published_utc"], TIME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"{where}: bad published_utc {head['published_utc']!r} "
                         f"(expected like 2026-10-23T19:03:00Z)") from None
    tickers = [t.strip() for t in head.get("tickers", "").split(",") if t.strip()]
    if not tickers and head.get("symbol"):
        tickers = [head["symbol"]]  # fall back to the symbol the file was fetched for
    if not tickers:
        raise ValueError(f"{where}: the record has neither tickers nor a symbol")
    headline, summary, insights = decode_entities(head["headline"]), decode_entities(summary), decode_entities(insights)
    return Document(
        id=head["id"],
        page_content="\n\n".join(part for part in (headline, summary, insights) if part),
        metadata={
            "article_id": head["id"],
            "headline": headline,
            "tickers": tickers,
            "published_utc": head["published_utc"],
            "published_ts": int(published.timestamp()),  # a number: Chroma range filters take numbers only
            "publisher": head.get("publisher", ""),
            "api": head.get("api", ""),
            "url": head["url"],
        },
    )


def load_press_releases(path: Path | str, skipped: list | None = None) -> list[Document]:
    """Read one press release file (<SYMBOL>_PRs.txt). Returns one Document per release, in file order. A release that fails is SKIPPED and ignored completely, as a news article is (see the module text; `skipped`
    receives one `Skipped` for each).

    The Documents have the SAME metadata keys as the news Documents (article_id, headline, tickers, published_utc, published_ts, publisher, api, url),
    so that chunking, storing, searching and citing treat them alike. The text is the headline, a blank line and the release's text, with HTML entities decoded."""
    path = Path(path)
    docs = []
    for record in _press_release_records(path):
        if isinstance(record, Skipped):
            if skipped is not None:
                skipped.append(record)
            continue
        where = f"{path}:{record[0]}"
        try:
            docs.append(_press_release_to_document(path, *record))
        except (ValueError, OverflowError) as e:
            if skipped is not None:
                skipped.append(Skipped(where, str(e).removeprefix(f"{where}: ")))
    return docs


def _press_release_records(path: Path):
    """Yield (first line number, header fields, text) for each press release: `_text_records` with the press release marker."""
    yield from _text_records(path, PRESS_RELEASE_START, lenient=True)


def _text_records(path: Path, start_marker: str, lenient: bool = False):
    """Yield (first line number, header fields, text) for each record of a file whose records are `start_marker`, "name: value" header lines, <<<TEXT>>>, the text,
    <<<END>>> (press releases and filings). Same rules as `_records`: a record that starts before the last one ended, text outside a record, an unknown marker, a
    header line without a colon, bad UTF-8 and a missing <<<END>>> are all errors that name the line. Each format calls this with its own start marker, so a record
    of one format is an "unknown marker" in the other's file and one format cannot be read as the other.

    `lenient` False (filings): the first of those errors raises. `lenient` True (press releases, design choices entry 83): a broken record is read to its end (or to the next start marker), discarded and
    yielded as a `Skipped`, and the next record is read as it should be; text between records is reported once per run of lines and ignored, and a record that ended and is followed by such text was cut short
    and is discarded too (see `_records`)."""
    state = start = None  # state: None between records, else "head" or "text"
    head: dict[str, str] = {}
    text: list[str] = []
    broken = ""           # lenient: why the record in progress failed; "" while it is fine
    junk = False          # lenient: a run of text between records was already reported
    pending = None        # lenient: the record that just ended, given when the next thing is a record, a blank line or the end of the file

    def fail(number: int, reason: str) -> None:
        nonlocal broken
        if not lenient:
            raise ValueError(f"{path}:{number}: {reason}")
        broken = broken or f"line {number}: {reason}"

    with open(path, "rb") as f:
        for number, raw in enumerate(f, 1):
            try:
                line = raw.decode("utf-8-sig" if number == 1 else "utf-8").rstrip("\r\n")
            except UnicodeDecodeError as e:
                reason = f"not valid UTF-8 ({e.reason}, at byte {e.start} of the line)"
                if not lenient:
                    raise ValueError(f"{path}:{number}: {reason}") from None
                if state is None:
                    if pending is not None:
                        yield Skipped(f"{path}:{pending[0]}", f"text follows its {PRESS_RELEASE_END} (line {number}: {reason}), so the record was cut short")
                        pending, junk = None, True
                    elif not junk:
                        yield Skipped(f"{path}:{number}", f"text outside a record: {reason}")
                        junk = True
                else:
                    fail(number, reason)
                continue
            if line == start_marker:
                if pending is not None:
                    yield pending
                    pending = None
                if state is not None:
                    if not lenient:
                        raise ValueError(f"{path}:{number}: a new record starts before the one at line {start} ended (it has no {PRESS_RELEASE_END})")
                    yield Skipped(f"{path}:{start}", broken or f"the record has no {PRESS_RELEASE_END} (a new record starts at line {number})")
                state, start, head, text, broken, junk = "head", number, {}, [], "", False
            elif state is None:
                if line.strip():
                    if not lenient:
                        raise ValueError(f"{path}:{number}: text outside a record: {line[:40]!r}")
                    if pending is not None:
                        yield Skipped(f"{path}:{pending[0]}", f"text follows its {PRESS_RELEASE_END} (line {number}: {line[:40]!r}), so the record was cut short")
                        pending, junk = None, True
                    elif not junk:
                        yield Skipped(f"{path}:{number}", f"text outside a record: {line[:40]!r}")
                        junk = True
            elif line == PRESS_TEXT_START:
                state = "text"
            elif line == PRESS_RELEASE_END:
                if broken:
                    yield Skipped(f"{path}:{start}", broken)
                elif lenient:
                    pending = (start, head, "\n".join(text).strip())
                else:
                    yield start, head, "\n".join(text).strip()
                state, junk = None, False
            elif line.startswith("<<<"):
                fail(number, f"unknown marker {line!r}")
            elif state == "head":
                name, colon, value = line.partition(":")  # the FIRST colon: headlines and urls contain colons
                if not colon:
                    fail(number, f"header line without a colon: {line[:40]!r}")
                else:
                    head[name.strip()] = value.strip()
            else:
                text.append(line)
    if pending is not None:
        yield pending
    if state is not None:
        if not lenient:
            raise ValueError(f"{path}:{start}: the record has no {PRESS_RELEASE_END}")
        yield Skipped(f"{path}:{start}", broken or f"the record has no {PRESS_RELEASE_END}")


def _press_release_to_document(path: Path, start: int, head: dict[str, str], text: str) -> Document:
    where = f"{path}:{start}"
    missing = [name for name in PRESS_REQUIRED_FIELDS if not head.get(name)]
    if missing:
        raise ValueError(f"{where}: the record is missing {', '.join(missing)}")
    try:
        published = datetime.strptime(head["published_utc"], TIME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"{where}: bad published_utc {head['published_utc']!r} (expected like 2026-10-23T19:03:00Z)") from None
    try:
        symbol = check_symbol(head["symbol"])
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    headline, text = decode_entities(head["headline"]), decode_entities(text)
    article_id = PRESS_RELEASE_ID_PREFIX + head["id"]
    return Document(
        id=article_id,
        page_content="\n\n".join(part for part in (headline, text) if part),
        metadata={
            "article_id": article_id,
            "headline": headline,
            "tickers": [symbol],  # a press release is one company's own: the symbol it was fetched for
            "published_utc": head["published_utc"],
            "published_ts": int(published.timestamp()),
            "publisher": PRESS_RELEASE_PUBLISHER,
            "api": "finnhub",
            "url": head.get("url", ""),
        },
    )


# ---------------------------------------------------------------- filings (design choices entry 35)
# The DataToolbox's fetch_filings writes <SYMBOL>.txt: <<<FILING>>>, "name: value" header lines (accession, symbol, form, filed, period_end, url,
# text_status), <<<TEXT>>>, the text as "== section ==" blocks, <<<END>>>. A filing with text becomes one Document per SECTION; a filing without text (the API
# gives none for a 10-Q, a Form 144, a proxy statement ...) becomes one short Document that says so.
FILING_START = "<<<FILING>>>"
FILING_REQUIRED_FIELDS = ("accession", "symbol", "form", "filed", "url", "text_status")
FILING_PUBLISHER = "SEC filing"   # what a citation shows ("[2] SEC filing, 2025-10-30, AAPL, ...")
FILING_ID_PREFIX = "sec:"         # so a filing and a news article or press release can never share a chunk id on one work order
ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")  # the SEC's id of a filing; it goes into a chunk id, so only this shape is accepted
SECTION_LINE = re.compile(r"== ([A-Za-z0-9_]+) ==")
DATE_FORMAT = "%Y-%m-%d"
# What a section is called in a headline; "items" (an 8-K's whole text) needs no label. The 10-K's two sections come from the Massive API, the 10-Q's eleven Items from EDGAR
# (agents/data_clerk/edgar_text.py, design choices entry 36); `risk_factors` is both a 10-K section and Part II Item 1A of a 10-Q.
SECTION_LABELS = {
    "business": "Business", "risk_factors": "Risk Factors", "items": "",
    "financial_statements": "Financial Statements", "mdna": "Management's Discussion and Analysis", "market_risk": "Market Risk Disclosures",
    "controls": "Controls and Procedures", "legal_proceedings": "Legal Proceedings", "unregistered_sales": "Unregistered Sales of Equity Securities",
    "defaults": "Defaults Upon Senior Securities", "mine_safety": "Mine Safety Disclosures", "other_information": "Other Information", "exhibits": "Exhibits",
}
# The SEC's own names for the forms the toolbox fetches (data_toolbox.FILING_FORMS), in plain words, for a filing that has no text: semantic search needs
# words to match, and "quarterly report" has to be in the Document for a question about quarterly reports to find a 10-Q. "/A" is an amendment.
FORM_NAMES = {
    "10-K": "annual report", "10-Q": "quarterly report", "8-K": "current report of a material event", "DEF 14A": "proxy statement for a shareholder meeting",
    "PRE 14C": "preliminary information statement", "DEF 14C": "information statement",
    "SCHEDULE 13D": "report by an investor who holds more than 5% of the shares and may seek influence",
    "SCHEDULE 13G": "report by an investor who holds more than 5% of the shares, as a passive holder",
    "144": "notice of a planned sale of restricted shares by an insider (insider selling)",
}


def filing_file(symbol: str) -> Path:
    """The filing file for one symbol: AAPL -> <filings dir>/AAPL.txt (the symbol is checked first)."""
    return filings_dir() / f"{check_symbol(symbol)}.txt"


def form_name(form: str) -> str:
    """The plain-words name of a form ('10-Q' -> 'quarterly report', '144/A' -> 'notice ..., amended'), or '' for a form that is not in FORM_NAMES."""
    base, amended = (form[:-2], True) if form.endswith("/A") else (form, False)
    name = FORM_NAMES.get(base, "")
    return f"{name}, amended" if name and amended else name


def load_filings(path: Path | str) -> list[Document]:
    """Read one filing file (<SYMBOL>.txt). Returns the Documents in file order: one per section of a filing that has text, one per filing that has none.

    The Documents have the SAME metadata keys as the news Documents (article_id, headline, tickers, published_utc, published_ts, publisher, api, url).
    `filed` is a date and the file has no time of day, so published_utc is that date at 00:00:00Z. A section's text is its headline, a blank line and the
    section, with HTML entities decoded. Raises ValueError, naming the file and line, for anything the DataToolbox would not have written.

    One filing can be in the file twice: the Massive record with no text (a 10-Q), and later the record `fetch_filing_text` appended with the text taken from EDGAR (design choices
    entry 36). The file is append-only, so the rule is here: for an accession, **the last record that has text wins**; a record without text never replaces one with text; the
    Documents come in the order the accessions first appear."""
    path = Path(path)
    chosen: dict[str, tuple[bool, list[Document]]] = {}
    for record in _text_records(path, FILING_START):
        docs = _filing_to_documents(path, *record)  # validates the record, the accession included
        has_text = record[1].get("text_status") == "ok" and bool(record[2])
        known = chosen.get(record[1]["accession"])
        if known is None or has_text or not known[0]:
            chosen[record[1]["accession"]] = (has_text, docs)
    return [doc for _, docs in chosen.values() for doc in docs]


FILING_HEADLINE = re.compile(r"(?P<symbol>\S+) (?P<form>.+?)(?:, (?P<label>[^,]+))?, filed (?P<filed>\d{4}-\d{2}-\d{2})(?:, period ending (?P<period_end>\d{4}-\d{2}-\d{2}))?")


def filing_parts(doc: Document) -> dict:
    """What a filing Document says about itself, read back from its id and headline (the format this module wrote them in): {accession, section, symbol, form, label,
    filed, period_end, url, has_text}. `section` is None for the short Document of a filing with no text. Raises ValueError for a Document that is not a filing's.
    The loader owns the headline format, so the reader of it lives next to the writer, and a test ties the two (design choices entry 38)."""
    parts = str(doc.id).split(":")
    found = FILING_HEADLINE.fullmatch(doc.metadata.get("headline", ""))
    if len(parts) not in (2, 3) or parts[0] != "sec" or not found:
        raise ValueError(f"not a filing Document: {str(doc.id)[:60]!r}")
    section = parts[2] if len(parts) == 3 else None
    return {"accession": parts[1], "section": section, "symbol": found["symbol"], "form": found["form"], "label": found["label"] or "", "filed": found["filed"],
            "period_end": found["period_end"] or "", "url": doc.metadata.get("url", ""), "has_text": section is not None}


def _sections(where: str, text: str) -> list[tuple[str, str]]:
    """The "== name ==" blocks of a filing's text as (name, body) pairs, in order. A name that repeats gets _2, _3 (an 8-K can have two blocks of items),
    so that two sections never get one id and one is never silently lost. A block with no body is left out."""
    sections: list[tuple[str, list[str]]] = []
    times: Counter = Counter()
    for line in text.split("\n"):
        match = SECTION_LINE.fullmatch(line)
        if match:
            times[match.group(1)] += 1
            sections.append((match.group(1) if times[match.group(1)] == 1 else f"{match.group(1)}_{times[match.group(1)]}", []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{where}: the text does not start with a section line like '== items ==': {line[:40]!r}")
    return [(name, "\n".join(body).strip()) for name, body in sections if "\n".join(body).strip()]


def _filing_to_documents(path: Path, start: int, head: dict[str, str], text: str) -> list[Document]:
    where = f"{path}:{start}"
    missing = [name for name in FILING_REQUIRED_FIELDS if not head.get(name)]
    if missing:
        raise ValueError(f"{where}: the record is missing {', '.join(missing)}")
    accession = head["accession"]
    if not ACCESSION.fullmatch(accession):
        raise ValueError(f"{where}: bad accession {accession[:40]!r} (expected like 0000320193-25-000077)")
    try:
        symbol = check_symbol(head["symbol"])
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    dates = {}
    for field in ("filed", "period_end"):
        if head.get(field) or field == "filed":
            try:
                dates[field] = datetime.strptime(head[field], DATE_FORMAT).replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(f"{where}: bad {field} {head[field]!r} (expected like 2026-10-23)") from None
    status = head["text_status"]
    if status not in ("ok", "not_available"):
        raise ValueError(f"{where}: bad text_status {status!r} (expected ok or not_available)")
    if (status == "ok") != bool(text):
        raise ValueError(f"{where}: text_status is {status} but the record has {'text' if text else 'no text'}")
    form, filed = head["form"], head["filed"]
    sections = _sections(where, text) if text else []

    def document(article_id: str, label: str, body: str) -> Document:
        headline = f"{symbol} {form}" + (f", {label}" if label else "") + f", filed {filed}" + (f", period ending {head['period_end']}" if head.get("period_end") else "")
        return Document(
            id=article_id,
            page_content="\n\n".join(part for part in (headline, decode_entities(body)) if part),
            metadata={
                "article_id": article_id,
                "headline": headline,
                "tickers": [symbol],  # a filing is one company's own: the symbol it was fetched for
                "published_utc": f"{filed}T00:00:00Z",
                "published_ts": int(dates["filed"].timestamp()),
                "publisher": FILING_PUBLISHER,
                "api": "polygon",
                "url": head["url"],
            },
        )

    if sections:
        return [document(f"{FILING_ID_PREFIX}{accession}:{name}", SECTION_LABELS.get(name, name.replace("_", " ")), body) for name, body in sections]
    name = form_name(form)  # no text (or only empty sections): one short Document that says so
    return [document(f"{FILING_ID_PREFIX}{accession}", "", f"Form {form}" + (f" ({name})" if name else "") + f" filed {filed}. "
                     "The text of this filing is not stored: only its form, date and link are known.")]


def _print_document(doc: Document, number: int, total: int, limit: int = 700) -> None:
    text = doc.page_content
    if len(text) > limit:
        text = f"{text[:limit]} ... [{len(doc.page_content) - limit} more characters]"
    print(f"\n--- Document {number} of {total} ---")
    print(f"id: {doc.id}")
    print(f"page_content ({len(doc.page_content)} characters):")
    print("  " + text.replace("\n", "\n  "))
    print("metadata:")
    print("  " + json.dumps(doc.metadata, indent=2, ensure_ascii=False).replace("\n", "\n  "))


def main(argv: list[str] | None = None) -> int:
    """Print what a news file (or, with --press-releases or --filings, a press release or filing file) turns into: python -m prometheus.rag.loaders AAPL [--show 2]"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("target", help="a ticker symbol such as AAPL, or the path of a news file")
    parser.add_argument("--show", type=int, default=2, help="how many Documents to print in full (default 2)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--press-releases", action="store_true", help="read <SYMBOL>_PRs.txt (press releases) instead of the news file")
    which.add_argument("--filings", action="store_true", help="read <SYMBOL>.txt in the filings folder (SEC filings) instead of the news file")
    args = parser.parse_args(argv)
    skipped: list[Skipped] = []
    try:
        is_path = "/" in args.target or args.target.endswith(".txt")
        if args.press_releases:
            path = Path(args.target) if is_path else press_release_file(args.target)
            docs = load_press_releases(path, skipped)
        elif args.filings:
            path = Path(args.target) if is_path else filing_file(args.target)
            docs = load_filings(path)
        else:
            path = Path(args.target) if is_path else news_file(args.target)
            docs = load_news(path, skipped)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(path)
    if skipped:
        print(f"{len(skipped)} record{'s' if len(skipped) != 1 else ''} failed and {'were' if len(skipped) != 1 else 'was'} discarded (never loaded):")
        for item in skipped[:10]:
            print(f"  {item.where}: {item.reason}")
    kind = "press releases" if args.press_releases else "filing documents" if args.filings else "articles"
    if not docs:
        print("no press releases" if args.press_releases else "no filings" if args.filings else "no articles")
        return 0
    times = sorted(d.metadata["published_utc"] for d in docs)
    with_insights = [i for i, d in enumerate(docs) if INSIGHT_LINE.search(d.page_content)]
    print(f"{len(docs)} {kind} | {times[0]} to {times[-1]} | {len(with_insights)} with insights | "
          f"{len({d.id for d in docs})} distinct ids")
    shown = list(range(min(max(args.show, 0), len(docs))))
    if shown and with_insights and with_insights[0] not in shown:
        shown.append(with_insights[0])  # also show one article that has insights
    for i in shown:
        _print_document(docs[i], i + 1, len(docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
