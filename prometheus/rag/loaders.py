"""Load the DataAgent's news files into LangChain Documents (the "Documents" box of the slide).

The DataAgent writes news/<SYMBOL>.txt as a series of records (format: docs/DataAgent.md):

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

This module turns each record into one Document. It never imports the DataAgent: the file format is
the contract between them. A record the DataAgent would not have written stops the load with a
ValueError that names the file and the line.

Look at the result:   python -m prometheus.rag.loaders AAPL
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from prometheus.rag.config import news_dir

ARTICLE_START, SUMMARY_START, INSIGHTS_START, ARTICLE_END = (
    "<<<ARTICLE>>>", "<<<SUMMARY>>>", "<<<INSIGHTS>>>", "<<<END>>>")
REQUIRED_FIELDS = ("id", "headline", "published_utc", "url")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")  # AAPL, BRK.B, BF-B
INSIGHT_LINE = re.compile(r"(?m)^[A-Z][A-Z0-9.\-]* \| (positive|neutral|negative) \| ")


def news_file(symbol: str) -> Path:
    """The news file for one symbol: AAPL -> <news dir>/AAPL.txt.

    The symbol becomes part of a file name and will come from an LLM's tool call, so it is checked:
    letters, digits, '.' and '-' only, at most 10 characters. Something like '../../.env' is refused."""
    symbol = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"not a valid ticker symbol: {symbol!r}")
    return news_dir() / f"{symbol}.txt"


def load_news(path: Path | str) -> list[Document]:
    """Read one news file. Returns one Document per article, in file order."""
    path = Path(path)
    return [_to_document(path, *record) for record in _records(path)]


def _records(path: Path):
    """Yield (first line number, header fields, summary text, insights text) for each record.

    Raises ValueError, naming the file and line, for anything the DataAgent would not have written."""
    state = start = None  # state: None between records, else "head", "summary" or "insights"
    head: dict[str, str] = {}
    summary: list[str] = []
    insights: list[str] = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if line == ARTICLE_START:
                if state is not None:
                    raise ValueError(f"{path}:{number}: a new record starts before the one at line "
                                     f"{start} ended (it has no {ARTICLE_END})")
                state, start, head, summary, insights = "head", number, {}, [], []
            elif state is None:
                if line.strip():
                    raise ValueError(f"{path}:{number}: text outside a record: {line[:40]!r}")
            elif line == SUMMARY_START:
                state = "summary"
            elif line == INSIGHTS_START:
                state = "insights"
            elif line == ARTICLE_END:
                yield start, head, "\n".join(summary).strip(), "\n".join(insights).strip()
                state = None
            elif line.startswith("<<<"):
                raise ValueError(f"{path}:{number}: unknown marker {line!r}")
            elif state == "head":
                name, colon, value = line.partition(":")  # split on the FIRST colon: headlines contain colons
                if not colon:
                    raise ValueError(f"{path}:{number}: header line without a colon: {line[:40]!r}")
                head[name.strip()] = value.strip()
            elif state == "summary":
                summary.append(line)
            else:
                insights.append(line)
    if state is not None:
        raise ValueError(f"{path}:{start}: the record has no {ARTICLE_END}")


def _to_document(path: Path, start: int, head: dict[str, str], summary: str, insights: str) -> Document:
    """One record -> one Document. The text is the headline, summary and insight lines, as written."""
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
    return Document(
        id=head["id"],
        page_content="\n\n".join(part for part in (head["headline"], summary, insights) if part),
        metadata={
            "article_id": head["id"],
            "headline": head["headline"],
            "tickers": tickers,
            "published_utc": head["published_utc"],
            "published_ts": int(published.timestamp()),  # a number: Chroma range filters take numbers only
            "publisher": head.get("publisher", ""),
            "api": head.get("api", ""),
            "url": head["url"],
        },
    )


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
    """Print what a news file turns into: python -m prometheus.rag.loaders AAPL [--show 2]"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("target", help="a ticker symbol such as AAPL, or the path of a news file")
    parser.add_argument("--show", type=int, default=2, help="how many Documents to print in full (default 2)")
    args = parser.parse_args(argv)
    try:
        is_path = "/" in args.target or args.target.endswith(".txt")
        path = Path(args.target) if is_path else news_file(args.target)
        docs = load_news(path)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(path)
    if not docs:
        print("no articles")
        return 0
    times = sorted(d.metadata["published_utc"] for d in docs)
    with_insights = [i for i, d in enumerate(docs) if INSIGHT_LINE.search(d.page_content)]
    print(f"{len(docs)} articles | {times[0]} to {times[-1]} | {len(with_insights)} with insights | "
          f"{len({d.id for d in docs})} distinct ids")
    shown = list(range(min(max(args.show, 0), len(docs))))
    if shown and with_insights and with_insights[0] not in shown:
        shown.append(with_insights[0])  # also show one article that has insights
    for i in shown:
        _print_document(docs[i], i + 1, len(docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
