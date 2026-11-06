"""The text of a 10-Q, out of EDGAR's complete submission file (design choices entry 36).

The Massive filings API gives the text of a 10-K and an 8-K and nothing for a 10-Q. The 10-Q is on EDGAR. The url stored in every filing record
(https://www.sec.gov/Archives/edgar/data/<cik>/<accession>.txt) is the COMPLETE SUBMISSION: an SGML header and every document of the filing (for a real AAPL 10-Q:
60 documents, 5.2 MB: the 10-Q as an inline-XBRL web page of 766,000 characters, three certifications, the XBRL files, 44 rendered statement pages ...).
This module takes the 10-Q document out of it and turns it into sections of plain text. It is plain Python: no network, no file, no LLM, and only the standard library
(`html.parser`), so a test can run it on a real submission.

    facts = submission_facts(submission)                       # accession, form, period_end from the SGML header
    sections = filing_sections(submission, "10-Q")             # [("financial_statements", "Item 1. Financial Statements\\n..."), ("mdna", ...), ...]

The real 10-Q looked at (design/week4.md section 23): the table of contents is a table (`Item 1. | Financial Statements | 1`); the real headings are lines of their
own, `PART I - FINANCIAL INFORMATION`, `Item 1. Financial Statements`, ... `Item 1A. Risk Factors` ...; item numbers repeat in Part II, so the part is tracked. A
table row is written as `cell | cell | cell` (the same way the Massive 8-K text writes its tables), the hidden XBRL header is dropped, and the cover page and the table of
contents are left out. A 10-Q with no Item heading at all is an error that says so: a layout nobody has looked at must be seen, not stored as one blob.
"""
import re
from html.parser import HTMLParser

SEC_PREFIX = "https://www.sec.gov/Archives/"  # the only place the identity header is ever sent
TEXT_FORMS = ("10-Q",)                        # the forms whose text this module can take out (design/week4.md decision G4)
SKIP_TAGS = {"head", "script", "style", "title", "ix:header"}   # ix:header is the hidden block of XBRL facts at the top of an inline-XBRL page
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "col", "area", "base", "wbr"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "table", "tr"}

# What each Item of a 10-Q is called as a section name: (part, item) -> name. A name that is not here is `part<n>_item<x>` (`item<x>` when no part was seen).
ITEM_NAMES = {
    ("I", "1"): "financial_statements", ("I", "2"): "mdna", ("I", "3"): "market_risk", ("I", "4"): "controls",
    ("II", "1"): "legal_proceedings", ("II", "1A"): "risk_factors", ("II", "2"): "unregistered_sales", ("II", "3"): "defaults",
    ("II", "4"): "mine_safety", ("II", "5"): "other_information", ("II", "6"): "exhibits",
}
# `Item 1A. Risk Factors`, `ITEM 2 - Management's ...`, `Item 3.`  The title, when there is one, starts with a capital letter: "Item 1A of the 2025 Form 10-K ..." at the start of
# a paragraph is a cross-reference, not a heading. A table row (the table of contents) has " | " in it and is not a heading either.
ITEM_LINE = re.compile(r"item\s+(\d{1,2}[a-z]?)\s*[.:\u2013\u2014-]*\s*(.*)", re.I)
# `PART I`, `PART II - OTHER INFORMATION`, `Part I`; the words after the dash must not contain a lower-case letter, so "Part I, Item 1A of the ..." is not a part line.
# Only the word and the numeral are case-insensitive: under re.I the class [^a-z] would exclude capitals too and "PART I - FINANCIAL INFORMATION" would not match (found by the
# first run on the real 10-Q: the part stayed at II from the table of contents, and Part I's Items were named as Part II's).
PART_LINE = re.compile(r"(?i:part)\s+(?i:(iv|iii|ii|i))\b(?:\s*[.:\u2013\u2014-]\s*(?:[^a-z]{0,70}|(?i:financial information|other information)))?")   # "Part I. Financial Information" (Restaurant Brands) too
MAX_HEADING = 150


class EdgarNotConfigured(RuntimeError):
    """SEC_USER_AGENT is not set (or is not a contact). EDGAR asks every client to say who it is, and the code never invents that."""


class EdgarLayoutError(ValueError):
    """The submission is not one this module can read (no such document, no Item headings). It stops the operation: the message says what was not found."""


def submission_facts(submission: str) -> dict:
    """{accession, form, period_end} from the SGML header of a complete submission. period_end is 'YYYY-MM-DD', or '' when the header has none."""
    header = submission[:submission.find("</SEC-HEADER>")] if "</SEC-HEADER>" in submission else submission[:5000]
    def field(name: str) -> str:
        found = re.search(rf"(?m)^{re.escape(name)}:\s*(\S[^\n]*)$", header)
        return found.group(1).strip() if found else ""
    period = field("CONFORMED PERIOD OF REPORT")
    return {"accession": field("ACCESSION NUMBER"), "form": field("CONFORMED SUBMISSION TYPE"),
            "period_end": f"{period[:4]}-{period[4:6]}-{period[6:8]}" if re.fullmatch(r"\d{8}", period) else ""}


def main_document(submission: str, form: str) -> str:
    """The text between <TEXT> and </TEXT> of the first document whose <TYPE> is `form` (the 10-Q itself, not its exhibits)."""
    for found in re.finditer(r"<DOCUMENT>\s*<TYPE>([^\n]*)\n(.*?)</DOCUMENT>", submission, re.S):
        if found.group(1).strip() == form:
            body = found.group(2)
            start, end = body.find("<TEXT>"), body.rfind("</TEXT>")
            if start < 0 or end < start:
                raise EdgarLayoutError(f"the {form} document of the submission has no <TEXT> block")
            return body[start + len("<TEXT>"):end]
    raise EdgarLayoutError(f"the submission has no document of type {form}")


class _Text(HTMLParser):
    """HTML to lines of text. Hidden parts (SKIP_TAGS, style display:none) are dropped; a table row becomes `cell | cell | cell` with empty cells left out;
    every block element starts a new line. Tables are tracked on a stack, so a table inside a cell does not lose the outer row."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.open: list[tuple[str, bool]] = []    # (tag, whether it is hidden): popped when the matching end tag comes, so bad HTML cannot leave a skip on
        self.tables: list[dict] = []              # one {"row": list | None, "cell": list | None} per table being read

    def handle_starttag(self, tag, attrs):
        hidden = tag in SKIP_TAGS or "display:none" in dict(attrs).get("style", "").replace(" ", "").lower()
        if tag not in VOID_TAGS:
            self.open.append((tag, hidden))
            self.skip += hidden
        if self.skip:
            return
        if tag == "table":
            self.tables.append({"row": None, "cell": None})
        elif tag == "tr" and self.tables:
            self.tables[-1]["row"] = []
        elif tag in ("td", "th") and self.tables:
            self.tables[-1]["cell"] = []
        if tag in BLOCK_TAGS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        for i in range(len(self.open) - 1, -1, -1):
            if self.open[i][0] == tag:
                self.skip -= sum(hidden for _, hidden in self.open[i:])
                del self.open[i:]
                break
        if self.skip:
            return
        table = self.tables[-1] if self.tables else None
        if tag in ("td", "th") and table and table["cell"] is not None and table["row"] is not None:
            text = " ".join("".join(table["cell"]).split())
            table["cell"] = None
            if text:
                table["row"].append(text)
        elif tag == "tr" and table and table["row"] is not None:
            if table["row"]:
                self.out.append("\n" + " | ".join(table["row"]) + "\n")
            table["row"] = None
        elif tag == "table" and self.tables:
            self.tables.pop()
        if tag in BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        table = self.tables[-1] if self.tables else None
        (table["cell"] if table and table["cell"] is not None else self.out).append(data)


def html_to_lines(html: str) -> list[str]:
    """The lines of text of an HTML page, whitespace collapsed (a no-break space is a space, a zero-width space is nothing), empty lines dropped."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    text = "".join(parser.out).replace("\u200b", "")
    return [line for line in (" ".join(raw.split()) for raw in text.split("\n")) if line]


def _two_cell_heading(line: str) -> str:
    """Procter & Gamble writes its real headings as a two-cell table row, `Item 1. | Financial Statements` (found on its real 10-Q: "no Item heading was found"). A row of exactly two cells,
    an item number and then a title that starts with a capital letter, is a heading and is returned as `Item 1. Financial Statements`. The table of contents has a page number in
    a third cell (`Item 1. | Financial Statements | 1`) and `PART I | Item 1. | ...` has four: neither is changed."""
    cells = line.split(" | ")
    if len(cells) == 2 and re.fullmatch(r"item\s+\d{1,2}[a-z]?\s*[.:\u2013\u2014-]*", cells[0], re.I) and cells[1][:1].isupper() and not cells[1].isdigit():
        return f"{cells[0]} {cells[1]}"
    return line


def split_items(lines: list[str]) -> list[tuple[str, str]]:
    """The Items of a 10-Q as (section name, text), in order. Text before the first Item heading (the cover page and the table of contents) is left out; the heading line
    stays at the top of its section. A name that comes twice keeps the longer text (a table of contents written as plain lines gives a heading that is a line long, and then
    the real one). A section with nothing after its heading is dropped. Raises EdgarLayoutError when there is no Item heading."""
    part = None
    sections: list[list] = []  # [name, lines]
    for line in lines:
        line = _two_cell_heading(line)
        part_line = PART_LINE.fullmatch(line) if len(line) <= 80 else None
        heading = ITEM_LINE.fullmatch(line) if len(line) <= MAX_HEADING and " | " not in line else None
        if part_line:
            part = part_line.group(1).upper()
        elif heading and (not heading.group(2) or heading.group(2)[0].isupper()):
            item = heading.group(1).upper()
            name = ITEM_NAMES.get((part, item)) or (f"part{part.lower()}_item{item.lower()}" if part else f"item{item.lower()}")
            sections.append([name, [line]])
        elif sections:
            sections[-1][1].append(line)
    if not sections:
        raise EdgarLayoutError("no Item heading (like 'Item 1. Financial Statements') was found in the text")
    best: dict[str, int] = {}
    for i, (name, body) in enumerate(sections):
        if name not in best or sum(map(len, body)) > sum(map(len, sections[best[name]][1])):
            best[name] = i
    return [(name, "\n".join(body)) for i, (name, body) in enumerate(sections) if best[name] == i and len(body) > 1]


def filing_sections(submission: str, form: str) -> list[tuple[str, str]]:
    """The sections of the `form` document of a complete submission, as (name, text). Only the forms in TEXT_FORMS are known."""
    if form not in TEXT_FORMS:
        raise EdgarLayoutError(f"the text of a {form} cannot be taken from EDGAR yet (only: {', '.join(TEXT_FORMS)})")
    sections = split_items(html_to_lines(main_document(submission, form)))
    if not sections:
        raise EdgarLayoutError(f"every Item of the {form} was empty")
    return sections
