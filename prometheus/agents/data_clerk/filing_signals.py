"""Filtered evidence from stored filings, for the Advisor (design choices entry 38). Plain Python: text in, facts out. No file, no network, no model, and NO SCORE.

    DataClerkTools.fetch_filings_recommendation(symbol)  ->  recommendation_evidence(...)          # the method the Data Clerk's tools expose (not an LLM tool)

What it returns is evidence, never a judgment: the figures of the latest balance sheet (both periods and the change), the capital actions of the cash-flow statement
(repurchases, dividends, share-based compensation, ...), the headline income figures, and the passages of the stored filings that are about stock dilution, executive
compensation and the like, each as an EXACT QUOTE with its filing. The Data Clerk's LLM will summarize it into the work order and the Advisor will score the sentiment
(Jacob, 2026-10-29): so nothing here is a score, a stance, a net result or a recommendation, and a test walks a whole result and refuses those words as keys.

The topics are in two fixed groups, `TOPIC_FILTER`: negative-type (dilution, executive compensation, employee equity compensation) and positive-type (buybacks, dividends).
That is the filter that was asked for, a table by TOPIC. It does not say a passage is good or bad news: a buyback that is smaller than last year's is still in the positive-type
group, with both numbers beside it.

Real pages this was written against (design/week4.md section 35): Apple's and META's 10-Q, whose balance sheets, units and layouts differ (shares in thousands in the units line
for Apple, `2,214 million` in the label for META; `Accumulated deficit` and `Retained earnings`; `Term debt` twice and `Long-term debt`).
"""
import re
from datetime import date, datetime, timedelta, timezone

from prometheus.rag.loaders import filing_parts

TOPIC_FILTER = {"dilution": "negative", "executive_compensation": "negative", "equity_compensation": "negative", "buyback": "positive", "dividend": "positive"}
TOPIC_PATTERNS = {
    "dilution": r"at-the-market|public offering|secondary offering|convertible (?:senior )?notes?|\bwarrants?\b|issuance of (?:additional )?(?:shares|common stock|class [ab] common stock)"
                r"|automatic increase|increase (?:in the number of shares|of \d[\d,]* (?:million )?shares)|anti-?dilutive|dilutive (?:effect|share|securit)|dilution"
                r"|(?:shares|number of shares) (?:available|reserved) for (?:future )?issuance",
    "executive_compensation": r"compensatory arrangements?|compensation committee|base salary|sign-?on bonus|signing bonus|target (?:annual )?(?:cash )?(?:bonus|incentive)|bonus plan"
                              r"|retention (?:award|bonus|grant)|severance|employment agreement|clawback|performance[- ]based (?:stock|restricted|equity|rsu|award)"
                              r"|equity incentive plan",
    "equity_compensation": r"restricted stock units?|\brsus?\b|share-based compensation|stock-based compensation|net share settlement|stock options?|performance (?:stock )?units?",
    "buyback": r"repurchas|buyback",
    "dividend": r"dividend",
}
TOPICS = {topic: re.compile(pattern, re.I) for topic, pattern in TOPIC_PATTERNS.items()}
SECTION_RANK = {"items": 0, "financial_statements": 1, "mdna": 2, "controls": 3, "other_information": 3, "unregistered_sales": 3, "risk_factors": 9}  # 8-K items and the statements first, the generic risk factors last: source type BEFORE recency (a newer 10-Q's risk factors must not push an officer's pay terms out of the shown passages: found on META)
EVIDENCE_PER_TOPIC = 5      # passages shown per topic: the digest must stay small enough to put in a prompt; `found` says how many there were
MAX_QUOTE = 400             # characters of a quote; a longer sentence is cut and marked `truncated`
MIN_QUOTE = 30              # a shorter line is a heading ("Restricted Stock Units") with nothing to quote
DEFAULT_WINDOW = 365
FORBIDDEN_KEYS = ("score", "stance", "sentiment", "recommendation", "rating", "verdict")  # no result of this module may use these as a key: it scores nothing

# ---------------------------------------------------------------- rows of the financial statements
NUMBER = re.compile(r"\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?%?")
DASH = {"\u2014", "\u2013", "-"}
STATEMENTS = [("balance_sheet", re.compile(r"balance sheets?", re.I)), ("cash_flow", re.compile(r"statements? of cash flows", re.I)),
              ("equity", re.compile(r"statements? of (?:share|stock)holders[\u2019'] equity", re.I)), ("comprehensive", re.compile(r"statements? of comprehensive (?:income|loss)", re.I)),
              ("income", re.compile(r"statements? of (?:operations|income|earnings)", re.I)), ("notes", re.compile(r"^notes to |^note \d+", re.I))]


def to_number(cell: str) -> float | None:
    """'45,317' -> 45317.0, '(2,058)' -> -2058.0, '$ 0.26' -> 0.26, a dash -> 0.0, anything else None."""
    text = cell.strip().replace("$", "").strip()
    if text in DASH:
        return 0.0
    if not NUMBER.fullmatch(text):
        return None
    core = text[:-1] if text.endswith("%") else text  # `(1.5)%` is negative: the parenthesis is inside the percent sign
    negative = core.startswith("(") and core.endswith(")")
    value = float(core.strip("()").replace(",", "").strip())
    return -value if negative else value


def statement_of(line: str) -> str | None:
    """Which statement a heading line starts: only a short, all-capitals line without a bar counts (a sentence of a note that says 'balance sheet' does not)."""
    plain = re.sub(r"\(.*?\)", "", line).strip()
    if " | " in line or len(plain) > 120 or not (plain.isupper() or re.match(r"(?i)^notes? ", plain)):
        return None
    for name, pattern in STATEMENTS:
        if pattern.search(plain):
            return name
    return None


def parse_rows(text: str) -> list[dict]:
    """The rows of a `financial_statements` section as {statement, group, label, cells, values, columns, units, line}. A row is a line `label | cell | cell`; the `$` cells are dropped;
    `group` is the last line that ends with a colon (`Current liabilities:`); `columns` are the header rows seen since the statement's heading (`December 27,2025 | September 27,2025`);
    `units` is the line like `(In millions, except ...)` under the heading."""
    rows, statement, group, columns, units = [], None, "", [], ""
    for line in text.split("\n"):
        heading = statement_of(line)
        if heading:
            statement, group, columns, units = heading, "", [], ""
            continue
        if " | " not in line:
            if line.endswith(":"):
                group = line.rstrip(":")
            elif line.startswith("(In ") and not units:
                units = line
            continue
        cells = [c.strip() for c in line.split(" | ")]
        data = [c for c in cells[1:] if c != "$"]
        numbers = [to_number(c) for c in data]
        if data and all(n is not None for n in numbers) and to_number(cells[0]) is None:
            rows.append({"statement": statement, "group": group, "label": cells[0], "cells": data, "values": numbers, "columns": list(columns), "units": units, "line": line})
        elif all(to_number(c) is None or re.fullmatch(r"20\d\d", c) for c in cells) and len(cells) >= 2:
            columns = (columns + [" | ".join(cells)])[-2:]  # a header row: dates, or `Three Months Ended | Nine Months Ended` then the years
    return rows


def change(current: float | None, prior: float | None) -> dict:
    if current is None or prior is None:
        return {"change": None, "change_pct": None}
    return {"change": round(current - prior, 4), "change_pct": round((current - prior) / abs(prior) * 100, 2) if prior else None}


def two_values(row: dict) -> dict:
    """A row of two columns as {label, group, current, prior, change, change_pct, as_reported}; a row of another width keeps its values as they are."""
    if len(row["values"]) == 2:
        return {"label": row["label"], "group": row["group"], "current": row["values"][0], "prior": row["values"][1], **change(*row["values"]), "as_reported": row["cells"]}
    return {"label": row["label"], "group": row["group"], "values": row["values"], "as_reported": row["cells"]}


BALANCE_ITEMS = [("cash", r"cash and cash equivalents"), ("marketable_securities", r"marketable securities"), ("total_current_assets", r"total current assets"),
                 ("total_assets", r"total assets"), ("total_current_liabilities", r"total current liabilities"), ("total_liabilities", r"total liabilities"),
                 ("total_equity", r"total (?:share|stock)holders[\u2019'] equity"), ("retained_earnings", r"retained earnings|accumulated deficit")]
DEBT_ROW = re.compile(r"term debt|long-term debt|commercial paper|notes payable|short-term (?:debt|borrowings)|borrowings|convertible", re.I)
SHARES_ROW = re.compile(r"([\d,]+)( million)?\s+and\s+([\d,]+)( million)?\s+shares issued and outstanding", re.I)
AUTHORIZED = re.compile(r"([\d,]+)( million)?\s+(?:class [ab] )?shares authorized", re.I)


def balance_sheet(rows: list[dict]) -> dict | None:
    """The balance sheet as figures for both periods: the key rows, every debt row, and the shares issued and outstanding (read from the label of the common stock row, in the unit
    that row or the units line gives). None when there is no balance sheet in these rows."""
    sheet = [r for r in rows if r["statement"] == "balance_sheet"]
    if not sheet:
        return None
    out = {"columns": sheet[0]["columns"][-1].split(" | ") if sheet[0]["columns"] else [], "units": sheet[0]["units"], "items": {}, "debt": [], "shares": None}
    # (the debt rows and the shares are filled below)
    for key, pattern in BALANCE_ITEMS:
        found = [two_values(r) for r in sheet if re.fullmatch(pattern, r["label"].strip().rstrip(":"), re.I) and len(r["values"]) == 2]
        if found:
            out["items"][key] = found if key == "marketable_securities" else found[0]  # Apple has current and non-current marketable securities
    for r in sheet:
        classes = share_classes(r["label"], r["units"])
        if classes:
            units = {c["unit"] for c in classes}
            total = ({"current": sum(c["current"] for c in classes), "prior": sum(c["prior"] for c in classes)} if len(units) == 1 else None)
            out["shares"] = {"label": r["label"], "unit": classes[0]["unit"] if len(units) == 1 else "mixed", "classes": classes, "as_reported": r["cells"],
                             **({**total, **change(total["current"], total["prior"])} if total else {"current": None, "prior": None, **change(None, None)})}
            break
    below = False  # a debt row is current or non-current by its place against `Total current liabilities` (a group line like `Non-current liabilities:` is not always there: META has none)
    for r in sheet:
        if re.fullmatch(r"total current liabilities", r["label"].strip(), re.I):
            below = True
        elif DEBT_ROW.search(r["label"]) and len(r["values"]) == 2 and not r["label"].lower().startswith(("total", "debt securities")):
            out["debt"].append({**two_values(r), "classification": "non-current" if below else "current"})
    return out


def share_classes(label: str, units: str) -> list[dict]:
    """The shares issued and outstanding in the label of the common stock row, one entry per class: `2,214 million and 2,187 million shares issued and outstanding` (META, Class A, then
    `342 million and 343 million` for Class B) or `14,702,703 and 14,773,260 shares issued and outstanding` (Apple, in the thousands of the units line). The authorized number is the
    nearest one before it. The label gives the current period first, then the prior, as the columns do."""
    out, taken = [], 0
    for found in SHARES_ROW.finditer(label):
        before = label[taken:found.start()]
        names = re.findall(r"class [ab]", before, re.I)
        authorized = list(AUTHORIZED.finditer(before))
        current, prior = (float(found.group(i).replace(",", "")) for i in (1, 3))
        out.append({"class": names[-1].title() if names else "common stock", "current": current, "prior": prior, **change(current, prior),
                    "authorized": float(authorized[-1].group(1).replace(",", "")) if authorized else None,
                    "unit": "million" if found.group(2) else "thousands" if "reflected in thousands" in units else "shares"})
        taken = found.end()
    return out


CASH_FLOW_ITEMS = [("repurchases of common stock", "buyback", r"repurchases? of |common stock repurchased|share repurchases?|purchases? of treasury"),
                   ("dividends paid", "dividend", r"payments? for dividends|dividends? paid|cash dividends"),
                   ("share-based compensation expense", "equity_compensation", r"share-based compensation|stock-based compensation"),
                   ("taxes paid for net share settlement", "equity_compensation", r"taxes (?:paid )?related to net share settlement"),
                   ("proceeds from issuing common stock", "dilution", r"proceeds from (?:the )?(?:issuance of (?:common|class [ab])|exercise of (?:stock )?options|employee stock)"),
                   ("proceeds from issuing debt", None, r"proceeds from (?:the )?issuance of (?:long-term|term) debt|proceeds from (?:the )?issuance of .*notes"),
                   ("repayments of debt", None, r"repayments? of (?:long-term |term )?debt|repayments? of commercial paper")]


def capital_actions(rows: list[dict]) -> list[dict]:
    """Rows of the cash-flow statement that are about the shares and the debt: repurchases, dividends paid, share-based compensation, taxes for net share settlement, proceeds from
    issuing stock or debt, repayments. The cash-flow statement is two columns in the real 10-Qs (the equity statement is 10 columns for META). `topic` is the topic the row belongs
    to, or None."""
    out = []
    for r in rows:
        if r["statement"] != "cash_flow" or len(r["values"]) != 2:
            continue
        for item, topic, pattern in CASH_FLOW_ITEMS:
            if re.search(pattern, r["label"], re.I):
                out.append({"item": item, "topic": topic, **two_values(r), "columns": r["columns"][-1].split(" | ") if r["columns"] else []})
                break
    return out


INCOME_ITEMS = [("revenue", None, r"(?:total )?(?:net sales|revenues?)"), ("net income", None, r"net income"), ("diluted earnings per share", "per share", r"diluted"),
                ("basic weighted shares", "shares used", r"basic"), ("diluted weighted shares", "shares used", r"diluted")]


def income_headlines(rows: list[dict]) -> list[dict]:
    """Revenue, net income, diluted earnings per share and the weighted-average shares (basic and diluted): rows of the income statement, with the columns as the filing gives them
    (META's have four: three and nine months, this year and last)."""
    out, taken = [], set()
    for r in rows:
        if r["statement"] != "income":
            continue
        for item, group, pattern in INCOME_ITEMS:
            if item not in taken and re.fullmatch(pattern, r["label"].strip(), re.I) and (group is None or group in r["group"].lower()):
                taken.add(item)
                out.append({"item": item, "label": r["label"], "values": r["values"], "as_reported": r["cells"], "columns": r["columns"][-2:]})
                break
    return out


# ---------------------------------------------------------------- passages
def quote_of(line: str, match: re.Match) -> tuple[str, bool]:
    """The sentence of `line` that holds the match, as a substring of the line (never reworded), at most MAX_QUOTE characters. A line of a table, or a short line, is the quote itself."""
    if len(line) <= MAX_QUOTE:
        return line, False
    start = max((m.end() for m in re.finditer(r"(?<=[.;:])\s+", line[:match.start()])), default=0)
    end = next((m.start() for m in re.finditer(r"(?<=[.;])\s+(?=[A-Z(])", line[match.end():])), None)
    sentence = line[start:match.end() + end] if end is not None else line[start:]
    return (sentence[:MAX_QUOTE], True) if len(sentence) > MAX_QUOTE else (sentence, False)


def passages(filings: list[dict]) -> dict[str, list[dict]]:
    """Every passage of the stored text that matches a topic, per topic, de-duplicated by quote, in this order: the source type first (8-K items, then the statements, then the MD&A,
    ..., the generic risk factors last), then the newest filing first, then the order in the text. A passage can be in several topics. A line shorter than MIN_QUOTE is a heading
    and is skipped. A line of a table also says which statement it is in and its columns."""
    found: dict[str, list[dict]] = {topic: [] for topic in TOPICS}
    seen: dict[str, set[str]] = {topic: set() for topic in TOPICS}
    for filing in sorted(filings, key=lambda f: f["filed"], reverse=True):
        for section, text in filing["sections"].items():
            table = {r["line"]: r for r in parse_rows(text)} if section == "financial_statements" else {}  # which statement a table row is in, and its columns
            for number, line in enumerate(text.split("\n")):
                if len(line) < MIN_QUOTE:
                    continue
                for topic, pattern in TOPICS.items():
                    match = pattern.search(line)
                    if match and line not in seen[topic]:
                        quote, cut = quote_of(line, match)
                        seen[topic].add(line)
                        row = table.get(line)
                        found[topic].append({"topic": topic, "filter": TOPIC_FILTER[topic], "matched": match.group(0).lower(), "quote": quote, "truncated": cut,
                                             "statement": row["statement"] if row else None, "columns": row["columns"][-1].split(" | ") if row and row["columns"] else [],
                                             "source": {"accession": filing["accession"], "form": filing["form"], "filed": filing["filed"], "section": section, "url": filing["url"]},
                                             "_rank": (SECTION_RANK.get(section, 5), -int(filing["filed"].replace("-", "")), number)})
    for items in found.values():
        items.sort(key=lambda item: item["_rank"])
    return found


# ---------------------------------------------------------------- the whole
def filings_from_documents(docs) -> list[dict]:
    """The Documents `load_filings` made, grouped by filing: {accession, form, filed, period_end, url, has_text, sections: {name: text}}. The headline that each Document starts with is
    taken off the text: what is left is what the filing says."""
    filings: dict[str, dict] = {}
    for doc in docs:
        parts = filing_parts(doc)
        filing = filings.setdefault(parts["accession"], {"accession": parts["accession"], "form": parts["form"], "filed": parts["filed"], "period_end": parts["period_end"],
                                                         "url": parts["url"], "has_text": False, "sections": {}})
        if parts["section"]:
            filing["has_text"] = True
            filing["sections"][parts["section"]] = doc.page_content.split("\n\n", 1)[1] if "\n\n" in doc.page_content else ""
    return list(filings.values())


def number_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def move(row: dict) -> str:
    if "current" not in row:
        return ", ".join(number_text(v) for v in row.get("values", []))
    pct = f", {row['change_pct']:+.2f}%" if row.get("change_pct") is not None else ""
    return f"{number_text(row['current'])} vs {number_text(row['prior'])} ({row['change']:+,.0f}{pct})" if row["change"] is not None else f"{number_text(row['current'])} vs {number_text(row['prior'])}"


def recommendation_evidence(symbol: str, filings: list[dict], today: date | None = None, since_days: int = DEFAULT_WINDOW) -> dict:
    """The evidence package for one company from its stored filings (see the module text). `filings` is `filings_from_documents(...)`, `since_days` the window of filing dates
    counted back from `today`. Returns {symbol, as_of, window_days, filings, balance_sheet, capital_actions, income, evidence, found, gaps, digest}; scores nothing."""
    today = today or datetime.now(timezone.utc).date()
    since = (today - timedelta(days=since_days)).isoformat()
    window = sorted((f for f in filings if f["filed"] >= since), key=lambda f: f["filed"], reverse=True)
    stored = [{"accession": f["accession"], "form": f["form"], "filed": f["filed"], "period_end": f["period_end"], "sections": list(f["sections"]), "url": f["url"]} for f in window if f["has_text"]]
    by_form: dict[str, dict] = {}
    for f in window:
        entry = by_form.setdefault(f["form"], {"form": f["form"], "count": 0, "with_text": 0, "latest": f["filed"]})
        entry["count"] += 1
        entry["with_text"] += f["has_text"]
    source = next((f for f in window if f["form"] in ("10-Q", "10-K") and "financial_statements" in f["sections"]), None)
    rows = parse_rows(source["sections"]["financial_statements"]) if source else []
    sheet, actions, income = balance_sheet(rows), capital_actions(rows), income_headlines(rows)
    if sheet:
        sheet["source"] = {"accession": source["accession"], "form": source["form"], "filed": source["filed"], "period_end": source["period_end"], "url": source["url"],
                           "section": "financial_statements"}
    found = passages(window)
    evidence, shown = [], {}
    derived = []
    shares = sheet and sheet["shares"]
    where = source and {"accession": source["accession"], "form": source["form"], "filed": source["filed"], "url": source["url"]}  # the filing the statements were read from
    if shares and shares["change"]:
        up = shares["change"] > 0
        every = "; ".join(f"{c['class']} {move(c)}" for c in shares["classes"]) if len(shares["classes"]) > 1 else ""
        derived.append({"topic": "dilution" if up else "buyback", "statement": "balance_sheet", "columns": sheet["columns"],
                        "what": f"total shares issued and outstanding {'rose' if up else 'fell'}: {move(shares)} {shares['unit']}" + (f" ({every})" if every else ""),
                        "quote": shares["label"]})
    for action in actions:
        if action["topic"] in ("buyback", "dividend", "dilution") and "current" in action and (action["current"] or action["prior"]):
            derived.append({"topic": action["topic"], "statement": "cash_flow", "columns": action["columns"],
                            "what": f"cash flow, {action['item']}: {move(action)} ({'; '.join(action['columns']) or 'columns as in the statement'})", "quote": action["label"]})
    for item in derived:
        evidence.append({"id": "", "topic": item["topic"], "filter": TOPIC_FILTER[item["topic"]], "what": item["what"], "matched": "", "quote": item["quote"], "truncated": False, "statement": item.get("statement"), "columns": item.get("columns", []),
                         "derived": True, "source": {"accession": where["accession"], "form": where["form"], "filed": where["filed"], "section": "financial_statements", "url": where["url"]}})
    for topic in TOPICS:
        for item in found[topic][:EVIDENCE_PER_TOPIC]:
            item = {k: v for k, v in item.items() if k != "_rank"}
            kind = {"notes": "table in the notes", None: ""}.get(item["statement"], f"{(item['statement'] or '').replace('_', ' ')} statement row")
            where_in = f"{item['source']['form']} {item['source']['section']}" + (f", {kind}" if kind else "")
            evidence.append({"id": "", **item, "what": f"{where_in}: mentions '{item['matched']}'", "derived": False})
        shown[topic] = min(len(found[topic]), EVIDENCE_PER_TOPIC)
    evidence.sort(key=lambda e: (e["filter"] != "negative", list(TOPICS).index(e["topic"]), not e["derived"]))
    for number, item in enumerate(evidence, 1):
        item["id"] = f"E{number}"
    counts = {topic: {"filter": TOPIC_FILTER[topic], "found": len(found[topic]), "shown": shown[topic], "derived": sum(1 for e in evidence if e["topic"] == topic and e["derived"])} for topic in TOPICS}
    gaps = find_gaps(symbol, filings, window, source, since_days)
    if source and not sheet:
        gaps.insert(0, f"the financial statements of the {source['form']} filed {source['filed']} have no balance sheet that could be read (the rows found are those of: {', '.join(sorted({r['statement'] or 'no statement' for r in rows})) or 'none'})")
    result = {"symbol": symbol, "as_of": max((f["filed"] for f in window), default=""), "window_days": since_days,
              "filings": {"in_window": len(window), "by_form": sorted(by_form.values(), key=lambda e: e["form"]), "with_text": stored},
              "balance_sheet": sheet, "capital_actions": actions, "income": income, "evidence": evidence, "found": counts, "gaps": gaps}
    result["digest"] = digest(result)
    return result


def find_gaps(symbol: str, filings: list[dict], window: list[dict], source: dict | None, since_days: int) -> list[str]:
    """What could not be read, and why, in words the next step can show: silence would read as 'nothing found'."""
    gaps = []
    if not filings:
        return [f"no filing of {symbol} is stored: run fetch_filings first"]
    if not window:
        gaps.append(f"no filing of {symbol} was filed in the last {since_days} days (the latest stored was filed {max(f['filed'] for f in filings)})")
    if source is None:
        no_text = [f for f in window if f["form"] == "10-Q" and not f["has_text"]]
        gaps.append("no balance sheet: no stored 10-Q or 10-K in the window has its financial statements (the Massive 10-K text is Business and Risk Factors only; a 10-Q's text comes from "
                    f"EDGAR when fetch_filings runs with SEC_USER_AGENT set; {len(no_text)} 10-Q filing(s) in the window are stored without text)")
    proxies = [f for f in window if f["form"].startswith(("DEF 14A", "PRE 14A", "DEF 14C", "PRE 14C")) and not f["has_text"]]
    if proxies:
        gaps.append(f"executive compensation plans are described in the proxy statement, and {len(proxies)} proxy/information statement(s) in the window are stored without text")
    if not any(f["form"] == "8-K" and f["has_text"] for f in window):
        gaps.append("no 8-K with text in the window (an 8-K is where an officer's compensation or an equity issue is reported)")
    return gaps


def digest(result: dict) -> str:
    """The same facts in fixed wording (a template filled from the result, not a model's prose), compact enough for a prompt. No score, no stance."""
    lines = [f"FILING EVIDENCE for {result['symbol']} (stored filings, last {result['window_days']} days, as of {result['as_of'] or 'n/a'}). This is evidence, not a score: nothing here rates the stock."]
    forms = ", ".join(f"{e['form']} x{e['count']} ({e['with_text']} with text)" for e in result["filings"]["by_form"]) or "none"
    lines.append(f"Filings in the window: {forms}.")
    sheet = result["balance_sheet"]
    if sheet:
        src = sheet["source"]
        lines.append(f"BALANCE SHEET from the {src['form']} filed {src['filed']} (period ended {src['period_end'] or 'n/a'}); columns {' vs '.join(sheet['columns'])}; {sheet['units']}")
        for key, value in sheet["items"].items():
            for row in (value if isinstance(value, list) else [value]):
                lines.append(f"  {row['label']}{' (' + row['group'] + ')' if row['group'] and key == 'marketable_securities' else ''}: {move(row)}")
        for row in sheet["debt"]:
            lines.append(f"  debt, {row['label']} ({row['classification']}): {move(row)}")
        if sheet["shares"]:
            s = sheet["shares"]
            for c in s["classes"]:
                lines.append(f"  shares issued and outstanding, {c['class']}: {move(c)} {c['unit']}" + (f"; authorized {number_text(c['authorized'])}" if c["authorized"] else ""))
            if len(s["classes"]) > 1:
                lines.append(f"  shares issued and outstanding, all classes: {move(s)} {s['unit']}")
    if result["capital_actions"]:
        lines.append("CAPITAL ACTIONS (cash-flow statement" + (f", columns {' vs '.join(result['capital_actions'][0]['columns'])}" if result["capital_actions"][0]["columns"] else "")
                     + "; as the filing shows them, so a cash outflow is negative):")
        lines.extend(f"  {a['item']}{' [row: ' + a['label'] + ']' if a['label'].lower() != a['item'] else ''}: {move(a)}" for a in result["capital_actions"])
    if result["income"]:
        lines.append("INCOME STATEMENT:")
        lines.extend(f"  {i['item']}: {', '.join(number_text(v) for v in i['values'])}" + (f" ({' / '.join(i['columns'])})" if i["columns"] else "") for i in result["income"])
    for group, title in (("negative", "NEGATIVE-TYPE TOPICS (dilution, executive compensation, employee equity compensation)"), ("positive", "POSITIVE-TYPE TOPICS (buybacks, dividends)")):
        counts = ", ".join(f"{t} {c['shown']} of {c['found']}" + (f" +{c['derived']} from the statements" if c["derived"] else "") for t, c in result["found"].items() if c["filter"] == group)
        lines.append(f"{title}: passages shown of found: {counts}")
        for e in (e for e in result["evidence"] if e["filter"] == group):
            s = e["source"]
            lines.append(f"  [{e['id']}] {e['topic']} | {s['form']} filed {s['filed']}, {s['section']} | {e['what']} | \"{e['quote']}\"" + (" [cut]" if e["truncated"] else ""))
    lines.extend(f"NOT AVAILABLE: {g}" for g in result["gaps"])
    return "\n".join(lines)


def keys_of(value) -> set[str]:
    """Every dictionary key anywhere in a result (for the test that it scores nothing)."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_of(v)}
    return set()
