"""Step 2 of `Advisor.generate_recommendations`: which symbols to look at (Advisor v1 spec section 7, step 2, design choices entry 51).

    candidates, dropped = collect_candidates(order, portfolio, case, theme_tickers, limits)

Plain Python, no model, no network. A candidate is `{symbol, company, origin, held}`, in this order, de-duplicated, every symbol through `check_symbol`:

    1. subject    the work order's own symbols (a funds order names them; a theme order's subject is a TERM and names none);
    2. sources    a symbol in a source's `tickers` list that is ALSO named in the text of at least one source (the ticker as a word in capitals). A ticker list alone does not qualify
                  (entry 26): an article lists every company it touches, and most of them are not what it is about;
    3. discover   Case 2 only: the tickers the stored market news ties to the theme, with at least `discover_min_articles` articles (the caller reads them from the Data Clerk's
                  `discover_tickers`, which reads stored news and fetches nothing);
    4. held       every symbol of the portfolio, tagged `held: true`: they are for the holdings pass, not new ideas.

New ideas (1 to 3) are capped at `max_candidates` (15, and never so many that the metrics order for them and the holdings goes over 30 symbols): the subject first, then the symbols
more sources name, then alphabetical. What was cut is returned in `dropped`, so the run can say it.
"""
import re

from prometheus.agents.advisor.advisor import names_company
from prometheus.common.portfolio import Portfolio
from prometheus.common.work_order import MAX_METRIC_SYMBOLS, WorkOrder
from prometheus.rag.loaders import check_symbol


def subject_symbols(order: WorkOrder) -> list[str]:
    """The order's subject as checked symbols; a subject that is a term ("AI memory shortage") has none."""
    raw = [order.subject] if isinstance(order.subject, str) else list(order.subject or [])
    out = []
    for item in raw:
        if isinstance(item, str):
            try:
                out.append(check_symbol(item.strip().lstrip("$")))
            except ValueError:
                continue
    return list(dict.fromkeys(out))


def infer_case(order: WorkOrder) -> str:
    """funds when the subject is symbols, theme when it is a term (or there is none): the use cases of the week 1 design."""
    return "funds" if subject_symbols(order) else "theme"


def theme_term(order: WorkOrder) -> str | None:
    """The phrase a theme order is about: its subject when that is a term, else the start of the query."""
    if isinstance(order.subject, str) and order.subject.strip() and not subject_symbols(order):
        return order.subject.strip()[:60]
    return order.query.strip()[:60] if order.query and order.query.strip() else None


def named_in_sources(sources: list[dict]) -> dict[str, int]:
    """{symbol: number of sources that list it in `tickers` AND name it in their text}."""
    counts: dict[str, int] = {}
    for source in sources:
        for ticker in source.get("tickers") or []:
            try:
                symbol = check_symbol(ticker)
            except ValueError:
                continue
            if names_company(source.get("text") or "", symbol, None):
                counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def collect_candidates(order: WorkOrder, portfolio: Portfolio, case: str, theme_tickers: list[dict] | None, max_candidates: int, min_articles: int, only_subject: bool = False) -> tuple[list[dict], list[str]]:
    """The candidates and what was dropped. `theme_tickers` is the `tickers` list of the Data Clerk's discover result (Case 2), or None. `only_subject` (entry 53: the user named the companies, for a
    comparison) leaves out the symbols the sources name and the theme's tickers: the candidates are the subject and the holdings."""
    named = {p.symbol: p.name for p in portfolio.positions}
    subject = subject_symbols(order)
    company_of = {subject[0]: order.company} if order.company and len(subject) == 1 else {}
    sources = order.data.sources if order.data else []
    counts = named_in_sources(sources)
    ideas: dict[str, dict] = {}

    def add(symbol: str, origin: str):
        ideas.setdefault(symbol, {"symbol": symbol, "company": company_of.get(symbol) or named.get(symbol), "origin": origin, "held": symbol in named})

    for symbol in subject:
        add(symbol, "subject")
    for symbol in ([] if only_subject else sorted(counts, key=lambda s: (-counts[s], s))):
        add(symbol, "sources")
    if case == "theme" and not only_subject:
        for row in theme_tickers or []:
            if row.get("articles", 0) >= min_articles:
                try:
                    add(check_symbol(row["symbol"]), "discover")
                except (ValueError, KeyError):
                    continue
    held_only = [s for s in named if s not in ideas]
    ranked = sorted(ideas.values(), key=lambda c: (c["origin"] != "subject", -counts.get(c["symbol"], 0), c["symbol"]))
    room = max(0, min(max_candidates, MAX_METRIC_SYMBOLS - len(held_only)))   # the metrics order covers the new ideas and the holdings: at most 30 symbols
    new, cut = ranked[:room], ranked[room:]
    still_held = [c["symbol"] for c in cut if c["held"]]                      # a cut idea that the user holds is still reviewed, as a holding
    lost = [c["symbol"] for c in cut if not c["held"]]
    dropped = [f"{len(lost)} new idea(s) beyond the first {room}: {', '.join(lost)}"] if lost else []
    candidates = [*new, *({"symbol": s, "company": named[s], "origin": "held", "held": True} for s in [*held_only, *still_held])]
    return candidates, dropped
