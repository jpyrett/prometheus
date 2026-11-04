"""The Advisor, version 0: recommendations whose score is CALCULATED from evidence that code has checked (design choices entry 26).

    order = DataClerk().request_data(new_work_order("AI memory shortage. Which companies are affected?"))
    Advisor().advise(order)                       # appends order.recommendations; status becomes "recommended"
    Advisor().advise(order, use_evidence=False)   # the baseline for a comparison: the same Advisor, with no sources

    python labs/module3_rag/advisor_with_without.py     # Case 2 with and without the retrieved chunks, 5 runs each

The week 2 design says "a prompt will write a score that hallucinates and makes a fluent answer. The number is only reliable when it
is calculated from the criteria", so the model never writes the score here:

    1. the model reads the work order and the numbered sources and replies with SIGNALS for each company it recommends: a short quote
       copied word for word from one source, the source's number, and a direction (bullish or bearish);
    2. code checks every signal (`verify_signal`): the source exists, the quote is really in that source's text, and the quote itself
       names the company or its ticker. A signal that fails is kept, with the reason, and is not counted;
    3. code counts each STORY once per direction (`story_key`): the same article arrives through several feeds (Zacks through Polygon,
       and again through Yahoo and Finnhub) with different ids, and two quotes from one article are one voice. The extra signals are
       kept, marked `counted: false`, and do not add to the score;
    4. code calculates the score from the counted signals (`calculate_score`).

Entry 40: the Advisor also reads `data.results` (which sources are news, press releases and filings: the prompt says so) and `data.filing_evidence` (the figures and exact quotes
read from the company's stored filings). A signal may cite an evidence item, "source": "E3" (`verify_evidence_signal`): the item must exist for THAT company, the quote must be in
the item, and the direction must be the one the item's type fixes (buyback and dividend items are bullish, dilution and compensation items bearish). The quote need not name the company:
the evidence is that company's by construction. An evidence signal counts once per topic and direction, because five passages about dilution are one voice. Only an order
that carries evidence gets the evidence rules in its prompt.

The Advisor reads the CHUNKS, not the Data Clerk's summary of them. In the week 3 re-run (design/week3_evidence.md, section 3) the Data
Clerk's answers said Micron, SK Hynix and SanDisk fell a third and cited a chunk that names no company: a summary can stretch its source,
and an agent that read the summary would inherit the stretch. Check 2's last rule is the repair: a quote that does not name the company
does not count.

`model_score` is the model's own number, recorded next to the calculated one so that the two can be compared. It is never used.

WHAT THIS IS NOT. It is the smallest Advisor that lets the week 3 question be answered: "what does retrieval change in the Advisor's
recommendation and score". The real Advisor (criteria parsing, `calculate_score` over several kinds of evidence, the price and financial
signals, memory) comes with the Critic. The formula below is a stand-in that is simple enough to audit by hand.
"""
import json
import re
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prometheus.common.work_order import WorkOrder
from prometheus.models.chat import get_chat_model, is_rate_limited
from prometheus.prompts.advisor import RETRY, evidence_text, kinds_text, system_prompt, user_message
from prometheus.rag.citations import HEADER, _defuse, format_source
from prometheus.rag.loaders import check_symbol
from prometheus.rag.retriever import Hit

STANCES = ("bullish", "bearish", "neutral")
DIRECTIONS = ("bullish", "bearish")
MAX_COMPANIES = 5
MAX_SIGNALS = 3
MIN_QUOTE = 12  # characters: a shorter quote proves nothing
MAX_TEXT = 60
# Words of a company name that name nothing in particular: "Bloom Energy" is recognised by "Bloom", not by "Energy".
GENERIC = {"inc", "corp", "corporation", "technology", "technologies", "holdings", "group", "company", "systems", "energy", "electric",
           "electronics", "semiconductor", "semiconductors", "international", "limited", "platforms", "networks", "solutions"}


def normalize(text: str) -> str:
    """Lower case, straight quotes and hyphens, single spaces: the model's typography must not decide whether a quote is 'in' a source."""
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019\u02bc]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u00a0\u202f\u2009]", " ", text)
    return " ".join(text.lower().split())


NAME_SUFFIXES = {"co", "corp", "corporation", "inc", "incorporated", "ltd", "limited", "plc", "the", "company"}


def short_names(company: str | None) -> list[str]:
    """The short name a company's own name gives, when it is two words joined by "&": "Procter & Gamble Co" -> ["P&G"], "Johnson & Johnson" -> ["J&J"],
    "Barnes & Noble Inc" -> ["B&N"]. Nothing is looked up or guessed: it is the initials of the two words around the "&", in capitals, and only when the
    name is exactly those two words (after Co, Inc, Corp, Ltd, plc, The) and each is at least 3 letters. "Bath & Body Works" gives nothing (three words
    after the "&"), nor does "AT&T Inc" (its parts are 2 and 1 letters long and "AT&T" is already the name)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+|&", company or "") if w.lower() not in NAME_SUFFIXES]
    if len(words) == 3 and words[1] == "&" and words[0] != "&" and words[2] != "&" and len(words[0]) >= 3 and len(words[2]) >= 3:
        return [words[0][0].upper() + "&" + words[2][0].upper()]
    return []


def names_company(quote: str, symbol: str, company: str | None, known_company: str | None = None) -> bool:
    """The quote names the company: its ticker (as a word, in capitals), a distinctive word of its name (4 letters or more, not generic), or the short name
    the name gives ("P&G" for "Procter & Gamble Co", in capitals, as a word). `company` is what the model wrote; `known_company` is what the work order
    carries from the name lookup (design choices entry 31), and both count."""
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", quote):
        return True
    for name in (company, known_company):
        words = [w for w in re.findall(r"[A-Za-z0-9]+", name or "") if len(w) >= 4 and w.lower() not in GENERIC]
        if any(re.search(rf"(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])", quote, re.I) for w in words):
            return True
        for short in short_names(name):
            initials = re.escape(short[0]) + r"\s?&\s?" + re.escape(short[-1])
            if re.search(rf"(?<![A-Za-z0-9&]){initials}(?![A-Za-z0-9&])", quote):
                return True
    return False


EVIDENCE_ID = re.compile(r"E\d+")
DIRECTION_OF_TYPE = {"positive": "bullish", "negative": "bearish"}  # the direction an evidence item's type fixes (filing_signals.TOPIC_FILTER)


def evidence_id(signal: dict) -> str | None:
    """"E3" (or "e3", " E3 ") when the signal cites an evidence item, else None."""
    number = signal.get("source")
    return number.strip().upper() if isinstance(number, str) and EVIDENCE_ID.fullmatch(number.strip().upper()) else None


def verify_evidence_signal(signal: dict, items: dict[str, dict]) -> tuple[bool, str | None]:
    """(True, None) when a signal that cites an evidence item can be counted: the item exists for this company, the direction is the one its type fixes, and the quote is in the item
    (its quote or its `what`, which carries the figures of a derived item). `items` is {id: item} for this company's evidence."""
    item = items.get(evidence_id(signal) or "")
    if item is None:
        return False, f"evidence {signal.get('source')!r} is not one of this company's evidence items"
    if signal.get("direction") not in DIRECTIONS:
        return False, "the direction must be bullish or bearish"
    if signal["direction"] != DIRECTION_OF_TYPE.get(item.get("filter")):
        return False, f"the direction contradicts the evidence: a {item.get('filter')}-type item ({item.get('topic')}) is {DIRECTION_OF_TYPE.get(item.get('filter'))}"
    quote = signal.get("quote")
    if not isinstance(quote, str) or len(normalize(quote)) < MIN_QUOTE:
        return False, "the quote is missing or too short to check"
    if not any(normalize(quote) in normalize(item.get(field) or "") for field in ("quote", "what")):
        return False, f"the quote is not in evidence {evidence_id(signal)}"
    return True, None


def verify_signal(signal: dict, symbol: str, company: str | None, sources: dict[int, dict], known_company: str | None = None) -> tuple[bool, str | None]:
    """(True, None) when the signal can be counted, else (False, the reason)."""
    if not sources:
        return False, "no sources were provided"
    number = signal.get("source")
    if isinstance(number, str) and number.strip().isdigit():
        number = int(number)
    if not isinstance(number, int) or isinstance(number, bool) or number not in sources:
        return False, f"source {number!r} is not one of the sources shown"
    if signal.get("direction") not in DIRECTIONS:
        return False, "the direction must be bullish or bearish"
    quote = signal.get("quote")
    if not isinstance(quote, str) or len(normalize(quote)) < MIN_QUOTE:
        return False, "the quote is missing or too short to check"
    text = sources[number]["text"]
    if normalize(quote) not in normalize(text) and normalize(quote) not in normalize(_defuse(text)):
        return False, f"the quote is not in source {number}"
    if not names_company(quote, symbol, company, known_company):
        return False, "the quote does not name the company"
    return True, None


def calculate_score(bullish: int, bearish: int) -> int | None:
    """0 to 100 from the verified signals: the share that is bullish, smoothed so that one signal is not certainty
    ((bullish + 1) / (signals + 2)). 50 is balanced. None when there are no verified signals: there is nothing to calculate from.
    1 bullish: 67. 2 bullish: 75. 3 bullish: 80. 1 bearish: 33. 1 of each: 50."""
    if bullish + bearish == 0:
        return None
    return round(100 * (bullish + 1) / (bullish + bearish + 2))


def story_key(headline: str) -> str:
    """The same story through two feeds has the same headline up to case and punctuation, so that identifies it."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalize(headline)).split())


def _side(score: int) -> str:
    return "bullish" if score > 50 else "bearish" if score < 50 else "neutral"


def parse_reply(text: str) -> list:
    """The `recommendations` list from the model's reply. Raises ValueError, with a short reason, if there is none."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
        raise ValueError("no recommendations list")
    return data["recommendations"]


def build_recommendation(raw, sources: dict[int, dict], dropped: list[str], known: dict[str, str] | None = None, evidence: dict[str, dict[str, dict]] | None = None) -> dict | None:
    """One raw recommendation from the model -> a checked one with its calculated score. Returns None (and says why in `dropped`) if
    it is not usable at all. `evidence` is {symbol: {id: item}}: the filing evidence a signal may cite (entry 40)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("symbol"), str):
        dropped.append("a recommendation without a symbol")
        return None
    try:
        symbol = check_symbol(raw["symbol"].strip().lstrip("$"))
    except ValueError:
        dropped.append(f"a recommendation with an invalid symbol {raw['symbol'][:12]!r}")
        return None
    company = raw.get("company") if isinstance(raw.get("company"), str) else None
    company = company.strip()[:MAX_TEXT] if company else None
    stance = raw["stance"].lower() if isinstance(raw.get("stance"), str) and raw["stance"].lower() in STANCES else None
    model_score = raw.get("model_score")
    model_score = round(model_score) if isinstance(model_score, (int, float)) and not isinstance(model_score, bool) and 0 <= model_score <= 100 else None
    items = (evidence or {}).get(symbol) or {}
    signals, seen, taken = [], set(), {"source": 0, "evidence": 0}
    for signal in (raw.get("signals") if isinstance(raw.get("signals"), list) else []):
        if not isinstance(signal, dict):
            taken["source"] += 1  # it still uses up a place, as when only the first MAX_SIGNALS entries were read
            continue
        kind = "evidence" if items and evidence_id(signal) else "source"  # without evidence for this company an "E3" is an ordinary (failing) source number, as it always was
        if taken[kind] >= MAX_SIGNALS:
            continue
        taken[kind] += 1
        key = (str(signal.get("source")), signal.get("direction"), normalize(str(signal.get("quote", ""))))
        if key in seen:
            continue
        seen.add(key)
        number = signal.get("source") if isinstance(signal.get("source"), (int, str)) else None
        if kind == "evidence":
            ok, reason = verify_evidence_signal(signal, items)
            item = items[evidence_id(signal)] if ok else None
            signals.append({"source": number, "chunk_id": f"evidence:{item['source']['accession']}:{item['id']}" if item else None, "direction": signal.get("direction"),
                            "quote": str(signal.get("quote", ""))[:300], "verified": ok, "reason": reason, "evidence_id": evidence_id(signal), "topic": item["topic"] if item else None})
            continue
        ok, reason = verify_signal(signal, symbol, company, sources, (known or {}).get(symbol))
        source = sources.get(int(number)) if ok else None
        signals.append({"source": number, "chunk_id": source["chunk_id"] if source else None, "direction": signal.get("direction"),
                        "quote": str(signal.get("quote", ""))[:300], "verified": ok, "reason": reason})
    counted_stories = set()
    for s in signals:  # a story counts once per direction: a syndicated copy, or a second quote from the same article, adds nothing. Evidence counts once per topic and direction
        s["counted"] = False
        if s["verified"]:
            key = (("evidence", s["topic"]) if s.get("evidence_id") else story_key(sources[int(s["source"])]["headline"]), s["direction"])
            if key in counted_stories:
                s["note"] = "not counted: the same " + ("topic" if s.get("evidence_id") else "story") + " as an earlier signal in the same direction"
            else:
                counted_stories.add(key)
                s["counted"] = True
    bullish = sum(s["counted"] and s["direction"] == "bullish" for s in signals)
    bearish = sum(s["counted"] and s["direction"] == "bearish" for s in signals)
    score = calculate_score(bullish, bearish)
    return {"symbol": symbol, "company": company, "stance": stance, "model_score": model_score, "score": score,
            "supported": score is not None, "consistent": None if score is None or stance is None else stance == _side(score),
            "bullish_signals": bullish, "bearish_signals": bearish, "signals_claimed": len(signals),
            "signals_verified": sum(s["verified"] for s in signals), "signals_counted": bullish + bearish, "signals": signals}


class Advisor:
    def __init__(self, model=None, model_id: str | None = None):
        """model: a chat model (anything with `invoke`; a scripted one in tests); otherwise OpenRouter through `get_chat_model` with the
        model of the `advisor` role (ADVISOR_MODEL), or `model_id`."""
        self.model = model or get_chat_model("advisor", model=model_id)

    @staticmethod
    def _evidence(order: WorkOrder) -> dict[str, dict[str, dict]]:
        """The filing evidence a signal may cite: {symbol: {"E3": item}}, from `data.filing_evidence` (written by the Data Clerk's code, entry 39 and 40). Empty when there is none."""
        if not order.data:
            return {}
        return {symbol: {item["id"]: item for item in ev.get("evidence", []) if item.get("id")} for symbol, ev in order.data.filing_evidence.items()}

    @staticmethod
    def _sources(order: WorkOrder) -> tuple[dict[int, dict], str | None]:
        """The chunks the Data Clerk found, by number, and the text the model reads (formatted exactly as the Data Clerk's model saw them)."""
        if not order.data or not order.data.sources:
            return {}, None
        by_number = {s["number"]: s for s in order.data.sources}
        fields = Hit.__dataclass_fields__
        blocks = [format_source(n, Hit(**{k: v for k, v in s.items() if k in fields})) for n, s in sorted(by_number.items())]
        return by_number, HEADER + "\n\n" + "\n\n".join(blocks)

    def advise(self, order: WorkOrder, use_evidence: bool = True, today: date | None = None, bare: bool = False) -> WorkOrder:
        """Append recommendations to the order and return the SAME order. With `use_evidence` (the normal case) the order must be
        `data_ready`. `use_evidence=False` gives the Advisor no sources at all: a baseline for "what does retrieval change". `bare=True`
        swaps in the ordinary prompt-only prompt ("recommend as well as you can"), the other baseline; the desk never uses it."""
        if use_evidence and order.status != "data_ready":
            order.errors = [*order.errors, f"the Advisor needs an order whose data is ready (status {order.status})"]
            return order
        sources, text = self._sources(order) if use_evidence else ({}, None)
        items = self._evidence(order) if use_evidence and not bare else {}
        messages = [SystemMessage(content=system_prompt(today, bare, evidence=bool(items))),
                    HumanMessage(content=user_message(order, text, kinds_text(order) if use_evidence and not bare else None, evidence_text(order) if items else None))]
        run = {"model": getattr(self.model, "model_name", None), "evidence": use_evidence, "sources": len(sources), "filing_evidence": sorted(items), "tokens_in": 0,
               "tokens_out": 0, "calls": 0, "dropped": [], "error": None, "bare_prompt": bare}
        raws = None
        for attempt in range(2):  # one retry if the reply is not the JSON we asked for
            try:
                reply = self.model.invoke(messages)
            except Exception as error:  # a model or network failure: say what kind, never print request details
                run["error"], run["rate_limited"] = f"the advisor model call failed ({type(error).__name__})", is_rate_limited(error)
                break
            run["calls"] += 1
            usage = getattr(reply, "usage_metadata", None) or {}
            run["tokens_in"] += usage.get("input_tokens", 0)
            run["tokens_out"] += usage.get("output_tokens", 0)
            try:
                raws = parse_reply(reply.content if isinstance(reply.content, str) else "")
                break
            except ValueError as problem:
                run["error"] = f"the advisor's reply was not usable ({problem})"
                messages = [*messages, AIMessage(content=reply.content if isinstance(reply.content, str) else ""),
                            HumanMessage(content=RETRY.format(problem=problem))]
        order.advisor_runs.append(run)
        if raws is None:
            order.errors = [*order.errors, run["error"]]
            return order
        run["error"] = None
        known = {order.subject[0]: order.company} if order.company and isinstance(order.subject, list) and len(order.subject) == 1 else {}
        recommendations = [r for r in (build_recommendation(raw, sources, run["dropped"], known, items) for raw in raws[:MAX_COMPANIES]) if r]
        if len(raws) > MAX_COMPANIES:
            run["dropped"].append(f"{len(raws) - MAX_COMPANIES} recommendation(s) beyond the first {MAX_COMPANIES}")
        for r in recommendations:
            r["advisor"] = {"model": run["model"], "evidence": use_evidence}
        order.recommendations = [*order.recommendations, *recommendations]
        order.status = "recommended"
        return order
