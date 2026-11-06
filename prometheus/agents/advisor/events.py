"""Corporate events: the taxonomy, and the code that checks a model's claim that a quote shows one (Advisor v1 spec section 8.4, design choices entry 51).

The model may SAY "this sentence is a guidance raise" (plan call, `events` list). Code decides whether it is: `verify_event` accepts a class only when

  1. the class is in the taxonomy below and is a TEXT class (the two filing classes come from the filing index, see `filing_index_events`),
  2. the source number exists and the quote is really in that source (as v0's `verify_signal` checks it),
  3. the quote itself names the company (v0's `names_company`): "Micron beat estimates" is not evidence about Sandisk,
  4. the class's own patterns match the QUOTE: every regex of `all_of` and none of `none_of` (so "did not raise guidance" is not a raise, and "intends to increase its dividend" is not
     a dividend increase),
  5. the direction the model gave, if any, is the one the class fixes. The direction is never the model's choice.

This is the same idea as `filing_signals.TOPIC_PATTERNS`: a fixed table of words decides, the model only points. A pattern is deliberately narrow: a missed event is a gap, a false one is
a wrong trade. The table is tested against real sentences of the stored news, press releases and filings (tests/test_events.py), and a near miss for each class.

Direction: a class is bullish, bearish, or neutral (a cause that explains a fall without being news about the company: reversion uses it). The horizon is in bars (sessions).
"""
import re
from dataclasses import dataclass, field

from prometheus.agents.advisor.advisor import MIN_QUOTE, names_company, normalize
from prometheus.rag.citations import _defuse

HEDGE = r"\b(?:could|may|might|would|if)\b"  # a sentence that only says what could happen is not an event


@dataclass(frozen=True)
class EventClass:
    name: str
    direction: str                              # bullish | bearish | neutral
    horizon: int | None                         # bars; None for a neutral class (it is a cause, not a trade)
    all_of: tuple[str, ...] = ()                # every one of these regexes must match the quote
    none_of: tuple[str, ...] = ()               # none of these may
    kind: str = "text"                          # "text" (a quote) or "filing_index" (a filing's form and date, no text needed)
    description: str = ""


def _c(name, direction, horizon, all_of=(), none_of=(), kind="text", description=""):
    return EventClass(name, direction, horizon, tuple(all_of), tuple(none_of), kind, description)


# The words. A verb and its object are required to be NEAR each other in ONE clause (`[^.;]{0,50}`): "a 4% increase in organic sales ... while maintaining its full-year outlook" is not a
# raise of the outlook, and "history of missed forecasts raises skepticism" is not a missed estimate. They were widened and narrowed against the real stored text (tests/test_events.py).
GUIDANCE = r"(?:guidance|outlook|forecasts?|financial targets?)"
ESTIMATES = r"(?:estimates?|expectations?|consensus)"                      # not "forecasts": a person's forecast being missed is not a result missing the analysts'
RESULTS_CONTEXT = r"\b(?:earnings|revenues?|sales|eps|profits?|results|quarter|income)\b"
PREDICTION = r"\b(?:expects?|expected|expect|projected?|poised|likely|anticipat\w+|hopes?|should|will)\b[^;]{0,40}\b(?:beat|exceed|top|surpass|miss|fall short)"  # a forecast of a beat is not a beat
NEGATED = lambda verb: rf"\b(?:did not|didn't|does not|doesn't|not|no|never|without)\b[^.;]{{0,20}}\b{verb}"  # "did not raise guidance"
FUTURE = r"\b(?:intends?|plans?|expects?|may|might|could|would|subject to|required to|forced to|if)\b[^.;]{0,25}\b(?:rais|increas|hik|boost|cut|reduc|suspend)"

EVENT_CLASSES: dict[str, EventClass] = {c.name: c for c in (
    # bullish, text
    _c("guidance_raise", "bullish", 40, [rf"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|hik(?:e|es|ed|ing)|upgrad(?:e|es|ed|ing))\b[^.;]{{0,50}}\b{GUIDANCE}\b"
                                         rf"|\b{GUIDANCE} (?:raise|hike|upgrade)\b"], [NEGATED("(?:rais|lift|boost|hik)"), HEDGE], description="the company raises its own guidance or outlook"),
    _c("earnings_beat", "bullish", 40, [rf"\b(?:beat\w*|topped|tops|exceed(?:s|ed)?|surpass(?:es|ed)?|outpac\w+)\b[^.;]{{0,35}}\b{ESTIMATES}\b|\bearnings (?:beat|surprise)\b", RESULTS_CONTEXT],
       [HEDGE, PREDICTION], description="results beat analysts' estimates or expectations"),
    _c("buyback_authorization", "bullish", 40, [r"\b(?:(?:share|stock) (?:repurchase|buyback)|buyback|repurchase program)\b", r"\$\s?[\d.,]+\s?(?:billion|million)",
                                                r"\b(?:announc\w+|authoriz\w+|approv\w+|launch\w+|new|additional|expand\w+|increas\w+)\b"], [HEDGE],
       description="a new share repurchase programme with a dollar amount"),
    _c("dividend_increase", "bullish", 40, [r"\b(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|hik(?:e|es|ed|ing)|boost(?:s|ed|ing)?)\b[^.;]{0,40}\bdividend|\bdividend (?:increase|hike|raise)\b|\bdividend (?:was |is |has been )?(?:raised|increased|hiked|boosted)\b"],
       [FUTURE, r"\b(?:every (?:single )?year|each year|consecutive|for \d+ (?:straight |consecutive )?years?|for (?:seven )?decades?|streak)\b"], description="the dividend is raised (not a history of raises)"),
    _c("dividend_initiation", "bullish", 40, [r"\b(?:initiat(?:e|es|ed|ing|ion)|institut(?:e|es|ed|ing)|begin\w* (?:paying|to pay)|start\w* (?:paying|to pay))\b[^.;]{0,40}\bdividend|\bfirst[- ](?:ever )?(?:quarterly |cash )?dividend\b"],
       [HEDGE], description="the company starts paying a dividend"),
    _c("activist_stake_13d", "bullish", 40, kind="filing_index", description="a Schedule 13D (an investor over 5% who may seek influence), from the filing index"),
    # bearish, text
    _c("guidance_cut", "bearish", 40, [rf"\b(?:cut\w*|lower(?:s|ed|ing)?|slash\w*|reduc(?:e|es|ed|ing)|trim\w*|withdr\w+|warn\w*|downgrad\w+)\b[^.;]{{0,50}}\b{GUIDANCE}\b"
                                       rf"|\b{GUIDANCE} (?:cut|reduction|downgrade|warning)\b|\bprofit warning\b"], [NEGATED("(?:cut|lower|reduc|trim)"), HEDGE, r"\b(?:lower|low) end\b|\blower[- ]than\b|\b(?:unchanged|maintain\w*|reaffirm\w*)\b"], description="the company cuts or withdraws its guidance"),
    _c("earnings_miss", "bearish", 40, [rf"\b(?:miss(?:es|ed)?|fell short of|falls? short of|came in (?:below|under)|below)\b[^.;]{{0,35}}\b{ESTIMATES}\b|\bearnings miss\b", RESULTS_CONTEXT],
       [HEDGE, PREDICTION], description="results miss analysts' estimates or expectations"),
    _c("dividend_cut", "bearish", 40, [r"\b(?:cut\w*|reduc(?:e|es|ed|ing)|suspend\w*|eliminat\w+|slash\w*|omit\w*)\b[^.;]{0,30}\bdividend|\bdividend (?:cut|suspension|reduction)\b"], [HEDGE, FUTURE],
       description="the dividend is cut or suspended"),
    _c("dilution_offering", "bearish", 40, [r"\b(?:public offering|secondary offering|at[- ]the[- ]market|registered direct|follow-on offering|convertible (?:senior )?notes?|equity offering|stock offering|private placement)\b",
                                            r"\b(?:announc\w+|priced|pricing|launch\w+|commenc\w+|propos\w+|filed|entered|sold|issu\w+)\b"], [r"\binitial public offering\b", HEDGE, r"\b(?:scrap\w*|cancel\w*|withdr\w+)\b"],
       description="the company sells new shares or convertible debt"),
    _c("ceo_departure_unplanned", "bearish", 40, [r"\b(?:ceo|chief executive(?: officer)?)\b", r"\b(?:resign\w*|steps? down|stepped down|depart\w*|exit\w*|ousted|fired|terminated|removed)\b",
                                                  r"\b(?:abrupt\w*|sudden\w*|unexpected\w*|immediate\w*|ousted|fired|terminated|removed|without notice)\b"], [HEDGE],
       description="the chief executive leaves abruptly"),
    _c("material_weakness", "bearish", 40, [r"\bmaterial weakness(?:es)?\b"], [HEDGE, r"\b(?:no material weakness|not identified|no longer)\b"], description="a material weakness in internal control is reported"),
    _c("restatement", "bearish", 40, [r"\b(?:restate\w*|restatement)\b", r"\b(?:financial statements|previously issued|results|earnings|non-reliance)\b"],
       [r"\bamended and restated\b", r"\b(?:plan|agreement|charter|bylaws|certificate)\b", HEDGE], description="previously issued financial statements are restated"),
    _c("going_concern", "bearish", 40, [r"\bsubstantial doubt\b", r"\bgoing concern\b"], [HEDGE, r"\b(?:no substantial doubt|alleviat\w+)\b"], description="doubt about the company's ability to continue"),
    _c("form_144_sale_notice", "bearish", 40, kind="filing_index", description="an insider's Form 144 notice of a planned sale, from the filing index"),
    # neutral: a cause, not a trade (reversion reads these as "the fall was not about the company")
    _c("analyst_rating_change", "neutral", None, [r"\b(?:upgrad\w+|downgrad\w+|initiat(?:e|es|ed|ing) (?:coverage|with)|price targets?|reiterat\w+|overweight|underweight|outperform|underperform)\b",
                                                  r"\b(?:analysts?|ratings?|price targets?|coverage)\b"], [], description="an analyst changes a rating or a target"),
    _c("sector_sympathy", "neutral", None, [r"\b(?:sympathy|sector[- ]wide|industry[- ]wide|read[- ]through|knock-on|contagion)\b"
                                            r"|\b(?:peer|rival|competitor)(?:'s|s')?\b[^.;]{0,40}\b(?:results|earnings|guidance|warning|plunge\w*|fell|drop\w*|slump\w*|sell-?off)\b"], [], description="the stock moved with its sector or a peer"),
    _c("macro", "neutral", None, [r"\b(?:fed|federal reserve|interest rates?|rate (?:cut|hike)|inflation|cpi|jobs report|tariffs?|treasury yields?|recession|shutdown|geopolitical|oil prices?)\b"], [], description="a market-wide macro cause"),
    _c("lawsuit_headline", "neutral", None, [r"\b(?:lawsuit|sued|sues|class[- ]action|securities fraud|subpoena|investigation|probe|antitrust|litigation)\b"], [HEDGE], description="a lawsuit or investigation headline"),
    _c("index_flow", "neutral", None, [r"\bindex (?:rebalanc\w+|reconstitution|inclusion|deletion)\b|\b(?:S&P 500|Nasdaq-100|Russell \d+|MSCI)\b[^.;]{0,40}\b(?:inclusion|added|joins|removed|rebalanc\w+)\b"
                                       r"|\b(?:added to|removed from|deleted from|joins?)\b[^.;]{0,30}\b(?:S&P|Nasdaq-100|Russell|MSCI)\b"], [], description="a flow caused by an index change"),
)}
TEXT_CLASSES = tuple(n for n, c in EVENT_CLASSES.items() if c.kind == "text")
BULLISH = tuple(n for n, c in EVENT_CLASSES.items() if c.direction == "bullish")
BEARISH = tuple(n for n, c in EVENT_CLASSES.items() if c.direction == "bearish")
NEUTRAL = tuple(n for n, c in EVENT_CLASSES.items() if c.direction == "neutral")
FILING_INDEX_FORMS = {"form_144_sale_notice": re.compile(r"^Form 144(?:/A)?\b"), "activist_stake_13d": re.compile(r"^Form SCHEDULE 13D\b(?!/A)")}  # the first words of a filing Document that has no text (loaders._filing_to_documents)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$(\"'])")
AMOUNT = re.compile(r"\$\s?([\d.,]+)\s?(billion|million)", re.I)


def class_matches(name: str, quote: str) -> bool:
    """Whether the class's patterns match this quote: every `all_of`, no `none_of`. A filing-index class never matches text."""
    cls = EVENT_CLASSES.get(name)
    if cls is None or cls.kind != "text" or not quote:
        return False
    return all(re.search(p, quote, re.I) for p in cls.all_of) and not any(re.search(p, quote, re.I) for p in cls.none_of)


def match_events(text: str) -> list[dict]:
    """Every text class that matches a sentence of `text`: [{class, direction, quote}], the first matching sentence of each class, in the order of the taxonomy. The code-side reading of a text,
    used to test the table against real sentences; the Advisor's events are the model's claims checked by `verify_event`."""
    sentences = [s.strip() for s in SENTENCE.split(" ".join(text.split())) if s.strip()]
    found = []
    for name in TEXT_CLASSES:
        quote = next((s for s in sentences if class_matches(name, s)), None)
        if quote:
            found.append({"class": name, "direction": EVENT_CLASSES[name].direction, "quote": quote[:300]})
    return found


def verify_event(event: dict, symbol: str, company: str | None, sources: dict[int, dict], known_company: str | None = None) -> tuple[bool, str | None]:
    """(True, None) when a model's event claim {class, source, quote, direction?} can be counted, else (False, the reason). See the module text for the five checks."""
    cls = EVENT_CLASSES.get(event.get("class")) if isinstance(event.get("class"), str) else None
    if cls is None:
        return False, f"{event.get('class')!r} is not an event class"
    if cls.kind != "text":
        return False, f"{cls.name} comes from the filing index, not from a quote"
    number = event.get("source")
    if isinstance(number, str) and number.strip().isdigit():
        number = int(number)
    if not isinstance(number, int) or isinstance(number, bool) or number not in sources:
        return False, f"source {number!r} is not one of the sources shown"
    quote = event.get("quote")
    if not isinstance(quote, str) or len(normalize(quote)) < MIN_QUOTE:
        return False, "the quote is missing or too short to check"
    text = sources[number]["text"]
    if normalize(quote) not in normalize(text) and normalize(quote) not in normalize(_defuse(text)):
        return False, f"the quote is not in source {number}"
    if not names_company(quote, symbol, company, known_company):
        return False, "the quote does not name the company"
    if not class_matches(cls.name, quote):
        return False, f"the quote does not show {cls.name} (its words are not there, or it only says what could happen)"
    if event.get("direction") not in (None, cls.direction):
        return False, f"the direction contradicts the class: {cls.name} is {cls.direction}"
    return True, None


def filing_index_events(symbol: str, sources: list[dict]) -> list[dict]:
    """The events of the filing INDEX: a Form 144 (an insider plans to sell) or a Schedule 13D (an investor over 5% who may seek influence) among the filing sources of this symbol,
    straight from the form and the date: no model and no quote needed. `sources` are the order's filing sources ({number, text, tickers, published_utc, ...})."""
    out = []
    for source in sources:
        if symbol not in (source.get("tickers") or []):
            continue
        for name, pattern in FILING_INDEX_FORMS.items():
            if pattern.search((source.get("text") or "").strip()):
                out.append({"class": name, "direction": EVENT_CLASSES[name].direction, "source": source.get("number"), "quote": (source["text"].strip().split(". ")[0])[:200],
                            "published_utc": source.get("published_utc"), "origin": "filing_index"})
    return out


def last_earnings_release(symbol: str, press_sources: list[dict]) -> str | None:
    """The date (YYYY-MM-DD) of the newest press release of `symbol` that reports results ("announces ... results", "reports earnings"), or None. Used only to ESTIMATE when the next
    report is due when the earnings calendar is not available (the event-risk gate says so in its note)."""
    dates = [s["published_utc"][:10] for s in press_sources
             if symbol in (s.get("tickers") or []) and re.search(r"\b(?:announces?|reports?|reported)\b[^.]{0,80}\b(?:results|earnings)\b", (s.get("headline") or "") + " " + (s.get("text") or "")[:300], re.I) and s.get("published_utc")]
    return max(dates) if dates else None


def amount_of(quote: str) -> float | None:
    """The dollar amount a quote states ("$20 billion" -> 20e9), the first one; None when there is none."""
    found = AMOUNT.search(quote or "")
    if not found:
        return None
    return float(found.group(1).replace(",", "").rstrip(".")) * (1e9 if found.group(2).lower() == "billion" else 1e6)


def timing(published_utc: str | None, recent: list[list]) -> dict | None:
    """When an event happened against the stored bars. `recent` is the `recent_bars` metric: [[date, close, volume], ...] oldest first. Returns
    {age_bars, reaction, event_volume_ratio, bar_date}. The EVENT BAR is the first bar on or after the day the source was published (a weekend item: Monday's); `age_bars` is the number
    of bars after it (0: published on or after the last bar, nothing has traded since); `reaction` is the last close over the close of the bar BEFORE the event bar, minus 1 (an item
    published after the close counts that day's move as its reaction: conservative, it makes a move look more priced in); `event_volume_ratio` is the event bar's volume over the mean of the
    up to 20 bars before it. An event older than the bars has `age_bars == len(recent)` (at least that old) and no reaction. None when the date or the bars are missing."""
    if not published_utc or len(recent) < 2:
        return None
    day = published_utc[:10]
    if day < recent[0][0]:
        return {"age_bars": len(recent), "reaction": None, "event_volume_ratio": None, "bar_date": None}
    index = next((i for i, bar in enumerate(recent) if bar[0] >= day), None)
    if index is None:
        return {"age_bars": 0, "reaction": recent[-1][1] / recent[-1][1] - 1, "event_volume_ratio": None, "bar_date": None}
    out = {"age_bars": len(recent) - 1 - index, "reaction": None, "event_volume_ratio": None, "bar_date": recent[index][0]}
    if index >= 1 and recent[index - 1][1] > 0:
        out["reaction"] = recent[-1][1] / recent[index - 1][1] - 1
    prior = [bar[2] for bar in recent[max(0, index - 20):index]]
    if len(prior) >= 3 and sum(prior) > 0:
        out["event_volume_ratio"] = recent[index][2] / (sum(prior) / len(prior))
    return out
