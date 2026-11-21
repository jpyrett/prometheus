"""Evidence: do the quotes say what the Advisor says they say? And is there newer news it did not see? (Critic v1 spec section 8.2 and 11, step 5.)

Code proved each quote exists and names the company. Whether it is good news, and whether it is about this company's prospects at all, nothing in the code can see: that is the model's one job,
and a model of ANOTHER producer than the Advisor's does it (so the reviewer does not simply agree with text a model of its own family wrote).

    result = assess(model, rec, claims, source_checks, verification, order, advisor_config, config, now, critic_model, independent, run)

ONE call per record. The model labels each supporting claim (supports / weak / unrelated / contradicts) from the source text as the Data Clerk read it again today, and each FRESH source (stored
news and press releases the Advisor never saw) as bearish (with an exact quote) or none. Then code:

  * checks the reply: a claim it was not asked about is dropped, a missing label is `unreviewed`, a "bearish" label counts only when its quote is really in that source and names the company;
  * reads the fresh sources itself (`events.match_events`): a bearish event class in a sentence that names the company is a finding whatever the model says;
  * plants ONE control claim whose right label code knows (an unrelated notice): a reply that does not call it unrelated did not read the text, and is thrown away (spec 8.2);
  * calculates the expected return again WITHOUT the quotes that did not hold (`counterfactual`). It can only be lower than the Advisor's.

THE MODEL NEVER DECIDES: it labels, and every label can only LOWER a result. It sees no score, no thesis, no stance reasoning and nothing about the user.

CRITIC v2 (`design/critic_v2_evidence_spec.md`). Three rules of this step changed, each because the outcomes of four six-month tests said a reason did not earn its place:
  C2  `fresh_negative_news` (the model's label of a newer article, or a bad word of the text with no blocking class) INFORMS: it stays a warning and no longer asks for a person, so it no longer denies. `fresh_negative_event`
      (going concern, restatement, material weakness, guidance cut, earnings miss, dilution, dividend cut: code finds it in the text) is still a blocker.
  C3  `blend_not_supported` is a WARNING. The blend worked again without the quotes that did not hold is still calculated and shown, the evidence dimension still loses its points, and the review still says so.
  C6  The same quote gets the same label: a label is kept in the label store (`labels.py`, `evidence.label_store`) and read back, so the model is asked about a quote once; a call with nothing left to ask is not made. The control claim
      still runs in every call that is made, and only a reply whose control was read right is kept.
"""
import re
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime

from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.agents.advisor.advisor import names_company, story_key
from prometheus.agents.advisor.calls import ask_object, new_run
from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.events import BEARISH, SENTENCE as EVENT_SENTENCE, TEXT_CLASSES, class_matches   # the Advisor's own splitter, so a sentence is cut exactly where the event matcher cut it (this module's `SENTENCE` below is another, simpler one)
from prometheus.agents.advisor.record import Recommendation
from prometheus.agents.critic.claims import Claim, stance_direction, supporting
from prometheus.agents.critic.common import blocker, info, producer, quote_in_text, warning
from prometheus.agents.critic.config import CriticConfig
from prometheus.agents.critic.labels import LabelStore, prompt_version
from prometheus.agents.critic.record import ClaimLabel, Control, Counterfactual, EvidenceDetail, Finding, FreshItem, Measure, SourceCheck
from prometheus.common.market_calendar import eastern_date
from prometheus.common.work_order import WorkOrder, corpus_of
from prometheus.prompts.critic import RETRY_REVIEW, review_message, review_system

REAL_LABELS = ("supports", "weak", "unrelated", "contradicts")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class EvidenceResult:
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    measures: list[Measure] = field(default_factory=list)
    detail: EvidenceDetail = field(default_factory=EvidenceDetail)
    called: bool = False


# ---------------------------------------------------------------- what to show the model

def seed_of(record_id: str) -> int:
    """A number made from the record's id, so that the control claim's text and place are the same on every run of the same record (the id is a hex string; anything else is hashed)."""
    try:
        return int(record_id, 16)
    except ValueError:
        return zlib.crc32(record_id.encode())


def excerpt(text: str, quote: str, limit: int) -> str:
    """At most `limit` characters of a source, around the quote (the head when the quote cannot be located). A text that fits is returned whole."""
    if len(text) <= limit:
        return text
    at = text.lower().find(quote.lower()[:80])
    start = 0 if at < 0 else max(0, min(at - (limit - len(quote)) // 2, len(text) - limit))
    return ("... " if start else "") + text[start:start + limit].strip() + (" ..." if start + limit < len(text) else "")


def control_claim(rec: Recommendation, order: WorkOrder, config: CriticConfig, direction: str, today: date) -> tuple[Claim, str, str]:
    """(the control claim, its source text, its heading): a quote that has nothing to do with the company, given a ref of the ordinary form ([N], N one more than the highest source on the order)."""
    seed = seed_of(rec.identity.record_id)
    notice = config.evidence.control_texts[seed % len(config.evidence.control_texts)]
    top = max([s.get("number") for s in (order.data.sources if order.data else []) if isinstance(s.get("number"), int)] + [0])
    ref = f"[{top + 1}]"
    return Claim(ref=ref, kind="signal", direction=direction, quote=SENTENCE.split(notice.text.strip())[0], source=top + 1), notice.text.strip(), f"{notice.headline}, {today.isoformat()}"


def _claim_source(c: Claim, rec: Recommendation, verification: dict, config: CriticConfig) -> tuple[str, str]:
    """(the heading, the text) the model is shown for one claim: the source as the Data Clerk read it again today, cut to the part around the quote."""
    if c.kind == "evidence":
        found = ((verification.get("evidence") or {}).get(rec.symbol) or {}).get(c.evidence_id) or {}
        return "SEC filing evidence", excerpt(f"{found.get('what') or ''}\nQuote: {found.get('quote') or ''}".strip(), c.quote, config.limits.max_source_chars)
    found = (verification.get("chunks") or {}).get(c.chunk_id) or {}
    return _heading(found), excerpt(found.get("text") or "", c.quote, config.limits.max_source_chars)


def fresh_candidates(rec: Recommendation, order: WorkOrder, verification: dict, config: CriticConfig) -> list[dict]:
    """The newest stored text of the symbol that the Advisor did not see: not a source of the order (by article id) and not a copy of one (by the story of its headline), newest first."""
    on_order = order.data.sources if order.data else []
    articles, stories = {s.get("article_id") for s in on_order}, {story_key(s.get("headline") or "") for s in on_order}
    new = [h for h in (verification.get("fresh") or {}).get(rec.symbol, []) if h.get("article_id") not in articles and story_key(h.get("headline") or "") not in stories]
    return new[:config.limits.max_fresh]


# One company, one clause: "Procter & Gamble declared its dividend, while PepsiCo cut its outlook" says P&G declared a dividend and PepsiCo cut its outlook. A sentence is cut at these words.
CLAUSE_BREAK = re.compile(r",?\s+(?:while|whereas|but|although|though)\s+|;\s*", re.I)


def about_the_company(name: str, sentence: str, symbol: str, company: str | None) -> bool:
    """Whether a bearish class that matches `sentence` is about THIS company. `names_company` alone only says the company is somewhere in the sentence, and a sentence about two companies names both: "Procter & Gamble
    declared its dividend, while PepsiCo cut its 2026 outlook" matched `guidance_cut` and blocked a Procter & Gamble purchase for PepsiCo's news (found in the first live run of 2026-11-14). So when the sentence has
    more than one clause, the clause that SHOWS the class must name the company. When no single clause shows it (the words of the pattern are spread over several), the old rule stands: the sentence names the company."""
    clauses = [c.strip() for c in CLAUSE_BREAK.split(sentence) if c.strip()]
    if len(clauses) > 1:
        shown = [c for c in clauses if class_matches(name, c)]
        if shown:
            return any(names_company(c, symbol, company) for c in shown)
    return names_company(sentence, symbol, company)


def bearish_classes(text: str, symbol: str, company: str | None) -> list[str]:
    """The bearish event classes CODE finds in a text, in a sentence that names the company and is about it (`about_the_company`; the patterns are the Advisor's: `events.class_matches`). Every sentence is
    looked at, not only the first one that matches a class: a cut by another company in the first sentence must not hide this company's own cut in a later one. In the order of the taxonomy."""
    sentences = [s.strip() for s in EVENT_SENTENCE.split(" ".join(text.split())) if s.strip()]
    return [name for name in TEXT_CLASSES if name in BEARISH and any(class_matches(name, s) and about_the_company(name, s, symbol, company) for s in sentences)]


# ---------------------------------------------------------------- reading the reply

def read_reply(reply: dict, asked: set[str], fresh: dict[str, dict], symbol: str, company: str | None, max_words: int, dropped: list[str]) -> tuple[dict[str, tuple[str, str | None]], dict[str, tuple[str, str | None]]]:
    """(labels {ref: (label, reason)}, fresh labels {id: ("bearish" | "none", quote)}) from the model's reply, after the checks. A ref that was not asked and a label that is not on the list are dropped;
    a "bearish" without a quote that is in that source and names the company becomes "none"."""
    labels: dict[str, tuple[str, str | None]] = {}
    for item in reply.get("claims") if isinstance(reply.get("claims"), list) else []:
        ref = str(item.get("ref", "")).strip() if isinstance(item, dict) else ""
        if ref not in asked:
            dropped.append(f"a claim label for {ref or 'no ref'!r}, which was not asked")
            continue
        label, reason = str(item.get("label", "")).strip().lower(), item.get("reason")
        if label not in REAL_LABELS:
            dropped.append(f"{ref}: {label!r} is not a label")
            continue
        labels.setdefault(ref, (label, " ".join(str(reason).split()[:max_words]) if isinstance(reason, str) and reason.strip() else None))
    marks: dict[str, tuple[str, str | None]] = {}
    for item in reply.get("fresh") if isinstance(reply.get("fresh"), list) else []:
        key = str(item.get("id", "")).strip() if isinstance(item, dict) else ""
        if key not in fresh:
            dropped.append(f"a label for the fresh source {key or 'no id'!r}, which was not shown")
            continue
        if str(item.get("label", "")).strip().lower() != "bearish":
            marks.setdefault(key, ("none", None))
            continue
        quote = item.get("quote")
        if isinstance(quote, str) and quote_in_text(quote, fresh[key].get("text")) and names_company(quote, symbol, company):
            marks.setdefault(key, ("bearish", quote[:300]))
        else:
            dropped.append(f"{key}: bearish without a quote that is in the source and names the company: ignored")
            marks.setdefault(key, ("none", None))
    return labels, marks


# ---------------------------------------------------------------- the blend without the quotes that did not hold

def _claim_key(base: str, index: int) -> str:
    """The claim ref for the `index`-th blend row that shares `base`. The claim list keeps the first quote as `base` and numbers the later quotes from the same source `base#2`, `base#3`, ..."""
    return base if index == 0 else f"{base}#{index + 1}"


def counterfactual(rec: Recommendation, claims: list[Claim], labels: dict[str, str], untagged: set[str], fresh_bearish: list[tuple[float, int]], advisor: AdvisorConfig,
                   config: CriticConfig) -> Counterfactual | None:
    """The record's blend, worked again from its own parts with the review's labels applied (spec 8.2 steps 1 to 5; Amendment 3 removed the expected return after cost). `labels` is {ref: label} of the
    supporting claims; `untagged` the refs whose source does not list the symbol (they count for nothing); `fresh_bearish` is [(kind weight, age in days)] of the newer bad news."""
    blend = rec.blend
    if not blend:
        return None
    comps, cfg, weights = blend["components"], advisor.blend, advisor.blend.weights
    weight_of = config.evidence.label_weight
    # 1 and 2: the news part, with each supporting quote's weight scaled by its label, and the newer bad news added. Two quotes from one source are two rows with the same ref on the blend and two
    # claim refs (`[4]` and `[4]#2`): each row takes its own label, in that order.
    votes: list[tuple[int, float]] = []
    seen: dict[str, int] = {}
    for p in comps["news"]["parts"]:
        sign, weight = (1 if p["direction"] == "bullish" else -1), p["weight"]
        ref, index = p["ref"], seen.get(p["ref"], 0)
        seen[ref] = index + 1
        key = _claim_key(ref, index)
        label = labels.get(key)
        if key in untagged:
            weight = 0.0
        elif label == "contradicts":
            sign = -sign                                   # it says the opposite: it votes the other way, with its full weight
        elif label is not None:
            weight *= weight_of[label]
        votes.append((sign, weight))
    votes += [(-1, kind_weight * 0.5 ** (age / cfg.half_life_days)) for kind_weight, age in fresh_bearish]
    # 3
    total = sum(w for _, w in votes)
    news = sum(s * w for s, w in votes) / (total + cfg.news_prior_n) if votes and total + cfg.news_prior_n > 0 else 0.0
    # 4: the filings part: an evidence vote that is weak counts as much as its label weight, one that is unrelated or contradicted is removed
    kept = []
    seen = {}
    for p in comps["filings"]["parts"]:
        ref = p.get("ref")
        if not ref:
            label = None
        else:
            index = seen.get(ref, 0)
            seen[ref] = index + 1
            label = labels.get(_claim_key(ref, index))
        if label in ("unrelated", "contradicts"):
            continue
        kept.append(p["value"] * (weight_of[label] if label else 1.0))
    filings = sum(kept) / len(kept) if kept else 0.0
    # 5
    blend2 = blend["value"] - weights.news * (comps["news"]["value"] - news) - weights.filings * (comps["filings"]["value"] - filings)
    return Counterfactual(news=round(news, 4), filings=round(filings, 4), blend=round(blend2, 4), blend_recorded=blend["value"])


# ---------------------------------------------------------------- the step

def _heading(found: dict) -> str:
    return f"{found.get('publisher') or 'unknown publisher'}, {str(found.get('published_utc') or '')[:10]}, {', '.join((found.get('tickers') or [])[:4])}"


def _age(published: str | None, today: date) -> int | None:
    try:
        return max(0, (today - date.fromisoformat(str(published)[:10])).days)
    except ValueError:
        return None


def assess(model, rec: Recommendation, claims: list[Claim], checks: list[SourceCheck], verification: dict, order: WorkOrder, advisor: AdvisorConfig, config: CriticConfig, now: datetime,
           critic_model: str | None, independent: bool | None, run: dict, store: LabelStore | None = None) -> EvidenceResult:
    """The evidence findings of one record, its score, and the expected return worked again. `claims` are ALL the record's claims; the model is asked about the supporting ones. `checks` are the
    source checks of recheck (a claim whose source does not list the symbol counts for nothing). `run` is the review's `critic_runs` entry: tokens, calls and drops are added to it, and `labels_from_store`, the number of labels
    the store answered. `store` is the label store (`labels.py`, rule C6): a quote it has a label for is not sent to the model; None keeps and reads none."""
    ev, out = config.evidence, EvidenceResult()
    today = eastern_date(now)
    direction = stance_direction(rec)
    mine = supporting(claims, direction)
    by_ref = {c.ref: c for c in claims}
    check_of = {c.ref: c for c in checks}
    dropped: list[str] = []

    # what the model reads: the claims whose source could be read again, the control, and the fresh sources
    readable = [c for c in mine if (check_of.get(c.ref) and check_of[c.ref].found)]
    fresh_hits = fresh_candidates(rec, order, verification, config)
    fresh = {f"F{i}": h for i, h in enumerate(fresh_hits, 1)}
    labels: dict[str, tuple[str, str | None]] = {}
    marks: dict[str, tuple[str, str | None]] = {}
    control: Control | None = None
    failure: str | None = None
    had_something = bool(readable or fresh)

    # what the model is shown of each claim and each fresh source (the text the label depends on), and the labels the store already has (rule C6): those are not asked again
    shown_of: dict[str, tuple[str, str]] = {c.ref: _claim_source(c, rec, verification, config) for c in readable}
    fresh_shown = {k: (f"{_heading(h)}: {h.get('headline') or ''}", excerpt(h.get("text") or "", "", config.limits.max_source_chars)) for k, h in fresh.items()}
    version = prompt_version(ev.max_reason_words)
    claim_keys = {c.ref: LabelStore.claim_key(critic_model, version, rec.symbol, direction, c.quote, shown_of[c.ref][1]) for c in readable}
    fresh_keys = {k: LabelStore.fresh_key(critic_model, version, rec.symbol, direction, text[0], text[1]) for k, text in fresh_shown.items()}
    if store is not None:
        for ref, key in claim_keys.items():
            kept = store.get(key)
            if kept is not None and kept["label"] in REAL_LABELS:
                labels[ref] = (kept["label"], kept.get("reason") if isinstance(kept.get("reason"), str) else None)
        for key_id, key in fresh_keys.items():
            kept = store.get(key)
            if kept is not None and kept["label"] in ("bearish", "none"):
                marks[key_id] = (kept["label"], kept.get("quote") if isinstance(kept.get("quote"), str) else None)
    from_store = len(labels) + len(marks)
    run["labels_from_store"] = run.get("labels_from_store", 0) + from_store
    ask_claims = [c for c in readable if c.ref not in labels]
    ask_fresh = {k: h for k, h in fresh.items() if k not in marks}
    need_call = bool(ask_claims or ask_fresh)

    if need_call:
        if model is None or run["records_called"] >= config.limits.max_model_calls:
            failure = "the review's model calls were used up" if model is not None else "there is no model to ask"
        else:
            control_c, control_text, control_head = control_claim(rec, order, config, direction, today)
            shown = list(ask_claims)
            shown.insert(seed_of(rec.identity.record_id) % (len(shown) + 1), control_c)
            sources = [(c.ref, control_head, control_text) if c is control_c else (c.ref, *shown_of[c.ref]) for c in shown]
            fresh_block = [(k, *fresh_shown[k]) for k in ask_fresh]
            messages = [SystemMessage(content=review_system(today, rec.symbol, rec.company, direction, ev.max_reason_words)),
                        HumanMessage(content=review_message(rec.symbol, direction, [(c.ref, c.quote) for c in shown], sources, fresh_block))]
            record_run = new_run("review", model)
            reply = ask_object(model, messages, record_run, "claims", RETRY_REVIEW)
            run["records_called"] += 1
            for key in ("tokens_in", "tokens_out", "calls"):
                run[key] += record_run[key]
            out.called = True
            if reply is None:
                failure = record_run["error"] or "the model gave no usable reply"
            else:
                said, noted = read_reply(reply, {c.ref for c in shown}, ask_fresh, rec.symbol, rec.company, ev.max_reason_words, dropped)
                got = said.pop(control_c.ref, None)
                control = Control(ref=control_c.ref, label=got[0] if got else None, passed=bool(got and got[0] == "unrelated"))
                run["controls_failed"] += not control.passed
                if control.passed:
                    labels.update(said)
                    marks.update(noted)
                    if store is not None:      # only a reply whose control was read right is kept: a failed control keeps nothing, so the next review asks again
                        for ref, (label, reason) in said.items():
                            store.put(claim_keys[ref], {"label": label, "reason": reason})
                        for key_id, (label, quote) in noted.items():
                            store.put(fresh_keys[key_id], {"label": label, "quote": quote})
                else:
                    out.findings.append(warning("evidence", "control_failed", f"The model did not read the text: it did not call a planted, unrelated notice {control.expected!r} (it said {control.label!r}). "
                                                f"Its labels are not used.", value=control.label, requires_human=True))
    run["dropped"].extend(f"{rec.symbol}: {d}" for d in dropped)
    if failure:
        run["errors"].append(f"{rec.symbol}: {failure}")
    reviewed = had_something and failure is None and (control is None or control.passed)      # no control when the store answered everything: every label it holds was made by a reply whose control was read right

    # the labels, and what each claim counts for
    unanswered = [c for c in readable if c.ref not in labels]
    label_of = {c.ref: (labels[c.ref][0] if c.ref in labels else "unreviewed") for c in mine}
    untagged = {c.ref for c in checks if c.tagged is False}
    out.detail.claims = [ClaimLabel(ref=c.ref, direction=c.direction, label=label_of[c.ref], reason=(labels.get(c.ref) or (None, None))[1]) for c in mine]
    out.detail.control = control
    if failure:
        out.findings.append(warning("evidence", "evidence_not_reviewed", f"The quotes could not be reviewed by the model: {failure}.", requires_human=True))
    elif reviewed and unanswered:
        out.findings.append(warning("evidence", "evidence_not_reviewed", f"The model left {len(unanswered)} claim(s) unlabelled: {', '.join(c.ref for c in unanswered)}.", requires_human=True))

    # the fresh sources: what the model said, and what CODE finds in them
    fresh_bearish: list[tuple[float, int]] = []
    items: list[FreshItem] = []
    for key, hit in fresh.items():
        classes = bearish_classes(hit.get("text") or "", rec.symbol, rec.company)
        label, quote = marks.get(key, ("none", None))
        items.append(FreshItem(id=key, chunk_id=hit.get("chunk_id"), headline=hit.get("headline"), published_utc=hit.get("published_utc"), corpus=hit.get("corpus") or corpus_of(hit.get("chunk_id") or ""),
                               classes=classes, label=label, quote=quote))
        age = _age(hit.get("published_utc"), today)
        if label == "bearish" or classes:
            kind = 1.0 if (hit.get("corpus") or corpus_of(hit.get("chunk_id") or "")) == "news" else advisor.blend.press_release_weight
            fresh_bearish.append((kind, 0 if age is None else age))
            blocking = [c for c in classes if c in ev.blocking_classes]
            when = f"published {str(hit.get('published_utc'))[:10]}" if hit.get("published_utc") else "undated"
            if blocking and age is not None and age <= ev.fresh_event_max_age_days:
                out.findings.append(blocker("evidence", "fresh_negative_event", f"Newer stored text the Advisor did not see ({key}, {when}) reports {', '.join(blocking)} for {rec.symbol}: {hit.get('headline') or ''}.",
                                            value=blocking, threshold=ev.fresh_event_max_age_days))
            else:
                out.findings.append(warning("evidence", "fresh_negative_news", f"Newer stored text the Advisor did not see ({key}, {when}) looks bad for {rec.symbol}: {hit.get('headline') or ''}.",
                                            value=classes or label))      # Critic v2, rule C2: it informs and asks for nobody (it was `requires_human=True`, a denial since entry 103: 60 of the 72 it fired on and that closed were winners)
    out.detail.fresh = items

    # the event family stands on one event: if it was not supported, the position has no entry
    if rec.trade.family == "event":
        wanted = next((f.features_used.get("event_class") for f in rec.families if f.family == "event" and f.passed), None)
        bad = [c.ref for c in claims if c.event_class == wanted and label_of.get(c.ref) in ("unrelated", "contradicts")]
        if bad:
            out.findings.append(blocker("evidence", "event_not_supported", f"The event {wanted} that opened this position was not supported by its source ({', '.join(bad)}: {label_of[bad[0]]}).", value=wanted))

    # the conviction family stands on the blend alone: if the blend, worked again with this review's labels, falls under the family's own mark, it has no entry
    cf_early = counterfactual(rec, claims, label_of, untagged, fresh_bearish, advisor, config)
    if rec.trade.family == "conviction" and cf_early is not None and cf_early.blend < advisor.families.conviction.min_blend:
        out.findings.append(warning("evidence", "blend_not_supported", f"This position was opened on the blend alone ({rec.blend['value']:+.2f}); with the quotes the review did not accept it is {cf_early.blend:+.2f}, "
                                    f"under the {advisor.families.conviction.min_blend:g} the conviction strategy needs.", value=cf_early.blend, threshold=advisor.families.conviction.min_blend))      # Critic v2, rule C3: a warning (it was a blocker: 37 of the 41 it fired on and that closed were winners; the evidence dimension still loses its points)

    # about the claims themselves
    if independent is False:
        out.findings.append(warning("evidence", "same_model_family", f"The critic's model ({critic_model}) is from the same producer as the Advisor's ({rec.advisor.get('model')}): it may agree with its own family's text."))      # a WARNING, never a person (Jacob, 2026-11-21: GLM was too expensive, CRITIC_MODEL is openai/gpt-oss-120b like the Advisor; with requires_human every review was denied, entry 103)
    dated = [(found.get("published_utc"), found.get("publisher")) for c in mine for found in [(verification.get("chunks") or {}).get(c.chunk_id) or {}] if c.chunk_id and found.get("found")]
    ages = [a for a in (_age(p, today) for p, _ in dated) if a is not None]
    if ages and min(ages) > ev.old_evidence_days:
        out.findings.append(warning("evidence", "evidence_old", f"The newest quote the recommendation stands on is {min(ages)} days old (the limit is {ev.old_evidence_days}).", value=min(ages), threshold=ev.old_evidence_days))
    publishers = {p for _, p in dated if p}
    if len(dated) >= 2 and len(publishers) == 1:   # the quotes of articles and releases; a filing-evidence item has no publisher
        out.findings.append(warning("evidence", "single_publisher", f"All {len(dated)} supporting quotes from articles and releases come from one publisher ({next(iter(publishers))})."))
    if rec.evidence.counter_evidence:
        out.findings.append(info("evidence", "counter_evidence", "The record lists counter-evidence: " + ", ".join(str(c.get("ref")) for c in rec.evidence.counter_evidence[:4]) + ".", value=len(rec.evidence.counter_evidence)))

    # the blend worked again, and the score
    out.detail.counterfactual = cf_early
    out.detail.reviewed = reviewed

    weights = ev.parts
    unknown = config.verdict.unknown_credit
    credit = [unknown if label_of[c.ref] == "unreviewed" else ev.label_weight[label_of[c.ref]] for c in mine]
    support = sum(credit) / len(credit) if credit else unknown
    clean = 1 - min(1.0, sum(1 for item in items if item.label == "bearish" or item.classes) / 2)
    breadth = min(1.0, sum(1 for c in mine if label_of[c.ref] in ("supports", "weak")) / ev.breadth_full_at)
    out.measures = [Measure(name="support", x=round(support, 4), value=round(support, 4), weight=weights.support * 100, available=bool(credit)),
                    Measure(name="clean", x=sum(1 for item in items if item.label == "bearish" or item.classes), value=round(clean, 4), weight=weights.clean * 100),
                    Measure(name="breadth", x=sum(1 for c in mine if label_of[c.ref] in ("supports", "weak")), value=round(breadth, 4), weight=weights.breadth * 100)]
    out.score = round(100 * (weights.support * support + weights.clean * clean + weights.breadth * breadth), 2)
    return out


def independence(critic_model: str | None, advisor_model: str | None) -> bool | None:
    """Whether the critic's model is from a different producer than the Advisor's: True, False, or None when either is unknown."""
    a, b = producer(critic_model), producer(advisor_model)
    return None if a is None or b is None else a != b
