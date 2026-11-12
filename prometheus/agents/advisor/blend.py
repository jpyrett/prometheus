"""The blend: what the score is made of (Advisor v1 amendment, design choices entry 55).

    blend = compute_blend(metrics, signals, events, sources, today, config.blend, negative_equity)
    blend["value"]      # -1 to +1: 0.40 x news + 0.30 x performance + 0.20 x historical + 0.10 x filings
    blend["score"]      # the same on the 0 to 100 scale: round(50 + 50 x value)

Plain Python, no model, no network. Four parts, each -1 (all bearish) to +1 (all bullish), and each shown with the pieces that made it:

    news          the VERIFIED signals and events that cite a news article or a press release, each weighted by how new it is: weight = kind x 0.5 ** (age_days / half_life_days),
                  kind 1.0 for an article and `press_release_weight` for a company's own release. value = sum(direction x weight) / (sum(weight) + news_prior_n): the prior shrinks
                  a single old quote toward zero. A signal and an event that speak for the same source in the same direction are one voice. The model chose WHICH quotes to
                  offer; code checked every quote, took the direction from the event class, and computes every weight.
    performance   the stock's own prices: the measures of `config.blend.measures` in the group `performance` (12-1 month momentum, one-month return, the distance from the 52-week high,
                  the 200-day mean).
    historical    the Finnhub history of the company (earnings yield, return on equity, revenue growth, margin, current ratio, debt to equity: not for negative equity, entry 54).
    filings       the SEC filings: the measures of the group `filings` (net payout yield, share count change, payout coverage, cash flow against profit), plus the verified
                  filing-evidence items and filing-index events (Form 144, Schedule 13D) as one vote each.

A measure is `clip((x - neutral) / span, -1, 1)` (a negative span: lower is better) or, for a yes/no metric, +1 when it equals `good` and -1 when it does not (`only_bad`: the good state is
silence). A part's value is the mean of its measures that exist. A part with nothing to say is UNAVAILABLE: it counts as neutral (0) and is listed as unavailable, so the Critic can see how
much of the blend was actually measured. Every number is in `configs/advisor_v1.json` (`blend`).
"""
from datetime import date

from prometheus.agents.advisor.config import Blend as BlendConfig
from prometheus.agents.advisor.gates import value as metric_value
from prometheus.common.work_order import corpus_of

GROUPS = ("news", "performance", "historical", "filings")


def clip(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def recency(age_days: int | None, half_life_days: float) -> float:
    """1.0 for news of today, 0.5 after one half-life, 0.25 after two; a date that cannot be read counts as one half-life old (it is not dropped and not trusted as new)."""
    return 0.5 ** ((half_life_days if age_days is None else max(0, age_days)) / half_life_days)


def age_in_days(published_utc: str | None, today: date) -> int | None:
    try:
        return max(0, (today - date.fromisoformat(str(published_utc)[:10])).days)
    except ValueError:
        return None


def _speaker(source_number, sources: dict[int, dict]) -> tuple[str | None, str | None]:
    """(corpus, published_utc) of a numbered source: "news", "press_releases" or "filings"; (None, None) for a source that is not on the order."""
    source = sources.get(source_number) if isinstance(source_number, int) else None
    return (corpus_of(source.get("chunk_id") or ""), source.get("published_utc")) if source else (None, None)


def text_items(signals: list[dict], events: list[dict], sources: dict[int, dict], today: date, cfg: BlendConfig) -> tuple[list[dict], list[dict]]:
    """The votes of the text, sorted into (news and press releases, filings): each {ref, kind, direction, age_days, recency, weight}. Only a verified, counted signal and a bullish or bearish
    event vote; one source in one direction is one voice."""
    news, filings, seen = [], [], set()

    def add(ref: str, direction: str, corpus: str | None, published: str | None, origin_filing: bool = False):
        if direction not in ("bullish", "bearish"):
            return
        age = age_in_days(published, today)
        if origin_filing or corpus == "filings":
            filings.append({"ref": ref, "kind": "filings", "direction": direction, "age_days": age, "recency": None, "weight": 1.0})
            return
        kind = 1.0 if corpus in (None, "news") else cfg.press_release_weight
        rec = recency(age, cfg.half_life_days)
        news.append({"ref": ref, "kind": "press_release" if corpus == "press_releases" else "news", "direction": direction, "age_days": age, "recency": round(rec, 4), "weight": round(kind * rec, 4)})

    for s in signals:
        if not (s.get("verified") and s.get("counted")):
            continue
        seen.add((s.get("source"), s["direction"]))
        if s.get("evidence_id"):  # an evidence item of the filings, written by code from a filing
            add(s["evidence_id"], s["direction"], "filings", None, origin_filing=True)
        else:
            corpus, published = _speaker(s.get("source"), sources)
            add(f"[{s.get('source')}]", s["direction"], corpus or corpus_of(s.get("chunk_id") or ""), published)
    for e in events:
        if (e.get("source"), e.get("direction")) in seen:
            continue
        corpus, published = _speaker(e.get("source"), sources)
        add(e["class"], e.get("direction"), corpus, e.get("published_utc") or published, origin_filing=e.get("origin") == "filing_index")
    return news, filings


def measure_values(metrics: dict | None, cfg: BlendConfig, group: str, negative_equity: bool) -> list[dict]:
    """The measures of one group that have a value: {metric, raw, value, waived?}. A measure whose metric is missing is left out, not counted as neutral."""
    out = []
    for m in cfg.measures:
        if m.group != group:
            continue
        if m.waived_by_negative_equity and negative_equity:
            continue
        raw = metric_value(metrics, m.metric)
        if raw is None:
            continue
        if m.good is not None:
            if m.only_bad and raw == m.good:
                continue
            out.append({"metric": m.metric, "raw": raw, "value": 1.0 if raw == m.good else -1.0})
        else:
            out.append({"metric": m.metric, "raw": round(float(raw), 6), "value": round(clip((float(raw) - m.neutral) / m.span), 4)})
    return out


def compute_blend(metrics: dict | None, signals: list[dict], events: list[dict], sources: dict[int, dict], today: date, cfg: BlendConfig, negative_equity: bool = False) -> dict:
    """The blend of one company: {value, score, weights, components: {news|performance|historical|filings: {weight, value, available, contribution, n, parts}}, half_life_days}."""
    news_items, filing_items = text_items(signals, events, sources, today, cfg)
    sign = lambda item: 1.0 if item["direction"] == "bullish" else -1.0
    total = sum(i["weight"] for i in news_items)
    news_value = sum(sign(i) * i["weight"] for i in news_items) / (total + cfg.news_prior_n) if news_items else None
    parts = {"news": news_items}
    values = {"news": None if news_value is None else round(news_value, 4)}
    for group in ("performance", "historical"):
        parts[group] = measure_values(metrics, cfg, group, negative_equity)
        values[group] = round(sum(p["value"] for p in parts[group]) / len(parts[group]), 4) if parts[group] else None
    measures = measure_values(metrics, cfg, "filings", negative_equity)
    votes = [{"ref": i["ref"], "direction": i["direction"], "value": sign(i)} for i in filing_items]
    parts["filings"] = [*measures, *votes]
    values["filings"] = round(sum(p["value"] for p in parts["filings"]) / len(parts["filings"]), 4) if parts["filings"] else None
    weights = {g: getattr(cfg.weights, g) for g in GROUPS}
    components = {g: {"weight": weights[g], "value": values[g] if values[g] is not None else 0.0, "available": values[g] is not None, "n": len(parts[g]),
                      "contribution": round(weights[g] * (values[g] or 0.0), 4), "parts": parts[g]} for g in GROUPS}
    blended = round(sum(c["contribution"] for c in components.values()), 4)
    return {"value": blended, "score": round(50 + 50 * blended), "weights": weights, "half_life_days": cfg.half_life_days, "components": components}
