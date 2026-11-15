"""`Advisor.generate_recommendations`: the whole step, in order (Advisor v1 spec section 7, design choices entry 51; the walk-through is design/week5_advisor_implementation.md).

    order = Advisor().generate_recommendations(order, portfolio, case="funds", today=date.today())

ONE work order in (status `data_ready`), the SAME work order out (status `recommended`). A LangGraph node is `lambda state: advisor.generate_recommendations(state["order"], portfolio, case)`. The twelve steps:

     1  preconditions       the order has its data; the case (funds, theme; `market` needs the market brief, which is not built); today
     2  candidates          who to look at (candidates.py)
     3  PLAN call           MODEL call 1: signals, events and a drop cause, with exact quotes; every quote checked by code (v0's verify_signal, events.verify_event)
     4  metrics             ONE `calculate_financials` order to the Data Clerk for every number every strategy, gate and the plan will read (DataClerk.request_metrics)
     5  gates               freshness, red flags, event risk (gates.py; no liquidity gate, entry 93): a failure is recorded, never silent
     6  families            five strategy families, each one score from fixed rules (families.py)
     7  quality             the haircut multiplier (quality.py)
     8  rank                composite edge, cost, stance, the blended score that ranks, the regime (ranking.py)
     9  holdings + select   what to do about what is already held; the cooldown; the sleeve of the case picks the new ideas (ranking.py)
    10  trade plan          entry, stop, target, time stop, order style, invalidation rules: prices and dates, NO size (trade_plan.py; how much is the Account Manager's, entry 80)
    11  THESIS call         MODEL call 2: three sentences and bullets from the FINISHED record, every number and ref checked by code (thesis.py)
    12  write out           records onto `order.recommendations` (a reduction first, then additions, then new ideas by rank), the rest onto `order.candidates`, the runs onto
                            `order.advisor_runs`, one line per record onto the ledger

The Advisor does not use the user's money and recommends no amount (entry 80): `portfolio` is read only for WHICH symbols are held (the holdings pass and the names of the candidates), their shares
and their cost basis (is a holding in profit, how much has it gained), never the cash or the equity. A reduction, like an opening, carries no number of shares.

The model is called exactly twice. It writes no number that is used: the numbers come from the Data Clerk's metrics and from this module's arithmetic. A failed model call does not stop
the step: without the plan call there are no text signals (the families that need none still run), without the thesis call a record carries `thesis: null` and says why.
"""
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from prometheus.agents.advisor import families as fam
from prometheus.agents.advisor import ranking as rk
from prometheus.agents.advisor import trade_plan as tp
from prometheus.agents.advisor.advisor import Advisor, build_recommendation, names_company
from prometheus.agents.advisor.calls import ask_json, new_run
from prometheus.agents.advisor.candidates import collect_candidates, infer_case, theme_term
from prometheus.agents.advisor.events import EVENT_CLASSES, filing_index_events, last_earnings_release, timing, verify_event
from prometheus.agents.advisor.blend import compute_blend
from prometheus.agents.advisor.gates import negative_equity, run_gates, value
from prometheus.agents.advisor.quality import quality_multiplier
from prometheus.agents.advisor.record import Action, Agreement, Audit, Candidate, Evidence, Identity, PortfolioFit, Ranking, Recommendation, Risk, Thesis
from prometheus.agents.advisor.thesis import write_theses
from prometheus.common.portfolio import Portfolio
from prometheus.common.work_order import MAX_METRIC_SYMBOLS, WorkOrder, corpus_of, new_work_order, sector_map
from prometheus.prompts.advisor_v1 import RETRY_PLAN, plan_message, plan_system
from prometheus.prompts.advisor import evidence_text, kinds_text, user_message
from prometheus.rag.loaders import check_symbol

GATE_METRICS = ("last_close", "days_to_next_earnings", "debt_to_equity", "negative_equity", "current_ratio", "cfo_negative_with_profit")
PLAN_METRICS = ("atr_14", "sma_50", "above_sma_200", "earnings_move_abs")
RISK_METRICS = ("vol_63d_ann", "atr_pct", "beta_63d", "max_drawdown_63d")
REGIME_METRICS = ("spy_above_sma_200", "spy_vol_21d_ann", "spy_drawdown_63d", "spy_ret_21d", "breadth_200")
EVIDENCE_FIELDS = ("value", "unit", "as_of", "source")


@dataclass
class Checked:
    """What the plan call said about one company, after code checked it."""
    v0: dict | None = None                          # `build_recommendation`'s record: the signals, verified and counted
    company: str | None = None
    events: list[dict] = field(default_factory=list)
    drop_cause: dict | None = None
    archetypes: list[str] = field(default_factory=list)
    model_score: int | None = None


def _now(today: date | None) -> str:
    """When the records are as of: the date the caller passed (at midnight UTC: a rerun for that date is the same date), else this moment."""
    return f"{today.isoformat()}T00:00:00Z" if today else datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _metric_names(config) -> list[str]:
    names = [*fam.required_metrics(), *GATE_METRICS, *PLAN_METRICS, *RISK_METRICS, *REGIME_METRICS, "sector_etf", *(c.metric for c in config.quality.checks), *(m.metric for m in config.blend.measures)]
    return list(dict.fromkeys(names))


def _verify_claims(raw: dict, symbol: str, company: str | None, known: str | None, sources: dict[int, dict], run: dict, limit: int) -> tuple[list[dict], dict | None]:
    """The events and the drop cause the model gave for one company, each checked by `verify_event`. A claim that fails is recorded in `run["dropped"]` with its reason."""
    events = []
    for claim in (raw.get("events") if isinstance(raw.get("events"), list) else [])[:limit]:
        if not isinstance(claim, dict):
            continue
        ok, reason = verify_event(claim, symbol, company, sources, known)
        if not ok:
            run["dropped"].append(f"{symbol}: event {claim.get('class')!r} dropped: {reason}")
            continue
        number = int(claim["source"]) if str(claim["source"]).isdigit() else claim["source"]
        events.append({"class": claim["class"], "source": number, "quote": str(claim["quote"])[:300], "origin": "model", "published_utc": sources[number].get("published_utc")})
    cause = raw.get("drop_cause")
    drop = None
    if isinstance(cause, dict):
        ok, reason = verify_event(cause, symbol, company, sources, known)
        if ok:
            drop = {"class": cause["class"], "source": int(cause["source"]), "quote": str(cause["quote"])[:300]}
        else:
            run["dropped"].append(f"{symbol}: drop_cause dropped: {reason}")
    return events, drop


def _plan_call(advisor: Advisor, order: WorkOrder, candidates: list[dict], case: str, today: date, sources: dict[int, dict], sources_text: str | None, evidence: dict, known: dict[str, str], run: dict
               ) -> dict[str, Checked]:
    """MODEL call 1 and the checks of everything it said. Returns {symbol: Checked} for every company the model gave that is a candidate or has a verified signal."""
    limits = advisor.config.limits
    messages = [SystemMessage(content=plan_system(today, limits.max_companies, limits.max_signals_per_kind, evidence=bool(evidence))),
                HumanMessage(content=plan_message(user_message(order, sources_text, kinds_text(order) if sources_text else None, evidence_text(order) if evidence else None), candidates, case))]
    raws = ask_json(advisor.model, messages, run, "companies", RETRY_PLAN)
    if raws is None:
        order.errors = [*order.errors, run["error"]]
        return {}
    if len(raws) > limits.max_companies:
        run["dropped"].append(f"{len(raws) - limits.max_companies} company(ies) beyond the first {limits.max_companies}")
    candidate_symbols = {c["symbol"] for c in candidates}
    out: dict[str, Checked] = {}
    for raw in raws[:limits.max_companies]:
        if not isinstance(raw, dict) or not isinstance(raw.get("symbol"), str):
            run["dropped"].append("a company without a symbol")
            continue
        v0 = build_recommendation({"symbol": raw["symbol"], "company": raw.get("company"), "stance": None, "model_score": raw.get("model_score"), "signals": raw.get("signals")}, sources, run["dropped"], known, evidence)
        if v0 is None:
            continue
        symbol = v0["symbol"]
        if symbol not in candidate_symbols and v0["signals_verified"] == 0:
            run["dropped"].append(f"{symbol}: not a candidate and no verified signal")
            continue
        events, drop = _verify_claims(raw, symbol, v0["company"], known.get(symbol), sources, run, limits.max_events)
        archetypes = [a for a in (raw.get("archetypes") if isinstance(raw.get("archetypes"), list) else []) if a in fam.FAMILIES]
        out[symbol] = Checked(v0, v0["company"], events, drop, list(dict.fromkeys(archetypes)), v0["model_score"])
    return out


def _timed(events: list[dict], metrics: dict | None, config) -> list[dict]:
    """Each checked event with its direction (fixed by the class), its age in bars, the reaction since it and the event bar's volume ratio, from the symbol's `recent_bars`."""
    recent, atr = value(metrics, "recent_bars") or [], value(metrics, "atr_pct")
    out = []
    for e in events:
        t = timing(e.get("published_utc"), recent) or {"age_bars": None, "reaction": None, "event_volume_ratio": None, "bar_date": None}
        reaction_atr = abs(t["reaction"]) / atr if t["reaction"] is not None and atr else None
        out.append({**e, "direction": EVENT_CLASSES[e["class"]].direction, **t, "reaction_atr": reaction_atr})
    return out


def _votes(signals: list[dict], events: list[dict], sources: dict[int, dict], max_age: int) -> list[tuple[str, float]]:
    """The text evidence as (direction, weight): a counted news signal 1.0, the company's own words (a press release, a filing, an evidence item) 0.5; a bullish or bearish event of the last
    `max_age` bars the same, unless a counted signal already speaks for the same source in the same direction (one story is one voice)."""
    votes, seen = [], set()
    for s in signals:
        if not (s.get("verified") and s.get("counted")):
            continue
        weight = 0.5 if s.get("evidence_id") else 1.0 if corpus_of(s.get("chunk_id") or "") == "news" else 0.5
        votes.append((s["direction"], weight))
        seen.add((s.get("source"), s["direction"]))
    for e in events:
        if e["direction"] not in ("bullish", "bearish") or e.get("age_bars") is None or e["age_bars"] > max_age or (e.get("source"), e["direction"]) in seen:
            continue
        source = sources.get(e["source"]) if isinstance(e.get("source"), int) else None
        votes.append((e["direction"], 1.0 if source and corpus_of(source.get("chunk_id", "")) == "news" else 0.5))
    return votes


def _refs(signals: list[dict], events: list[dict]) -> list[dict]:
    out = [{"ref": s.get("evidence_id") or f"[{s['source']}]", "direction": s["direction"], "quote": s["quote"]} for s in signals if s.get("verified") and s.get("counted")]
    out += [{"ref": e["class"], "direction": e["direction"], "quote": e["quote"]} for e in events if e["direction"] in ("bullish", "bearish")]
    return out


def _evidence(s: rk.Scored, direction: str, items: dict[str, dict]) -> Evidence:
    """The evidence section: verified signals, the filing items they cite, the events, what runs against the stance, and the pairs of opposed counted quotes (derived by code)."""
    counted = [x for x in s.signals if x.get("verified") and x.get("counted")]
    refs = _refs(s.signals, s.events)
    against = "bearish" if direction == "bullish" else "bullish"
    bull = [r["ref"] for r in refs if r["direction"] == "bullish"]
    bear = [r["ref"] for r in refs if r["direction"] == "bearish"]
    return Evidence(signals=[x for x in s.signals if x.get("verified")], e_items=[items[x["evidence_id"]] for x in counted if x.get("evidence_id") in items],
                    events=s.events, counter_evidence=[r for r in refs if r["direction"] == against], contradictions=[[a, b] for a in bull for b in bear][:4])


def _features(metrics: dict | None) -> dict[str, dict]:
    return {name: {k: result.get(k) for k in EVIDENCE_FIELDS} for name, result in (metrics or {}).items() if result.get("value") is not None}


def _risk(s: rk.Scored, plan) -> Risk:
    flags = next((g.value for g in s.gates.gates if g.gate == "red_flag"), None) or []
    return Risk(**{n: value(s.metrics, n) for n in ("vol_63d_ann", "atr_pct", "beta_63d", "max_drawdown_63d", "days_to_next_earnings", "earnings_move_abs")},
                gap_risk=plan.gap_risk, red_flags=list(flags))


def _record(s: rk.Scored, checked: Checked | None, family, action: str, reason: str, plan, ident: dict, run_ids: dict, regime: tuple[str, str | None], items: dict, config, price: float | None,
            cooldown: str | None = None, extra: dict | None = None) -> Recommendation:
    extra = extra or {}
    v0 = (checked.v0 if checked and checked.v0 else {})
    direction = s.stance if s.stance in ("bullish", "bearish") else (family.direction if family else "bullish")
    return Recommendation(
        symbol=s.symbol, company=s.company, negative_equity=negative_equity(s.metrics), blend=s.blend, stance=s.stance, model_score=checked.model_score if checked else None, score=s.score, supported=bool(s.passed()), consistent=None,
        bullish_signals=v0.get("bullish_signals", 0), bearish_signals=v0.get("bearish_signals", 0), signals_claimed=v0.get("signals_claimed", 0), signals_verified=v0.get("signals_verified", 0),
        signals_counted=v0.get("signals_counted", 0), signals=v0.get("signals", []), advisor={"model": run_ids["model"], "version": "v1", "proposed_archetypes": checked.archetypes if checked else []},
        identity=Identity(record_id=uuid.uuid4().hex[:12], request_id=ident["request_id"], case=ident["case"], sleeve=config.sleeves[ident["case"]].sleeve, as_of_utc=ident["as_of_utc"],
                          sector_etf=s.sector_etf, regime=regime[0], regime_note=regime[1]),
        action=Action(action=action, reason=reason, held_shares=s.held_shares, cost_basis=s.cost_basis, unrealized_pct=round(price / s.cost_basis - 1, 4) if price and s.cost_basis else None, cooldown_of=cooldown),
        families=s.families, quality=s.quality, evidence=_evidence(s, direction, items), features=_features(s.metrics),
        agreement=Agreement(agreement=round(s.agreement, 4), evidence_n=s.evidence_n, completeness=s.passed()[0].completeness if s.passed() else None, confidence=round(s.confidence, 4)),
        ranking=Ranking(composite=s.composite, conflict=s.conflict, stance=s.stance, stance_reason=s.stance_reason,
                        diversity_adjustment=extra.get("diversity_adjustment", False), displaced_by=extra.get("displaced_by")),
        trade=plan, risk=_risk(s, plan), portfolio_fit=PortfolioFit(note=extra.get("portfolio_note")),
        gates=s.gates.gates, thesis=Thesis(thesis=None, thesis_error="not written yet"),
        audit=Audit(metric_order=run_ids.get("metric_order"), plan_run=run_ids.get("plan"), config_version=config.config_version, family_versions={f: "v1" for f in fam.FAMILIES}))


def generate(advisor: Advisor, order: WorkOrder, portfolio: Portfolio, case: str | None = None, today: date | None = None, top_k: int | None = None, only_subject: bool = False) -> WorkOrder:
    """The twelve steps (see the module text). Returns the SAME order. `top_k` replaces the sleeve's number of new ideas (1 to 20). `only_subject`: look only at the order's own symbols and the
    holdings (a comparison of named companies, entry 53): no symbol is added because the sources name it, the theme finds it or the model proposes it."""
    config, as_of = advisor.config, _now(today)
    today = today or date.today()
    # ---- 1 preconditions
    if order.status != "data_ready":
        order.errors = [*order.errors, f"the Advisor needs an order whose data is ready (status {order.status})"]
        return order
    case = case or order.case or infer_case(order)
    if case not in ("funds", "theme", "industry"):
        order.errors = [*order.errors, f"case {case!r} is not supported: 'market' needs the market brief (Advisor v1 spec section 13), which is not built; use 'funds', 'theme' or 'industry'"]
        return order
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20):
        order.errors = [*order.errors, "top_k must be a whole number from 1 to 20"]
        return order
    if top_k is not None:
        config = config.model_copy(update={"sleeves": {**config.sleeves, case: config.sleeves[case].model_copy(update={"top_k": top_k})}})
    order.schema_version = max(order.schema_version, 2)
    # ---- 2 candidates
    clerk = advisor.clerk
    theme_rows, theme_note = None, None
    if case == "theme" and theme_term(order):
        try:
            theme_rows = clerk.tools.discover_tickers(theme_term(order), from_date=30, top=20)["tickers"]
        except Exception as error:  # the stored market news cannot be read: the theme basket is empty, and the run says so
            theme_note = f"the theme's tickers could not be discovered ({type(error).__name__})"
            order.errors = [*order.errors, theme_note]
    candidates, dropped = collect_candidates(order, portfolio, case, theme_rows, config.limits.max_candidates, config.limits.discover_min_articles, only_subject)
    plan_run = new_run("plan", advisor.model)
    plan_run["dropped"].extend(dropped)
    order.advisor_runs = [*order.advisor_runs, plan_run]
    sources, sources_text = advisor._sources(order)
    evidence = advisor._evidence(order)
    held = {p.symbol: p for p in portfolio.positions}
    known = {c["symbol"]: c["company"] for c in candidates if c.get("company")}
    # ---- 3 plan call
    checked = _plan_call(advisor, order, candidates, case, today, sources, sources_text, evidence, known, plan_run) if candidates else {}
    for symbol, chk in checked.items():
        if symbol not in {c["symbol"] for c in candidates} and not only_subject:
            candidates.append({"symbol": symbol, "company": chk.company, "origin": "model", "held": symbol in held})
    if not candidates:
        order.status = "recommended"   # nothing to look at: a result, not a failure
        return order
    # ---- 4 metrics (one order to the Data Clerk)
    symbols = [c["symbol"] for c in candidates][:MAX_METRIC_SYMBOLS]
    given = sector_map(order) or {}
    sectors = {sym: given[sym] for sym in symbols if sym in given}
    metric_order = new_work_order(f"metrics for {order.request_id}", requester="advisor", action_type="calculate_financials", subject=symbols, metrics=[{"name": n} for n in _metric_names(config)],
                                  sector=sectors or None)
    clerk.request_metrics(metric_order, today)
    plan_run["metric_order"] = metric_order.request_id
    metrics = (metric_order.data.metrics if metric_order.data else {})
    if metric_order.status == "data_failed":
        order.errors = [*order.errors, *(f"metrics: {e}" for e in metric_order.errors)]
    if order.data is not None:
        order.data.metrics = metrics
    quotes = order.data.quotes if order.data else {}
    filing_sources = order.data.sources_of("filings") if order.data and order.data.results else []
    press_sources = order.data.sources_of("press_releases") if order.data and order.data.results else []
    regime_metrics = next((m for m in metrics.values() if m.get("spy_above_sma_200")), None)
    regime = rk.regime_of(regime_metrics, config)
    # ---- 5-8 gates, families, quality, rank
    basket = [c["symbol"] for c in candidates if any(r.get("symbol") == c["symbol"] and r.get("articles", 0) >= config.limits.discover_min_articles for r in (theme_rows or []))]
    scored: dict[str, rk.Scored] = {}
    for c in candidates:
        symbol, m, chk = c["symbol"], metrics.get(c["symbol"]), checked.get(c["symbol"])
        events = _timed([*(chk.events if chk else []), *filing_index_events(symbol, filing_sources)], m, config)
        signals = chk.v0["signals"] if chk and chk.v0 else []
        quality = quality_multiplier(m, today, config.quality)
        gates = run_gates(m, events, today, config.gates, last_earnings_release(symbol, press_sources))
        recent = value(m, "recent_bars") or []
        recent_company = sum(1 for src in sources.values() if symbol in (src.get("tickers") or []) and names_company(src.get("text") or "", symbol, c.get("company"))
                             and (timing(src.get("published_utc"), recent) or {}).get("age_bars", 99) <= config.families.reversion.cause_lookback_bars)
        row = next((r for r in (theme_rows or []) if r.get("symbol") == symbol), None)
        theme = None
        if case == "theme" and row and row.get("articles", 0) >= 1:
            total = sum(r["articles"] for r in theme_rows if r["articles"] >= config.limits.discover_min_articles) or 1
            theme = {"articles": row["articles"], "positive": row.get("positive", 0), "neutral": row.get("neutral", 0), "negative": row.get("negative", 0), "article_share": row["articles"] / total,
                     "basket_ret_21d": {b: value(metrics.get(b), "ret_21d") for b in basket if value(metrics.get(b), "ret_21d") is not None}}
        blend = compute_blend(m, signals, events, sources, today, config.blend, negative_equity(m))      # before the families: `conviction` reads it
        inp = fam.FamilyInput(symbol=symbol, metrics=m or {}, events=events, signals=signals, quality=quality, regime=regime[0], config=config, today=today, drop_cause=chk.drop_cause if chk else None, blend=blend,
                              recent_company_sources=recent_company, theme=theme)
        results = [fam.evaluate(f, inp) for f in fam.FAMILIES if f != "theme" or case == "theme"]
        position = held.get(symbol)
        s = rk.Scored(symbol=symbol, company=c.get("company") or (chk.company if chk else None), origin=c["origin"], held=c["held"], metrics=m or {}, gates=gates, quality=quality, families=results,
                      events=events, signals=signals, votes=_votes(signals, events, sources, config.families.event.max_age_bars), sector_etf=sectors.get(symbol),
                      held_shares=position.shares if position else 0, cost_basis=position.cost_basis if position else None)
        theme_result = next((r for r in results if r.family == "theme"), None)
        s.theme_role = (theme_result.features_used.get("theme_role") if theme_result else None)
        s.blend = blend
        scored[symbol] = rk.rank_symbol(s)
    # ---- 9 holdings, cooldown, select
    ledger = advisor.ledger
    ident = {"request_id": order.request_id, "case": case, "as_of_utc": as_of}
    run_ids = {"model": plan_run["model"], "metric_order": metric_order.request_id, "plan": {k: plan_run[k] for k in ("model", "tokens_in", "tokens_out", "calls", "error")}}
    new_ideas = [s for s in scored.values() if not s.held]
    rejected: dict[str, dict] = {}
    cooling = {}
    eligible = []
    for s in new_ideas:
        family = rk.open_family(s, case, regime[0], config)
        old = rk.cooldown_of(s, family.family, ledger.recent(s.symbol, family.family), today.isoformat()) if family else None
        if old:
            cooling[s.symbol] = old
            rejected[s.symbol] = {"reason": f"cooldown: an earlier {family.family} recommendation ({old}) still stands", "cooldown_of": old}
        else:
            eligible.append(s)
    picked, why = rk.select(eligible, case, regime[0], config)
    universe = {s.symbol: n for n, s in enumerate(sorted(new_ideas, key=lambda s: (bool(s.gates.failed), *rk.sort_key(s))), 1)}   # every new idea, best to worst: those that passed every gate first
    for symbol, reason in why.items():
        rejected.setdefault(symbol, {"reason": reason})
    add_floor = min((rk.rank_value(s) for s, _, _ in picked), default=None)
    records: list[tuple[str, Recommendation]] = []
    for s in (x for x in scored.values() if x.held):
        family = rk.open_family(s, case, regime[0], config) or next(iter(sorted(s.passed("bullish"), key=lambda f: -f.edge_20d)), None)
        price = tp.price_of(s.symbol, s.metrics, quotes.get(s.symbol))[0]
        action, reason = rk.holding_action(s, ledger.recent(s.symbol), add_floor)
        if action == "hold":
            rejected[s.symbol] = {"reason": f"hold: {reason}"}
            continue
        plan = tp.build(family if action == "add" else None, s.metrics, quotes.get(s.symbol), today, config, s.symbol)      # a reduction has an entry zone and no amount, like every plan
        records.append((action, _record(s, checked.get(s.symbol), family if action == "add" else None, action, reason, plan, ident, run_ids, regime, evidence.get(s.symbol, {}), config, price)))
    # ---- 10 trade plans for the new ideas, then the records
    for position, (s, family, extra) in enumerate(picked, 1):
        plan = tp.build(family, s.metrics, quotes.get(s.symbol), today, config, s.symbol, reaction_atr=next((e.get("reaction_atr") for e in s.events if e["class"] == family.features_used.get("event_class")), None))
        price = tp.price_of(s.symbol, s.metrics, quotes.get(s.symbol))[0]
        if case == "funds" and len(picked) > 1:
            extra = {**extra, "portfolio_note": f"{len(picked)} names are one bet: sectors {[p[0].sector_etf or 'unknown' for p in picked]}, betas {[value(p[0].metrics, 'beta_63d') for p in picked]}"}
        rec = _record(s, checked.get(s.symbol), family, "open", f"{family.family}: blended score {s.score} (news, performance, history and filings together)", plan, ident, run_ids, regime, evidence.get(s.symbol, {}), config, price, extra=extra)
        if case == "funds":
            rec.ranking.concentrated = True
        rec.ranking.universe_rank = universe.get(s.symbol)
        records.append(("open", rec))
    order_of = {"reduce": 0, "add": 1, "open": 2}
    records.sort(key=lambda pair: (order_of[pair[0]], pair[1].score is None, -(pair[1].score or 0)))
    for number, (_, rec) in enumerate(records, 1):
        rec.ranking.rank = number
    # ---- 11 thesis call
    recorded = {rec.symbol for _, rec in records}
    dumped = [rec.model_dump(mode="json") for _, rec in records]
    thesis_run = new_run("thesis", advisor.model)
    if dumped:
        order.advisor_runs = [*order.advisor_runs, thesis_run]
        theses = write_theses(advisor.model, dumped, thesis_run)
        for _, rec in records:
            rec.thesis = theses[rec.symbol]
            rec.audit.thesis_run = {k: thesis_run[k] for k in ("model", "tokens_in", "tokens_out", "calls", "error")}
    # ---- 12 write out
    final = [rec.model_dump(mode="json") for _, rec in records]
    order.recommendations = [*order.recommendations, *final]
    leftovers = []
    for s in scored.values():
        if s.symbol in recorded:
            continue
        info = rejected.get(s.symbol, {})
        gate = s.gates.first_failed()
        parts = ([f"gate {gate.gate} failed" + (f": {gate.note}" if gate.note else "")] if gate else []) + ([info["reason"]] if info.get("reason") and not (gate and info["reason"].startswith("gate ")) else [])
        reason = "; ".join(parts) or "not selected"
        leftovers.append(Candidate(symbol=s.symbol, company=s.company, origin=next(c["origin"] for c in candidates if c["symbol"] == s.symbol), held=s.held, gate=gate.gate if gate else None,
                                   gate_value=gate.value if gate else None, families=s.families, negative_equity=negative_equity(s.metrics), blend=s.blend, score=s.score, stance=s.stance, rank=universe.get(s.symbol), reason=reason, cooldown_of=info.get("cooldown_of"),
                                   theme_role=s.theme_role))
    leftovers.sort(key=lambda c: (c.score is None, -(c.score or 0), c.symbol))
    order.candidates = [c.model_dump(mode="json") for c in leftovers[:config.limits.max_candidates_on_order]]
    for rec in final:
        ledger.append(rec)
    if ledger.problems:
        order.errors = [*order.errors, *(f"ledger: {p}" for p in ledger.problems)]
    order.status = "recommended"
    return order
