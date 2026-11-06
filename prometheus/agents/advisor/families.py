"""The strategy families: one score per family per symbol, each from fixed rules over the Data Clerk's numbers (Advisor v1 spec section 8, design choices entry 51).

    result = evaluate("capital_return", FamilyInput(symbol, metrics, events, signals, quality, regime, config, today))      # a FamilyResult

A FAMILY is one way a stock can be worth a trade. It has ONE vote per symbol: correlated features (RSI, Bollinger %B and the z-score all say "stretched down") are folded into one score
by the family's own formula, so that three of them do not count three times. The families:

    trend           momentum with a quality filter and an anti-chase rule (do not buy what already ran 15% in a month)
    event           a corporate event (guidance, buyback, ...) that the price has not fully reacted to yet
    reversion       a sound company that fell for a reason that is not about the company
    capital_return  a company that gives money back (buybacks, dividends) without diluting its holders
    theme           Case 2 only: a company in the theme's basket that has lagged it

WHAT A RESULT SAYS. `computable` is False when a REQUIRED number is missing: the family then says nothing and `missing` names what it needed (with 17 stored sessions `trend` needs
`mom_12_1`, so it is `computable: false`: that is correct, not a failure). `passed` is True only when every rule of the family held; only then is there a `score` (0 to 1), a
`direction` and an `edge_20d`. A computable family that did not pass lists, in `notes`, each rule that failed and the value that failed it: the Critic reads why.

    edge_20d = prior_20d x regime_multiplier x score x quality_multiplier        (negative for a bearish family)

`prior_20d` is the family's expected 20-day excess return over SPY, a number from the research the spec cites (`basis_ref`); `basis` is "prior" because this system has not measured
its own outcomes yet. The thesis must call a prior-based number "unmeasured". All thresholds, weights and priors are in `configs/advisor_v1.json`; this file has no number a strategy uses.
"""
from dataclasses import dataclass, field
from datetime import date

from prometheus.agents.advisor.config import AdvisorConfig
from prometheus.agents.advisor.events import BEARISH, BULLISH, NEUTRAL, amount_of
from prometheus.agents.advisor.gates import WAIVED_NOTE, negative_equity
from prometheus.agents.advisor.gates import note as note_of
from prometheus.agents.advisor.gates import value as value_of
from prometheus.agents.advisor.record import FamilyResult, Quality

REQUIRED = {
    "trend": ("mom_12_1", "pct_of_52w_high", "above_sma_200", "ret_21d", "rsi_14", "atr_pct"),
    "event": ("atr_pct", "ret_1d", "volume_ratio_1d"),
    "reversion": ("ret_5d", "zscore_21d", "rsi_14", "atr_pct", "roe", "debt_to_equity", "current_ratio", "revenue_growth"),
    "capital_return": ("earnings_yield", "debt_to_equity", "dilution_current"),
    "theme": ("ret_21d",),
}
OPTIONAL = {
    "trend": ("residual_mom_12_1", "macd_hist", "adx_14", "obv_20d_change", "vol_63d_ann", "breakout_20d", "bollinger_pct_b"),
    "event": ("gap_1d", "sue_4q", "eps_surprise_last", "market_cap"),
    "reversion": ("residual_5d", "sector_ret_5d", "spy_ret_5d", "above_sma_200", "above_sma_50", "bollinger_pct_b", "volume_ratio_1d", "gap_1d", "candle_reversal_1d", "stoch_k_14"),
    "capital_return": ("payout_coverage", "revenue_growth", "net_payout_yield_ttm", "share_count_change", "sbc_to_revenue", "market_cap", "cash_balance"),
    "theme": ("residual_21d", "volume_ratio_1d", "roe", "operating_margin"),
}
ANY_OF = {"capital_return": ("share_count_change", "net_payout_yield_ttm")}   # at least one of these must be present
EXTRA_REQUIRED = {"recent_bars": "dates the events and measures their reaction"}  # asked for every candidate (the Advisor times events with it), not a family's own input
FAMILIES = tuple(REQUIRED)


@dataclass
class FamilyInput:
    symbol: str
    metrics: dict                                   # data.metrics[symbol]: {name: metric result}
    events: list[dict]                              # the CHECKED events: {class, direction, source, age_bars, reaction, reaction_atr, event_volume_ratio, origin, quote}
    signals: list[dict]                             # the verified, counted signals (v0 shape; E-items have `topic`)
    quality: Quality
    regime: str
    config: AdvisorConfig
    today: date
    drop_cause: dict | None = None                  # {class, source, quote}: the model's checked explanation of a recent fall (reversion)
    recent_company_sources: int = 0                 # sources about this company in the last 5 bars (reversion: "no company-specific news" means it is sympathy)
    current_dilution_ids: frozenset = frozenset()   # the ids of the filing-evidence items that show stock being issued NOW: a derived item (shares outstanding rose, proceeds from an issuance) or a passage of an 8-K. A passage of a 10-Q or 10-K is not one (see capital_return)
    theme: dict | None = None                       # Case 2: {articles, positive, neutral, negative, article_share, basket_ret_21d: {symbol: value}}
    extra: dict = field(default_factory=dict)

    def v(self, name: str):
        return value_of(self.metrics, name)


def _start(family: str, inp: FamilyInput) -> FamilyResult:
    """The result with everything that does not depend on the rules: the inputs that were missing, the completeness, the features, the horizon, the prior and the regime."""
    cfg = getattr(inp.config.families, family)
    required, optional = REQUIRED[family], OPTIONAL[family]
    waived = ["debt_to_equity"] if "debt_to_equity" in required and negative_equity(inp.metrics) else []   # entry 54: a stockholders' deficit makes debt / equity meaningless: not required, not tested
    required = tuple(n for n in required if n not in waived)
    missing = [n for n in required if inp.v(n) is None]
    any_of = ANY_OF.get(family)
    if any_of and all(inp.v(n) is None for n in any_of):
        missing.append(" or ".join(any_of))
    present_optional = [n for n in optional if inp.v(n) is not None]
    prior = getattr(cfg, "prior_20d", None)
    if prior is None:
        prior = cfg.prior_event
    return FamilyResult(
        family=family, computable=not missing, missing=missing, horizon_bars=cfg.horizon, prior_20d=prior, basis_ref=cfg.basis_ref,
        regime_multiplier=inp.config.regime.multipliers[family][inp.regime], completeness=round(len(present_optional) / len(optional), 4) if optional else None,
        features_used={n: inp.v(n) for n in (*required, *optional) if inp.v(n) is not None},
        waived=waived, notes=[*([WAIVED_NOTE] if waived else []), *(f"{n}: {note_of(inp.metrics, n)}" for n in missing if n in inp.metrics)])


def _finish(result: FamilyResult, inp: FamilyInput, score: float, direction: str, prior: float | None = None, haircut: float = 1.0) -> FamilyResult:
    """The family passed: set its score, direction and edge. `edge_20d = prior x regime x score x quality (x haircut)`, negative when bearish."""
    if prior is not None:
        result.prior_20d = prior
    result.passed, result.direction, result.score = True, direction, round(max(0.0, min(1.0, score * haircut)), 4)
    sign = 1.0 if direction == "bullish" else -1.0
    result.edge_20d = round(sign * abs(result.prior_20d) * result.regime_multiplier * result.score * inp.quality.multiplier, 8)
    return result


def _clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def _fail(result: FamilyResult, failures: list[str]) -> FamilyResult:
    result.passed = False
    result.notes.extend(failures)
    return result


# ---------------------------------------------------------------- trend

def trend(inp: FamilyInput) -> FamilyResult:
    """Momentum that is not chased. Pass when ALL: 12-1 momentum >= 15%, the price within 10% of its 52-week high, above the 200-day mean, NOT up more than 15% in the last 21 bars (the
    anti-chase rule), RSI under 75, residual momentum (against the sector) positive and %B at most 1.05 when they are known.
    Score = 0.40 + 0.20 x clip(momentum / 50%) + 0.10 each for: MACD histogram positive, ADX >= 20, OBV rising, volatility under 35%. A missing optional input adds nothing and lowers completeness."""
    result, cfg = _start("trend", inp), inp.config.families.trend
    if not result.computable:
        return result
    mom, high, above, ret21, rsi = inp.v("mom_12_1"), inp.v("pct_of_52w_high"), inp.v("above_sma_200"), inp.v("ret_21d"), inp.v("rsi_14")
    residual, pct_b = inp.v("residual_mom_12_1"), inp.v("bollinger_pct_b")
    failures = [msg for ok, msg in (
        (mom >= cfg.min_mom_12_1, f"mom_12_1 {mom:.3f} is under {cfg.min_mom_12_1}"), (high >= cfg.min_pct_of_52w_high, f"pct_of_52w_high {high:.3f} is under {cfg.min_pct_of_52w_high}"),
        (above is True, "the price is not above its 200-day mean"), (ret21 <= cfg.max_ret_21d, f"ret_21d {ret21:.3f} is over {cfg.max_ret_21d}: it already ran (anti-chase)"),
        (rsi < cfg.max_rsi, f"rsi_14 {rsi:.1f} is {cfg.max_rsi} or more"),
        (residual is None or residual > 0, f"residual_mom_12_1 {residual} is not above the sector's"), (pct_b is None or pct_b <= cfg.max_pct_b, f"bollinger_pct_b {pct_b} is over {cfg.max_pct_b}")) if not ok]
    if failures:
        return _fail(result, failures)
    macd, adx, obv, vol = inp.v("macd_hist"), inp.v("adx_14"), inp.v("obv_20d_change"), inp.v("vol_63d_ann")
    score = (cfg.base + cfg.w_mom * _clip(mom / cfg.mom_scale) + cfg.w_macd * (macd is not None and macd > 0) + cfg.w_adx * (adx is not None and adx >= cfg.adx_min)
             + cfg.w_obv * (obv is not None and obv > 0) + cfg.w_lowvol * (vol is not None and vol < cfg.lowvol))
    return _finish(result, inp, score, "bullish")


# ---------------------------------------------------------------- event

def _counted_events(inp: FamilyInput) -> list[dict]:
    """The checked events that count for the event family: a bullish or bearish class, young enough, with its timing known. Neutral classes are causes, not trades."""
    cfg = inp.config.families.event
    return [e for e in inp.events if e.get("class") in (*BULLISH, *BEARISH) and e.get("age_bars") is not None and e["age_bars"] <= cfg.max_age_bars]


def event(inp: FamilyInput) -> FamilyResult:
    """A corporate event the price has not fully reacted to. Needs a verified event no more than 10 bars old. A bullish event passes when the move since it is at most 3 ATRs
    (above that it is priced in); a bearish one is always a (bearish) vote. When a bullish and a bearish event are both young, the newer one votes and the other is a contradiction
    in the notes. Score = 0.50 + 0.30 x (1 - clip(reaction in ATRs / 3)) + 0.20 if the volume on the event bar was at least twice normal; x 0.5 for a company worth more than $200 billion
    (a big company's surprise is absorbed faster). The surprise path (`sue_4q`) needs the earnings dates, which are not available: it never fires in this build."""
    result, cfg = _start("event", inp), inp.config.families.event
    counted = _counted_events(inp)
    if not counted:
        result.computable = False
        result.missing.append(f"a verified bullish or bearish event of the last {cfg.max_age_bars} bars (or sue_4q)")
        result.notes.append("no counted event: " + (", ".join(f"{e['class']} age {e.get('age_bars')}" for e in inp.events) or "none were verified"))
        return result
    if not result.computable:
        return result
    newest = min(e["age_bars"] for e in counted)
    chosen = sorted((e for e in counted if e["age_bars"] == newest), key=lambda e: (e["direction"] != "bearish", abs(e.get("reaction") or 0.0)))[0]  # a tie goes to the bearish one: the cautious reading
    for other in counted:
        if other["direction"] != chosen["direction"]:
            result.notes.append(f"contradiction: {chosen['class']} (age {chosen['age_bars']}) against {other['class']} (age {other['age_bars']}): the newer votes")
    atr = inp.v("atr_pct")
    reaction = chosen.get("reaction")
    reaction_atr = abs(reaction) / atr if reaction is not None and atr else None
    chosen["reaction_atr"] = reaction_atr
    result.evidence_ids = [f"[{chosen['source']}]" if isinstance(chosen.get("source"), int) else str(chosen.get("source"))]
    result.counter_evidence_ids = [f"[{o['source']}]" if isinstance(o.get("source"), int) else str(o.get("source")) for o in counted if o["direction"] != chosen["direction"]]
    result.features_used.update({"event_class": chosen["class"], "event_age_bars": chosen["age_bars"], "reaction_atr": reaction_atr, "event_volume_ratio": chosen.get("event_volume_ratio")})
    if chosen["direction"] == "bullish" and (reaction_atr is None or reaction_atr > cfg.max_reaction_atr):
        why = "the reaction since the event cannot be computed" if reaction_atr is None else f"the move since the event is {reaction_atr:.1f} ATRs (over {cfg.max_reaction_atr:g}): priced in"
        return _fail(result, [f"{chosen['class']}: {why}"])
    volume = chosen.get("event_volume_ratio")
    score = cfg.base + cfg.w_unpriced * (1 - _clip((reaction_atr or 0.0) / cfg.max_reaction_atr)) + cfg.w_volume * (volume is not None and volume >= cfg.volume_ratio_min)
    cap = inp.v("market_cap")
    haircut = cfg.large_cap_factor if cap is not None and cap > cfg.large_cap else 1.0
    if haircut != 1.0:
        result.notes.append(f"score halved: market cap ${cap:,.0f} is over ${cfg.large_cap:,.0f}")
    return _finish(result, inp, score, chosen["direction"], prior=cfg.prior_event, haircut=haircut)


# ---------------------------------------------------------------- reversion

def reversion(inp: FamilyInput) -> FamilyResult:
    """A sound company on sale. Pass when ALL: it is oversold (5-day return <= -7%, or z-score <= -2, or RSI <= 30); the fall is its own and not the sector's (residual <= -5% with the sector
    not down 3%, else the stock down 7% with SPY not down 3%); the company is sound (ROE > 10, debt to equity < 1.5, current ratio > 1, revenue growth > 0); it is in an uptrend (above its
    200-day mean, else its 50-day, else the filter is skipped with a note); the cause is not about the company (the model's checked `drop_cause` is a neutral class, or there is no news
    about the company in the last 5 bars: sympathy) and not a fundamental one (guidance cut, offering, restatement, going concern, weakness, earnings miss); and no earnings-quality check
    switched reversion off. Score = 0.40 + 0.20 x clip((-z - 2) / 2) + 0.10 each for: RSI <= 30, %B <= 0, volume >= 2x, a reversal candle."""
    result, cfg = _start("reversion", inp), inp.config.families.reversion
    if not result.computable:
        return result
    ret5, z, rsi = inp.v("ret_5d"), inp.v("zscore_21d"), inp.v("rsi_14")
    failures = []
    if not (ret5 <= cfg.ret_5d or z <= cfg.zscore or rsi <= cfg.rsi):
        failures.append(f"not oversold: ret_5d {ret5:.3f}, zscore_21d {z:.2f}, rsi_14 {rsi:.1f}")
    residual, sector, spy = inp.v("residual_5d"), inp.v("sector_ret_5d"), inp.v("spy_ret_5d")
    if residual is not None and sector is not None:
        if not (residual <= cfg.residual_5d and sector > cfg.sector_ret_5d_min):
            failures.append(f"not idiosyncratic: residual_5d {residual:.3f} against the sector's {sector:.3f}")
    elif spy is None:
        failures.append("cannot tell whether the fall is the market's: spy_ret_5d is missing")
    elif not (ret5 <= cfg.ret_5d and spy > cfg.spy_ret_5d_min):
        failures.append(f"not idiosyncratic: ret_5d {ret5:.3f} with SPY at {spy:.3f}")
    leverage_ok = "debt_to_equity" in result.waived or inp.v("debt_to_equity") < cfg.debt_to_equity
    if not (inp.v("roe") > cfg.roe and leverage_ok and inp.v("current_ratio") > cfg.current_ratio and inp.v("revenue_growth") > cfg.revenue_growth):
        failures.append(f"not sound: roe {inp.v('roe')}, debt_to_equity {'waived' if 'debt_to_equity' in result.waived else inp.v('debt_to_equity')}, current_ratio {inp.v('current_ratio')}, revenue_growth {inp.v('revenue_growth')}")
    if inp.v("above_sma_200") is not None:
        if not inp.v("above_sma_200"):
            failures.append("not in an uptrend: below the 200-day mean")
    elif inp.v("above_sma_50") is not None:
        if not inp.v("above_sma_50"):
            failures.append("not in an uptrend: below the 50-day mean")
    else:
        result.notes.append(f"trend filter skipped: {note_of(inp.metrics, 'above_sma_50')}")
    cause = (inp.drop_cause or {}).get("class")
    fundamental = cause in cfg.fundamental_causes or any(e.get("class") in cfg.fundamental_causes and (e.get("age_bars") or 0) <= cfg.cause_lookback_bars for e in inp.events)
    if fundamental:
        failures.append("fundamental cause: " + (cause if cause in cfg.fundamental_causes else next(e["class"] for e in inp.events if e.get("class") in cfg.fundamental_causes)))
    elif cause in NEUTRAL:
        result.notes.append(f"cause: {cause}")
    elif inp.recent_company_sources > 0:
        failures.append(f"company-specific news in the last {cfg.cause_lookback_bars} bars ({inp.recent_company_sources}) and no neutral cause was given")
    if "reversion" in inp.quality.turned_off:
        failures.append("switched off by an earnings-quality check: " + ", ".join(h.check for h in inp.quality.haircuts if h.check in ("accrual_ratio", "receivables_vs_sales", "inventory_vs_sales", "cfo_negative_with_profit")))
    if failures:
        return _fail(result, failures)
    pct_b, volume, candle = inp.v("bollinger_pct_b"), inp.v("volume_ratio_1d"), inp.v("candle_reversal_1d")
    score = (cfg.base + cfg.w_zscore * _clip((-z - cfg.zscore_floor) / cfg.zscore_span) + cfg.w_rsi * (rsi <= cfg.rsi) + cfg.w_pct_b * (pct_b is not None and pct_b <= 0)
             + cfg.w_volume * (volume is not None and volume >= cfg.volume_ratio) + cfg.w_candle * (candle is True))
    if cause in NEUTRAL and inp.drop_cause:
        result.evidence_ids = [f"[{inp.drop_cause['source']}]"]
    return _finish(result, inp, score, "bullish")


# ---------------------------------------------------------------- capital_return

def capital_return(inp: FamilyInput) -> FamilyResult:
    """A company that returns cash and does not dilute. Pass when ALL: the net payout yield (buybacks + dividends - stock issued, over market cap) is at least 3% (or, when it cannot be
    computed, the share count fell at least 0.5%); the share count did not rise; the company is not issuing stock now; debt to equity <= 2; earnings yield > 0; dividends are covered
    (payout coverage < 1) and revenue is not falling more than 5%, when known.
    Score = 0.40 + 0.30 x clip(payout yield / 8%) + 0.10 if the share count fell more than 1% + 0.10 for a buyback authorization or dividend raise in the last 60 bars (halved when the
    authorization is larger than the company's cash) + 0.10 for revenue growth. Without the payout yield the 0.30 term is 0.15 if the share count fell more than 1%.
    A company that IS issuing stock (dilution_current, or a verified bearish dilution item of its filings that shows stock being issued NOW: `current_dilution_ids`) is a BEARISH vote of this family,
    score 0.6. A dilution PASSAGE of a 10-Q is not such an item: the real Procter & Gamble run showed the model citing "the diluted shares include the dilutive effect of stock options" (the earnings-per-share
    note every 10-Q has) and flipping the family bearish; spec 8.6 says "a dilution E-item that passes the Advisor's check", and entry 51 narrows it as `dilution_current` is narrowed."""
    result, cfg = _start("capital_return", inp), inp.config.families.capital_return
    if not result.computable:
        return result
    npy, shares, growth = inp.v("net_payout_yield_ttm"), inp.v("share_count_change"), inp.v("revenue_growth")
    dilution_items = [s for s in inp.signals if s.get("topic") == "dilution" and s.get("direction") == "bearish" and s.get("evidence_id") in inp.current_dilution_ids]
    if inp.v("dilution_current") is True or dilution_items:
        result.evidence_ids = [s.get("evidence_id") or f"[{s.get('source')}]" for s in dilution_items]
        result.notes.append("issuing stock: " + ("dilution_current" if inp.v("dilution_current") is True else "a verified dilution item of the filings"))
        return _finish(result, inp, cfg.bearish_score, "bearish", prior=cfg.prior_bearish)
    failures = []
    if npy is not None and npy < cfg.min_net_payout_yield:
        failures.append(f"net_payout_yield_ttm {npy:.4f} is under {cfg.min_net_payout_yield}")
    if npy is None and not (shares is not None and shares <= cfg.min_share_count_change):
        failures.append(f"net_payout_yield_ttm is missing and share_count_change {shares} is not at or under {cfg.min_share_count_change}")
    if shares is not None and shares > 0:
        failures.append(f"share_count_change {shares:.4f}: the share count rose")
    if "debt_to_equity" not in result.waived and inp.v("debt_to_equity") > cfg.max_debt_to_equity:
        failures.append(f"debt_to_equity {inp.v('debt_to_equity')} is over {cfg.max_debt_to_equity:g}")
    if inp.v("earnings_yield") <= 0:
        failures.append("earnings_yield is not positive")
    coverage = inp.v("payout_coverage")
    if coverage is not None and coverage >= cfg.max_payout_coverage:
        failures.append(f"payout_coverage {coverage:.2f}: dividends are not covered by earnings")
    if growth is not None and growth < cfg.min_revenue_growth:
        failures.append(f"revenue_growth {growth} is under {cfg.min_revenue_growth}")
    if failures:
        return _fail(result, failures)
    fresh = next((e for e in inp.events if e.get("class") in ("buyback_authorization", "dividend_increase") and e.get("age_bars") is not None and e["age_bars"] <= cfg.event_bars), None)
    event_term = cfg.w_event if fresh else 0.0
    if fresh and fresh["class"] == "buyback_authorization":
        amount, cash = amount_of(fresh.get("quote")), inp.v("cash_balance")
        if amount is not None and cash is not None and cash < amount:
            event_term /= 2
            result.notes.append(f"authorization exceeds cash: ${amount:,.0f} against ${cash:,.0f}")
    if fresh:
        result.evidence_ids = [f"[{fresh['source']}]" if isinstance(fresh.get("source"), int) else str(fresh.get("source"))]
    payout_term = cfg.w_payout * _clip(npy / cfg.payout_scale) if npy is not None else cfg.w_missing_payout * (shares is not None and shares < cfg.shares_bonus)
    if npy is None:
        result.notes.append("net payout yield not computable: the share count stands in for it and completeness is lower")
    score = cfg.base + payout_term + cfg.w_shares * (shares is not None and shares < cfg.shares_bonus) + event_term + cfg.w_growth * (growth is not None and growth > 0)
    return _finish(result, inp, score, "bullish")


# ---------------------------------------------------------------- theme

def theme(inp: FamilyInput) -> FamilyResult:
    """Case 2: a company of the theme's basket that has not moved with it. Needs the theme's discovery result (`inp.theme`) and a verified signal about the company. Pass when ALL: the theme
    has at least 3 articles about it; at least 60% of the articles on the theme are positive; the company's 21-bar return is at or under the basket's median (the laggard rule); the quality
    multiplier is at least 0.5. A company that fails ONLY the laggard rule is a `leader`: kept as a candidate, not recommended.
    Score = 0.40 + 0.20 x clip(its share of the basket's articles / 30%) + 0.20 x (1 - its place in the basket's 21-bar return range) + 0.10 if 80% of the articles are positive + 0.10 if
    today's volume is 1 to 2 times normal."""
    result, cfg = _start("theme", inp), inp.config.families.theme
    data = inp.theme
    if data is None:
        result.computable = False
        result.missing.append("the theme's discover result (a theme order)")
        return result
    if not any(s.get("verified") and s.get("counted") for s in inp.signals):
        result.computable = False
        result.missing.append("a verified signal naming the company in the theme's sources")
    if not result.computable:
        return result
    total = data["positive"] + data["neutral"] + data["negative"]
    share = data["positive"] / total if total else 0.0
    ret21 = inp.v("ret_21d")
    basket = sorted(data["basket_ret_21d"].values())
    median = (basket[(len(basket) - 1) // 2] + basket[len(basket) // 2]) / 2 if basket else None
    failures = [msg for ok, msg in ((data["articles"] >= cfg.min_articles, f"only {data['articles']} articles on the theme (at least {cfg.min_articles})"),
                                    (share >= cfg.min_positive_share, f"{share:.0%} of the articles on the theme are positive (at least {cfg.min_positive_share:.0%})"),
                                    (inp.quality.multiplier >= cfg.min_quality, f"quality multiplier {inp.quality.multiplier:.2f} is under {cfg.min_quality}")) if not ok]
    laggard = median is not None and ret21 <= median
    if median is None:
        failures.append("the basket has no 21-bar returns to compare with")
    result.features_used.update({"theme_articles": data["articles"], "theme_positive_share": round(share, 4), "basket_median_ret_21d": median})
    if not laggard and median is not None:
        if failures:
            return _fail(result, failures + [f"not a laggard: ret_21d {ret21:.3f} is over the basket median {median:.3f}"])
        result.notes.append(f"leader: already moved {ret21:.0%} in 21 bars")
        result.features_used["theme_role"] = "leader"
        return _fail(result, [f"already moved {ret21:.0%} in 21 bars (the basket median is {median:.0%})"])
    if failures:
        return _fail(result, failures)
    low, high = basket[0], basket[-1]
    place = (ret21 - low) / (high - low) if high > low else 0.0
    volume = inp.v("volume_ratio_1d")
    score = (cfg.base + cfg.w_article * _clip(data["article_share"] / cfg.article_scale) + cfg.w_laggard * (1 - _clip(place)) + cfg.w_positive * (share >= cfg.positive_high)
             + cfg.w_volume * (volume is not None and 1 <= volume <= 2))
    result.features_used["theme_role"] = "laggard"
    return _finish(result, inp, score, "bullish")


EVALUATORS = {"trend": trend, "event": event, "reversion": reversion, "capital_return": capital_return, "theme": theme}


def evaluate(family: str, inp: FamilyInput) -> FamilyResult:
    return EVALUATORS[family](inp)


def required_metrics(families=FAMILIES) -> list[str]:
    """Every metric name the given families read (required and optional), plus the ones every candidate needs: what the Advisor asks the Data Clerk for."""
    names: list[str] = []
    for family in families:
        names.extend((*REQUIRED[family], *OPTIONAL[family], *ANY_OF.get(family, ())))
    names.extend(EXTRA_REQUIRED)
    return list(dict.fromkeys(names))
