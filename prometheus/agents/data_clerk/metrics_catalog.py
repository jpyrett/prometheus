"""The metrics catalogue: every number the Advisor's strategies may ask the Data Clerk to calculate (Advisor v1 spec sections 4 and 5, design choices entry 51).

    results = compute(["SNDK", "PG"], requests, bars, financials, evidence, today)     # {symbol: {metric name: result}}

Adding a metric = one entry in `METRICS` and one test. Nothing here rates anything and no model is involved.

THE RESULT OF ONE METRIC (spec 4.2), the same shape for all:

    {"value": float | bool | str | None (a list for `recent_bars`), "unit": str, "as_of": "YYYY-MM-DD" | None, "window": str, "source": str, "inputs_summary": dict, "note": str | None}

  * `source` says where the number came from: `day_aggs:<first date>..<last date>` (the stored bars), `finnhub_metric:<date>`, `filing:<accession>`, or `derived:<the inputs joined by +>`;
  * `inputs_summary` holds what a reader needs to recompute the number (for RSI the bars used and the last close; for a payout yield the dollars of each cash flow);
  * `note` is a gap in words, never a judgement. `value: None` always has a note.

A metric that cannot be computed is `value: None` with the reason (`needs 30 bars, 17 stored`, `no sector ETF given`, `no market capitalization from Finnhub`): it is a result, not an
error. A test walks every result and refuses the keys score, stance, sentiment, recommendation, rating and verdict.
"""
import functools
import json
from dataclasses import dataclass
from datetime import date
from typing import Callable

from prometheus.agents.data_clerk import fundamentals as fund
from prometheus.agents.data_clerk import technicals as tech
from prometheus.agents.data_clerk.bars import Bar, BarStore
from prometheus.agents.data_clerk.data_toolbox import FINANCIAL_METRICS
from prometheus.agents.data_clerk.fundamentals import Missing
from prometheus.rag.config import REPO_ROOT

FORBIDDEN_KEYS = ("score", "stance", "sentiment", "recommendation", "rating", "verdict")  # no metric result may use these as a key
SPY = "SPY"
RECENT_BARS = 35        # how many of the latest bars the `recent_bars` metric carries: enough to date an event up to 10 bars old and to average the 20 bars of volume before it
SECTOR_ETF_FILE = REPO_ROOT / "configs" / "sector_etfs.json"


@functools.lru_cache(maxsize=None)
def sector_etfs() -> tuple[str, ...]:
    """The 11 sector ETF tickers (the keys of configs/sector_etfs.json; a key that starts with an underscore is a note)."""
    return tuple(k for k in json.loads(SECTOR_ETF_FILE.read_text(encoding="utf-8")) if not k.startswith("_"))


@dataclass
class Context:
    """What one symbol's metrics are computed from."""
    symbol: str
    today: date
    bars: list[Bar]                     # the symbol's bars on or before today, oldest first
    spy: list[Bar]
    sector_etf: str | None = None       # the ticker of the symbol's sector ETF, when the work order gave one
    sector_bars: list[Bar] | None = None
    financials: dict | None = None      # the result of `Financials.get`: {symbol, as_of, metrics: {name: {value, unit, description}}}
    evidence: dict | None = None        # the result of `filing_signals.recommendation_evidence`


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str                          # price | oscillator | risk | market | fundamentals | filings | unavailable
    inputs: tuple[str, ...]             # which of bars, financials, evidence the metric reads
    unit: str
    min_bars: int
    description: str
    fn: Callable[[Context], tuple]      # -> (value, inputs_summary); raises Missing(reason)
    window: str


# ---------------------------------------------------------------- the entries

def _bar_spec(name, group, unit, fn, description, window=None):
    minimum = tech.MIN_BARS.get(name, 1)
    return MetricSpec(name, group, ("bars",), unit, minimum, description, fn, window or f"last {minimum} bars")


def _last(c: Context) -> dict:
    return {"bars": len(c.bars), "last_close": c.bars[-1].close}


def _simple(fn):
    """A technical indicator f(bars) -> value, with the bars it used as the inputs summary."""
    def run(c: Context):
        return fn(c.bars), _last(c)
    return run


def _need(series: list[Bar] | None, n: int, who: str) -> list[Bar]:
    if not series or len(series) < n:
        raise Missing(f"needs {n} bars of {who}, {len(series or [])} stored")
    return series


def _sector(c: Context) -> list[Bar]:
    if not c.sector_etf:
        raise Missing("no sector ETF given for this symbol (WorkOrder.sector)")
    return _need(c.sector_bars, 2, c.sector_etf)


def _market(fn, n: int):
    """A metric of SPY's own bars."""
    def run(c: Context):
        _need(c.spy, n, SPY)
        return fn(c.spy), {"bars": len(c.spy), "last_close": c.spy[-1].close}
    return run


def _beta(index: int):
    def run(c: Context):
        _need(c.spy, tech.MIN_BARS["beta_63d"], SPY)
        value = tech.beta_and_corr(c.bars, c.spy)[index]
        if value is None:
            raise Missing("fewer than 64 dates shared with SPY, or SPY did not move")
        return value, {"bars": len(c.bars), "spy_bars": len(c.spy)}
    return run


def _sector_return(n: int):
    def run(c: Context):
        etf = _sector(c)
        _need(etf, n + 1, c.sector_etf)
        return tech.ret(etf, n), {"etf": c.sector_etf, "bars": len(etf)}
    return run


def _residual(fn, n: int):
    """The symbol's value minus the sector ETF's over the same window."""
    def run(c: Context):
        etf = _sector(c)
        _need(etf, n, c.sector_etf)
        own, theirs = fn(c.bars), fn(etf)
        if own is None or theirs is None:
            raise Missing(f"needs {n} bars of {c.symbol} and of {c.sector_etf}")
        return own - theirs, {"symbol_value": own, "etf": c.sector_etf, "etf_value": theirs}
    return run


def _finnhub(name: str):
    def run(c: Context):
        if c.financials is None:
            raise Missing(f"no Finnhub metric payload for {c.symbol} (it was not fetched or the call failed)")
        value = c.financials["metrics"].get(name, {}).get("value")
        if value is None:
            raise Missing(f"Finnhub has no value for {name} for {c.symbol}")
        return value, {}
    return run


def _evidence(fn):
    def run(c: Context):
        if c.evidence is None:
            raise Missing(f"no stored filings of {c.symbol} (fetch_filings has not been run for it)")
        return fn(c.evidence)
    return run


def _finnhub_value(c: Context, name: str) -> float | None:
    return ((c.financials or {}).get("metrics", {}).get(name) or {}).get("value")


def _need_evidence(c: Context) -> dict:
    if c.evidence is None:
        raise Missing(f"no stored filings of {c.symbol} (fetch_filings has not been run for it)")
    return c.evidence


def _sector_etf(c: Context):
    if not c.sector_etf:
        raise Missing("no sector ETF given for this symbol (WorkOrder.sector)")
    return c.sector_etf, {}


def _earnings_yield(c: Context):
    if c.financials is None:
        raise Missing(f"no Finnhub metric payload for {c.symbol} (it was not fetched or the call failed)")
    return fund.earnings_yield(_finnhub_value(c, "pe_ratio"))


def _net_payout(c: Context):
    return fund.net_payout_yield_ttm(_need_evidence(c), _finnhub_value(c, "market_cap"))


def _dilution(c: Context):
    return fund.dilution_current(_need_evidence(c), c.today)


UNAVAILABLE = {  # names the strategies read but that this build cannot compute yet, each with the reason: they are in the catalogue so that a gap is a visible result, not a missing key
    "sue_4q": "earnings history is not available: it needs the Finnhub earnings endpoints (not added yet)",
    "eps_surprise_last": "earnings history is not available: it needs the Finnhub earnings endpoints (not added yet)",
    "days_to_next_earnings": "the earnings calendar is not available: it needs the Finnhub earnings endpoints (not added yet)",
    "earnings_move_abs": "the earnings dates are not available: it needs the Finnhub earnings endpoints (not added yet)",
    "median_spread_bps_1d": "intraday quotes are not read in this build (spec 5.6, phase 10)",
    "breadth_200": "needs an index universe (configs/universe_sp500.json), which does not exist yet",
    "risk_factor_change": "not built: needs the embedding comparison of the risk factors of the two latest 10-Ks (spec 5.5)",
}


def _unavailable(name: str):
    def run(c: Context):
        raise Missing(UNAVAILABLE[name])
    return run


def _build() -> dict[str, MetricSpec]:
    specs: list[MetricSpec] = []
    for n, name in ((1, "ret_1d"), (5, "ret_5d"), (21, "ret_21d"), (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")):
        specs.append(_bar_spec(name, "price", "fraction", _simple(lambda b, n=n: tech.ret(b, n)), f"close-to-close return over {n} sessions"))
    specs += [
        _bar_spec("recent_bars", "price", "[date, close, volume]", lambda c: ([[b.date.isoformat(), b.close, b.volume] for b in c.bars[-RECENT_BARS:]], _last(c)),
                  f"the latest {RECENT_BARS} bars as [date, close, volume], oldest first: the Advisor dates an event and measures its reaction against them (the one metric whose value is a list)", f"last {RECENT_BARS} bars"),
        _bar_spec("last_close", "price", "USD", lambda c: (c.bars[-1].close, {"bar_date": c.bars[-1].date.isoformat()}), "the latest stored close", "1 bar"),
        _bar_spec("mom_12_1", "price", "fraction", _simple(tech.mom_12_1), "12-month return skipping the latest month: close[-21] / close[-252] - 1"),
        _bar_spec("pct_of_52w_high", "price", "fraction", _simple(tech.pct_of_52w_high), "close / the highest high of the last 252 bars"),
        _bar_spec("sma_50", "price", "USD", _simple(lambda b: tech.sma(b, 50)), "simple mean of the last 50 closes"),
        _bar_spec("sma_200", "price", "USD", _simple(lambda b: tech.sma(b, 200)), "simple mean of the last 200 closes"),
        _bar_spec("above_sma_50", "price", "bool", _simple(lambda b: tech.above_sma(b, 50)), "the close is above its 50-bar mean"),
        _bar_spec("above_sma_200", "price", "bool", _simple(lambda b: tech.above_sma(b, 200)), "the close is above its 200-bar mean"),
        _bar_spec("breakout_20d", "price", "bool", _simple(tech.breakout_20d), "the close is at or above the highest high of the 20 sessions before"),
        _bar_spec("gap_1d", "price", "fraction", _simple(tech.gap_1d), "today's open over yesterday's close, minus 1"),
        _bar_spec("candle_reversal_1d", "price", "bool", _simple(tech.candle_reversal_1d), "traded below yesterday's low and closed above today's open"),
        _bar_spec("rsi_14", "oscillator", "0-100", _simple(tech.rsi_14), "Wilder RSI, 14 periods"),
        _bar_spec("macd_hist", "oscillator", "USD", _simple(tech.macd_hist), "MACD line (EMA12 - EMA26) minus its EMA9"),
        _bar_spec("bollinger_pct_b", "oscillator", "ratio", _simple(tech.bollinger_pct_b), "(close - lower band) / (upper - lower), bands SMA20 +- 2 population std"),
        _bar_spec("zscore_21d", "oscillator", "std", _simple(tech.zscore_21d), "(close - mean of 21) / sample std of 21"),
        _bar_spec("adx_14", "oscillator", "0-100", _simple(tech.adx_14), "Wilder ADX, 14 periods"),
        _bar_spec("atr_14", "risk", "USD", _simple(tech.atr_14), "Wilder average true range, 14 periods"),
        _bar_spec("atr_pct", "risk", "fraction", _simple(tech.atr_pct), "atr_14 / close"),
        _bar_spec("obv_20d_change", "oscillator", "fraction", _simple(tech.obv_20d_change), "change of on-balance volume over 20 bars over the volume of those bars"),
        _bar_spec("stoch_k_14", "oscillator", "0-100", _simple(tech.stoch_k_14), "stochastic %K, 14 periods"),
        _bar_spec("vol_21d_ann", "risk", "fraction", _simple(lambda b: tech.vol_ann(b, 21)), "annualized sample std of the last 21 daily log returns"),
        _bar_spec("vol_63d_ann", "risk", "fraction", _simple(lambda b: tech.vol_ann(b, 63)), "annualized sample std of the last 63 daily log returns"),
        _bar_spec("max_drawdown_63d", "risk", "fraction", _simple(tech.max_drawdown_63d), "deepest fall from a running peak in the last 63 bars (negative)"),
        _bar_spec("beta_63d", "risk", "ratio", _beta(0), "OLS slope of the daily returns on SPY's, last 63 returns"),
        _bar_spec("corr_63d_spy", "risk", "ratio", _beta(1), "correlation of the same daily returns with SPY's"),
        _bar_spec("adv_20d_dollars", "risk", "USD", _simple(tech.adv_20d_dollars), "mean of close x volume over 20 bars"),
        _bar_spec("volume_ratio_1d", "risk", "ratio", _simple(tech.volume_ratio_1d), "today's volume over the mean of the 20 sessions before"),
        # the sector ETF and the market (SPY)
        MetricSpec("sector_etf", "market", ("bars",), "ticker", 0, "the sector ETF the work order gave for this symbol", _sector_etf, "n/a"),
        MetricSpec("sector_ret_5d", "market", ("bars",), "fraction", 0, "the sector ETF's 5-session return", _sector_return(5), "6 bars of the ETF"),
        MetricSpec("sector_ret_21d", "market", ("bars",), "fraction", 0, "the sector ETF's 21-session return", _sector_return(21), "22 bars of the ETF"),
        MetricSpec("sector_ret_63d", "market", ("bars",), "fraction", 0, "the sector ETF's 63-session return", _sector_return(63), "64 bars of the ETF"),
        MetricSpec("residual_5d", "market", ("bars",), "fraction", 6, "the symbol's 5-session return minus its sector ETF's", _residual(lambda b: tech.ret(b, 5), 6), "6 bars"),
        MetricSpec("residual_21d", "market", ("bars",), "fraction", 22, "the symbol's 21-session return minus its sector ETF's", _residual(lambda b: tech.ret(b, 21), 22), "22 bars"),
        MetricSpec("residual_mom_12_1", "market", ("bars",), "fraction", 252, "the symbol's 12-1 momentum minus its sector ETF's", _residual(tech.mom_12_1, 252), "252 bars"),
        MetricSpec("spy_ret_5d", "market", ("bars",), "fraction", 0, "SPY's 5-session return", _market(lambda b: tech.ret(b, 5), 6), "6 bars of SPY"),
        MetricSpec("spy_ret_21d", "market", ("bars",), "fraction", 0, "SPY's 21-session return", _market(lambda b: tech.ret(b, 21), 22), "22 bars of SPY"),
        MetricSpec("spy_above_sma_200", "market", ("bars",), "bool", 0, "SPY's close is above its 200-bar mean", _market(lambda b: tech.above_sma(b, 200), 200), "200 bars of SPY"),
        MetricSpec("spy_vol_21d_ann", "market", ("bars",), "fraction", 0, "SPY's annualized 21-day volatility", _market(lambda b: tech.vol_ann(b, 21), 22), "22 bars of SPY"),
        MetricSpec("spy_drawdown_63d", "market", ("bars",), "fraction", 0, "SPY's deepest fall from a peak in 63 bars", _market(tech.max_drawdown_63d, 63), "63 bars of SPY"),
    ]
    for name, (_, unit, description) in FINANCIAL_METRICS.items():
        specs.append(MetricSpec(name, "fundamentals", ("financials",), unit, 0, description, _finnhub(name), "as Finnhub reports"))
    specs += [
        MetricSpec("earnings_yield", "fundamentals", ("financials",), "fraction", 0, "1 / pe_ratio when the P/E is positive",
                   _earnings_yield, "TTM"),
        MetricSpec("net_payout_yield_ttm", "filings", ("evidence", "financials"), "fraction", 0, "(buybacks + dividends - stock issued) / market cap, annualized",
                   _net_payout, "year to date, annualized"),
        MetricSpec("cash_balance", "filings", ("evidence",), "USD", 0, "cash and cash equivalents of the latest stored balance sheet, in dollars", _evidence(fund.cash_balance), "latest balance sheet"),
        MetricSpec("share_count_change", "filings", ("evidence",), "fraction", 0, "change of the shares outstanding between the two stored balance sheets", _evidence(fund.share_count_change), "latest balance sheet vs the earlier one"),
        MetricSpec("payout_coverage", "filings", ("evidence",), "ratio", 0, "dividends paid / net income, year to date", _evidence(fund.payout_coverage), "year to date"),
        MetricSpec("sbc_to_revenue", "filings", ("evidence",), "fraction", 0, "share-based compensation / revenue, year to date", _evidence(fund.sbc_to_revenue), "year to date"),
        MetricSpec("accrual_ratio", "filings", ("evidence",), "ratio", 0, "(net income - cash from operations) / total assets", _evidence(fund.accrual_ratio), "year to date"),
        MetricSpec("receivables_vs_sales", "filings", ("evidence",), "points", 0, "% change of receivables minus % change of sales", _evidence(fund.receivables_vs_sales), "latest balance sheet vs earlier; sales year over year"),
        MetricSpec("inventory_vs_sales", "filings", ("evidence",), "points", 0, "% change of inventories minus % change of sales", _evidence(fund.inventory_vs_sales), "latest balance sheet vs earlier; sales year over year"),
        MetricSpec("cfo_negative_with_profit", "filings", ("evidence",), "bool", 0, "cash from operations below zero while net income is above", _evidence(fund.cfo_negative_with_profit), "year to date"),
        MetricSpec("negative_equity", "filings", ("evidence",), "bool", 0, "total stockholders' equity of the latest stored balance sheet is below zero (a deficit): debt / equity has no meaning", _evidence(fund.negative_equity), "latest balance sheet"),
        MetricSpec("dilution_current", "filings", ("evidence",), "bool", 0, "the company is issuing stock now, from the stored filings", _dilution, "latest filings"),
    ]
    for name in UNAVAILABLE:
        group = "unavailable"
        specs.append(MetricSpec(name, group, (), "n/a", 0, UNAVAILABLE[name], _unavailable(name), "n/a"))
    return {s.name: s for s in specs}


METRICS: dict[str, MetricSpec] = _build()
METRIC_NAMES = tuple(METRICS)


# ---------------------------------------------------------------- compute

def _source(spec: MetricSpec, c: Context) -> str:
    """Where the number came from: one input is named as it is, several are joined as `derived:a+b`."""
    parts = []
    for kind in spec.inputs:
        if kind == "bars" and c.bars:
            parts.append(f"day_aggs:{c.bars[0].date}..{c.bars[-1].date}")
        elif kind == "financials" and c.financials:
            parts.append(f"finnhub_metric:{c.financials.get('as_of')}")
        elif kind == "evidence" and c.evidence:
            sheet = c.evidence.get("balance_sheet") or {}
            accession = (sheet.get("source") or {}).get("accession") or next((f["accession"] for f in c.evidence.get("filings", {}).get("with_text", [])), None)
            parts.append(f"filing:{accession}")
    return "n/a" if not parts else parts[0] if len(parts) == 1 else "derived:" + "+".join(parts)


def _as_of(spec: MetricSpec, c: Context) -> str | None:
    """The date of the newest input: the last bar (of SPY or of the sector ETF for their metrics), the Finnhub fetch date, or the filing date of the statements the figures were read from."""
    series = c.spy if spec.name.startswith("spy_") else c.sector_bars if spec.name.startswith(("sector_ret", "residual")) else c.bars
    if "bars" in spec.inputs and series:
        return series[-1].date.isoformat()
    if "financials" in spec.inputs and c.financials:
        return c.financials.get("as_of")
    if "evidence" in spec.inputs and c.evidence:
        statement = ((c.evidence.get("balance_sheet") or {}).get("source") or {}).get("filed")  # the filing the figures were read from, not the newest filing of any kind
        return statement or c.evidence.get("as_of") or None
    return None


def metric_result(spec: MetricSpec, c: Context) -> dict:
    """One metric of one symbol, as the result object of spec 4.2. Never raises for a missing input: that is `value: None` with the note."""
    base = {"unit": spec.unit, "as_of": _as_of(spec, c), "window": spec.window, "source": _source(spec, c)}
    if "bars" in spec.inputs and len(c.bars) < spec.min_bars:
        return {"value": None, **base, "inputs_summary": {"bars": len(c.bars)}, "note": f"needs {spec.min_bars} bars, {len(c.bars)} stored"}
    try:
        value, summary = spec.fn(c)
    except Missing as reason:
        return {"value": None, **base, "inputs_summary": {}, "note": str(reason)}
    if value is None:
        return {"value": None, **base, "inputs_summary": summary, "note": "not defined for these bars (a zero range, or no volume)"}
    if "financials" in spec.inputs and spec.group == "fundamentals" and c.financials:
        base["unit"] = c.financials["metrics"].get(spec.name, {}).get("unit", spec.unit)
    return {"value": value, **base, "inputs_summary": summary, "note": None}


def compute(symbols: list[str], requests: list[dict], bars: BarStore, financials: dict[str, dict], evidence: dict[str, dict], today: date,
            sectors: dict[str, str] | None = None) -> dict[str, dict[str, dict]]:
    """{symbol: {metric name: result}} for every requested metric of every symbol. `financials` and `evidence` are {symbol: payload} (a symbol without one has its metrics that need it
    as `None` with a note); `sectors` is {symbol: sector ETF ticker}. A name that is not in the catalogue raises ValueError (the work order check refuses it first)."""
    names = [r["name"] for r in requests]
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}; known metrics: {', '.join(METRIC_NAMES)}")
    out: dict[str, dict[str, dict]] = {}
    spy = bars.bars(SPY, end=today)
    for symbol in symbols:
        etf = (sectors or {}).get(symbol)
        context = Context(symbol, today, bars.bars(symbol, end=today), spy, etf, bars.bars(etf, end=today) if etf else None, financials.get(symbol), evidence.get(symbol))
        out[symbol] = {name: metric_result(METRICS[name], context) for name in dict.fromkeys(names)}
    return out
