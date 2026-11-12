"""Technical indicators over daily bars: pure functions, no file, no network, no model (Advisor v1 spec section 5.1 to 5.3, design choices entry 51).

Every function takes a list of `Bar` (oldest first, the last one is "today") and returns a float, a bool, or None. None means "not computable": there are fewer bars than the indicator
needs (`MIN_BARS[name]`) or the quantity is undefined for these bars (a zero range). A window is NEVER shortened to make a number appear: with 17 stored sessions `rsi_14` is None.

Every indicator is a FEATURE of a strategy family (prometheus/agents/advisor/families.py); none is a vote by itself, and none of them rates anything.
Returns are simple, close-to-close, unless a name says log. "N bars" means N trading sessions, not calendar days.
"""
import math
from statistics import fmean

from prometheus.agents.data_clerk.bars import Bar

TRADING_DAYS = 252

MIN_BARS = {
    "ret_1d": 2, "ret_5d": 6, "ret_21d": 22, "ret_63d": 64, "ret_126d": 127, "ret_252d": 253,
    "mom_12_1": 252, "pct_of_52w_high": 252, "sma_50": 50, "sma_200": 200, "above_sma_50": 50, "above_sma_200": 200,
    "breakout_20d": 21, "gap_1d": 2, "candle_reversal_1d": 2,
    "rsi_14": 30, "macd_hist": 60, "bollinger_pct_b": 20, "zscore_21d": 21, "adx_14": 40, "atr_14": 20, "atr_pct": 20, "obv_20d_change": 21, "stoch_k_14": 14,
    "vol_21d_ann": 22, "vol_63d_ann": 64, "max_drawdown_63d": 63, "beta_63d": 64, "corr_63d_spy": 64,
    "adv_20d_dollars": 20, "volume_ratio_1d": 21,
    "sma_50_vs_200": 200, "cross_50_200_20d": 220, "structure_63d": 63,   # chart patterns the Critic reads (Critic v1 Amendment 1, design choices entry 62)
    "decline_from_peak_63d": 63, "rebound_from_trough_63d": 63, "sessions_since_trough_63d": 63,
    "sma_126": 126, "high_126d": 126, "low_126d": 126,   # the six-month average, high and low: what the Critic judges a price against (Critic v1 spec 5.1)
}


def _closes(bars: list[Bar]) -> list[float]:
    return [b.close for b in bars]


def _enough(bars: list[Bar], name: str) -> bool:
    return len(bars) >= MIN_BARS[name]


# ---------------------------------------------------------------- price, return, trend

def ret(bars: list[Bar], n: int) -> float | None:
    """`close[-1] / close[-1-n] - 1`. None when there are fewer than n+1 bars."""
    if len(bars) < n + 1 or bars[-1 - n].close <= 0:
        return None
    return bars[-1].close / bars[-1 - n].close - 1


def mom_12_1(bars: list[Bar]) -> float | None:
    """12-month momentum that skips the latest month: `close[-21] / close[-252] - 1`."""
    if not _enough(bars, "mom_12_1") or bars[-252].close <= 0:
        return None
    return bars[-21].close / bars[-252].close - 1


def pct_of_52w_high(bars: list[Bar]) -> float | None:
    """`close[-1] / max(high over the last 252 bars)`."""
    if not _enough(bars, "pct_of_52w_high"):
        return None
    top = max(b.high for b in bars[-252:])
    return bars[-1].close / top if top > 0 else None


def sma(bars: list[Bar], n: int) -> float | None:
    """The simple mean of the last n closes."""
    return fmean(_closes(bars[-n:])) if len(bars) >= n else None


def high_126d(bars: list[Bar]) -> float | None:
    """The highest HIGH of the last 126 bars (six months of sessions). None below 126 bars: the window is never shortened."""
    return max(b.high for b in bars[-126:]) if _enough(bars, "high_126d") else None


def low_126d(bars: list[Bar]) -> float | None:
    """The lowest LOW of the last 126 bars (six months of sessions). None below 126 bars."""
    return min(b.low for b in bars[-126:]) if _enough(bars, "low_126d") else None


CROSS_WINDOW = 20   # a crossing of the 50- and 200-day means counts as recent for this many sessions


def sma_50_vs_200(bars: list[Bar]) -> float | None:
    """`sma_50 / sma_200 - 1`: positive when the 50-day mean is above the 200-day mean (the "golden" side), negative on the "death" side."""
    if not _enough(bars, "sma_50_vs_200"):
        return None
    slow = sma(bars, 200)
    return sma(bars, 50) / slow - 1 if slow else None


def cross_50_200_20d(bars: list[Bar]) -> int | None:
    """When the 50-day mean last crossed the 200-day mean, within the last 20 sessions: `+age` for a golden cross (it went from at-or-under to above), `-age` for a death cross (from above to
    at-or-under), `0` when there was none. `age` is 1 for the latest bar, 2 for the one before, up to 20. The two means are worked out again at every bar, from that bar's own closes."""
    if not _enough(bars, "cross_50_200_20d"):
        return None
    closes = _closes(bars)

    def above(end: int) -> bool:   # the means at the bar closes[end - 1]
        return fmean(closes[end - 50:end]) > fmean(closes[end - 200:end])

    for age in range(1, CROSS_WINDOW + 1):
        end = len(closes) - (age - 1)
        now, before = above(end), above(end - 1)
        if now != before:
            return age if now else -age
    return 0


def structure_63d(bars: list[Bar]) -> str | None:
    """The shape of the last 63 sessions as three windows of 21: `"up"` when the highest high AND the lowest low both rise from each window to the next (higher highs and higher lows),
    `"down"` when both fall (lower highs and lower lows), else `"mixed"`."""
    if not _enough(bars, "structure_63d"):
        return None
    windows = (bars[-63:-42], bars[-42:-21], bars[-21:])
    highs, lows = [max(b.high for b in w) for w in windows], [min(b.low for b in w) for w in windows]
    if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
        return "up"
    if highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
        return "down"
    return "mixed"


def _peak_and_trough(bars: list[Bar]) -> tuple[list[float], int, int | None]:
    """(the last 63 closes, the index of their peak (the first of equal ones), the index of the lowest close AFTER the peak (the latest of equal ones), None when the peak is the latest bar))."""
    closes = _closes(bars[-63:])
    peak = max(range(len(closes)), key=lambda i: (closes[i], -i))
    after = range(peak + 1, len(closes))
    return closes, peak, (min(after, key=lambda i: (closes[i], -i)) if len(after) else None)


def decline_from_peak_63d(bars: list[Bar]) -> float | None:
    """`trough / peak - 1` over the last 63 closes: how far it fell from the peak close to the lowest close after it (a negative number; 0.0 when the peak is the latest bar)."""
    if not _enough(bars, "decline_from_peak_63d"):
        return None
    closes, peak, trough = _peak_and_trough(bars)
    return 0.0 if trough is None else closes[trough] / closes[peak] - 1


def rebound_from_trough_63d(bars: list[Bar]) -> float | None:
    """`close[-1] / trough - 1`: how far the price has bounced off that lowest close (0.0 when the peak is the latest bar)."""
    if not _enough(bars, "rebound_from_trough_63d"):
        return None
    closes, _, trough = _peak_and_trough(bars)
    return 0.0 if trough is None else closes[-1] / closes[trough] - 1


def sessions_since_trough_63d(bars: list[Bar]) -> int | None:
    """How many sessions ago that lowest close was (0 for the latest bar, and 0 when the peak is the latest bar)."""
    if not _enough(bars, "sessions_since_trough_63d"):
        return None
    closes, _, trough = _peak_and_trough(bars)
    return 0 if trough is None else len(closes) - 1 - trough


def above_sma(bars: list[Bar], n: int) -> bool | None:
    average = sma(bars, n)
    return None if average is None else bars[-1].close > average


def breakout_20d(bars: list[Bar]) -> bool | None:
    """`close[-1] >= max(high over bars -21..-2)`: today closes at or above the highest high of the 20 sessions before it."""
    if not _enough(bars, "breakout_20d"):
        return None
    return bars[-1].close >= max(b.high for b in bars[-21:-1])


def gap_1d(bars: list[Bar]) -> float | None:
    """`open[-1] / close[-2] - 1`."""
    if not _enough(bars, "gap_1d") or bars[-2].close <= 0:
        return None
    return bars[-1].open / bars[-2].close - 1


def candle_reversal_1d(bars: list[Bar]) -> bool | None:
    """Sold off below yesterday's low and closed up: `close[-1] > open[-1]` and `low[-1] < low[-2]`."""
    if not _enough(bars, "candle_reversal_1d"):
        return None
    return bars[-1].close > bars[-1].open and bars[-1].low < bars[-2].low


# ---------------------------------------------------------------- oscillators and trend strength

def rsi_14(bars: list[Bar]) -> float | None:
    """Wilder's RSI: the first average gain and loss are the simple means of the first 14 changes, then `avg = (prev * 13 + current) / 14`;
    `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`, and 100 when `avg_loss == 0`."""
    if not _enough(bars, "rsi_14"):
        return None
    closes = _closes(bars)
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gains, losses = [max(c, 0.0) for c in changes], [max(-c, 0.0) for c in changes]
    avg_gain, avg_loss = fmean(gains[:14]), fmean(losses[:14])
    for gain, loss in zip(gains[14:], losses[14:]):
        avg_gain, avg_loss = (avg_gain * 13 + gain) / 14, (avg_loss * 13 + loss) / 14
    return 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)


def ema(values: list[float], period: int) -> list[float]:
    """The exponential moving average, seeded with the simple mean of the first `period` values; the result starts at index `period - 1` of `values`."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [fmean(values[:period])]
    for value in values[period:]:
        out.append(value * k + out[-1] * (1 - k))
    return out


def macd_hist(bars: list[Bar]) -> float | None:
    """`EMA12(close) - EMA26(close)` minus its own `EMA9`, the EMAs seeded with the simple mean of their first period."""
    if not _enough(bars, "macd_hist"):
        return None
    closes = _closes(bars)
    fast, slow = ema(closes, 12), ema(closes, 26)
    line = [fast[i + 14] - slow[i] for i in range(len(slow))]  # fast[k] is the bar k+11, slow[i] the bar i+25
    signal = ema(line, 9)
    return line[-1] - signal[-1] if signal else None


def bollinger_pct_b(bars: list[Bar]) -> float | None:
    """`(close - lower) / (upper - lower)`, the bands `SMA20 +- 2 * std20` (population standard deviation). None when the bands are the same price."""
    if not _enough(bars, "bollinger_pct_b"):
        return None
    window = _closes(bars[-20:])
    mean = fmean(window)
    std = math.sqrt(fmean([(c - mean) ** 2 for c in window]))
    lower, upper = mean - 2 * std, mean + 2 * std
    return None if upper == lower else (bars[-1].close - lower) / (upper - lower)


def zscore_21d(bars: list[Bar]) -> float | None:
    """`(close[-1] - mean(close, 21)) / std(close, 21)` with the sample standard deviation."""
    if not _enough(bars, "zscore_21d"):
        return None
    window = _closes(bars[-21:])
    mean = fmean(window)
    std = math.sqrt(sum((c - mean) ** 2 for c in window) / 20)
    return None if std == 0 else (bars[-1].close - mean) / std


def _true_ranges(bars: list[Bar]) -> list[float]:
    return [max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close)) for a, b in zip(bars, bars[1:])]


def atr_14(bars: list[Bar]) -> float | None:
    """Wilder's average true range: the first value is the simple mean of the first 14 true ranges, then `(prev * 13 + current) / 14`."""
    if not _enough(bars, "atr_14"):
        return None
    ranges = _true_ranges(bars)
    value = fmean(ranges[:14])
    for tr in ranges[14:]:
        value = (value * 13 + tr) / 14
    return value


def atr_pct(bars: list[Bar]) -> float | None:
    """`atr_14 / close[-1]`."""
    value = atr_14(bars)
    return None if value is None or bars[-1].close <= 0 else value / bars[-1].close


def adx_14(bars: list[Bar]) -> float | None:
    """Wilder's ADX from +DI and -DI with 14-period smoothing (the first smoothed value is the sum of the first 14, then `prev - prev / 14 + current`);
    the first ADX is the mean of the first 14 DX values, then `(prev * 13 + DX) / 14`."""
    if not _enough(bars, "adx_14"):
        return None
    ranges, plus, minus = _true_ranges(bars), [], []
    for a, b in zip(bars, bars[1:]):
        up, down = b.high - a.high, a.low - b.low
        plus.append(up if up > down and up > 0 else 0.0)
        minus.append(down if down > up and down > 0 else 0.0)
    smooth_tr, smooth_plus, smooth_minus = sum(ranges[:14]), sum(plus[:14]), sum(minus[:14])
    dx = []
    for i in range(14, len(ranges) + 1):
        if i > 14:
            smooth_tr += ranges[i - 1] - smooth_tr / 14
            smooth_plus += plus[i - 1] - smooth_plus / 14
            smooth_minus += minus[i - 1] - smooth_minus / 14
        if smooth_tr == 0:
            return None
        di_plus, di_minus = 100 * smooth_plus / smooth_tr, 100 * smooth_minus / smooth_tr
        total = di_plus + di_minus
        dx.append(0.0 if total == 0 else 100 * abs(di_plus - di_minus) / total)
    if len(dx) < 14:
        return None
    value = fmean(dx[:14])
    for d in dx[14:]:
        value = (value * 13 + d) / 14
    return value


def obv_20d_change(bars: list[Bar]) -> float | None:
    """`(OBV[-1] - OBV[-21]) / sum(volume, last 20)`, OBV the cumulative volume signed by the direction of the close; a number in [-1, 1]. None when the 20 days had no volume."""
    if not _enough(bars, "obv_20d_change"):
        return None
    obv = [0.0]
    for a, b in zip(bars, bars[1:]):
        obv.append(obv[-1] + (b.volume if b.close > a.close else -b.volume if b.close < a.close else 0.0))
    total = sum(b.volume for b in bars[-20:])
    return None if total == 0 else (obv[-1] - obv[-21]) / total


def stoch_k_14(bars: list[Bar]) -> float | None:
    """`(close - min(low, 14)) / (max(high, 14) - min(low, 14)) * 100`. None when the 14-day range is zero."""
    if not _enough(bars, "stoch_k_14"):
        return None
    low, high = min(b.low for b in bars[-14:]), max(b.high for b in bars[-14:])
    return None if high == low else (bars[-1].close - low) / (high - low) * 100


# ---------------------------------------------------------------- volatility, risk, liquidity

def vol_ann(bars: list[Bar], n: int) -> float | None:
    """The sample standard deviation of the last n daily LOG returns times the square root of 252."""
    if len(bars) < n + 1:
        return None
    closes = _closes(bars[-(n + 1):])
    if min(closes) <= 0:
        return None
    logs = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    mean = fmean(logs)
    return math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1)) * math.sqrt(TRADING_DAYS)


def max_drawdown_63d(bars: list[Bar]) -> float | None:
    """The deepest fall from a running peak inside the last 63 bars, as a negative number (0.0 when it never fell)."""
    if not _enough(bars, "max_drawdown_63d"):
        return None
    peak, worst = 0.0, 0.0
    for close in _closes(bars[-63:]):
        peak = max(peak, close)
        worst = min(worst, close / peak - 1)
    return worst


def beta_and_corr(bars: list[Bar], spy: list[Bar], n: int = 63) -> tuple[float | None, float | None]:
    """(beta, correlation) of the symbol's daily returns on SPY's over the last n returns. The two series are matched by DATE (a day one of them lacks is dropped);
    fewer than n+1 shared dates gives (None, None)."""
    spy_close = {b.date: b.close for b in spy}
    shared = [(b.close, spy_close[b.date]) for b in bars if b.date in spy_close][-(n + 1):]
    if len(shared) < n + 1 or any(x <= 0 or y <= 0 for x, y in shared):
        return None, None
    own = [b[0] / a[0] - 1 for a, b in zip(shared, shared[1:])]
    market = [b[1] / a[1] - 1 for a, b in zip(shared, shared[1:])]
    mean_own, mean_market = fmean(own), fmean(market)
    cov = sum((x - mean_own) * (y - mean_market) for x, y in zip(own, market))
    var_market = sum((y - mean_market) ** 2 for y in market)
    var_own = sum((x - mean_own) ** 2 for x in own)
    beta = None if var_market == 0 else cov / var_market
    corr = None if var_market == 0 or var_own == 0 else cov / math.sqrt(var_market * var_own)
    return beta, corr


def adv_20d_dollars(bars: list[Bar]) -> float | None:
    """The mean of `close * volume` over the last 20 bars."""
    if not _enough(bars, "adv_20d_dollars"):
        return None
    return fmean([b.close * b.volume for b in bars[-20:]])


def volume_ratio_1d(bars: list[Bar]) -> float | None:
    """`volume[-1] / mean(volume, bars -21..-2)`. None when the 20 days before had no volume."""
    if not _enough(bars, "volume_ratio_1d"):
        return None
    before = fmean([b.volume for b in bars[-21:-1]])
    return None if before == 0 else bars[-1].volume / before
