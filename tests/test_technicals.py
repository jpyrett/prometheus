"""Tests for prometheus.agents.data_clerk.technicals (Advisor v1 spec section 5, phase 1).

Each indicator is checked against a value worked out by hand on a series short enough to do it on paper (a ramp, an alternating series, a flat one), and against
`None` below MIN_BARS. The series are literals that test arithmetic; they are not market data.
"""


from bar_helpers import make_bars
from prometheus.agents.data_clerk import technicals as t


def ramp(n, start=100.0, step=1.0):
    return [start + step * i for i in range(n)]


def test_every_indicator_is_none_below_its_min_bars_and_never_shortens_the_window():
    """The 17-bar case: only the short ones may answer. A long window is never shrunk to fit."""
    bars = make_bars(ramp(17))
    calls = {"ret_1d": lambda b: t.ret(b, 1), "ret_5d": lambda b: t.ret(b, 5), "ret_21d": lambda b: t.ret(b, 21), "ret_63d": lambda b: t.ret(b, 63), "ret_126d": lambda b: t.ret(b, 126),
             "ret_252d": lambda b: t.ret(b, 252), "mom_12_1": t.mom_12_1, "pct_of_52w_high": t.pct_of_52w_high, "sma_50": lambda b: t.sma(b, 50), "sma_200": lambda b: t.sma(b, 200),
             "above_sma_50": lambda b: t.above_sma(b, 50), "above_sma_200": lambda b: t.above_sma(b, 200), "breakout_20d": t.breakout_20d, "gap_1d": t.gap_1d,
             "candle_reversal_1d": t.candle_reversal_1d, "rsi_14": t.rsi_14, "macd_hist": t.macd_hist, "bollinger_pct_b": t.bollinger_pct_b, "zscore_21d": t.zscore_21d,
             "adx_14": t.adx_14, "atr_14": t.atr_14, "atr_pct": t.atr_pct, "obv_20d_change": t.obv_20d_change, "stoch_k_14": t.stoch_k_14,
             "vol_21d_ann": lambda b: t.vol_ann(b, 21), "vol_63d_ann": lambda b: t.vol_ann(b, 63), "max_drawdown_63d": t.max_drawdown_63d,
             "beta_63d": lambda b: t.beta_and_corr(b, b)[0], "corr_63d_spy": lambda b: t.beta_and_corr(b, b)[1],
             "volume_ratio_1d": t.volume_ratio_1d,
             "sma_126": lambda b: t.sma(b, 126), "high_126d": t.high_126d, "low_126d": t.low_126d,
             "sma_50_vs_200": t.sma_50_vs_200, "cross_50_200_20d": t.cross_50_200_20d, "structure_63d": t.structure_63d, "decline_from_peak_63d": t.decline_from_peak_63d,
             "rebound_from_trough_63d": t.rebound_from_trough_63d, "sessions_since_trough_63d": t.sessions_since_trough_63d}
    assert set(calls) == set(t.MIN_BARS)  # a new indicator needs its MIN_BARS and a line here
    for name, fn in calls.items():
        answer = fn(bars)
        if len(bars) < t.MIN_BARS[name]:
            assert answer is None, name
        else:
            assert answer is not None, name


def test_ema_is_seeded_with_the_simple_mean():
    assert t.ema([1, 2, 3, 4, 5], 3) == [2, 3, 4]       # seed (1+2+3)/3 = 2, then k = 0.5
    assert t.ema([1, 2], 3) == []


# ---------------------------------------------------------------- chart patterns the Critic reads (Critic v1 Amendment 1)


def test_a_golden_cross_is_found_with_its_age_and_not_when_older_than_twenty_sessions():
    assert t.cross_50_200_20d(make_bars([100.0] * 219)) is None                               # 220 bars needed: the means of the bar before the oldest one in the window
    jump = [100.0] * 245 + [200.0] * 5                                                         # both means are 100 (equal, so not above) until the jump bar, 5th from the end
    assert t.cross_50_200_20d(make_bars(jump)) == 5
    assert t.cross_50_200_20d(make_bars([100.0] * 249 + [200.0])) == 1                        # the latest bar crossed
    assert t.cross_50_200_20d(make_bars([100.0] * 230 + [200.0] * 20)) == 20                  # the oldest bar the window looks at
    assert t.cross_50_200_20d(make_bars([100.0] * 229 + [200.0] * 21)) == 0                   # one session older: out of the window
    assert t.cross_50_200_20d(make_bars(ramp(250))) == 0                                       # above all along: no crossing
    assert t.cross_50_200_20d(make_bars([100.0] * 250)) == 0


def test_the_structure_of_63_sessions_is_up_down_or_mixed():
    assert t.structure_63d(make_bars(ramp(62))) is None
    assert t.structure_63d(make_bars(ramp(63))) == "up"                                        # highs and lows rise window to window
    assert t.structure_63d(make_bars(ramp(63, 200.0, -1.0))) == "down"
    assert t.structure_63d(make_bars([100.0] * 63)) == "mixed"                                 # equal is not rising
    assert t.structure_63d(make_bars(ramp(42) + ramp(21, 141.0, -1.0))) == "mixed"            # rose, then fell: the last window's high is lower, its low is lower, the middle's higher
    assert t.structure_63d(make_bars(ramp(63), lows=[50.0 if i == 62 else 100 + i - 0.5 for i in range(63)])) == "mixed"   # higher highs but a lower low


