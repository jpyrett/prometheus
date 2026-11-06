"""Tests for prometheus.agents.data_clerk.technicals (Advisor v1 spec section 5, phase 1).

Each indicator is checked against a value worked out by hand on a series short enough to do it on paper (a ramp, an alternating series, a flat one), and against
`None` below MIN_BARS. The series are literals that test arithmetic; they are not market data.
"""
import math
from statistics import stdev

import pytest

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
             "adv_20d_dollars": t.adv_20d_dollars, "volume_ratio_1d": t.volume_ratio_1d}
    assert set(calls) == set(t.MIN_BARS)  # a new indicator needs its MIN_BARS and a line here
    for name, fn in calls.items():
        answer = fn(bars)
        if len(bars) < t.MIN_BARS[name]:
            assert answer is None, name
        else:
            assert answer is not None, name


def test_returns_and_momentum_skip_the_last_month():
    closes = ramp(252)
    bars = make_bars(closes)
    assert t.ret(bars, 5) == pytest.approx(351 / 346 - 1)
    assert t.ret(make_bars(ramp(6)), 5) == pytest.approx(105 / 100 - 1)
    assert t.ret(make_bars(ramp(5)), 5) is None
    # close[-21] / close[-252] - 1: index 231 over index 0 = 331 / 100 - 1
    assert t.mom_12_1(bars) == pytest.approx(2.31)
    # the latest month is skipped: whatever the last 20 closes do, close[-21] (index 231) and close[-252] (index 0) are the only inputs
    assert t.mom_12_1(make_bars(closes[:-20] + [1.0] * 20)) == pytest.approx(2.31)
    assert t.mom_12_1(make_bars(closes[:-20] + [999.0] * 20)) == pytest.approx(2.31)


def test_52_week_high_and_moving_averages():
    bars = make_bars(ramp(252))  # highs are close + 0.5; the best is 351.5
    assert t.pct_of_52w_high(bars) == pytest.approx(351 / 351.5)
    assert t.sma(make_bars([1, 2, 3, 4, 5]), 5) == 3
    assert t.sma(make_bars([1, 2, 3, 4, 5]), 3) == 4
    assert t.above_sma(make_bars([1, 2, 3, 4, 5]), 5) is True
    assert t.above_sma(make_bars([5, 4, 3, 2, 1]), 5) is False
    assert t.above_sma(make_bars([5, 4, 3, 2, 1]), 6) is None


def test_breakout_gap_and_reversal_candle():
    flat = [10.0] * 21
    assert t.breakout_20d(make_bars(flat[:-1] + [10.4])) is False        # the highest high of the 20 before is 10.5
    assert t.breakout_20d(make_bars(flat[:-1] + [10.6])) is True
    assert t.breakout_20d(make_bars(flat[:20])) is None
    assert t.gap_1d(make_bars([100, 101], opens=[100, 103])) == pytest.approx(103 / 100 - 1)
    up_candle = make_bars([10, 10, 10], opens=[10, 10, 9.5], lows=[9, 9, 8.5], highs=[11, 11, 10.5])  # opens 9.5, trades down to 8.5 < yesterday's 9, closes 10 > open
    assert t.candle_reversal_1d(up_candle) is True
    down_candle = make_bars([10, 10, 9], opens=[10, 10, 9.5], lows=[9, 9, 8.5])                      # closes under its open
    assert t.candle_reversal_1d(down_candle) is False
    no_new_low = make_bars([10, 10, 10], opens=[10, 10, 9.5], lows=[9, 9, 9.2])
    assert t.candle_reversal_1d(no_new_low) is False


def test_rsi_is_100_on_a_ramp_0_on_a_fall_and_50_when_gains_equal_losses(monkeypatch):
    assert t.rsi_14(make_bars(ramp(40))) == 100.0                       # avg_loss == 0
    assert t.rsi_14(make_bars(ramp(40, 200.0, -1.0))) == pytest.approx(0.0)
    alternating = [100.0 + (i % 2) for i in range(40)]                  # +1, -1, +1, ...: 7 gains and 7 losses in the first 14 changes
    assert t.rsi_14(make_bars(alternating)) == pytest.approx(50.0, abs=3.5)  # 7/14 then smoothed: stays close to 50
    # worked by hand: the first 14 changes are +1 x8 and -1 x6 -> 8/14 and 6/14; with exactly 14 changes (15 bars) RSI = 100 - 100 / (1 + 8/6) = 57.142857
    first = 100.0
    closes = [first]
    for step in [1] * 8 + [-1] * 6:
        closes.append(closes[-1] + step)
    monkeypatch.setitem(t.MIN_BARS, "rsi_14", 15)   # the textbook example has only 15 bars; the shipped minimum is 30
    assert t.rsi_14(make_bars(closes)) == pytest.approx(100 - 100 / (1 + 8 / 6))
    # one more change (+1) is the Wilder smoothing step: avg_gain = (8/14 * 13 + 1) / 14, avg_loss = (6/14 * 13 + 0) / 14
    monkeypatch.setitem(t.MIN_BARS, "rsi_14", 16)
    gain, loss = (8 / 14 * 13 + 1) / 14, (6 / 14 * 13) / 14
    assert t.rsi_14(make_bars(closes + [closes[-1] + 1])) == pytest.approx(100 - 100 / (1 + gain / loss))


def test_ema_is_seeded_with_the_simple_mean():
    assert t.ema([1, 2, 3, 4, 5], 3) == [2, 3, 4]       # seed (1+2+3)/3 = 2, then k = 0.5
    assert t.ema([1, 2], 3) == []


def test_macd_histogram_is_zero_on_a_ramp_and_negative_when_the_ramp_stops():
    assert t.macd_hist(make_bars(ramp(80))) == pytest.approx(0.0, abs=1e-9)   # an EMA of a ramp lags by a constant: the line and its signal are equal
    assert t.macd_hist(make_bars([100.0] * 80)) == pytest.approx(0.0)
    stalled = make_bars(ramp(70) + [169.0] * 10)
    assert t.macd_hist(stalled) < 0


def test_bollinger_pct_b_and_zscore():
    closes = [10.0] * 19 + [12.0]
    mean, std = 10.1, math.sqrt(0.19)                                      # population std of the 20: (19 * 0.01 + 3.61) / 20 = 0.19
    assert t.bollinger_pct_b(make_bars(closes)) == pytest.approx((12 - (mean - 2 * std)) / (4 * std))
    assert t.bollinger_pct_b(make_bars([10.0] * 20)) is None                # the bands are the same price
    assert t.zscore_21d(make_bars(ramp(21, 1.0))) == pytest.approx(10 / math.sqrt(770 / 20))   # mean 11, sample std sqrt(38.5)
    assert t.zscore_21d(make_bars([5.0] * 21)) is None


def test_atr_smoothing_step_by_hand(monkeypatch):
    monkeypatch.setitem(t.MIN_BARS, "atr_14", 16)
    bars = make_bars([100.0] * 16, highs=[101.0] * 15 + [109.0], lows=[99.0] * 15 + [93.0])   # 14 true ranges of 2, then one of 16
    assert t.atr_14(bars) == pytest.approx((2 * 13 + 16) / 14)    # (prev * 13 + current) / 14 = 3.0


def test_atr_is_the_constant_true_range_and_adx_is_100_in_a_one_way_trend():
    flat = make_bars([100.0] * 30, highs=[101.0] * 30, lows=[99.0] * 30)   # a true range of 2 every bar
    assert t.atr_14(flat) == pytest.approx(2.0)
    assert t.atr_pct(flat) == pytest.approx(0.02)
    assert t.adx_14(make_bars(ramp(60))) == pytest.approx(100.0)            # only +DM, no -DM: DX is 100 each day
    assert t.adx_14(make_bars([100.0] * 60, highs=[101.0] * 60, lows=[99.0] * 60)) == 0.0   # no directional movement at all


def test_obv_stochastic_volume_and_dollar_volume():
    assert t.obv_20d_change(make_bars(ramp(30), volumes=[100.0] * 30)) == pytest.approx(1.0)       # 20 up days: +2000 over a total of 2000
    assert t.obv_20d_change(make_bars(ramp(30, 200.0, -1.0), volumes=[100.0] * 30)) == pytest.approx(-1.0)
    assert t.obv_20d_change(make_bars([10.0] * 30, volumes=[0.0] * 30)) is None
    bars = make_bars([10.0] * 13 + [10.0], highs=[12.0] * 14, lows=[8.0] * 13 + [8.0])
    assert t.stoch_k_14(bars) == pytest.approx(50.0)
    assert t.stoch_k_14(make_bars([10.0] * 14, highs=[10.0] * 14, lows=[10.0] * 14)) is None
    volumes = [100.0] * 20 + [300.0]
    assert t.volume_ratio_1d(make_bars([10.0] * 21, volumes=volumes)) == pytest.approx(3.0)
    assert t.adv_20d_dollars(make_bars([10.0] * 20, volumes=[50.0] * 20)) == pytest.approx(500.0)


def test_volatility_drawdown_beta_and_correlation():
    assert t.vol_ann(make_bars([100.0 * 1.01 ** i for i in range(30)]), 21) == pytest.approx(0.0, abs=1e-12)  # constant growth: no spread
    closes = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    logs = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    assert t.vol_ann(make_bars(closes), 5) == pytest.approx(stdev(logs) * math.sqrt(252))
    drawdown = make_bars(ramp(40, 100.0, 0.5) + [90.0] * 23)               # up to 119.5 then 90: 90 / 119.5 - 1
    assert t.max_drawdown_63d(drawdown) == pytest.approx(90 / 119.5 - 1)
    assert t.max_drawdown_63d(make_bars(ramp(63))) == 0.0
    market = [100.0]
    for i in range(70):
        market.append(market[-1] * (1 + (0.01 if i % 3 else -0.015)))
    twice = [100.0]
    for a, b in zip(market, market[1:]):
        twice.append(twice[-1] * (1 + 2 * (b / a - 1)))                       # twice the market's daily return
    beta, corr = t.beta_and_corr(make_bars(twice), make_bars(market))
    assert beta == pytest.approx(2.0) and corr == pytest.approx(1.0)
    assert t.beta_and_corr(make_bars(twice[:40]), make_bars(market[:40])) == (None, None)
    assert t.beta_and_corr(make_bars(twice), make_bars([100.0] * 71)) == (None, None)   # a market that never moves has no beta


def test_beta_matches_series_by_date():
    """A day one series lacks is dropped from both: the returns are over the shared dates only."""
    market = make_bars([100.0 + i + (i % 4) for i in range(80)])
    own = [b for i, b in enumerate(make_bars([50.0 + 0.5 * i + (i % 4) / 2 for i in range(80)])) if i != 40]
    beta, corr = t.beta_and_corr(own, market)
    assert beta is not None and corr is not None and -1.0 <= corr <= 1.0
