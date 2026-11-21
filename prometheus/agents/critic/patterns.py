"""Chart patterns calculated in code from the stored closes (Critic v3, design/critic_v3_two_reasons_spec.md section 4, design choices entry 137). No network, no model.

Jacob asked for chart patterns from Finnhub or Massive if the licence has a classification call. It has none (`/scan/pattern`, `/scan/support-resistance` and Massive's Benzinga endpoints answer 403: the spec's section 1), so
the patterns are made here, from the closing prices the Data Clerk calculated for the Critic (the metric `close_history`: the last stored sessions as [date, close], oldest first). That keeps them point in time, as every
number the Critic reads is: the Time Machine's bound day cuts the bars before the metric is made (P2).

    held, text = double_bottom(facts, config.chart)       # (True, "two lows of 21.40 ... and the last close ... is above it"), (False, "..."), or (None, "") when the closes are too few

SWING POINTS. A swing low (high) is a close lower (higher) than the `pivot_window` closes before it and no higher (lower) than the `pivot_window` closes after it, looked for in the last `pattern_lookback` sessions; the last
`pivot_window` sessions cannot be one yet (nothing confirms them). Two levels are "the same" when they differ by no more than `level_tolerance` of the higher. The eight patterns (points are in the config; bullish positive):

    double_bottom                two swing lows of the same level at least `min_gap` sessions apart, a swing high between them at least `min_depth` above both, the last close above the highest swing high between them
    double_top                   the mirror image: two swing highs, a swing low between them `min_depth` under both, the last close under the lowest swing low between them
    inverse_head_and_shoulders   three swing lows in a row, the middle one lower than both others by `min_depth`, the outer two of the same level, the last close above the higher of the two swing highs between them
    head_and_shoulders           the mirror image
    ascending_triangle           the last three swing highs of the same level (a flat top) and the last three swing lows each higher than the one before
    descending_triangle          the last three swing lows of the same level and the last three swing highs each lower
    breakout_52w                 the last close above the highest close of the `year_sessions` before it
    breakdown_52w                the last close under the lowest close of those sessions

A pattern whose closes are not there (a recent listing) is `None`: not judged, never guessed (as every pattern of `chart.py`).

A PATTERN MUST BE COMPLETE RECENTLY TO COUNT (`chart.pattern_recent_sessions`, 10). A double top or bottom, a head and shoulders and its inverse fire only while the neckline was broken within the last `pattern_recent_sessions` sessions
(the first close beyond it after the pattern's last swing point is that recent) and the last close is still beyond it; a pattern whose neckline was broken a month ago and is still "broken" is old news and does not count. A triangle fires
only when its last swing point (the newest of its three highs and three lows) is within the last 2 x `pivot_window` + `pattern_recent_sessions` sessions. "Within the last K sessions" counts the last close as the first. Every swing point is already
inside the last `pattern_lookback` sessions, the only ones looked at. The 52-week breakout and breakdown are the last close itself and need no such rule.

The closes come in the share basis of the last day (the Data Clerk puts a stored split into them: `metrics_catalog.close_history`), so a split inside the window is not a crash and not a pattern.
"""
from prometheus.agents.critic.config import Chart

Judgement = tuple[bool | None, str]
UNKNOWN: Judgement = (None, "")
Point = tuple[int, float]      # (the index in the segment, the close)


def _series(f: dict, c: Chart, minimum: int) -> tuple[list[str], list[float]] | None:
    """(dates, closes) of the last `pattern_lookback` sessions, or None when `minimum` closes are not there."""
    history = f.get("close_history")
    if not isinstance(history, list) or len(history) < minimum:
        return None
    rows = [r for r in history[-c.pattern_lookback:] if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[1], (int, float)) and not isinstance(r[1], bool)]
    return ([str(r[0]) for r in rows], [float(r[1]) for r in rows]) if len(rows) >= minimum else None


def swing_points(closes: list[float], window: int) -> tuple[list[Point], list[Point]]:
    """(swing lows, swing highs), each [(index, close)] oldest first. A swing low is strictly under the `window` closes before it and no higher than the `window` closes after it (the first of a flat run counts); a swing high mirrors it."""
    lows, highs = [], []
    for i in range(window, len(closes) - window):
        before, after, x = closes[i - window:i], closes[i + 1:i + 1 + window], closes[i]
        if all(x < y for y in before) and all(x <= y for y in after):
            lows.append((i, x))
        if all(x > y for y in before) and all(x >= y for y in after):
            highs.append((i, x))
    return lows, highs


def same_level(a: float, b: float, tolerance: float) -> bool:
    """Two levels are the same when they differ by no more than `tolerance` of the higher."""
    return abs(a - b) <= tolerance * max(a, b)


def _between(points: list[Point], first: int, last: int) -> list[Point]:
    return [p for p in points if first < p[0] < last]


def _within(index: int, n: int, sessions: int) -> bool:
    """Whether the session at `index` of `n` is within the last `sessions` sessions, the last close counting as the first."""
    return index >= n - sessions


def _broke_recently(closes: list[float], level: float, above: bool, after: int, c: Chart) -> int | None:
    """The index of the neckline's BREAK, the first close beyond `level` (above it, or under it when `above` is False) after the pattern's last swing point (`after`), when that was within the last `pattern_recent_sessions` sessions
    and the last close is still beyond the level; else None. The pattern is complete at that first break: one broken a month ago, or broken and lost again, does not count."""
    n = len(closes)
    if not (closes[-1] > level if above else closes[-1] < level):
        return None
    j = next((k for k in range(after + 1, n) if (closes[k] > level if above else closes[k] < level)), None)
    return j if j is not None and _within(j, n, c.pattern_recent_sessions) else None


def _broke_text(dates: list[str], j: int, n: int) -> str:
    ago = n - j
    return f"{'today' if ago == 1 else f'{ago} sessions ago'} ({dates[j]})"


def _need(f: dict, c: Chart):
    return _series(f, c, 2 * c.pivot_window + 1)


def double_bottom(f: dict, c: Chart) -> Judgement:
    """Two swing lows of one level, `min_gap` sessions apart or more, a swing high between them `min_depth` above both, and the neckline (the highest such swing high) broken upward within the last `pattern_recent_sessions` sessions with
    the last close still above it."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    dates, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    last, n = closes[-1], len(closes)
    for k in range(len(lows) - 1, 0, -1):
        for j in range(k - 1, -1, -1):
            (i1, a), (i2, b) = lows[j], lows[k]
            if i2 - i1 < c.min_gap or not same_level(a, b, c.level_tolerance):
                continue
            middle = [h for h in _between(highs, i1, i2) if h[1] >= max(a, b) * (1 + c.min_depth)]
            if not middle:
                continue
            neck = max(h[1] for h in middle)
            broke = _broke_recently(closes, neck, True, i2, c)
            if broke is not None:
                return True, (f"two lows of {a:.2f} ({dates[i1]}) and {b:.2f} ({dates[i2]}), {i2 - i1} sessions apart, a high of {neck:.2f} between them, and the neckline was broken {_broke_text(dates, broke, n)}; "
                              f"the last close {last:.2f} is above it")
    return False, (f"no pair of lows of one level (within {c.level_tolerance:.0%}) at least {c.min_gap} sessions apart with a high {c.min_depth:.0%} above them whose neckline was broken upward "
                   f"in the last {c.pattern_recent_sessions} sessions (the last close is {last:.2f})")


def double_top(f: dict, c: Chart) -> Judgement:
    """The mirror image of `double_bottom`: two swing highs of one level, a swing low between them `min_depth` under both, the neckline (the lowest such swing low) broken downward within the last `pattern_recent_sessions` sessions with
    the last close still under it."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    dates, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    last, n = closes[-1], len(closes)
    for k in range(len(highs) - 1, 0, -1):
        for j in range(k - 1, -1, -1):
            (i1, a), (i2, b) = highs[j], highs[k]
            if i2 - i1 < c.min_gap or not same_level(a, b, c.level_tolerance):
                continue
            middle = [v for v in _between(lows, i1, i2) if v[1] <= min(a, b) * (1 - c.min_depth)]
            if not middle:
                continue
            neck = min(v[1] for v in middle)
            broke = _broke_recently(closes, neck, False, i2, c)
            if broke is not None:
                return True, (f"two highs of {a:.2f} ({dates[i1]}) and {b:.2f} ({dates[i2]}), {i2 - i1} sessions apart, a low of {neck:.2f} between them, and the neckline was broken {_broke_text(dates, broke, n)}; "
                              f"the last close {last:.2f} is under it")
    return False, (f"no pair of highs of one level (within {c.level_tolerance:.0%}) at least {c.min_gap} sessions apart with a low {c.min_depth:.0%} under them whose neckline was broken downward "
                   f"in the last {c.pattern_recent_sessions} sessions (the last close is {last:.2f})")


def inverse_head_and_shoulders(f: dict, c: Chart) -> Judgement:
    """Three swing lows in a row, the middle lower than both others by `min_depth`, the outer two of one level, and the neckline (the higher of the two swing highs between them) broken upward within the last
    `pattern_recent_sessions` sessions with the last close still above it."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    dates, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    last, n = closes[-1], len(closes)
    for k in range(len(lows) - 1, 1, -1):
        (i1, a), (i2, head), (i3, b) = lows[k - 2], lows[k - 1], lows[k]
        if head > min(a, b) * (1 - c.min_depth) or not same_level(a, b, c.level_tolerance):
            continue
        left, right = _between(highs, i1, i2), _between(highs, i2, i3)
        if left and right:
            neck = max(max(h[1] for h in left), max(h[1] for h in right))
            broke = _broke_recently(closes, neck, True, i3, c)
            if broke is not None:
                return True, (f"lows of {a:.2f} ({dates[i1]}), {head:.2f} ({dates[i2]}, the head) and {b:.2f} ({dates[i3]}), and the neckline of {neck:.2f} was broken {_broke_text(dates, broke, n)}; "
                              f"the last close {last:.2f} is above it")
    return False, (f"no three lows with a middle one {c.min_depth:.0%} lower than two of one level whose neckline was broken upward in the last {c.pattern_recent_sessions} sessions (the last close is {last:.2f})")


def head_and_shoulders(f: dict, c: Chart) -> Judgement:
    """The mirror image: three swing highs in a row, the middle higher than both others by `min_depth`, the outer two of one level, and the neckline (the lower of the two swing lows between them) broken downward within the last
    `pattern_recent_sessions` sessions with the last close still under it."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    dates, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    last, n = closes[-1], len(closes)
    for k in range(len(highs) - 1, 1, -1):
        (i1, a), (i2, head), (i3, b) = highs[k - 2], highs[k - 1], highs[k]
        if head < max(a, b) * (1 + c.min_depth) or not same_level(a, b, c.level_tolerance):
            continue
        left, right = _between(lows, i1, i2), _between(lows, i2, i3)
        if left and right:
            neck = min(min(v[1] for v in left), min(v[1] for v in right))
            broke = _broke_recently(closes, neck, False, i3, c)
            if broke is not None:
                return True, (f"highs of {a:.2f} ({dates[i1]}), {head:.2f} ({dates[i2]}, the head) and {b:.2f} ({dates[i3]}), and the neckline of {neck:.2f} was broken {_broke_text(dates, broke, n)}; "
                              f"the last close {last:.2f} is under it")
    return False, (f"no three highs with a middle one {c.min_depth:.0%} higher than two of one level whose neckline was broken downward in the last {c.pattern_recent_sessions} sessions (the last close is {last:.2f})")


def _triangle_is_recent(lows: list[Point], highs: list[Point], n: int, c: Chart) -> tuple[bool, int]:
    """(whether the newest swing point of the triangle's six is within the last 2 x `pivot_window` + `pattern_recent_sessions` sessions, how many sessions ago it was: 1 is the last close)."""
    newest = max(highs[-1][0], lows[-1][0])
    return _within(newest, n, 2 * c.pivot_window + c.pattern_recent_sessions), n - newest


def ascending_triangle(f: dict, c: Chart) -> Judgement:
    """The last three swing highs of one level (a flat top) and the last three swing lows each higher than the one before, the newest of the six within the last 2 x `pivot_window` + `pattern_recent_sessions` sessions."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    _, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    if len(highs) < 3 or len(lows) < 3:
        return False, "fewer than three swing highs and three swing lows in the last sessions"
    top, bottom = [p[1] for p in highs[-3:]], [p[1] for p in lows[-3:]]
    flat, rising = max(top) - min(top) <= c.level_tolerance * max(top), bottom[0] < bottom[1] < bottom[2]
    recent, ago = _triangle_is_recent(lows, highs, len(closes), c)
    return flat and rising and recent, (f"the last three highs are {', '.join(f'{x:.2f}' for x in top)} ({'a flat top' if flat else 'not flat'}) and the last three lows {', '.join(f'{x:.2f}' for x in bottom)} "
                                        f"({'each higher' if rising else 'not each higher'}); the newest of them was {ago} sessions ago ({'recent' if recent else f'older than {2 * c.pivot_window + c.pattern_recent_sessions}'})")


def descending_triangle(f: dict, c: Chart) -> Judgement:
    """The last three swing lows of one level (a flat bottom) and the last three swing highs each lower than the one before, the newest of the six within the last 2 x `pivot_window` + `pattern_recent_sessions` sessions."""
    got = _need(f, c)
    if got is None:
        return UNKNOWN
    _, closes = got
    lows, highs = swing_points(closes, c.pivot_window)
    if len(highs) < 3 or len(lows) < 3:
        return False, "fewer than three swing highs and three swing lows in the last sessions"
    top, bottom = [p[1] for p in highs[-3:]], [p[1] for p in lows[-3:]]
    flat, falling = max(bottom) - min(bottom) <= c.level_tolerance * max(bottom), top[0] > top[1] > top[2]
    recent, ago = _triangle_is_recent(lows, highs, len(closes), c)
    return flat and falling and recent, (f"the last three lows are {', '.join(f'{x:.2f}' for x in bottom)} ({'a flat bottom' if flat else 'not flat'}) and the last three highs {', '.join(f'{x:.2f}' for x in top)} "
                                         f"({'each lower' if falling else 'not each lower'}); the newest of them was {ago} sessions ago ({'recent' if recent else f'older than {2 * c.pivot_window + c.pattern_recent_sessions}'})")


def _year(f: dict, c: Chart):
    history = f.get("close_history")
    if not isinstance(history, list) or len(history) < c.year_sessions + 1:
        return None
    closes = [float(r[1]) for r in history[-(c.year_sessions + 1):] if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[1], (int, float)) and not isinstance(r[1], bool)]
    return closes if len(closes) == c.year_sessions + 1 else None


def breakout_52w(f: dict, c: Chart) -> Judgement:
    """The last close above the highest close of the `year_sessions` sessions before it."""
    closes = _year(f, c)
    if closes is None:
        return UNKNOWN
    high = max(closes[:-1])
    return closes[-1] > high, f"the last close {closes[-1]:.2f} against the highest close of the {c.year_sessions} sessions before it, {high:.2f}"


def breakdown_52w(f: dict, c: Chart) -> Judgement:
    """The last close under the lowest close of the `year_sessions` sessions before it."""
    closes = _year(f, c)
    if closes is None:
        return UNKNOWN
    low = min(closes[:-1])
    return closes[-1] < low, f"the last close {closes[-1]:.2f} against the lowest close of the {c.year_sessions} sessions before it, {low:.2f}"


JUDGES = {"double_bottom": double_bottom, "double_top": double_top, "inverse_head_and_shoulders": inverse_head_and_shoulders, "head_and_shoulders": head_and_shoulders,
          "ascending_triangle": ascending_triangle, "descending_triangle": descending_triangle, "breakout_52w": breakout_52w, "breakdown_52w": breakdown_52w}
