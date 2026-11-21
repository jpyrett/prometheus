# Critic v3: a denial needs two reasons, analysts and chart patterns score (spec)

Jacob, 2026-11-20 (night), in his words:

> Use finnhub or massive.com analyst recommendations in the Critic Scoring.
> Leverage chart patterns in the Critic scoring. Bullish ones like uptrending are positive bonus scoring and bearish are negative scoring. Expand the current chart patterns to use ones by FinnHub or Massive if they provide a chart classification API call in my paid license with them.
> Critic shouldn't nitpick about individual financial metrics. There should be more than one reason for a denial.
> A denial should be mostly due to poor news sentiment and poor stock performance or is too risky for the user's published profile risk level or denied because the price is not a good buy any longer (too expensive).

His answer to the one design question: **a denial needs bad signs in TWO different groups; "too risky for the user's profile" still denies alone** (it is about who the user is, not the stock).

Config version 3 (`configs/critic_v1.json`, `"config_version": "3"`). Design choices entry 137 (written by the main session, not the worker).

## 1. What the license covers (probed 2026-11-20, one GET each, nothing written)

| Endpoint | Answer |
|---|---|
| Finnhub `/stock/recommendation` (analyst recommendation trends: monthly `period`, `strongBuy`, `buy`, `hold`, `sell`, `strongSell`) | **200: in the license** |
| Finnhub `/stock/price-target`, `/stock/upgrade-downgrade` | 403 |
| Finnhub `/scan/pattern`, `/scan/support-resistance`, `/scan/technical-indicator` | 403 |
| Massive `/benzinga/v1/ratings`, `/benzinga/v1/consensus-ratings/{ticker}` | 403 ("not entitled") |

So: the analyst input is Finnhub recommendation trends; there is NO chart-classification API, and the expanded chart patterns are computed in code from the stored day bars (which also keeps them point in time for the Time Machine, P2). Do not call any 403 endpoint.

## 2. The rules

| # | Rule |
|---|---|
| V1 | Every finding that can deny belongs to exactly ONE class, in a closed table in code (`critic/reasons.py`, new): `fault`, `profile`, or one of the three judgment GROUPS `news`, `performance`, `price`. A test fails if a blocker code is in no class or two. |
| V2 | **fault** denies alone: the recommendation cannot be traded safely or cannot be checked. `record_defect`, `record_not_reviewable`, `symbol_unknown`, `price_mismatch`, `source_missing`, `source_altered`, `market_closed`, `quote_stale`, `no_live_price`, `bars_stale`, `record_expired`, `time_stop_passed`, `stop_already_hit`, and every finding that asks for a person (`needs_human_review`, entry 103: unchanged). |
| V3 | **profile** denies alone (Jacob's exception): `risk_above_profile`, `sector_excluded`, `symbol_excluded`, `price_below_profile_min`. |
| V4 | The three judgment GROUPS. A group is BAD when at least one of its signs fired: **news** = `fresh_negative_event`, `fresh_negative_news`, `event_not_supported`, `analysts_bearish` (new, section 3); **performance** = `trend_not_confirmed`, `dead_cat_bounce`, `falling_knife`, `price_below_zone`, `chart_bearish` (new: the net chart-pattern points at or under `chart.bearish_at`), `poor_performance` (new: `ret_63d <= performance.poor_ret_63d` AND `ret_21d < 0`, both the Critic's own re-asked metrics); **price** = `overpriced`, `valuation_extreme`, `price_left_zone` (above the zone: "too expensive now"). |
| V5 | **The verdict:** DENIED when any fault or profile finding fired, OR at least TWO of the three groups are bad. Otherwise APPROVED. Nothing between (entry 103 unchanged). |
| V6 | ONE bad group does NOT deny: the approval's summary names it ("Note: one reason against it: performance (trend_not_confirmed: ...)"), the same way `INFORMS` notes work today. |
| V7 | **No nitpicking on one financial metric.** `data_mismatch` and `data_unverified` on a single metric become WARNINGS (the accuracy dimension still scores them). They stay a FAULT only when the metric is the price itself (`last_close`, the quote) or when the record's match rate is under `accuracy.min_match_rate` (the record as a whole cannot be trusted): one finding code `data_unreliable` (fault) for that case, so the report card sees it apart. `blend_not_supported`, `rich_price`, `single_publisher`, `evidence_old`, `source_not_tagged`, `weak_chart`, `death_cross`, `high_52w_differs` stay warnings and belong to no group (they inform). |
| V8 | **The score no longer denies.** `score_below_minimum` is gone as a reason; `approve_min_score` is removed from the config. The score and the grade stay on the card (they rank and inform). The blocked cap stays: a DENIED review shows at most `blocked_score_cap`; an approved one with one bad group is NOT capped. |
| V9 | `denial_reasons` lists the codes that denied, in `BLOCKER_ORDER`, and for a two-group denial ALSO the group names first (`"group:news"`, `"group:performance"`), so a reader sees WHY two reasons were enough. The DENY summary sentence names the two (or three) groups in words with the heaviest code of each: `DENY XOM: poor news (fresh_negative_event: ...) and poor performance (trend_not_confirmed: ...).` |
| V10 | Every threshold is in the config; none is tuned on the outcomes (entry 133 still holds: the report card judges; Jacob chose these rules by reason, entry 137 says so). |

## 3. Analysts (a new dimension and a news sign)

- **Data.** `DataClerk.request_analyst_ratings(symbols)` (new; the Critic reaches the world only through the Data Clerk, as today) calls Finnhub `/stock/recommendation` through the toolbox (`DataToolbox.fetch_analyst_ratings`), the same key handling and error handling as the other Finnhub calls. Each answer is kept, per symbol, in `data/analysts/{SYMBOL}.json` (gitignored, derived: a dict keyed by `period`, newest answer for a period wins, written atomically). Reading is POINT IN TIME: only periods `<= trading_clock.horizon()` (a period `2026-11-01` is usable from 2026-11-01). While a day is bound (`as_of`) the fetcher refuses as every fetcher does and only the stored file is read. A caveat joins the closed list in `proving/record.py`: `analysts_not_point_in_time` ("the analyst counts are the period's last revision, not what was known on the day"), shown when a Time Machine review used them.
- **Score.** A new dimension `analysts`, weight 10; the weights become `accuracy 25, evidence 15, risk_fit 15, timing 10, value 15, chart 10, analysts 10` (total 100; a test pins the total). For the newest usable period: `n = strongBuy+buy+hold+sell+strongSell`; `mean = (5*strongBuy + 4*buy + 3*hold + 2*sell + 1*strongSell) / n` (1 to 5); score `= 100 * (mean - 1) / 4`. Not applicable (out of the weighted average, as other dimensions) when there is no usable period or `n < analysts.min_analysts` (3). The measure lists the counts, the period and `n`. A change since the period `analysts.trend_months` (3) earlier is shown as a measure (`mean_change`), not scored.
- **The sign.** `analysts_bearish` (a `news` sign, severity warning, it denies only through V5): `mean <= analysts.bearish_mean` (2.5: more sell than buy) OR `mean_change <= analysts.bearish_drop` (-0.5 over `trend_months`).
- Config section: `"analysts": {"min_analysts": 3, "trend_months": 3, "bearish_mean": 2.5, "bearish_drop": -0.5, "store": "data/analysts"}`.

## 4. Chart patterns (more patterns, and a bonus)

- **Bonus.** The net of the fired patterns' points (the existing list plus the new ones) becomes ALSO a bonus on the final score, like the great-deal bonus: `pattern_bonus = clamp(net * chart.bonus.per_point, -chart.bonus.max_points, +chart.bonus.max_points)` with `per_point 0.25`, `max_points 8`. A bullish chart raises the score, a bearish one lowers it. Shown on the card (`bonus` keeps the great-deal bonus; add `pattern_bonus`). The chart dimension stays as today.
- **`chart_bearish`** (performance sign, V4): `net <= chart.bearish_at` (-20).
- **New patterns, in code, on the stored day bars only** (`critic/chart.py`, or a new `critic/patterns.py` if chart.py grows past ~300 lines). Swing highs/lows are pivots with `chart.pivot_window` (5) bars each side, over the last `chart.pattern_lookback` (120) sessions; every tolerance is in the config. Points (bullish positive, bearish negative):

| pattern | points | definition (all thresholds in config) |
|---|---|---|
| `double_bottom` | +10 | two swing lows within `level_tolerance` (3%) of each other, at least `min_gap` (10) sessions apart, a swing high between them at least `min_depth` (5%) above them, and the last close above that middle high (the neckline broken) |
| `double_top` | -10 | the mirror image, last close below the middle low |
| `inverse_head_and_shoulders` | +10 | three swing lows, the middle the lowest by at least `min_depth`, the outer two within `level_tolerance`, last close above the higher of the two highs between them |
| `head_and_shoulders` | -10 | the mirror image |
| `ascending_triangle` | +5 | the last three swing highs within `level_tolerance` (flat top) and the last three swing lows each higher than the one before |
| `descending_triangle` | -5 | the last three swing lows within `level_tolerance` (flat bottom) and the last three swing highs each lower |
| `breakout_52w` | +10 | last close above the highest close of the 252 sessions before it |
| `breakdown_52w` | -10 | last close below the lowest close of the 252 sessions before it |

  `uptrend_structure` (+10) and `downtrend_structure` (-10) already exist and stay ("uptrending" in Jacob's words).
- Tests on REAL stored bars: for each new pattern find a real symbol and day in the stored bars where it fires and one where it does not (write the finder as a throwaway script in the scratchpad, put the found (symbol, day) pairs in the test with a comment saying how they were found). No invented bars.

## 5. What changes, module by module

- `configs/critic_v1.json`: version 3, `_version_3` note, weights (section 3), `verdict` without `approve_min_score`, `analysts` section, `performance` section `{"poor_ret_63d": -0.15}`, `chart` additions (`bearish_at`, `bonus`, `pivot_window`, `pattern_lookback`, `level_tolerance`, `min_gap`, `min_depth`, the new pattern points). `critic/config.py` validates every key (missing or unknown key is an error, as today).
- `critic/reasons.py` (new): the class table (V1 to V4) and `groups_of(findings) -> dict[group, list[Finding]]`.
- `critic/scoring.py`: `decide` implements V5, V8, V9; `summarize` implements V6 and V9; `weighted_score` adds `pattern_bonus`.
- `critic/recheck.py`: V7 (`data_unreliable`).
- `critic/timing.py`: `price_below_zone` stays a warning but counts as a performance sign (V4); nothing else.
- `critic/analysts.py` (new): the dimension and the sign; `review.py` step 3 asks the Data Clerk for the ratings with the other requests (all records at once), step 8 scores them.
- `critic/record.py`: `Review` gains `pattern_bonus`, `groups` (`{"news": [codes], "performance": [...], "price": [...]}`, empty lists when clean) and an `AnalystsDetail`; `BLOCKER_ORDER` gains the new codes. Old reviews (no `groups`) still load.
- `prometheus/proving/report_card.py`: the new codes and the groups are scored like the others; a line per GROUP ("denied with this group bad") and the two-group verdict. The old versions' rows stay as they are.
- `application/web/views.py` + `ui/src`: the Critic's card shows the three groups (bad / clean) and the analyst counts in one line each; no new rule on the page. Front-end tests updated.
- `prometheus/agents/account_manager/decisions.py` `denied_lines`: says the groups in words (it reads `denial_reasons`).
- The Advisor is NOT touched (the Critic imports nothing of its scoring code: test). The Critic still never receives the portfolio and sizes nothing.

## 6. Tests (real data only; AGENTS.md)

- Update the existing Critic tests whose expected verdict changes, and LIST each changed expected value in the report (which test, old, new, why).
- New: the class table is closed (V1); one bad group approves with a note, two deny, a fault or profile denies alone (on real captured reviews from `tests/fixtures/` re-scored, not invented findings where a real one exists); `data_mismatch` on one non-price metric no longer denies; the analyst score on a REAL captured Finnhub answer (capture 6 symbols of the stored universe once into `tests/fixtures/finnhub_recommendation_trends.json` with a `--save`-style script: Finnhub only, cheap, allowed); point in time: a period after the bound day is never read; the eight patterns on real bars; the pattern bonus sign and cap.
- Report card: runs on the saved results (free) and shows the new groups.

## 7. Order of work

1. reasons table + scoring (V1 to V9) with the existing findings; tests. 2. V7 in recheck. 3. analysts (clerk, toolbox, store, dimension, sign, caveat). 4. patterns + bonus. 5. report card, views, UI. 6. Full Python suite and front-end tests; `./test_proving.sh critic-report` (free) before/after numbers in the report.

## 8. Open for Jacob (not decided here)

- The new thresholds (`bearish_mean 2.5`, `bearish_drop -0.5`, `chart.bearish_at -20`, `poor_ret_63d -0.15`, the bonus 0.25 a point capped at 8) are conventions, not tuned.
- Under V5, `trend_not_confirmed` (v2's best rule at catching losers, -4.2 points) no longer denies alone; the report card will show what that costs.
- A backfill of the analyst store for the stored symbols (one Finnhub call each, writes `data/analysts/`) is NOT run by the worker: Jacob's to say.
